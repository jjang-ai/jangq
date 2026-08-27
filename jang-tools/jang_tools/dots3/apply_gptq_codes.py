"""Swap GPTQ codes + grid into a converted dots3 bundle (byte-safe).

For each layer with a codes file:
  - GPTQ owns the routed grid (per-unit best of activation-weighted imatrix
    fit vs min-max), so scales/biases are replaced ALONGSIDE the codes;
    shapes and f16 dtype are asserted against the converter layout (the ABI);
  - pack codes to uint32 and overwrite the switch_mlp weight tensors;
  - rewrite affected shards via temp + atomic rename (no in-place mmap
    clobber), preserving every other tensor byte.

    python -m jang_tools.dots3.apply_gptq_codes <bundle> <codes_dir>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from .gptq_dots3 import pack_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", type=Path)
    ap.add_argument("codes_dir", type=Path)
    a = ap.parse_args()

    idx_p = a.bundle / "model.safetensors.index.json"
    index = json.loads(idx_p.read_text())
    wm = index["weight_map"]

    t0 = time.time()
    checked = 0
    files = sorted(p for p in a.codes_dir.glob("layer_*.npz")
                   if ".tmp." not in p.name)
    if len(files) != 45:
        raise SystemExit(f"expected 45 layer code files, found {len(files)} "
                         f"— GPTQ incomplete, refusing to apply")
    for codes_file in files:
        li = int(codes_file.stem.split("_")[1])
        z = np.load(codes_file, allow_pickle=False)
        specs = json.loads(str(z["specs"]))
        # stage this LAYER's packed units, grouped by shard, then write —
        # bounded memory (one layer ≈ 2-3 GB packed).
        staged: dict[str, dict[str, mx.array]] = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            bits, gs, mode = specs[proj]
            stem = f"model.layers.{li}.mlp.switch_mlp.{proj}"
            wname, sname, bname = (stem + ".weight", stem + ".scales",
                                   stem + ".biases")
            shard = wm[wname]
            cur = mx.load(str(a.bundle / shard))
            # GPTQ owns the routed grid (activation-weighted imatrix fit, or
            # min-max where that measured better), so scales/biases are
            # REPLACED alongside the codes. Shapes/dtypes must still match the
            # converter's layout exactly — that is the ABI contract.
            s_g = z[f"{proj}_scales"]
            b_g = z[f"{proj}_biases"]
            for arr, ten, nm in ((s_g, cur[sname], sname),
                                 (b_g, cur[bname], bname)):
                if arr.shape != tuple(ten.shape):
                    raise SystemExit(f"GRID SHAPE MISMATCH {nm}: "
                                     f"{arr.shape} vs {tuple(ten.shape)}")
                if str(ten.dtype).split(".")[-1] != "float16":
                    raise SystemExit(f"{nm} is {ten.dtype}, expected float16")
            codes = z[f"{proj}_codes"]
            E, out_d, in_d = codes.shape
            packed = pack_rows(codes.reshape(E * out_d, in_d), bits)
            packed = packed.reshape(E, out_d, in_d * bits // 32)
            old = np.asarray(cur[wname]).view(np.uint32)
            assert packed.shape == old.shape, (stem, packed.shape, old.shape)
            sub = staged.setdefault(shard, {})
            sub[wname] = mx.array(packed)
            sub[sname] = mx.array(s_g.astype(np.float16))
            sub[bname] = mx.array(b_g.astype(np.float16))
            checked += 1
            del cur, old
        for shard, subs in staged.items():
            tensors = dict(mx.load(str(a.bundle / shard)))
            tensors.update(subs)
            mx.eval(list(tensors.values()))
            tmp = a.bundle / shard.replace(".safetensors", ".tmp.safetensors")
            mx.save_safetensors(str(tmp), tensors)
            tmp.rename(a.bundle / shard)
        print(f"  layer {li}: 3 units swapped "
              f"({len(staged)} shard rewrites)", flush=True)
        del staged, z
        mx.clear_cache()

    # provenance stamp
    jc_p = a.bundle / "jang_config.json"
    jc = json.loads(jc_p.read_text()) if jc_p.exists() else {}
    jc["qat"] = {
        "method": "gptq_error_compensated_codes",
        "grid": "per_unit_best_of{imatrix_activation_weighted, minmax}_f16",
        "sequencing": "brecq_w1w3_then_w2",
        "units_replaced": checked,
        "codes_dir": str(a.codes_dir),
        "applied": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    jc_p.write_text(json.dumps(jc, indent=1))
    print(f"APPLY_DONE {checked} units in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
