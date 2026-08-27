"""imatrix refit for the routed experts of Ornith-1.5-35B-A3B (`qwen3_5_moe`).

Created by Jinho Jang (eric@osaurus.ai) — 2026-08-19.

`qwen36_imatrix_refit` resolves bundle module paths to source tensors with a
DENSE-only `source_key()` and `continue`s on a miss. On the 35B that silently
skipped **every routed expert**: `refit 474 modules` out of 596, with all 120
misses being `mlp.switch_mlp.*`. AWQ and the fold survive that (skipped modules
keep the build's codes), but the experts — 91.8 % of the parameters — never got
the activation-weighted fit.

This module refits exactly those 120, and nothing else, so it composes with the
dense pass rather than replacing it:

    source                                   bundle module
    experts.gate_up_proj (E,1024,2048)  ->   switch_mlp.gate_proj  = [:, :512, :]
                                             switch_mlp.up_proj    = [:, 512:, :]
    experts.down_proj    (E,2048, 512)  ->   switch_mlp.down_proj  (whole)

Verified shapes at 4-bit gs64: `gate_proj.weight (256,512,256) uint32` with
`scales (256,512,32)` — 2048/8 codes per uint32 and 2048/64 groups; `down_proj`
`(256,2048,64)` + `(256,2048,8)` for in_features 512. The capture agrees:
in_features 2048 for gate/up, 512 for down.

The 2-D quantizer is applied to a `(E*out, in)` reshape — affine groups run
along the input axis, so flattening the expert and output axes is exact — then
packed/scales/biases are reshaped back to 3-D.

🚨 `--awq-scales` is required if the bundle records an AWQ fold, for the same
reason as the dense pass: this reloads W from SOURCE, which knows nothing about
the fold, so without re-applying `s` it would revert `W*s` to `W` while the
norms stay divided by `s`.

    python -m jang_tools.ornith_moe_imatrix_refit <src> <calib.safetensors> \
        <bundle> [--group-size 64] [--awq-scales <path>]
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors.numpy import load_file

from .affine import quantize_imatrix_affine_numpy


def _plan(path: str) -> tuple[str, str | None]:
    """bundle module -> (source tensor key, half)."""
    p = path.replace("language_model.model.", "model.language_model.")
    if p.endswith(".mlp.switch_mlp.gate_proj"):
        return p[: -len(".switch_mlp.gate_proj")] + ".experts.gate_up_proj", "gate"
    if p.endswith(".mlp.switch_mlp.up_proj"):
        return p[: -len(".switch_mlp.up_proj")] + ".experts.gate_up_proj", "up"
    if p.endswith(".mlp.switch_mlp.down_proj"):
        return p[: -len(".switch_mlp.down_proj")] + ".experts.down_proj", None
    return "", None


def main(argv) -> int:
    if len(argv) < 4:
        print(__doc__)
        return 1
    src, calib_p, bundle = Path(argv[1]), Path(argv[2]), Path(argv[3])
    gs = 64
    awq_scales_p = None
    for i, a in enumerate(argv):
        if a == "--group-size":
            gs = int(argv[i + 1])
        if a == "--awq-scales":
            awq_scales_p = argv[i + 1]

    calib = load_file(str(calib_p))
    cfg = json.loads((bundle / "config.json").read_text())
    q = cfg["quantization"]

    awq_s = {}
    if awq_scales_p:
        awq_s = {k[: -len(".awq_scale")]: v
                 for k, v in load_file(awq_scales_p).items()
                 if k.endswith(".awq_scale")}
    if cfg.get("awq") and not awq_s:
        print("  !! bundle records an AWQ fold but no --awq-scales given: "
              "refitting from unscaled source would BREAK the fold. Refusing.")
        return 2

    targets = {k: v for k, v in q.items()
               if isinstance(v, dict) and "switch_mlp" in k}
    if not targets:
        print("  no routed-expert modules in this bundle — nothing to do")
        return 0
    print(f"  routed-expert modules: {len(targets)}")
    if awq_s:
        n = sum(1 for k in targets if k in awq_s)
        print(f"  AWQ scales available for {n}/{len(targets)} of them")

    src_wm = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    bun_wm = json.loads((bundle / "model.safetensors.index.json").read_text())["weight_map"]

    by_shard: dict[str, list[str]] = {}
    for path in targets:
        wk = f"{path}.weight"
        if wk in bun_wm:
            by_shard.setdefault(bun_wm[wk], []).append(path)

    t0, done, errs, skipped = time.time(), 0, [], []
    for shard, paths in sorted(by_shard.items()):
        shard_p = bundle / shard
        # lazy-mmap self-clobber: materialize, write temp, atomic rename.
        tensors = dict(mx.load(str(shard_p)))
        mx.eval(list(tensors.values()))
        for path in paths:
            skey, half = _plan(path)
            if not skey or skey not in src_wm:
                skipped.append(path)
                continue
            imp_key = f"{path}.second_moment"
            if imp_key not in calib:
                skipped.append(path)
                continue
            bits = targets[path]["bits"]
            g = targets[path].get("group_size", gs)

            W = mx.load(str(src / src_wm[skey]))[skey].astype(mx.float32)
            W = np.array(W)
            if half:                       # split the fused gate_up
                mid = W.shape[1] // 2
                W = W[:, :mid, :] if half == "gate" else W[:, mid:, :]

            s_awq = awq_s.get(path)
            if s_awq is not None and s_awq.shape[0] == W.shape[-1]:
                W = W * s_awq[None, None, :].astype(np.float32)

            imp = calib[imp_key].astype(np.float32)
            if imp.shape[0] != W.shape[-1]:
                skipped.append(path)
                continue

            E, out, in_f = W.shape
            packed, scales, biases, werr = quantize_imatrix_affine_numpy(
                W.reshape(E * out, in_f), imp, bits=bits, group_size=g)
            tensors[f"{path}.weight"] = mx.array(packed.reshape(E, out, -1))
            tensors[f"{path}.scales"] = mx.array(scales.reshape(E, out, -1)).astype(mx.bfloat16)
            tensors[f"{path}.biases"] = mx.array(biases.reshape(E, out, -1)).astype(mx.bfloat16)
            errs.append(werr)
            done += 1
            if done % 20 == 0:
                print(f"    {done}/{len(targets)}  ({time.time()-t0:.0f}s)", flush=True)

        mx.eval(list(tensors.values()))
        tmp = str(shard_p).replace(".safetensors", ".tmp.safetensors")
        mx.save_safetensors(tmp, tensors, metadata={"format": "pt"})
        shutil.move(tmp, str(shard_p))
        print(f"  rewrote {shard}", flush=True)

    if skipped:
        print(f"  !! {len(skipped)} routed-expert modules were SKIPPED "
              f"(unresolved source or missing capture) — refusing to report "
              f"success: {skipped[:5]}")
        return 2

    print(f"\n  refit {done} routed-expert modules in {time.time()-t0:.0f}s")
    print(f"  mean weighted rel-err after fit: {float(np.mean(errs)):.4f}")
    cfg.setdefault("imatrix", {})["routed_experts"] = {
        "modules": done, "group_size": gs,
        "awq_scales_reapplied": bool(awq_s),
    }
    (bundle / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
