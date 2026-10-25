"""DeepSeek-V4.1-Flash DSpark MTP draft -> a self-contained ``dspark/`` subdir in a JANG bundle.

The main JANG bundle (`convert_dsv41_jang.py`) DROPS the `mtp.*` draft on purpose (it is a
speculative head, not needed for plain autoregressive serving). This tool carries the draft into
the bundle as its own component subdir — the same convention the engram tables use (`engram/`):

  <bundle>/dspark/dspark.safetensors   the 3-stage DSpark draft (routed experts 4-bit, rest bf16/f32)
  <bundle>/dspark/README.md

and patches `<bundle>/config.json` `runtime` so the bundle advertises `bundle_has_mtp: True`.

Layout (matches what `dspark.load_dspark` assigns):
  - routed experts: per (stage, proj) the 128 per-expert weights are dequantized, re-quantized to
    4-bit affine (group 64) and STACKED into the SwitchProj layout
      mtp.{s}.ffn.experts.{gate,up,down}_proj.{weight,scales,biases}
    (w1->gate_proj, w3->up_proj, w2->down_proj), so load is a direct assign, not a per-expert quant.
  - everything else under `mtp.*` is copied at its load precision: F8_E4M3 weights -> bf16,
    stored BF16/F32 tensors pass through. `.scale` siblings are consumed by the dequant.

This is tiny next to the backbone (~5-8 GB), so it writes in place with no full-bundle rebuild.

Usage:
  python -m jang_tools.dsv41.convert_dspark_jang \
      --src ~/models/DeepSeek-V4.1-Flash-1017 \
      --dst ~/models/DeepSeek-V4.1-Flash-1017-JANG-Q2Q3-IMATRIX
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import mlx.core as mx

from jang_tools.dsv41.st_io import Shards

GROUP_SIZE = 64
# source per-expert projection name -> prestacked SwitchProj name (same map the backbone uses)
PROJ = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
EXPERT_EVAL_CHUNK = 32  # eval cadence while quantizing experts, bounds transient memory


def _stage_ids(keys) -> list[int]:
    return sorted({int(m.group(1)) for m in (re.match(r"mtp\.(\d+)\.", k) for k in keys) if m})


def _quant_stack_experts(sh: Shards, stage: int, mat: str, n_experts: int, bits: int):
    """Dequant the 128 per-expert bf16 weights for one projection and re-quantize to `bits`,
    stacked into the SwitchProj layout (weight/scales/biases each stacked on axis 0)."""
    qs, ss, bs = [], [], []
    for e in range(n_experts):
        w = sh.dequant(f"mtp.{stage}.ffn.experts.{e}.{mat}.weight", mx.bfloat16)
        qw, s, b = mx.quantize(w, group_size=GROUP_SIZE, bits=bits)
        qs.append(qw)
        ss.append(s)
        bs.append(b)
        if e % EXPERT_EVAL_CHUNK == EXPERT_EVAL_CHUNK - 1:
            mx.eval(qs[-1], ss[-1], bs[-1])
    weight, scales, biases = mx.stack(qs), mx.stack(ss), mx.stack(bs)
    mx.eval(weight, scales, biases)
    return weight, scales, biases


def build_dspark_subdir(src: Path, dst: Path, bits: int = 4) -> dict:
    src, dst = Path(src).expanduser(), Path(dst).expanduser()
    if not (dst / "config.json").exists():
        raise FileNotFoundError(f"destination is not a JANG bundle: {dst}/config.json missing")
    out_dir = dst / "dspark"
    out_dir.mkdir(exist_ok=True)

    sh = Shards(src, allow_missing=True)
    mtp = [n for n in sh.keys() if n.startswith("mtp.")]
    if not mtp:
        raise ValueError(f"no mtp.* tensors in source {src}")
    stages = _stage_ids(mtp)
    scfg = json.loads((src / "config.json").read_text())["text_config"]
    n_experts = int(scfg["dspark_n_routed_experts"])

    tensors: dict[str, mx.array] = {}
    t0 = time.time()

    # 1) routed experts -> prestacked 4-bit
    expert_err = {}
    for s in stages:
        for mat, proj in PROJ.items():
            weight, scales, biases = _quant_stack_experts(sh, s, mat, n_experts, bits)
            base = f"mtp.{s}.ffn.experts.{proj}"
            tensors[f"{base}.weight"] = weight
            tensors[f"{base}.scales"] = scales
            tensors[f"{base}.biases"] = biases
            # cheap fidelity probe on expert 0
            w0 = sh.dequant(f"mtp.{s}.ffn.experts.0.{mat}.weight", mx.float32)
            dq = mx.dequantize(weight[0], scales[0], biases[0], group_size=GROUP_SIZE, bits=bits).astype(mx.float32)
            d = dq - w0
            expert_err[base] = float(mx.sqrt(mx.sum(d * d)) / mx.maximum(mx.sqrt(mx.sum(w0 * w0)), 1e-30))
            print(f"[{time.time()-t0:6.0f}s] experts {base}: {tuple(weight.shape)} rel_l2={expert_err[base]:.4f}", flush=True)
            del weight, scales, biases, w0, dq
            mx.clear_cache()

    # 2) everything else under mtp.* copied at load precision
    n_copied = 0
    for name in sorted(mtp):
        if ".ffn.experts." in name:
            continue  # handled above
        if name.endswith(".scale"):
            continue  # consumed by dequant of its .weight
        m = sh.meta[name]
        if name.endswith(".weight") and m.dtype in ("F8_E4M3", "I8"):
            tensors[name] = sh.dequant(name, mx.bfloat16)
        else:
            tensors[name] = sh.to_mx(name)  # BF16 / F32 / small tensors pass through
        n_copied += 1
        if n_copied % 16 == 0:
            mx.eval(list(tensors.values())[-16:])
    mx.eval(list(tensors.values())[-16:])
    print(f"[{time.time()-t0:6.0f}s] copied {n_copied} non-expert mtp tensors", flush=True)

    out_file = out_dir / "dspark.safetensors"
    tmp = out_dir / "dspark.tmp.safetensors"  # mx.save_safetensors requires a .safetensors name
    mx.save_safetensors(str(tmp), tensors, metadata={"format": "mlx"})
    tmp.rename(out_file)
    size_gb = out_file.stat().st_size / 1e9

    (out_dir / "README.md").write_text(
        "DeepSeek-V4.1-Flash DSpark MTP draft (3 stages), carried as a self-contained bundle "
        "component (like `engram/`). Routed experts are 4-bit affine (group 64), stacked in the "
        "SwitchProj layout; attention/norms/shared/markov are bf16, small heads f32. Loaded by "
        "`jang_tools.dsv41.dspark.load_dspark(<bundle>)` when the bundle advertises "
        "`runtime.bundle_has_mtp`.\n"
    )

    # patch bundle config.json runtime to advertise the draft
    cfg_path = dst / "config.json"
    cfg = json.loads(cfg_path.read_text())
    rt = cfg.setdefault("runtime", {})
    rt["bundle_has_mtp"] = True
    rt["mtp_mode"] = "dspark_subdir"
    rt["dspark_dir"] = "dspark"
    cfg.setdefault("text_config", {})["num_nextn_predict_layers"] = len(stages)
    cfg_path.write_text(json.dumps(cfg, indent=2))

    summary = {
        "stages": stages,
        "n_experts": n_experts,
        "bits": bits,
        "tensors": len(tensors),
        "size_gb": round(size_gb, 2),
        "expert_rel_l2": expert_err,
    }
    (out_dir / "dspark_convert.json").write_text(json.dumps(summary, indent=2))
    print(f"[{time.time()-t0:6.0f}s] wrote {out_file} ({size_gb:.2f} GB), patched config runtime", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source DeepSeek-V4.1-Flash model dir (has mtp.*)")
    ap.add_argument("--dst", required=True, help="JANG bundle dir to add dspark/ into")
    ap.add_argument("--bits", type=int, default=4)
    a = ap.parse_args()
    build_dspark_subdir(Path(a.src), Path(a.dst), a.bits)


if __name__ == "__main__":
    main()
