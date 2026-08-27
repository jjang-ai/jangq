"""GPTQ codes for the routed experts of an Ornith 1.5 MoE bundle.

Created by Jinho Jang (eric@osaurus.ai) — 2026-08-19.

Replaces the routed-expert codes with GPTQ error-compensated rounding using the
full ``H = XᵀX`` captured by `ornith_moe_hessians`. This is the pass that the
diagonal-only capture could not support: Hessian-trace allocation, the imatrix
fit and AWQ all use ``E[x_c²]`` (the diagonal), GPTQ needs the whole matrix.

Why not `dots3/gptq_dots3.py`: that one builds **per-expert routing-weighted**
Hessians and therefore wants raw per-layer activations plus router weights —
a different capture. We use the pooled shared-H form, which is what
`capture_gemma4_hessians` was designed around and what `gptq_mlx` consumes:
all experts in a layer share one H. Per-expert H would be 256x the memory for a
second-order refinement of an already second-order method, and at top-8 routing
most experts see too few tokens for a stable estimate.

Layout, verified on the real bundle:

    source  experts.gate_up_proj (E,1024,2048) -> gate = [:, :512, :]
                                                  up   = [:, 512:, :]
            experts.down_proj    (E,2048, 512)
    H       H_L{layer}_in_d2048   feeds gate + up
            H_L{layer}_mid_d512   feeds down_proj

Each expert is quantized against the shared H for its projection. Codes are
emitted through the same `mx.quantize` ABI (uint32 + fp16 scales/biases) so the
runtime is unchanged.

🚨 AWQ scales are re-applied to the source weight first, for the same reason as
the imatrix refit: this reads W from SOURCE, which knows nothing about the fold,
so without `--awq-scales` it would revert `W*s` to `W` while the bundle's norms
stay `/s`. Refuses if the bundle records an AWQ fold and scales are absent.

🚨 Never-worse-than-RTN guard per tensor: GPTQ on a rank-deficient or badly
conditioned H can be worse than plain rounding. Each expert keeps whichever
reconstructs the source better on the calibration Hessian.

    python -m jang_tools.ornith_moe_gptq <src> <hessians_dir> <bundle> \
        [--group-size 64] [--awq-scales <path>] [--layers 0-39]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors.numpy import load_file

from .gptq_mlx import gptq_quantize_mlx_fast


def _parse_layers(spec: str | None, n: int) -> set[int]:
    if not spec:
        return set(range(n))
    out: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def _dequant_err(W: np.ndarray, packed, scales, biases, bits: int,
                 gs: int, H: np.ndarray) -> float:
    """Hessian-weighted reconstruction error, for the best-of guard."""
    q = mx.dequantize(mx.array(packed), mx.array(scales.astype(np.float16)),
                      mx.array(biases.astype(np.float16)),
                      group_size=gs, bits=bits, mode="affine")
    D = np.array(q.astype(mx.float32)) - W
    # tr(D H Dᵀ) without forming D H Dᵀ
    return float(np.einsum("ij,jk,ik->", D, H, D, optimize=True))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("hess", type=Path)
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--awq-scales", type=str, default=None)
    ap.add_argument("--layers", default=None)
    a = ap.parse_args()

    cfg = json.loads((a.bundle / "config.json").read_text())
    q = cfg["quantization"]
    awq_s = {}
    if a.awq_scales:
        awq_s = {k[: -len(".awq_scale")]: v
                 for k, v in load_file(a.awq_scales).items()
                 if k.endswith(".awq_scale")}
    if cfg.get("awq") and not awq_s:
        print("  !! bundle records an AWQ fold but no --awq-scales given: "
              "GPTQ from unscaled source would BREAK the fold. Refusing.")
        return 2

    targets = {k: v for k, v in q.items()
               if isinstance(v, dict) and "switch_mlp" in k}
    if not targets:
        print("  no routed-expert modules — nothing to do")
        return 0

    src_wm = json.loads((a.src / "model.safetensors.index.json").read_text())["weight_map"]
    bun_wm = json.loads((a.bundle / "model.safetensors.index.json").read_text())["weight_map"]
    n_layers = cfg.get("text_config", cfg).get("num_hidden_layers", 40)
    want = _parse_layers(a.layers, n_layers)

    by_shard: dict[str, list[str]] = {}
    for path in targets:
        li = int(path.split(".layers.")[1].split(".")[0])
        if li not in want:
            continue
        wk = f"{path}.weight"
        if wk in bun_wm:
            by_shard.setdefault(bun_wm[wk], []).append(path)

    total = sum(len(v) for v in by_shard.values())
    print(f"  routed-expert modules to GPTQ: {total}")
    print(f"  AWQ scales available for "
          f"{sum(1 for s in by_shard.values() for p in s if p in awq_s)}/{total}")

    t0, done, kept_rtn, errs = time.time(), 0, 0, []
    unpackable: set = set()
    for shard, paths in sorted(by_shard.items()):
        shard_p = a.bundle / shard
        tensors = dict(mx.load(str(shard_p)))
        mx.eval(list(tensors.values()))
        for path in sorted(paths):
            li = int(path.split(".layers.")[1].split(".")[0])
            leaf = path.rsplit(".", 1)[-1]
            which = "mid" if leaf == "down_proj" else "in"
            hs = list(a.hess.glob(f"H_L{li:02d}_{which}_d*.npy"))
            if not hs:
                print(f"  !! no Hessian for L{li} {which} — refusing")
                return 2
            H = np.load(hs[0]).astype(np.float64)
            H = (H / max(np.trace(H) / H.shape[0], 1e-12)).astype(np.float32)

            base = path.replace("language_model.model.", "model.language_model.")
            if leaf in ("gate_proj", "up_proj"):
                skey = base[: -len(f".switch_mlp.{leaf}")] + ".experts.gate_up_proj"
            else:
                skey = base[: -len(".switch_mlp.down_proj")] + ".experts.down_proj"
            if skey not in src_wm:
                print(f"  !! unresolved source for {path} -> {skey}; refusing")
                return 2

            W = np.array(mx.load(str(a.src / src_wm[skey]))[skey].astype(mx.float32))
            if leaf in ("gate_proj", "up_proj"):
                mid = W.shape[1] // 2
                W = W[:, :mid, :] if leaf == "gate_proj" else W[:, mid:, :]
            s_awq = awq_s.get(path)
            if s_awq is not None and s_awq.shape[0] == W.shape[-1]:
                W = W * s_awq[None, None, :].astype(np.float32)

            bits = targets[path]["bits"]
            gs = targets[path].get("group_size", a.group_size)
            E, out, in_f = W.shape
            W2 = W.reshape(E * out, in_f)

            # RTN reference first — also gives us the authoritative packed shape.
            rq, rs, rb = mx.quantize(mx.array(W2), group_size=gs, bits=bits,
                                     mode="affine")
            mx.eval(rq, rs, rb)

            # 🚨 PACK SELF-TEST. gptq_mlx's _pack_uint32 only produces MLX's
            # layout for bit widths that divide 32 evenly: verified 2/4/8 OK,
            # 3/5/6 MISMATCH (e.g. 5-bit gives 342 words where MLX wants 320).
            # Writing those would corrupt the bundle, so GPTQ is skipped and
            # RTN kept — never a silent mismatched layout.
            packed, scales, biases = gptq_quantize_mlx_fast(
                W2, H, bits=bits, group_size=gs)
            if tuple(packed.shape) != tuple(rq.shape):
                tensors[f"{path}.weight"] = mx.array(
                    np.array(rq).reshape(E, out, -1))
                tensors[f"{path}.scales"] = mx.array(
                    np.array(rs.astype(mx.float32)).reshape(E, out, -1)).astype(mx.bfloat16)
                tensors[f"{path}.biases"] = mx.array(
                    np.array(rb.astype(mx.float32)).reshape(E, out, -1)).astype(mx.bfloat16)
                unpackable.add(bits)
                kept_rtn += 1
                done += 1
                continue

            e_gptq = _dequant_err(W2, packed, scales, biases, bits, gs, H)
            e_rtn = _dequant_err(W2, np.array(rq), np.array(rs.astype(mx.float32)),
                                 np.array(rb.astype(mx.float32)), bits, gs, H)
            if e_rtn <= e_gptq:
                packed, scales, biases = (np.array(rq),
                                          np.array(rs.astype(mx.float32)),
                                          np.array(rb.astype(mx.float32)))
                kept_rtn += 1
            else:
                errs.append(1.0 - e_gptq / max(e_rtn, 1e-30))

            tensors[f"{path}.weight"] = mx.array(packed.reshape(E, out, -1))
            tensors[f"{path}.scales"] = mx.array(
                scales.reshape(E, out, -1).astype(np.float32)).astype(mx.bfloat16)
            tensors[f"{path}.biases"] = mx.array(
                biases.reshape(E, out, -1).astype(np.float32)).astype(mx.bfloat16)
            done += 1
            if done % 10 == 0:
                print(f"    {done}/{total}  ({time.time()-t0:.0f}s, "
                      f"{kept_rtn} kept RTN)", flush=True)

        mx.eval(list(tensors.values()))
        tmp = str(shard_p).replace(".safetensors", ".tmp.safetensors")
        mx.save_safetensors(tmp, tensors, metadata={"format": "pt"})
        shutil.move(tmp, str(shard_p))
        print(f"  rewrote {shard}", flush=True)

    print(f"\n  GPTQ on {done} routed-expert modules in {time.time()-t0:.0f}s")
    print(f"  kept RTN (GPTQ not better, or unpackable): {kept_rtn}")
    if unpackable:
        print(f"  !! bit widths GPTQ could not pack MLX-compatibly, RTN kept: "
              f"{sorted(unpackable)} — gptq_mlx._pack_uint32 only matches MLX "
              f"for widths dividing 32 (2/4/8).")
    if errs:
        print(f"  mean Hessian-weighted error reduction vs RTN: "
              f"{100*float(np.mean(errs)):.1f}%")
    cfg.setdefault("gptq", {})["routed_experts"] = {
        "modules": done, "kept_rtn": kept_rtn,
        "awq_scales_reapplied": bool(awq_s),
        "hessian": "pooled per-layer XtX (all experts share one H)",
        "unpackable_bits_kept_rtn": sorted(unpackable),
    }
    (a.bundle / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
