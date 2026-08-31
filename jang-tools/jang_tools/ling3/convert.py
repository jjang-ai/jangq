"""Build JANG bundles for Ling-3.0 (BailingMoeV3): MXFP8 / JANG_6M / JANG_4M.

Created by Jinho Jang (eric@jangq.ai) — 2026-08-26.

Pipeline per bundle:

  bf16 source
    -> AWQ fold (with router + shared-expert compensation)
    -> per-expert stacking into SwitchGLU layout
    -> per-tensor bit widths from the Hessian-driven allocation
    -> imatrix-weighted clip-range refit
    -> mx.quantize

## Keeps (never quantized)

`conv1d` kernels, `A_log`, `dt_bias`, the KDA gate projections (`f/b/g_proj`),
all norms, and the router (`mlp.gate.weight`, `expert_bias`, fp32). These are
state-forming or gating parameters whose error compounds through a recurrence or
flips which experts execute, rather than averaging out across a matmul. Together
they are a small fraction of the model — the router alone is 128x1536 per layer.

## imatrix refit

MLX's affine quantizer takes the group's min/max as the range. That is optimal
for unweighted L2, but the channels are not equally important: a channel with a
large 2nd moment contributes proportionally more output error. So for each group
the clip range is shrunk by a searched factor and the **activation-weighted**
error is scored:

    err(f) = sum_j a_j^2 * (w_j - dequant(quant(clip(w_j, f))))^2

Shrinking trades a little clipping of the extremes for finer steps everywhere
else, which wins when the mass is in the middle. The chosen range is applied by
**clipping the weights and then calling `mx.quantize`**, so the official packer
computes exactly the intended scale/bias and the bit packing is never hand-rolled.

    python -m jang_tools.ling3.convert <src> <out> --profile JANG_4M \
        --calib <calib.safetensors> [--bitmap <bitmap.json>] [--awq <scales.safetensors>]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np

from ..format.aligned_safetensors import rewrite_aligned_safetensors

GROUP_SIZE = 64
MX_GROUP_SIZE = 32   # MX spec: one shared e8m0 scale per 32 elements

# Substrings of tensor names that are never quantized.
KEEP_PATTERNS = (
    "_conv1d.weight", ".A_log", ".dt_bias",
    ".f_proj.weight", ".b_proj.weight", ".g_proj.weight",
    "layernorm.weight", ".o_norm.weight", "model.norm.weight",
    ".mlp.gate.weight", ".mlp.gate.expert_bias",
)
FP32_KEEPS = (".A_log", ".dt_bias", ".mlp.gate.weight", ".mlp.gate.expert_bias")

# Non-expert quantized tensors, by profile.
NON_EXPERT_BITS = {"MXFP8": 8, "JANG_6M": 8, "JANG_4M": 8}
EMBED_BITS = {"MXFP8": 8, "JANG_6M": 8, "JANG_4M": 6}

SHRINK_GRID = [1.0, 0.98, 0.95, 0.92, 0.89, 0.86, 0.82, 0.78]


def is_keep(name: str) -> bool:
    return any(p in name for p in KEEP_PATTERNS)


def imatrix_quantize(
    w: mx.array, a2: mx.array | None, bits: int, group_size: int = GROUP_SIZE
) -> tuple[mx.array, mx.array, mx.array, float]:
    """Quantize with an activation-weighted clip-range search.

    `w` is ``[..., in_features]``; `a2` is the per-input-channel 2nd moment or
    None (falls back to plain min/max, i.e. shrink factor 1.0).

    Returns ``(packed, scales, biases, chosen_shrink)``.
    """
    if a2 is None:
        packed, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
        return packed, scales, biases, 1.0

    orig_shape = w.shape
    flat = w.reshape(-1, orig_shape[-1]).astype(mx.float32)
    aw = a2.astype(mx.float32)                                   # [in_features]

    best_err = None
    best = None
    best_f = 1.0
    for f in SHRINK_GRID:
        if f == 1.0:
            clipped = flat
        else:
            g = flat.reshape(flat.shape[0], -1, group_size)
            lo = mx.min(g, axis=-1, keepdims=True)
            hi = mx.max(g, axis=-1, keepdims=True)
            mid = (lo + hi) * 0.5
            half = (hi - lo) * 0.5 * f
            clipped = mx.clip(g, mid - half, mid + half).reshape(flat.shape)

        packed, scales, biases = mx.quantize(clipped, group_size=group_size, bits=bits)
        deq = mx.dequantize(packed, scales, biases, group_size=group_size, bits=bits)
        d = flat - deq
        err = mx.sum((d * d) * aw)
        mx.eval(err)
        e = float(err)
        if best_err is None or e < best_err:
            best_err, best, best_f = e, (packed, scales, biases), f

    packed, scales, biases = best
    lead = orig_shape[:-1]
    return (
        packed.reshape(*lead, -1),
        scales.reshape(*lead, -1),
        biases.reshape(*lead, -1),
        best_f,
    )


def load_source(src: Path) -> tuple[dict[str, mx.array], dict]:
    idx = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    weights: dict[str, mx.array] = {}
    for shard in sorted(set(idx.values())):
        weights.update(mx.load(str(src / shard)))
    cfg = json.loads((src / "config.json").read_text())
    return weights, cfg


def apply_awq_fold(weights: dict[str, mx.array], scales: dict[str, mx.array], cfg: dict) -> int:
    """Fold `1/s` into post_attention_layernorm and `s` into EVERY consumer.

    The router is a consumer. Missing it would leave the model's routing subtly
    shifted — with group-limited top-k that changes which experts execute, and no
    weight-error metric would show it.
    """
    n = 0
    for norm_path, s in scales.items():
        layer = norm_path.split(".")[2]
        nw = f"{norm_path}.weight"
        if nw not in weights:
            raise KeyError(f"AWQ fold target missing: {nw}")
        weights[nw] = (weights[nw].astype(mx.float32) / s).astype(weights[nw].dtype)

        consumers = [f"model.layers.{layer}.mlp.gate.weight"]              # router
        for proj in ("gate_proj", "up_proj"):
            consumers.append(f"model.layers.{layer}.mlp.shared_experts.{proj}.weight")
            consumers += [
                k for k in weights
                if k.startswith(f"model.layers.{layer}.mlp.experts.")
                and k.endswith(f"{proj}.weight")
            ]
        for c in consumers:
            if c not in weights:
                raise KeyError(f"AWQ consumer missing (fold would be unbalanced): {c}")
            weights[c] = (weights[c].astype(mx.float32) * s).astype(weights[c].dtype)
            n += 1
    return n


def stack_experts(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    experts: dict[str, dict[int, mx.array]] = defaultdict(dict)
    for k, v in weights.items():
        if ".mlp.experts." in k:
            head, tail = k.split(".mlp.experts.")
            eid, proj = tail.split(".", 1)
            experts[head + ".mlp.switch_mlp." + proj][int(eid)] = v
        else:
            out[k] = v
    for name, per in experts.items():
        out[name] = mx.stack([per[i] for i in sorted(per)], axis=0)
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="jang_tools.ling3.convert")
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--profile", required=True, choices=["MXFP8", "JANG_6M", "JANG_4M"])
    ap.add_argument("--calib", default=None)
    ap.add_argument("--bitmap", default=None)
    ap.add_argument("--awq", default=None)
    ap.add_argument("--no-imatrix", action="store_true")
    args = ap.parse_args(argv)

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    weights, cfg = load_source(src)
    print(f"[load] {len(weights)} tensors", flush=True)

    calib = mx.load(args.calib) if args.calib else {}
    if args.awq:
        n = apply_awq_fold(weights, mx.load(args.awq), cfg)
        print(f"[awq] folded into {len(mx.load(args.awq))} norms, {n} consumers scaled", flush=True)

    bitmap = {}
    if args.bitmap:
        bitmap = json.loads(Path(args.bitmap).read_text())["bits"]

    weights = stack_experts(weights)
    print(f"[stack] {len(weights)} tensors after expert stacking", flush=True)

    quantization: dict[str, dict] = {}
    outw: dict[str, mx.array] = {}
    counts: dict[str, int] = defaultdict(int)
    shrinks: list[float] = []

    for name in sorted(weights):
        w = weights[name]
        if is_keep(name):
            dtype = mx.float32 if any(p in name for p in FP32_KEEPS) else mx.float16
            outw[name] = w.astype(dtype)
            counts["keep"] += 1
            continue
        if w.ndim < 2:
            outw[name] = w.astype(mx.float16)
            counts["keep"] += 1
            continue

        base = name[: -len(".weight")] if name.endswith(".weight") else name

        if "switch_mlp" in name:
            bits = bitmap.get(base, 4 if args.profile == "JANG_4M" else 6)
        elif "word_embeddings" in name or name.startswith("lm_head"):
            bits = EMBED_BITS[args.profile]
        else:
            bits = NON_EXPERT_BITS[args.profile]

        if args.profile == "MXFP8":
            # Real MXFP8: e4m3 elements with a shared power-of-two (e8m0) scale per
            # 32-wide block, via MLX's native mode. NOT affine-8 relabelled.
            # The imatrix clip search is skipped here on purpose: it assumes the
            # affine min/max range, whereas the MX scale is a power of two derived
            # from the block max, so shrinking the range does not map onto it.
            packed, sc = mx.quantize(w, group_size=MX_GROUP_SIZE, bits=8, mode="mxfp8")
            mx.eval(packed, sc)
            outw[f"{base}.weight"] = packed
            outw[f"{base}.scales"] = sc
            quantization[base] = {"mode": "mxfp8", "bits": 8, "group_size": MX_GROUP_SIZE}
            counts["mxfp8"] += 1
            continue

        a2 = None if args.no_imatrix else calib.get(base)
        if a2 is not None and a2.shape[-1] != w.shape[-1]:
            a2 = None                       # never silently misalign the weighting

        packed, sc, bi, f = imatrix_quantize(w, a2, bits)
        mx.eval(packed, sc, bi)
        outw[f"{base}.weight"] = packed
        outw[f"{base}.scales"] = sc.astype(mx.float16)
        outw[f"{base}.biases"] = bi.astype(mx.float16)
        quantization[base] = {"mode": "affine", "bits": bits, "group_size": GROUP_SIZE}
        counts[f"affine{bits}"] += 1
        if a2 is not None:
            shrinks.append(f)

    out_cfg = dict(cfg)
    out_cfg.pop("quantization_config", None)
    is_mx = args.profile == "MXFP8"
    out_cfg["weight_format"] = "mxfp8" if is_mx else "affine"
    out_cfg["quantization"] = {
        "group_size": MX_GROUP_SIZE if is_mx else GROUP_SIZE,
        "bits": 8 if is_mx else NON_EXPERT_BITS[args.profile],
        "mode": "mxfp8" if is_mx else "affine",
    }
    out_cfg["quantization"].update({"per_tensor": quantization})
    out_cfg["jang_profile"] = args.profile

    model_path = out / "model.safetensors"
    mx.save_safetensors(str(model_path), outw, metadata={"format": "mlx"})
    rewrite_aligned_safetensors(model_path)
    (out / "config.json").write_text(json.dumps(out_cfg, indent=2))

    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "chat_template.jinja", "generation_config.json"):
        if (src / f).exists():
            shutil.copy2(src / f, out / f)

    print(f"[done] {dict(counts)}")
    if shrinks:
        print(f"[imatrix] refit on {len(shrinks)} tensors, "
              f"mean shrink {np.mean(shrinks):.3f}, min {min(shrinks):.2f}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
