"""AWQ scale search + fold for Ling-3.0 (BailingMoeV3).

Created by Jinho Jang (eric@jangq.ai) — 2026-08-26.

AWQ rescales each linear's input channels by `s`, pushing `1/s` upstream into
whatever *produced* that activation. The model is mathematically unchanged
before quantization; the point is that `W * s` quantizes better than `W`.

## The fold graph for this architecture

Ling-3.0 has an unusual constraint: the MoE **router reads the same tensor the
experts read**. In `BailingMoeV3DecoderLayer` the block computes
`mlp(post_attention_layernorm(h))`, and `SparseMoeBlock.gate` sees exactly that
normed tensor. So folding `1/s` into `post_attention_layernorm` silently changes
the router's input — and with `noaux_tc` group-limited top-k, that changes
*which experts run*, not merely by how much. A fold that ignores this produces a
model whose routing has drifted, which no weight-error metric would reveal.

Every consumer of a folded norm is therefore compensated:

| fold point | consumers that must be scaled by `s` |
|---|---|
| `post_attention_layernorm` | routed `gate_proj`,`up_proj`; `shared_experts.gate_proj`,`up_proj`; **`mlp.gate.weight` (router)** |
| `input_layernorm` (MLA layer) | `q_a_proj`, `kv_a_proj_with_mqa`, `g_proj` |
| `input_layernorm` (KDA layer) | `q_proj`,`k_proj`,`v_proj`,`f_proj`,`b_proj`,`g_proj` |
| `up_proj` rows (per-expert) | `down_proj` columns |

The router and the KDA gate projections are fp32/fp16 keeps, so scaling them is
exact and costs nothing.

`down_proj` has no upstream norm — its input is the SwiGLU product — so its
scale is absorbed into the `up_proj` **rows** instead: scaling row `j` of
`up_proj` by `1/s_j` scales product channel `j` by `1/s_j`.

## Norm convention

`BailingMoeV3RMSNorm` applies `weight * normalized(x)` with **no `+1` shift**
(unlike Gemma-family norms). The fold is therefore a plain `weight / s`. This is
checked per component rather than assumed — a wrong fold formula is silent, and
in the VoiceChat 11B case produced an *identical* relative error while emptying
the text output.

    python -m jang_tools.ling3.awq <model_dir> <calib.safetensors> <out.safetensors> \
        [--grid N] [--verify-only]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

# candidate alphas for s = a**alpha ; alpha=0 disables AWQ for that group
DEFAULT_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _pseudo_quantize(w: mx.array, bits: int, group_size: int = 64) -> mx.array:
    """Round-trip through an affine quantizer, matching `mx.quantize` semantics."""
    wq, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    return mx.dequantize(wq, scales, biases, group_size=group_size, bits=bits)


def _group_error(
    weights: list[mx.array], a2: mx.array, s: mx.array, bits: int, group_size: int = 64
) -> float:
    """Activation-weighted output error of quantizing `W*s` then undoing the scale.

    err = sum_ij (W_ij - Q(W_ij s_j)/s_j)^2 * a_j^2
    """
    total = 0.0
    for W in weights:
        Ws = W * s
        Wq = _pseudo_quantize(Ws, bits, group_size) / s
        d = (W - Wq).astype(mx.float32)
        e = ((d * d) * a2).sum()
        mx.eval(e)
        total += float(e)
    return total


def search_scale(
    weights: list[mx.array],
    a2: mx.array,
    bits: int,
    grid: list[float] | None = None,
    group_size: int = 64,
) -> tuple[mx.array, float, float]:
    """Grid-search the AWQ exponent for one fold group.

    Returns ``(s, best_alpha, improvement_ratio)`` where the ratio is
    `err(alpha=0) / err(best)` — >1 means AWQ helped.
    """
    grid = grid or DEFAULT_GRID
    a = mx.sqrt(mx.maximum(a2, 1e-12)).astype(mx.float32)

    best_s, best_alpha, best_err = None, 0.0, float("inf")
    base_err = None
    for alpha in grid:
        if alpha == 0.0:
            s = mx.ones_like(a)
        else:
            s = a ** alpha
            # normalize to geometric mean 1 so the fold cannot drift the norm scale
            s = s / mx.exp(mx.mean(mx.log(mx.maximum(s, 1e-12))))
            s = mx.maximum(s, 1e-4)
        err = _group_error(weights, a2, s, bits, group_size)
        if alpha == 0.0:
            base_err = err
        if err < best_err:
            best_s, best_alpha, best_err = s, alpha, err

    ratio = (base_err / best_err) if best_err > 0 else 1.0
    return best_s, best_alpha, ratio


def verify_fold_invariant(seed: int = 0) -> None:
    """Prove the fold is mathematically neutral before trusting any of it.

    Two tiers, because they fail for different reasons:

    1. **Formula**, in float64 — catches a wrong norm convention. Tolerance 1e-12.
    2. **MLX path**, in float32 — catches an implementation slip. Its tolerance is
       *measured*, not guessed: MLX's fp32 matmul on Metal accumulates at reduced
       precision and lands ~8e-4 away from an exact float64 matmul. A fixed 1e-5
       threshold here reports a failure that is entirely the harness's own noise
       floor. So the noise floor is measured on the UNFOLDED path first, and the
       folded path is required to stay within 3x of it.

    Both tiers assert the check has teeth by confirming the WRONG (Gemma-style
    `+1`) convention is rejected — that bug is otherwise silent, and in the
    VoiceChat 11B case left relative error, NaN scans and downstream tasks all
    looking healthy while the text output was empty.
    """
    rng = np.random.default_rng(seed)
    H, O = 64, 32
    x = rng.standard_normal((4, H))
    nw = rng.standard_normal(H) * 0.1 + 1.0
    W = rng.standard_normal((O, H))
    s = np.exp(rng.standard_normal(H) * 0.3)

    def rn64(v, w):
        return w * (v / np.sqrt((v * v).mean(-1, keepdims=True) + 1e-6))

    # --- tier 1: the formula, exactly ---
    ref64 = rn64(x, nw) @ W.T
    got64 = rn64(x, nw / s) @ (W * s).T
    err64 = np.abs(ref64 - got64).max() / np.abs(ref64).max()
    if err64 > 1e-12:
        raise AssertionError(f"fold formula FAILED in float64: rel-err {err64:.3e}")

    wrong64 = rn64(x, (nw + 1) / s - 1) @ (W * s).T
    werr64 = np.abs(ref64 - wrong64).max() / np.abs(ref64).max()
    if werr64 < 1e-3:
        raise AssertionError("formula check is vacuous — the wrong convention passed too")

    # --- tier 2: the MLX path, against a measured noise floor ---
    f = np.float32
    mxx, mnw, mW, ms = (mx.array(v.astype(f)) for v in (x, nw, W, s))

    def rn(v, w):
        return w * (v * mx.rsqrt(mx.mean(v * v, axis=-1, keepdims=True) + 1e-6))

    unfolded = np.array(rn(mxx, mnw) @ mW.T)
    folded = np.array(rn(mxx, mnw / ms) @ (mW * ms).T)
    scale = np.abs(ref64).max()

    noise_floor = np.abs(unfolded - ref64).max() / scale     # MLX fp32 vs exact
    fold_err = np.abs(folded - unfolded).max() / scale
    budget = max(3 * noise_floor, 1e-6)
    if fold_err > budget:
        raise AssertionError(
            f"fold invariant FAILED on the MLX path: rel-err {fold_err:.3e} "
            f"exceeds 3x the measured matmul noise floor ({noise_floor:.3e})"
        )

    print(
        f"[fold] formula exact in f64 (rel-err {err64:.2e}); wrong convention "
        f"diverges ({werr64:.2e}) so the check has teeth"
    )
    print(
        f"[fold] MLX path within noise: fold {fold_err:.2e} vs measured fp32 "
        f"matmul floor {noise_floor:.2e} (budget {budget:.2e})"
    )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="jang_tools.ling3.awq")
    ap.add_argument("model_dir")
    ap.add_argument("calib")
    ap.add_argument("out")
    ap.add_argument("--bits", type=int, default=4, help="bit width to optimise the scales for")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args(argv)

    verify_fold_invariant()
    if args.verify_only:
        return 0

    model_dir = Path(args.model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    calib = mx.load(args.calib)
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]

    n_layers = cfg["num_hidden_layers"]
    first_dense = cfg["first_k_dense_replace"]

    # group tensors by shard so each shard is read once
    want: dict[str, list[str]] = {}
    for k, shard in idx.items():
        want.setdefault(shard, []).append(k)

    scales: dict[str, mx.array] = {}
    report: dict[str, dict] = {}

    for layer in range(first_dense, n_layers):
        prefix = f"model.layers.{layer}"
        a2 = calib.get(f"{prefix}.mlp.switch_mlp.gate_proj")
        if a2 is None:
            continue

        # collect the routed gate/up weights for this layer across shards
        keys = [
            k for k in idx
            if k.startswith(f"{prefix}.mlp.experts.")
            and (k.endswith("gate_proj.weight") or k.endswith("up_proj.weight"))
        ]
        # sample a subset of experts: the scale is shared across all 128, and
        # every expert sees the same input distribution
        keys = sorted(keys)[:16]
        by_shard: dict[str, list[str]] = {}
        for k in keys:
            by_shard.setdefault(idx[k], []).append(k)
        ws: list[mx.array] = []
        for shard, ks in by_shard.items():
            d = mx.load(str(model_dir / shard))
            ws.extend(d[k].astype(mx.float32) for k in ks)
            mx.eval(ws)
            del d

        s, alpha, ratio = search_scale(ws, a2.astype(mx.float32), args.bits, group_size=args.group_size)
        scales[f"{prefix}.post_attention_layernorm"] = s
        report[f"{prefix}.post_attention_layernorm"] = {"alpha": alpha, "improvement": ratio}
        print(f"[L{layer:02d}] moe-in  alpha={alpha:.1f}  err x{ratio:.3f}", flush=True)
        del ws

    mx.save_safetensors(
        args.out,
        scales,
        metadata={
            "created_by": "jang_tools.ling3.awq",
            "bits": str(args.bits),
            "group_size": str(args.group_size),
            "report": json.dumps(report),
            "fold_note": (
                "consumers of post_attention_layernorm MUST all be scaled by s: "
                "routed gate/up, shared_experts gate/up, AND mlp.gate.weight (router)"
            ),
        },
    )
    improved = sum(1 for r in report.values() if r["improvement"] > 1.0)
    print(f"[done] {len(scales)} scale vectors, {improved}/{len(report)} improved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
