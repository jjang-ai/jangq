"""AWQ scale selection for Hy3, chosen by MEASURED quantization error.

Created by Jinho Jang (eric@jangq.ai) — 2026-07-09.

Background
----------
The MiniMax-M3 converter derived AWQ scales as ``s = clip(max|x|^0.5, min=1.0)``.
Measured on Hy3 (2026-07-09) that formula is inert: Hy3's post-attention-norm
activations have median ``max|x| = 0.62``, and in layers 1..40 *every* channel
has ``max|x| < 1.0`` — so the floor pins ``s = 1.0`` on 70.7% of all channels
and on 100% of channels in half the network. AWQ would silently do nothing
exactly where the 2-bit routed experts need protection most.

Rather than swap in another guessed formula, this module *measures*. AWQ's
scaling is an exact identity on the unquantized forward:

    y = W @ x = (W * s) @ (x / s)

so the only thing a scale choice changes is how the quantizer rounds ``W * s``.
The true objective is therefore

    err(s) = || dequant(quant(W * s)) @ (x / s)  -  W @ x ||_F

evaluated on REAL routed-expert weights and REAL captured activation rows
(``act_sample`` from ``jang_tools.hy3.awq_capture``), at the exact bit width /
group size the converter will use.

Candidate scale families (all normalized so ``s`` is geometrically centered on
1 — a uniform rescale of every channel is a no-op for the identity but shifts
the weight magnitude into or out of the quantizer's sweet spot):

    s_j = (stat_j + eps)^alpha,   stat in {max|x|, mean|x|},  alpha grid
    s   = s / sqrt(s.max() * s.min())          # llm-awq normalization
    s   = clip(s, 1/clip_max, clip_max)        # guard dead channels

``alpha = 0`` reduces to ``s = 1`` (i.e. no AWQ), so the search always includes
the do-nothing baseline and can only pick AWQ when it actually helps.

Usage:
  python -m jang_tools.hy3.awq_search \
      --model /Volumes/EricsLLMDrive/sources/Hy3 \
      --stats /Volumes/EricsLLMDrive/sources/Hy3-awq-stats.safetensors \
      --out   /Volumes/EricsLLMDrive/sources/Hy3-awq-scales.safetensors \
      --bits 2 --group-size 128 --experts 4 --probe-layers 1,20,40,60,79
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

EPS = 1e-6
_ALPHA_GRID = (0.0, 0.15, 0.25, 0.35, 0.5, 0.65, 0.8)
_STATS = ("absmean", "max")


def make_scale(stat: np.ndarray, alpha: float, clip_max: float = 8.0) -> np.ndarray:
    """(stat + eps)^alpha, geometrically centered on 1, dead-channel clipped."""
    if alpha == 0.0:
        return np.ones_like(stat, dtype=np.float32)
    s = np.power(stat.astype(np.float64) + EPS, alpha)
    s = s / np.sqrt(s.max() * s.min())
    s = np.clip(s, 1.0 / clip_max, clip_max)
    return s.astype(np.float32)


def _quant_dequant(w: np.ndarray, bits: int, group_size: int) -> np.ndarray:
    """Round-trip through the exact kernel the converter uses (mx.quantize)."""
    import mlx.core as mx

    a = mx.array(w.astype(np.float32))
    qw, qs, qb = mx.quantize(a, group_size=group_size, bits=bits)
    deq = mx.dequantize(qw, qs, qb, group_size=group_size, bits=bits)
    out = np.array(deq).astype(np.float32)
    del a, qw, qs, qb, deq
    mx.clear_cache()
    return out


def scale_error(
    w: np.ndarray,       # (out, in) fp32 expert weight
    x: np.ndarray,       # (n, in)   fp32 real activation rows
    s: np.ndarray,       # (in,)     fp32 candidate scale
    bits: int,
    group_size: int,
) -> float:
    """Relative Frobenius error of the AWQ-transformed quantized forward."""
    y_ref = x @ w.T                                   # (n, out) exact
    w_q = _quant_dequant(w * s[None, :], bits, group_size)
    y_q = (x / s[None, :]) @ w_q.T
    denom = np.linalg.norm(y_ref)
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm(y_q - y_ref) / denom)


def _load_expert(src: Path, wm: dict, name: str) -> np.ndarray:
    import torch  # noqa: F401
    from safetensors import safe_open

    with safe_open(str(src / wm[name]), framework="pt") as f:
        return f.get_tensor(name).float().numpy()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Measured AWQ scale search for Hy3")
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--stats", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--bits", type=int, default=2, help="routed gate/up bit width")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--experts", type=int, default=4, help="experts probed per layer")
    ap.add_argument("--probe-layers", default="1,20,40,60,79")
    ap.add_argument("--clip-max", type=float, default=8.0)
    args = ap.parse_args(argv)

    from safetensors.numpy import load_file, save_file

    src = args.model
    cfg = json.loads((src / "config.json").read_text())
    NL = int(cfg["num_hidden_layers"])
    first_dense = int(cfg.get("first_k_dense_replace", 0))
    wm = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]

    stats = load_file(str(args.stats))
    probe = [int(x) for x in args.probe_layers.split(",") if x.strip()]

    print(f"  AWQ search: bits={args.bits} gs={args.group_size} "
          f"experts/layer={args.experts} probe_layers={probe}", flush=True)
    print(f"  objective: ||Q(W*s)@(x/s) - W@x||_F / ||W@x||_F  "
          f"(alpha=0 == no AWQ baseline)\n", flush=True)

    # ── grid search on probe layers ──
    t0 = time.time()
    totals: dict[tuple[str, float], list[float]] = {}
    for li in probe:
        x = stats[f"model.layers.{li}.mlp.act_sample"]  # (n, H)
        ws = [
            _load_expert(src, wm, f"model.layers.{li}.mlp.experts.{e}.gate_proj.weight")
            for e in range(args.experts)
        ]
        line = []
        for stat_name in _STATS:
            stat = stats[f"model.layers.{li}.mlp.input_{stat_name}"]
            for alpha in _ALPHA_GRID:
                s = make_scale(stat, alpha, args.clip_max)
                errs = [scale_error(w, x, s, args.bits, args.group_size) for w in ws]
                e = float(np.mean(errs))
                totals.setdefault((stat_name, alpha), []).append(e)
                if stat_name == "absmean":
                    line.append(f"a{alpha:g}={e:.4f}")
        print(f"    L{li:<3d} absmean: {'  '.join(line)}", flush=True)

    print(f"\n  mean relative error across probe layers "
          f"({time.time()-t0:.0f}s):", flush=True)
    ranked = sorted(totals.items(), key=lambda kv: float(np.mean(kv[1])))
    baseline = float(np.mean(totals[("absmean", 0.0)]))
    for (stat_name, alpha), errs in ranked:
        m = float(np.mean(errs))
        tag = "  <-- no-AWQ baseline" if alpha == 0.0 else ""
        gain = (baseline - m) / baseline * 100.0
        print(f"    stat={stat_name:<8} alpha={alpha:<5g} err={m:.5f} "
              f"({gain:+.1f}% vs baseline){tag}", flush=True)

    (best_stat, best_alpha), best_errs = ranked[0]
    best = float(np.mean(best_errs))
    if best_alpha == 0.0:
        raise SystemExit(
            "\n  REFUSING to emit scales: the no-AWQ baseline (alpha=0) won the "
            "search. AWQ would not improve this quantization — investigate "
            "(wrong stat? wrong bits? bad calibration?) before shipping."
        )
    print(f"\n  WINNER: stat={best_stat} alpha={best_alpha} "
          f"err={best:.5f} vs no-AWQ {baseline:.5f} "
          f"({(baseline-best)/baseline*100:+.1f}%)", flush=True)

    # ── emit scales for every sparse layer ──
    out: dict[str, np.ndarray] = {}
    for li in range(first_dense, NL):
        stat = stats[f"model.layers.{li}.mlp.input_{best_stat}"]
        s = make_scale(stat, best_alpha, args.clip_max)
        out[f"model.layers.{li}.mlp.input_scale"] = s
    n_at_one = sum(int((v == 1.0).sum()) for v in out.values())
    n_total = sum(v.size for v in out.values())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(args.out))
    meta = {
        "stat": best_stat,
        "alpha": best_alpha,
        "bits": args.bits,
        "group_size": args.group_size,
        "clip_max": args.clip_max,
        "probe_layers": probe,
        "experts_per_layer": args.experts,
        "err_awq": best,
        "err_no_awq": baseline,
        "improvement_pct": (baseline - best) / baseline * 100.0,
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n  wrote {len(out)} layer scales -> {args.out}", flush=True)
    print(f"  inert channels (s==1.0): {n_at_one}/{n_total} "
          f"({n_at_one/n_total*100:.2f}%)", flush=True)
    print(f"  meta -> {args.out.with_suffix('.meta.json')}", flush=True)


if __name__ == "__main__":
    main()
