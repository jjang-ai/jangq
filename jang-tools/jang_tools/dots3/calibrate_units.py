"""Per-unit quantization option measurement for dots3 routed experts.

For every (layer, projection) routed unit, decode sampled source experts,
quantize at each candidate {bits, group, mode}, and measure ACTIVATION-
WEIGHTED output NMSE on that expert's actually-routed calibration rows
(router weights as row weights — the error that reaches the residual).

down_proj inputs are derived through the SOURCE gate/up (limited-SwiGLU),
BRECQ-consistent with how the final GPTQ pass will sequence.

Scale/bias sidecars round through f16 exactly like converter storage
(DSV4 F16-correction lesson). mxfp4 measured at equal footing for the
~4-bit tier (e8m0 utilization decision is per-measurement, per user).

Output: <out>/unit_scores.json — rows keyed "layer:proj" with
{option: {nmse, bytes}} + routing coverage stats.

    python -m jang_tools.dots3.calibrate_units <src> <capture_dir> <out> \
        [--experts 0,31,63,95,127,159,191,223,255] [--layers all]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from .config import Dots3Config
from .fp8 import ShardIndex

OPTIONS = [
    ("2b_g64", dict(bits=2, group_size=64, mode="affine")),
    ("2b_g32", dict(bits=2, group_size=32, mode="affine")),
    ("3b_g128", dict(bits=3, group_size=128, mode="affine")),
    ("3b_g64", dict(bits=3, group_size=64, mode="affine")),
    ("4b_g64", dict(bits=4, group_size=64, mode="affine")),
    ("mxfp4_g32", dict(bits=4, group_size=32, mode="mxfp4")),
]


def eff_bytes(n_weights: int, bits: int, gs: int, mode: str) -> int:
    if mode == "mxfp4":
        return int(n_weights * (bits + 8 / gs) / 8)      # e8m0 scale only
    return int(n_weights * (bits + 32 / gs) / 8)         # f16 scale + bias


def qdq(w: mx.array, bits: int, group_size: int, mode: str) -> mx.array:
    out = mx.quantize(w, group_size=group_size, bits=bits, mode=mode)
    if mode == "affine":
        wq, s, b = out
        s = s.astype(mx.float16).astype(mx.float32)
        b = b.astype(mx.float16).astype(mx.float32)
        return mx.dequantize(wq, s, b, group_size=group_size, bits=bits)
    wq, s = out
    return mx.dequantize(wq, s, group_size=group_size, bits=bits, mode=mode)


def unit_nmse(W: mx.array, W_hat: mx.array, X: mx.array,
              row_w: mx.array) -> tuple[float, float]:
    """(weighted ||X W^T - X Ŵ^T||^2, weighted ||X W^T||^2)"""
    Y = X @ W.T
    D = X @ (W_hat - W).T
    wgt = row_w[:, None]
    num = float((wgt * D.astype(mx.float32) ** 2).sum())
    den = float((wgt * Y.astype(mx.float32) ** 2).sum()) + 1e-12
    return num, den


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("capture", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--experts", default="0,31,63,95,127,159,191,223,255")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--max-rows-per-expert", type=int, default=512)
    ap.add_argument("--folds", type=Path, default=None,
                    help="measure FOLDED weights on fold-domain inputs")
    a = ap.parse_args()

    from .folds import Folds
    folds = Folds.load(a.folds) if a.folds else Folds.none()
    cfg = Dots3Config.load(a.src)
    idx = ShardIndex(a.src)
    experts = [int(e) for e in a.experts.split(",")]
    layers = (range(cfg.first_k_dense_replace, cfg.num_hidden_layers)
              if a.layers == "all" else [int(x) for x in a.layers.split(",")])
    a.out.mkdir(parents=True, exist_ok=True)
    mx.set_memory_limit(int(40 * 1024**3))

    scores: dict = {}
    t0 = time.time()
    for li in layers:
        x1 = np.load(a.capture / "x1" / f"layer_{li:02d}.npy")
        rt = np.load(a.capture / "router" / f"layer_{li:02d}.npz")
        inds, weights = rt["inds"], rt["weights"].astype(np.float32)
        n_unit_weights = (cfg.n_routed_experts * cfg.moe_intermediate_size *
                          cfg.hidden_size)

        # per-sampled-expert routed rows
        per_e: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for e in experts:
            hit_r, hit_c = np.where(inds == e)
            if hit_r.size == 0:
                continue
            if hit_r.size > a.max_rows_per_expert:
                sel = np.random.default_rng(li * 1000 + e).choice(
                    hit_r.size, a.max_rows_per_expert, replace=False)
                hit_r, hit_c = hit_r[sel], hit_c[sel]
            xe = x1[hit_r].astype(np.float32)
            if folds.awq is not None:
                xe = xe / folds.awq[li][None, :]      # fold-domain inputs
            per_e[e] = (xe, weights[hit_r, hit_c])

        Wg, Wu, Wd = {}, {}, {}
        for e in per_e:
            p = f"model.layers.{li}.mlp.experts.{e}."
            Wg[e] = mx.array(folds.apply(p + "gate_proj.weight",
                                         idx.read_dequant(p + "gate_proj.weight")))
            Wu[e] = mx.array(folds.apply(p + "up_proj.weight",
                                         idx.read_dequant(p + "up_proj.weight")))
            Wd[e] = mx.array(folds.apply(p + "down_proj.weight",
                                         idx.read_dequant(p + "down_proj.weight")))

        for proj in ("gate_proj", "up_proj", "down_proj"):
            row = {}
            for opt_name, spec in OPTIONS:
                num_sum, den_sum = 0.0, 0.0
                for e, (Xe_np, we_np) in per_e.items():
                    Xe = mx.array(Xe_np)
                    we = mx.array(we_np)
                    if proj == "down_proj":
                        g = Xe @ Wg[e].T
                        Xin = mx.multiply(g * mx.sigmoid(g), Xe @ Wu[e].T)
                        W = Wd[e]
                    else:
                        Xin = Xe
                        W = Wg[e] if proj == "gate_proj" else Wu[e]
                    W_hat = qdq(W, **spec)
                    num, den = unit_nmse(W, W_hat, Xin, we)
                    num_sum += num
                    den_sum += den
                row[opt_name] = {
                    "nmse": num_sum / max(den_sum, 1e-12),
                    "wnum": num_sum, "wden": den_sum,
                    "bytes": eff_bytes(n_unit_weights, spec["bits"],
                                       spec["group_size"], spec["mode"]),
                }
            scores[f"{li}:{proj}"] = {
                "options": row,
                "sampled_experts": sorted(per_e.keys()),
                "rows_used": int(sum(v[0].shape[0] for v in per_e.values())),
            }
            print(f"L{li:02d} {proj:10s} " + " ".join(
                f"{k}={v['nmse']:.4f}" for k, v in row.items()), flush=True)
        del Wg, Wu, Wd, per_e, x1
        mx.clear_cache()
        (a.out / "unit_scores.json").write_text(json.dumps(scores, indent=1))
    print(f"done {len(scores)} units in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
