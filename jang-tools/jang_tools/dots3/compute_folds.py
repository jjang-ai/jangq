"""Compute folds.npz (AWQ per-layer + diagonal-imatrix per-expert scales).

Inputs: capture dir (moments.npz + x1/ + router/) and the fp8 source.
- awq[L, H]: from mlp_in second moments; identity (1.0) for dense layer 0.
- imx[L, E+1, I]: per-expert derived-X2 second moments through SOURCE
  gate/up on that expert's routed rows; index E (last) = shared expert.
  Experts with < 8 routed rows keep identity scales.

    python -m jang_tools.dots3.compute_folds <src> <capture_dir> <out.npz>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from .config import Dots3Config
from .folds import Folds, scales_from_moments
from .fp8 import ShardIndex


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("capture", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--max-rows-per-expert", type=int, default=512)
    ap.add_argument("--shared-rows", type=int, default=4096)
    a = ap.parse_args()

    cfg = Dots3Config.load(a.src)
    idx = ShardIndex(a.src)
    mx.set_memory_limit(int(40 * 1024**3))
    L, H, I, E = (cfg.num_hidden_layers, cfg.hidden_size,
                  cfg.moe_intermediate_size, cfg.n_routed_experts)

    mom = np.load(a.capture / "moments.npz")
    counts = mom["counts"].astype(np.float64)
    awq = np.ones((L, H), np.float32)
    for li in range(cfg.first_k_dense_replace, L):
        awq[li] = scales_from_moments(mom["mlp_in"][li] / max(counts[li], 1))

    imx = np.ones((L, E + 1, I), np.float32)
    t0 = time.time()
    for li in range(cfg.first_k_dense_replace, L):
        x1 = np.load(a.capture / "x1" / f"layer_{li:02d}.npy")
        rt = np.load(a.capture / "router" / f"layer_{li:02d}.npz")
        inds = rt["inds"]
        p = f"model.layers.{li}.mlp."
        for e in range(E):
            rows = np.where((inds == e).any(axis=1))[0]
            if rows.size < 8:
                continue
            if rows.size > a.max_rows_per_expert:
                rows = np.random.default_rng(li * 1000 + e).choice(
                    rows, a.max_rows_per_expert, replace=False)
            Xe = mx.array(x1[rows].astype(np.float32))
            Wg = mx.array(idx.read_dequant(p + f"experts.{e}.gate_proj.weight"))
            Wu = mx.array(idx.read_dequant(p + f"experts.{e}.up_proj.weight"))
            g = Xe @ Wg.T
            x2 = mx.multiply(g * mx.sigmoid(g), Xe @ Wu.T)
            m2 = np.asarray((x2.astype(mx.float32) ** 2).mean(axis=0))
            imx[li, e] = scales_from_moments(m2)
            del Wg, Wu, Xe, x2
        # shared expert on a subsample of all rows
        rows = np.random.default_rng(li).choice(
            x1.shape[0], min(a.shared_rows, x1.shape[0]), replace=False)
        Xe = mx.array(x1[rows].astype(np.float32))
        Wg = mx.array(idx.read_dequant(p + "shared_experts.gate_proj.weight"))
        Wu = mx.array(idx.read_dequant(p + "shared_experts.up_proj.weight"))
        g = Xe @ Wg.T
        x2 = mx.multiply(g * mx.sigmoid(g), Xe @ Wu.T)
        imx[li, E] = scales_from_moments(
            np.asarray((x2.astype(mx.float32) ** 2).mean(axis=0)))
        del Wg, Wu, Xe, x2
        mx.clear_cache()
        print(f"  layer {li:2d} folds done ({(time.time()-t0)/60:.1f} min)",
              flush=True)

    f = Folds(awq, imx)
    ident = f.identity_audit()
    finite = bool(np.isfinite(awq).all() and np.isfinite(imx).all())
    np.savez(a.out, awq=awq, imx=imx)
    report = {"identity_max_rel_err": ident, "finite": finite,
              "awq_range": [float(awq.min()), float(awq.max())],
              "imx_range": [float(imx.min()), float(imx.max())],
              "elapsed_min": round((time.time() - t0) / 60, 1)}
    Path(str(a.out) + ".report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    assert finite and ident < 1e-4, "fold audit FAILED"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
