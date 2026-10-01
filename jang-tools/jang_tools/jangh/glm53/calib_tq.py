"""GLM-5.3-Flash JANGTQ v2 calibration.

Per routed-MoE layer, using the FP8-reference capture (diag.safetensors: 4096 reservoir rows of the MoE input with each
row's top-8 routing + weights; per-expert E[x^2]) and the bf16 source weights:

 1. AWQ alpha search on the TRUE TQ objective. s = m^alpha / geomean, m = pooled E[x^2] of the MoE input.
    Objective = router-weighted output NMSE of TQ2(W*s) vs W on the rows that routed to each sampled expert,
    summed over gate+up. alpha=0 IS the no-AWQ baseline, so AWQ is only chosen when it measurably wins.
 2. Unit error curves: units = (layer, gate_up) [one unit: the fused kernel needs equal bits] and (layer, down).
    Options tq2/tq3/tq4. down inputs go through the bf16 gate/up with the clamped SwiGLU (BRECQ-consistent).
    Row scales use each expert's own imatrix (x-side importance), divided by s^2 after the AWQ fold.
 3. Ablation on the same rows: unweighted vs imatrix-weighted LS scale (proves the imatrix contribution).

Output: calib_tq.json {layer: {awq: {...}, units: {...}}} + awq_tq_scales.safetensors {'model.layers.L.mlp': s}.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from jang_tools.jangh.encode import encode, dequant
from jang_tools.jangh.format import h32

E, D_IN, D_H, LIMIT = 288, 4096, 2048, 10.0
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
BITS = (2, 3, 4)


def unit_bytes(n_rows_total: int, k: int, bits: int) -> int:
    return n_rows_total * k * bits // 8 + n_rows_total * 2   # packed + fp16 row scales


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--diag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--rotation", default="none", choices=["none", "hadamard32"])
    ap.add_argument("--diag2", default=None)
    ap.add_argument("--w2", type=float, default=2.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model)
    wm = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    d = mx.load(args.diag)
    d2 = mx.load(args.diag2) if args.diag2 else None
    from jang_tools.jangh.glm53.pool_stats import pooled
    mods = sorted({k[: -len(".rows")] for k in d if k.endswith(".mlp.experts.rows")},
                  key=lambda p: int(re.search(r"layers\.(\d+)\.", p).group(1)))
    if args.layers != "all":
        want = {int(x) for x in args.layers.split(",")}
        mods = [m for m in mods if int(re.search(r"layers\.(\d+)\.", m).group(1)) in want]
    ROT = args.rotation == "hadamard32"
    rot = (lambda M: h32(M)) if ROT else (lambda M: M)
    res_path = out_dir / (("calib_tq.json" if not ROT else "calib_tq_h32.json").replace(".json", f"{args.tag}.json"))
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    awq_path = out_dir / ("awq_tq_scales.safetensors" if not ROT else f"awq_tq_scales_h32{args.tag}.safetensors")
    awq = dict(mx.load(str(awq_path))) if awq_path.exists() else {}
    mx.eval(list(awq.values()))   # materialize: never let a lazy mmap of awq_path be written back into awq_path
    cache: dict = {}

    def get(name):
        f = wm[name]
        if f not in cache:
            cache.clear(); mx.clear_cache()
            cache[f] = mx.load(str(model_dir / f))
        return cache[f][name]

    t0 = time.time()
    for mod in mods:
        layer = int(re.search(r"layers\.(\d+)\.", mod).group(1))
        if str(layer) in res:
            continue
        X, tidx, tw, ex_imx, m_pool, _ = pooled(d, d2, mod, args.w2)
        counts = np.bincount(tidx.ravel(), minlength=E)
        by_freq = np.argsort(-counts)
        sampled = list(by_freq[: args.experts // 2]) + list(np.linspace(0, E - 1, args.experts - args.experts // 2).astype(int))
        sampled = sorted({int(e) for e in sampled if counts[e] >= 8})
        data = []
        for e in sampled:
            rmask, slot = np.nonzero(tidx == e)
            base = f"model.language_model.layers.{layer}.mlp.experts.{e}"
            Wg = get(base + ".gate_proj.weight").astype(mx.float32)
            Wu = get(base + ".up_proj.weight").astype(mx.float32)
            Wd = get(base + ".down_proj.weight").astype(mx.float32)
            xe = X[mx.array(rmask.astype(np.uint32))]
            we = mx.array(tw[rmask, slot].astype(np.float32))[:, None]
            imx = ex_imx[e]
            g = xe @ Wg.T; u = xe @ Wu.T
            gc = mx.minimum(g, LIMIT); uc = mx.clip(u, -LIMIT, LIMIT)
            a = gc * mx.sigmoid(gc) * uc
            imx_d = mx.mean(a * a, axis=0)          # down-input importance on routed rows (fallback: reservoir)
            mx.eval(xe, we, g, u, a, imx_d)
            data.append(dict(e=e, Wg=Wg, Wu=Wu, Wd=Wd, xe=xe, we=we, imx=imx, g=g, u=u, a=a, dn=a @ Wd.T, imx_d=imx_d))
        mx.eval([x["dn"] for x in data])

        def nmse(ref, got, we):
            return float((we * (got - ref) ** 2).sum()), float((we * ref ** 2).sum())

        # ---- 1. AWQ alpha search (TQ2, gate+up)
        geo = lambda v: v / mx.exp(mx.mean(mx.log(mx.maximum(v, 1e-30))))
        awq_tbl = {}
        for alpha in (ALPHAS if not ROT else [0.0]):
            s = geo(mx.power(mx.maximum(m_pool, 1e-12), alpha)) if alpha > 0 else mx.ones((D_IN,))
            num = den = 0.0
            for x in data:
                for W, ref in ((x["Wg"], x["g"]), (x["Wu"], x["u"])):
                    q, sc = encode(W * s, 2, x["imx"] / (s * s))
                    got = (x["xe"] / s) @ dequant(q, sc, 2).T
                    a_, b_ = nmse(ref, got, x["we"]); num += a_; den += b_
            awq_tbl[alpha] = num / max(den, 1e-12)
        best_alpha = min(awq_tbl, key=awq_tbl.get)
        s_best = geo(mx.power(mx.maximum(m_pool, 1e-12), best_alpha)) if best_alpha > 0 else mx.ones((D_IN,))
        awq[f"model.layers.{layer}.mlp"] = s_best.astype(mx.float32)

        # ---- 3. imatrix ablation (TQ2, gate+up, no AWQ)
        abl = {}
        for mode in (("unweighted", "imatrix") if not ROT else ()):
            num = den = 0.0
            for x in data:
                for W, ref in ((x["Wg"], x["g"]), (x["Wu"], x["u"])):
                    q, sc = encode(W, 2, None if mode == "unweighted" else x["imx"])
                    a_, b_ = nmse(ref, x["xe"] @ dequant(q, sc, 2).T, x["we"]); num += a_; den += b_
            abl[mode] = num / max(den, 1e-12)
        abl = abl or {"unweighted": float("nan"), "imatrix": float("nan")}

        # ---- 2. unit curves with the chosen AWQ
        units = {}
        for bits in BITS:
            ng = dg = nd = dd = 0.0
            for x in data:
                xr = rot(x["xe"] / s_best); ar = rot(x["a"])
                for W, ref in ((x["Wg"], x["g"]), (x["Wu"], x["u"])):
                    q, sc = encode(rot(W * s_best), bits, None if ROT else x["imx"] / (s_best * s_best))
                    a_, b_ = nmse(ref, xr @ dequant(q, sc, bits).T, x["we"]); ng += a_; dg += b_
                q, sc = encode(rot(x["Wd"]), bits, None if ROT else x["imx_d"])
                a_, b_ = nmse(x["dn"], ar @ dequant(q, sc, bits).T, x["we"]); nd += a_; dd += b_
            units[f"tq{bits}"] = {
                "gate_up": {"nmse": ng / max(dg, 1e-12), "bytes": unit_bytes(2 * E * D_H, D_IN, bits)},
                "down": {"nmse": nd / max(dd, 1e-12), "bytes": unit_bytes(E * D_IN, D_H, bits)},
            }
        res[str(layer)] = {"experts_measured": len(data), "awq": {"alpha_nmse": {str(k): v for k, v in awq_tbl.items()},
                           "best_alpha": best_alpha, "gain_vs_none": 1 - awq_tbl[best_alpha] / awq_tbl[0.0]},
                           "imatrix_ablation_tq2_gate_up": abl, "units": units}
        res_path.write_text(json.dumps(res, indent=1))
        mx.save_safetensors(str(awq_path), awq)
        print(f"L{layer:2d} n={len(data)} awq a*={best_alpha} gain={res[str(layer)]['awq']['gain_vs_none']*100:+.1f}% "
              f"imx {abl['unweighted']:.4f}->{abl['imatrix']:.4f} | gu tq2/3/4 "
              + "/".join(f"{units[f'tq{b}']['gate_up']['nmse']:.4f}" for b in BITS) + " | dn "
              + "/".join(f"{units[f'tq{b}']['down']['nmse']:.4f}" for b in BITS)
              + f" ({(time.time()-t0)/60:.1f} min)", flush=True)
        del data; mx.clear_cache()


if __name__ == "__main__":
    main()
