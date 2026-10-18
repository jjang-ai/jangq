"""JANGT error curves for the N0.5 allocator. MLX_ENABLE_TF32=0, JANGT_HB=128, JANGT_L=12.

Mirrors jang_tools.jangh.n05.curves exactly where it matters, so the numbers are interchangeable with curves.json:
  same experts per layer (curves.json[L]["experts"]), same held-out rows (stats te_rows), same recipe m_min
  (curves.json[L]["gu_m_min"]), same NMSE: router-w^2 weighted expert output, gate+up quantized, down BF16.
Adds for every layer: jt_gate_up2, jt_gate_up3 (JANGT v2 encoder: unit-gain refit, block 32, per-layer signs and
gate/up outlier columns). Resumable per layer (writes <out> after every layer).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from jang_tools.jangt import minitest as MT
from jang_tools.jangt import moe_encode as ME


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", required=True)
    ap.add_argument("--headers", required=True, help="tensor -> [shard file] map of the source")
    ap.add_argument("--stats", required=True)
    ap.add_argument("--hess-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--layers", default="all")
    ap.add_argument("--kbits", default="2,3"); ap.add_argument("--block", type=int, default=32)
    a = ap.parse_args()
    assert os.environ.get("MLX_ENABLE_TF32") == "0"
    from jang_tools.jangh.n05.hess_cpu import HessStore
    from jang_tools.jangh.n05.recipe import hessian, TAU
    from jang_tools.jangh.n05.gptq_stable import quantize_group
    from jang_tools.jangh.format import h32
    mx.set_cache_limit(2 * 2 ** 30)
    C = json.loads(Path(a.curves).read_text()); Hh = json.loads(Path(a.headers).read_text())
    out = Path(a.out); res = json.loads(out.read_text()) if out.exists() else {}
    st = mx.load(a.stats); src = Path(a.model); kbs = [int(k) for k in a.kbits.split(",")]
    layers = sorted(int(k) for k in C) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    t0 = time.time()
    for L in layers:
        if str(L) in res:
            continue
        tl = time.time(); rec = C[str(L)]; ex = rec["experts"]; b = f"model.layers.{L}.mlp.experts"
        HS = HessStore(str(Path(a.hess_dir) / f"L{L:02d}.safetensors"), "gu"); pool = HS.pooled()
        rng = np.random.default_rng(10_000 + L); su = rng.choice([-1.0, 1.0], HS.n).astype(np.float32)
        cols = ME.layer_outlier_cols(HS.mean, HS.wsum, pool)
        Wf = {}
        def get(name):
            f = Hh[name][0]
            if f not in Wf:
                Wf.clear(); Wf[f] = mx.load(str(src / f))
            return Wf[f][name]
        W = {p: [np.array(get(f"{b}.{e}.{p}_proj.weight").astype(mx.float32)) for e in ex] for p in ("gate", "up", "down")}
        Xte = st[b + ".te_rows"].astype(mx.float32); ti = np.asarray(st[b + ".te_rows_topk_idx"]); tw = np.asarray(st[b + ".te_rows_topk_w"])
        te = []
        for i, e in enumerate(ex):
            r, s = np.nonzero(ti == e)
            X = Xte[mx.array(r.astype(np.int32))]; w2 = mx.array((tw[r, s] ** 2).astype(np.float32))[:, None]
            Wg, Wu, Wd = (mx.array(W[p][i]) for p in ("gate", "up", "down"))
            g, u = X @ Wg.T, X @ Wu.T; y = (g * mx.sigmoid(g) * u) @ Wd.T; mx.eval(y)
            te.append(dict(X=X, w2=w2, y=y, Wd=Wd))
        den = sum(float((t["w2"] * t["y"] ** 2).sum()) for t in te)
        mm = float(rec["gu_m_min"])
        preps, Ts, Hgs = [], [], []
        for e in ex:
            Hg = hessian(HS.cov(e), HS.mean[e], HS.rows[e], pool, mm, TAU["gu"]); Hgs.append(Hg)
            pr = MT.prep_hessian(Hg, su, cols); preps.append(pr); Ts.append(ME.ldl_feedback(pr[1], a.block))
        r_out = {"m_min": mm, "outlier_cols": cols.tolist()}
        # JANGH on the SAME Hessians and rows (the reference curves.json used the September capture)
        q = quantize_group([mx.stack([mx.array(w) for w in W["gate"]]), mx.stack([mx.array(w) for w in W["up"]])],
                           Hgs, HS.mean[ex], (2, 3, 4), True)
        for bits in (2, 3, 4):
            num = 0.0
            for i, t in enumerate(te):
                xr = h32(t["X"]); g, u = xr @ q[bits][0][3][i].T, xr @ q[bits][1][3][i].T
                num += float((t["w2"] * ((g * mx.sigmoid(g) * u) @ t["Wd"].T - t["y"]) ** 2).sum())
            r_out[f"jh_gate_up{bits}"] = num / den
        for kb in kbs:
            n = len(ex)
            encs = ME.encode_batch(W["gate"] + W["up"], [su] * 2 * n, [cols] * 2 * n, preps * 2, Ts * 2, block=a.block, kbits=kb)
            num = 0.0
            for i, t in enumerate(te):
                Wg = mx.array(ME.dequant(encs[i], W["gate"][i].shape[1]).astype(np.float32))
                Wu = mx.array(ME.dequant(encs[n + i], W["up"][i].shape[1]).astype(np.float32))
                g, u = t["X"] @ Wg.T, t["X"] @ Wu.T
                num += float((t["w2"] * ((g * mx.sigmoid(g) * u) @ t["Wd"].T - t["y"]) ** 2).sum())
            r_out[f"jt_gate_up{kb}"] = num / den
            r_out[f"jt_relH{kb}"] = float(np.median([x["rel_err_H"] for x in encs]))
        res[str(L)] = r_out
        tmp = out.with_suffix(".tmp"); tmp.write_text(json.dumps(res, indent=1)); tmp.replace(out)
        msg = "  ".join(f"K{kb} {r_out[f'jt_gate_up{kb}']:.5f} vs JANGH {r_out[f'jh_gate_up{kb}']:.5f} (ratio {r_out[f'jt_gate_up{kb}']/r_out[f'jh_gate_up{kb}']:.3f}; Sept curve {rec[f'gate_up{kb}']:.5f})" for kb in kbs)
        print(f"L{L:2d} gate_up  {msg}  ({(time.time()-tl):.0f}s, total {(time.time()-t0)/60:.1f} min)", flush=True)
        del HS, pool, preps, Ts, te, W, Wf, Xte; mx.clear_cache()
    print("JT CURVES DONE", flush=True)


if __name__ == "__main__":
    main()
