"""Per-layer error-vs-bits curves with the ACTUAL recipe, on held-out rows. MLX_ENABLE_TF32=0.

For each MoE layer and a sample of experts: expert-output NMSE (router-w^2 weighted, unseen documents) for
  gate@b (up, down BF16), up@b (gate, down BF16), gate_up@b (both, down BF16), down@b (gate, up BF16), b in 2,3,4
plus the MLX affine RTN references (2-bit g128, 3-bit g64) for gate_up and down, and one joint check (2,2,3).
Resumable: writes <out> after every layer; layers whose Hessian file does not exist yet are skipped.
"""
from __future__ import annotations

import os
import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


from jang_tools.jangh.n05.gptq_stable import HessStore, quantize_group  # noqa: E402
from jang_tools.jangh.n05.recipe import hessian, CAND, TAU  # noqa: E402
from jang_tools.jangh.n05 import gptq_stable  # noqa: E402
from jang_tools.jangh.format import h32  # noqa: E402

BITS = (2, 3, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True); ap.add_argument("--hess-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--experts", type=int, default=6); ap.add_argument("--layers", default="all")
    a = ap.parse_args()
    import os
    assert os.environ.get("MLX_ENABLE_TF32") == "0"
    SRC = Path(os.environ["JANGH_SOURCE"]); Hh = json.loads(Path(os.environ["JANGH_HEADERS"]).read_text())
    out = Path(a.out); res = json.loads(out.read_text()) if out.exists() else {}
    st = mx.load(a.stats)
    layers = range(1, 48) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    t0 = time.time()
    for L in layers:
        b = f"model.layers.{L}.mlp.experts"
        hp = Path(a.hess_dir) / f"L{L:02d}.safetensors"
        if str(L) in res or not hp.exists() or b + ".te_rows" not in st:
            continue
        tl = time.time()
        Wf = mx.load(str(SRC / Hh[f"{b}.0.gate_proj.weight"][0]))
        HG, HD = HessStore(str(hp), "gu"), HessStore(str(hp), "dn")
        Xte = st[b + ".te_rows"].astype(mx.float32); ti = np.asarray(st[b + ".te_rows_topk_idx"]); tw = np.asarray(st[b + ".te_rows_topk_w"])
        tc = np.bincount(ti.ravel(), minlength=256)
        ok = [int(e) for e in np.argsort(-tc) if tc[e] >= 32]
        rng = np.random.default_rng(500 + L)
        ex = sorted(set(ok[:2]) | set(int(e) for e in rng.choice(ok[2:], a.experts - 2, replace=False)))
        B = len(ex)
        W = {p: mx.stack([Wf[f"{b}.{e}.{p}_proj.weight"].astype(mx.float32) for e in ex]) for p in ("gate", "up", "down")}
        te = []
        for i, e in enumerate(ex):
            r, s = np.nonzero(ti == e)
            X = Xte[mx.array(r.astype(np.int32))]; w2 = mx.array((tw[r, s] ** 2).astype(np.float32))[:, None]
            g, u = X @ W["gate"][i].T, X @ W["up"][i].T
            act = g * mx.sigmoid(g) * u; y = act @ W["down"][i].T
            mx.eval(X, g, u, act, y); te.append(dict(X=X, w2=w2, g=g, u=u, a=act, y=y))
        den = sum(float((t["w2"] * t["y"] ** 2).sum()) for t in te)
        pg, pd = HG.pooled(), HD.pooled()
        Sx = [HG.cov(e) for e in ex]; Sa = [HD.cov(e) for e in ex]
        def nm(fg=None, fu=None, fd=None):
            num = 0.0
            for i, t in enumerate(te):
                xr = h32(t["X"])
                g = xr @ fg[i].T if fg is not None else t["g"]
                u = xr @ fu[i].T if fu is not None else t["u"]
                act = g * mx.sigmoid(g) * u if (fg is not None or fu is not None) else t["a"]
                y = h32(act) @ fd[i].T if fd is not None else act @ W["down"][i].T
                num += float((t["w2"] * (y - t["y"]) ** 2).sum())
            return num / den

        rec = {"experts": ex, "te_rows": [int(tc[e]) for e in ex], "hess_rows": [int(HG.rows[e]) for e in ex], "select": {}}
        # ---- recipe selection per projection group on held-out rows (bits 2 and 3), then bits 4 with the winner
        best = {}
        for grp, names, S_, HS_, pl in (("gu", ("gate", "up"), Sx, HG, pg), ("dn", ("down",), Sa, HD, pd)):
            trials = {}
            for m_min in CAND[grp]:
                Hs = [hessian(S_[i], HS_.mean[e], HS_.rows[e], pl, m_min, TAU[grp]) for i, e in enumerate(ex)]
                q = quantize_group([W[n] for n in names], Hs, HS_.mean[ex], (2, 3), True)
                extra = max(gptq_stable.LAST_EXTRA_DAMP)
                if grp == "gu":
                    sc = [nm(fg=q[b_][0][3], fu=q[b_][1][3]) for b_ in (2, 3)]
                else:
                    sc = [nm(fd=q[b_][0][3]) for b_ in (2, 3)]
                trials[m_min] = dict(q=q, score=sc, extra=extra, Hs=Hs)
                rec["select"][f"{grp}_m{m_min:g}"] = {"nmse2": sc[0], "nmse3": sc[1], "extra_damp_max": extra}
            # winner: lowest geometric mean of the 2-bit and 3-bit held-out errors
            win = min(trials, key=lambda m_: np.sqrt(trials[m_]["score"][0] * trials[m_]["score"][1]))
            q4 = quantize_group([W[n] for n in names], trials[win]["Hs"], HS_.mean[ex], (4,), True)
            best[grp] = {**trials[win]["q"], **q4}
            rec[f"{grp}_m_min"] = win
            del trials
        gu, dn = best["gu"], best["dn"]
        for bits in BITS:
            Wg, Wu, Wd = gu[bits][0][3], gu[bits][1][3], dn[bits][0][3]
            rec[f"gate{bits}"] = nm(fg=Wg); rec[f"up{bits}"] = nm(fu=Wu); rec[f"gate_up{bits}"] = nm(fg=Wg, fu=Wu)
            rec[f"down{bits}"] = nm(fd=Wd)
        rec["joint_g2u2d2"] = nm(fg=gu[2][0][3], fu=gu[2][1][3], fd=dn[2][0][3])
        rec["joint_g2u2d3"] = nm(fg=gu[2][0][3], fu=gu[2][1][3], fd=dn[3][0][3])
        rec["joint_g2u3d3"] = nm(fg=gu[2][0][3], fu=gu[3][1][3], fd=dn[3][0][3])
        rec["joint_g3u3d3"] = nm(fg=gu[3][0][3], fu=gu[3][1][3], fd=dn[3][0][3])
        for bits, gs in ((2, 128), (3, 64)):
            aq = lambda M: mx.stack([(lambda q, s, bi: mx.dequantize(q, s, bi, group_size=gs, bits=bits).astype(mx.float32))(
                *mx.quantize(M[i].astype(mx.bfloat16), group_size=gs, bits=bits)) for i in range(B)])
            Ag, Au, Ad = aq(W["gate"]), aq(W["up"]), aq(W["down"])
            num_gu = num_d = num_j = 0.0
            for i, t in enumerate(te):
                g, u = t["X"] @ Ag[i].T, t["X"] @ Au[i].T
                act = g * mx.sigmoid(g) * u
                num_gu += float((t["w2"] * (act @ W["down"][i].T - t["y"]) ** 2).sum())
                num_d += float((t["w2"] * (t["a"] @ Ad[i].T - t["y"]) ** 2).sum())
                num_j += float((t["w2"] * (act @ Ad[i].T - t["y"]) ** 2).sum())
            rec[f"affine_gate_up{bits}"] = num_gu / den; rec[f"affine_down{bits}"] = num_d / den; rec[f"affine_joint{bits}{bits}"] = num_j / den
        res[str(L)] = rec
        tmp = out.with_suffix(".tmp"); tmp.write_text(json.dumps(res, indent=1)); tmp.replace(out)
        print(f"L{L:2d} [gu m{rec['gu_m_min']:g} dn m{rec['dn_m_min']:g}] gate 2/3/4 {rec['gate2']:.4f}/{rec['gate3']:.4f}/{rec['gate4']:.4f}  up {rec['up2']:.4f}/{rec['up3']:.4f}/{rec['up4']:.4f}  "
              f"gate_up {rec['gate_up2']:.4f}/{rec['gate_up3']:.4f}/{rec['gate_up4']:.4f}  down {rec['down2']:.4f}/{rec['down3']:.4f}/{rec['down4']:.4f} | "
              f"joint g2u2d3 {rec['joint_g2u2d3']:.4f} g2u3d3 {rec['joint_g2u3d3']:.4f} g3u3d3 {rec['joint_g3u3d3']:.4f} | "
              f"affine gu2 {rec['affine_gate_up2']:.4f} dn2 {rec['affine_down2']:.4f} dn3 {rec['affine_down3']:.4f} joint22 {rec['affine_joint22']:.4f} 33 {rec['affine_joint33']:.4f} "
              f"({(time.time()-tl)/60:.1f} min, total {(time.time()-t0)/60:.1f})", flush=True)
        del Wf, HG, HD, pg, pd, Sx, Sa, gu, dn, best; mx.clear_cache()
    print("CURVES PASS DONE", flush=True)


if __name__ == "__main__":
    main()
