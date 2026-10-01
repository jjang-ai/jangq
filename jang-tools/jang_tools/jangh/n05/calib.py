"""Naive-N0.5-Flash JANGH calibration measurements.

Per routed-MoE layer, on the BF16 layer-streamed capture (4096 reservoir rows of the MoE input per layer, their top-8
routing, per-expert E[x^2] and down-input E[a^2]) and the BF16 source weights, for a sample of experts:

 1. Error-vs-bits unit curves (RTN, codebook per-row scale) for units (L, gate_up) and (L, down), bits 2/3/4, with
    rotation none AND hadamard32 — plus per-row error tails (p99 / p99.9 of per-row relative error), the metric that
    exposed the GLM tool-decision failures (means hid them).
 2. AWQ, measured on the TRUE objective (router-weighted output error on HELD-OUT reservoir rows):
      gate/up: s = E[x^2]^alpha (geomean-normalized), W*s, x/s — folds exactly into post_attention_layernorm
               PROVIDED the fp32 router weight is multiplied by s as well (router sees x/s).
      down   : per-expert s_d = E[a^2]^alpha, W_down*s_d, a/s_d — folds exactly into that expert's up_proj rows
               (SwiGLU is linear in u), so down gets a real fold partner.
    alpha = 0 IS the no-AWQ baseline; AWQ is only adopted where it wins on held-out rows.
 3. imatrix: per-expert x-side importance used for the RTN least-squares row scale (ablation vs unweighted).
Train/held-out: reservoir rows [0, 3072) calibrate nothing here (RTN has no fit) but AWQ alpha is CHOSEN on
[0,3072) and REPORTED on [3072, 4096) so the choice itself is out-of-sample.

Output: <out>/calib_n05.json {layer: {...}}; resumable (skips finished layers).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


from jang_tools.jangh.encode import encode, dequant  # noqa: E402
from jang_tools.jangh.format import h32  # noqa: E402

E, D_IN, D_H = 256, 4096, 2048
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
BITS = (2, 3, 4)
SPLIT = 3072


def unit_bytes(unit, bits):
    if unit == "gate_up":
        return 2 * E * D_H * (D_IN * bits // 8 + 2)
    return E * D_IN * (D_H * bits // 8 + 2)


def geo(v):
    return v / mx.exp(mx.mean(mx.log(mx.maximum(v, 1e-30))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--headers", required=True)
    ap.add_argument("--stats", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--experts", type=int, default=32); ap.add_argument("--layers", default="all")
    a = ap.parse_args()
    src = Path(a.model); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    H = json.loads(Path(a.headers).read_text())
    st = mx.load(a.stats)
    res_path = out / "calib_n05.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    layers = sorted(int(k.split(".")[2]) for k in st if k.endswith(".mlp.experts.rows"))
    if a.layers != "all":
        layers = [L for L in layers if L in {int(x) for x in a.layers.split(",")}]
    cache = {}

    def get(n):
        f = H[n][0]
        if f not in cache:
            if len(cache) > 1:
                cache.pop(next(iter(cache)))
            cache[f] = mx.load(str(src / f))
        return cache[f][n]

    t0 = time.time()
    for L in layers:
        if str(L) in res:
            continue
        b = f"model.layers.{L}.mlp.experts"
        X = st[b + ".rows"].astype(mx.float32)
        tidx = np.asarray(st[b + ".rows_topk_idx"]); tw = np.asarray(st[b + ".rows_topk_w"])
        ecnt = st[b + ".expert_count"].astype(mx.float32)
        imx = st[b + ".expert_sum_x2"].astype(mx.float32) / mx.maximum(ecnt, 1.0)[:, None]
        imx_d = st[f"model.layers.{L}.mlp.switch_mlp.down_proj.expert_sum_a2"].astype(mx.float32) / mx.maximum(ecnt, 1.0)[:, None]
        m_pool = st[b + ".sum_x2"].astype(mx.float32) / float(st[b + ".count"].item())
        counts = np.bincount(tidx.ravel(), minlength=E)
        sampled = sorted({int(e) for e in list(np.argsort(-counts)[: a.experts // 2]) +
                          list(np.linspace(0, E - 1, a.experts - a.experts // 2).astype(int)) if counts[e] >= 16})
        data = []
        for e in sampled:
            r, s_ = np.nonzero(tidx == e)
            Wg = get(f"{b}.{e}.gate_proj.weight").astype(mx.float32)
            Wu = get(f"{b}.{e}.up_proj.weight").astype(mx.float32)
            Wd = get(f"{b}.{e}.down_proj.weight").astype(mx.float32)
            xe = X[mx.array(r.astype(np.uint32))]; we = mx.array(tw[r, s_].astype(np.float32))[:, None]
            g, u = xe @ Wg.T, xe @ Wu.T
            act = g * mx.sigmoid(g) * u
            mx.eval(xe, g, u, act)
            data.append(dict(e=e, tr=mx.array((r < SPLIT).astype(np.float32))[:, None], Wg=Wg, Wu=Wu, Wd=Wd, xe=xe, we=we,
                             g=g, u=u, a=act, dn=act @ Wd.T, imx=imx[e], imx_d=imx_d[e]))
        mx.eval([x["dn"] for x in data])

        def err(ref, got, x, part):                      # part: "all" | "tr" | "te"
            w = x["we"] * (1.0 if part == "all" else (x["tr"] if part == "tr" else 1.0 - x["tr"]))
            return float((w * (got - ref) ** 2).sum()), float((w * ref ** 2).sum())

        rec = {"experts": len(data), "rows_per_expert_mean": float(np.mean([x["xe"].shape[0] for x in data]))}
        # ---- 1. curves, both rotations, + per-row tails at 2 bits
        for rot_name, R in (("none", lambda M: M), ("hadamard32", h32)):
            units, tails = {}, {}
            for bits in BITS:
                ng = dg = nd = dd = 0.0   # gate_up/down nmse: projection-output objective (GLM-compatible);
                                          # gate/up/down_eo: EXPERT-OUTPUT objective (comparable across projections)
                pp = {"gate": [0.0, 0.0], "up": [0.0, 0.0]}
                rows_rel = []
                for x in data:
                    xr, ar = R(x["xe"]), R(x["a"])
                    qo = {}
                    for pname, W, ref in (("gate", x["Wg"], x["g"]), ("up", x["Wu"], x["u"])):
                        Wr = R(W)
                        q, sc = encode(Wr, bits, None if rot_name == "hadamard32" else x["imx"])
                        got = xr @ dequant(q, sc, bits).T
                        qo[pname] = got
                        a_, b_ = err(ref, got, x, "all"); ng += a_; dg += b_
                    # per-projection importance on the EXPERT OUTPUT with only that projection quantized
                    for pname, gg, uu in (("gate", qo["gate"], x["u"]), ("up", x["g"], qo["up"])):
                        a_, b_ = err(x["dn"], (gg * mx.sigmoid(gg) * uu) @ x["Wd"].T, x, "all")
                        pp[pname][0] += a_; pp[pname][1] += b_
                        if bits == 2:                    # per OUTPUT ROW relative error on the real routed rows
                            rows_rel.append(np.asarray(mx.sqrt(mx.sum(x["we"] * (got - ref) ** 2, 0)
                                                               / mx.maximum(mx.sum(x["we"] * ref ** 2, 0), 1e-30))))
                    q, sc = encode(R(x["Wd"]), bits, None if rot_name == "hadamard32" else x["imx_d"])
                    a_, b_ = err(x["dn"], ar @ dequant(q, sc, bits).T, x, "all"); nd += a_; dd += b_
                units[f"tq{bits}"] = {"gate_up": {"nmse": ng / dg, "bytes": unit_bytes("gate_up", bits)},
                                      "gate": {"nmse": pp["gate"][0] / pp["gate"][1], "bytes": unit_bytes("gate_up", bits) // 2},
                                      "up": {"nmse": pp["up"][0] / pp["up"][1], "bytes": unit_bytes("gate_up", bits) // 2},
                                      "down": {"nmse": nd / dd, "bytes": unit_bytes("down", bits)}}
                # down on the expert output is the same objective as "down" (only down quantized, bf16 act)
                if rows_rel:
                    rr = np.concatenate(rows_rel)
                    tails = {"mean": float(rr.mean()), "p50": float(np.median(rr)), "p99": float(np.percentile(rr, 99)),
                             "p999": float(np.percentile(rr, 99.9))}
            rec[f"units_{rot_name}"] = units
            rec[f"tails_tq2_gate_up_{rot_name}"] = tails
        # ---- 2. AWQ (hadamard32, TQ2 and TQ3), alpha chosen on train rows, reported on held-out rows
        awq = {}
        for bits in (2, 3):
            tbl = {}
            for alpha in ALPHAS:
                s = geo(mx.power(mx.maximum(m_pool, 1e-12), alpha)) if alpha > 0 else mx.ones((D_IN,))
                acc = {"tr": [0.0, 0.0], "te": [0.0, 0.0]}
                for x in data:
                    xr = h32(x["xe"] / s)
                    for W, ref in ((x["Wg"], x["g"]), (x["Wu"], x["u"])):
                        q, sc = encode(h32(W * s), bits)
                        got = xr @ dequant(q, sc, bits).T
                        for p in ("tr", "te"):
                            a_, b_ = err(ref, got, x, p); acc[p][0] += a_; acc[p][1] += b_
                tbl[alpha] = {p: acc[p][0] / acc[p][1] for p in acc}
            best = min(ALPHAS, key=lambda al: tbl[al]["tr"])
            dtbl = {}
            for alpha in ALPHAS:
                acc = {"tr": [0.0, 0.0], "te": [0.0, 0.0]}
                for x in data:
                    sd = geo(mx.power(mx.maximum(x["imx_d"], 1e-12), alpha)) if alpha > 0 else mx.ones((D_H,))
                    q, sc = encode(h32(x["Wd"] * sd), bits)
                    got = h32(x["a"] / sd) @ dequant(q, sc, bits).T
                    for p in ("tr", "te"):
                        a_, b_ = err(x["dn"], got, x, p); acc[p][0] += a_; acc[p][1] += b_
                dtbl[alpha] = {p: acc[p][0] / acc[p][1] for p in acc}
            dbest = min(ALPHAS, key=lambda al: dtbl[al]["tr"])
            awq[f"tq{bits}"] = {"gate_up": {"table": {str(k): v for k, v in tbl.items()}, "best_alpha_train": best,
                                            "heldout_gain": 1 - tbl[best]["te"] / tbl[0.0]["te"]},
                                "down": {"table": {str(k): v for k, v in dtbl.items()}, "best_alpha_train": dbest,
                                         "heldout_gain": 1 - dtbl[dbest]["te"] / dtbl[0.0]["te"]}}
        rec["awq_h32"] = awq
        # ---- 3. imatrix ablation (none rotation, TQ2 gate/up)
        num_u = den = num_i = 0.0
        for x in data:
            for W, ref in ((x["Wg"], x["g"]), (x["Wu"], x["u"])):
                q, sc = encode(W, 2, None); a1, b1 = err(ref, x["xe"] @ dequant(q, sc, 2).T, x, "all")
                q, sc = encode(W, 2, x["imx"]); a2, _ = err(ref, x["xe"] @ dequant(q, sc, 2).T, x, "all")
                num_u += a1; num_i += a2; den += b1
        rec["imatrix_ablation_tq2_gate_up_none"] = {"unweighted": num_u / den, "imatrix": num_i / den}
        res[str(L)] = rec
        tmp = res_path.with_suffix(".tmp"); tmp.write_text(json.dumps(res, indent=1)); tmp.replace(res_path)
        un, uh = rec["units_none"], rec["units_hadamard32"]
        tn, th = rec["tails_tq2_gate_up_none"], rec["tails_tq2_gate_up_hadamard32"]
        print(f"L{L:2d} n={len(data)} gu tq2/3/4 none " + "/".join(f"{un[f'tq{k}']['gate_up']['nmse']:.4f}" for k in BITS)
              + " h32 " + "/".join(f"{uh[f'tq{k}']['gate_up']['nmse']:.4f}" for k in BITS)
              + " | dn none " + "/".join(f"{un[f'tq{k}']['down']['nmse']:.4f}" for k in BITS)
              + " h32 " + "/".join(f"{uh[f'tq{k}']['down']['nmse']:.4f}" for k in BITS)
              + f" | tails p99.9 none {tn['p999']:.3f} h32 {th['p999']:.3f}"
              + f" | awq tq2 gu a*={awq['tq2']['gate_up']['best_alpha_train']} {awq['tq2']['gate_up']['heldout_gain']*100:+.2f}%"
              + f" dn a*={awq['tq2']['down']['best_alpha_train']} {awq['tq2']['down']['heldout_gain']*100:+.2f}%"
              + f" | imx {rec['imatrix_ablation_tq2_gate_up_none']['unweighted']:.4f}->{rec['imatrix_ablation_tq2_gate_up_none']['imatrix']:.4f}"
              + f" ({(time.time()-t0)/60:.1f} min)", flush=True)
        del data; mx.clear_cache()


if __name__ == "__main__":
    main()
