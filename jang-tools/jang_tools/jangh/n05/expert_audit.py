"""Audit EVERY expert of a converted bundle on held-out rows. MLX_ENABLE_TF32=0, run alone.

convert.py verifies 6 experts per layer; the Hessian validator flagged one expert (L47 / 173) whose quantized down_proj
turned out 100x worse than its neighbours. This scans all 47 x 256 experts: expert-output NMSE (curves.py definition)
with only gate+up quantized, only down quantized, and all three, for every expert with >= 16 held-out rows.
Outlier = NMSE > 8 x the layer median of that column AND > 0.05.
"""
import argparse, json, os, sys, time
from pathlib import Path
import mlx.core as mx, numpy as np

from jang_tools.jangh.n05.model import Args, sanitize_layer
from jang_tools.jangh.n05.stream_capture import load_layer_raw
from jang_tools.jangh.encode import dequant
from jang_tools.jangh.format import h32, unpack_bitstream

ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--layers", default="all"); a = ap.parse_args()
assert os.environ.get("MLX_ENABLE_TF32") == "0"
SRC = Path(os.environ["JANGH_SOURCE"])
A = Args.from_config(json.loads((SRC / "config.json").read_text())); H = json.loads((home / "ref/headers.json").read_text())
st = mx.load(str(home / "work/stats_final.safetensors"))
layers = range(1, 48) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
res, cache, t0, flagged = {}, {}, time.time(), []
for L in layers:
    w = sanitize_layer(load_layer_raw(SRC, H, L, cache), L, A.n_routed_experts)
    Wb = {p: w[f"mlp.switch_mlp.{p}_proj.weight"] for p in ("gate", "up", "down")}
    sf = mx.load(str(Path(a.stage) / f"L{L:02d}.safetensors")); Wq = {}
    for p in Wb:
        m = f"model.layers.{L}.mlp.switch_mlp.{p}_proj"; K = Wb[p].shape[-1]; bits = sf[m + ".tq2_packed"].shape[-1] * 32 // K
        parts = []
        for e0 in range(0, A.n_routed_experts, 16):
            parts.append(h32(dequant(unpack_bitstream(sf[m + ".tq2_packed"][e0:e0 + 16], bits, K), sf[m + ".tq2_scales"][e0:e0 + 16], bits)).astype(mx.bfloat16))
            mx.eval(parts[-1])
        Wq[p] = mx.concatenate(parts, axis=0)
    b = f"model.layers.{L}.mlp.experts"
    X = st[b + ".te_rows"].astype(mx.float32); ti = np.asarray(st[b + ".te_rows_topk_idx"]); tw = np.asarray(st[b + ".te_rows_topk_w"])
    rec = {}
    for e in range(A.n_routed_experts):
        r_, s_ = np.nonzero(ti == e)
        if len(r_) < 16:
            continue
        x = X[mx.array(r_.astype(np.int32))]; w2 = mx.array((tw[r_, s_] ** 2).astype(np.float32))[:, None]
        f = lambda p, q: (Wq[p][e] if q else Wb[p][e]).astype(mx.float32)
        g, u = x @ f("gate", 0).T, x @ f("up", 0).T; act = g * mx.sigmoid(g) * u; y = act @ f("down", 0).T
        gq, uq = x @ f("gate", 1).T, x @ f("up", 1).T; aq = gq * mx.sigmoid(gq) * uq
        den = float((w2 * y * y).sum())
        rec[e] = {"rows": int(len(r_)), "gu": float((w2 * (aq @ f("down", 0).T - y) ** 2).sum()) / den,
                  "dn": float((w2 * (act @ f("down", 1).T - y) ** 2).sum()) / den, "all": float((w2 * (aq @ f("down", 1).T - y) ** 2).sum()) / den}
    med = {k: float(np.median([v[k] for v in rec.values()])) for k in ("gu", "dn", "all")}
    out = [(e, k, v[k]) for e, v in rec.items() for k in ("gu", "dn") if v[k] > 8 * med[k] and v[k] > 0.05]
    flagged += [(L,) + o for o in out]
    res[str(L)] = {"experts_scored": len(rec), "median": med, "max": {k: float(max(v[k] for v in rec.values())) for k in med},
                   "outliers": [{"expert": e, "group": k, "nmse": v, "rows": rec[e]["rows"]} for e, k, v in out], "per_expert": rec}
    print(f"L{L:2d} scored {len(rec):3d}  median gu {med['gu']:.4f} dn {med['dn']:.4f} all {med['all']:.4f}  max gu {res[str(L)]['max']['gu']:.4f} dn {res[str(L)]['max']['dn']:.4f}  "
          f"outliers {[(e, k, round(v, 3)) for e, k, v in out]}  ({(time.time()-t0)/60:.1f} min)", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    del w, Wb, Wq, sf, X; mx.clear_cache()
print(f"AUDIT DONE: {len(flagged)} outlier (layer, expert, group): {flagged}", flush=True)
