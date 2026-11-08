"""Routing-flip sensitivity for the bit allocation . Run with MLX_ENABLE_TF32=0.

WHY (measured on Naive-N0.5-Flash): this model's top-8 expert selection sits on very thin
margins (8th vs 9th choice gap: median 1e-3..1e-2, 7-48% of tokens below 1e-3) and the 8th expert carries 5-12% of the
MoE output. ANY perturbation (relative weight noise 1e-3 in ONE layer, or just fp32 instead of bf16 arithmetic on the
unquantized weights) is amplified by selection flips to KL ~0.3 / ~21% top-1 flips. With the selection FROZEN to the
baseline's the same perturbation costs KL 0.001. So the quantity a quantizer can control is: how many selection flips
its error causes downstream, to first order.

Method: one BRANCH per (MoE layer L, group g in gu|dn): layer L's experts get mean-free relative weight noise (the
response to each expert's mean input is removed, as the uncentered-Hessian GPTQ does); every branch then runs through
the later layers with the BASELINE's expert selection and routing weights (linear propagation). At every later MoE
layer M the branch's OWN selection is computed and compared with the baseline's:
    flips[L,g][M] = share of tokens whose top-8 set would change
    k[L,g][M]     = flips / sqrt(expert-output NMSE of the perturbation)      (curves.py NMSE definition)
plan.py --flips: predicted flips of a plan = sum_M sqrt( sum_{L<M,g} k[L,g][M]^2 * NMSE[L,g](bits) ).
Controls (fail closed): null branch (no noise) must show 0 flips and KL 0; batched == per-sequence forward.
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

from jang_tools.jangh.n05.model import Args, DecoderLayer, sanitize_layer, route  # noqa: E402
from jang_tools.jangh.n05.stream_capture import load_layer_raw  # noqa: E402
from jang_tools.jangh.encode import encode, dequant  # noqa: E402
from jang_tools.jangh.format import h32  # noqa: E402

# (T, minimum document length, {domain: documents})
BUCKETS = [(256, 150, {"coding": 6, "cybersec": 5, "chinese": 2, "science": 1, "general": 1, "academic_mc": 1}),
           (768, 400, {"tool_conv": 4, "agentic": 3, "general": 2, "coding": 1})]
PROJ = {"gu": ("gate", "up"), "dn": ("down",)}
SIGMA = {"gu": 0.6, "dn": 0.3}     # chosen so that the expert-output NMSE is at the level of 2-3 bit GPTQ (1e-2): well above
                                    # the bf16 rounding noise that every branch carries anyway (overridable on the CLI)


def fwd(layer, x, moe, base_rt=None):
    """DecoderLayer forward; with base_rt=(idx, w) the MoE uses THAT selection and reports where its own would differ."""
    h = x + layer.self_attn(layer.input_layernorm(x))
    xm = layer.post_attention_layernorm(h)
    if not moe:
        return h + layer.mlp(xm), None, None
    a = layer.mlp.a
    idx, w = route(layer.mlp.gate, xm, a.num_experts_per_tok, a.norm_topk_prob, a.routed_scaling_factor)
    flip = None
    if base_rt is not None:
        flip = mx.any(mx.sort(idx, axis=-1) != mx.sort(base_rt[0], axis=-1), axis=-1)
        idx, w = base_rt
    y = layer.mlp.switch_mlp(xm, idx)
    return h + mx.sum(y * w[..., None].astype(y.dtype), axis=-2).astype(x.dtype), (idx, w), flip


def mean_free_noise(W, mu, sigma, seed):
    """W (E,o,i) bf16, mu (E,i) fp32 -> W + sigma * rms_row * N with (noise @ mu) = 0 per expert, bf16."""
    mx.random.seed(seed)
    outs = []
    for e0 in range(0, W.shape[0], 32):
        Wf = W[e0:e0 + 32].astype(mx.float32); m = mu[e0:e0 + 32]
        n = sigma * mx.sqrt(mx.mean(Wf * Wf, axis=-1, keepdims=True)) * mx.random.normal(Wf.shape)
        n = n - (n @ m[:, :, None]) * m[:, None, :] / mx.maximum(mx.sum(m * m, axis=-1), 1e-30)[:, None, None]
        outs.append((Wf + n).astype(mx.bfloat16)); mx.eval(outs[-1])
    return mx.concatenate(outs, axis=0)


def rtn(W, bits):
    """(E, o, i) bf16 -> codebook RTN in the rotated basis, returned DEQUANTIZED IN THE ORIGINAL BASIS (bf16)."""
    outs = []
    for e0 in range(0, W.shape[0], 32):
        q, s = encode(h32(W[e0:e0 + 32].astype(mx.float32)), bits, None)
        outs.append(h32(dequant(q, s, bits)).astype(mx.bfloat16)); mx.eval(outs[-1])
    return mx.concatenate(outs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--headers", required=True); ap.add_argument("--seqs", required=True)
    ap.add_argument("--stats", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--sigma-gu", type=float, default=SIGMA["gu"]); ap.add_argument("--sigma-dn", type=float, default=SIGMA["dn"])
    ap.add_argument("--max-layers", type=int, default=0, help="debug: stop after this many layers (weights then NOT valid)")
    a = ap.parse_args()
    SIGMA["gu"], SIGMA["dn"] = a.sigma_gu, a.sigma_dn
    assert os.environ.get("MLX_ENABLE_TF32") == "0", "run with MLX_ENABLE_TF32=0"
    src = Path(a.model)
    A = Args.from_config(json.loads((src / "config.json").read_text()))
    H = json.loads(Path(a.headers).read_text())
    st = mx.load(a.stats)
    rows = [json.loads(l) for l in open(a.seqs)]
    n_cal, by = 0, {}
    for j, r in enumerate(rows):
        if not r["ref"]:
            if n_cal % 10 == 9:                                   # identical to stream_capture's cal_te rule
                by.setdefault(r["domain"], []).append(j)
            n_cal += 1
    rng = np.random.default_rng(77)
    taken, IDS, VALID, sel_all, npos = set(), [], [], [], 0
    for T, lmin, quota in BUCKETS:
        sel = []
        for d, n in quota.items():
            c = [j for j in by.get(d, []) if len(rows[j]["ids"]) >= lmin and j not in taken]
            if len(c) < n:
                print(f"INVALID: domain {d} has {len(c)} held-out documents of >= {lmin} tokens, need {n}"); sys.exit(2)
            pick = sorted(c, key=lambda j: len(rows[j]["ids"]))[:n] if d == "tool_conv" else [int(x) for x in rng.choice(c, n, replace=False)]
            sel += pick; taken |= set(pick)
        ids = np.full((len(sel), T), 151645, np.int32); nreal = np.zeros(len(sel), np.int64)
        for b, j in enumerate(sel):
            x = rows[j]["ids"][:T]; ids[b, :len(x)] = x; nreal[b] = len(x)
        IDS.append(ids); VALID.append(mx.array(np.arange(T)[None, :] < (nreal[:, None] - 1)))
        npos += int(np.maximum(nreal - 1, 0).sum()); sel_all += sel
        print(f"bucket T={T}: {len(sel)} documents {dict(quota)}, lengths {[len(rows[j]['ids']) for j in sel]}, decisions inside T: "
              f"{sum(1 for j in sel for p_, _ in rows[j].get('decisions', []) if p_ < T - 1)}", flush=True)
    NB = [x.shape[0] for x in IDS]
    print(f"scored positions {npos}", flush=True)
    emb = mx.load(str(src / H["model.embed_tokens.weight"][0]))["model.embed_tokens.weight"]
    HS = [[emb[mx.array(x)].astype(mx.bfloat16) for x in IDS]]; mx.eval(HS[0]); del emb     # HS[branch][bucket]
    tags = ["base"]; FL = {}
    res = {"documents": sel_all, "buckets": [[t_, n_] for (t_, _, _), n_ in zip(BUCKETS, NB)], "positions": npos, "layers": {}}
    cache, t0 = {}, time.time()
    nL = a.max_layers or A.num_hidden_layers
    for i in range(nL):
        tl = time.time()
        layer = DecoderLayer(A, i)
        layer.load_weights(list(sanitize_layer(load_layer_raw(src, H, i, cache), i, A.n_routed_experts).items()), strict=True)
        mx.eval(layer.parameters())
        new, rec = [], {}
        moe = bool(A.moe_layer_freq[i])
        # baseline first (own routing); its selection is what every branch uses in this layer
        base_out, base_rt = [], []
        for h in HS[0]:
            o, rt, _ = fwd(layer, h, moe); mx.eval(o, rt); base_out.append(o); base_rt.append(rt)
        if i == 1:
            one = layer(HS[0][0][:1]); mx.eval(one)
            d = float(mx.abs(one.astype(mx.float32) - base_out[0][:1].astype(mx.float32)).max()) / float(mx.abs(one.astype(mx.float32)).max())
            print(f"control batch: batched vs single-sequence layer-1 output rel max diff {d:.2e}", flush=True)
            if not d < 2e-2:
                print("INVALID: batched forward differs from the per-sequence forward"); sys.exit(2)
        nb = len(HS)
        adv = []
        for q, (T, _, _) in enumerate(BUCKETS):
            if nb == 1:
                adv.append([base_out[q]]); continue
            flat = mx.concatenate([HS[k][q] for k in range(1, nb)], axis=0)
            rt_i = mx.concatenate([base_rt[q][0]] * (nb - 1), axis=0) if moe else None
            rt_w = mx.concatenate([base_rt[q][1]] * (nb - 1), axis=0) if moe else None
            outs, fl, ch = [], [], max(1, 16384 // T)
            for s_ in range(0, flat.shape[0], ch):
                o, _, f = fwd(layer, flat[s_:s_ + ch], moe, (rt_i[s_:s_ + ch], rt_w[s_:s_ + ch]) if moe else None)
                mx.eval(o, f); outs.append(o)
                if moe:
                    fl.append(f)
            flat = mx.concatenate(outs, axis=0)
            adv.append([base_out[q]] + [flat[k * NB[q]:(k + 1) * NB[q]] for k in range(nb - 1)])
            if moe:
                f = np.asarray(mx.concatenate(fl, axis=0)).reshape(nb - 1, NB[q], T) & np.asarray(VALID[q])[None]
                for k in range(nb - 1):
                    FL[tags[k + 1]][i] = FL[tags[k + 1]].get(i, 0) + int(f[k].sum())
            del outs, flat
        if moe:
            sw = layer.mlp.switch_mlp
            Wb = {p: getattr(sw, p + "_proj").weight for p in ("gate", "up", "down")}
            bk = f"model.layers.{i}.mlp.experts"
            Xte = st[bk + ".te_rows"].astype(mx.float32); ti = np.asarray(st[bk + ".te_rows_topk_idx"]); tw = np.asarray(st[bk + ".te_rows_topk_w"])
            erows = {}
            mux = np.tile(np.asarray(mx.mean(Xte, axis=0))[None], (A.n_routed_experts, 1)).astype(np.float32)
            mua = np.zeros((A.n_routed_experts, A.moe_intermediate_size), np.float32); pa, pn = 0.0, 0
            for e in range(A.n_routed_experts):
                r_, s_ = np.nonzero(ti == e)
                if len(r_) < 8:
                    continue
                X = Xte[mx.array(r_.astype(np.int32))]; w2 = mx.array((tw[r_, s_].astype(np.float32)) ** 2)[:, None]
                g0, u0 = X @ Wb["gate"][e].astype(mx.float32).T, X @ Wb["up"][e].astype(mx.float32).T
                a0 = g0 * mx.sigmoid(g0) * u0; mx.eval(a0)
                erows[e] = (X, w2, g0, u0, a0)
                mux[e] = np.asarray(mx.sum(w2 * X, axis=0) / mx.sum(w2)); mua[e] = np.asarray(mx.sum(w2 * a0, axis=0) / mx.sum(w2))
                pa = pa + mua[e]; pn += 1
            if pn < 128:
                print(f"INVALID: layer {i}: only {pn} experts have held-out rows"); sys.exit(2)
            for e in range(A.n_routed_experts):
                if e not in erows:
                    mua[e] = pa / pn
            Wn = {"gate": mean_free_noise(Wb["gate"], mx.array(mux), SIGMA["gu"], 9000 + 10 * i),
                  "up": mean_free_noise(Wb["up"], mx.array(mux), SIGMA["gu"], 9001 + 10 * i),
                  "down": mean_free_noise(Wb["down"], mx.array(mua), SIGMA["dn"], 9002 + 10 * i)}
            num = {"gu": 0.0, "dn": 0.0}; den = 0.0
            for e, (X, w2, g0, u0, a0) in erows.items():
                f = lambda p, n_=False: (Wn[p][e] if n_ else Wb[p][e]).astype(mx.float32)
                y = a0 @ f("down").T
                gq, uq = X @ f("gate", True).T, X @ f("up", True).T
                den += float((w2 * y * y).sum())
                num["gu"] += float((w2 * ((gq * mx.sigmoid(gq) * uq) @ f("down").T - y) ** 2).sum())
                num["dn"] += float((w2 * (a0 @ f("down", True).T - y) ** 2).sum())
            for g in ("gu", "dn") + (("null",) if i == 1 else ()):
                for p in (PROJ[g] if g != "null" else ()):
                    getattr(sw, p + "_proj").weight = Wn[p]
                o = []
                for q in range(len(BUCKETS)):
                    oo, _, _ = fwd(layer, HS[0][q], True, base_rt[q]); mx.eval(oo); o.append(oo)
                for p in ("gate", "up", "down"):
                    getattr(sw, p + "_proj").weight = Wb[p]
                tag = f"{i}:{g}" if g != "null" else "null"
                new.append(o); tags.append(tag); FL[tag] = {}
                if g != "null":
                    rec[g] = {"nmse": num[g] / den}
            rec["experts_scored"] = pn
            del Wn, Xte, erows
        HS = [[adv[q][k] for q in range(len(BUCKETS))] for k in range(nb)] + new
        mx.eval(HS)
        if rec:
            res["layers"][str(i)] = rec
        del layer, adv, new; mx.clear_cache()
        print(f"layer {i:2d} branches {len(HS):3d}  {time.time()-tl:6.1f}s  total {(time.time()-t0)/60:5.1f} min  "
              + " ".join(f"{k} nmse {v['nmse']:.2e}" for k, v in rec.items() if isinstance(v, dict))
              + (f" | would-flip share in this layer: oldest branch {FL[tags[1]].get(i, 0) / npos:.4f}, newest {FL[tags[nb - 1]].get(i, 0) / npos:.4f}" if nb > 2 and moe else ""), flush=True)
    assert len(tags) == len(HS)
    normw = mx.load(str(src / H["model.norm.weight"][0]))["model.norm.weight"].astype(mx.bfloat16)
    head = mx.load(str(src / H["lm_head.weight"][0]))["lm_head.weight"].astype(mx.bfloat16)

    def logp(h):
        lg = (mx.fast.rms_norm(h, normw, A.layernorm_epsilon) @ head.T).astype(mx.float32)
        return lg - mx.logsumexp(lg, axis=-1, keepdims=True)

    kls = [[] for _ in HS]; flips = [0] * len(HS)
    for q, (T, _, _) in enumerate(BUCKETS):
        ch = max(1, 1536 // T)
        for s_ in range(0, NB[q], ch):
            lb = logp(HS[0][q][s_:s_ + ch]); pb = mx.exp(lb); ab = mx.argmax(lb, axis=-1); m = VALID[q][s_:s_ + ch]; mx.eval(lb, pb, ab)
            mn = np.asarray(m)
            for k in range(1, len(HS)):
                lq = logp(HS[k][q][s_:s_ + ch])
                kl = mx.sum(pb * (lb - lq), axis=-1); fl = (mx.argmax(lq, axis=-1) != ab) & m
                mx.eval(kl, fl)
                kls[k].append(np.asarray(kl)[mn]); flips[k] += int(fl.sum().item())
            mx.clear_cache()
    out = {}
    for k in range(1, len(HS)):
        v = np.concatenate(kls[k])
        if v.size != npos or not np.isfinite(v).all():
            print(f"INVALID: branch {tags[k]}: {v.size} positions (expected {npos}) or non-finite KL"); sys.exit(2)
        out[tags[k]] = {"kl_frozen_mean": float(v.mean()), "kl_frozen_median": float(np.median(v)), "top1_flip_pct_frozen": 100.0 * flips[k] / npos,
                        "would_flip_share": {str(M): c / npos for M, c in sorted(FL[tags[k]].items())}}
    res["null"] = out.pop("null", None)
    if res["null"] is None and not a.max_layers:
        print("INVALID: the null control branch is missing"); sys.exit(2)
    if res["null"] is not None and (not res["null"]["kl_frozen_mean"] < 1e-5 or any(v > 0 for v in res["null"]["would_flip_share"].values())):
        print(f"INVALID: null control {res['null']}"); sys.exit(2)
    for tag, v in out.items():
        L, g = tag.split(":")
        r = res["layers"][L][g]; r.update(v); r["amp"] = r["nmse"] ** 0.5
        r["k"] = {M: f / r["amp"] for M, f in v["would_flip_share"].items()}
        r["flips_total"] = sum(v["would_flip_share"].values())
        r["c"] = v["kl_frozen_mean"] / max(r["nmse"], 1e-30)
    res["valid_for_planning"] = not a.max_layers; res["sigma"] = SIGMA
    res["minutes"] = (time.time() - t0) / 60
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"null control: {res['null']}", flush=True)
    for L, r in res["layers"].items():
        print(f"L{int(L):2d} " + "  ".join(f"{k}: nmse {v['nmse']:.2e} frozen KL {v['kl_frozen_mean']:.5f} c {v['c']:.3f} would-flip layers*share {v['flips_total']:.3f} "
                                          f"per unit amplitude {v['flips_total'] / v['amp']:.2f} next-layer {next(iter(v['would_flip_share'].values()), 0):.4f}"
                                          for k, v in r.items() if isinstance(v, dict)), flush=True)
    print("SENS2 DONE", flush=True)


if __name__ == "__main__":
    main()
