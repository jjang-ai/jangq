"""Held-out A/B: RTN (LS scale) vs GPTQ variants with the TQ codebook, on real GLM-5.3 experts.
Hessians built from reservoir rows [0, 3072); evaluated on rows [3072, 4096) that routed to the expert
(router-weighted output NMSE). Never trust in-sample numbers for GPTQ on MoE."""
import os
import json, sys, time, numpy as np, mlx.core as mx
from jang_tools.jangh.encode import encode, dequant
from jang_tools.jangh.gptq import gptq_encode
SRC=os.environ["JANGH_SOURCE"]
with open(f"{SRC}/model.safetensors.index.json") as fh:
    wm=json.load(fh)["weight_map"]
d=mx.load(os.environ["JANGH_DIAG"])
LIMIT=10.0; SPLIT=3072
cache={}
def get(name):
    f=wm[name]
    if f not in cache: cache.clear(); cache[f]=mx.load(f"{SRC}/{f}")
    return cache[f][name]
res={}
for L in (5, 20, 40, 44):
    mod=f"model.language_model.layers.{L}.mlp.experts"
    X=d[mod+".rows"].astype(mx.float32); tidx=np.asarray(d[mod+".rows_topk_idx"]); tw=np.asarray(d[mod+".rows_topk_w"])
    imx=(d[mod+".expert_sum_x2"].astype(mx.float32)/mx.maximum(d[mod+".expert_count"].astype(mx.float32),1.0)[:,None])
    gate_w=get(f"model.language_model.layers.{L}.mlp.gate.weight").astype(mx.float32)
    bias=get(f"model.language_model.layers.{L}.mlp.gate.e_score_correction_bias").astype(mx.float32)
    probs=mx.sigmoid(X@gate_w.T)                               # (R, E) soft router probability
    counts=np.bincount(tidx[:SPLIT].ravel(),minlength=288)
    experts=[int(e) for e in np.argsort(-counts)[:6]]+[int(e) for e in np.random.default_rng(L).choice(288,6,replace=False)]
    for bits in (2,3):
        for e in experts:
            base=f"model.language_model.layers.{L}.mlp.experts.{e}"
            Wg=get(base+".gate_proj.weight").astype(mx.float32); Wd=get(base+".down_proj.weight").astype(mx.float32); Wu=get(base+".up_proj.weight").astype(mx.float32)
            r_tr,s_tr=np.nonzero(tidx[:SPLIT]==e); r_te,s_te=np.nonzero(tidx[SPLIT:]==e); r_te=r_te+SPLIT
            if len(r_te)<4 or len(r_tr)<8: continue
            Xtr=X[mx.array(r_tr.astype(np.uint32))]; wtr=mx.array(tw[r_tr,s_tr].astype(np.float32))
            Xte=X[mx.array(r_te.astype(np.uint32))]; wte=mx.array(tw[r_te-0,s_te].astype(np.float32)) if False else mx.array(tw[r_te,s_te].astype(np.float32))
            Xall=X[:SPLIT]; pe=probs[:SPLIT,e]
            n_tr=float(wtr.sum().item())
            H_routed=(Xtr*wtr[:,None]).T@Xtr/n_tr
            H_soft=(Xall*pe[:,None]).T@Xall/float(pe.sum().item())
            D=mx.diag(imx[e]); H_prior=H_routed+ (mx.trace(H_routed)/mx.trace(D))*D*0.5
            def ev_gate(Q,s):
                got=Xte@dequant(Q,s,bits).T; ref=Xte@Wg.T
                return float((wte[:,None]*(got-ref)**2).sum()/(wte[:,None]*ref**2).sum())
            q,s=encode(Wg,bits); rec={"rtn":ev_gate(q,s)}
            for name,H in (("gptq_routed",H_routed),("gptq_soft",H_soft),("gptq_prior",H_prior)):
                Q,S=gptq_encode(Wg[None],H[None],bits); rec[name]=ev_gate(Q[0],S[0])
            # down: inputs = clamped swiglu of bf16 gate/up on routed rows
            def act(Xr):
                g=mx.minimum(Xr@Wg.T,LIMIT); u=mx.clip(Xr@Wu.T,-LIMIT,LIMIT); return g*mx.sigmoid(g)*u
            Atr=act(Xtr); Ate=act(Xte)
            Hd=(Atr*wtr[:,None]).T@Atr/n_tr
            def ev_down(Q,s):
                got=Ate@dequant(Q,s,bits).T; ref=Ate@Wd.T
                return float((wte[:,None]*(got-ref)**2).sum()/(wte[:,None]*ref**2).sum())
            q,s=encode(Wd,bits); rec["down_rtn"]=ev_down(q,s)
            Q,S=gptq_encode(Wd[None],Hd[None],bits); rec["down_gptq_routed"]=ev_down(Q[0],S[0])
            res.setdefault((L,bits),[]).append(rec)
            print(f"L{L} b{bits} e{e:3d} ntr={len(r_tr):3d} nte={len(r_te):3d} "+" ".join(f"{k}={v:.4f}" for k,v in rec.items()),flush=True)
print("\n== held-out summary (mean ratio vs RTN; <1 = GPTQ better)")
for (L,bits),recs in sorted(res.items()):
    for k in ("gptq_routed","gptq_soft","gptq_prior"):
        r=[x[k]/x["rtn"] for x in recs]; print(f"  L{L} b{bits} gate {k:12s} {np.mean(r):.3f} (worse in {sum(v>1 for v in r)}/{len(r)})")
    r=[x["down_gptq_routed"]/x["down_rtn"] for x in recs]; print(f"  L{L} b{bits} down gptq_routed  {np.mean(r):.3f} (worse in {sum(v>1 for v in r)}/{len(r)})")
