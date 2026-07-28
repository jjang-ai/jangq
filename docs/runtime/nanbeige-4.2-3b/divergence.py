import json, numpy as np, mlx.core as mx
from jang_tools.nanbeige import mlx_register  # noqa: F401
from mlx_lm import load
from pathlib import Path

SRC="/Users/eric/models/Nanbeige/Nanbeige4.2-3B"
PROMPTS=["The capital of France is Paris. The capital of Japan is",
         "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
         "84 * 3 = 252. 252 / 2 =",
         "The three primary colors are red, blue, and",
         "In 1969 humans first landed on the"]
model,tok=load(SRC)
ids=[tok.encode(p, add_special_tokens=False) for p in PROMPTS]
ref=[]
for i in ids:
    lg=model(mx.array([i]))[0,-1].astype(mx.float32)
    ref.append(np.array(mx.softmax(lg)))
del model; mx.clear_cache()
print(f"{'bundle':10s} {'top1 agree':>11s} {'mean KL':>10s} {'max KL':>9s}")
for p in ["MXFP8","JANG_6M","JANG_4M"]:
    m,_=load(f"/Users/eric/models/JANGQ-AI/Nanbeige4.2-3B-{p}")
    kls=[];agree=0
    for i,r in zip(ids,ref):
        q=np.array(mx.softmax(m(mx.array([i]))[0,-1].astype(mx.float32)))
        kl=float(np.sum(r*(np.log(r+1e-12)-np.log(q+1e-12))))
        kls.append(kl); agree += int(q.argmax()==r.argmax())
    print(f"{p:10s} {agree}/{len(ids):>9d} {np.mean(kls):10.4f} {np.max(kls):9.4f}")
    del m; mx.clear_cache()
