import numpy as np, mlx.core as mx
from jang_tools.nanbeige import mlx_register  # noqa: F401
from mlx_lm import load
from mlx_lm.models.cache import KVCache
SRC="/Users/eric/models/Nanbeige/Nanbeige4.2-3B"
OUT="/private/tmp/claude-501/-Users-eric-jang/85ab95b4-d966-4eb6-b700-255f21436c4b/scratchpad/nanbeige"
mm,_=load(SRC); mm.set_dtype(mx.float32)
ids=mx.array(np.load(f"{OUT}/hf_ids.npy")); hf=np.load(f"{OUT}/hf_logits.npy")
def rel(a,b):
    a=np.asarray(a,np.float64); b=np.asarray(b,np.float64)
    return np.abs(a-b).max()/max(np.abs(b).max(),1e-9)

full=np.array(mm(ids)[0,-1].astype(mx.float32))
print(f"A  no-cache full prefill      vs HF: rel={rel(full,hf):.3e} argmax {int(full.argmax())}/{int(hf.argmax())}")

c=mm.make_cache(); print("   make_cache slots:", len(c))
mm(ids[:,:-1], cache=c)
inc=np.array(mm(ids[:,-1:], cache=c)[0,-1].astype(mx.float32))
print(f"B  44-slot incremental        vs HF: rel={rel(inc,hf):.3e} argmax {int(inc.argmax())}/{int(hf.argmax())}")

# token-by-token, the real decode path
c2=mm.make_cache()
for i in range(ids.shape[1]-1):
    mm(ids[:, i:i+1], cache=c2)
tok1=np.array(mm(ids[:,-1:], cache=c2)[0,-1].astype(mx.float32))
print(f"C  token-by-token decode      vs HF: rel={rel(tok1,hf):.3e} argmax {int(tok1.argmax())}/{int(hf.argmax())}")
print("   cache offsets:", sorted({x.offset for x in c2}))

# NEGATIVE CONTROL: 22 slots shared by both loops (the naive port)
c22=[KVCache() for _ in range(22)]
shared=c22+c22
mm.model(ids[:,:-1], cache=shared)
bad=np.array(mm.lm_head(mm.model(ids[:,-1:], cache=shared))[0,-1].astype(mx.float32))
print(f"D  NEG CTRL 22 shared slots   vs HF: rel={rel(bad,hf):.3e} argmax {int(bad.argmax())}/{int(hf.argmax())}  -> must differ")
print("   negative control detected:", rel(bad,hf) > 1e-2 or int(bad.argmax())!=int(hf.argmax()))
