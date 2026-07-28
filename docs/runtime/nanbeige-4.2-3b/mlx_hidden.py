import numpy as np, mlx.core as mx
from jang_tools.nanbeige import mlx_register  # noqa: F401
from mlx_lm import load
SRC="/Users/eric/models/Nanbeige/Nanbeige4.2-3B"
OUT="/private/tmp/claude-501/-Users-eric-jang/85ab95b4-d966-4eb6-b700-255f21436c4b/scratchpad/nanbeige"
model,_=load(SRC); model.set_dtype(mx.float32)
ids=mx.array(np.load(f"{OUT}/hf_ids.npy"))

hf_seq=np.load(f"{OUT}/hf_seq.npz"); hf_norms=np.load(f"{OUT}/hf_norms.npz")
hf_emb=np.load(f"{OUT}/hf_emb.npy")

inner=model.model
h=inner.embed_tokens(ids)
def rel(a,b):
    b=np.asarray(b,dtype=np.float64); a=np.asarray(a,dtype=np.float64)
    return np.abs(a-b).max()/max(np.abs(b).max(),1e-9)
print(f"embed        rel={rel(np.array(h),hf_emb):.3e}")
from mlx_lm.models.base import create_attention_mask
mask=create_attention_mask(h,None)
n=0
for loop in range(2):
    for i,layer in enumerate(inner.layers):
        r=rel(np.array(h), hf_seq[f"c{n}_l{i}"])
        if r>1e-4 or i in (0,21):
            print(f"loop{loop} layer{i:2d} input rel={r:.3e}")
        h=layer(h,mask,cache=None); n+=1
    h=inner.norm(h)
    print(f"loop{loop} post-norm    rel={rel(np.array(h),hf_norms[f'n{loop}']):.3e}")
lg=model.lm_head(h)
hf_lg=np.load(f"{OUT}/hf_logits_all.npy")
print("logits rel", rel(np.array(lg)[0], hf_lg))
