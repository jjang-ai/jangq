import numpy as np, torch, mlx.core as mx
from transformers import AutoModelForCausalLM, AutoConfig
from jang_tools.nanbeige import mlx_register  # noqa: F401
from mlx_lm import load
SRC="/Users/eric/models/Nanbeige/Nanbeige4.2-3B"
OUT="/private/tmp/claude-501/-Users-eric-jang/85ab95b4-d966-4eb6-b700-255f21436c4b/scratchpad/nanbeige"
ids_np=np.load(f"{OUT}/hf_ids.npy")

cfg=AutoConfig.from_pretrained(SRC, trust_remote_code=True); cfg.rope_scaling=None; cfg._attn_implementation="eager"
hm=AutoModelForCausalLM.from_pretrained(SRC, config=cfg, trust_remote_code=True, dtype=torch.float32, device_map="cpu").eval()
mm,_=load(SRC); mm.set_dtype(mx.float32)

def rel(a,b):
    a=np.asarray(a,np.float64); b=np.asarray(b,np.float64)
    return np.abs(a-b).max()/max(np.abs(b).max(),1e-9)

x_t=hm.model.embed_tokens(torch.tensor(ids_np))
x_m=mm.model.embed_tokens(mx.array(ids_np))
print("embed", rel(np.array(x_m), x_t.detach().numpy()))

hl=hm.model.layers[0]; ml=mm.model.layers[0]
n_t=hl.input_layernorm(x_t); n_m=ml.input_layernorm(x_m)
print("input_layernorm", rel(np.array(n_m), n_t.detach().numpy()))

B,L,_=n_t.shape
q_t=hl.self_attn.q_proj(n_t).view(B,L,48,128).transpose(1,2)
k_t=hl.self_attn.k_proj(n_t).view(B,L,8,128).transpose(1,2)
v_t=hl.self_attn.v_proj(n_t).view(B,L,8,128).transpose(1,2)
q_m=ml.self_attn.q_proj(n_m).reshape(B,L,48,128).transpose(0,2,1,3)
k_m=ml.self_attn.k_proj(n_m).reshape(B,L,8,128).transpose(0,2,1,3)
v_m=ml.self_attn.v_proj(n_m).reshape(B,L,8,128).transpose(0,2,1,3)
print("q_proj", rel(np.array(q_m), q_t.detach().numpy()))
print("k_proj", rel(np.array(k_m), k_t.detach().numpy()))
print("v_proj", rel(np.array(v_m), v_t.detach().numpy()))

pos=torch.arange(L)[None]
cos,sin=hl.self_attn.rotary_emb(v_t,pos)
import importlib
mod=importlib.import_module(hl.self_attn.__class__.__module__)
qr_t,kr_t=mod.apply_rotary_pos_emb(q_t,k_t,cos,sin)
qr_m=ml.self_attn.rope(q_m); kr_m=ml.self_attn.rope(k_m)
print("rope(q)", rel(np.array(qr_m), qr_t.detach().numpy()))
print("rope(k)", rel(np.array(kr_m), kr_t.detach().numpy()))
print("  inv_freq0", float(hl.self_attn.rotary_emb.inv_freq[0]), float(hl.self_attn.rotary_emb.inv_freq[-1]))
print("  mlx rope base", getattr(ml.self_attn.rope,'base',None), "dims", getattr(ml.self_attn.rope,'dims',None),
      "traditional", getattr(ml.self_attn.rope,'traditional',None), "scale", getattr(ml.self_attn.rope,'scale',None))

r_t=hl.self_attn(n_t, attention_mask=None, position_ids=pos)[0]
from mlx_lm.models.base import create_attention_mask
mask=create_attention_mask(n_m,None)
r_m=ml.self_attn(n_m, mask, None)
print("attn out", rel(np.array(r_m), r_t.detach().numpy()))

h_t=x_t+r_t; h_m=x_m+r_m
m_t=hl.mlp(hl.post_attention_layernorm(h_t)); m_m=ml.mlp(ml.post_attention_layernorm(h_m))
print("mlp out", rel(np.array(m_m), m_t.detach().numpy()))
