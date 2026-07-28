import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
SRC="/Users/eric/models/Nanbeige/Nanbeige4.2-3B"
OUT="/private/tmp/claude-501/-Users-eric-jang/85ab95b4-d966-4eb6-b700-255f21436c4b/scratchpad/nanbeige"
tok=AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)
cfg=AutoConfig.from_pretrained(SRC, trust_remote_code=True)
cfg.rope_scaling=None            # transformers>=5 injects {"rope_type":"default"}; 4.42-era code wants None
cfg._attn_implementation="eager"
m=AutoModelForCausalLM.from_pretrained(SRC, config=cfg, trust_remote_code=True,
                                       dtype=torch.float32, device_map="cpu")
m.eval()
print("num_loops",m.config.num_loops,"layers",m.config.num_hidden_layers,
      "skip_loop_final_norm",m.config.skip_loop_final_norm)

text="The capital of France is Paris. The capital of Japan is"
ids=tok(text, return_tensors="pt").input_ids
print("ids", ids.shape, ids.tolist())
with torch.no_grad():
    out=m(ids, use_cache=False)
lg=out.logits[0,-1].float().numpy()
np.save(f"{OUT}/hf_logits.npy", lg); np.save(f"{OUT}/hf_ids.npy", ids.numpy())
np.save(f"{OUT}/hf_logits_all.npy", out.logits[0].float().numpy())
top=np.argsort(-lg)[:5]
print("top5", top.tolist(), [tok.decode([int(i)]) for i in top])
