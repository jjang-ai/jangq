from transformers import AutoTokenizer
SRC="/Users/eric/models/Nanbeige/Nanbeige4.2-3B"
t=AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)
m=[{"role":"user","content":"hi"}]
enc=t.apply_chat_template(m, add_generation_prompt=True, tokenize=True, return_dict=True)
ids_tmpl=list(enc["input_ids"])
if ids_tmpl and isinstance(ids_tmpl[0], list): ids_tmpl=ids_tmpl[0]
s=t.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
print("rendered head:", repr(s[:60]))
print("template ids[:6]", ids_tmpl[:6], "n_bos:", ids_tmpl.count(166100))
a=list(t(s).input_ids); b=list(t(s, add_special_tokens=False).input_ids)
print("encode add_special=True [:6]", a[:6], "n_bos:", a.count(166100))
print("encode add_special=False[:6]", b[:6], "n_bos:", b.count(166100))
print("bos_id", t.bos_token_id, "eos_id", t.eos_token_id, "add_bos_token", t.add_bos_token)
for kw in ({}, {"enable_thinking": False}, {"enable_thinking": True}):
    r=t.apply_chat_template(m, add_generation_prompt=True, tokenize=False, **kw)
    print(kw, "tail ->", repr(r[-32:]))
