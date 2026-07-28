"""Is turn-4 truncation a bundle defect or prompt ambiguity + budget?

A: original ambiguous prompt ("returning KV cache bytes"), budget 14000.
B: disambiguated prompt ("returns an int"), budget 14000.
Same 4-turn history, thinking rail, JANG_6M.
"""
import time, json, sys
import mlx.core as mx
from jang_tools.nanbeige import mlx_register  # noqa: F401
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

AMBIG = ("Write a Python function `kv_bytes(tokens, layers, loops, kv_heads, head_dim, dtype_bytes=2)` "
         "returning KV cache bytes for a looped transformer, with a docstring and one doctest. "
         "Then output a JSON object {\"name\": <my name>, \"toolkit\": <my toolkit>} on the last line.")
CLEAR = ("Write a Python function `kv_bytes(tokens, layers, loops, kv_heads, head_dim, dtype_bytes=2)` "
         "that returns an int: the total KV cache size in bytes for a looped transformer. "
         "Include a docstring and exactly one doctest. "
         "Then output a JSON object {\"name\": <my name>, \"toolkit\": <my toolkit>} on the last line.")
PRIOR = [
    "My name is Eric and I'm building a quantization toolkit called JANG. In one short paragraph, "
    "what is the capital of Japan and why did it move there?",
    "What is 84 * 3 / 2? Show the two steps. Also, what did I say my toolkit is called?",
    "Write a 4-bullet release checklist for shipping a quantized model bundle. "
    "Each bullet must be one sentence. Then, on a final line, restate my name.",
]
model, tok = load("/Users/eric/models/JANGQ-AI/Nanbeige4.2-3B-JANG_6M")
sampler = make_sampler(temp=0.6, top_p=0.95, top_k=20)
eos = {166101, 166102}

for tag, q4 in (("A_ambiguous", AMBIG), ("B_disambiguated", CLEAR)):
    cache = model.make_cache()
    head = tok.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": PRIOR[0]}],
        add_generation_prompt=True, tokenize=False, enable_thinking=True)
    print(f"\n{'='*70}\n{tag}")
    for i, user in enumerate(PRIOR + [q4]):
        delta = head if i == 0 else f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        ids = tok.encode(delta, add_special_tokens=False)
        budget = 4000 if i < 3 else 14000
        t0, text, n, finish = time.time(), "", 0, "length"
        for r in stream_generate(model, tok, prompt=mx.array(ids), max_tokens=budget,
                                 prompt_cache=cache, sampler=sampler):
            text += r.text; n += 1
            if r.token in eos:
                finish = "stop"; break
        vis = text.split("</think>")[-1].strip() if "</think>" in text else "[REASONING NEVER CLOSED]"
        print(f"  t{i+1}: gen={n} finish={finish} closed={'</think>' in text} "
              f"{n/(time.time()-t0):.1f}tok/s off={sorted({c.offset for c in cache})}")
        if i == 3:
            print("  ---- FULL VISIBLE ----")
            print(vis)
            print("  ---- END ----")
        model(mx.array([tok.encode("<|im_end|>\n", add_special_tokens=False)]), cache=cache)
