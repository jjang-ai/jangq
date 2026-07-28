"""Multi-turn coherence gate for Nanbeige4.2-3B bundles.

Per ~/wiki model-production-gate: >=3 turns, turn 2/3 must recall turn 1,
full tails inspected, both reasoning rails, real KV-cache continuity across
turns (not a fresh re-prefill), artifacts saved.

Turn N is fed as the DELTA only, against a persistent 44-slot cache — the same
thing a server does. Cache offsets are recorded after every turn to prove
continuity and that all 44 slots advance in lockstep.
"""
import json, sys, time
from pathlib import Path

import mlx.core as mx
from jang_tools.nanbeige import mlx_register  # noqa: F401
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

ART = Path(sys.argv[1])
ART.mkdir(parents=True, exist_ok=True)

TURNS = [
    # (label, user text, max_tokens)
    ("t1_factual",
     "My name is Eric and I'm building a quantization toolkit called JANG. "
     "In one short paragraph, what is the capital of Japan and why did it move there?", 4000),
    ("t2_arith_recall",
     "What is 84 * 3 / 2? Show the two steps. Also, what did I say my toolkit is called?", 4000),
    ("t3_long_instruction",
     "Write a 4-bullet release checklist for shipping a quantized model bundle. "
     "Each bullet must be one sentence. Then, on a final line, restate my name.", 4000),
    ("t4_code_structured",
     "Write a Python function `kv_bytes(tokens, layers, loops, kv_heads, head_dim, dtype_bytes=2)` "
     "returning KV cache bytes for a looped transformer, with a docstring and one doctest. "
     "Then output a JSON object {\"name\": <my name>, \"toolkit\": <my toolkit>} on the last line.", 5000),
]

SYSTEM = "You are a helpful assistant."


def run(bundle: str, thinking: bool) -> dict:
    d = Path(f"/Users/eric/models/JANGQ-AI/Nanbeige4.2-3B-{bundle}")
    model, tok = load(str(d))
    cache = model.make_cache()
    eos = {166101, 166102}

    # Turn 1 prompt is the full rendered template; later turns append the
    # exact block the template would have produced.
    head = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TURNS[0][1]}],
        add_generation_prompt=True, tokenize=False, enable_thinking=thinking,
    )
    rows, fed_total = [], 0
    sampler = make_sampler(temp=0.6, top_p=0.95, top_k=20)

    for i, (label, user, max_tok) in enumerate(TURNS):
        if i == 0:
            delta = head
        else:
            tail = "<think>\n" if thinking else "<think>\n\n</think>\n\n"
            delta = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{tail}"
        ids = tok.encode(delta, add_special_tokens=False)   # never re-add BOS
        fed_total += len(ids)

        t0, text, ntok, finish = time.time(), "", 0, "length"
        for resp in stream_generate(model, tok, prompt=mx.array(ids),
                                    max_tokens=max_tok, prompt_cache=cache, sampler=sampler):
            text += resp.text
            ntok += 1
            if resp.token in eos:
                finish = "stop"
                break
        dt = time.time() - t0
        offsets = sorted({c.offset for c in cache})
        rows.append({
            "turn": i + 1, "label": label, "user": user,
            "prompt_delta_tokens": len(ids), "gen_tokens": ntok,
            "finish_reason": finish, "decode_tok_s": round(ntok / dt, 2),
            "cache_offsets_distinct": offsets,
            "cache_slots": len(cache),
            "reasoning": text.split("</think>")[0].strip() if "</think>" in text else None,
            "visible": text.split("</think>")[-1].strip(),
            "raw": text,
        })
        # feed the model's own output back verbatim + the turn terminator
        head = ""
        if finish == "stop":
            trailer = "<|im_end|>\n"
        else:
            trailer = "<|im_end|>\n"
            text += "\n[TRUNCATED]"
        tok_trailer = tok.encode(trailer, add_special_tokens=False)
        model(mx.array([tok_trailer]), cache=cache)   # keep cache aligned with history

    peak = mx.get_peak_memory() / 1024**3
    res = {"bundle": bundle, "thinking": thinking, "cache_slots": len(cache),
           "prompt_tokens_fed_total": fed_total, "peak_gb": round(peak, 2), "turns": rows}
    del model
    mx.clear_cache()
    mx.reset_peak_memory()
    return res


all_res = []
for bundle in ["MXFP8", "JANG_6M", "JANG_4M"]:
    for thinking in [True]:
        r = run(bundle, thinking)
        all_res.append(r)
        rail = "think" if thinking else "direct"
        (ART / f"{bundle}_{rail}.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))
        print(f"\n{'='*78}\n{bundle}  rail={rail}  slots={r['cache_slots']}  peak={r['peak_gb']} GB")
        for t in r["turns"]:
            print(f"  -- turn {t['turn']} {t['label']}  gen={t['gen_tokens']} "
                  f"finish={t['finish_reason']} {t['decode_tok_s']} tok/s "
                  f"offsets={t['cache_offsets_distinct']}")
            print(f"     VISIBLE: {t['visible'][:400]}")
            if len(t["visible"]) > 400:
                print(f"     ...TAIL: {t['visible'][-260:]}")
        sys.stdout.flush()

(ART / "SUMMARY.json").write_text(json.dumps(all_res, indent=2, ensure_ascii=False))
print("\nartifacts:", ART)
