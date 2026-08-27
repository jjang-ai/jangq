"""Greedy decode for Qwen4-Exp — the RUNTIME-BEFORE-QUANT coherence probe.

The 360 GB bf16 model streams from SSD via mmap: only touched pages load
(top-10/512 experts + trunk + 16 n-gram rows per token ≈ 12 GB/token), so a
short greedy probe is feasible on 128 GB RAM. Expect seconds per token.

Usage:
  python -m jang_tools.qwen4_exp.generate --model ~/models/Qwen3.8-Flash-Next \
      --prompt "Explain why the sky is blue in one sentence." --max-tokens 40
"""

import argparse
import time

import mlx.core as mx
import numpy as np

from .load import load_model
from .modeling import Model


def build_inputs(tokenizer_dir: str, prompt: str, thinking: bool = True):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False, enable_thinking=thinking
    )
    return tok, tok.encode(text, add_special_tokens=False)


def greedy(model: Model, ids, max_tokens: int, eos_ids, tok=None, chunk: int = 32):
    cache = model.make_cache()
    ids = mx.array([ids])
    # chunked prefill; stream_cold pages each layer via CPU first so the GPU
    # never hits the Metal watchdog on cold SSD faults
    logits = None
    for start in range(0, ids.shape[1], chunk):
        logits = model(ids[:, start: start + chunk], cache=cache, stream_cold=True)
        mx.eval(logits)
    out = []
    t0 = time.time()
    tokens_meta = []
    next_id = int(mx.argmax(logits[0, -1]))
    for i in range(max_tokens):
        out.append(next_id)
        if tok is not None:
            print(tok.decode([next_id]), end="", flush=True)
        if next_id in eos_ids:
            break
        step_t = time.time()
        logits = model(mx.array([[next_id]]), cache=cache, eval_layers=True)
        next_id = int(mx.argmax(logits[0, -1]))
        tokens_meta.append(time.time() - step_t)
    dt = time.time() - t0
    if tok is not None:
        print()
    return out, dt, tokens_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--prompt", default="Explain why the sky is blue in one sentence.")
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--raw", action="store_true", help="no chat template")
    args = ap.parse_args()

    tok_dir = args.tokenizer or args.model
    if args.raw:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tok_dir)
        ids = tok.encode(args.prompt)
    else:
        tok, ids = build_inputs(tok_dir, args.prompt, thinking=not args.no_thinking)
    print(f"prompt tokens: {len(ids)}")

    t0 = time.time()
    model = load_model(args.model, lazy=True)
    print(f"model constructed (lazy) in {time.time() - t0:.1f}s")

    eos = {248046, 248044}
    out, dt, per_tok = greedy(model, ids, args.max_tokens, eos, tok=tok)
    print(f"\n{len(out)} tokens in {dt:.1f}s "
          f"({len(out)/max(dt,1e-9):.2f} tok/s; median step "
          f"{np.median(per_tok) if per_tok else 0:.2f}s)")
    print("token ids:", out)


if __name__ == "__main__":
    main()
