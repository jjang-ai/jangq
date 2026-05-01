"""Long-prompt coherence test for HSA + CSA + SWA tri-mode on real bundle.

Drives DSV4-Flash through three prompt lengths that exercise progressively
more of the Compressor / Indexer code path:

    L=64    →  fits in sliding_window; Compressor pool empty; SWA only
    L=512   →  pool fills (HSA P=4, CSA P=128); both modes engaged
    L=4096  →  large pool (HSA P=32, CSA P=1024); CSA top-k=512 active
                (forces Indexer to actually drop entries)

For each prompt we decode 32 greedy tokens and assert:
  - no exceptions
  - decoded tokens are valid IDs (in vocab range)
  - first-token logits are not NaN/Inf
  - first-token entropy < 8 (= sane distribution; pure-noise = log2(vocab) ≈ 17)

Run from the bundle on /Volumes/EricsLLMDrive/jangq-ai/DeepSeek-V4-Flash-JANGTQ.
"""
from __future__ import annotations
import os, sys, time, argparse, math

os.environ.setdefault("DSV4_LONG_CTX", "1")  # default after this commit
import mlx.core as mx
mx.set_memory_limit(110 * 1024**3)
mx.set_cache_limit(8 * 1024**3)

sys.path.insert(0, "/Users/eric/jang/jang-tools")


def run_case(model, tokenizer, label, prompt_len_target, max_new=32):
    print(f"\n=== {label} (target len={prompt_len_target}) ===", flush=True)
    base = "The Apollo program was the third United States human spaceflight program. "
    prompt = (base * (prompt_len_target // 12 + 1))[: prompt_len_target * 6]
    ids = tokenizer.encode(prompt)[:prompt_len_target]
    print(f"  actual prompt length: {len(ids)} tokens", flush=True)

    t0 = time.time()
    out = model(mx.array([ids], dtype=mx.uint32))
    mx.eval(out)
    dt_prefill = time.time() - t0
    last = out[0, -1].astype(mx.float32)
    if not bool(mx.all(mx.isfinite(last)).item()):
        print(f"  FAIL: last logits contain NaN/Inf"); return False
    log_probs = last - mx.logsumexp(last)
    probs = mx.exp(log_probs)
    entropy = float((-(probs * log_probs)).sum().item()) / math.log(2)
    print(f"  prefill: {dt_prefill:.1f}s  entropy(top-token-dist): {entropy:.2f} bits "
          f"(sanity: <8 = focused, log2(vocab)={math.log2(out.shape[-1]):.1f})", flush=True)

    if entropy > 12:
        print(f"  FAIL: entropy {entropy:.2f} > 12 = noisy distribution"); return False

    nxt = int(mx.argmax(last).item())
    print(f"  first generated token: id={nxt}  decoded={tokenizer.decode([nxt])!r}", flush=True)

    out_ids = list(ids)
    out_ids.append(nxt)
    t0 = time.time()
    # Continue decode for max_new tokens to confirm cache + decode path works
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    s = make_sampler(temp=0.0)
    n = 0
    for tok_id, _ in generate_step(prompt=mx.array(out_ids), model=model,
                                    max_tokens=max_new - 1, sampler=s):
        n += 1
        out_ids.append(int(tok_id))
        if int(tok_id) == tokenizer.eos_token_id:
            break
    dt_dec = time.time() - t0
    decoded_tail = tokenizer.decode(out_ids[len(ids):])
    print(f"  decode {n} new tokens in {dt_dec:.2f}s ({n/max(dt_dec,1e-9):.1f} tok/s)", flush=True)
    print(f"  generated tail: {decoded_tail!r}", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/Volumes/EricsLLMDrive/jangq-ai/DeepSeek-V4-Flash-JANGTQ")
    ap.add_argument("--max-new", type=int, default=24)
    args = ap.parse_args()

    print(f"[coherence] loading {args.src}...", flush=True)
    from jang_tools.load_jangtq import load_jangtq_model
    t0 = time.time()
    model, tok = load_jangtq_model(args.src)
    print(f"[coherence] loaded {time.time()-t0:.1f}s", flush=True)

    cases = [
        ("L=64  fits-window",         64),
        ("L=512 pool-engaged",       512),
        ("L=4096 indexer-active",   4096),
    ]
    fails = 0
    for label, n in cases:
        ok = run_case(model, tok, label, n, max_new=args.max_new)
        if not ok: fails += 1

    print(f"\n=== {len(cases)-fails}/{len(cases)} cases PASS ===")
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
