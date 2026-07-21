"""Laguna S 2.1 JANG bundle — Python runtime example + verification drivers.

Three modes (see NOTES.md for the full verification protocol):

  # raw greedy smoke (step 3/4 of the protocol)
  python python_example.py --src ~/.mlxstudio/models/JANGQ-AI/Laguna-S-2.1-JANG_2L \
      --prompt 'def fibonacci(n):' --max-new 32

  # chat mode with the bundle's stamped defaults (thinking ON per vendor)
  python python_example.py --src <bundle> --chat "Explain GQA in two sentences."

  # long-prompt cache parity — THE test that exercises SWA + RotatingKVCache
  # (prompt >> 512-token window; cached and no-cache greedy must match)
  python python_example.py --src <bundle> --parity

Notes:
- eos is [2, 24]; id 2 doubles as bos (template leads with 〈|EOS|〉).
- The chat template is GLM-style: generation prompt ends '<assistant><think>'
  when thinking, '<assistant></think>' when not. Vendor default is thinking
  ON (jang_config.chat.template_kwargs_defaults), template fallback is off.
- Greedy only here — this is a correctness driver, not a serving example.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

from jang_tools.laguna.runtime import load


def _eos_set(src: Path) -> set[int]:
    cfg = json.loads((src / "config.json").read_text())
    v = cfg.get("eos_token_id", 2)
    return set(v) if isinstance(v, list) else {v}


def _greedy_cached(model, ids, max_new, eos, echo_tok=None):
    out = list(ids)
    logits, caches = model(mx.array([ids], dtype=mx.uint32), caches=None)
    t0 = time.time()
    for _ in range(max_new):
        nxt = int(mx.argmax(logits[0, -1]).item())
        out.append(nxt)
        if echo_tok is not None:
            print(echo_tok.decode([nxt]), end="", flush=True)
        if nxt in eos:
            break
        logits, caches = model(mx.array([[nxt]], dtype=mx.uint32), caches=caches)
    dt = time.time() - t0
    n = len(out) - len(ids)
    print(f"\n[{n} tok in {dt:.1f}s = {n / dt:.1f} tok/s]", flush=True)
    return out


def _greedy_no_cache(model, ids, max_new, eos):
    out = list(ids)
    for _ in range(max_new):
        logits, _ = model(mx.array([out], dtype=mx.uint32), caches=None)
        nxt = int(mx.argmax(logits[0, -1]).item())
        out.append(nxt)
        if nxt in eos:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--chat", default=None, metavar="MESSAGE")
    ap.add_argument("--no-think", action="store_true",
                    help="chat mode: override the vendor thinking-on default")
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--max-new", type=int, default=200)
    args = ap.parse_args()
    src = args.src.expanduser()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(src), trust_remote_code=True)
    eos = _eos_set(src)

    model, cfg, fmt = load(str(src))
    print(f"[example] format={fmt} eos={sorted(eos)}", flush=True)

    if args.parity:
        # TEACHER-FORCED parity: feed the SAME token sequence through the
        # one-shot prefill path (T>1: banded/causal masks) and the
        # incremental cached path (T=1 steps: RotatingKVCache eviction),
        # compare per-position argmax. Free-running greedy comparison is
        # the wrong test on a quantized bundle: bf16 kernel-order noise
        # legitimately flips near-tied tokens (measured on 2L: p=0.41 vs
        # 0.31 flip at step 9), and one flip diverges the whole tail.
        # The discriminative signal for a mask/cache bug is agreement
        # COLLAPSING past the 512 window while staying high below it —
        # tie-flips are position-independent.
        chunk = "def f%d(x):\n    return x * %d + f%d(x - 1)\n\n"
        prompt = "".join(chunk % (i, i, i + 1) for i in range(120))
        ids = tok.encode(prompt)
        W = cfg.sliding_window
        assert len(ids) > W + 300, f"prompt only {len(ids)} tokens; need >> {W}"
        print(f"[parity] teacher-forced, {len(ids)} tokens, window {W}",
              flush=True)

        logits_full, _ = model(mx.array([ids], dtype=mx.uint32), caches=None)
        full_top = mx.argmax(logits_full[0], axis=-1)
        mx.eval(full_top)

        step_top = []
        logits, caches = model(mx.array([ids[:1]], dtype=mx.uint32), caches=None)
        step_top.append(int(mx.argmax(logits[0, -1]).item()))
        for t in ids[1:]:
            logits, caches = model(mx.array([[t]], dtype=mx.uint32), caches=caches)
            step_top.append(int(mx.argmax(logits[0, -1]).item()))

        full_np = [int(x) for x in full_top.tolist()]
        agree = [a == b for a, b in zip(full_np, step_top)]
        pre = agree[:W]
        post = agree[W:]
        r_pre = sum(pre) / len(pre)
        r_post = sum(post) / len(post)
        print(f"[parity] top-1 agreement: pre-window {r_pre:.3f} "
              f"({len(pre)} pos), post-window {r_post:.3f} ({len(post)} pos)")
        # Tie-flips cost a few % uniformly. A mask/cache bug specifically
        # tanks the post-window segment.
        if r_post < r_pre - 0.05 or r_post < 0.85:
            raise SystemExit(
                "[parity] FAIL — post-window agreement collapsed: SWA cache "
                "eviction or banded prefill mask is broken. Run "
                "tests/test_laguna_hybrid_attention_cache.py first.")
        print("[parity] OK — no post-window degradation")
        return

    if args.chat is not None:
        think = not args.no_think  # vendor default is ON
        ids = tok.apply_chat_template(
            [{"role": "user", "content": args.chat}],
            tokenize=True, add_generation_prompt=True, enable_thinking=think)
        # Some fast-tokenizer paths return [tokenizers.Encoding], not
        # list[int] — same normalization the converter gate needs.
        if ids and not isinstance(ids[0], int):
            first = ids[0]
            ids = list(getattr(first, "ids", first))
        print(f"[chat] enable_thinking={think} prompt={len(ids)} tokens\n",
              flush=True)
        _greedy_cached(model, list(ids), args.max_new, eos, echo_tok=tok)
        return

    prompt = args.prompt or "def fibonacci(n):"
    ids = tok.encode(prompt)
    print(prompt, end="", flush=True)
    _greedy_cached(model, ids, args.max_new, eos, echo_tok=tok)


if __name__ == "__main__":
    main()
