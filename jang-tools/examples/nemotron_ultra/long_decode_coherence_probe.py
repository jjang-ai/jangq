"""Long-decode coherence probe for Nemotron Ultra JANGTQ_1L.

This is a bounded runtime check, not a benchmark leaderboard. It loads the real
bundle, generates longer answers, and records simple coherence flags:

- EOS reached
- visible think-marker leaks
- short n-gram repetition
- expected answer substring for smoke prompts
- tokens/sec excluding first token

Use this after runtime/kernel changes to catch token salad, runaway repetition,
and parser/template regressions.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

from jang_tools.load_jangtq import load_jangtq_model


DEFAULT_BUNDLE = Path(
    "/Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L"
)


ROWS = [
    {
        "id": "factual_japan",
        "enable_thinking": False,
        "prompt": "What is the capital of Japan? Answer in one sentence.",
        "expected_substrings": ["Tokyo", "Japan"],
    },
    {
        "id": "arithmetic_brief",
        "enable_thinking": False,
        "prompt": "What is 17 + 25? Answer briefly.",
        "expected_substrings": ["42"],
    },
    {
        "id": "reasoning_apples",
        "enable_thinking": True,
        "prompt": "I have 3 apples, buy 4 more, then give away 2. How many apples remain?",
        "expected_substrings": ["5"],
    },
]


def _render(tokenizer, prompt: str, enable_thinking: bool) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return tokenizer.encode(rendered)


def _eos_ids(tokenizer) -> set[int]:
    ids = getattr(tokenizer, "eos_token_ids", None)
    if ids is None:
        eos = getattr(tokenizer, "eos_token_id", None)
        ids = eos if isinstance(eos, list) else ([eos] if eos is not None else [])
    return {int(x) for x in ids if x is not None}


def _ngram_stats(token_ids: list[int], n: int = 4) -> dict:
    if len(token_ids) < n:
        return {"n": n, "total": 0, "unique": 0, "max_count": 0, "repeat_fraction": 0.0}
    grams = [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]
    counts = Counter(grams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return {
        "n": n,
        "total": len(grams),
        "unique": len(counts),
        "max_count": max(counts.values()) if counts else 0,
        "repeat_fraction": repeated / max(1, len(grams)),
    }


def _run_row(model, tokenizer, row: dict, max_tokens: int, sampler) -> dict:
    prompt_ids = _render(tokenizer, row["prompt"], bool(row["enable_thinking"]))
    eos = _eos_ids(tokenizer)
    start = time.perf_counter()
    first = None
    token_ids: list[int] = []
    for token, _prob in generate_step(
        mx.array(prompt_ids),
        model,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        if first is None:
            first = time.perf_counter()
        token_id = int(token)
        token_ids.append(token_id)
        if token_id in eos:
            break
    end = time.perf_counter()
    text = tokenizer.decode(token_ids)
    expected = row.get("expected_substrings", [])
    decode_tps = None
    if first is not None and len(token_ids) > 1:
        decode_tps = (len(token_ids) - 1) / max(end - first, 1e-9)
    marker_leaks = [
        marker for marker in ("<think>", "</think>", "<tool_call>", "<tool_response>")
        if marker in text
    ]
    ngram = _ngram_stats(token_ids, 4)
    return {
        "id": row["id"],
        "enable_thinking": row["enable_thinking"],
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(token_ids),
        "eos_reached": bool(token_ids and token_ids[-1] in eos),
        "ttft_s": round(first - start, 3) if first is not None else None,
        "wall_s": round(end - start, 3),
        "decode_tps_excluding_first": round(decode_tps, 3) if decode_tps else None,
        "expected_substrings": expected,
        "expected_found": {s: (s in text) for s in expected},
        "visible_marker_leaks": marker_leaks,
        "ngram_repeat": ngram,
        "text": text,
        "token_ids": token_ids,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--wired-limit-gb", type=int, default=105)
    ap.add_argument("--sampler", choices=("greedy", "default"), default="greedy")
    ap.add_argument("--rows", choices=("short", "full"), default="full")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    os.environ.setdefault("JANGTQ_WIRED_LIMIT_GB", str(args.wired_limit_gb))
    model, tokenizer = load_jangtq_model(args.bundle, skip_params_eval=True)
    sampler = make_sampler(temp=0.0) if args.sampler == "greedy" else make_sampler(temp=1.0, top_p=0.95)
    rows = ROWS[:2] if args.rows == "short" else ROWS
    result = {
        "bundle": str(args.bundle),
        "max_tokens": args.max_tokens,
        "sampler": args.sampler,
        "weighted_moe_disable_env": os.environ.get("JANGTQ_DISABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH", ""),
        "activation_bf16_disable_env": os.environ.get("JANGTQ_DISABLE_NEMOTRON_ACTIVATION_BF16", ""),
        "switchmlp_fastpath_env": os.environ.get("JANGTQ_ENABLE_NEMOTRON_SWITCHMLP_FASTPATH", ""),
        "rows": [_run_row(model, tokenizer, row, args.max_tokens, sampler) for row in rows],
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
