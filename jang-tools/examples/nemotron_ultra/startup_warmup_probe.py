"""Measure Nemotron Ultra loader warmup vs first-request latency.

Use this to separate startup compile/warmup time from steady request TTFT.
The normal live speed probe uses skip_params_eval=True for bounded iteration;
this probe intentionally uses the default loader warmup path.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

from jang_tools.load_jangtq import load_jangtq_model


DEFAULT_BUNDLE = Path(
    "/Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L"
)


def _render(tokenizer, prompt: str) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer.encode(rendered)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--wired-limit-gb", type=int, default=105)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    os.environ.setdefault("JANGTQ_WIRED_LIMIT_GB", str(args.wired_limit_gb))
    load_start = time.perf_counter()
    model, tokenizer = load_jangtq_model(args.bundle, skip_params_eval=False)
    load_s = time.perf_counter() - load_start

    prompt_ids = _render(
        tokenizer,
        "What is the capital of Japan? Answer in one short sentence.",
    )
    eos_ids = getattr(tokenizer, "eos_token_ids", None)
    if eos_ids is None:
        eos = getattr(tokenizer, "eos_token_id", None)
        eos_ids = eos if isinstance(eos, list) else ([eos] if eos is not None else [])
    eos_ids = {int(x) for x in eos_ids if x is not None}

    start = time.perf_counter()
    first = None
    token_ids: list[int] = []
    for token, _prob in generate_step(
        mx.array(prompt_ids),
        model,
        max_tokens=args.max_tokens,
        sampler=make_sampler(temp=1.0, top_p=0.95),
    ):
        if first is None:
            first = time.perf_counter()
        token_id = int(token)
        token_ids.append(token_id)
        if token_id in eos_ids:
            break
    end = time.perf_counter()
    decode_tps = None
    if first is not None and len(token_ids) > 1:
        decode_tps = (len(token_ids) - 1) / max(end - first, 1e-9)

    result = {
        "bundle": str(args.bundle),
        "wired_limit_gb": args.wired_limit_gb,
        "load_s": load_s,
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(token_ids),
        "ttft_s": (first - start) if first is not None else None,
        "wall_s": end - start,
        "decode_tps_excluding_first": decode_tps,
        "text": tokenizer.decode(token_ids),
        "token_ids": token_ids,
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
