"""Bounded live speed/coherence probe for Nemotron Ultra JANGTQ_1L.

This loads the real bundle and runs a small fixed prompt set. Use it for A/B
checks around `JANGTQ_ENABLE_NEMOTRON_SWITCHMLP_FASTPATH`.
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


DEFAULT_BUNDLE = Path("/Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L")


def _render(tokenizer, prompt: str, enable_thinking: bool) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return tokenizer.encode(rendered)


def _run_row(model, tokenizer, row: dict, max_tokens: int, sampler) -> dict:
    prompt_ids = _render(tokenizer, row["prompt"], bool(row["enable_thinking"]))
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
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        if first is None:
            first = time.perf_counter()
        token_id = int(token)
        token_ids.append(token_id)
        if token_id in eos_ids:
            break
    end = time.perf_counter()
    generated = len(token_ids)
    decode_tps = None
    if first is not None and generated > 1:
        decode_tps = (generated - 1) / max(end - first, 1e-9)
    return {
        "id": row["id"],
        "enable_thinking": row["enable_thinking"],
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": generated,
        "wall_s": round(end - start, 3),
        "ttft_s": round(first - start, 3) if first is not None else None,
        "decode_tps_excluding_first": round(decode_tps, 3) if decode_tps else None,
        "text": tokenizer.decode(token_ids),
        "token_ids": token_ids,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--wired-limit-gb", type=int, default=105)
    ap.add_argument("--rows", choices=("short", "full"), default="short")
    ap.add_argument(
        "--sampler",
        choices=("default", "greedy"),
        default="default",
        help="default uses generation_config-style temp=1/top_p=0.95; greedy uses temp=0.",
    )
    args = ap.parse_args()

    os.environ.setdefault("JANGTQ_WIRED_LIMIT_GB", str(args.wired_limit_gb))
    model, tokenizer = load_jangtq_model(args.bundle, skip_params_eval=True)
    sampler = make_sampler(temp=0.0) if args.sampler == "greedy" else make_sampler(temp=1.0, top_p=0.95)
    rows = [
        {
            "id": "nt_math_default",
            "enable_thinking": False,
            "prompt": "What is 2+2? Answer briefly.",
        },
        {
            "id": "nt_capital_default",
            "enable_thinking": False,
            "prompt": "What is the capital of Japan? Answer in one short sentence.",
        },
    ]
    if args.rows == "full":
        rows.append(
            {
                "id": "think_math_default",
                "enable_thinking": True,
                "prompt": "If I have 3 apples and buy 4 more, how many apples do I have?",
            }
        )
    result = {
        "bundle": str(args.bundle),
        "generation_config_sampler": (
            "make_sampler(temp=0.0)"
            if args.sampler == "greedy"
            else "make_sampler(temp=1.0, top_p=0.95)"
        ),
        "sampler": args.sampler,
        "switchmlp_fastpath_env": os.environ.get("JANGTQ_ENABLE_NEMOTRON_SWITCHMLP_FASTPATH", ""),
        "activation_bf16_disable_env": os.environ.get("JANGTQ_DISABLE_NEMOTRON_ACTIVATION_BF16", ""),
        "weighted_moe_disable_env": os.environ.get("JANGTQ_DISABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH", ""),
        "wired_limit_gb": args.wired_limit_gb,
        "max_tokens": args.max_tokens,
        "rows": [_run_row(model, tokenizer, row, args.max_tokens, sampler) for row in rows],
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
