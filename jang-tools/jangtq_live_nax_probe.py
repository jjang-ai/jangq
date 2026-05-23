#!/usr/bin/env python3
"""Live JANGTQ MPP/NAX probe for a real model.

This is intentionally narrow: load one local JANGTQ model, run the same long
prompt with the current kernel path and with ``JANGTQ_MPP_NAX=auto``, and emit
timing plus generated text. It is a live gate, not a synthetic proof.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

from jang_tools.load_jangtq import load_jangtq_model


@contextmanager
def _env(name: str, value: str | None):
    old = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def _token_ids(tokenizer, text: str) -> list[int]:
    encoded = tokenizer.encode(text)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    return [int(x) for x in encoded]


def _build_prompt(tokenizer, target_tokens: int) -> tuple[str, int]:
    seed = (
        "You are validating a local JANGTQ runtime on Apple Silicon. "
        "Keep the final answer short and say READY then the answer. "
    )
    payload = (
        "Context marker alpha: CERULEAN. "
        "Context marker beta: AMBER. "
        "Context marker gamma: VIOLET. "
        "The arithmetic check is 17 plus 28. "
    )
    text = seed
    while len(_token_ids(tokenizer, text)) < target_tokens:
        text += payload
    ids = _token_ids(tokenizer, text)
    return text, len(ids)


def _chat_prompt(tokenizer, body: str) -> tuple[mx.array, int, str]:
    messages = [
        {
            "role": "user",
            "content": body
            + "\nFinal answer: repeat CERULEAN and give 17+28 only.",
        }
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        rendered = body + "\nAssistant:"
    ids = _token_ids(tokenizer, rendered)
    return mx.array(ids, dtype=mx.uint32), len(ids), rendered


def _decode(tokenizer, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(token_ids)
    except Exception:
        return "".join(tokenizer.decode([tid]) for tid in token_ids)


def _token_to_int(token) -> int:
    if hasattr(token, "item"):
        return int(token.item())
    return int(token)


def _run_once(model, tokenizer, prompt: mx.array, max_tokens: int) -> dict:
    sampler = make_sampler(temp=0.0, top_p=0.0)
    new_ids: list[int] = []
    t0 = time.perf_counter()
    first = None
    for token, _probs in generate_step(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        sampler=sampler,
        prefill_step_size=2048,
    ):
        if hasattr(token, "shape"):
            mx.eval(token)
        if first is None:
            first = time.perf_counter()
        new_ids.append(_token_to_int(token))
    end = time.perf_counter()
    ttft = (first - t0) if first is not None else None
    total = end - t0
    decode_window = max(total - (ttft or 0.0), 1e-9)
    return {
        "new_tokens": len(new_ids),
        "ttft_s": ttft,
        "total_s": total,
        "approx_prefill_tok_s": (
            float(prompt.size) / ttft if ttft and ttft > 0 else None
        ),
        "decode_tok_s_after_first": (
            max(len(new_ids) - 1, 0) / decode_window if len(new_ids) > 1 else None
        ),
        "output": _decode(tokenizer, new_ids),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    load_t0 = time.perf_counter()
    model, tokenizer = load_jangtq_model(str(args.model))
    mx.eval(model.parameters())
    load_s = time.perf_counter() - load_t0

    body, body_tokens = _build_prompt(tokenizer, args.prompt_tokens)
    prompt, prompt_tokens, rendered = _chat_prompt(tokenizer, body)
    mx.eval(prompt)

    results = {
        "model": str(args.model),
        "load_s": load_s,
        "target_body_tokens": args.prompt_tokens,
        "body_tokens": body_tokens,
        "prompt_tokens": prompt_tokens,
        "prompt_prefix": rendered[:500],
        "runs": [],
    }

    run_modes = [
        {
            "label": "legacy_prefill",
            "global": None,
            "prefill": "0",
            "strict": None,
        },
        {
            "label": "default_prefill",
            "global": None,
            "prefill": None,
            "strict": "1",
        },
        {
            "label": "global_auto",
            "global": "auto",
            "prefill": None,
            "strict": "1",
        },
    ]

    for mode in run_modes:
        with _env("JANGTQ_MPP_NAX", mode["global"]), _env(
            "JANGTQ_MPP_NAX_PREFILL", mode["prefill"]
        ), _env(
            "JANGTQ_MPP_NAX_STRICT", mode["strict"]
        ):
            label = mode["label"]
            # Warm compile on the same prompt so measured rows are not just
            # kernel compilation. This deliberately does not reuse KV cache.
            _run_once(model, tokenizer, prompt, max_tokens=2)
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            else:
                mx.metal.clear_cache()
            run = _run_once(model, tokenizer, prompt, args.max_tokens)
            run["mode"] = label
            results["runs"].append(run)

    text = json.dumps(results, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
