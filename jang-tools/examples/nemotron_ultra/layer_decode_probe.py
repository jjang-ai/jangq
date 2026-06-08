"""Profile one real Nemotron Ultra decode step by layer type.

The probe prefills a short prompt to create real hybrid cache state, then runs
one manual decode token through the backbone and records synchronized wall time
for each layer. This is intentionally diagnostic: per-layer synchronization
adds overhead, but it localizes where the current Python/MLX decode path spends
time.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx

from mlx_lm.models.nemotron_h import create_attention_mask, create_ssm_mask

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


def _summarize(rows: list[dict]) -> dict:
    by_type: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_type[row["block_type"]].append(float(row["ms"]))
    return {
        block_type: {
            "count": len(samples),
            "total_ms": sum(samples),
            "median_ms": sorted(samples)[len(samples) // 2],
            "min_ms": min(samples),
            "max_ms": max(samples),
        }
        for block_type, samples in sorted(by_type.items())
    }


def _decode_step(model, cache, token_id: int, *, collect_rows: bool) -> tuple[list[dict], float, float]:
    token = mx.array([[token_id]], dtype=mx.int32)
    hidden_states = model.backbone.embeddings(token)
    mx.eval(hidden_states)
    attn_mask = create_attention_mask(hidden_states, cache[model.backbone.fa_idx])
    ssm_mask = create_ssm_mask(hidden_states, cache[model.backbone.ssm_idx])

    rows = []
    cache_counter = 0
    for i, layer in enumerate(model.backbone.layers):
        if layer.block_type == "M" or layer.block_type == "*":
            c = cache[cache_counter]
            cache_counter += 1
        else:
            c = None
        mask = attn_mask if layer.block_type == "*" else ssm_mask
        start = time.perf_counter()
        hidden_states = layer(hidden_states, mask=mask, cache=c)
        mx.eval(hidden_states)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if collect_rows:
            rows.append(
                {
                    "layer_index": i,
                    "block_type": layer.block_type,
                    "ms": elapsed_ms,
                }
            )

    norm_start = time.perf_counter()
    hidden_states = model.backbone.norm_f(hidden_states)
    logits = model.lm_head(hidden_states)
    mx.eval(logits)
    norm_lm_head_ms = (time.perf_counter() - norm_start) * 1000.0
    total_ms = sum(r["ms"] for r in rows) + norm_lm_head_ms if collect_rows else 0.0
    return rows, norm_lm_head_ms, total_ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--prompt", default="What is the capital of Japan? Answer briefly.")
    ap.add_argument("--decode-token", type=int, default=42)
    ap.add_argument("--decode-warmup-steps", type=int, default=1)
    ap.add_argument("--wired-limit-gb", type=int, default=105)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    os.environ.setdefault("JANGTQ_WIRED_LIMIT_GB", str(args.wired_limit_gb))
    model, tokenizer = load_jangtq_model(args.bundle, skip_params_eval=True)
    cache = model.make_cache()
    prompt_ids = _render(tokenizer, args.prompt)

    prefill_start = time.perf_counter()
    prefill_logits = model(mx.array([prompt_ids], dtype=mx.int32), cache=cache)
    mx.eval(prefill_logits)
    prefill_s = time.perf_counter() - prefill_start

    for _ in range(args.decode_warmup_steps):
        _decode_step(model, cache, args.decode_token, collect_rows=False)

    rows, norm_lm_head_ms, manual_decode_total_ms = _decode_step(
        model,
        cache,
        args.decode_token,
        collect_rows=True,
    )

    result = {
        "bundle": str(args.bundle),
        "prompt_tokens": len(prompt_ids),
        "decode_token": args.decode_token,
        "decode_warmup_steps": args.decode_warmup_steps,
        "prefill_s": prefill_s,
        "layer_rows": rows,
        "summary_by_block_type": _summarize(rows),
        "norm_lm_head_ms": norm_lm_head_ms,
        "manual_decode_total_ms": manual_decode_total_ms,
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
