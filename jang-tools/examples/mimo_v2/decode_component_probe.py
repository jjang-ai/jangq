"""MiMo JANGTQ decode component timing probe.

Loads one local MiMo bundle, fills the cache with a prompt, then times steady
single-token decode as backbone-only versus lm_head projection. This is an
inspection tool; it does not change generation behavior.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx


def _format_prompt(tokenizer: Any, prompt: str) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt


def _time_eval(fn) -> tuple[Any, float]:
    t0 = time.perf_counter()
    out = fn()
    mx.eval(out)
    return out, time.perf_counter() - t0


def run_probe(
    model_path: str,
    *,
    prompt: str,
    steps: int,
    warmup_steps: int,
    quantize_lm_head_bits: int | None,
    quantize_lm_head_group_size: int,
    output: str | Path,
) -> dict[str, Any]:
    from mlx_lm.utils import load
    from jang_tools.mimo_v2 import mlx_register  # noqa: F401

    model, tokenizer = load(
        model_path,
        lazy=True,
        tokenizer_config={"trust_remote_code": True},
    )
    lm_head_quantized = False
    lm_head_quantize_seconds = 0.0
    if quantize_lm_head_bits is not None:
        import mlx.nn as nn

        t0 = time.perf_counter()
        model.lm_head = nn.QuantizedLinear.from_linear(
            model.lm_head,
            group_size=quantize_lm_head_group_size,
            bits=quantize_lm_head_bits,
            mode="affine",
        )
        mx.eval(model.lm_head.parameters())
        lm_head_quantize_seconds = time.perf_counter() - t0
        lm_head_quantized = True
    rendered = _format_prompt(tokenizer, prompt)
    ids = tokenizer.encode(rendered)
    if len(ids) < 2:
        raise ValueError("prompt must encode to at least two tokens")

    cache = model.make_cache() if hasattr(model, "make_cache") else None
    input_ids = mx.array(ids, dtype=mx.int32)[None, :]
    _, prefill_seconds = _time_eval(lambda: model(input_ids, cache=cache))

    token = input_ids[:, -1:]
    records: list[dict[str, float]] = []
    for index in range(steps + warmup_steps):
        h, backbone_seconds = _time_eval(lambda: model.model(token, cache=cache))
        logits, lm_head_seconds = _time_eval(lambda: model.lm_head(h))
        token, sample_seconds = _time_eval(lambda: mx.argmax(logits[:, -1, :], axis=-1)[:, None])
        if index >= warmup_steps:
            records.append(
                {
                    "backbone_seconds": backbone_seconds,
                    "lm_head_seconds": lm_head_seconds,
                    "sample_seconds": sample_seconds,
                    "total_seconds": backbone_seconds + lm_head_seconds + sample_seconds,
                }
            )

    def avg(key: str) -> float:
        return sum(row[key] for row in records) / max(len(records), 1)

    report = {
        "model_path": model_path,
        "prompt_tokens": len(ids),
        "prefill_seconds": prefill_seconds,
        "lm_head": {
            "quantized": lm_head_quantized,
            "quantize_bits": quantize_lm_head_bits,
            "quantize_group_size": quantize_lm_head_group_size if quantize_lm_head_bits is not None else None,
            "quantize_seconds": lm_head_quantize_seconds,
        },
        "warmup_steps": warmup_steps,
        "measured_steps": len(records),
        "average_seconds": {
            "backbone": avg("backbone_seconds"),
            "lm_head": avg("lm_head_seconds"),
            "sample": avg("sample_seconds"),
            "total": avg("total_seconds"),
        },
        "average_ms": {
            "backbone": avg("backbone_seconds") * 1000.0,
            "lm_head": avg("lm_head_seconds") * 1000.0,
            "sample": avg("sample_seconds") * 1000.0,
            "total": avg("total_seconds") * 1000.0,
        },
        "records": records,
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Time MiMo decode backbone vs lm_head.")
    parser.add_argument("model", help="Path to local MiMo bundle")
    parser.add_argument("--prompt", default="Count from 1 to 20, separated by commas.")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--quantize-lm-head-bits", type=int, choices=[2, 3, 4, 6, 8])
    parser.add_argument("--quantize-lm-head-group-size", type=int, default=64)
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()

    report = run_probe(
        args.model,
        prompt=args.prompt,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        quantize_lm_head_bits=args.quantize_lm_head_bits,
        quantize_lm_head_group_size=args.quantize_lm_head_group_size,
        output=args.json_out,
    )
    ms = report["average_ms"]
    print(
        "[mimo-decode-probe] "
        f"steps={report['measured_steps']} "
        f"backbone_ms={ms['backbone']:.3f} "
        f"lm_head_ms={ms['lm_head']:.3f} "
        f"sample_ms={ms['sample']:.3f} "
        f"total_ms={ms['total']:.3f} "
        f"report={args.json_out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
