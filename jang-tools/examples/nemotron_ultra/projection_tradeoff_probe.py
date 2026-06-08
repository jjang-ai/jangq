"""Benchmark Nemotron Ultra 8-bit affine projections against BF16 dequantized copies.

This is a runtime tradeoff probe only. It does not modify the model bundle.
Use it before deciding to spend RAM on dequantized Mamba or shared-expert
projections.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from jang_tools.load_jangtq import load_jangtq_model


DEFAULT_BUNDLE = Path(
    "/Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L"
)


def _dequantized_linear(q):
    out_dims, packed_in = q.weight.shape
    in_dims = (packed_in * 32) // q.bits
    lin = nn.Linear(in_dims, out_dims, bias="bias" in q)
    w = mx.dequantize(
        q.weight,
        q.scales,
        q.get("biases"),
        group_size=q.group_size,
        bits=q.bits,
        mode=q.mode,
    )
    lin.weight = w.astype(mx.bfloat16)
    if "bias" in q:
        lin.bias = q.bias.astype(mx.bfloat16)
    return lin


def _bench(fn, repeats: int, warmup: int) -> dict:
    for _ in range(warmup):
        mx.eval(fn())
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        y = fn()
        mx.eval(y)
        samples.append(time.perf_counter() - start)
    ordered = sorted(samples)
    return {
        "samples_s": samples,
        "median_ms": ordered[len(ordered) // 2] * 1000.0,
        "min_ms": ordered[0] * 1000.0,
        "max_ms": ordered[-1] * 1000.0,
    }


def _probe(name: str, q, x, repeats: int, warmup: int) -> dict:
    lin = _dequantized_linear(q)
    mx.eval(x, q.weight, q.scales, lin.weight)
    q_stats = _bench(lambda: q(x), repeats, warmup)
    bf16_stats = _bench(lambda: lin(x), repeats, warmup)
    return {
        "name": name,
        "quantized": {
            "class": type(q).__name__,
            "weight_shape": list(q.weight.shape),
            "bits": int(q.bits),
            "group_size": int(q.group_size),
            **q_stats,
        },
        "bf16_dequantized": {
            "weight_shape": list(lin.weight.shape),
            **bf16_stats,
        },
        "quantized_speedup_vs_bf16": (
            bf16_stats["median_ms"] / q_stats["median_ms"]
            if q_stats["median_ms"] > 0
            else None
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--wired-limit-gb", type=int, default=105)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    os.environ.setdefault("JANGTQ_WIRED_LIMIT_GB", str(args.wired_limit_gb))
    model, _ = load_jangtq_model(args.bundle, skip_params_eval=True)
    mamba = next(l.mixer for l in model.backbone.layers if l.block_type == "M")
    moe = next(l.mixer for l in model.backbone.layers if l.block_type == "E")

    hidden = int(model.args.hidden_size)
    probes = [
        (
            "mamba_in_proj",
            mamba.in_proj,
            mx.ones((1, 1, hidden), dtype=mx.bfloat16),
        ),
        (
            "mamba_out_proj",
            mamba.out_proj,
            mx.ones((1, 1, int(mamba.intermediate_size)), dtype=mx.bfloat16),
        ),
        (
            "shared_up",
            moe.shared_experts.up_proj,
            mx.ones((1, 1, hidden), dtype=mx.bfloat16),
        ),
        (
            "shared_down",
            moe.shared_experts.down_proj,
            mx.ones(
                (1, 1, int(moe.config.moe_shared_expert_intermediate_size)),
                dtype=mx.bfloat16,
            ),
        ),
    ]

    result = {
        "bundle": str(args.bundle),
        "repeats": args.repeats,
        "warmup": args.warmup,
        "results": [
            _probe(name, q, x, args.repeats, args.warmup)
            for name, q, x in probes
        ],
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
