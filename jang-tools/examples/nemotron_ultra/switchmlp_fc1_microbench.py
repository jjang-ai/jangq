"""Microbench Nemotron Ultra SwitchMLP fc1 gather modes.

This isolates the decode bottleneck fixed in `load_jangtq.py`: old experimental
SwitchMLP fast path repeated one rotated input row K times before fc1, while the
new path uses the gather kernel's broadcast mode directly.

It does not load the 98G model.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

from jang_tools.turboquant.gather_tq_kernel import (
    make_gather_tq_decode_broadcast,
    make_gather_tq_decode_per_row,
)


def _time(fn, repeats: int) -> list[float]:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        y = fn()
        mx.eval(y)
        times.append(time.perf_counter() - start)
    return times


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-features", type=int, default=2048)
    ap.add_argument("--out-features", type=int, default=5120)
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--bits", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=22)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    vals_per_u32 = 32 // args.bits
    packed_cols = (args.in_features + vals_per_u32 - 1) // vals_per_u32
    packed = mx.zeros((args.experts, args.out_features, packed_cols), dtype=mx.uint32)
    norms = mx.ones((args.experts, args.out_features), dtype=mx.float16)
    codebook = mx.array([-1.0, 1.0], dtype=mx.float32)
    x_rot = mx.ones((1, args.in_features), dtype=mx.float32)
    idx = mx.arange(args.top_k, dtype=mx.uint32)
    mx.eval(packed, norms, codebook, x_rot, idx)

    per_row = make_gather_tq_decode_per_row(
        args.in_features, args.out_features, args.bits, args.top_k
    )
    broadcast = make_gather_tq_decode_broadcast(
        args.in_features, args.out_features, args.bits, args.top_k
    )

    def old_repeat_per_row():
        return per_row(mx.repeat(x_rot, args.top_k, axis=0), packed, norms, codebook, idx)

    def new_broadcast():
        return broadcast(x_rot, packed, norms, codebook, idx)

    _time(old_repeat_per_row, args.warmup)
    _time(new_broadcast, args.warmup)
    old_s = _time(old_repeat_per_row, args.repeats)
    new_s = _time(new_broadcast, args.repeats)

    result = {
        "shape": {
            "experts": args.experts,
            "in_features": args.in_features,
            "out_features": args.out_features,
            "bits": args.bits,
            "top_k": args.top_k,
            "packed_cols": packed_cols,
        },
        "old_repeat_per_row_s": old_s,
        "new_broadcast_s": new_s,
        "old_median_ms": sorted(old_s)[len(old_s) // 2] * 1000.0,
        "new_median_ms": sorted(new_s)[len(new_s) // 2] * 1000.0,
        "speedup": (sorted(old_s)[len(old_s) // 2] / sorted(new_s)[len(new_s) // 2])
        if sorted(new_s)[len(new_s) // 2] > 0
        else None,
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
