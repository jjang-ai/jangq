"""Thunderbolt 5 / RDMA bandwidth + latency probe.

Run BEFORE any correctness experiment (per feedback_runtime_before_quant
spirit — verify the transport at every layer).

Usage (launch on both nodes via mlx.launch):
    mlx.launch --hostfile hostfile.json -m jang_tools.distributed.tb5_probe \
        --rounds 5

Reports per-pair:
    - Median round-trip latency (small all_sum)
    - Sustained all_sum bandwidth at 16 KB / 1 MB / 1 GB

The numbers are sanity-checks against TB5 spec (~80 Gb/s = 10 GB/s usable).
If RDMA is misconfigured, you'll see ~10x lower throughput and ~5x higher
latency — that's the alarm.
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx

from .jaccl_init import init_world, print_zero, realize


def _bw_test(world, n_bytes: int, rounds: int = 3):
    n_floats = max(1, n_bytes // 4)
    x = mx.ones((n_floats,), dtype=mx.float32)
    realize(x)
    # warm
    realize(mx.distributed.all_sum(x))

    t0 = time.perf_counter()
    for _ in range(rounds):
        y = mx.distributed.all_sum(x)
        realize(y)
    dt = (time.perf_counter() - t0) / rounds
    bytes_moved = n_bytes * 2 * (world.size - 1)  # all_sum ring estimate
    return dt, bytes_moved / dt / 1e9  # GB/s


def _latency_test(world, rounds: int = 64) -> float:
    s = mx.array(1, dtype=mx.int32)
    realize(s)
    t0 = time.perf_counter()
    for _ in range(rounds):
        realize(mx.distributed.all_sum(s))
    return (time.perf_counter() - t0) / rounds * 1e6  # us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--sizes", nargs="*",
                    default=["16K", "1M", "16M", "256M", "1G"])
    args = ap.parse_args()

    def _parse(s: str) -> int:
        u = s[-1].upper()
        n = int(s[:-1]) if u in "KMG" else int(s)
        return n * {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}.get(u, 1)

    world = init_world()
    print_zero(world, f"world: rank {world.rank} of {world.size} "
                      f"on backend {world.backend}")

    lat = _latency_test(world)
    print_zero(world, f"latency (small all_sum, 64 rounds): {lat:.1f} us")

    for s in args.sizes:
        dt, bw = _bw_test(world, _parse(s), rounds=args.rounds)
        print_zero(world, f"  {s:>5}  all_sum dt={dt*1e3:7.2f} ms  bw={bw:6.2f} GB/s")


if __name__ == "__main__":
    main()
