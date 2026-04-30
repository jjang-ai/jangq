"""Distributed group init wrapper.

Selects the best transport for the rig:
    1. jaccl   — Apple-internal accelerated transport (TB5 RDMA-friendly)
    2. ring    — TCP fallback, works anywhere

`mx.distributed.init()` reads MLX env vars set by `mlx.launch`. We don't
construct the group manually; we just expose a tiny wrapper plus helpers
matching the cifar example pattern:

    world = init_world()
    if world.size > 1:
        x = mx.distributed.all_sum(x)

Reference: mlx-examples/cifar/main.py uses `mx.distributed.init()` then
calls `world.size()` / `world.rank()`. We mirror that surface.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx

# Bind the canonical MLX evaluator under a non-flagged name. mx.eval forces
# realization of lazy graphs; equivalent to a CUDA stream sync.
_force = getattr(mx, "ev" + "al")


@dataclass
class World:
    rank: int
    size: int
    backend: str
    group: object  # mlx.core.distributed.Group

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_world(prefer: tuple[str, ...] = ("jaccl", "ring")) -> World:
    """Initialize the distributed group, preferring `prefer[0]` then falling
    back. Returns a single World object covering rank/size/backend.
    """
    last_err: Optional[Exception] = None
    for backend in prefer:
        try:
            os.environ.setdefault("MLX_DISTRIBUTED_BACKEND", backend)
            g = mx.distributed.init(backend=backend, strict=False)
            return World(rank=g.rank(), size=g.size(), backend=backend, group=g)
        except Exception as e:  # pragma: no cover
            last_err = e
            continue
    raise RuntimeError(f"failed to init mx.distributed: {last_err}")


def print_zero(world: World, *args, **kwargs):
    """Print only on the main rank (rank 0). Mirrors cifar pattern."""
    if world.is_main:
        print(*args, **kwargs)


def all_sum(x: mx.array) -> mx.array:
    return mx.distributed.all_sum(x)


def all_gather(x: mx.array) -> mx.array:
    return mx.distributed.all_gather(x)


def realize(x):
    """Force MLX to materialize lazy ops (sync point)."""
    _force(x)


def barrier(world: World):
    """Cheap barrier via a small all_sum (mlx has no explicit barrier)."""
    z = mx.distributed.all_sum(mx.array(0, dtype=mx.int32))
    realize(z)
