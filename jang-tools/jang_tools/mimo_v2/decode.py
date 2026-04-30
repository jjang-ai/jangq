"""Decode helpers: greedy + temperature/top-p sampling, with KV-cache.

Reused by `runtime.py`, the eval scripts (HumanEval+/MMLU), and the
distributed launcher. Operates on `MiMoV2ForCausalLM` and the per-layer
cache returned by its `__call__`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import mlx.core as mx

from .model import MiMoV2ForCausalLM


@dataclass
class SamplerCfg:
    temperature: float = 0.0   # 0 = greedy
    top_p: float = 1.0
    repetition_penalty: float = 1.0  # off by default per glm51 finding
    seed: Optional[int] = None


def _sample(logits: mx.array, cfg: SamplerCfg) -> int:
    if cfg.temperature <= 0.0:
        return int(mx.argmax(logits).item())
    logits = logits / cfg.temperature
    probs = mx.softmax(logits, axis=-1)
    if cfg.top_p < 1.0:
        order = mx.argsort(-probs)
        sorted_p = mx.take(probs, order)
        cum = mx.cumsum(sorted_p)
        cutoff = mx.argmax((cum > cfg.top_p).astype(mx.int32)).item() + 1
        keep = order[:cutoff]
        kept = mx.take(probs, keep)
        kept = kept / mx.sum(kept)
        if cfg.seed is not None:
            mx.random.seed(cfg.seed)
        choice = mx.random.categorical(mx.log(kept)).item()
        return int(keep[choice].item())
    if cfg.seed is not None:
        mx.random.seed(cfg.seed)
    return int(mx.random.categorical(mx.log(probs)).item())


def stream_decode(
    model: MiMoV2ForCausalLM,
    prompt_ids: list[int],
    *,
    max_new: int,
    eos_ids: tuple[int, ...] = (),
    sampler: SamplerCfg = SamplerCfg(),
) -> Iterator[int]:
    """Yield generated token ids one at a time (KV-cache enabled)."""
    x = mx.array([prompt_ids], dtype=mx.uint32)
    logits, caches = model(x, caches=None)
    for _ in range(max_new):
        nxt = _sample(logits[0, -1], sampler)
        yield nxt
        if nxt in eos_ids:
            return
        x = mx.array([[nxt]], dtype=mx.uint32)
        logits, caches = model(x, caches=caches)
