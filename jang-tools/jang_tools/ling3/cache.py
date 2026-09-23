"""Decode caches for the Ling-3.0 MLX runtime (Python-side eval harness).

Created by Jinho Jang (eric@jangq.ai) — 2026-08-26.

Two cache kinds, matching the hybrid layer split:

  * `MLACache`   — standard growing KV for the 6 full-attention layers.
  * `KDACache`   — fixed-size recurrent state `[B,H,K,V]` + three conv tails.

The correctness proof for the KDA seam (split at non-aligned lengths, then
continue) lives in `tests/test_ling3_kda_vs_torch.py`; the model-level test in
`tests/test_ling3_cache.py` proves cached decode matches full recompute on the
real weights.
"""

from __future__ import annotations

import mlx.core as mx


class MLACache:
    """Append-only KV cache, grown in steps to avoid per-token reallocation."""

    step = 256

    def __init__(self):
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        prev = self.offset
        B, H, T, Dk = keys.shape
        Dv = values.shape[-1]
        if self.keys is None or prev + T > self.keys.shape[2]:
            n_steps = (self.step + T - 1) // self.step
            new_k = mx.zeros((B, H, n_steps * self.step, Dk), keys.dtype)
            new_v = mx.zeros((B, H, n_steps * self.step, Dv), values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[:, :, :prev]
                    self.values = self.values[:, :, :prev]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
        self.keys[:, :, prev : prev + T] = keys
        self.values[:, :, prev : prev + T] = values
        self.offset += T
        return self.keys[:, :, : self.offset], self.values[:, :, : self.offset]


class KDACache:
    """Recurrent state + conv tails. Fixed size; `offset` tracks position only."""

    def __init__(self):
        self.rec_state: mx.array | None = None
        self.conv_state: tuple = (None, None, None)
        self.offset = 0


def make_cache(model) -> list:
    """One cache per layer, matching each layer's attention kind."""
    caches = []
    for layer in model.layers:
        caches.append(MLACache() if layer.is_full_attention else KDACache())
    return caches
