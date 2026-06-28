"""Pool-quantized DeepSeek-V4 long-context cache.

DSV4's long-context path keeps local SWA K/V in a RotatingKVCache and stores
CSA/HSA compressed global-context pools in ``compressor_state`` and
``indexer_state``. The pools are averaged vectors, so 4-bit affine
round-trips preserve direction well while reducing live pool memory.

This module intentionally subclasses ``DeepseekV4Cache``. Older builds used a
peer class, which failed ``isinstance(cache, DeepseekV4Cache)`` in
``DeepseekV4Attention`` and silently disabled CSA/HSA.
"""
from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from typing import Any

import mlx.core as mx

from .mlx_model import DeepseekV4Cache


_STATE_KEYS = ("buffer_kv", "buffer_gate", "pooled")


def _quant_pool(pool: mx.array, group_size: int = 32, bits: int = 4):
    """Quantize a pool tensor along its last dimension with affine groups."""
    if pool is None:
        return None
    x = pool.astype(mx.float32)
    shape = tuple(x.shape)
    last = shape[-1]
    if last % group_size != 0:
        group_size = last
    groups = last // group_size
    xr = x.reshape(*shape[:-1], groups, group_size)
    mn = mx.min(xr, axis=-1, keepdims=True)
    mxv = mx.max(xr, axis=-1, keepdims=True)
    qmax = (1 << bits) - 1
    scale = (mxv - mn) / qmax
    scale = mx.where(scale > 1e-8, scale, mx.ones_like(scale))
    q = mx.round((xr - mn) / scale)
    q = mx.clip(q, 0, qmax).astype(mx.uint8)
    mx.eval(q, scale, mn)
    return (q, scale.astype(mx.float16), mn.astype(mx.float16), shape, group_size, bits)


def _dequant_pool(qpool):
    """Dequantize a tuple produced by ``_quant_pool``."""
    if qpool is None:
        return None
    q, scale, mn, shape, group_size, _bits = qpool
    x = q.astype(mx.float32) * scale.astype(mx.float32) + mn.astype(mx.float32)
    return x.reshape(shape).astype(mx.bfloat16)


class _StateProxy(MutableMapping[str, Any]):
    """Dict-like state object that quantizes only the ``pooled`` slot."""

    def __init__(self, initial: dict[str, Any] | None = None):
        self._data = {"buffer_kv": None, "buffer_gate": None}
        self._pooled_q_segments: list[Any] = []
        self._pooled_materialized: Any = None
        if initial:
            for key, value in initial.items():
                self[key] = value

    def __getitem__(self, key: str) -> Any:
        if key == "pooled":
            if self._pooled_materialized is not None:
                return self._pooled_materialized
            if not self._pooled_q_segments:
                return None
            parts = [
                part
                for part in (_dequant_pool(qpool) for qpool in self._pooled_q_segments)
                if part is not None
            ]
            if not parts:
                return None
            if len(parts) == 1:
                self._pooled_materialized = parts[0]
            else:
                self._pooled_materialized = mx.concatenate(parts, axis=1)
            return self._pooled_materialized
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "pooled":
            self._pooled_q_segments = [] if value is None else [_quant_pool(value)]
            self._pooled_materialized = value
        else:
            self._data[key] = value

    def append_pooled(self, value: Any) -> None:
        """Quantize and append only newly produced pool rows."""
        if value is None or value.shape[1] <= 0:
            return
        self._pooled_q_segments.append(_quant_pool(value))
        if self._pooled_materialized is None:
            self._pooled_materialized = value
        else:
            self._pooled_materialized = mx.concatenate(
                [self._pooled_materialized, value],
                axis=1,
            )

    def __delitem__(self, key: str) -> None:
        self[key] = None

    def __iter__(self) -> Iterator[str]:
        return iter(_STATE_KEYS)

    def __len__(self) -> int:
        return len(_STATE_KEYS)

    def values(self):
        return [self[key] for key in _STATE_KEYS]

    def quant_nbytes(self) -> int:
        total = 0
        for key in ("buffer_kv", "buffer_gate"):
            value = self._data.get(key)
            if value is not None:
                total += getattr(value, "nbytes", 0)
        for qpool in self._pooled_q_segments:
            if qpool is None:
                continue
            for part in qpool[:3]:
                total += getattr(part, "nbytes", 0)
        total += getattr(self._pooled_materialized, "nbytes", 0)
        return total


class PoolQuantizedV4Cache(DeepseekV4Cache):
    """DeepseekV4Cache with 4-bit affine storage for compressed pools."""

    def __init__(self, sliding_window, compress_ratio=None):
        super().__init__(sliding_window, compress_ratio=compress_ratio)

    @property
    def compressor_state(self):
        return self._compressor_state

    @compressor_state.setter
    def compressor_state(self, value):
        if isinstance(value, _StateProxy):
            self._compressor_state = value
        else:
            self._compressor_state = _StateProxy(value or {})

    @property
    def indexer_state(self):
        return self._indexer_state

    @indexer_state.setter
    def indexer_state(self, value):
        if isinstance(value, _StateProxy):
            self._indexer_state = value
        else:
            self._indexer_state = _StateProxy(value or {})

    @property
    def state(self):
        local_state = None if self.local.empty() else self.local.state
        return (
            local_state,
            tuple(self.compressor_state[k] for k in _STATE_KEYS),
            tuple(self.indexer_state[k] for k in _STATE_KEYS),
        )

    @state.setter
    def state(self, value):
        local_state, compressor_state, indexer_state = value
        if local_state is None:
            self.local.keys = None
            self.local.values = None
        else:
            self.local.state = local_state
        self.compressor_state = dict(zip(_STATE_KEYS, compressor_state))
        self.indexer_state = dict(zip(_STATE_KEYS, indexer_state))

    def update_pool(self, new_pooled, state_key):
        state = self._branch_state(state_key)
        if new_pooled.shape[1] > 0:
            if isinstance(state, _StateProxy):
                state.append_pooled(new_pooled)
                pool = state["pooled"]
            else:
                pool = state["pooled"]
                pool = new_pooled if pool is None else mx.concatenate([pool, new_pooled], axis=1)
                state["pooled"] = pool
        else:
            pool = state["pooled"]
        if pool is None:
            pool = mx.zeros((new_pooled.shape[0], 0, new_pooled.shape[-1]), new_pooled.dtype)
        return pool

    @property
    def nbytes(self):
        total = self.local.nbytes
        total += self.compressor_state.quant_nbytes()
        total += self.indexer_state.quant_nbytes()
        return total
