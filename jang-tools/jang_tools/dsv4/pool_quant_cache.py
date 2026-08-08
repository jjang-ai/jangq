"""Pool-quantized DeepSeek-V4 long-context cache.

DSV4's long-context path keeps local SWA K/V in a RotatingKVCache and stores
CSA/HSA compressed global-context pools in ``compressor_state`` and
``indexer_state``. Pool codes use ``uint8`` containers, so the full 8-bit
range costs the same retained bytes as the historical 4-bit path while
materially reducing accumulated long-context attention error.

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
_POOL_SEGMENT_ROWS = 64
# Quantization has a fixed conversion cost and codes are stored in uint8
# containers. Keep small pools in their attention-ready BF16
# form until one state would retain more than 2 MiB. At ratio 4, the wide
# compressor pool is 1 MiB for a 4K-token prompt and exactly 2 MiB for 8K, so
# ordinary <=4K prompts avoid quantize/dequantize work while a 12K prompt
# (about 3 MiB of pooled BF16) still promotes. The byte budget also lets the
# narrower indexer and high-ratio pools remain hot for substantially longer.
_POOL_BF16_MAX_BYTES = 2 * 1024 * 1024


def _quant_pool(pool: mx.array, group_size: int = 32, bits: int = 8):
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
    scale = scale.astype(mx.float16)
    mn = mn.astype(mx.float16)
    mx.eval(q, scale, mn)
    return (q, scale, mn, shape, group_size, bits)


def _dequant_pool(qpool):
    """Dequantize a tuple produced by ``_quant_pool``."""
    if qpool is None:
        return None
    q, scale, mn, shape, group_size, _bits = qpool
    x = q.astype(mx.float32) * scale.astype(mx.float32) + mn.astype(mx.float32)
    return x.reshape(shape).astype(mx.bfloat16)


def _qpool_rows(qpool) -> int:
    """Return the number of pool rows represented by a quantized segment."""
    return 0 if qpool is None else int(qpool[3][1])


def _slice_qpool_rows(qpool, stop: int):
    """Keep ``[:stop]`` pool rows without dequantizing the segment.

    Affine groups span only the final feature dimension, so slicing the pool
    row axis preserves the original quantization parameters exactly.
    """
    q, scale, mn, shape, group_size, bits = qpool
    stop = max(0, min(int(stop), int(shape[1])))

    def _row_slice(value):
        slices = [slice(None)] * value.ndim
        slices[1] = slice(0, stop)
        return value[tuple(slices)]

    q = _row_slice(q)
    scale = _row_slice(scale)
    mn = _row_slice(mn)
    shape = tuple(stop if axis == 1 else dim for axis, dim in enumerate(shape))
    mx.eval(q, scale, mn)
    return (q, scale, mn, shape, group_size, bits)


class _StateProxy(MutableMapping[str, Any]):
    """Dict-like state object that quantizes only the ``pooled`` slot."""

    def __init__(self, initial: dict[str, Any] | None = None):
        self._data = {"buffer_kv": None, "buffer_gate": None}
        self._pooled_bf16 = None
        self._pooled_q_segments: list[Any] = []
        self._pooled_empty_spec: tuple[tuple[int, ...], Any] | None = None
        if initial:
            for key, value in initial.items():
                self[key] = value

    def __getitem__(self, key: str) -> Any:
        if key == "pooled":
            if self._pooled_bf16 is not None:
                return self._pooled_bf16
            if not self._pooled_q_segments:
                if self._pooled_empty_spec is not None:
                    shape, dtype = self._pooled_empty_spec
                    return mx.zeros(shape, dtype=dtype)
                return None
            parts = [
                part
                for part in (_dequant_pool(qpool) for qpool in self._pooled_q_segments)
                if part is not None
            ]
            if not parts:
                return None
            if len(parts) == 1:
                return parts[0]
            return mx.concatenate(parts, axis=1)
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "pooled":
            self._pooled_bf16 = None
            self._pooled_q_segments = []
            self._pooled_empty_spec = None
            if value is not None:
                rows = int(value.shape[1])
                if rows == 0:
                    self._pooled_empty_spec = (tuple(value.shape), value.dtype)
                elif int(value.nbytes) <= _POOL_BF16_MAX_BYTES:
                    self._pooled_bf16 = value
                else:
                    self._replace_quantized(value)
        else:
            self._data[key] = value

    def _replace_quantized(self, value: Any) -> None:
        """Replace pooled storage with bounded quantized row segments."""
        self._pooled_bf16 = None
        self._pooled_q_segments = []
        rows = int(value.shape[1])
        for start in range(0, rows, _POOL_SEGMENT_ROWS):
            self._pooled_q_segments.append(
                _quant_pool(value[:, start:start + _POOL_SEGMENT_ROWS])
            )

    def _tail_segments(self) -> tuple[int, int]:
        """Return ``(start, rows)`` for trailing sub-chunk segments."""
        start = len(self._pooled_q_segments)
        rows = 0
        while start > 0:
            segment_rows = _qpool_rows(self._pooled_q_segments[start - 1])
            if segment_rows >= _POOL_SEGMENT_ROWS:
                break
            start -= 1
            rows += segment_rows
        return start, rows

    def _compact_full_tail(self) -> None:
        """Merge one completed tail chunk without retaining its BF16 form."""
        start, rows = self._tail_segments()
        if rows != _POOL_SEGMENT_ROWS:
            return
        parts = [_dequant_pool(qpool) for qpool in self._pooled_q_segments[start:]]
        materialized = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)
        compacted = _quant_pool(materialized)
        self._pooled_q_segments[start:] = [compacted]

    def append_pooled(self, value: Any) -> None:
        """Append rows in BF16 until the byte threshold, then quantize once.

        Decode commonly appends one row at a time. Small pools remain directly
        usable by attention, avoiding a full quantize/dequantize round-trip on
        every token. Once the combined BF16 state exceeds the hot-tier byte
        budget it is promoted to segmented quantized storage. Subsequent small
        quantized segments are compacted every ``_POOL_SEGMENT_ROWS`` rows.
        """
        if value is None or value.shape[1] <= 0:
            return
        self._pooled_empty_spec = None

        if self._pooled_bf16 is not None or not self._pooled_q_segments:
            combined = (
                value
                if self._pooled_bf16 is None
                else mx.concatenate([self._pooled_bf16, value], axis=1)
            )
            if int(combined.nbytes) <= _POOL_BF16_MAX_BYTES:
                self._pooled_bf16 = combined
                return
            self._replace_quantized(combined)
            return

        rows = int(value.shape[1])
        offset = 0

        _, tail_rows = self._tail_segments()
        if tail_rows:
            take = min(_POOL_SEGMENT_ROWS - tail_rows, rows)
            self._pooled_q_segments.append(_quant_pool(value[:, :take]))
            offset += take
            self._compact_full_tail()

        while rows - offset >= _POOL_SEGMENT_ROWS:
            self._pooled_q_segments.append(
                _quant_pool(value[:, offset:offset + _POOL_SEGMENT_ROWS])
            )
            offset += _POOL_SEGMENT_ROWS

        if offset < rows:
            self._pooled_q_segments.append(_quant_pool(value[:, offset:]))

    def trim_pooled(self, rows_to_drop: int) -> None:
        """Drop trailing pool rows without changing the active storage tier."""
        remaining = max(0, int(rows_to_drop))
        if not remaining:
            return
        self._pooled_empty_spec = None
        if self._pooled_bf16 is not None:
            keep = max(0, int(self._pooled_bf16.shape[1]) - remaining)
            self._pooled_bf16 = (
                self._pooled_bf16[:, :keep]
                if keep > 0
                else None
            )
            return
        while remaining and self._pooled_q_segments:
            last = self._pooled_q_segments[-1]
            rows = _qpool_rows(last)
            if rows <= remaining:
                self._pooled_q_segments.pop()
                remaining -= rows
                continue
            self._pooled_q_segments[-1] = _slice_qpool_rows(
                last,
                rows - remaining,
            )
            remaining = 0

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
        if self._pooled_bf16 is not None:
            total += getattr(self._pooled_bf16, "nbytes", 0)
        for qpool in self._pooled_q_segments:
            if qpool is None:
                continue
            for part in qpool[:3]:
                total += getattr(part, "nbytes", 0)
        return total


class PoolQuantizedV4Cache(DeepseekV4Cache):
    """DeepseekV4Cache with 8-bit affine storage for compressed pools."""

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

    def trim(self, n):
        """Trim local KV and pooled rows without materializing quantized storage."""
        rv = self.local.trim(n)
        for state in (self.compressor_state, self.indexer_state):
            state["buffer_kv"] = None
            state["buffer_gate"] = None

        ratio = self.compress_ratio
        if ratio is None or ratio <= 0:
            for state in (self.compressor_state, self.indexer_state):
                state["pooled"] = None
            return rv

        rows_to_drop = max(1, n // ratio) if n > 0 else 0
        if rows_to_drop:
            self.compressor_state.trim_pooled(rows_to_drop)
            self.indexer_state.trim_pooled(rows_to_drop)
        return rv

    @property
    def nbytes(self):
        total = self.local.nbytes
        total += self.compressor_state.quant_nbytes()
        total += self.indexer_state.quant_nbytes()
        return total
