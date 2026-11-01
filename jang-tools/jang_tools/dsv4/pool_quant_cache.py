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
import numpy as np

from .mlx_model import DeepseekV4Cache


_STATE_KEYS = ("buffer_kv", "buffer_gate", "pooled")
_POOL_SEGMENT_ROWS = 64
# Ceiling for lossless segment compaction. Without it the q8 tier keeps one
# 64-row segment per append chunk forever: at 1M tokens (ratio 4) that is
# ~3.9K segments per pool state, and decode pays a Python loop over ALL of
# them per layer per token in both iter_dequantized_tiles (indexer scan) and
# gather_dequantized_rows (CSA gather) — the measured ~100 ms/tok op flood.
# Equal-size neighbors merge via _concat_qpools (native q8, no requantize),
# binary-counter style, so each row is copied at most log2(16384/64)=8 times
# and the segment count stays O(rows/16K + log) instead of O(rows/64).
_POOL_SLAB_MAX_ROWS = 16 * 1024
_POOL_ATTENTION_TILE_ROWS = 16 * 1024
POOL_STORAGE_SCHEMA = "dsv4_pool_q8_segmented_v2"
_BRANCH_STORAGE_SCHEMA = "dsv4_pool_q8_branch_v2"
# Quantization has a fixed conversion cost and, worse, a per-token decode
# cost: once a pool promotes, every CSA layer pays a q8 dequant/gather op
# flood on the host each token (measured ~100 ms/tok at 24K ctx on M5 Max vs
# ~36 ms/tok on the BF16 fused-SDPA path). BF16 pools are cheap to retain: at
# ratio 4 the wide compressor pool holds 1 MiB per 4K prompt tokens, so 128K
# tokens is ~32 MiB per layer (~1.5 GB across all layers) — far inside the
# ~6-8 GB live-cache RAM budget. Keep pools attention-ready BF16 until one
# state would retain more than 64 MiB (~256K tokens at ratio 4); segmented q8
# remains for genuinely huge contexts and for restored q8 block state.
# Override with DSV4_POOL_BF16_MAX_BYTES.
def _pool_bf16_max_bytes() -> int:
    import os

    raw = os.environ.get("DSV4_POOL_BF16_MAX_BYTES", "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return 64 * 1024 * 1024


_POOL_BF16_MAX_BYTES = _pool_bf16_max_bytes()


# The 64 MiB hot tier keeps DECODE on the fused-SDPA fast path, but a dense
# pool returned to a PREFILL forward routes attention through the layer's
# dense fallback and the indexer's dense top-k scoring, whose transients grow
# with pool rows (~6-7.5 GB per layer past ~180K tokens on a 128 GB M5 Max —
# Metal OOM against a ~107.5 GB working-set limit). Cap how many pool rows a
# multi-token forward may see densely; beyond it, return the bounded
# _QuantizedPoolView so attention and selection tile over the SAME hot BF16
# storage (no quantization happens — the view slices _pooled_bf16 directly).
# Rows, not bytes: compressor (head_dim 512) and indexer (head_dim 128) pools
# must flip to tiling at the same token count or selection stays dense.
# 16384 rows = 65K tokens at ratio 4; dense transients are proven safe at 40K.
def _pool_prefill_dense_max_rows() -> int:
    import os

    raw = os.environ.get("DSV4_POOL_PREFILL_DENSE_MAX_ROWS", "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return 16384


_POOL_PREFILL_DENSE_MAX_ROWS = _pool_prefill_dense_max_rows()


def _quant_pool(pool: mx.array, group_size: int = 32, bits: int = 8):
    """Quantize a pool tensor along its last dimension with affine groups."""
    if pool is None:
        return None
    original_dtype = str(pool.dtype)
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
    mx.async_eval(q, scale, mn)
    return (q, scale, mn, shape, group_size, bits, original_dtype)


def _dequant_pool(qpool):
    """Dequantize a tuple produced by ``_quant_pool``."""
    if qpool is None:
        return None
    if len(qpool) == 7:
        q, scale, mn, shape, group_size, _bits, original_dtype = qpool
        target_dtype = _dtype_from_storage_name(original_dtype)
    else:
        # Backward compatibility for pre-storage-schema in-process states.
        q, scale, mn, shape, group_size, _bits = qpool
        target_dtype = mx.bfloat16
    x = q.astype(mx.float32) * scale.astype(mx.float32) + mn.astype(mx.float32)
    return x.reshape(shape).astype(target_dtype)


def _dequant_qpool_selected(qpool, batch_indices, row_indices):
    """Decode only selected batch/row occurrences from one q8 segment.

    DSV4's affine metadata is independent for every pool row.  Gathering the
    code, scale, and minimum leaves before dequantization is therefore exactly
    equivalent to dequantizing the segment first and then gathering, while the
    temporary geometry is bounded by the selected CSA top-k rather than the
    historical pool length.
    """
    q, scale, mn, shape, group_size, _bits, original_dtype = _validated_qpool(
        qpool
    )
    batch_indices = batch_indices.astype(mx.int32)
    row_indices = row_indices.astype(mx.int32)
    q = q[batch_indices, row_indices]
    scale = scale[batch_indices, row_indices]
    mn = mn[batch_indices, row_indices]
    x = q.astype(mx.float32) * scale.astype(mx.float32) + mn.astype(mx.float32)
    return x.reshape(int(row_indices.size), int(shape[-1])).astype(
        _dtype_from_storage_name(original_dtype)
    )


def _qpool_rows(qpool) -> int:
    """Return the number of pool rows represented by a quantized segment."""
    return 0 if qpool is None else int(qpool[3][1])


def _slice_qpool_rows(qpool, stop: int):
    """Keep ``[:stop]`` pool rows without dequantizing the segment.

    Affine groups span only the final feature dimension, so slicing the pool
    row axis preserves the original quantization parameters exactly.
    """
    return _slice_qpool_row_range(qpool, 0, stop)


def _slice_qpool_row_range(qpool, start: int, stop: int):
    """Keep ``[start:stop]`` pool rows without dequantizing the segment."""
    q, scale, mn, shape, group_size, bits, *dtype_tail = qpool
    start = max(0, min(int(start), int(shape[1])))
    stop = max(start, min(int(stop), int(shape[1])))

    def _row_slice(value):
        slices = [slice(None)] * value.ndim
        slices[1] = slice(start, stop)
        return value[tuple(slices)]

    q = _row_slice(q)
    scale = _row_slice(scale)
    mn = _row_slice(mn)
    rows = stop - start
    shape = tuple(rows if axis == 1 else dim for axis, dim in enumerate(shape))
    mx.async_eval(q, scale, mn)
    return (q, scale, mn, shape, group_size, bits, *dtype_tail)


def _dtype_from_storage_name(name: str):
    """Resolve an MLX dtype serialized as a stable string."""
    dtype = getattr(mx, str(name).replace("mlx.core.", ""), None)
    if dtype is None:
        raise ValueError(f"unsupported MLX dtype in DSV4 pool state: {name!r}")
    return dtype


def _validated_qpool(qpool):
    """Validate and normalize one encoded pool segment without decoding it."""
    if not isinstance(qpool, (tuple, list)) or len(qpool) not in (6, 7):
        raise ValueError("invalid DSV4 q8 pool segment")
    q, scale, mn, raw_shape, raw_group_size, raw_bits, *dtype_tail = qpool
    if not all(hasattr(part, "shape") for part in (q, scale, mn)):
        raise ValueError("DSV4 q8 pool segment is missing tensor leaves")
    try:
        shape = tuple(int(dim) for dim in raw_shape)
        group_size = int(raw_group_size)
        bits = int(raw_bits)
    except (TypeError, ValueError) as exc:
        raise ValueError("DSV4 q8 pool segment metadata is not integral") from exc
    if len(shape) < 2 or group_size <= 0 or shape[-1] % group_size:
        raise ValueError(
            "invalid DSV4 q8 pool geometry: "
            f"shape={shape} group_size={group_size}"
        )
    if bits != 8 or not str(q.dtype).endswith("uint8"):
        raise ValueError(
            f"invalid DSV4 pool codec: bits={bits} q_dtype={q.dtype}"
        )
    groups = shape[-1] // group_size
    expected_q = (*shape[:-1], groups, group_size)
    expected_params = (*shape[:-1], groups, 1)
    if tuple(q.shape) != expected_q:
        raise ValueError(
            f"invalid DSV4 q8 code shape: got={tuple(q.shape)} expected={expected_q}"
        )
    if tuple(scale.shape) != expected_params or tuple(mn.shape) != expected_params:
        raise ValueError(
            "invalid DSV4 q8 affine metadata shape: "
            f"scale={tuple(scale.shape)} min={tuple(mn.shape)} "
            f"expected={expected_params}"
        )
    original_dtype = (
        str(dtype_tail[0]) if dtype_tail else str(mx.bfloat16)
    )
    _dtype_from_storage_name(original_dtype)
    return (q, scale, mn, shape, group_size, bits, original_dtype)


def _qpool_merge_key(qpool):
    """Concat-compatibility key (everything but the row count), or ``None``.

    Compaction is an optimization pass and must never add a failure mode: a
    corrupt or mismatched pool has to reach the attention-time fail-closed
    misalignment guard, not die inside ``_concat_qpools``. Segments whose key
    is ``None`` or differs from their neighbor are simply left unmerged.
    """
    try:
        shape = tuple(int(dim) for dim in qpool[3])
        group_size = int(qpool[4])
        bits = int(qpool[5])
        dtype = str(qpool[6]) if len(qpool) > 6 else str(mx.bfloat16)
    except (TypeError, ValueError, IndexError):
        return None
    if len(shape) < 2:
        return None
    return (shape[:1], shape[2:], group_size, bits, dtype)


def _concat_qpools(qpools):
    """Join adjacent encoded rows without decoding or requantizing them.

    Pool affine parameters are per row and per final-dimension group.  The row
    axis is therefore a lossless concatenation boundary: codes, scales, and
    minima can be joined directly while retaining their original codec values.
    """
    normalized = [_validated_qpool(qpool) for qpool in qpools]
    if not normalized:
        raise ValueError("cannot concatenate an empty DSV4 q8 pool sequence")
    first = normalized[0]
    batch_shape = first[3][:1]
    tail_shape = first[3][2:]
    group_size = first[4]
    bits = first[5]
    original_dtype = first[6]
    for qpool in normalized[1:]:
        if (
            qpool[3][:1] != batch_shape
            or qpool[3][2:] != tail_shape
            or qpool[4] != group_size
            or qpool[5] != bits
            or qpool[6] != original_dtype
        ):
            raise ValueError("incompatible DSV4 q8 pool segments")
    rows = sum(_qpool_rows(qpool) for qpool in normalized)
    q = mx.concatenate([qpool[0] for qpool in normalized], axis=1)
    scale = mx.concatenate([qpool[1] for qpool in normalized], axis=1)
    mn = mx.concatenate([qpool[2] for qpool in normalized], axis=1)
    shape = (*batch_shape, rows, *tail_shape)
    mx.async_eval(q, scale, mn)
    return (q, scale, mn, shape, group_size, bits, original_dtype)


class _QuantizedPoolView:
    """Attention-facing view over bounded BF16 or segmented native-q8 rows.

    This object deliberately is not an ``mx.array``.  Consumers must iterate
    bounded tiles, making a full historical BF16 pool an explicit compatibility
    operation rather than an accidental decode-time side effect.
    """

    is_dsv4_quantized_pool_view = True

    def __init__(self, state: "_StateProxy", empty_like=None):
        self._state = state
        self._empty_like = empty_like

    @property
    def shape(self):
        shape, _dtype = self._state.pool_shape_dtype(self._empty_like)
        return shape

    @property
    def dtype(self):
        _shape, dtype = self._state.pool_shape_dtype(self._empty_like)
        return dtype

    def iter_dequantized_tiles(self, max_rows: int = _POOL_ATTENTION_TILE_ROWS):
        yield from self._state.iter_dequantized_tiles(max_rows=max_rows)

    def gather_dequantized_rows(self, indices):
        return self._state.gather_dequantized_rows(indices)

    def materialize(self):
        return self._state.materialize_pooled(self._empty_like)


class _StateProxy(MutableMapping[str, Any]):
    """Dict-like state object that quantizes only the ``pooled`` slot."""

    def __init__(self, initial: dict[str, Any] | None = None):
        self._data = {"buffer_kv": None, "buffer_gate": None}
        self._pooled_bf16 = None
        self._pooled_q_segments: list[Any] = []
        self._pooled_empty_spec: tuple[tuple[int, ...], Any] | None = None
        self._q_rows_deferred: int | None = None
        if initial:
            for key, value in initial.items():
                self[key] = value

    def __getitem__(self, key: str) -> Any:
        if key == "pooled":
            return self.materialize_pooled()
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "pooled":
            self._pooled_bf16 = None
            self._pooled_q_segments = []
            self._pooled_empty_spec = None
            self._q_rows_deferred = None
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
        """Replace pooled storage with bounded quantized row slabs."""
        self._pooled_bf16 = None
        self._pooled_q_segments = []
        rows = int(value.shape[1])
        for start in range(0, rows, _POOL_SLAB_MAX_ROWS):
            self._pooled_q_segments.append(
                _quant_pool(value[:, start:start + _POOL_SLAB_MAX_ROWS])
            )

    def pool_shape_dtype(self, empty_like=None):
        """Return logical array geometry without decoding native q8 storage."""
        if self._pooled_bf16 is not None:
            return tuple(self._pooled_bf16.shape), self._pooled_bf16.dtype
        if self._pooled_q_segments:
            first = self._pooled_q_segments[0]
            rows = sum(_qpool_rows(segment) for segment in self._pooled_q_segments)
            shape = tuple(rows if axis == 1 else dim for axis, dim in enumerate(first[3]))
            return shape, _dtype_from_storage_name(first[6])
        if self._pooled_empty_spec is not None:
            return self._pooled_empty_spec
        if empty_like is not None:
            return (
                (int(empty_like.shape[0]), 0, int(empty_like.shape[-1])),
                empty_like.dtype,
            )
        return None, None

    def pool_view(self, empty_like=None):
        return _QuantizedPoolView(self, empty_like=empty_like)

    def iter_dequantized_tiles(self, max_rows: int = _POOL_ATTENTION_TILE_ROWS):
        """Yield ``(global_row_start, BF16_tile)`` under a strict row bound."""
        max_rows = max(1, int(max_rows))
        if self._pooled_bf16 is not None:
            rows = int(self._pooled_bf16.shape[1])
            for start in range(0, rows, max_rows):
                yield start, self._pooled_bf16[:, start:start + max_rows]
            return

        cursor = 0
        pending = []
        pending_start = 0
        pending_rows = 0

        def _flush():
            nonlocal pending, pending_start, pending_rows
            if not pending:
                return None
            parts = [_dequant_pool(segment) for segment in pending]
            tile = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)
            result = (pending_start, tile)
            pending = []
            pending_rows = 0
            return result

        for raw_segment in self._pooled_q_segments:
            segment = _validated_qpool(raw_segment)
            segment_rows = _qpool_rows(segment)
            segment_offset = 0
            while segment_offset < segment_rows:
                available = max_rows - pending_rows
                take = min(available, segment_rows - segment_offset)
                piece = (
                    segment
                    if segment_offset == 0 and take == segment_rows
                    else _slice_qpool_row_range(
                        segment,
                        segment_offset,
                        segment_offset + take,
                    )
                )
                if not pending:
                    pending_start = cursor + segment_offset
                pending.append(piece)
                pending_rows += take
                segment_offset += take
                if pending_rows == max_rows:
                    result = _flush()
                    if result is not None:
                        yield result
            cursor += segment_rows
        result = _flush()
        if result is not None:
            yield result

    def gather_dequantized_rows(self, indices):
        """Gather CSA-selected rows without decoding an unselected pool row.

        ``indices`` has shape ``(B, ..., K)`` and contains global pool-row
        indices.  The indexer result has already been materialized at the
        tiled top-k boundary, so one small device-to-host copy lets us map each
        occurrence to its owning q8 segment.  Only those code/scale/minimum
        rows are gathered and decoded; the ordered result is returned with
        shape ``(*indices.shape, D)``.
        """
        logical_shape, _dtype = self.pool_shape_dtype()
        if logical_shape is None:
            raise ValueError("cannot gather rows from an absent DSV4 pool")
        if int(indices.ndim) < 2:
            raise ValueError("DSV4 pool gather indices must include batch and row axes")
        batch = int(logical_shape[0])
        rows = int(logical_shape[1])
        width = int(logical_shape[-1])
        if int(indices.shape[0]) != batch:
            raise ValueError(
                "DSV4 pool gather batch mismatch: "
                f"indices={int(indices.shape[0])} pool={batch}"
            )

        host_indices = np.asarray(indices, dtype=np.int64)
        if host_indices.size == 0:
            return mx.zeros((*indices.shape, width), dtype=_dtype)
        if host_indices.min() < 0 or host_indices.max() >= rows:
            raise IndexError(
                "DSV4 pool gather index is outside retained rows: "
                f"min={int(host_indices.min())} max={int(host_indices.max())} "
                f"rows={rows}"
            )

        if self._pooled_bf16 is not None:
            batch_shape = (batch,) + (1,) * (int(indices.ndim) - 1)
            batch_indices = mx.broadcast_to(
                mx.arange(batch, dtype=mx.int32).reshape(batch_shape),
                indices.shape,
            )
            return self._pooled_bf16[
                batch_indices,
                indices.astype(mx.int32),
            ]

        flat_indices = host_indices.reshape(-1)
        per_batch = int(host_indices[0].size)
        flat_positions = np.arange(flat_indices.size, dtype=np.int64)
        batch_indices = flat_positions // per_batch
        segment_rows = np.asarray(
            [_qpool_rows(segment) for segment in self._pooled_q_segments],
            dtype=np.int64,
        )
        segment_ends = np.cumsum(segment_rows)
        segment_ids = np.searchsorted(segment_ends, flat_indices, side="right")
        segment_starts = np.concatenate(
            [np.zeros((1,), dtype=np.int64), segment_ends[:-1]]
        )

        decoded_parts = []
        decoded_positions = []
        for segment_id in np.unique(segment_ids):
            positions = np.flatnonzero(segment_ids == segment_id)
            local_rows = (
                flat_indices[positions] - segment_starts[int(segment_id)]
            )
            decoded_parts.append(
                _dequant_qpool_selected(
                    self._pooled_q_segments[int(segment_id)],
                    mx.array(batch_indices[positions], dtype=mx.int32),
                    mx.array(local_rows, dtype=mx.int32),
                )
            )
            decoded_positions.append(positions)

        decoded = (
            decoded_parts[0]
            if len(decoded_parts) == 1
            else mx.concatenate(decoded_parts, axis=0)
        )
        positions = np.concatenate(decoded_positions)
        restore_order = np.argsort(positions, kind="stable")
        if restore_order.size > 1:
            decoded = mx.take(
                decoded,
                mx.array(restore_order, dtype=mx.int32),
                axis=0,
            )
        return decoded.reshape(*indices.shape, width)

    def materialize_pooled(self, empty_like=None):
        """Compatibility-only full pool materialization; never retained."""
        if self._pooled_bf16 is not None:
            return self._pooled_bf16
        parts = [tile for _start, tile in self.iter_dequantized_tiles()]
        if parts:
            return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)
        shape, dtype = self.pool_shape_dtype(empty_like)
        if shape is None:
            return None
        return mx.zeros(shape, dtype=dtype)

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
        """Losslessly keep the sub-slab tail merged as one q8 segment.

        Decode appends one pool row per window; leaving each as its own
        segment puts up to 63 tiny segments on every per-token tile scan.
        Merging the tail on every append keeps it a single segment, and once
        it completes a 64-row chunk the slab cascade takes over.
        """
        start, rows = self._tail_segments()
        tail = self._pooled_q_segments[start:]
        if len(tail) >= 2:
            keys = {_qpool_merge_key(segment) for segment in tail}
            if len(keys) == 1 and None not in keys:
                self._pooled_q_segments[start:] = [_concat_qpools(tail)]
        if rows >= _POOL_SEGMENT_ROWS:
            self._compact_slab_cascade()

    def _compact_slab_cascade(self) -> None:
        """Merge trailing equal-size segments into bounded slabs (lossless)."""
        segments = self._pooled_q_segments
        while len(segments) >= 2:
            prev_rows = _qpool_rows(segments[-2])
            last_rows = _qpool_rows(segments[-1])
            if (
                prev_rows != last_rows
                or prev_rows + last_rows > _POOL_SLAB_MAX_ROWS
            ):
                return
            key = _qpool_merge_key(segments[-2])
            if key is None or key != _qpool_merge_key(segments[-1]):
                return
            segments[-2:] = [_concat_qpools(segments[-2:])]

    def _compact_segments_to_slabs(self) -> None:
        """Greedily regroup imported segments into <=16K-row slabs.

        Restored L2/delta state arrives with whatever segmentation it was
        exported under (a near-1M pool can carry thousands of 64-row wire
        segments). One native-q8 regroup on import keeps every later decode
        token off the per-segment op flood.

        Each merged slab is evaluated before the next one is built. The concat
        itself only async-schedules; a near-1M import/restore would otherwise
        push the entire regroup into the Metal stream without one blocking
        sync, which can exhaust device memory next to ~95GB of weights.
        """
        segments = self._pooled_q_segments
        if len(segments) <= 1:
            return

        def _emit_slab(pending):
            if len(pending) == 1:
                return pending[0]
            slab = _concat_qpools(pending)
            mx.eval(slab[0], slab[1], slab[2])
            return slab

        regrouped = []
        pending = []
        pending_rows = 0
        pending_key = None
        for segment in segments:
            rows = _qpool_rows(segment)
            key = _qpool_merge_key(segment)
            if pending and (
                pending_rows + rows > _POOL_SLAB_MAX_ROWS
                or key is None
                or key != pending_key
            ):
                regrouped.append(_emit_slab(pending))
                pending = []
                pending_rows = 0
            pending.append(segment)
            pending_rows += rows
            pending_key = key
        if pending:
            regrouped.append(_emit_slab(pending))
        self._pooled_q_segments = regrouped

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
            # Quantize the largest 64*2^k chunk that fits so bulk (prefill)
            # appends produce canonical counter sizes that keep merging.
            full = ((rows - offset) // _POOL_SEGMENT_ROWS) * _POOL_SEGMENT_ROWS
            take = _POOL_SEGMENT_ROWS
            while take * 2 <= min(full, _POOL_SLAB_MAX_ROWS):
                take *= 2
            self._pooled_q_segments.append(
                _quant_pool(value[:, offset:offset + take])
            )
            offset += take
            self._compact_slab_cascade()

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

    def export_storage_state(self):
        """Return the retained pool representation without dequantizing it.

        ``MutableMapping.values()`` and ``self["pooled"]`` intentionally expose
        an attention-ready BF16 view.  Cache snapshot/L2 code must not use that
        view: assigning it back would perform a second lossy q8 encode.  This
        tagged tree contains the actual q codes and affine metadata instead.
        """
        empty_shape = None
        empty_dtype = None
        if self._pooled_empty_spec is not None:
            shape, dtype = self._pooled_empty_spec
            empty_shape = tuple(int(dim) for dim in shape)
            empty_dtype = str(dtype)
        return (
            _BRANCH_STORAGE_SCHEMA,
            self._data.get("buffer_kv"),
            self._data.get("buffer_gate"),
            self._pooled_bf16,
            tuple(self._pooled_q_segments),
            empty_shape,
            empty_dtype,
        )

    def export_pool_delta(self, start_row: int, end_row: int, _collector=None):
        """Export retained rows without dequantizing or re-quantizing q8 data.

        ``_collector`` (optional): batch the BF16 hot-tier snapshot eval with
        the rest of the block-boundary leaves into one async materialization
        instead of a blocking mx.eval per layer. See
        ``DeepseekV4Cache._copy_delta_tree``.
        """
        from .cache_delta import DSV4_POOL_DELTA_SCHEMA

        start = int(start_row)
        end = int(end_row)
        if start < 0 or end < start:
            raise ValueError("invalid DSV4 pool delta row range")
        if self._pooled_bf16 is not None:
            rows = int(self._pooled_bf16.shape[1])
            if end > rows:
                raise ValueError("DSV4 BF16 pool delta exceeds retained rows")
            value = self._pooled_bf16[:, start:end, :] + mx.zeros_like(
                self._pooled_bf16[:, start:end, :]
            )
            if _collector is None:
                mx.eval(value)
            else:
                _collector.append(value)
            return {
                "schema": DSV4_POOL_DELTA_SCHEMA,
                "storage": "bf16",
                "start_row": start,
                "end_row": end,
                "value": value,
            }

        segments = []
        cursor = 0
        for segment in self._pooled_q_segments:
            rows = _qpool_rows(segment)
            segment_start = max(start - cursor, 0)
            segment_end = min(end - cursor, rows)
            if segment_start < segment_end:
                segments.append(
                    _slice_qpool_row_range(segment, segment_start, segment_end)
                )
            cursor += rows
            if cursor >= end:
                break
        if end > cursor and end > start:
            raise ValueError("DSV4 q8 pool delta exceeds retained rows")
        return {
            "schema": DSV4_POOL_DELTA_SCHEMA,
            "storage": "q8",
            "start_row": start,
            "end_row": end,
            "segments": tuple(segments),
        }

    def append_pool_delta(self, delta, *, defer_compaction: bool = False) -> None:
        """Append a lossless BF16/q8 delta while preserving q8 codes directly.

        ``defer_compaction=True`` skips the per-append slab regroup and reuses
        a running row count for the contiguity check. A near-1M restore chain
        (~1.6K records per branch) otherwise re-concatenates the trailing slab
        on every append — an unsynchronized Metal burst that can OOM a
        weight-heavy host. Deferred callers MUST invoke
        :meth:`finalize_deferred_pool_appends` once the chain is applied.
        """
        from .cache_delta import DSV4_POOL_DELTA_SCHEMA

        if not isinstance(delta, dict) or delta.get("schema") != DSV4_POOL_DELTA_SCHEMA:
            raise ValueError("unsupported DSV4 pool delta")
        start = int(delta.get("start_row", -1))
        end = int(delta.get("end_row", -1))
        if self._pooled_bf16 is not None:
            current_rows = int(self._pooled_bf16.shape[1])
        elif defer_compaction and self._q_rows_deferred is not None:
            current_rows = self._q_rows_deferred
        else:
            current_rows = sum(
                _qpool_rows(segment) for segment in self._pooled_q_segments
            )
        if start != current_rows or end < start:
            raise ValueError(
                "DSV4 pool delta is not contiguous: "
                f"current={current_rows} start={start} end={end}"
            )
        expected_rows = end - start
        storage = delta.get("storage")
        if storage == "none":
            if start or end:
                raise ValueError("absent DSV4 pool delta cannot declare rows")
            return
        if storage == "bf16":
            value = delta.get("value")
            if value is None or int(value.shape[1]) != expected_rows:
                if expected_rows == 0 and value is None:
                    return
                raise ValueError("invalid DSV4 BF16 pool delta")
            if self._pooled_q_segments:
                self._pooled_q_segments.append(_quant_pool(value))
                self._compact_full_tail()
            else:
                self.append_pooled(value)
            if defer_compaction:
                self._q_rows_deferred = end
            return
        if storage != "q8":
            raise ValueError("unsupported DSV4 pool delta storage")
        segments = [
            _validated_qpool(segment) for segment in (delta.get("segments") or ())
        ]
        if sum(_qpool_rows(segment) for segment in segments) != expected_rows:
            raise ValueError("DSV4 q8 pool delta row count does not match metadata")
        if self._pooled_bf16 is not None:
            self._replace_quantized(self._pooled_bf16)
        self._pooled_empty_spec = None
        self._pooled_q_segments.extend(segments)
        if defer_compaction:
            self._q_rows_deferred = end
        else:
            self._q_rows_deferred = None
            self._compact_segments_to_slabs()

    def finalize_deferred_pool_appends(self) -> None:
        """Regroup once after a deferred ``append_pool_delta`` sequence."""
        self._q_rows_deferred = None
        self._compact_segments_to_slabs()

    def import_storage_state(self, value) -> None:
        """Install a lossless state tree produced by ``export_storage_state``."""
        if (
            not isinstance(value, (tuple, list))
            or len(value) != 7
            or value[0] != _BRANCH_STORAGE_SCHEMA
        ):
            raise ValueError("unsupported DSV4 pool branch storage state")
        (
            _schema,
            buffer_kv,
            buffer_gate,
            pooled_bf16,
            raw_segments,
            raw_empty_shape,
            raw_empty_dtype,
        ) = value
        if raw_segments is None:
            raw_segments = ()
        if not isinstance(raw_segments, (tuple, list)):
            raise ValueError("DSV4 q8 pool segments must be a sequence")
        segments = [_validated_qpool(segment) for segment in raw_segments]
        if pooled_bf16 is not None and segments:
            raise ValueError("DSV4 pool state cannot contain BF16 and q8 tiers together")

        empty_spec = None
        if raw_empty_shape is not None or raw_empty_dtype is not None:
            if raw_empty_shape is None or raw_empty_dtype is None:
                raise ValueError("incomplete DSV4 empty-pool storage metadata")
            try:
                empty_shape = tuple(int(dim) for dim in raw_empty_shape)
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid DSV4 empty-pool shape") from exc
            if len(empty_shape) < 2 or empty_shape[1] != 0:
                raise ValueError(f"invalid DSV4 empty-pool shape: {empty_shape}")
            empty_spec = (empty_shape, _dtype_from_storage_name(raw_empty_dtype))
        if empty_spec is not None and (pooled_bf16 is not None or segments):
            raise ValueError("DSV4 empty-pool metadata conflicts with retained pool data")

        self._data = {"buffer_kv": buffer_kv, "buffer_gate": buffer_gate}
        self._pooled_bf16 = pooled_bf16
        self._pooled_q_segments = segments
        self._pooled_empty_spec = empty_spec
        self._q_rows_deferred = None
        self._compact_segments_to_slabs()

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
        self.flush_pool_pending()
        local_state = None if self.local.empty() else self.local.state[:2]
        return (
            local_state,
            tuple(self.compressor_state[k] for k in _STATE_KEYS),
            tuple(self.indexer_state[k] for k in _STATE_KEYS),
        )

    @state.setter
    def state(self, value):
        self._pool_pending = None
        local_state, compressor_state, indexer_state = value
        if local_state is None:
            self.local.keys = None
            self.local.values = None
        else:
            self.local.keys, self.local.values = local_state
        self.compressor_state = dict(zip(_STATE_KEYS, compressor_state))
        self.indexer_state = dict(zip(_STATE_KEYS, indexer_state))

    @property
    def storage_state(self):
        """Lossless prompt/L2 state preserving native q8 pool segments."""
        self.flush_pool_pending()
        local_state = None if self.local.empty() else self.local.state[:2]
        return (
            POOL_STORAGE_SCHEMA,
            local_state,
            self.compressor_state.export_storage_state(),
            self.indexer_state.export_storage_state(),
        )

    @storage_state.setter
    def storage_state(self, value):
        if (
            not isinstance(value, (tuple, list))
            or len(value) != 4
            or value[0] != POOL_STORAGE_SCHEMA
        ):
            raise ValueError("unsupported DSV4 pool cache storage state")
        _schema, local_state, compressor_state, indexer_state = value
        self._pool_pending = None
        if local_state is None:
            self.local.keys = None
            self.local.values = None
        else:
            self.local.keys, self.local.values = local_state
        compressor = _StateProxy()
        compressor.import_storage_state(compressor_state)
        indexer = _StateProxy()
        indexer.import_storage_state(indexer_state)
        self._compressor_state = compressor
        self._indexer_state = indexer

    def _export_pool_delta(self, state, start_row, end_row, _collector=None):
        if not isinstance(state, _StateProxy):
            raise ValueError("PoolQuantizedV4Cache requires _StateProxy branches")
        return state.export_pool_delta(start_row, end_row, _collector)

    def _append_pool_delta(self, state, delta, *, defer_compaction: bool = False):
        if not isinstance(state, _StateProxy):
            raise ValueError("PoolQuantizedV4Cache requires _StateProxy branches")
        state.append_pool_delta(delta, defer_compaction=defer_compaction)

    def _finalize_pool_delta_appends(self, state):
        if not isinstance(state, _StateProxy):
            raise ValueError("PoolQuantizedV4Cache requires _StateProxy branches")
        state.finalize_deferred_pool_appends()

    def update_pool_view(self, new_pooled, state_key, seq_len=None):
        """Append rows and return the active attention representation.

        Pools below ``_POOL_BF16_MAX_BYTES`` are deliberately retained as a
        normal BF16 array.  For DECODE (``seq_len == 1``) return that array
        directly so single-token forwards keep the fused-SDPA fast path.
        For PREFILL (``seq_len > 1``) a dense return routes the layer through
        its dense attention fallback and dense top-k selection, whose
        transients scale with pool rows and Metal-OOM past ~180K tokens.
        Once the hot tier exceeds ``_POOL_PREFILL_DENSE_MAX_ROWS``, prefill
        gets the bounded view instead — it tiles over the same BF16 storage
        without quantizing anything, so decode keeps its dense hot tier.

        Once the state promotes to segmented q8 storage, return the bounded
        view so attention never materializes the complete historical pool.
        """
        state = self._branch_state(state_key)
        if seq_len is None:
            is_prefill = int(new_pooled.shape[1]) > 1
        else:
            is_prefill = int(seq_len) > 1
        if new_pooled.shape[1] > 0:
            if isinstance(state, _StateProxy):
                state.append_pooled(new_pooled)
            else:
                pool = state["pooled"]
                pool = new_pooled if pool is None else mx.concatenate([pool, new_pooled], axis=1)
                state["pooled"] = pool
                return pool
        if isinstance(state, _StateProxy):
            hot = state._pooled_bf16
            if hot is not None and (
                not is_prefill
                or int(hot.shape[1]) <= _POOL_PREFILL_DENSE_MAX_ROWS
            ):
                return hot
            return state.pool_view(empty_like=new_pooled)
        pool = state["pooled"]
        if pool is None:
            return mx.zeros(
                (new_pooled.shape[0], 0, new_pooled.shape[-1]),
                new_pooled.dtype,
            )
        return pool

    def update_pool(self, new_pooled, state_key):
        """Compatibility array API; inference uses ``update_pool_view``."""
        pool = self.update_pool_view(new_pooled, state_key)
        if getattr(pool, "is_dsv4_quantized_pool_view", False):
            return pool.materialize()
        return pool

    def trim(self, n):
        """Trim local KV and pooled rows without materializing quantized storage."""
        self._drop_pool_pending()
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
        if self._pool_pending:
            for slot in self._pool_pending.values():
                if slot:
                    for arr in slot["xs"]:
                        total += arr.nbytes
        return total
