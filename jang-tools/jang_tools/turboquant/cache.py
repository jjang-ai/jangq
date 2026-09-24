"""
TurboQuantKVCache -- JANG-exclusive KV cache with TurboQuant compression.

Three-phase lifecycle:
  1. FILL: Raw float buffer at baseline Metal speed (zero overhead).
  2. COMPRESS: Encode old tokens to ~3-bit packed format (5x memory reduction).
     Decode once into a read-only float buffer. No per-step re-decoding.
  3. GENERATE: New tokens append to float window. Attention reads
     concat(decoded_buffer, float_window) — both float, full Metal SDPA speed.

JANG-gated: only instantiated via JANG loader for JANG models.
"""

from typing import Optional

import mlx.core as mx

from .pipeline import (
    TurboQuantEncoder,
    EncodedKeys,
    EncodedValues,
    encode_keys,
    decode_keys,
    encode_values,
    decode_values,
)


class TurboQuantKVCache:
    """KV cache with on-demand TurboQuant compression.

    Baseline speed at all times. compress() saves 5x memory and decodes
    the compressed region ONCE into a persistent read-only buffer.
    """

    _vmlx_batch_api = "turboquant_kv_v1"
    step = 256

    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        key_bits: int = 3,
        value_bits: int = 3,
        seed: int = 42,
        compress_after: int = 0,
        sink_tokens: int = 0,
    ):
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.offset = 0
        self.compress_after = compress_after
        self.sink_tokens = sink_tokens

        self._key_encoder = None
        self._value_encoder = None
        self._seed = seed

        # Float buffer for active tokens (pre-allocated, in-place writes)
        self.keys = None
        self.values = None

        # Compressed storage (packed bits — 5x smaller)
        self._compressed_keys: Optional[EncodedKeys] = None
        self._compressed_values: Optional[EncodedValues] = None
        self._compressed_tokens: int = 0

        # Decoded buffer (read-only float — decoded ONCE after compress)
        self._decoded_k_buffer: Optional[mx.array] = None
        self._decoded_v_buffer: Optional[mx.array] = None

        # Joined buffer: [decoded_buffer | float_window] pre-allocated
        # Avoids mx.concatenate on every step after compress
        self._joined_k: Optional[mx.array] = None
        self._joined_v: Optional[mx.array] = None

        # TurboQuant's decode pipeline computes in float32. Preserve the model's
        # attention dtype so compressed caches do not silently promote every
        # subsequent SDPA operation to float32.
        self._vmlx_tq_key_dtype = None
        self._vmlx_tq_value_dtype = None

        # BatchGenerator compatibility. In single-sequence mode `offset` is an
        # int. After extend()/prepare() it becomes a per-row mx.array and
        # `_idx` is the shared right edge, matching mlx-lm BatchKVCache.
        self.left_padding = None
        self._right_padding = None
        self._idx = None

    def _ensure_encoders(self):
        if self._key_encoder is None:
            self._key_encoder = TurboQuantEncoder(
                dim=self.key_dim, key_bits=self.key_bits,
                value_bits=self.value_bits, seed=self._seed,
            )
            self._value_encoder = TurboQuantEncoder(
                dim=self.value_dim, key_bits=self.key_bits,
                value_bits=self.value_bits, seed=self._seed + 500,
            )

    @property
    def key_encoder(self):
        self._ensure_encoders()
        return self._key_encoder

    @property
    def value_encoder(self):
        self._ensure_encoders()
        return self._value_encoder

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Append new K/V. Returns full cache for attention at baseline speed."""
        if self._vmlx_tq_key_dtype is None:
            self._vmlx_tq_key_dtype = keys.dtype
        if self._vmlx_tq_value_dtype is None:
            self._vmlx_tq_value_dtype = values.dtype
        if self._is_batched:
            return self._update_and_fetch_batched(keys, values)

        prev = self.offset
        num_new = keys.shape[2]

        if self._compressed_tokens > 0:
            # --- POST-COMPRESS PATH ---
            # _joined_k/v is pre-allocated: [decoded_buffer | space for new tokens]
            # Write new token directly at position self.offset — no concatenation.
            if self._joined_k is None or (self.offset + num_new) > self._joined_k.shape[2]:
                # Grow joined buffer
                B = self._decoded_k_buffer.shape[0]
                n_kv_heads = self._decoded_k_buffer.shape[1]
                total_needed = self.offset + num_new
                n_steps = (self.step + total_needed - 1) // self.step
                jk = mx.zeros((B, n_kv_heads, n_steps * self.step, self.key_dim), keys.dtype)
                jv = mx.zeros((B, n_kv_heads, n_steps * self.step, self.value_dim), values.dtype)
                # Copy decoded buffer into front
                ct = self._compressed_tokens
                jk[..., :ct, :] = self._decoded_k_buffer
                jv[..., :ct, :] = self._decoded_v_buffer
                # Copy any existing window tokens
                if self._joined_k is not None and self.offset > ct:
                    jk[..., ct:self.offset, :] = self._joined_k[..., ct:self.offset, :]
                    jv[..., ct:self.offset, :] = self._joined_v[..., ct:self.offset, :]
                self._joined_k = jk
                self._joined_v = jv

            # In-place write at offset position (fast, no allocation)
            self._joined_k[..., self.offset:self.offset + num_new, :] = keys
            self._joined_v[..., self.offset:self.offset + num_new, :] = values
            self.offset += num_new

            return (
                self._joined_k[..., :self.offset, :],
                self._joined_v[..., :self.offset, :],
            )

        else:
            # --- PRE-COMPRESS PATH (standard KVCache behavior) ---
            if self.keys is None or (prev + num_new) > self.keys.shape[2]:
                B, n_kv_heads, _, k_head_dim = keys.shape
                v_head_dim = values.shape[3]
                n_steps = (self.step + num_new - 1) // self.step
                k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
                v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
                new_k = mx.zeros(k_shape, keys.dtype)
                new_v = mx.zeros(v_shape, values.dtype)
                if self.keys is not None:
                    if prev % self.step != 0:
                        self.keys = self.keys[..., :prev, :]
                        self.values = self.values[..., :prev, :]
                    self.keys = mx.concatenate([self.keys, new_k], axis=2)
                    self.values = mx.concatenate([self.values, new_v], axis=2)
                else:
                    self.keys, self.values = new_k, new_v

            self.offset += num_new
            self.keys[..., prev:self.offset, :] = keys
            self.values[..., prev:self.offset, :] = values

            # Auto-compress if threshold exceeded
            if self.compress_after > 0 and self.offset > self.compress_after:
                self.compress(self.compress_after)
                return self._get_full_cache()

            return (
                self.keys[..., :self.offset, :],
                self.values[..., :self.offset, :],
            )

    @property
    def _is_batched(self) -> bool:
        return self._idx is not None

    def _materialize_float_buffers(self):
        """Ensure keys/values hold the readable full float state.

        Batch operations slice and pad along the batch and sequence axes. The
        packed TurboQuant representation is per-cache, so filtering or merging
        first materializes the decoded float view and clears packed metadata.
        """
        full_k, full_v = self._get_full_cache()
        if full_k is None or full_v is None:
            return None, None
        self.keys = full_k
        self.values = full_v
        self._compressed_keys = None
        self._compressed_values = None
        self._compressed_tokens = 0
        self._decoded_k_buffer = None
        self._decoded_v_buffer = None
        self._joined_k = None
        self._joined_v = None
        return self.keys, self.values

    def _current_idx(self) -> int:
        if self._idx is not None:
            return int(self._idx)
        return int(self.offset)

    def _batch_state(self):
        keys, values = self._materialize_float_buffers()
        if keys is None or values is None:
            return None, None, 0, mx.array([], dtype=mx.int32), mx.array([], dtype=mx.int32)
        batch = int(keys.shape[0])
        idx = self._current_idx()
        if self._is_batched:
            offsets = self.offset
            left_padding = self.left_padding
        else:
            offsets = mx.array([int(self.offset)] * batch, dtype=mx.int32)
            left_padding = mx.array([0] * batch, dtype=mx.int32)
        return keys, values, idx, offsets, left_padding

    def _update_and_fetch_batched(self, keys: mx.array, values: mx.array):
        prev = int(self._idx)
        num_new = int(keys.shape[2])
        if self.keys is None or (prev + num_new) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + num_new - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset = self.offset + num_new
        self._idx += num_new
        self.keys[..., prev:self._idx, :] = keys
        self.values[..., prev:self._idx, :] = values
        return self.keys[..., :self._idx, :], self.values[..., :self._idx, :]

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        """BatchGenerator prepare() compatibility.

        `lengths` is accepted for API parity with ArraysCache and ignored for
        KV-style attention caches.
        """
        if left_padding is not None:
            if self.keys is not None or self._joined_k is not None:
                raise ValueError(
                    "Left padding can only be added to an empty TurboQuantKVCache"
                )
            lp = mx.array(left_padding)
            self.left_padding = lp
            self.offset = mx.array([0] * len(left_padding), dtype=mx.int32) - lp
            self._idx = 0
        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)

    def finalize(self):
        if self._right_padding is None:
            return
        keys, values = self._materialize_float_buffers()
        if keys is None or values is None:
            self._right_padding = None
            return
        from mlx_lm.models.cache import dynamic_roll

        padding = self._right_padding
        self.keys = dynamic_roll(keys, padding[:, None], axis=2)
        self.values = dynamic_roll(values, padding[:, None], axis=2)
        if not self._is_batched:
            self.offset = mx.array([int(self.offset)] * int(self.keys.shape[0]))
            self.left_padding = mx.array([0] * int(self.keys.shape[0]))
            self._idx = int(self.keys.shape[2])
        self.offset = self.offset - padding
        self.left_padding = self.left_padding + padding
        self._right_padding = None

    def filter(self, batch_indices):
        if isinstance(batch_indices, (list, tuple)):
            keep = list(batch_indices)
            keep_len = len(keep)
        else:
            keep = batch_indices
            try:
                keep_len = int(batch_indices.shape[0])
            except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
                keep_len = len(batch_indices)

        if keep_len == 0:
            self.keys = None
            self.values = None
            self._compressed_keys = None
            self._compressed_values = None
            self._compressed_tokens = 0
            self._decoded_k_buffer = None
            self._decoded_v_buffer = None
            self._joined_k = None
            self._joined_v = None
            self.offset = 0
            self.left_padding = None
            self._idx = None
            return

        keys, values, idx, offsets, left_padding = self._batch_state()
        if keys is None or values is None:
            self.offset = mx.array([0] * len(keep), dtype=mx.int32)
            self.left_padding = mx.array([0] * len(keep), dtype=mx.int32)
            self._idx = 0
            return

        self.keys = keys[keep]
        self.values = values[keep]
        self.offset = offsets[keep]
        self.left_padding = left_padding[keep]
        self._idx = idx

        min_left_pad = int(self.left_padding.min().item())
        if min_left_pad > 0:
            self.keys = self.keys[..., min_left_pad:, :]
            self.values = self.values[..., min_left_pad:, :]
            self._idx -= min_left_pad
            self.left_padding = self.left_padding - min_left_pad

        if keep_len == 1 and int(self.left_padding[0].item()) == 0:
            self.offset = int(self.offset[0].item())
            self.left_padding = None
            self._idx = None

    def extract(self, idx: int):
        keys, values, right_edge, offsets, left_padding = self._batch_state()
        extracted = TurboQuantKVCache(
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            key_bits=self.key_bits,
            value_bits=self.value_bits,
            seed=self._seed,
            compress_after=self.compress_after,
            sink_tokens=self.sink_tokens,
        )
        if keys is None or values is None:
            return extracted
        padding = int(left_padding[idx].item())
        extracted.keys = mx.contiguous(keys[idx:idx + 1, :, padding:right_edge, :])
        extracted.values = mx.contiguous(values[idx:idx + 1, :, padding:right_edge, :])
        extracted.offset = int(extracted.keys.shape[2])
        return extracted

    def extend(self, other):
        """In-place merge with another TurboQuant cache batch.

        Mirrors mlx-lm BatchKVCache.extend(): rows are right-justified to a
        common `_idx`, and per-row `offset`/`left_padding` carry the true
        sequence lengths for masks and later extract/filter operations.
        """
        self_k, self_v, self_idx, self_offsets, self_left = self._batch_state()
        other_k, other_v, other_idx, other_offsets, other_left = other._batch_state()

        if self_k is None and other_k is None:
            self.offset = mx.concatenate([self_offsets, other_offsets])
            self.left_padding = mx.concatenate([self_left, other_left])
            self._idx = max(self_idx, other_idx)
            return
        if self_k is None:
            self_k = mx.array([]).reshape(0, other_k.shape[1], 0, other_k.shape[3])
            self_v = mx.array([]).reshape(0, other_v.shape[1], 0, other_v.shape[3])
        if other_k is None:
            other_k = mx.array([]).reshape(0, self_k.shape[1], 0, self_k.shape[3])
            other_v = mx.array([]).reshape(0, self_v.shape[1], 0, self_v.shape[3])

        max_idx = max(int(self_idx), int(other_idx))
        max_size = max(int(self_k.shape[2]), int(other_k.shape[2]))

        def pad(keys, values, idx, offsets, left_padding):
            if keys.shape[0] == 0:
                return keys, values, offsets, left_padding
            left = max_idx - int(idx)
            right = max_size - int(keys.shape[2]) - left
            if right < 0:
                keys = keys[..., :right, :]
                values = values[..., :right, :]
                right = 0
            if left != 0 or right != 0:
                pad_width = [(0, 0), (0, 0), (left, right), (0, 0)]
                keys = mx.pad(keys, pad_width)
                values = mx.pad(values, pad_width)
            return keys, values, offsets, left_padding + left

        self_k, self_v, self_offsets, self_left = pad(
            self_k, self_v, self_idx, self_offsets, self_left
        )
        other_k, other_v, other_offsets, other_left = pad(
            other_k, other_v, other_idx, other_offsets, other_left
        )
        self.keys = mx.concatenate([self_k, other_k], axis=0)
        self.values = mx.concatenate([self_v, other_v], axis=0)
        self.offset = mx.concatenate([self_offsets, other_offsets])
        self.left_padding = mx.concatenate([self_left, other_left])
        self._idx = max_idx

    def compress(self, n_tokens: Optional[int] = None):
        """Compress oldest tokens. Decode ONCE into persistent read-only buffer.

        After compression:
          - Packed compressed storage (5x smaller, for serialization/eviction)
          - Decoded float buffer (read-only, for fast attention)
          - Float window for new tokens
          - Sink tokens (first N) preserved at full precision in the decoded buffer
        """
        if self.offset == 0 or self.keys is None:
            return
        self._ensure_encoders()

        n = n_tokens if n_tokens is not None else self.offset
        n = min(n, self.offset)

        # Preserve sink tokens — compress [sink..n), keep [0..sink) as float
        sink = min(self.sink_tokens, n)

        if n <= sink:
            return  # nothing to compress

        # Region to compress: [sink..n)
        k_region = self.keys[..., sink:n, :]
        v_region = self.values[..., sink:n, :]
        self._vmlx_tq_key_dtype = k_region.dtype
        self._vmlx_tq_value_dtype = v_region.dtype

        self._compressed_keys = encode_keys(k_region, self._key_encoder)
        self._compressed_values = encode_values(v_region, self._value_encoder)
        self._compressed_tokens = n  # total tokens accounted for (sink + compressed)

        # Decode compressed region ONCE into persistent buffer
        decoded_k = decode_keys(self._compressed_keys, self._key_encoder)
        decoded_v = decode_values(self._compressed_values, self._value_encoder)
        if decoded_k.dtype != self._vmlx_tq_key_dtype:
            decoded_k = decoded_k.astype(self._vmlx_tq_key_dtype)
        if decoded_v.dtype != self._vmlx_tq_value_dtype:
            decoded_v = decoded_v.astype(self._vmlx_tq_value_dtype)

        # Build decoded buffer: [sink_float, decoded_compressed]
        if sink > 0:
            sink_k = self.keys[..., :sink, :]
            sink_v = self.values[..., :sink, :]
            self._decoded_k_buffer = mx.concatenate([sink_k, decoded_k], axis=2)
            self._decoded_v_buffer = mx.concatenate([sink_v, decoded_v], axis=2)
        else:
            self._decoded_k_buffer = decoded_k
            self._decoded_v_buffer = decoded_v

        # Build joined buffer: [decoded | remaining window tokens | space]
        remaining = self.offset - n
        B, n_kv_heads = self._decoded_k_buffer.shape[:2]
        total = self.offset
        n_steps = (self.step + total - 1) // self.step
        self._joined_k = mx.zeros((B, n_kv_heads, n_steps * self.step, self.key_dim), self._decoded_k_buffer.dtype)
        self._joined_v = mx.zeros((B, n_kv_heads, n_steps * self.step, self.value_dim), self._decoded_v_buffer.dtype)
        # Copy decoded buffer into front [0..n)
        self._joined_k[..., :n, :] = self._decoded_k_buffer
        self._joined_v[..., :n, :] = self._decoded_v_buffer
        # Copy remaining float window [n..offset)
        if remaining > 0:
            self._joined_k[..., n:self.offset, :] = self.keys[..., n:self.offset, :]
            self._joined_v[..., n:self.offset, :] = self.values[..., n:self.offset, :]
        # Clear old separate buffers
        self.keys = None
        self.values = None

    def _get_full_cache(self) -> tuple[mx.array, mx.array]:
        """Return full K/V. Uses joined buffer if compressed, else float buffer."""
        end = self._current_idx()
        if self._joined_k is not None:
            return self._joined_k[..., :end, :], self._joined_v[..., :end, :]
        elif self.keys is not None:
            return self.keys[..., :end, :], self.values[..., :end, :]
        return None, None

    def make_mask(
        self,
        N: int,
        return_array: bool = True,
        window_size: Optional[int] = None,
        **kwargs,
    ):
        if self._is_batched:
            from mlx_lm.models.cache import create_causal_mask

            return create_causal_mask(
                N,
                offset=self._current_idx(),
                left_padding=self.left_padding,
                window_size=window_size,
                **kwargs,
            )
        from mlx_lm.models.cache import create_attention_mask
        return create_attention_mask(N, self.offset, return_array, window_size)

    def empty(self) -> bool:
        if self._is_batched:
            return bool(mx.all(self.offset == 0).item())
        return self.offset == 0

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self._current_idx(), n)
        if self._is_batched:
            self.offset = mx.maximum(self.offset - n, 0)
            self._idx = max(int(self._idx) - n, 0)
        else:
            self.offset -= n
        return n

    @property
    def state(self):
        if self.empty():
            return [], []
        return self._get_full_cache()

    @state.setter
    def state(self, v):
        if v is not None and len(v) == 2:
            self.keys, self.values = v
            if self.keys is not None:
                self._vmlx_tq_key_dtype = self.keys.dtype
                self._vmlx_tq_value_dtype = self.values.dtype
                self.offset = self.keys.shape[-2]
                self.left_padding = None
                self._right_padding = None
                self._idx = None
                self._compressed_keys = None
                self._compressed_values = None
                self._compressed_tokens = 0
                self._decoded_k_buffer = None
                self._decoded_v_buffer = None
                self._joined_k = None
                self._joined_v = None

    @property
    def meta_state(self):
        return str(self.offset), str(self.key_bits), str(self.value_bits)

    @meta_state.setter
    def meta_state(self, v):
        self.offset = int(v[0])
        self.key_bits = int(v[1])
        self.value_bits = int(v[2])

    def size(self):
        return self._current_idx()

    @property
    def nbytes(self):
        """Actual GPU memory: joined buffer OR float buffer + compressed packed."""
        total = 0
        if self._joined_k is not None:
            total += self._joined_k.nbytes + self._joined_v.nbytes
        elif self.keys is not None:
            total += self.keys.nbytes + self.values.nbytes
        if self._compressed_keys is not None:
            for arr in self._compressed_keys:
                if hasattr(arr, 'nbytes'):
                    total += arr.nbytes
            for arr in self._compressed_values:
                if hasattr(arr, 'nbytes'):
                    total += arr.nbytes
        return total

    @property
    def compressed_nbytes(self):
        """Memory of ONLY the packed compressed storage (not decoded buffer)."""
        total = 0
        if self._compressed_keys is not None:
            for arr in self._compressed_keys:
                if hasattr(arr, 'nbytes'):
                    total += arr.nbytes
            for arr in self._compressed_values:
                if hasattr(arr, 'nbytes'):
                    total += arr.nbytes
        return total

    @property
    def is_compressed(self):
        return self._compressed_tokens > 0
