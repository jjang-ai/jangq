"""Qwen4-Exp PLE n-gram hashing — exact port of the HF reference.

All integer math must match torch int64 two's-complement semantics:
multiply wraps, XOR is bitwise, remainder takes the divisor's sign
(positive here). numpy int64 gives identical wraparound (C semantics).

The hash is pure token-id arithmetic, so the same code serves the runtime
lookup AND the offline row-frequency histogram (no forward pass needed).
"""

import math

import numpy as np

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def build_layer_multipliers(unigram_vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int) -> np.ndarray:
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    multipliers = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
    # values fit in int64 after the % half_bound; store as int64 for wrapping math
    return np.array(multipliers, dtype=np.uint64).astype(np.int64)


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


class NGramHasher:
    """Maps token histories to rows of the concatenated n-gram embedding."""

    def __init__(
        self,
        vocab_size: int,
        eos_token_id: int,
        ngram_size: int = 3,
        heads_per_ngram: int = 8,
        ngram_vocab_size_base: int = 20_000_000,
        make_divisible_by: int = 128,
        seed: int = 1234,
        ple_layer_index: int = 0,
    ):
        self.ngram_size = ngram_size
        self.context_len = ngram_size - 1
        self.heads_per_ngram = heads_per_ngram
        self.ngram_heads = (ngram_size - 1) * heads_per_ngram
        self.eos_token_id = eos_token_id

        self.head_vocab_sizes = []
        self.head_offsets = []
        total = 0
        for head_idx in range(self.ngram_heads):
            global_head_idx = ple_layer_index * self.ngram_heads + head_idx
            size = _find_nth_prime_after(ngram_vocab_size_base - 1, global_head_idx + 1)
            self.head_vocab_sizes.append(size)
            self.head_offsets.append(total)
            total += size
        self.total_vocab_size = total
        self.padded_vocab_size = math.ceil(total / make_divisible_by) * make_divisible_by

        self.layer_multipliers = build_layer_multipliers(vocab_size, ngram_size, ple_layer_index, seed)
        self._head_sizes_np = np.array(self.head_vocab_sizes, dtype=np.int64)
        self._head_offsets_np = np.array(self.head_offsets, dtype=np.int64)

    def _shift_right_ignore_eos(self, token_ids: np.ndarray, shift: int) -> np.ndarray:
        """token_ids: [B, S] int64. Port of the HF reference (EOS-segmented)."""
        if shift == 0:
            return token_ids
        batch_size, seq_len = token_ids.shape
        positions = np.arange(seq_len, dtype=np.int64)
        eos_positions = np.where(token_ids == self.eos_token_id, positions[None, :], -1)
        previous_eos_inclusive = np.maximum.accumulate(eos_positions, axis=1)
        previous_eos = np.concatenate(
            [np.full((batch_size, 1), -1, dtype=np.int64), previous_eos_inclusive[:, :-1]], axis=1
        )
        segment_start = previous_eos + 1
        position_in_segment = positions[None, :] - segment_start
        source_positions = positions - shift
        gather_positions = np.broadcast_to(np.clip(source_positions, 0, None)[None, :], token_ids.shape)
        shifted = np.take_along_axis(token_ids, gather_positions, axis=1)
        valid = (position_in_segment >= shift) & (source_positions[None, :] >= 0)
        return np.where(valid, shifted, np.int64(self.eos_token_id))

    def hash_history(self, token_history: np.ndarray) -> np.ndarray:
        """token_history: [B, C+S] int64 (prev context + current ids).
        Returns row ids [B, C+S, ngram_heads] into the concatenated table."""
        token_history = token_history.astype(np.int64)
        shifted = [self._shift_right_ignore_eos(token_history, s) for s in range(self.ngram_size)]
        blocks = []
        with np.errstate(over="ignore"):
            for ngram in range(2, self.ngram_size + 1):
                start = (ngram - 2) * self.heads_per_ngram
                end = start + self.heads_per_ngram
                mixed = shifted[0] * self.layer_multipliers[0]
                for pos in range(1, ngram):
                    mixed = np.bitwise_xor(mixed, shifted[pos] * self.layer_multipliers[pos])
                ids = np.remainder(mixed[..., None], self._head_sizes_np[start:end][None, None, :])
                blocks.append(ids + self._head_offsets_np[start:end][None, None, :])
        return np.concatenate(blocks, axis=-1)

    def hash_tokens(self, input_ids: np.ndarray, prev_context: np.ndarray | None = None) -> np.ndarray:
        """input_ids: [B, S]. prev_context: [B, context_len] or None (EOS fill).
        Returns row ids [B, S, ngram_heads]."""
        input_ids = np.asarray(input_ids, dtype=np.int64)
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        if prev_context is None:
            prev_context = np.full(
                (input_ids.shape[0], self.context_len), self.eos_token_id, dtype=np.int64
            )
        history = np.concatenate([prev_context.astype(np.int64), input_ids], axis=-1)
        return self.hash_history(history)[:, -input_ids.shape[1]:]
