"""Parity test: jang_tools NGramHasher vs the HF qwen4_exp reference (torch).

The torch side below is copied VERBATIM (minus module plumbing) from
transformers-main modular_qwen4_exp.py so any divergence is ours.
"""

import numpy as np
import torch

from jang_tools.qwen4_exp.ngram import NGramHasher, build_layer_multipliers

VOCAB = 248320
EOS = 248044
SEED = 1234
NGRAM_SIZE = 3
HEADS = 8
BASE = 20_000_000


# ---- verbatim HF reference -------------------------------------------------
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


def _build_layer_multipliers_ref(unigram_vocab_size, ngram_size, ple_layer_index, seed):
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    multipliers = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
    return torch.tensor(multipliers, dtype=torch.long)


def _shift_right_ignore_eos_ref(token_ids, shift, eos_token_id):
    if shift == 0:
        return token_ids
    batch_size, seq_len = token_ids.shape
    positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
    eos_positions = torch.where(token_ids == eos_token_id, positions, -1)
    previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
    previous_eos = torch.cat([eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]], dim=1)
    segment_start = previous_eos + 1
    position_in_segment = positions.unsqueeze(0) - segment_start
    source_positions = positions - shift
    gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
    shifted = token_ids.gather(dim=1, index=gather_positions)
    valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
    return torch.where(valid, shifted, token_ids.new_full((), eos_token_id))


def hash_ref(input_ids, prev_context, hasher):
    """Verbatim port of Qwen4ExpTextNGramEmbedding.forward id computation."""
    multipliers = _build_layer_multipliers_ref(VOCAB, NGRAM_SIZE, 0, SEED)
    head_sizes = torch.tensor(hasher.head_vocab_sizes, dtype=torch.long)
    head_offsets = torch.tensor(hasher.head_offsets, dtype=torch.long)
    token_history = torch.cat([prev_context, input_ids], dim=-1)
    shifted_tokens = [_shift_right_ignore_eos_ref(token_history, s, EOS) for s in range(NGRAM_SIZE)]
    blocks = []
    for ngram in range(2, NGRAM_SIZE + 1):
        start_idx = (ngram - 2) * HEADS
        end_idx = start_idx + HEADS
        mixed_ids = shifted_tokens[0] * multipliers[0]
        for position in range(1, ngram):
            mixed_ids = torch.bitwise_xor(mixed_ids, shifted_tokens[position] * multipliers[position])
        ngram_ids = torch.remainder(mixed_ids.unsqueeze(-1), head_sizes[start_idx:end_idx].view(1, 1, -1))
        blocks.append(ngram_ids + head_offsets[start_idx:end_idx].view(1, 1, -1))
    return torch.cat(blocks, dim=-1)[:, -input_ids.shape[1]:]


# ---- tests -----------------------------------------------------------------
def main():
    rng = np.random.default_rng(0)
    hasher = NGramHasher(VOCAB, EOS, NGRAM_SIZE, HEADS, BASE, 128, SEED, 0)

    ours = build_layer_multipliers(VOCAB, NGRAM_SIZE, 0, SEED)
    ref = _build_layer_multipliers_ref(VOCAB, NGRAM_SIZE, 0, SEED).numpy()
    assert (ours == ref).all(), f"multipliers differ: {ours} vs {ref}"
    print(f"multipliers OK: {ours}")
    print(f"total_vocab={hasher.total_vocab_size} padded={hasher.padded_vocab_size} "
          f"head_sizes[:3]={hasher.head_vocab_sizes[:3]}")

    for trial in range(20):
        b, s = int(rng.integers(1, 4)), int(rng.integers(1, 64))
        ids = rng.integers(0, VOCAB, size=(b, s))
        # sprinkle EOS tokens to exercise segmentation
        eos_mask = rng.random((b, s)) < 0.15
        ids = np.where(eos_mask, EOS, ids)
        prev = rng.integers(0, VOCAB, size=(b, hasher.context_len))
        if trial % 3 == 0:
            prev[:] = EOS  # fresh-sequence case

        got = hasher.hash_tokens(ids, prev)
        want = hash_ref(torch.tensor(ids), torch.tensor(prev), hasher).numpy()
        assert (got == want).all(), f"trial {trial}: mismatch {np.argwhere(got != want)[:5]}"

    # chunked == full (cache semantics): hash(full) == [hash(c1, fresh), hash(c2, tail(c1))]
    for trial in range(10):
        s1, s2 = int(rng.integers(3, 40)), int(rng.integers(1, 40))
        ids = rng.integers(0, VOCAB, size=(1, s1 + s2))
        ids = np.where(rng.random((1, s1 + s2)) < 0.1, EOS, ids)
        full = hasher.hash_tokens(ids, None)
        c1 = hasher.hash_tokens(ids[:, :s1], None)
        c2 = hasher.hash_tokens(ids[:, s1:], ids[:, s1 - hasher.context_len: s1])
        chunked = np.concatenate([c1, c2], axis=1)
        assert (full == chunked).all(), f"chunk trial {trial} mismatch"

    # non-aligned single-token decode path
    ids = rng.integers(0, VOCAB, size=(1, 37))
    full = hasher.hash_tokens(ids, None)
    step = hasher.hash_tokens(ids[:, :1], None)
    outs = [step]
    for t in range(1, 37):
        prev = ids[:, max(0, t - hasher.context_len): t]
        if prev.shape[1] < hasher.context_len:
            prev = np.concatenate(
                [np.full((1, hasher.context_len - prev.shape[1]), EOS, dtype=np.int64), prev], axis=1
            )
        outs.append(hasher.hash_tokens(ids[:, t: t + 1], prev))
    assert (np.concatenate(outs, axis=1) == full).all(), "decode-path mismatch"

    print("ALL NGRAM PARITY TESTS PASSED")


if __name__ == "__main__":
    main()
