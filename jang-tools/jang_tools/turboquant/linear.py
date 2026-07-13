"""
TurboQuantLinear — Mixed-precision TQ weight quantization for MLX.
Created by Jinho Jang (eric@jangq.ai)

MXTQ format: rotation + Lloyd-Max codebook per layer, different bits per tier.
Drop-in replacement for nn.QuantizedLinear / QuantizedSwitchLinear.

Storage per weight: packed uint32 indices + 1 float16 norm per row
Dequant: unpack → codebook[indices] → scale by norm → inverse rotate → matmul
"""

import math
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Optional

from .codebook import compute_codebook
from .rotation import generate_random_signs, hadamard_rotate, hadamard_inverse
from .pipeline import pack_bits, unpack_bits


class TurboQuantLinear(nn.Module):
    """Linear layer with MXTQ weight quantization."""

    def __init__(self, in_features: int, out_features: int, bits: int = 2,
                 bias: bool = False, seed: int = 42):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self._seed = seed

        cb = compute_codebook(in_features, bits)
        self.codebook = mx.array(cb, dtype=mx.float32)
        self.signs = mx.array(generate_random_signs(in_features, seed=seed))

        vals_per_u32 = 32 // bits
        packed_cols = (in_features + vals_per_u32 - 1) // vals_per_u32
        self.packed = mx.zeros((out_features, packed_cols), dtype=mx.uint32)
        self.norms = mx.zeros((out_features,), dtype=mx.float16)

        if bias:
            self.bias = mx.zeros((out_features,))
        self.freeze()

    def __call__(self, x: mx.array) -> mx.array:
        # Unpack per-row (handles non-power-of-2 bit widths like 3-bit)
        rows = []
        for r in range(self.out_features):
            row_idx = unpack_bits(self.packed[r], self.bits, self.in_features)
            rows.append(row_idx)
        idx = mx.stack(rows)
        # Codebook lookup + norm + inverse rotate
        w = mx.take(self.codebook, idx.astype(mx.uint32))
        w = w * self.norms[:, None].astype(w.dtype)
        w = hadamard_inverse(w, self.signs)
        y = x @ w.T
        if "bias" in self:
            y = y + self.bias
        return y


class TurboQuantSwitchLinear(nn.Module):
    """MoE expert switch with MXTQ weights. Dequants only active experts."""

    def __init__(self, in_features: int, out_features: int, num_experts: int,
                 bits: int = 2, bias: bool = False, seed: int = 42):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.bits = bits

        cb = compute_codebook(in_features, bits)
        self.codebook = mx.array(cb, dtype=mx.float32)
        self.signs = mx.array(generate_random_signs(in_features, seed=seed))

        vals_per_u32 = 32 // bits
        packed_cols = (in_features + vals_per_u32 - 1) // vals_per_u32
        self.packed = mx.zeros((num_experts, out_features, packed_cols), dtype=mx.uint32)
        self.norms = mx.zeros((num_experts, out_features), dtype=mx.float16)

        if bias:
            self.bias = mx.zeros((num_experts, out_features))
        self.freeze()

    def _dequant_experts(self, expert_indices) -> mx.array:
        """Dequant selected experts. Returns (n_selected, out, in) float."""
        results = []
        for e in expert_indices:
            packed_e = self.packed[e]
            n_el = self.out_features * self.in_features
            idx = unpack_bits(packed_e.reshape(-1), self.bits, n_el)
            idx = idx.reshape(self.out_features, self.in_features)
            w = mx.take(self.codebook, idx.astype(mx.uint32))
            w = w * self.norms[e][:, None].astype(w.dtype)
            w = hadamard_inverse(w, self.signs)
            results.append(w)
        return mx.stack(results)

    def __call__(self, x: mx.array, indices: mx.array, sorted_indices=False) -> mx.array:
        """x: (B, S, in), indices: (B, S, K) → (B, S, K, out)"""
        B, S, K = indices.shape[0], indices.shape[1], indices.shape[-1]

        # Dequant all needed experts
        unique = mx.unique(indices.reshape(-1))
        mx.eval(unique)
        expert_list = unique.tolist()
        expert_weights = {int(e): self._dequant_experts([int(e)])[0] for e in expert_list}

        # Compute output per expert assignment
        out = mx.zeros((B, S, K, self.out_features), dtype=x.dtype)
        for k in range(K):
            for e_idx, w in expert_weights.items():
                mask = (indices[:, :, k] == e_idx)
                if mx.any(mask):
                    r = x @ w.T
                    out_k = out[:, :, k, :]
                    out = out.at[:, :, k, :].add(
                        mx.where(mask[:, :, None], r, mx.zeros_like(r))
                    )

        if "bias" in self:
            for k in range(K):
                for e_idx in expert_list:
                    mask = (indices[:, :, k] == int(e_idx))
                    if mx.any(mask):
                        out = out.at[:, :, k, :].add(
                            mx.where(mask[:, :, None], self.bias[int(e_idx)], mx.zeros_like(self.bias[0]))
                        )
        return out


# ── Conversion utilities ──────────────────────────────────────────

def tq_quantize_weight(weight: np.ndarray, bits: int = 2, seed: int = 42) -> dict:
    """TQ-quantize a single weight matrix.

    Returns dict ready to save as safetensors.
    """
    out_feat, in_feat = weight.shape
    w = mx.array(weight.astype(np.float32))

    signs = mx.array(generate_random_signs(in_feat, seed=seed))
    w_rot = hadamard_rotate(w, signs)

    norms = mx.sqrt(mx.sum(w_rot * w_rot, axis=1, keepdims=True))
    norms_safe = mx.maximum(norms, mx.array(1e-10))
    w_normed = w_rot / norms_safe

    cb = mx.array(compute_codebook(in_feat, bits))
    # Quantize
    boundaries = (cb[:-1] + cb[1:]) / 2.0
    indices = mx.zeros(w_normed.shape, dtype=mx.uint8)
    for b in boundaries:
        indices = indices + (w_normed > b).astype(mx.uint8)

    # Vectorized flat packing is valid only when each source row ends on a
    # uint32 pack boundary.  3-bit JANGTQ stores 10 values per uint32, and
    # Qwen/Zaya-style widths such as 2048 leave an 8-value tail.  In that
    # case flattening would pack the tail of row N together with the head of
    # row N+1, so use row-wise padding instead.
    vals_per_u32 = 32 // bits
    if in_feat % vals_per_u32 == 0:
        packed = pack_bits(indices.reshape(-1), bits).reshape(out_feat, -1)
    else:
        pad = (vals_per_u32 - (in_feat % vals_per_u32)) % vals_per_u32
        if pad:
            indices = mx.concatenate(
                [indices, mx.zeros((out_feat, pad), dtype=mx.uint8)],
                axis=1,
            )
        rows = indices.reshape(out_feat, -1, vals_per_u32).astype(mx.uint32)
        packed = mx.zeros((out_feat, rows.shape[1]), dtype=mx.uint32)
        for i in range(vals_per_u32):
            packed = packed | (rows[:, :, i] << (i * bits))

    mx.eval(packed, norms)

    return {
        "packed": np.array(packed),
        "norms": np.array(norms.squeeze(-1).astype(mx.float16)),
    }


def tq_quantize_experts(weights: np.ndarray, bits: int = 2, seed: int = 42) -> dict:
    """TQ-quantize stacked expert weights (n_experts, out, in).

    Returns dict with stacked packed/norms.
    """
    n_experts, out_feat, in_feat = weights.shape
    all_packed = []
    all_norms = []

    for e in range(n_experts):
        result = tq_quantize_weight(weights[e], bits=bits, seed=seed)
        all_packed.append(result["packed"])
        all_norms.append(result["norms"])

    return {
        "packed": np.stack(all_packed),
        "norms": np.stack(all_norms),
    }
