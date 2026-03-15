"""
MXQ GPTQ — Hessian-Guided Optimal Quantization
Created by Eric Jang (eric@vmlx.net)

Implements the GPTQ algorithm (Frantar et al., 2023) for per-layer
weight quantization with error compensation. This is the key algorithm
that makes MXQ's mixed-precision quantization produce high-quality
results at low bit widths.

The algorithm:
1. Collect calibration activations to compute the Hessian H = X^T X
2. For each column of the weight matrix (in sensitivity order):
   a. Quantize the column to the assigned bit width
   b. Compute the quantization error
   c. Compensate remaining columns using the inverse Hessian
3. The compensation ensures that the total OUTPUT error is minimized,
   not just the per-weight error

This is what makes the difference between "garbage at 2-bit" and
"usable at 2-bit" — the error compensation propagates corrections
so that the overall matrix-vector product is preserved.

References:
- GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained
  Transformers (Frantar et al., 2023)
- Optimal Brain Surgeon (Hassibi & Stork, 1993) — the theoretical
  foundation for the compensation formula
"""

import numpy as np
from typing import Optional

from .pack import pack_block
from .format.spec import DEFAULT_BLOCK_SIZE, bytes_per_block


def gptq_quantize_layer(
    weight: np.ndarray,
    hessian: np.ndarray,
    bit_allocation: np.ndarray,
    block_size: int = DEFAULT_BLOCK_SIZE,
    damping: float = 0.01,
    act_order: bool = True,
    block_column_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Quantize a weight matrix using GPTQ with per-block variable bit widths.

    Args:
        weight: (out_features, in_features) float32 weight matrix
        hessian: (in_features, in_features) float32 Hessian matrix (X^T X)
        bit_allocation: (n_blocks,) uint8 — bits per block from allocator
        block_size: weights per quantization block
        damping: regularization added to Hessian diagonal for stability
        act_order: if True, quantize columns in order of decreasing
                   Hessian diagonal (most sensitive first)
        block_column_size: process columns in blocks for efficiency

    Returns:
        qweight: packed quantized data (uint8 array)
        scales: per-block scale factors (float16)
        zeros: per-block zero points (float16)
        bit_map: per-block bit widths (uint8)
        block_offsets: byte offsets per block (uint32)
        metrics: dict with quantization quality metrics
    """
    out_features, in_features = weight.shape
    n_blocks = len(bit_allocation)
    blocks_per_row = in_features // block_size

    assert n_blocks == out_features * blocks_per_row, \
        f"Block count mismatch: {n_blocks} != {out_features} * {blocks_per_row}"

    W = weight.copy().astype(np.float64)  # work in float64 for precision

    # Add damping to Hessian for numerical stability
    H = hessian.astype(np.float64)
    damp = damping * np.mean(np.diag(H))
    H += damp * np.eye(in_features, dtype=np.float64)

    # Compute inverse Hessian via Cholesky decomposition
    try:
        L = np.linalg.cholesky(H)
        H_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(in_features)))
    except np.linalg.LinAlgError:
        # Fall back to direct inverse with more damping
        H_inv = np.linalg.inv(H + 0.1 * np.eye(in_features))

    # Determine column ordering
    if act_order:
        # Quantize most sensitive columns first (largest H diagonal)
        perm = np.argsort(np.diag(H))[::-1]  # descending sensitivity
        inv_perm = np.argsort(perm)  # to undo the permutation
        W = W[:, perm]
        H_inv = H_inv[np.ix_(perm, perm)]
    else:
        perm = np.arange(in_features)
        inv_perm = perm

    # Build the bit width map for each column position
    # bit_allocation is per-block (row-major). We need to know the bit width
    # for each (row, col) position in the weight matrix.
    col_bits = np.zeros(in_features, dtype=np.int32)
    for col in range(in_features):
        # The block index for row 0, column col (representative)
        block_in_row = col // block_size
        # All rows use the same bit width for this column position
        # (this is an approximation — true MXQ allows different bits per row)
        # Use the first row's block to determine bits
        block_idx = 0 * blocks_per_row + block_in_row
        col_bits[col] = int(bit_allocation[block_idx])

    # Quantized output
    Q = np.zeros_like(W)

    # Process in column blocks for efficiency
    total_error = 0.0

    for col_start in range(0, in_features, block_column_size):
        col_end = min(col_start + block_column_size, in_features)

        # Get the diagonal block of H_inv for this column range
        H_diag = np.diag(H_inv)[col_start:col_end]

        for j in range(col_start, col_end):
            bits = col_bits[perm[j]] if act_order else col_bits[j]
            n_levels = (1 << bits) - 1

            # Current column values
            w_col = W[:, j]

            # Quantize this column
            w_min = w_col.min()
            w_max = w_col.max()
            if w_max == w_min:
                Q[:, j] = 0
                continue

            scale = (w_max - w_min) / n_levels
            zero = round(-w_min / scale)
            zero = max(0, min(n_levels, zero))

            q_vals = np.clip(np.round(w_col / scale + zero), 0, n_levels)
            dq_vals = (q_vals - zero) * scale
            Q[:, j] = dq_vals

            # Error
            delta = w_col - dq_vals
            total_error += float(np.sum(delta ** 2))

            # Compensate remaining columns in this block
            h_jj = H_inv[j, j]
            if h_jj > 1e-15 and j + 1 < col_end:
                comp_row = H_inv[j, j+1:col_end] / h_jj
                W[:, j+1:col_end] -= np.outer(delta, comp_row)

    # Undo column permutation
    if act_order:
        Q = Q[:, inv_perm]

    # Now pack the quantized weights into MXQ format (per-block)
    Q_flat = Q.reshape(-1).astype(np.float32)
    packed_blocks = []
    scales_out = np.zeros(n_blocks, dtype=np.float32)
    zeros_out = np.zeros(n_blocks, dtype=np.float32)
    bit_map = np.asarray(bit_allocation, dtype=np.uint8)

    for i in range(n_blocks):
        bits = int(bit_map[i])
        n_levels = (1 << bits) - 1
        start = i * block_size
        end = start + block_size
        block_vals = Q_flat[start:end]

        # Re-quantize the GPTQ-optimized values to integer form for packing
        b_min = block_vals.min()
        b_max = block_vals.max()
        if b_max == b_min:
            q_ints = np.zeros(block_size, dtype=np.uint8)
            scale_val = 1.0
            zero_val = 0.0
        else:
            scale_val = (b_max - b_min) / n_levels
            zero_val = round(-b_min / scale_val)
            zero_val = max(0, min(n_levels, zero_val))
            q_ints = np.clip(np.round(block_vals / scale_val + zero_val), 0, n_levels).astype(np.uint8)

        packed = pack_block(q_ints, bits)
        packed_blocks.append(packed)
        scales_out[i] = scale_val
        zeros_out[i] = zero_val

    qweight = np.concatenate(packed_blocks)

    # Compute block offsets
    offsets = []
    current = 0
    for i in range(n_blocks):
        offsets.append(current)
        current += bytes_per_block(int(bit_map[i]), block_size)
    block_offsets = np.array(offsets, dtype=np.uint32)

    metrics = {
        "total_weight_error": total_error,
        "mean_weight_error": total_error / (out_features * in_features),
        "act_order": act_order,
        "damping": damping,
    }

    return (
        qweight,
        scales_out.astype(np.float16),
        zeros_out.astype(np.float16),
        bit_map,
        block_offsets,
        metrics,
    )


def compute_hessian(
    activations: list[np.ndarray],
    in_features: int,
) -> np.ndarray:
    """
    Compute the Hessian matrix H = (1/n) Σ X_i^T X_i from calibration activations.

    Args:
        activations: list of (seq_len, in_features) activation arrays
        in_features: input dimension

    Returns:
        (in_features, in_features) Hessian matrix
    """
    H = np.zeros((in_features, in_features), dtype=np.float64)
    n_samples = 0

    for act in activations:
        if act.ndim == 1:
            act = act.reshape(1, -1)
        # Flatten batch and sequence dims
        X = act.reshape(-1, in_features).astype(np.float64)
        H += X.T @ X
        n_samples += X.shape[0]

    if n_samples > 0:
        H /= n_samples

    return H
