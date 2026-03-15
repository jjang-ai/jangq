"""
MXQ Per-Block Quantization Engine
Created by Eric Jang (eric@vmlx.net)

Quantizes weight tensors block-by-block at variable bit widths.
Supports both round-to-nearest (RTN) and GPTQ-style optimal rounding.
"""

import numpy as np
from typing import NamedTuple

from .format.spec import (
    ALLOWED_BIT_WIDTHS,
    DEFAULT_BLOCK_SIZE,
    validate_bit_width,
    compute_block_offsets,
)
from .pack import pack_block


class QuantizedTensor(NamedTuple):
    """Result of quantizing a weight tensor."""

    qweight: np.ndarray      # packed quantized data (uint8)
    scales: np.ndarray        # per-block scale factors (float16)
    zeros: np.ndarray         # per-block zero points (float16)
    bit_map: np.ndarray       # per-block bit widths (uint8)
    block_offsets: np.ndarray  # byte offsets per block (uint32)
    shape: tuple              # original weight shape


def quantize_block_rtn(
    weights: np.ndarray,
    bits: int,
) -> tuple[np.ndarray, float, float]:
    """
    Quantize a block of weights using round-to-nearest (RTN).

    Asymmetric quantization:
        scale = (w_max - w_min) / (2^bits - 1)
        zero_point = round(-w_min / scale)
        quantized = clamp(round(w / scale + zero_point), 0, 2^bits - 1)
        dequantized = (quantized - zero_point) * scale

    Args:
        weights: float32 weight block (1D array, block_size elements)
        bits: bit width for this block

    Returns:
        (quantized_ints, scale, zero_point)
    """
    validate_bit_width(bits)
    n_levels = (1 << bits) - 1  # 2^bits - 1

    w_min = float(weights.min())
    w_max = float(weights.max())

    if w_max == w_min:
        # Constant block — all same value
        return np.zeros(len(weights), dtype=np.uint8), 1.0, 0.0

    # Compute scale and zero point
    scale = (w_max - w_min) / n_levels
    zero_point = round(-w_min / scale)
    zero_point = max(0, min(n_levels, zero_point))

    # Quantize
    quantized = np.round(weights / scale + zero_point).astype(np.int32)
    quantized = np.clip(quantized, 0, n_levels)

    return quantized.astype(np.uint8), scale, zero_point


def quantize_block_mse(
    weights: np.ndarray,
    bits: int,
    n_search: int = 20,
) -> tuple[np.ndarray, float, float]:
    """
    Quantize a block with MSE-optimal scale factor (grid search).

    Instead of using min/max directly, search over clipping ratios
    to find the scale that minimizes reconstruction error.

    Args:
        weights: float32 weight block
        bits: bit width for this block
        n_search: number of grid search points

    Returns:
        (quantized_ints, scale, zero_point)
    """
    validate_bit_width(bits)
    n_levels = (1 << bits) - 1

    w_min = float(weights.min())
    w_max = float(weights.max())

    if w_max == w_min:
        return np.zeros(len(weights), dtype=np.uint8), 1.0, 0.0

    best_mse = float("inf")
    best_result = None

    # Vectorized grid search over clipping ratios (0.8 to 1.0)
    ratios = np.linspace(0.8, 1.0, n_search)
    for clip_ratio in ratios:
        c_min = w_min * clip_ratio
        c_max = w_max * clip_ratio

        scale = (c_max - c_min) / n_levels
        if scale == 0:
            continue

        zero_point = round(-c_min / scale)
        zero_point = max(0, min(n_levels, zero_point))

        q = np.clip(np.round(weights / scale + zero_point), 0, n_levels)
        dq = (q - zero_point) * scale

        mse = float(np.mean((weights - dq) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_result = (q.astype(np.uint8), scale, float(zero_point))

    return best_result


def _quantize_blocks_vectorized(
    blocks: np.ndarray,
    bits: int,
    use_mse_optimal: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized quantization of many blocks at the same bit width.

    Args:
        blocks: (n_blocks, block_size) float32 array
        bits: bit width for ALL these blocks
        use_mse_optimal: if True, search for optimal clipping ratio

    Returns:
        (quantized_ints, scales, zeros) — all arrays over n_blocks
    """
    n_blocks, block_size = blocks.shape
    n_levels = (1 << bits) - 1

    # Per-block min/max
    w_min = blocks.min(axis=1)  # (n_blocks,)
    w_max = blocks.max(axis=1)  # (n_blocks,)

    # Handle constant blocks
    const_mask = (w_max == w_min)

    if use_mse_optimal and bits <= 4:
        # MSE-optimal clipping: search for best clipping ratio
        # This finds the clip range that minimizes reconstruction error,
        # not just the full min/max range. Crucial at low bit widths
        # where outliers waste quantization levels.
        best_q = np.zeros((n_blocks, block_size), dtype=np.uint8)
        best_scale = np.ones(n_blocks, dtype=np.float32)
        best_zero = np.zeros(n_blocks, dtype=np.float32)
        best_mse = np.full(n_blocks, np.inf, dtype=np.float64)

        for clip_ratio in np.linspace(0.7, 1.0, 15):
            c_min = w_min * clip_ratio
            c_max = w_max * clip_ratio

            c_range = c_max - c_min
            c_range[c_range == 0] = 1.0

            scale = c_range / n_levels
            zero = np.round(-c_min / scale)
            zero = np.clip(zero, 0, n_levels)

            q = np.round(blocks / scale[:, None] + zero[:, None])
            q = np.clip(q, 0, n_levels)
            dq = (q - zero[:, None]) * scale[:, None]

            mse = np.mean((blocks - dq) ** 2, axis=1)

            # Update where this ratio is better
            improved = mse < best_mse
            best_mse[improved] = mse[improved]
            best_q[improved] = q[improved].astype(np.uint8)
            best_scale[improved] = scale[improved]
            best_zero[improved] = zero[improved]

        best_scale[const_mask] = 1.0
        best_zero[const_mask] = 0.0
        best_q[const_mask] = 0

        return best_q, best_scale, best_zero
    else:
        # Standard min/max RTN (fast, used for 6+ bit)
        w_range = w_max - w_min
        w_range[const_mask] = 1.0

        scale = w_range / n_levels
        zero_point = np.round(-w_min / scale)
        zero_point = np.clip(zero_point, 0, n_levels)

        q = np.round(blocks / scale[:, None] + zero_point[:, None])
        q = np.clip(q, 0, n_levels).astype(np.uint8)

        scale[const_mask] = 1.0
        zero_point[const_mask] = 0.0
        q[const_mask] = 0

        return q, scale.astype(np.float32), zero_point.astype(np.float32)


def quantize_tensor(
    weights: np.ndarray,
    bit_allocation: np.ndarray,
    block_size: int = DEFAULT_BLOCK_SIZE,
    method: str = "mse",
) -> QuantizedTensor:
    """
    Quantize a full weight tensor with per-block variable bit widths.

    Uses vectorized quantization: groups all blocks by bit width,
    quantizes each group in one numpy operation. Fast even for
    millions of blocks.

    Args:
        weights: float32 weight tensor, shape (out_features, in_features)
        bit_allocation: uint8 array of bit widths per block
        block_size: weights per block
        method: "rtn" or "mse" (both use vectorized RTN for speed;
                MSE adds a clipping ratio search per bit-width group)

    Returns:
        QuantizedTensor with all companion arrays
    """
    original_shape = weights.shape
    flat_weights = weights.reshape(-1).astype(np.float32)
    n_weights = len(flat_weights)
    n_blocks = len(bit_allocation)

    # Verify block count matches
    expected_blocks = (n_weights + block_size - 1) // block_size
    if n_blocks != expected_blocks:
        raise ValueError(
            f"bit_allocation has {n_blocks} blocks but weight has "
            f"{n_weights} values ({expected_blocks} blocks at block_size={block_size})"
        )

    # Pad weights to multiple of block_size
    pad = n_blocks * block_size - n_weights
    if pad > 0:
        flat_weights = np.concatenate([flat_weights, np.zeros(pad, dtype=np.float32)])

    # Reshape into blocks: (n_blocks, block_size)
    all_blocks = flat_weights.reshape(n_blocks, block_size)
    bit_map = np.asarray(bit_allocation, dtype=np.uint8)

    # Pre-allocate output arrays
    all_q = np.zeros((n_blocks, block_size), dtype=np.uint8)
    scales = np.zeros(n_blocks, dtype=np.float32)
    zeros = np.zeros(n_blocks, dtype=np.float32)

    # Group blocks by bit width and quantize each group vectorized
    unique_bits = np.unique(bit_map)
    for bits in unique_bits:
        bits = int(bits)
        mask = (bit_map == bits)
        group_blocks = all_blocks[mask]  # (n_group, block_size)

        if len(group_blocks) == 0:
            continue

        q, s, z = _quantize_blocks_vectorized(group_blocks, bits,
                                                use_mse_optimal=(method == "mse"))
        all_q[mask] = q
        scales[mask] = s
        zeros[mask] = z

    # Pack blocks — vectorized per bit-width group
    # Pre-compute packed bytes per block for offset calculation
    from .format.spec import bytes_per_block as bpb_fn
    packed_sizes = np.array([bpb_fn(int(b), block_size) for b in bit_map], dtype=np.int64)
    total_packed = int(packed_sizes.sum())
    qweight = np.zeros(total_packed, dtype=np.uint8)

    # Compute offsets
    offsets = np.zeros(n_blocks, dtype=np.int64)
    offsets[1:] = np.cumsum(packed_sizes[:-1])

    # Pack each block (still per-block for packing, but quantization is vectorized)
    for i in range(n_blocks):
        bits = int(bit_map[i])
        packed = pack_block(all_q[i], bits)
        off = int(offsets[i])
        qweight[off:off + len(packed)] = packed
    block_offsets = np.array(
        compute_block_offsets(bit_map.tolist(), block_size), dtype=np.uint32
    )

    return QuantizedTensor(
        qweight=qweight,
        scales=scales.astype(np.float16),
        zeros=zeros.astype(np.float16),
        bit_map=bit_map,
        block_offsets=block_offsets,
        shape=original_shape,
    )


def dequantize_tensor(qt: QuantizedTensor, block_size: int = DEFAULT_BLOCK_SIZE) -> np.ndarray:
    """
    Dequantize an MXQ tensor back to float32.

    Used for validation and quality measurement.

    Args:
        qt: QuantizedTensor to dequantize
        block_size: weights per block

    Returns:
        float32 array of dequantized weights in original shape
    """
    from .pack import unpack_block

    n_blocks = len(qt.bit_map)
    all_weights = []

    for i in range(n_blocks):
        bits = int(qt.bit_map[i])
        offset = int(qt.block_offsets[i])
        block_bytes = (block_size * bits + 7) // 8
        packed = qt.qweight[offset : offset + block_bytes]

        q_ints = unpack_block(packed, bits, block_size)
        scale = float(qt.scales[i])
        zero = float(qt.zeros[i])

        dequantized = (q_ints.astype(np.float32) - zero) * scale
        all_weights.append(dequantized)

    flat = np.concatenate(all_weights)
    # Trim padding
    total_elements = 1
    for d in qt.shape:
        total_elements *= d
    flat = flat[:total_elements]

    return flat.reshape(qt.shape)
