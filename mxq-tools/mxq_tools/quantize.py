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
    n_search: int = 100,
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

    # Search over clipping ratios (0.8 to 1.0 of the full range)
    for clip_ratio in np.linspace(0.8, 1.0, n_search):
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


def quantize_tensor(
    weights: np.ndarray,
    bit_allocation: np.ndarray,
    block_size: int = DEFAULT_BLOCK_SIZE,
    method: str = "mse",
) -> QuantizedTensor:
    """
    Quantize a full weight tensor with per-block variable bit widths.

    Args:
        weights: float32 weight tensor, shape (out_features, in_features)
        bit_allocation: uint8 array of bit widths per block
        block_size: weights per block
        method: "rtn" for round-to-nearest, "mse" for MSE-optimal

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

    quantize_fn = quantize_block_mse if method == "mse" else quantize_block_rtn

    # Quantize each block
    packed_blocks = []
    scales = np.empty(n_blocks, dtype=np.float32)
    zeros = np.empty(n_blocks, dtype=np.float32)
    bit_map = np.asarray(bit_allocation, dtype=np.uint8)

    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size
        block_weights = flat_weights[start:end]
        bits = int(bit_map[i])

        q_ints, scale, zero = quantize_fn(block_weights, bits)
        packed = pack_block(q_ints, bits)

        packed_blocks.append(packed)
        scales[i] = scale
        zeros[i] = zero

    # Concatenate all packed blocks
    qweight = np.concatenate(packed_blocks)
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
