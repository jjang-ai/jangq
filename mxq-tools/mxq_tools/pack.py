"""
MXQ Bit Packing — Pack and unpack variable-width integers into byte arrays.
Created by Eric Jang (eric@vmlx.net)

This is the core bit manipulation engine. Every quantized weight in MXQ
passes through these functions.
"""

import numpy as np


def pack_bits(values: np.ndarray, bits: int) -> np.ndarray:
    """
    Pack an array of unsigned integers into a compact byte array at the given bit width.

    Args:
        values: uint8 or uint16 array of quantized values (each in range [0, 2^bits - 1])
        bits: bit width per value (2, 3, 4, 5, 6, or 8)

    Returns:
        uint8 array of packed bytes (LSB-first packing)

    Example (3-bit, 8 values):
        values = [5, 3, 7, 1, 0, 6, 2, 4]
        Bit stream: 101_011_111_001_000_110_010_100
        Packed: [0xEF, 0x91, 0x94]
    """
    values = np.asarray(values, dtype=np.uint64)
    n = len(values)
    total_bits = n * bits
    total_bytes = (total_bits + 7) // 8

    if bits == 8:
        return values.astype(np.uint8)

    if bits == 4:
        # Fast path: 2 values per byte
        if n % 2 != 0:
            values = np.append(values, np.uint64(0))
        packed = (values[0::2] | (values[1::2] << 4)).astype(np.uint8)
        return packed[:total_bytes]

    if bits == 2:
        # Fast path: 4 values per byte
        pad = (4 - n % 4) % 4
        if pad:
            values = np.append(values, np.zeros(pad, dtype=np.uint64))
        packed = (
            values[0::4]
            | (values[1::4] << 2)
            | (values[2::4] << 4)
            | (values[3::4] << 6)
        ).astype(np.uint8)
        return packed[:total_bytes]

    # General case: pack bits contiguously across byte boundaries
    result = np.zeros(total_bytes, dtype=np.uint8)
    for i, val in enumerate(values):
        bit_offset = i * bits
        byte_idx = bit_offset // 8
        bit_shift = bit_offset % 8

        # Write value across 1-2 bytes
        wide = int(val) << bit_shift
        result[byte_idx] |= wide & 0xFF
        if byte_idx + 1 < total_bytes:
            result[byte_idx + 1] |= (wide >> 8) & 0xFF
        if byte_idx + 2 < total_bytes:
            result[byte_idx + 2] |= (wide >> 16) & 0xFF

    return result


def unpack_bits(packed: np.ndarray, bits: int, count: int) -> np.ndarray:
    """
    Unpack a byte array into an array of unsigned integers at the given bit width.

    Args:
        packed: uint8 array of packed bytes
        bits: bit width per value (2, 3, 4, 5, 6, or 8)
        count: number of values to unpack

    Returns:
        uint8 or uint16 array of unpacked values
    """
    packed = np.asarray(packed, dtype=np.uint8)

    if bits == 8:
        return packed[:count].copy()

    if bits == 4:
        # Fast path
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        interleaved = np.empty(len(packed) * 2, dtype=np.uint8)
        interleaved[0::2] = low
        interleaved[1::2] = high
        return interleaved[:count]

    if bits == 2:
        # Fast path
        b0 = packed & 0x03
        b1 = (packed >> 2) & 0x03
        b2 = (packed >> 4) & 0x03
        b3 = (packed >> 6) & 0x03
        interleaved = np.empty(len(packed) * 4, dtype=np.uint8)
        interleaved[0::4] = b0
        interleaved[1::4] = b1
        interleaved[2::4] = b2
        interleaved[3::4] = b3
        return interleaved[:count]

    # General case
    mask = (1 << bits) - 1
    result = np.empty(count, dtype=np.uint8 if bits <= 8 else np.uint16)

    for i in range(count):
        bit_offset = i * bits
        byte_idx = bit_offset // 8
        bit_shift = bit_offset % 8

        # Read uint16 to handle cross-byte boundary
        if byte_idx + 1 < len(packed):
            raw = int(packed[byte_idx]) | (int(packed[byte_idx + 1]) << 8)
        else:
            raw = int(packed[byte_idx])

        result[i] = (raw >> bit_shift) & mask

    return result


def pack_block(weights_quantized: np.ndarray, bits: int) -> np.ndarray:
    """
    Pack a single block of quantized weights.

    Args:
        weights_quantized: quantized weight values for one block
        bits: bit width for this block

    Returns:
        packed byte array
    """
    max_val = (1 << bits) - 1
    clipped = np.clip(weights_quantized, 0, max_val).astype(np.uint64)
    return pack_bits(clipped, bits)


def unpack_block(packed: np.ndarray, bits: int, block_size: int) -> np.ndarray:
    """
    Unpack a single block of quantized weights.

    Args:
        packed: packed byte array for one block
        bits: bit width for this block
        block_size: number of weights in the block

    Returns:
        array of quantized integer values
    """
    return unpack_bits(packed, bits, block_size)
