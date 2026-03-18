"""
JANG FP8 Support — Dequantize FP8 E4M3 weights to float32.
Created by Jinho Jang (eric@jangq.ai)

Handles FP8 prequantized models (MiniMax-M2.5, DeepSeek-V3) by reading
raw uint8 bytes and applying E4M3 → float32 conversion with block scales.
"""

import json
import struct
from pathlib import Path

import numpy as np


def fp8_e4m3_to_float32(x: np.ndarray) -> np.ndarray:
    """Convert FP8 E4M3 (uint8) to float32.

    E4M3 format: 1 sign, 4 exponent (bias=7), 3 mantissa.
    """
    x = x.astype(np.uint32)
    sign = (x >> 7) & 1
    exp = (x >> 3) & 0xF
    mant = x & 0x7

    result = np.zeros_like(x, dtype=np.float32)

    # Normal numbers: (-1)^sign * 2^(exp-7) * (1 + mant/8)
    normal = exp > 0
    result[normal] = (
        ((-1.0) ** sign[normal])
        * (2.0 ** (exp[normal].astype(np.float32) - 7))
        * (1.0 + mant[normal].astype(np.float32) / 8.0)
    )

    # Subnormal numbers: (-1)^sign * 2^(-6) * (mant/8)
    subnorm = (exp == 0) & (mant > 0)
    result[subnorm] = (
        ((-1.0) ** sign[subnorm])
        * (2.0 ** -6)
        * (mant[subnorm].astype(np.float32) / 8.0)
    )

    return result


def load_fp8_tensor(
    sf_path: Path | str,
    tensor_name: str,
    shape: list[int],
    scale_inv: np.ndarray | None = None,
) -> np.ndarray:
    """Load an FP8 E4M3 tensor and dequantize to float32.

    Args:
        sf_path: path to safetensors file
        tensor_name: tensor name in safetensors
        shape: tensor shape
        scale_inv: block-wise inverse scale [rows/B, cols/B] where B=128.
                   If None, returns raw FP8→float32 without scaling.

    Returns:
        float32 array of dequantized weights
    """
    with open(sf_path, "rb") as fh:
        header_size = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_size))
        data_offset = 8 + header_size

        info = header[tensor_name]
        fh.seek(data_offset + info["data_offsets"][0])
        raw = np.frombuffer(
            fh.read(info["data_offsets"][1] - info["data_offsets"][0]),
            dtype=np.uint8,
        )

    fp8_vals = raw.reshape(shape)
    result = fp8_e4m3_to_float32(fp8_vals)

    if scale_inv is not None and len(shape) == 2:
        # Block scaling: scale_inv is [rows/B, cols/B], each covers BxB block
        sh, sw = scale_inv.shape
        bh = shape[0] // sh
        bw = shape[1] // sw
        scale_full = np.repeat(np.repeat(scale_inv, bh, axis=0), bw, axis=1)
        result *= scale_full

    return result


def is_fp8_model(sf_path: Path | str) -> bool:
    """Check if a safetensors file contains FP8 weights."""
    with open(sf_path, "rb") as fh:
        header_size = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_size))

    for info in header.values():
        if isinstance(info, dict) and info.get("dtype") == "F8_E4M3":
            return True
    return False
