"""FP4 e2m1fn dequantization (DeepSeek-V4 expert weights).

Input format per DSV4-Flash:
  weight: int8, shape (out_dim, in_dim // 2)  —  2 FP4 nibbles per byte
          low nibble = packed_byte & 0x0F       (even column)
          high nibble = (packed_byte >> 4) & 0x0F  (odd column)
  scale:  float8_e8m0fnu (UE8M0), shape (out_dim, in_dim // 32)
          each scale covers 32 consecutive FP4 values along the input dim.

FP4_TABLE (from inference/convert.py):
  nibble 0..7  → [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]   positive
  nibble 8..15 → [0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]  negative

Scale decode (UE8M0 = unsigned 8-bit exponent, no mantissa):
  scale_float = 2 ** ue8m0_byte     (when byte != 0xFF; 0xFF is NaN/reserved)

This module works on torch tensors (since source safetensors are torch).
"""

from __future__ import annotations

import torch

FP4_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


def _ue8m0_to_fp32(s: torch.Tensor) -> torch.Tensor:
    """UE8M0 byte → fp32 via 2**exp. Input is torch.float8_e8m0fnu."""
    return s.float()  # torch's float8_e8m0fnu cast already interprets as 2^exp


def dequant_fp4_blockwise(
    w_int8: torch.Tensor,
    scale_ue8m0: torch.Tensor,
    *,
    fp4_block: int = 32,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize a packed-FP4 int8 matrix to bf16 (or fp32) via UE8M0 scales.

    Arguments
    ---------
    w_int8 : int8 tensor, shape (out_dim, packed_in // 2)
    scale_ue8m0 : float8_e8m0fnu tensor, shape (out_dim, in_dim // fp4_block)
    fp4_block : inner-dim block size (default 32 per DSV4 spec)
    out_dtype : output precision (bfloat16 for JANG downstream, fp32 for verify)

    Returns
    -------
    tensor of shape (out_dim, in_dim) in out_dtype.
    """
    assert w_int8.dtype == torch.int8, f"expected int8, got {w_int8.dtype}"
    assert w_int8.ndim == 2, f"expected 2D, got {w_int8.ndim}D"
    out_dim, packed_in = w_int8.shape
    in_dim = packed_in * 2
    assert in_dim % fp4_block == 0, \
        f"in_dim {in_dim} not a multiple of fp4_block {fp4_block}"
    assert scale_ue8m0.shape == (out_dim, in_dim // fp4_block), \
        f"scale shape mismatch: {scale_ue8m0.shape} vs expected " \
        f"({out_dim}, {in_dim // fp4_block})"

    device = w_int8.device
    table = FP4_TABLE.to(device)

    # Unpack nibbles
    w_u8 = w_int8.view(torch.uint8)
    low = (w_u8 & 0x0F).long()
    high = ((w_u8 >> 4) & 0x0F).long()
    # Interleave low, high along dim 1 to reconstruct in_dim
    vals = torch.stack([table[low], table[high]], dim=-1).flatten(1)  # (out_dim, in_dim) // fp4_block)

    # Apply block-wise scale (UE8M0 → fp32, one scale per fp4_block chunk)
    scale_fp32 = _ue8m0_to_fp32(scale_ue8m0)  # (out_dim, in_dim // fp4_block)
    # Expand each scale entry to cover fp4_block columns
    scale_expanded = scale_fp32.repeat_interleave(fp4_block, dim=1)  # (out_dim, in_dim)
    out = vals * scale_expanded

    return out.to(out_dtype)


def detect_fp4(w: torch.Tensor) -> bool:
    """Heuristic: int8 weight tensor == FP4 packed (vs plain int8 which
    DSV4 doesn't use for weights)."""
    return w.dtype == torch.int8
