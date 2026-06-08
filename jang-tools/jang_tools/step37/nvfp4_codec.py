"""ModelOpt NVFP4 tensor decode helpers for Step 3.7 Flash.

ModelOpt stores NVFP4 weights as:
  - ``<base>.weight``: uint8 packed FP4 E2M1 values, last dim = in_dim / 2
  - ``<base>.weight_scale``: float8_e4m3fn scales, one per 16 input values
  - ``<base>.weight_scale_2``: float32 global/per-expert second-level scale

The activation ``input_scale`` tensors are not needed to reconstruct weights.
"""

from __future__ import annotations

import torch


FP4_E2M1_TABLE = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def dequant_nvfp4_modelopt(
    weight_u8: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    *,
    block_size: int = 16,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize a ModelOpt NVFP4 packed tensor.

    Supports rank-2 and rank-3 tensors. The final axis is the packed input
    dimension. ``weight_scale`` must match the unpacked input dimension divided
    by ``block_size``. ``weight_scale_2`` may be scalar, per-expert, or already
    broadcastable to the decoded tensor.
    """
    if weight_u8.dtype != torch.uint8:
        raise TypeError(f"expected uint8 weight, got {weight_u8.dtype}")
    if weight_u8.ndim not in (2, 3):
        raise ValueError(f"expected rank-2 or rank-3 weight, got {weight_u8.ndim}D")

    in_dim = int(weight_u8.shape[-1]) * 2
    if in_dim % block_size != 0:
        raise ValueError(f"unpacked in_dim {in_dim} is not divisible by block_size={block_size}")
    expected_scale_shape = tuple(weight_u8.shape[:-1]) + (in_dim // block_size,)
    if tuple(weight_scale.shape) != expected_scale_shape:
        raise ValueError(
            f"scale shape mismatch: got {tuple(weight_scale.shape)}, "
            f"expected {expected_scale_shape}"
        )

    table = FP4_E2M1_TABLE.to(weight_u8.device)
    low = (weight_u8 & 0x0F).long()
    high = ((weight_u8 >> 4) & 0x0F).long()
    vals = torch.stack([table[low], table[high]], dim=-1).flatten(-2)

    scale = weight_scale.float().repeat_interleave(block_size, dim=-1)
    scale2 = weight_scale_2.float()
    while scale2.ndim < vals.ndim:
        scale2 = scale2.unsqueeze(-1)

    return (vals * scale * scale2).to(out_dtype)
