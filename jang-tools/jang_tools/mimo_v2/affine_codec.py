"""CPU min/max affine packing for MLX quantized modules."""

from __future__ import annotations

import torch


def _pack_codes_lsb(codes: torch.Tensor, *, bits: int, group_size: int) -> torch.Tensor:
    words_per_group = (group_size * bits) // 32
    flat = codes.to(torch.int64).reshape(-1, group_size)
    packed = torch.zeros((flat.shape[0], words_per_group), dtype=torch.int64)
    mask = (1 << bits) - 1
    for offset in range(group_size):
        bit_offset = offset * bits
        word = bit_offset // 32
        shift = bit_offset % 32
        vals = (flat[:, offset] & mask) << shift
        packed[:, word] |= vals
        spill = shift + bits - 32
        if spill > 0:
            packed[:, word + 1] |= (flat[:, offset] & mask) >> (bits - spill)
    return packed.to(torch.uint32)


def quantize_minmax_affine(
    weight: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    sidecar_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight to MLX-compatible affine sidecars.

    The grid is the standard CPU min/max grid used by the MiMo probes. MLX
    affine dequantization computes ``q * scale + bias``, so equivalent min/max
    reconstruction is represented as ``scale=(min-max)/levels`` and
    ``bias=max`` with inverted integer codes.
    """
    if bits not in {2, 3, 4, 5, 6, 8}:
        raise ValueError(f"unsupported affine bits={bits}")
    if group_size not in {32, 64, 128}:
        raise ValueError(f"unsupported affine group_size={group_size}")
    if weight.ndim != 2:
        raise ValueError(f"expected rank-2 weight, got shape={tuple(weight.shape)}")
    rows, cols = weight.shape
    if cols % group_size != 0:
        raise ValueError(f"cols={cols} is not divisible by group_size={group_size}")

    x = weight.float().reshape(rows, cols // group_size, group_size)
    minv = x.amin(dim=2, keepdim=True)
    maxv = x.amax(dim=2, keepdim=True)
    levels = (1 << bits) - 1
    positive_scale = ((maxv - minv) / float(levels)).clamp_min(1e-7)
    minmax_codes = torch.round((x - minv) / positive_scale).clamp_(0, levels).to(torch.int64)
    mlx_codes = levels - minmax_codes

    qweight = _pack_codes_lsb(mlx_codes, bits=bits, group_size=group_size)
    qweight = qweight.reshape(rows, (cols // group_size) * ((group_size * bits) // 32))
    scales = (-positive_scale.squeeze(-1)).to(sidecar_dtype).contiguous()
    biases = maxv.squeeze(-1).to(sidecar_dtype).contiguous()
    return qweight.contiguous(), scales, biases
