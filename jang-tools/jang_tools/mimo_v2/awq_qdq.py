"""Source-side AWQ-style affine QDQ helpers for MiMo diagnostics."""

from __future__ import annotations

from collections.abc import Callable

import torch


def awq_channel_scale(
    act_max: torch.Tensor,
    weight: torch.Tensor,
    *,
    alpha: float = 0.5,
    eps: float = 1e-6,
    floor: float = 1.0,
) -> torch.Tensor:
    """Return per-input-channel AWQ scale for a linear weight.

    The scale balances observed activation magnitude against per-input-column
    weight magnitude. It is normalized to avoid changing the overall layer
    scale too aggressively; callers preserve the exact linear contract by
    applying ``x / scale`` with a quantized ``weight * scale``.
    """
    if weight.ndim != 2:
        raise ValueError(f"expected rank-2 weight, got shape={tuple(weight.shape)}")
    if act_max.ndim != 1 or act_max.shape[0] != weight.shape[1]:
        raise ValueError(
            f"act_max shape {tuple(act_max.shape)} does not match weight input dim {weight.shape[1]}"
        )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    a = act_max.float().clamp_min(eps)
    w = weight.float().abs().amax(dim=0).clamp_min(eps)
    scale = (a.pow(alpha) / w.pow(1.0 - alpha)).clamp_min(floor)
    finite = torch.isfinite(scale)
    if not bool(finite.all()):
        scale = torch.where(finite, scale, torch.ones_like(scale))
    # Keep the geometric center around one so this remains a salience reshape,
    # not an unbounded global gain change.
    center = torch.sqrt(scale.max().clamp_min(eps) * scale.min().clamp_min(eps))
    return (scale / center).clamp_min(eps).contiguous()


def quant_dequant_minmax_affine(weight: torch.Tensor, *, bits: int, group_size: int) -> torch.Tensor:
    """CPU QDQ that mirrors the converter's min/max MLX affine sidecars."""
    if bits == 0:
        return weight.float()
    if bits not in {2, 3, 4, 5, 6, 8}:
        raise ValueError(f"unsupported affine bits={bits}")
    if group_size not in {32, 64, 128}:
        raise ValueError(f"unsupported group_size={group_size}")
    x = weight.float()
    if x.ndim != 2:
        raise ValueError(f"expected rank-2 weight, got shape={tuple(x.shape)}")
    rows, cols = x.shape
    if cols % group_size != 0:
        raise ValueError(f"cols={cols} not divisible by group_size={group_size}")
    xr = x.reshape(rows, cols // group_size, group_size)
    minv = xr.amin(dim=2, keepdim=True)
    maxv = xr.amax(dim=2, keepdim=True)
    levels = (1 << bits) - 1
    scale = ((maxv - minv) / float(levels)).clamp_min(1e-7)
    q = torch.round((xr - minv) / scale).clamp_(0, levels)
    mlx_scale = (-scale).to(torch.bfloat16).to(torch.float32)
    mlx_bias = maxv.to(torch.bfloat16).to(torch.float32)
    return ((levels - q) * mlx_scale + mlx_bias).reshape_as(x)


def quant_dequant_awq_weight(
    weight: torch.Tensor,
    *,
    input_scale: torch.Tensor,
    bits: int,
    group_size: int,
) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    """Quantize ``weight * input_scale`` and return its matching input transform."""
    if input_scale.ndim != 1 or input_scale.shape[0] != weight.shape[1]:
        raise ValueError(
            f"input_scale shape {tuple(input_scale.shape)} does not match weight input dim {weight.shape[1]}"
        )
    scale = input_scale.float().reshape(1, -1)
    q_weight = quant_dequant_minmax_affine(weight.float() * scale, bits=bits, group_size=group_size)

    def transform_input(x: torch.Tensor) -> torch.Tensor:
        return x.float() / input_scale.to(device=x.device, dtype=torch.float32)

    return q_weight, transform_input
