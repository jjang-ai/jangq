"""MiMo-V2 FP8 E4M3 block dequantization.

MiMo stores most text weights as torch float8_e4m3fn tensors with fp32
``*_weight_scale_inv`` companions. Each scale covers a 128 x 128 block and is
multiplied into the decoded FP8 value.
"""

from __future__ import annotations

from typing import Any


def dequant_fp8_e4m3_scale_inv(
    weight: Any,
    scale_inv: Any,
    *,
    block_size: tuple[int, int] = (128, 128),
    out_dtype: Any | None = None,
):
    """Return ``weight.float() * expanded(scale_inv)`` for MiMo FP8 tensors.

    The implementation is torch-first because safetensors exposes MiMo's
    ``F8_E4M3`` tensors through the PyTorch framework without losing dtype
    information. It intentionally accepts partial edge blocks by trimming the
    expanded scale tensor back to the weight shape.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch.
        raise RuntimeError("torch is required to decode MiMo FP8 tensors") from exc

    if not isinstance(weight, torch.Tensor) or not isinstance(scale_inv, torch.Tensor):
        raise TypeError("weight and scale_inv must be torch.Tensor instances")
    if weight.ndim != 2:
        raise ValueError(f"expected a 2-D FP8 matrix, got shape={tuple(weight.shape)}")
    if scale_inv.ndim != 2:
        raise ValueError(f"expected a 2-D scale matrix, got shape={tuple(scale_inv.shape)}")
    if weight.dtype != torch.float8_e4m3fn:
        raise TypeError(f"expected torch.float8_e4m3fn weight, got {weight.dtype}")

    rows, cols = weight.shape
    brow, bcol = block_size
    min_scale = ((rows + brow - 1) // brow, (cols + bcol - 1) // bcol)
    s_rows, s_cols = scale_inv.shape
    if s_rows < min_scale[0] or s_cols < min_scale[1]:
        raise ValueError(
            f"scale_inv shape {tuple(scale_inv.shape)} is smaller than "
            f"ceil(weight/block_size) {min_scale} for weight {tuple(weight.shape)}"
        )
    # MiMo full-attention qkv_proj has trailing scale-row padding beyond the
    # weight rows (e.g. weight (13568,4096), scale (108,32) when ceil=106).
    # The extra scale rows have no corresponding weight rows, so the
    # repeat_interleave + [:rows,:cols] slice trims them correctly.
    scale_full = (
        scale_inv.float()
        .repeat_interleave(brow, dim=0)
        .repeat_interleave(bcol, dim=1)
    )
    out = weight.float() * scale_full[:rows, :cols]
    return out if out_dtype is None else out.to(out_dtype)


def dequant_fp8_e4m3_scale_inv_segments(
    weight: Any,
    scale_inv: Any,
    *,
    row_segments: list[tuple[int, int]],
    block_size: tuple[int, int] = (128, 128),
    out_dtype: Any | None = None,
):
    """Dequant a fused 2-D weight whose scale rows include segment padding.

    ``row_segments`` is a list of ``(weight_rows, scale_rows)`` pairs. This is
    needed for MiMo full-attention fused ``qkv_proj`` tensors: Q/K/V are stored
    as one matrix, but the FP8 scale rows are padded at the projection segment
    boundary, not just at the end of the fused matrix.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required to decode MiMo FP8 tensors") from exc

    outs = []
    w_pos = 0
    s_pos = 0
    for w_rows, s_rows in row_segments:
        w_part = weight[w_pos : w_pos + w_rows]
        s_part = scale_inv[s_pos : s_pos + s_rows]
        outs.append(
            dequant_fp8_e4m3_scale_inv(
                w_part,
                s_part,
                block_size=block_size,
                out_dtype=None,
            )
        )
        w_pos += w_rows
        s_pos += s_rows

    if w_pos != weight.shape[0]:
        raise ValueError(f"row segments consumed {w_pos} weight rows, expected {weight.shape[0]}")
    if s_pos != scale_inv.shape[0]:
        raise ValueError(f"row segments consumed {s_pos} scale rows, expected {scale_inv.shape[0]}")

    out = torch.cat(outs, dim=0)
    return out if out_dtype is None else out.to(out_dtype)
