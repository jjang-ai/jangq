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
    if s_rows != min_scale[0]:
        raise ValueError(
            f"scale_inv rows {s_rows} != ceil(rows/{brow}) {min_scale[0]} for weight "
            f"{tuple(weight.shape)}: this tensor was quantized per tensor-parallel rank; "
            "use dequant_fp8_rank_blocked()"
        )
    scale_full = (
        scale_inv.float()
        .repeat_interleave(brow, dim=0)
        .repeat_interleave(bcol, dim=1)
    )
    out = weight.float() * scale_full[:rows, :cols]
    return out if out_dtype is None else out.to(out_dtype)


def dequant_fp8_rank_blocked(
    weight: Any,
    scale_inv: Any,
    *,
    tp_size: int,
    block_size: tuple[int, int] = (128, 128),
    out_dtype: Any | None = None,
):
    """Dequantize a fused tensor whose rows were FP8-quantized PER TP RANK.

    MiMo stores fused ``qkv_proj`` as ``tp_size`` row blocks (one per rank) and
    quantizes each block separately, so every rank's rows start a fresh
    128-row scale block. On full-attention layers a rank block is 3392 rows
    (26.5 blocks), giving 4 x 27 = 108 scale rows for 13568 weight rows.
    Applying the scales on the global row grid (as the V2.5 converter did)
    mis-scales ranks 1..3. Measured 2026-09-22 (golden perplexity test).
    """
    import torch

    rows, cols = weight.shape
    if rows % tp_size:
        raise ValueError(f"rows {rows} not divisible by tp_size {tp_size}")
    rank_rows = rows // tp_size
    per_rank_scale_rows = (rank_rows + block_size[0] - 1) // block_size[0]
    if scale_inv.shape[0] != tp_size * per_rank_scale_rows:
        raise ValueError(
            f"scale rows {scale_inv.shape[0]} != tp_size*ceil(rank_rows/{block_size[0]}) "
            f"= {tp_size * per_rank_scale_rows} for weight {tuple(weight.shape)}"
        )
    parts = []
    for r in range(tp_size):
        w = weight[r * rank_rows:(r + 1) * rank_rows]
        s = scale_inv[r * per_rank_scale_rows:(r + 1) * per_rank_scale_rows]
        parts.append(dequant_fp8_e4m3_scale_inv(w, s, block_size=block_size, out_dtype=torch.float32))
    out = torch.cat(parts, dim=0)
    return out if out_dtype is None else out.to(out_dtype)
