"""Mistral 3.5 FP8 dequant: per-tensor scale (NOT [128,128] block).

Source format:
    *.weight       : float8_e4m3fn  (out, in)
    *.weight_scale : float32        (1,)  -- ONE scalar per tensor

Implication: dequant is `w_fp8 * scale_scalar`, much cheaper than the
DSV4/MiMo block path. Vision tower + multi_modal_projector + lm_head are
already bf16 in source (per modules_to_not_convert), so this path only
runs for text decoder linears.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import torch


def fp8_e4m3_to_fp32(w_u8: np.ndarray) -> np.ndarray:
    return torch.from_numpy(w_u8.view(np.uint8)).view(torch.float8_e4m3fn).float().numpy()


def dequant_fp8_per_tensor(w_u8: np.ndarray, scale: np.ndarray,
                           out_dtype: mx.Dtype = mx.bfloat16) -> mx.array:
    assert w_u8.dtype == np.uint8
    s = float(scale.reshape(-1)[0])
    out = fp8_e4m3_to_fp32(w_u8) * s
    return mx.array(out).astype(out_dtype)
