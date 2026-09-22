"""MiMo-V2.6 routed-expert MXFP4 decoding.

MiMo-V2.6 ships routed experts as OCP MXFP4: ``weight`` is uint8 with two
e2m1 codes per byte, ``weight_scale`` is uint8 e8m0 with one exponent per
32-element block along the input dimension.

Conventions below were MEASURED against the parent MiMo-V2.5 FP8 weights
(docs/runtime/mimo-v26-flash-2026-09-22/scripts/probe_v26_vs_v25.py):
low nibble = even element (cos 0.988 vs 0.006 swapped), e8m0 bias 127
(norm ratio 1.02).

The same bytes, viewed as little-endian uint32, are bit-exact MLX
``mode="mxfp4"`` packed weights with ``group_size=32``. Re-quantizing the
decoded values with ``mx.quantize`` does NOT reproduce them (MLX picks
different block exponents on ~36% of bytes), so native passthrough must copy
the raw bytes via :func:`mxfp4_raw_to_mlx`.
"""

from __future__ import annotations

import numpy as np
import torch

MXFP4_BLOCK = 32
_FP4_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _check(packed: torch.Tensor, scale: torch.Tensor) -> None:
    if packed.dtype != torch.uint8 or scale.dtype != torch.uint8:
        raise TypeError(f"expected uint8 packed/scale, got {packed.dtype}/{scale.dtype}")
    if packed.ndim != 2 or scale.ndim != 2 or packed.shape[0] != scale.shape[0]:
        raise ValueError(f"shape mismatch packed={tuple(packed.shape)} scale={tuple(scale.shape)}")
    if packed.shape[1] * 2 != scale.shape[1] * MXFP4_BLOCK:
        raise ValueError(
            f"packed cols {packed.shape[1]}*2 != scale cols {scale.shape[1]}*{MXFP4_BLOCK}"
        )


def dequant_mxfp4(packed: torch.Tensor, scale: torch.Tensor, *, out_dtype=torch.float32) -> torch.Tensor:
    """Decode MiMo MXFP4 (uint8 codes + uint8 e8m0) to a dense [out, in] tensor."""
    _check(packed, scale)
    p = packed.to(torch.int64)
    vals = torch.stack((_FP4_E2M1[p & 15], _FP4_E2M1[p >> 4]), dim=-1).reshape(packed.shape[0], -1)
    s = torch.exp2(scale.to(torch.float32) - 127.0).repeat_interleave(MXFP4_BLOCK, dim=1)
    return (vals * s).to(out_dtype)


def mxfp4_raw_to_mlx(packed: torch.Tensor, scale: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return (uint32 weight, uint8 scales) for MLX mode='mxfp4', group_size=32. Bit-exact."""
    _check(packed, scale)
    w = np.ascontiguousarray(packed.numpy()).view(np.uint32)
    return w, np.ascontiguousarray(scale.numpy())
