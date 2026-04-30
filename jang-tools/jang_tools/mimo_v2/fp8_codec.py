"""FP8 E4M3 with FP32 [128,128] block scales — MiMo-V2.5-Pro source format.

Per upstream config:
    quantization_config = {
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        "activation_scheme": "dynamic",
    }

Per-layer storage (HF safetensors keys, all FP32 scales — NOT UE8M0):
    *.weight        : float8_e4m3fn, (out_dim, in_dim)
    *.weight_scale_inv : float32, (out_dim // 128, in_dim // 128)

`o_proj` keys appearing in `quantization_config.ignored_layers` are bf16 floats
with no scale tensor — pass through.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import torch  # only used for fp8 decode; safetensors mmaps already bypass GPU


def fp8_e4m3_to_fp32(w_fp8_bytes: np.ndarray) -> np.ndarray:
    """Decode raw fp8_e4m3fn bytes (uint8 view) into fp32 numpy.

    Uses torch's native fp8 cast (CPU) for correctness; numpy lacks fp8.
    """
    t = torch.from_numpy(w_fp8_bytes.view(np.uint8)).view(torch.float8_e4m3fn)
    return t.float().numpy()


def dequant_fp8_block(
    w_fp8_u8: np.ndarray,
    scale_fp32: np.ndarray,
    *,
    block: tuple[int, int] = (128, 128),
    out_dtype: mx.Dtype = mx.bfloat16,
) -> mx.array:
    """Dequantize one FP8 weight tensor to bf16 mx.array.

    Arguments
    ---------
    w_fp8_u8 : ndarray of dtype uint8 (raw fp8 bits), shape (out_dim, in_dim)
    scale_fp32 : ndarray of dtype float32, shape (out_dim // 128, in_dim // 128)
    """
    assert w_fp8_u8.dtype == np.uint8 and w_fp8_u8.ndim == 2
    out_dim, in_dim = w_fp8_u8.shape
    b0, b1 = block
    assert out_dim % b0 == 0 and in_dim % b1 == 0, (
        f"weight shape {w_fp8_u8.shape} not divisible by block {block}"
    )
    expected = (out_dim // b0, in_dim // b1)
    assert scale_fp32.shape == expected, (
        f"scale shape {scale_fp32.shape} != {expected}"
    )

    w_fp32 = fp8_e4m3_to_fp32(w_fp8_u8)            # (O, I) fp32
    scale_full = np.repeat(np.repeat(scale_fp32, b0, axis=0), b1, axis=1)
    out_fp32 = w_fp32 * scale_full
    return mx.array(out_fp32).astype(out_dtype)


def is_ignored(weight_key: str, ignored_layers: list[str]) -> bool:
    """Return True if `weight_key` (e.g. `model.layers.0.self_attn.o_proj.weight`)
    matches one of the upstream `ignored_layers` patterns."""
    base = weight_key.rstrip(".weight").rstrip(".")
    return base in ignored_layers
