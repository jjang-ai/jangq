"""Shared SSM/Mamba layout helpers for JANG converters and loaders.

These helpers intentionally handle only small state tensors whose semantics are
not Linear matmul weights. They must stay idempotent because old artifacts are
sanitized at load time while new artifacts may already be written in MLX layout.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


_STATE_TENSOR_LEAVES = {"A_log", "D", "dt_bias"}


def forced_passthrough_bits(tensor_name: str) -> int | None:
    """Return the storage bit width for state tensors that must not quantize."""
    leaf = tensor_name.rsplit(".", 1)[-1]
    if tensor_name.endswith("conv1d.weight") or tensor_name.endswith("conv1d.bias"):
        return 16
    # Inkling short convolutions: `{attn,mlp,k,v}_sconv.weight`, shape (C, 1, 4).
    # Same grouped-Conv1d semantics as Mamba's conv1d but a different leaf name,
    # so the rule above misses them. Last dim is 4, which no MLX group size
    # divides — they must never reach mx.quantize. Tiny (1.7M params total).
    if tensor_name.endswith("_sconv.weight") or tensor_name.endswith("_sconv.bias"):
        return 16
    # Inkling relative-position bias bank, shape (d_rel=16, rel_extent). 2-D, so
    # neither the ndim==1 nor the tiny-ndim>2 passthrough gate catches it, and it
    # would be quantized. With no RoPE in Inkling this tensor IS the position
    # encoding — quantizing it destroys positional attention.
    if tensor_name.endswith("rel_logits_proj.proj"):
        return 16
    if leaf in _STATE_TENSOR_LEAVES:
        return 16
    return None


def prepare_mlx_passthrough_tensor(tensor_name: str, tensor: np.ndarray) -> np.ndarray:
    """Prepare a passthrough state tensor for MLX bundle storage.

    HF grouped Conv1d weights are `(out_channels, 1, kernel)`. MLX Conv1d
    expects `(out_channels, kernel, 1)`. Bias/state vectors do not need a shape
    transform, but still stay fp16 passthrough.
    """
    out = tensor
    if (
        tensor_name.endswith("conv1d.weight")
        or tensor_name.endswith("_sconv.weight")
    ) and getattr(out, "ndim", None) == 3 and out.shape[-1] != 1:
        # (C, 1, K) -> (C, K, 1). MLX conv1d is NLC with weight
        # (out_channels, kernel, in_channels/groups); torch grouped Conv1d is
        # (out_channels, in_channels/groups, kernel). Idempotent: already-MLX
        # tensors have shape[-1] == 1 and are left alone.
        out = np.transpose(out, (0, 2, 1))
    if getattr(out, "dtype", None) != np.float16:
        out = out.astype(np.float16)
    return out


def sanitize_grouped_conv1d_layout(
    weights: dict,
    transpose_3d: Callable[[object], object],
) -> dict:
    """Return weights with leftover HF Conv1d tensors converted to MLX layout."""
    fixed = None
    for key, value in weights.items():
        if (
            "conv1d.weight" in key
            and getattr(value, "ndim", None) == 3
            and value.shape[-1] != 1
        ):
            if fixed is None:
                fixed = dict(weights)
            fixed[key] = transpose_3d(value)
    return weights if fixed is None else fixed
