"""Quantized linear layers used by all JANG runtimes.

JANGLinear: standard mx.quantize affine — wraps the codebook + scales as
shipped by `mx.quantize(..., bits=..., group_size=...)`. Identical to the
mlx_lm built-in QuantizedLinear; we duplicate here so distributed code can
import without mlx_lm dependency.

JANGTQLinear: TurboQuant — uint8 packed indices + bf16 codebook + optional
Hadamard rotation flag. Delegates to `jang_tools.turboquant.linear` for
the actual matmul kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class JANGLinear(nn.Module):
    """Affine-quantized linear (mx.quantize)."""

    def __init__(self, in_features: int, out_features: int,
                 bits: int = 4, group_size: int = 64, bias: bool = False):
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.in_features = in_features
        self.out_features = out_features
        # placeholders; filled by loader
        self.weight = mx.zeros((out_features, in_features // (32 // bits)),
                               dtype=mx.uint32)
        self.scales = mx.zeros((out_features, in_features // group_size),
                               dtype=mx.bfloat16)
        self.biases = mx.zeros((out_features, in_features // group_size),
                               dtype=mx.bfloat16)
        self.bias = mx.zeros((out_features,)) if bias else None

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.quantized_matmul(x, self.weight, self.scales, self.biases,
                                transpose=True, group_size=self.group_size,
                                bits=self.bits)
        return y if self.bias is None else (y + self.bias)


class JANGTQLinear(nn.Module):
    """TurboQuant linear; thin shim over jang_tools.turboquant.linear."""

    def __init__(self, in_features: int, out_features: int,
                 bits: int, group_size: int, *, hadamard: bool = False,
                 bias: bool = False):
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.in_features = in_features
        self.out_features = out_features
        self.hadamard = hadamard
        # placeholders; populated by loader
        self.indices = mx.zeros((out_features, in_features // group_size),
                                dtype=mx.uint8)
        self.codebook = mx.zeros((1 << bits, group_size), dtype=mx.bfloat16)
        self.bias = mx.zeros((out_features,)) if bias else None

    def __call__(self, x: mx.array) -> mx.array:
        from ..turboquant.linear import tq_matmul  # type: ignore
        y = tq_matmul(x, self.indices, self.codebook,
                      group_size=self.group_size, hadamard=self.hadamard)
        return y if self.bias is None else (y + self.bias)
