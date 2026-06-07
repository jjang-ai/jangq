"""MiMo-V2 runtime loading helpers.

This module keeps post-load runtime tweaks available to serving code, not just
benchmark scripts. The default loader is conservative; optional knobs must be
requested explicitly.
"""

from __future__ import annotations

import time
from typing import Any


def quantize_lm_head(
    model: Any,
    *,
    bits: int | None = None,
    group_size: int = 64,
    quantized_linear_cls: Any | None = None,
    eval_fn: Any | None = None,
) -> dict[str, Any]:
    """Optionally quantize MiMo's output projection after weights load."""
    if bits is None:
        return {
            "enabled": False,
            "bits": None,
            "group_size": None,
            "seconds": 0.0,
        }

    if not hasattr(model, "lm_head"):
        raise ValueError("model has no lm_head to quantize")

    import mlx.core as mx
    import mlx.nn as nn

    if quantized_linear_cls is None or eval_fn is None:
        quantized_linear_cls = nn.QuantizedLinear
        eval_fn = mx.eval

    t0 = time.perf_counter()
    source_lm_head = model.lm_head
    if hasattr(source_lm_head, "scales"):
        weight = mx.dequantize(
            source_lm_head.weight,
            source_lm_head.scales,
            getattr(source_lm_head, "biases", None),
            group_size=source_lm_head.group_size,
            bits=source_lm_head.bits,
            mode=getattr(source_lm_head, "mode", "affine"),
        )
        floating_lm_head = nn.Linear(weight.shape[1], weight.shape[0], bias="bias" in source_lm_head)
        floating_lm_head.weight = weight
        if "bias" in source_lm_head:
            floating_lm_head.bias = source_lm_head.bias
        source_lm_head = floating_lm_head

    model.lm_head = quantized_linear_cls.from_linear(
        source_lm_head,
        group_size=group_size,
        bits=bits,
        mode="affine",
    )
    eval_fn(model.lm_head.parameters())
    return {
        "enabled": True,
        "bits": int(bits),
        "group_size": int(group_size),
        "seconds": time.perf_counter() - t0,
    }


def load(
    model_path: str,
    *,
    lazy: bool = True,
    tokenizer_config: dict[str, Any] | None = None,
    quantize_lm_head_bits: int | None = None,
    quantize_lm_head_group_size: int = 64,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load a MiMo model and apply optional post-load runtime tweaks."""
    from mlx_lm.utils import load as mlx_load
    from jang_tools.mimo_v2 import mlx_register  # noqa: F401

    model, tokenizer = mlx_load(
        model_path,
        lazy=lazy,
        tokenizer_config=tokenizer_config or {"trust_remote_code": True},
    )
    lm_head_quantization = quantize_lm_head(
        model,
        bits=quantize_lm_head_bits,
        group_size=quantize_lm_head_group_size,
    )
    return model, tokenizer, {"lm_head_quantization": lm_head_quantization}


__all__ = ["load", "quantize_lm_head"]
