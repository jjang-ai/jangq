"""Auto-detecting bundle loader.

Detects bundle format by scanning config.json:
    - "fp8"      : quantization_config.quant_method == "fp8" or "compressed-tensors"
    - "jangtq"   : has mxtq_bits or routed_expert_bits keys (2026-04-25 invariant)
    - "jang"     : quantization.bits + quantization.group_size present
    - "bf16"     : neither — vanilla weights

Returns a triple (model, config, tokenizer). The model arch is dispatched
by config.architectures / config.model_type via a small registry that
each model package (mimo_v2, mistral3, laguna, ...) registers itself in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

ARCH_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_arch(model_type: str):
    def _wrap(fn):
        ARCH_REGISTRY[model_type] = fn
        return fn
    return _wrap


def detect_format(src: str) -> str:
    cfg = json.loads((Path(src) / "config.json").read_text())
    qc = cfg.get("quantization_config") or {}
    method = qc.get("quant_method") or qc.get("format")
    if method in ("fp8", "compressed-tensors", "float-quantized"):
        return "fp8"
    if "mxtq_bits" in cfg or "routed_expert_bits" in cfg:
        return "jangtq"
    qq = cfg.get("quantization") or {}
    if isinstance(qq, dict) and qq.get("bits") and qq.get("group_size"):
        return "jang"
    return "bf16"


def load_bundle(src: str, *, plan=None):
    cfg_dict = json.loads((Path(src) / "config.json").read_text())
    inner = cfg_dict.get("text_config") or cfg_dict
    mt = inner.get("model_type") or cfg_dict.get("model_type")
    if mt not in ARCH_REGISTRY:
        raise ValueError(
            f"unknown model_type {mt!r}. Registered: {sorted(ARCH_REGISTRY)}. "
            f"Import the matching jang_tools.<arch> package to register it."
        )
    fmt = detect_format(src)
    builder = ARCH_REGISTRY[mt]
    return builder(src, fmt=fmt, plan=plan)
