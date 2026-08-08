"""Inkling (thinkingmachines) JANG family — MLX runtime + converter support.

Spec and verification gates: `docs/runtime/inkling-small/`.
"""

from __future__ import annotations

from .model import (  # noqa: F401
    InklingAttention,
    InklingCache,
    InklingDecoderLayer,
    InklingLayerCache,
    InklingModel,
    InklingMoE,
    InklingRouter,
    Model,
    ModelArgs,
    RelLogits,
    ShortConv,
)
from . import mlx_register  # noqa: F401  (registers mlx_lm.models.inkling)

__all__ = [
    "Model",
    "ModelArgs",
    "InklingCache",
    "InklingLayerCache",
    "InklingModel",
    "InklingDecoderLayer",
    "InklingAttention",
    "InklingMoE",
    "InklingRouter",
    "RelLogits",
    "ShortConv",
]
