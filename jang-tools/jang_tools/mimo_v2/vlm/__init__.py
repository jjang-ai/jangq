"""MiMo-V2.5 multimodal (vision) MLX runtime package.

Importing this package registers the module under
``sys.modules["mlx_vlm.models.mimo_v2"]`` so ``mlx_vlm`` model resolution
finds it (same pattern as ``jang_tools.zaya1_vl``).
"""

from __future__ import annotations

import sys

from .config import ModelConfig, TextConfig, VisionConfig
from .vision import VisionModel
from .model import LanguageModel, Model

sys.modules.setdefault("mlx_vlm.models.mimo_v2", sys.modules[__name__])

__all__ = ["Model", "ModelConfig", "TextConfig", "VisionConfig", "VisionModel", "LanguageModel"]
