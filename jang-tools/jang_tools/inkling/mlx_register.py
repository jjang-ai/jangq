"""Register the Inkling family with mlx_lm.

Import side effect:

    from jang_tools.inkling import mlx_register  # noqa: F401

makes ``mlx_lm.utils.load_model`` able to resolve ``model_type`` values of
``inkling`` and ``inkling_mm_model``. ``load_jang.py`` / ``load_jangtq.py`` must
import ``jang_tools.inkling`` BEFORE constructing the MLX skeleton, or loading
fails with an unknown architecture.
"""

from __future__ import annotations

import importlib
import sys

# The checkpoint reports `inkling_mm_model` at the top level (multimodal wrapper)
# with the text stack nested under `text_config`. Both resolve to the same text
# runtime; the vision/audio towers are dropped by Model.sanitize().
_MODEL_TYPES = ("inkling", "inkling_mm_model")


def register() -> None:
    from jang_tools.inkling import model as mlx_model

    for name in _MODEL_TYPES:
        sys.modules[f"mlx_lm.models.{name}"] = mlx_model
    try:
        mlx_lm_models = importlib.import_module("mlx_lm.models")
        if hasattr(mlx_lm_models, "_MODEL_MAPPING"):
            for name in _MODEL_TYPES:
                mlx_lm_models._MODEL_MAPPING[name] = mlx_model  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - registration is best-effort
        setattr(mlx_model, "_jang_register_warning", repr(exc))


register()
