"""Register `nanbeige` with mlx_lm so `mlx_lm.utils.load()` finds the model.

Import before calling `mlx_lm.load`:

    from jang_tools.nanbeige import mlx_register  # noqa: F401
"""

from __future__ import annotations


def register() -> None:
    import importlib
    import sys

    from jang_tools.nanbeige import model as mlx_model

    sys.modules["mlx_lm.models.nanbeige"] = mlx_model
    try:
        mlx_lm_models = importlib.import_module("mlx_lm.models")
        if hasattr(mlx_lm_models, "_MODEL_MAPPING"):
            mlx_lm_models._MODEL_MAPPING["nanbeige"] = mlx_model  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover
        setattr(mlx_model, "_jang_register_warning", repr(exc))


register()
