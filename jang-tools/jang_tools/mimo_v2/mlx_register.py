"""Monkey-patch mlx_lm.models to register `mimo_v2` model_type.

Import this module (or `from jang_tools.mimo_v2 import mlx_register`) before
calling mlx_lm.utils.load to make mlx_lm aware of MiMo-V2 JANG bundles.
"""

from __future__ import annotations


def register() -> None:
    import sys
    import importlib

    from jang_tools.mimo_v2 import mlx_model
    sys.modules["mlx_lm.models.mimo_v2"] = mlx_model
    try:
        mlx_lm_models = importlib.import_module("mlx_lm.models")
        if hasattr(mlx_lm_models, "_MODEL_MAPPING"):
            mlx_lm_models._MODEL_MAPPING["mimo_v2"] = mlx_model  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover
        setattr(mlx_model, "_jang_register_warning", repr(exc))


register()
