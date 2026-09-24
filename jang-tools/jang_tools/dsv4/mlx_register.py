"""Monkey-patch mlx_lm.models to register `deepseek_v4` model_type."""

from __future__ import annotations


def register() -> None:
    """Inject deepseek_v4 into mlx_lm.models so AutoModel can find it."""
    import sys
    import importlib

    # Register our module as mlx_lm.models.deepseek_v4
    from jang_tools.dsv4 import mlx_model
    sys.modules["mlx_lm.models.deepseek_v4"] = mlx_model
    # Also expose under _MODEL_MAPPING if mlx_lm uses one
    try:
        mlx_lm_models = importlib.import_module("mlx_lm.models")
        # Some mlx_lm versions have a factory dict; guard both shapes
        if hasattr(mlx_lm_models, "_MODEL_MAPPING"):
            mlx_lm_models._MODEL_MAPPING["deepseek_v4"] = mlx_model  # type: ignore
    except ImportError as exc:
        setattr(mlx_model, "_jang_register_warning", repr(exc))


# Register immediately on import so `load_jangtq_model` can find it
register()
