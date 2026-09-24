"""Monkey-patch mlx_lm.models to register `mimo_v2` model_type.

Import this module (or `from jang_tools.mimo_v2 import mlx_register`) before
calling mlx_lm.utils.load to make mlx_lm aware of MiMo-V2 JANG bundles.
"""

from __future__ import annotations


def register() -> None:
    import sys
    import importlib

    # V2.6 runtime (2026-09-22) replaces the V2.5 mlx_model, which mis-decoded
    # full-attention qkv and is not trusted (see v26_model.py docstring).
    from jang_tools.mimo_v2 import v26_model as mlx_model
    sys.modules["mlx_lm.models.mimo_v2"] = mlx_model
    try:
        mlx_lm_models = importlib.import_module("mlx_lm.models")
        if hasattr(mlx_lm_models, "_MODEL_MAPPING"):
            mlx_lm_models._MODEL_MAPPING["mimo_v2"] = mlx_model  # type: ignore[attr-defined]
    except ImportError as exc:  # pragma: no cover
        setattr(mlx_model, "_jang_register_warning", repr(exc))


register()
