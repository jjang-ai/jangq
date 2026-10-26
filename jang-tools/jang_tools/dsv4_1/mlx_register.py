"""Register the DeepSeek-V4.1 model class with mlx_lm so ``model_type: deepseek_v41`` resolves.

Mirror of ``jang_tools.dsv4.mlx_register``: pre-populate ``sys.modules`` so
``mlx_lm.utils.load_model`` finds ``.Model`` / ``.ModelArgs`` for the new type, and add the mapping
entry when mlx_lm exposes one. Loading the actual weights goes through ``mlx_model.load_deepseek_v41``;
this shim only makes the class discoverable.
"""

from __future__ import annotations

import sys

from jang_tools.dsv4_1 import mlx_model

_MODEL_TYPE = "deepseek_v41"


def register() -> None:
    sys.modules.setdefault(f"mlx_lm.models.{_MODEL_TYPE}", mlx_model)
    sys.modules[f"mlx_lm.models.{_MODEL_TYPE}"] = mlx_model
    try:  # optional: some mlx_lm builds keep an explicit mapping dict
        import mlx_lm.models as mlx_lm_models

        mapping = getattr(mlx_lm_models, "_MODEL_MAPPING", None)
        if isinstance(mapping, dict):
            mapping[_MODEL_TYPE] = mlx_model
    except ImportError:  # no mlx_lm: nothing to register with
        return


register()
