"""DeepSeek-V4.1-Flash engine model (model_type ``deepseek_v41``). Importing registers it with mlx_lm."""

from jang_tools.dsv4_1 import mlx_register  # noqa: F401  (side effect: register with mlx_lm)
from jang_tools.dsv4_1.mlx_model import Model, ModelArgs, load_deepseek_v41

__all__ = ["Model", "ModelArgs", "load_deepseek_v41", "mlx_register"]
