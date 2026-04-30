"""Laguna (poolside, model_type=laguna) JANG arch."""
from .config import LagunaConfig
from .model import LagunaForCausalLM

__all__ = ["LagunaConfig", "LagunaForCausalLM"]
