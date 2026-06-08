"""Local MLX bridge for Step3p7 bundles.

The current MLX runtime has a Step3p5 text model but no Step3p7 VLM wrapper.
For text coherence proof, this module exposes a custom ``model_file`` that
loads the nested ``text_config`` through ``mlx_lm.models.step3p5`` and drops
vision tensors during sanitize. Vision runtime remains a separate follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass

from mlx_lm.models import step3p5


@dataclass
class ModelArgs(step3p5.ModelArgs):
    @classmethod
    def from_dict(cls, params):
        text_config = dict(params.get("text_config") or params)
        text_config["model_type"] = "step3p5"
        return super().from_dict(text_config)


class Model(step3p5.Model):
    def sanitize(self, weights):
        text_weights = {}
        prefix = "model.language_model."
        for key, value in weights.items():
            if key.startswith(prefix):
                text_weights["model." + key[len(prefix):]] = value
            elif key.startswith("model.vision_model.") or key.startswith("model.vit_large_projector."):
                continue
            elif key.startswith("vision_model.") or key.startswith("vit_large_projector."):
                continue
            else:
                text_weights[key] = value
        return super().sanitize(text_weights)
