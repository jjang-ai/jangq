"""Mistral 3.5 — text + pixtral VL config dataclasses."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class MinistralTextConfig:
    model_type: str = "ministral3"
    vocab_size: int = 131072
    hidden_size: int = 12288
    intermediate_size: int = 28672
    num_hidden_layers: int = 88
    num_attention_heads: int = 96
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-5
    sliding_window: Optional[int] = None
    tie_word_embeddings: bool = False
    rope_parameters: dict = field(default_factory=dict)


@dataclass
class PixtralVisionConfig:
    model_type: str = "pixtral"
    hidden_size: int = 1664
    intermediate_size: int = 8192
    num_hidden_layers: int = 48
    num_attention_heads: int = 16
    head_dim: int = 104
    image_size: int = 1540
    patch_size: int = 14
    num_channels: int = 3
    rope_parameters: dict = field(default_factory=lambda: {
        "rope_theta": 10000.0, "rope_type": "default"
    })


@dataclass
class Mistral3Config:
    architectures: tuple = ("Mistral3ForConditionalGeneration",)
    model_type: str = "mistral3"
    image_token_index: int = 10
    spatial_merge_size: int = 2
    multimodal_projector_bias: bool = False
    projector_hidden_act: str = "gelu"
    vision_feature_layer: int = -1
    text_config: MinistralTextConfig = field(default_factory=MinistralTextConfig)
    vision_config: PixtralVisionConfig = field(default_factory=PixtralVisionConfig)

    fp8_per_tensor: bool = True  # weight_block_size = null
    fp8_ignored_modules: tuple = (
        "model.vision_tower",
        "model.multi_modal_projector",
        "lm_head",
    )

    @classmethod
    def from_json(cls, path: str | Path) -> "Mistral3Config":
        d = json.loads(Path(path).read_text())
        text = d.get("text_config", {})
        vision = d.get("vision_config", {})
        qc = d.get("quantization_config") or {}
        cfg = cls(
            text_config=MinistralTextConfig(**{k: v for k, v in text.items()
                                               if k in MinistralTextConfig.__dataclass_fields__}),
            vision_config=PixtralVisionConfig(**{k: v for k, v in vision.items()
                                                 if k in PixtralVisionConfig.__dataclass_fields__}),
            image_token_index=d.get("image_token_index", 10),
            spatial_merge_size=d.get("spatial_merge_size", 2),
            multimodal_projector_bias=d.get("multimodal_projector_bias", False),
            projector_hidden_act=d.get("projector_hidden_act", "gelu"),
            vision_feature_layer=d.get("vision_feature_layer", -1),
            fp8_per_tensor=(qc.get("weight_block_size") is None),
            fp8_ignored_modules=tuple(qc.get("modules_to_not_convert") or
                                      cls.fp8_ignored_modules),
        )
        return cfg
