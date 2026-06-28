"""Config dataclasses for the MiMo-V2.5 MLX VLM module (mlx_vlm-shaped)."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field


class _FromDict:
    @classmethod
    def from_dict(cls, params: dict):
        valid = inspect.signature(cls.__init__).parameters
        return cls(**{k: v for k, v in params.items() if k in valid})


@dataclass
class VisionConfig(_FromDict):
    depth: int = 28
    hidden_size: int = 1280
    intermediate_size: int = 4608
    hidden_act: str = "silu"
    num_heads: int = 32
    num_key_value_heads: int = 8
    qk_channels: int = 64
    in_chans: int = 3
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    out_hidden_size: int = 4096
    rms_norm_eps: float = 1e-6
    use_sink: bool = True
    visual_token_window_size: int = 64
    tokens_per_second: int = 2
    fullatt_block_indexes: list = field(default_factory=lambda: [0, 9, 18, 27])
    vit_window_attn_types: list | None = None

    def resolved_window_attn_types(self) -> list[int]:
        if self.vit_window_attn_types:
            return list(self.vit_window_attn_types)
        return [-1] * self.depth


@dataclass
class TextConfig(_FromDict):
    """Marker holder; the text side reuses jang_tools.mimo_v2.mlx_model.ModelArgs."""

    raw: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, params: dict):
        return cls(raw=dict(params))


@dataclass
class ModelConfig(_FromDict):
    model_type: str = "mimo_v2"
    vision_config: VisionConfig | dict | None = None
    text_config: TextConfig | dict | None = None
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    eos_token_id: int | list | None = None
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, params: dict):
        cfg = super().from_dict(params)
        if isinstance(cfg.vision_config, dict):
            cfg.vision_config = VisionConfig.from_dict(cfg.vision_config)
        elif cfg.vision_config is None:
            cfg.vision_config = VisionConfig()
        if isinstance(cfg.text_config, dict):
            cfg.text_config = TextConfig.from_dict(cfg.text_config)
        else:
            # MiMo stores text fields at the top level of config.json.
            cfg.text_config = TextConfig.from_dict(params)
        cfg.raw = dict(params)
        return cfg
