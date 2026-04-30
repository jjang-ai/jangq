"""Laguna config — captures the unusual per-layer head count and dual RoPE."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class LagunaConfig:
    model_type: str = "laguna"
    vocab_size: int = 100352
    hidden_size: int = 2048
    intermediate_size: int = 8192   # dense layer 0
    num_hidden_layers: int = 40
    num_attention_heads: int = 48
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    sliding_window: int = 512
    partial_rotary_factor: float = 0.5
    tie_word_embeddings: bool = False

    num_experts: int = 256
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    moe_routed_scaling_factor: float = 2.5
    moe_apply_router_weight_on_input: bool = False
    gating: bool = True

    layer_types: List[str] = field(default_factory=list)             # full_attention | sliding_attention
    mlp_layer_types: List[str] = field(default_factory=list)          # dense | sparse
    num_attention_heads_per_layer: List[int] = field(default_factory=list)

    rope_parameters: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.layer_types:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if not self.mlp_layer_types:
            self.mlp_layer_types = ["dense"] + ["sparse"] * (self.num_hidden_layers - 1)
        if not self.num_attention_heads_per_layer:
            self.num_attention_heads_per_layer = [self.num_attention_heads] * self.num_hidden_layers

    @classmethod
    def from_json(cls, path: str | Path) -> "LagunaConfig":
        d = json.loads(Path(path).read_text())
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
