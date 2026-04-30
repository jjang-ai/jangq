"""MiMoV2Config — minimal MLX-side mirror of upstream MiMoV2Config.

Mirrors the relevant fields the MLX runtime needs; ignores HF-specific knobs
(rope_config_validation, attribute_map, tp_plan) that don't apply here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MiMoV2Config:
    # Core
    model_type: str = "mimo_v2"
    vocab_size: int = 152576
    hidden_size: int = 6144
    intermediate_size: int = 16384  # dense (layer 0)
    num_hidden_layers: int = 70
    num_attention_heads: int = 128
    num_key_value_heads: int = 8
    head_dim: int = 192
    v_head_dim: int = 128
    hidden_act: str = "silu"
    layernorm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    max_position_embeddings: int = 1_048_576
    tie_word_embeddings: bool = False

    # Attention
    attention_bias: bool = False
    attention_dropout: float = 0.0
    attention_value_scale: Optional[float] = 0.612
    attention_projection_layout: str = "fused_qkv"  # MiMo-V2.5-Pro uses fused
    partial_rotary_factor: float = 0.334
    rope_theta: float = 10_000_000.0
    rope_scaling: Optional[dict] = None

    # Hybrid attn (SWA)
    swa_num_attention_heads: int = 128
    swa_num_key_value_heads: int = 8
    swa_head_dim: int = 192
    swa_v_head_dim: int = 128
    swa_rope_theta: float = 10_000.0
    sliding_window: Optional[int] = 128
    sliding_window_size: Optional[int] = 128
    add_full_attention_sink_bias: bool = False
    add_swa_attention_sink_bias: bool = True
    hybrid_block_size: Optional[int] = None
    hybrid_layer_pattern: List[int] = field(default_factory=list)  # 0=full, 1=swa

    # MoE
    n_routed_experts: int = 384
    n_shared_experts: Optional[int] = None
    moe_intermediate_size: int = 2048
    num_experts_per_tok: int = 8
    routed_scaling_factor: Optional[float] = None
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    moe_layer_freq: List[int] = field(default_factory=list)

    # MTP
    num_mtp_layers: int = 3

    # Quant source meta (read-only — set by loader)
    fp8_weight_block_size: tuple = (128, 128)
    fp8_ignored_layers: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.hybrid_layer_pattern:
            self.hybrid_layer_pattern = [0] * self.num_hidden_layers
        if len(self.hybrid_layer_pattern) != self.num_hidden_layers:
            raise ValueError(
                f"hybrid_layer_pattern length {len(self.hybrid_layer_pattern)} "
                f"!= num_hidden_layers {self.num_hidden_layers}"
            )
        if not self.moe_layer_freq:
            # Default: layer 0 dense, all others MoE (matches upstream)
            self.moe_layer_freq = [0] + [1] * (self.num_hidden_layers - 1)

    @classmethod
    def from_dict(cls, d: dict) -> "MiMoV2Config":
        kept = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**kept)

    @classmethod
    def from_json(cls, path: str) -> "MiMoV2Config":
        import json
        with open(path) as f:
            d = json.load(f)
        # Carry FP8 quant metadata
        qc = d.get("quantization_config") or {}
        if qc:
            d["fp8_weight_block_size"] = tuple(qc.get("weight_block_size", (128, 128)))
            d["fp8_ignored_layers"] = list(qc.get("ignored_layers", []))
        return cls.from_dict(d)
