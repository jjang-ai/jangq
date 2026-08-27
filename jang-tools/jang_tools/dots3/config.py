"""dots3_note config parsing — geometry namespaces for full/SWA MLA, MoE, towers."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AttnGeom:
    num_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    rope_theta: float
    gate_type: str          # "headwise" | "elementwise"
    sliding_window: int | None = None   # None => full attention

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def scale(self) -> float:
        return self.qk_head_dim ** -0.5


@dataclass
class Dots3Config:
    hidden_size: int
    num_hidden_layers: int          # 46 backbone (MTP = index 46 extra)
    vocab_size: int
    rms_norm_eps: float
    intermediate_size: int          # dense layer 0 (and MTP FFN)
    first_k_dense_replace: int
    n_routed_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    n_shared_experts: int
    norm_topk_prob: bool
    routed_scaling_factor: float
    layer_types: list[str]
    full: AttnGeom
    swa: AttnGeom
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    apply_lora_rescale: bool
    k_rope_only_layernorm: bool
    image_token_id: int = 151660
    video_token_id: int = 151680
    audio_token_id: int = 151720
    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, model_dir: str | Path) -> "Dots3Config":
        c = json.loads((Path(model_dir) / "config.json").read_text())
        full = AttnGeom(
            num_heads=c["num_attention_heads"],
            q_lora_rank=c["q_lora_rank"],
            kv_lora_rank=c["kv_lora_rank"],
            qk_nope_head_dim=c["qk_nope_head_dim"],
            qk_rope_head_dim=c["qk_rope_head_dim"],
            v_head_dim=c["v_head_dim"],
            rope_theta=c["rope_theta"],
            gate_type=c.get("attention_gate_type", "headwise"),
            sliding_window=None,
        )
        swa = AttnGeom(
            num_heads=c["swa_num_attention_heads"],
            q_lora_rank=c["swa_q_lora_rank"],
            kv_lora_rank=c["swa_kv_lora_rank"],
            qk_nope_head_dim=c["swa_qk_nope_head_dim"],
            qk_rope_head_dim=c["swa_qk_rope_head_dim"],
            v_head_dim=c["swa_v_head_dim"],
            rope_theta=c["swa_rope_theta"],
            gate_type=c.get("swa_attention_gate_type", "headwise"),
            sliding_window=c.get("sliding_window", c.get("sliding_window_size", 513)),
        )
        return cls(
            hidden_size=c["hidden_size"],
            num_hidden_layers=c["num_hidden_layers"],
            vocab_size=c["vocab_size"],
            rms_norm_eps=c["rms_norm_eps"],
            intermediate_size=c["intermediate_size"],
            first_k_dense_replace=c.get("first_k_dense_replace", 1),
            n_routed_experts=c["n_routed_experts"],
            num_experts_per_tok=c["num_experts_per_tok"],
            moe_intermediate_size=c["moe_intermediate_size"],
            n_shared_experts=c.get("n_shared_experts", 1),
            norm_topk_prob=c.get("norm_topk_prob", True),
            routed_scaling_factor=c.get("routed_scaling_factor", 1.0),
            layer_types=list(c["layer_types"]),
            full=full,
            swa=swa,
            index_n_heads=c.get("index_n_heads", 64),
            index_head_dim=c.get("index_head_dim", 128),
            index_topk=c.get("index_topk", 2048),
            apply_lora_rescale=c.get("apply_mla_qkv_lora_rescale", True),
            k_rope_only_layernorm=c.get("k_rope_only_layernorm", True),
            raw=c,
        )

    def is_sliding(self, layer_idx: int) -> bool:
        if layer_idx >= len(self.layer_types):
            return False        # MTP layer 46: full-attention geometry
        return self.layer_types[layer_idx] == "sliding_attention"

    def is_moe(self, layer_idx: int) -> bool:
        if layer_idx >= self.num_hidden_layers:
            return False        # MTP layer FFN is dense
        return layer_idx >= self.first_k_dense_replace

    def geom(self, layer_idx: int) -> AttnGeom:
        return self.swa if self.is_sliding(layer_idx) else self.full
