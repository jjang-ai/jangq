# Copyright (c) 2026 Jinho Jang <eric@jangq.ai>
# Native MLX implementation of Mistral Small 4 (119B MoE with MLA)
#
# Architecture: Multi-head Latent Attention (MLA) + Mixture of Experts (MoE)
# This is a standalone model file for mlx-lm, NOT a DeepSeek V2 patch.
#
# Key differences from DeepSeek V2:
#   - Interleaved RoPE (traditional=False)
#   - mscale == mscale_all_dim == 1.0 → no mscale on attention or rope
#   - Llama 4 position-dependent query scaling
#   - n_group=1, topk_group=1 (trivial group routing)
#   - routed_scaling_factor=1.0
#   - norm_topk_prob=True

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "mistral4"
    vocab_size: int = 131072
    hidden_size: int = 4096
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    n_shared_experts: int = 1
    n_routed_experts: int = 128
    num_experts_per_tok: int = 4
    routed_scaling_factor: float = 1.0
    kv_lora_rank: int = 256
    q_lora_rank: int = 1024
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_nope_head_dim: int = 64
    qk_head_dim: int = 128
    n_group: int = 1
    topk_group: int = 1
    first_k_dense_replace: int = 0
    moe_layer_freq: int = 1
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_interleave: bool = True
    rope_scaling: Optional[Dict] = None
    rope_parameters: Optional[Dict] = None
    attention_bias: bool = False
    norm_topk_prob: bool = True
    tie_word_embeddings: bool = False
    head_dim: int = 128

    def __post_init__(self):
        # Merge rope_parameters into rope_scaling if needed
        if self.rope_parameters is not None and self.rope_scaling is None:
            rp = self.rope_parameters
            self.rope_scaling = {
                "type": rp.get("type", rp.get("rope_type", "yarn")),
                "factor": rp.get("factor", 128.0),
                "original_max_position_embeddings": rp.get(
                    "original_max_position_embeddings", 8192
                ),
                "beta_fast": rp.get("beta_fast", 32.0),
                "beta_slow": rp.get("beta_slow", 1.0),
                "mscale": rp.get("mscale", 1.0),
                "mscale_all_dim": rp.get("mscale_all_dim", 1.0),
                "llama_4_scaling_beta": rp.get("llama_4_scaling_beta", 0.0),
            }
            if "rope_theta" in rp:
                self.rope_theta = rp["rope_theta"]


# --------------------------------------------------------------------------- #
# YarnRoPE for Mistral 4
# --------------------------------------------------------------------------- #


def _yarn_find_correction_dim(num_rotations, dim, base, max_pos):
    return (dim * math.log(max_pos / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(low_rot, high_rot, dim, base, max_pos):
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_pos))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_pos))
    return max(low, 0), min(high, dim - 1)


def _yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class Mistral4YarnRoPE(nn.Module):
    """Yarn RoPE for Mistral 4.

    For Mistral 4, mscale == mscale_all_dim == 1.0, so the mscale factor is
    exactly 1.0 (they cancel). This means no scaling is applied to the rope
    embeddings or attention scores — just the frequency interpolation.
    """

    def __init__(
        self,
        dim,
        max_position_embeddings=1048576,
        base=10000,
        scaling_factor=128.0,
        original_max_position_embeddings=8192,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=1,
    ):
        super().__init__()
        # For Mistral 4: mscale=1, mscale_all_dim=1 → self.mscale = 1.0
        self.mscale = _yarn_get_mscale(scaling_factor, mscale) / _yarn_get_mscale(
            scaling_factor, mscale_all_dim
        )

        freq_extra = base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim)
        freq_inter = scaling_factor * base ** (
            mx.arange(0, dim, 2, dtype=mx.float32) / dim
        )
        low, high = _yarn_find_correction_range(
            beta_fast, beta_slow, dim, base, original_max_position_embeddings
        )
        freq_mask = 1.0 - mx.clip(
            (mx.arange(dim // 2, dtype=mx.float32) - low) / max(high - low, 0.001),
            0,
            1,
        )
        self._freqs = (freq_inter * freq_extra) / (
            freq_inter * freq_mask + freq_extra * (1 - freq_mask)
        )

    def __call__(self, x, offset=0):
        # For Mistral 4, mscale is 1.0 so no scaling needed
        if self.mscale != 1.0:
            x = self.mscale * x
        return mx.fast.rope(
            x,
            x.shape[-1],
            traditional=False,  # interleaved RoPE for Mistral 4
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


# --------------------------------------------------------------------------- #
# MLA Attention
# --------------------------------------------------------------------------- #


class Mistral4Attention(nn.Module):
    """Multi-head Latent Attention for Mistral 4.

    Q path: x → q_a_proj → layernorm → q_b_proj → reshape (B,H,L,qk_head_dim)
            split into q_nope (64d) + q_pe (64d), apply RoPE to q_pe

    KV path: x → kv_a_proj_with_mqa → split into compressed_kv (256d) + k_pe (64d)
             compressed_kv → layernorm → kv_b_proj → reshape (B,H,L,qk_nope+v)
             split into k_nope (64d) + values (128d)
             k_pe gets RoPE, then expand from 1 head to 32 heads

    Assemble: keys = cat(k_nope, k_pe) → (B,H,L,128)
              queries = cat(q_nope, q_pe) → (B,H,L,128)

    Llama 4 scaling: queries *= 1 + beta * log(1 + floor(pos/max_pos))
    """

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim  # 128

        self.scale = self.q_head_dim ** -0.5  # 0.0884

        # Q path with low-rank compression
        if self.q_lora_rank is not None and self.q_lora_rank > 0:
            self.q_a_proj = nn.Linear(
                config.hidden_size, self.q_lora_rank, bias=config.attention_bias
            )
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )
        else:
            self.q_proj = nn.Linear(
                config.hidden_size, self.num_heads * self.q_head_dim, bias=False
            )

        # KV path: single projection for compressed KV + rope key
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        # Build Yarn RoPE
        rope_cfg = config.rope_scaling or {}
        self.rope = Mistral4YarnRoPE(
            dim=self.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_factor=rope_cfg.get("factor", 128.0),
            original_max_position_embeddings=rope_cfg.get(
                "original_max_position_embeddings", 8192
            ),
            beta_fast=rope_cfg.get("beta_fast", 32.0),
            beta_slow=rope_cfg.get("beta_slow", 1.0),
            mscale=rope_cfg.get("mscale", 1.0),
            mscale_all_dim=rope_cfg.get("mscale_all_dim", 1.0),
        )

        # Llama 4 position-dependent query scaling
        self._llama4_beta = rope_cfg.get("llama_4_scaling_beta", 0.0)
        self._llama4_max_pos = rope_cfg.get(
            "original_max_position_embeddings", 8192
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        # --- Q path ---
        if self.q_lora_rank is not None and self.q_lora_rank > 0:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        else:
            q = self.q_proj(x)

        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        # --- KV path ---
        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)

        # k_pe is a single head: (B, L, qk_rope_head_dim) → (B, 1, L, qk_rope_head_dim)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)

        # Decompress: (B, L, kv_lora_rank) → (B, L, H, qk_nope+v_head_dim)
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        kv = kv.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        k_nope, values = mx.split(kv, [self.qk_nope_head_dim], axis=-1)

        # --- RoPE (interleaved) ---
        if cache is not None:
            q_pe = self.rope(q_pe, cache.offset)
            k_pe = self.rope(k_pe, cache.offset)
        else:
            q_pe = self.rope(q_pe)
            k_pe = self.rope(k_pe)

        # Expand k_pe from 1 head → num_heads
        k_pe = mx.repeat(k_pe, self.num_heads, axis=1)

        # --- Assemble full keys and queries ---
        keys = mx.concatenate([k_nope, k_pe], axis=-1)
        queries = mx.concatenate([q_nope, q_pe], axis=-1)

        # --- Cache update ---
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        # --- Llama 4 position-dependent query scaling ---
        if self._llama4_beta > 0:
            offset = cache.offset if cache is not None else 0
            # offset is the position AFTER cache update, so the query position
            # is (offset - L) for prefill or (offset - 1) for decode
            # For the scaling formula, we use the current offset which represents
            # the end position of the current tokens
            l4_scale = 1.0 + self._llama4_beta * math.log(
                1.0 + math.floor(offset / self._llama4_max_pos)
            )
            if l4_scale != 1.0:
                queries = queries * l4_scale

        # --- Attention ---
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


# --------------------------------------------------------------------------- #
# MLP
# --------------------------------------------------------------------------- #


class Mistral4MLP(nn.Module):
    def __init__(self, config: ModelArgs, intermediate_size: int = None):
        super().__init__()
        hidden = config.hidden_size
        inter = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def __call__(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


# --------------------------------------------------------------------------- #
# MoE Gate — kept in float, NOT quantized
# --------------------------------------------------------------------------- #


class MoEGate(nn.Module):
    def to_quantized(self, group_size, bits, **kwargs):
        # Gate weights stay as float — dequantized at load time
        return self

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = mx.zeros((self.n_routed_experts, config.hidden_size))

    def __call__(self, x):
        # Gate matmul — stays in native dtype (bfloat16)
        gates = x @ self.weight.T
        scores = mx.softmax(gates, axis=-1, precise=True)

        # Group routing (for Mistral 4: n_group=1, topk_group=1 → trivial)
        if self.n_group > 1 and self.topk_group < self.n_group:
            bsz = x.shape[0] if x.ndim == 2 else x.shape[0] * x.shape[1]
            seq_len = 1 if x.ndim == 2 else x.shape[1]
            scores_grouped = scores.reshape(-1, self.n_group, self.n_routed_experts // self.n_group)
            group_scores = scores_grouped.max(axis=-1, keepdims=True)
            k = self.n_group - self.topk_group
            group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
            scores_grouped = mx.put_along_axis(
                scores_grouped, group_idx, mx.array(0.0, scores_grouped.dtype), axis=-2
            )
            scores = scores_grouped.reshape(*scores.shape)

        # Top-k expert selection
        k = self.top_k
        inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        weights = mx.take_along_axis(scores, inds, axis=-1)

        # Normalize weights (Mistral 4: norm_topk_prob=True)
        if self.norm_topk_prob:
            weights = weights / mx.sum(weights, axis=-1, keepdims=True)

        weights = weights * self.routed_scaling_factor

        return inds, weights


# --------------------------------------------------------------------------- #
# MoE Block
# --------------------------------------------------------------------------- #


class Mistral4MoE(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.gate = MoEGate(config)

        # Routed experts via SwitchGLU (efficient batched expert dispatch)
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
        )

        # Shared expert(s)
        if config.n_shared_experts is not None and config.n_shared_experts > 0:
            shared_inter = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = Mistral4MLP(config, intermediate_size=shared_inter)

    def __call__(self, x):
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        if hasattr(self, "shared_experts"):
            y = y + self.shared_experts(x)
        return y


# --------------------------------------------------------------------------- #
# Decoder Layer
# --------------------------------------------------------------------------- #


class Mistral4DecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Mistral4Attention(config)

        # All layers are MoE in Mistral 4 (moe_layer_freq=1, first_k_dense_replace=0)
        is_moe = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        )
        if is_moe:
            self.mlp = Mistral4MoE(config)
        else:
            self.mlp = Mistral4MLP(config)

        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


# --------------------------------------------------------------------------- #
# Full Model
# --------------------------------------------------------------------------- #


class Mistral4TextModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Mistral4DecoderLayer(config, idx)
            for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(x)

        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = Mistral4TextModel(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        # Convert HF expert weights into SwitchGLU format
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"

            # Handle separate per-expert weights → stacked SwitchGLU
            for n, m in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
                for k in ["weight", "scales", "biases"]:
                    key0 = f"{prefix}.mlp.experts.0.{m}.{k}"
                    if key0 in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(self.args.n_routed_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(
                            to_join
                        )

            # Handle fused gate_up_proj format (some HF checkpoints)
            gate_up_key = f"{prefix}.mlp.experts.gate_up_proj"
            if gate_up_key in weights:
                gate_up = weights.pop(gate_up_key)
                mid = gate_up.shape[-2] // 2
                weights[f"{prefix}.mlp.switch_mlp.gate_proj.weight"] = gate_up[
                    ..., :mid, :
                ]
                weights[f"{prefix}.mlp.switch_mlp.up_proj.weight"] = gate_up[
                    ..., mid:, :
                ]
                down_key = f"{prefix}.mlp.experts.down_proj"
                if down_key in weights:
                    weights[f"{prefix}.mlp.switch_mlp.down_proj.weight"] = (
                        weights.pop(down_key)
                    )

        return weights

    @property
    def layers(self):
        return self.model.layers
