"""MLX runtime for MiMo-V2 (Xiaomi).

Text-only LLM body — drives the JANG bundle's main decode path. The vision
(`visual.*`) and audio (`audio_encoder.*`, `speech_embeddings.*`) towers ship
their weights through the bundle but are NOT wired into this forward path yet;
they live in `mimo_v2_multimodal.py` (follow-up). MTP weights are likewise
loaded-but-ignored — speculative decode is a separate runtime path.

Architecture (MiMo-V2.5, 310B/15B-active):
- 48 hybrid layers: full-attention at layers {0,5,11,17,23,29,35,41,47},
  sliding-window (w=128) at the other 39
- Asymmetric KV per layer-type:
    full: 64H / 4KV / qk_dim=192 / v_dim=128
    swa:  64H / 8KV / qk_dim=192 / v_dim=128
- Partial RoPE: rope_dim = int(192*0.334) = 64 on first 64 head dims
- attention_value_scale = 0.707 (V pre-multiplier)
- Sink bias on SWA only (cat as last column, softmax, drop last)
- MoE: 256 experts × top-8, sigmoid + e_score_correction_bias + noaux_tc
  (n_group=1, topk_group=1 → degenerates to plain top-8 from all 256)
- No shared expert; layer 0 is dense (interm=16384)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache
from mlx_lm.models.switch_layers import SwitchGLU


_INPUTS_EMBEDS_UNSET = object()


class MiMoV2CausalLMOutput:
    def __init__(self, logits: mx.array):
        self.logits = logits
        self.cross_attention_states = None
        self.encoder_outputs = None


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "mimo_v2"
    vocab_size: int = 152576
    hidden_size: int = 4096
    intermediate_size: int = 16384  # layer-0 dense
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 48
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    swa_num_attention_heads: int = 64
    swa_num_key_value_heads: int = 8
    head_dim: int = 192
    v_head_dim: int = 128
    swa_head_dim: int = 192
    swa_v_head_dim: int = 128
    layernorm_epsilon: float = 1e-5
    rope_theta: float = 10_000_000.0
    swa_rope_theta: float = 10_000.0
    partial_rotary_factor: float = 0.334
    sliding_window: int = 128
    attention_value_scale: float = 0.707
    add_full_attention_sink_bias: bool = False
    add_swa_attention_sink_bias: bool = True
    hybrid_layer_pattern: list[int] = field(default_factory=list)
    moe_layer_freq: list[int] = field(default_factory=list)
    n_routed_experts: int = 256
    num_experts_per_tok: int = 8
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    routed_scaling_factor: Optional[float] = None
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"
    max_position_embeddings: int = 1_048_576
    tie_word_embeddings: bool = False
    # Multimodal flags (used by sanitize / future runtime wiring).
    vision_config: Optional[dict] = None
    audio_config: Optional[dict] = None
    processor_config: Optional[dict] = None
    image_token_id: Optional[int] = None
    video_token_id: Optional[int] = None
    vision_start_token_id: Optional[int] = None
    vision_end_token_id: Optional[int] = None
    # Quantization metadata.
    quantization: Optional[dict] = None
    mxtq_bits: Optional[int] = None
    routed_expert_bits: Optional[int] = None
    debug_disable_attention_sink: bool = False


# --------------------------------------------------------------------------
# Rotary embedding (partial)
# --------------------------------------------------------------------------


class MiMoV2RoPE(nn.Module):
    """Partial rotary on first `rope_dim` head_dim positions.

    rope_dim = int(head_dim * partial_rotary_factor). For MiMo this is 64 of 192.
    Per-layer-type theta: full uses rope_theta (1e7), SWA uses swa_rope_theta (1e4).
    """

    def __init__(self, head_dim: int, rope_dim: int, base: float):
        super().__init__()
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.base = base

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        # x shape: (B, H, T, head_dim)
        if self.rope_dim == 0:
            return x
        rope_part = x[..., : self.rope_dim]
        nope_part = x[..., self.rope_dim :]
        rope_out = mx.fast.rope(
            rope_part,
            dims=self.rope_dim,
            traditional=False,
            base=self.base,
            scale=1.0,
            offset=offset,
        )
        return mx.concatenate([rope_out, nope_part], axis=-1)


# --------------------------------------------------------------------------
# Attention (fused qkv, asymmetric KV, V scale, sink bias)
# --------------------------------------------------------------------------


class MiMoV2Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_swa = args.hybrid_layer_pattern[layer_idx] == 1
        if self.is_swa:
            self.num_heads = args.swa_num_attention_heads
            self.num_kv_heads = args.swa_num_key_value_heads
            self.head_dim = args.swa_head_dim
            self.v_head_dim = args.swa_v_head_dim
            theta = args.swa_rope_theta
            self.sliding_window: Optional[int] = args.sliding_window
            has_sink_parameter = args.add_swa_attention_sink_bias
            self.use_sink = has_sink_parameter
        else:
            self.num_heads = args.num_attention_heads
            self.num_kv_heads = args.num_key_value_heads
            self.head_dim = args.head_dim
            self.v_head_dim = args.v_head_dim
            theta = args.rope_theta
            self.sliding_window = None
            has_sink_parameter = args.add_full_attention_sink_bias
            self.use_sink = has_sink_parameter
        if args.debug_disable_attention_sink or os.environ.get("JANG_MIMO_DISABLE_SINK") in {"1", "true", "yes"}:
            self.use_sink = False
        self.rope_dim = int(self.head_dim * args.partial_rotary_factor)
        if self.rope_dim % 2 != 0:
            raise ValueError(f"rope_dim must be even, got {self.rope_dim}")
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim ** -0.5
        self.v_scale = args.attention_value_scale
        self.q_size = self.num_heads * self.head_dim
        self.k_size = self.num_kv_heads * self.head_dim
        self.v_size = self.num_kv_heads * self.v_head_dim
        self.o_hidden = self.num_heads * self.v_head_dim
        self.qkv_proj = nn.Linear(args.hidden_size, self.q_size + self.k_size + self.v_size, bias=False)
        self.o_proj = nn.Linear(self.o_hidden, args.hidden_size, bias=False)
        self.rope = MiMoV2RoPE(self.head_dim, self.rope_dim, theta)
        if has_sink_parameter:
            # Loaded from `self_attn.attention_sink_bias` (shape (num_heads,)).
            self.attention_sink_bias = mx.zeros((self.num_heads,))

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, T, _ = x.shape
        qkv = self.qkv_proj(x)
        q = qkv[..., : self.q_size]
        k = qkv[..., self.q_size : self.q_size + self.k_size]
        v = qkv[..., self.q_size + self.k_size :]
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_kv_heads, self.v_head_dim).transpose(0, 2, 1, 3)
        if self.v_scale is not None:
            v = v * self.v_scale

        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        if self.use_sink and os.environ.get("JANG_MIMO_MANUAL_SINK_SDPA") not in {"1", "true", "yes"}:
            attn_out = scaled_dot_product_attention(
                q,
                k,
                v,
                cache=None,
                scale=self.scale,
                mask=mask,
                sinks=self.attention_sink_bias,
            )
        elif self.use_sink:
            attn_out = _sdpa_with_sink(
                q, k, v, mask, self.scale, self.attention_sink_bias,
                sliding_window=self.sliding_window,
            )
        else:
            attn_out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, self.o_hidden)
        return self.o_proj(attn_out)


def _sdpa_with_sink(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    mask,
    scale: float,
    sink_bias: mx.array,
    sliding_window: Optional[int] = None,
) -> mx.array:
    """SDPA with a learned per-head sink bias appended as one extra softmax slot.

    Matches torch reference: cat sink as last column → softmax over all → drop last.

    `mask` may be: an mx.array, the string "causal", or None. SWA layers also pass
    a `sliding_window` length to enforce |i-j| <= window.
    """
    # Repeat KV groups so dims match Q heads.
    H = q.shape[1]
    K = k.shape[1]
    if K != H:
        rep = H // K
        k = mx.repeat(k, rep, axis=1)
        v = mx.repeat(v, rep, axis=1)
    # Compute attn logits manually.
    attn = (q @ k.transpose(0, 1, 3, 2)) * scale  # (B, H, T, S)
    B, _, T, S = attn.shape
    # Build mask array if needed (mlx-lm uses "causal" string sentinel).
    if isinstance(mask, str):
        if mask == "causal":
            # Row i (= query pos S - T + i) attends to key positions 0..(S-T+i).
            i = mx.arange(T)[:, None]
            j = mx.arange(S)[None, :]
            offset = S - T
            allowed = j <= (i + offset)
            if sliding_window is not None:
                allowed = allowed & (j >= (i + offset - sliding_window + 1))
            cm = mx.where(allowed, mx.array(0.0, dtype=attn.dtype),
                          mx.array(-mx.inf, dtype=attn.dtype))
            attn = attn + cm
    elif mask is not None:
        if mask.dtype == mx.bool_:
            additive_mask = mx.where(
                mask,
                mx.array(0.0, dtype=attn.dtype),
                mx.array(-mx.inf, dtype=attn.dtype),
            )
            attn = attn + additive_mask
        else:
            attn = attn + mask
        # If a fully-formed mask was provided, sliding_window was already baked in by the caller.
    # Build sink column: shape (B, H, T, 1), value = sink_bias broadcast across (T,1).
    sink_col = mx.broadcast_to(sink_bias.reshape(1, H, 1, 1), (B, H, T, 1))
    attn_full = mx.concatenate([attn, sink_col], axis=-1)  # (B,H,T,S+1)
    # Stabilize + softmax.
    attn_full = attn_full - attn_full.max(axis=-1, keepdims=True)
    probs = mx.softmax(attn_full.astype(mx.float32), axis=-1).astype(q.dtype)
    probs = probs[..., :-1]  # drop sink
    return probs @ v  # (B, H, T, v_dim)


# --------------------------------------------------------------------------
# MoE — sigmoid + noaux_tc with n_group=1 degenerate top-8
# --------------------------------------------------------------------------


class MiMoV2MoEGate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.n_experts = args.n_routed_experts
        self.norm_topk_prob = args.norm_topk_prob
        self.routed_scaling = args.routed_scaling_factor if args.routed_scaling_factor is not None else 1.0
        # Router weight stored as fp32 in source; loaded as bf16 here unless explicitly preserved.
        self.weight = mx.zeros((self.n_experts, args.hidden_size))
        # e_score_correction_bias for noaux_tc routing (added BEFORE topk choose).
        self.e_score_correction_bias = mx.zeros((self.n_experts,))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        # x: (B*T, H). Returns (topk_indices, topk_weights).
        # Compute in fp32 for routing precision.
        x_fp32 = x.astype(mx.float32)
        w_fp32 = self.weight.astype(mx.float32)
        logits = x_fp32 @ w_fp32.T  # (N, n_experts)
        scores = mx.sigmoid(logits)
        # noaux_tc with n_group=1 degenerates to plain top-k over all experts.
        scores_for_choice = scores + self.e_score_correction_bias.astype(mx.float32).reshape(1, -1)
        # Top-k indices by corrected score.
        topk_indices = mx.argpartition(-scores_for_choice, kth=self.top_k - 1, axis=-1)[..., : self.top_k]
        # Original (un-corrected) sigmoid scores for the weights.
        topk_weights = mx.take_along_axis(scores, topk_indices, axis=-1)
        if self.top_k > 1 and self.norm_topk_prob:
            denom = topk_weights.sum(axis=-1, keepdims=True) + 1e-20
            topk_weights = topk_weights / denom
        topk_weights = topk_weights * self.routed_scaling
        return topk_indices, topk_weights.astype(x.dtype)


class MiMoV2MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate = MiMoV2MoEGate(args)
        # SwitchGLU default activation is SwiGLU(), which is silu(gate) * up — matches MiMo.
        # Bias=False per MiMo (no biases on expert proj weights).
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.n_routed_experts, bias=False,
        )

    def __call__(self, x: mx.array) -> mx.array:
        B, T, H = x.shape
        x_flat = x.reshape(-1, H)
        topk_idx, topk_w = self.gate(x_flat)
        # SwitchGLU returns per-token per-expert outputs of shape (N, K, H).
        out = self.switch_mlp(x_flat, topk_idx)
        # Weighted sum across the K selected experts.
        out = (out * topk_w[..., None]).sum(axis=1)
        return out.reshape(B, T, H)


# --------------------------------------------------------------------------
# Layer 0 dense MLP
# --------------------------------------------------------------------------


class MiMoV2MLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


# --------------------------------------------------------------------------
# Decoder layer
# --------------------------------------------------------------------------


class MiMoV2DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_swa = args.hybrid_layer_pattern[layer_idx] == 1
        self.self_attn = MiMoV2Attention(args, layer_idx)
        if args.moe_layer_freq[layer_idx] and args.n_routed_experts is not None:
            self.mlp: nn.Module = MiMoV2MoE(args)
        else:
            self.mlp = MiMoV2MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.layernorm_epsilon)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.layernorm_epsilon)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        h = h + self.mlp(self.post_attention_layernorm(h))
        return h


# --------------------------------------------------------------------------
# Top-level model
# --------------------------------------------------------------------------


class MiMoV2Backbone(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [MiMoV2DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.layernorm_epsilon)
        self.args = args

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        cache: Optional[list[KVCache]] = None,
        mask: Optional[mx.array] = None,
        **kwargs: Any,
    ) -> mx.array:
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Specify input_ids or inputs_embeds")
        h = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, c in zip(self.layers, cache):
            layer_mask = mask
            if layer_mask is None:
                layer_mask = create_attention_mask(
                    h,
                    c,
                    window_size=layer.self_attn.sliding_window,
                )
            h = layer(h, mask=layer_mask, cache=c)
        return self.norm(h)


class Model(nn.Module):
    """Entry point — mlx_lm picks this up as `Model(args)`."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MiMoV2Backbone(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Any = _INPUTS_EMBEDS_UNSET,
        cache: Optional[list[KVCache]] = None,
        mask: Optional[mx.array] = None,
        **kwargs: Any,
    ) -> mx.array | MiMoV2CausalLMOutput:
        mllm_style_call = inputs_embeds is not _INPUTS_EMBEDS_UNSET
        normalized_inputs_embeds = None if inputs_embeds is _INPUTS_EMBEDS_UNSET else inputs_embeds
        h = self.model(
            input_ids,
            inputs_embeds=normalized_inputs_embeds,
            cache=cache,
            mask=mask,
            **kwargs,
        )
        logits = self.lm_head(h)
        if mllm_style_call:
            return MiMoV2CausalLMOutput(logits)
        return logits

    @property
    def layers(self) -> list[nn.Module]:
        return self.model.layers

    def make_cache(self) -> list[KVCache]:
        caches: list[KVCache] = []
        for layer in self.model.layers:
            # SWA layers MUST use a plain KVCache, NOT RotatingKVCache.
            # RotatingKVCache(max_size=window) corrupts generation when the
            # PREFILL exceeds max_size: _update_concat stores the full prompt,
            # then the first decode step trims+rotates inconsistently, desyncing
            # the rope'd key positions from the rotated buffer -> degenerate
            # newline/im_end loops once total length > sliding_window (128).
            # KVCache stores all keys and lets create_attention_mask(window_size)
            # enforce the sliding window via make_mask in BOTH prefill and decode,
            # which matches the (coherent) no-cache full-reforward path exactly.
            # (A correct windowed rotating cache is a future memory optimization.)
            caches.append(KVCache())
        return caches

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Adapt JANG bundle weights to this MLX layout.

        Mainly: stack the 256 individual `experts.{0..255}.{gate,up,down}_proj.weight`
        tensors per MoE layer into the SwitchGLU's three batched weight tensors.
        Also drops MTP weights, vision weights, and audio weights (loaded in
        separate multimodal runtime, see future mimo_v2_multimodal.py).
        """
        new_weights: dict[str, mx.array] = {}
        # Defer expert stacking — collect them by (layer, projection) first.
        expert_stash: dict[tuple[int, str], dict[int, mx.array]] = {}
        n_experts = self.args.n_routed_experts
        for key, value in weights.items():
            if key.startswith("model.mtp.") or key.startswith("visual.") \
               or key.startswith("audio_encoder.") or key.startswith("speech_embeddings."):
                # Carry forward as opaque — runtime will leave them unmapped on the text-only path.
                # Mlx_lm's load_weights with strict=False will tolerate these extras.
                continue
            if ".mlp.experts." in key:
                # Pattern: model.layers.{L}.mlp.experts.{E}.{gate|up|down}_proj.weight
                # (plus .scales / .biases for quantized tensors)
                parts = key.split(".")
                layer_idx = int(parts[2])
                expert_idx = int(parts[5])
                proj = parts[6]  # gate_proj / up_proj / down_proj
                suffix = ".".join(parts[7:])  # weight / scales / biases
                stash_key = (layer_idx, proj, suffix)
                expert_stash.setdefault(stash_key, {})[expert_idx] = value
                continue
            new_weights[key] = value

        # Stack experts: produce (n_experts, ...) tensors.
        for (layer_idx, proj, suffix), per_expert in expert_stash.items():
            if len(per_expert) != n_experts:
                # Partial set — fail loudly; bundle is malformed.
                raise RuntimeError(
                    f"layer {layer_idx} {proj}.{suffix}: expected {n_experts} experts, got {len(per_expert)}"
                )
            stacked = mx.stack([per_expert[i] for i in range(n_experts)], axis=0)
            # SwitchGLU expects: gate_proj / up_proj / down_proj on the switch_mlp module.
            out_key = f"model.layers.{layer_idx}.mlp.switch_mlp.{proj}.{suffix}"
            new_weights[out_key] = stacked

        return new_weights
