"""MiMo-V2.6 text backbone for MLX (fresh implementation, 2026-09-22).

Written from Xiaomi's reference ``modeling_mimo_v2.py`` shipped with
MiMo-V2.6-Flash-RL, NOT from the older V2.5 runtime in this package.

Reference semantics mirrored here (line refs are to the V2.6 reference file):
  * RMSNorm: fp32 variance, ``weight * x_normed`` (no +1 shift).
  * Attention: fused qkv split contiguously as [q | k | v] AFTER the
    converter's TP-rank de-interleave (the checkpoint stores rank blocks,
    measured 2026-09-22); q/k head_dim 192, v head_dim 128,
    ``scale = head_dim ** -0.5``, V pre-multiplied by ``attention_value_scale``.
  * Partial RoPE on the first ``int(head_dim * partial_rotary_factor)`` = 64
    dims, rotate-half (non-traditional). Full layers use ``rope_theta``,
    SWA layers use ``swa_rope_theta``. Plain 1-D positions for every token,
    including image/audio placeholders (no M-RoPE).
  * Sink bias: learned per-head logit appended to the softmax and dropped,
    on SWA layers only (add_swa_attention_sink_bias=true,
    add_full_attention_sink_bias=false). Implemented with MLX SDPA ``sinks``.
  * Sliding window 128: token i attends j with i-j < 128 (HF sliding mask).
  * MoE: fp32 router logits, sigmoid, top-8 chosen on
    ``sigmoid + e_score_correction_bias`` (n_group=topk_group=1), weights =
    un-biased sigmoid scores renormalised, times routed_scaling_factor (1.0).
    Expert outputs are accumulated in fp32 then cast back, as in the reference.
  * Layer 0 is dense (moe_layer_freq[0] == 0). No shared experts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "mimo_v2"
    vocab_size: int = 152576
    hidden_size: int = 4096
    intermediate_size: int = 16384
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 48
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    head_dim: int = 192
    v_head_dim: int = 128
    swa_num_attention_heads: int = 64
    swa_num_key_value_heads: int = 8
    swa_head_dim: int = 192
    swa_v_head_dim: int = 128
    layernorm_epsilon: float = 1e-6
    rope_theta: float = 10_000_000.0
    swa_rope_theta: float = 10_000.0
    partial_rotary_factor: float = 0.334
    sliding_window: int = 128
    attention_value_scale: Optional[float] = 0.707
    add_full_attention_sink_bias: bool = False
    add_swa_attention_sink_bias: bool = True
    hybrid_layer_pattern: list = field(default_factory=list)
    moe_layer_freq: list = field(default_factory=list)
    n_routed_experts: int = 256
    num_experts_per_tok: int = 8
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    routed_scaling_factor: Optional[float] = None
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"
    tie_word_embeddings: bool = False
    rope_parameters: Optional[dict] = None

    def __post_init__(self):
        if self.scoring_func != "sigmoid" or self.topk_method != "noaux_tc":
            raise ValueError(f"unsupported routing {self.scoring_func}/{self.topk_method}")
        if self.n_group != 1 or self.topk_group != 1:
            # Grouped routing is not needed by Flash; refuse rather than silently mis-route.
            raise ValueError(f"grouped routing n_group={self.n_group} not implemented")
        rp = self.rope_parameters or {}
        if "partial_rotary_factor" in rp:
            self.partial_rotary_factor = float(rp["partial_rotary_factor"])
        if "rope_theta" in rp:
            self.rope_theta = float(rp["rope_theta"])


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_swa = args.hybrid_layer_pattern[layer_idx] == 1
        if self.is_swa:
            self.n_heads = args.swa_num_attention_heads
            self.n_kv = args.swa_num_key_value_heads
            self.head_dim = args.swa_head_dim
            self.v_head_dim = args.swa_v_head_dim
            base = args.swa_rope_theta
            has_sink = args.add_swa_attention_sink_bias
        else:
            self.n_heads = args.num_attention_heads
            self.n_kv = args.num_key_value_heads
            self.head_dim = args.head_dim
            self.v_head_dim = args.v_head_dim
            base = args.rope_theta
            has_sink = args.add_full_attention_sink_bias
        self.rope_dim = int(self.head_dim * args.partial_rotary_factor)
        if self.rope_dim % 2:
            raise ValueError(f"odd rope_dim {self.rope_dim}")
        self.rope_base = float(base)
        self.scale = self.head_dim ** -0.5
        self.v_scale = args.attention_value_scale
        self.q_size = self.n_heads * self.head_dim
        self.k_size = self.n_kv * self.head_dim
        self.v_size = self.n_kv * self.v_head_dim
        self.qkv_proj = nn.Linear(args.hidden_size, self.q_size + self.k_size + self.v_size, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.v_head_dim, args.hidden_size, bias=False)
        if has_sink:
            self.attention_sink_bias = mx.zeros((self.n_heads,))
        self.has_sink = has_sink

    def _rope(self, x, offset):
        # mx.fast.rope with dims < head_dim rotates only the first `dims`
        # features (rotate-half pairing inside them) and passes the rest through.
        return mx.fast.rope(x, self.rope_dim, traditional=False, base=self.rope_base,
                            scale=1.0, offset=offset)

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = mx.split(qkv, [self.q_size, self.q_size + self.k_size], axis=-1)
        q = q.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv, self.v_head_dim).transpose(0, 2, 1, 3)
        if self.v_scale is not None:
            v = v * self.v_scale
        offset = cache.offset if cache is not None else 0
        q = self._rope(q, offset)
        k = self._rope(k, offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        sinks = self.attention_sink_bias.astype(q.dtype) if self.has_sink else None
        o = scaled_dot_product_attention(q, k, v, cache, self.scale, mask=mask, sinks=sinks)
        o = o.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(o)


class DenseMLP(nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Router(nn.Module):
    """Holds `weight` [E, H] and `e_score_correction_bias` [E]; computes in fp32."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.scaling = 1.0 if args.routed_scaling_factor is None else float(args.routed_scaling_factor)
        self.weight = mx.zeros((args.n_routed_experts, args.hidden_size))
        self.e_score_correction_bias = mx.zeros((args.n_routed_experts,))

    def __call__(self, x):
        logits = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
        scores = mx.sigmoid(logits)
        choice = scores + self.e_score_correction_bias.astype(mx.float32)
        idx = mx.stop_gradient(mx.argpartition(-choice, kth=self.top_k - 1, axis=-1)[..., : self.top_k])
        w = mx.take_along_axis(scores, idx, axis=-1)
        if self.top_k > 1 and self.norm_topk_prob:
            w = w / (w.sum(axis=-1, keepdims=True) + 1e-20)
        return idx, w * self.scaling


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate = Router(args)
        self.switch_mlp = SwitchGLU(args.hidden_size, args.moe_intermediate_size, args.n_routed_experts, bias=False)

    def __call__(self, x):
        idx, w = self.gate(x)
        y = self.switch_mlp(x, idx)  # [..., K, H]
        y = (y.astype(mx.float32) * w[..., None]).sum(axis=-2)
        return y.astype(x.dtype)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_swa = args.hybrid_layer_pattern[layer_idx] == 1
        self.self_attn = Attention(args, layer_idx)
        self.mlp = MoE(args) if args.moe_layer_freq[layer_idx] else DenseMLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.layernorm_epsilon)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.layernorm_epsilon)

    def __call__(self, x, mask=None, cache=None):
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class MiMoV2Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.layernorm_epsilon)
        pat = args.hybrid_layer_pattern
        self.swa_idx = pat.index(1) if 1 in pat else None
        self.full_idx = pat.index(0) if 0 in pat else None

    def __call__(self, inputs, cache=None, input_embeddings=None):
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        full_mask = create_attention_mask(h, cache[self.full_idx]) if self.full_idx is not None else None
        swa_mask = (create_attention_mask(h, cache[self.swa_idx], window_size=self.args.sliding_window)
                    if self.swa_idx is not None else None)
        for layer, c in zip(self.layers, cache):
            h = layer(h, swa_mask if layer.is_swa else full_mask, c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MiMoV2Model(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        return self.lm_head(self.model(inputs, cache, input_embeddings))

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            RotatingKVCache(max_size=self.args.sliding_window, keep=0) if layer.is_swa else KVCache()
            for layer in self.model.layers
        ]

    def sanitize(self, weights: dict) -> dict:
        """Accept a JANG MiMo-V2.6 bundle (pre-stacked switch_mlp, de-interleaved qkv).

        Drops tensors this text backbone does not own (MTP, vision, audio);
        those are served by their own modules. Fails loudly on the per-expert
        V2.5 layout instead of silently re-stacking it.
        """
        out = {}
        for k, v in weights.items():
            if k.startswith(("model.mtp.", "visual.", "audio_encoder.", "speech_embeddings.")):
                continue
            if ".mlp.experts." in k:
                raise ValueError(
                    f"per-expert tensor {k}: the V2.6 bundle format stores routed experts "
                    "pre-stacked under mlp.switch_mlp.*"
                )
            out[k] = v
        return out
