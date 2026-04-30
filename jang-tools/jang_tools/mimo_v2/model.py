"""MiMoV2 — MLX port of modeling_mimo_v2.py.

Single-node reference. Distributed sharding lives in
`jang_tools.distributed.mimo_v2_dist` and wraps this module.

Key differences vs upstream PyTorch:
- `mx.fast.rms_norm` instead of manual variance op
- `mx.fast.rope` with WAVELENGTH (= rope_theta) convention, NOT frequency
- Two RoPE objects: full (rope_theta=10M) + SWA (rope_theta=10K)
- Asymmetric Q/V head_dim → V is V-only, Q/K share head_dim=192 with
  partial rotary on first 64 dims (`rope_dim = head_dim * 0.334`)
- Sink bias is a learnable per-head scalar concatenated into attn logits
  before softmax then sliced off after — matches upstream eager path
- All `o_proj` layers stay bf16 (kept out of any quant schedule)
- Routing: sigmoid + noaux_tc with topk over a single group
- MTP head loaded from a sibling `model_mtp.safetensors`
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import MiMoV2Config


# --- Norms / RoPE ----------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class RoPE:
    """Partial-rotary RoPE with split-theta full/SWA paths.

    rope_dim = int(head_dim * partial_rotary_factor); only the first rope_dim
    of each head are rotated. mx.fast.rope expects WAVELENGTH (rope_theta).
    """

    def __init__(self, head_dim: int, rope_dim: int, base: float, max_pos: int):
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.base = float(base)
        self.max_pos = max_pos

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        # x: (B, H, T, head_dim)
        if self.rope_dim == self.head_dim:
            return mx.fast.rope(x, self.head_dim, traditional=False,
                                base=self.base, scale=1.0, offset=offset)
        # Partial rotary: split last dim, rotate first rope_dim, concat back
        rot = x[..., : self.rope_dim]
        keep = x[..., self.rope_dim :]
        rot = mx.fast.rope(rot, self.rope_dim, traditional=False,
                           base=self.base, scale=1.0, offset=offset)
        return mx.concatenate([rot, keep], axis=-1)


# --- Attention -------------------------------------------------------------


def _causal_mask(T: int, dtype) -> mx.array:
    return mx.triu(mx.full((T, T), -mx.inf, dtype=dtype), k=1)


def _swa_mask(T: int, window: int, dtype, offset: int = 0) -> mx.array:
    """Sliding-window causal mask of size T x (T+offset)."""
    full_T = T + offset
    pos_q = mx.arange(offset, offset + T)[:, None]
    pos_k = mx.arange(full_T)[None, :]
    delta = pos_q - pos_k
    keep = (delta >= 0) & (delta < window)
    return mx.where(keep, mx.zeros_like(delta, dtype=dtype),
                    mx.full_like(delta, -mx.inf, dtype=dtype))


class MiMoV2Attention(nn.Module):
    def __init__(self, config: MiMoV2Config, layer_idx: int, is_swa: bool):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_swa = is_swa

        self.head_dim = config.swa_head_dim if is_swa else config.head_dim
        self.v_head_dim = config.swa_v_head_dim if is_swa else config.v_head_dim
        self.n_heads = (config.swa_num_attention_heads if is_swa
                        else config.num_attention_heads)
        self.n_kv_heads = (config.swa_num_key_value_heads if is_swa
                           else config.num_key_value_heads)
        self.n_kv_groups = self.n_heads // self.n_kv_heads
        self.rope_dim = int(self.head_dim * config.partial_rotary_factor)
        if self.rope_dim % 2 != 0:
            raise ValueError(f"rope_dim must be even, got {self.rope_dim}")
        self.scaling = self.head_dim ** -0.5
        self.v_scale = config.attention_value_scale
        self.window = config.sliding_window if is_swa else None

        self.q_size = self.n_heads * self.head_dim
        self.k_size = self.n_kv_heads * self.head_dim
        self.v_size = self.n_kv_heads * self.v_head_dim
        self.o_in = self.n_heads * self.v_head_dim

        self.layout = config.attention_projection_layout
        bias = config.attention_bias
        if self.layout == "fused_qkv":
            self.qkv_proj = nn.Linear(config.hidden_size,
                                      self.q_size + self.k_size + self.v_size,
                                      bias=bias)
        else:
            self.q_proj = nn.Linear(config.hidden_size, self.q_size, bias=bias)
            self.k_proj = nn.Linear(config.hidden_size, self.k_size, bias=bias)
            self.v_proj = nn.Linear(config.hidden_size, self.v_size, bias=bias)
        self.o_proj = nn.Linear(self.o_in, config.hidden_size, bias=False)

        # Sink bias (per-head learned scalar); only present per upstream config
        has_sink = (
            (config.add_full_attention_sink_bias and not is_swa)
            or (config.add_swa_attention_sink_bias and is_swa)
        )
        self.has_sink = has_sink
        if has_sink:
            self.attention_sink_bias = mx.zeros((self.n_heads,))

        base = config.swa_rope_theta if is_swa else config.rope_theta
        self.rope = RoPE(self.head_dim, self.rope_dim, base,
                         config.max_position_embeddings)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[tuple] = None,
    ) -> tuple[mx.array, tuple]:
        B, T, _ = x.shape
        if self.layout == "fused_qkv":
            qkv = self.qkv_proj(x)
            q, k, v = mx.split(qkv, [self.q_size, self.q_size + self.k_size], axis=-1)
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_kv_heads, self.v_head_dim).transpose(0, 2, 1, 3)

        if self.v_scale is not None:
            v = v * self.v_scale

        offset = 0 if cache is None else cache[0].shape[2]
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        new_cache = (k, v)

        # GQA expand
        if self.n_kv_groups > 1:
            k = mx.repeat(k, self.n_kv_groups, axis=1)
            v = mx.repeat(v, self.n_kv_groups, axis=1)

        if self.has_sink:
            # Manual SDPA so we can fold the sink bias into the softmax denom
            scores = (q @ k.swapaxes(-1, -2)) * self.scaling
            if mask is not None:
                scores = scores + mask
            sink = mx.broadcast_to(
                self.attention_sink_bias.reshape(1, self.n_heads, 1, 1),
                (B, self.n_heads, scores.shape[2], 1),
            )
            scores = mx.concatenate([scores, sink], axis=-1)
            scores = scores - mx.max(scores, axis=-1, keepdims=True)
            probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
            probs = probs[..., :-1]
            out = probs @ v
        else:
            out = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.scaling, mask=mask
            )

        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.o_in)
        return self.o_proj(out), new_cache


# --- MoE -------------------------------------------------------------------


class DenseMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEGate(nn.Module):
    """sigmoid + noaux_tc routing over a single group (n_group=1, topk_group=1).

    Simplified vs upstream because n_group=1 ⇒ group selection is a no-op.
    """

    def __init__(self, config: MiMoV2Config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_routed = config.n_routed_experts
        self.scale = config.routed_scaling_factor or 1.0
        self.norm = config.norm_topk_prob
        self.weight = mx.zeros((self.n_routed, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((self.n_routed,))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        B, T, H = x.shape
        flat = x.reshape(-1, H).astype(mx.float32)
        logits = flat @ self.weight.astype(mx.float32).T
        scores = mx.sigmoid(logits)
        scores_for_choice = scores + self.e_score_correction_bias[None, :]
        # noaux_tc with n_group=1: skip group selection (group 0 always chosen)
        topk_idx = mx.argsort(-scores_for_choice, axis=-1)[:, : self.top_k]
        topk_w = mx.take_along_axis(scores, topk_idx, axis=-1)
        if self.top_k > 1 and self.norm:
            denom = mx.sum(topk_w, axis=-1, keepdims=True) + 1e-20
            topk_w = topk_w / denom
        topk_w = topk_w * self.scale
        return topk_idx.astype(mx.uint32), topk_w


class MiMoV2MoE(nn.Module):
    """MoE block. Reference (slow) implementation; production swaps in
    fused expert kernels from `jang_tools.turboquant.gather_tq_kernel` /
    `fused_gate_up_kernel` once weights are quantized."""

    def __init__(self, config: MiMoV2Config):
        super().__init__()
        self.config = config
        self.gate = MoEGate(config)
        self.experts = [DenseMLP(config.hidden_size, config.moe_intermediate_size)
                        for _ in range(config.n_routed_experts)]

    def __call__(self, x: mx.array) -> mx.array:
        B, T, H = x.shape
        topk_idx, topk_w = self.gate(x)  # (B*T, K), (B*T, K)
        flat = x.reshape(-1, H)
        out = mx.zeros_like(flat, dtype=topk_w.dtype)
        # Reference scatter: iterate experts. Production kernel replaces this.
        for e, expert in enumerate(self.experts):
            mask = (topk_idx == e)  # (B*T, K)
            row_has = mx.any(mask, axis=-1)
            if not bool(mx.any(row_has).item()):
                continue
            rows = mx.where(row_has)[0]
            sel = mx.take(flat, rows, axis=0)
            y = expert(sel)
            w = mx.sum(mx.where(mask, topk_w, mx.zeros_like(topk_w)), axis=-1)
            w_sel = mx.take(w, rows, axis=0)[:, None]
            out = out.at[rows].add(y * w_sel)
        return out.reshape(B, T, H).astype(x.dtype)


# --- Decoder layer ---------------------------------------------------------


class DecoderLayer(nn.Module):
    def __init__(self, config: MiMoV2Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        is_swa = config.hybrid_layer_pattern[layer_idx] == 1
        self.is_swa = is_swa
        self.self_attn = MiMoV2Attention(config, layer_idx, is_swa)
        is_moe = (config.n_routed_experts is not None
                  and config.moe_layer_freq[layer_idx])
        self.mlp = (MiMoV2MoE(config) if is_moe
                    else DenseMLP(config.hidden_size, config.intermediate_size))
        self.input_layernorm = RMSNorm(config.hidden_size, config.layernorm_epsilon)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                config.layernorm_epsilon)

    def __call__(self, x, mask=None, cache=None):
        h, new_cache = self.self_attn(self.input_layernorm(x), mask, cache)
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_cache


# --- Top-level model -------------------------------------------------------


class MiMoV2Model(nn.Module):
    def __init__(self, config: MiMoV2Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = RMSNorm(config.hidden_size, config.layernorm_epsilon)

    def __call__(self, input_ids: mx.array, caches=None) -> mx.array:
        h = self.embed_tokens(input_ids)
        T = h.shape[1]
        full_mask = _causal_mask(T, h.dtype) if T > 1 else None
        swa_mask = _swa_mask(T, self.config.sliding_window, h.dtype) if T > 1 else None

        if caches is None:
            caches = [None] * len(self.layers)
        new_caches = []
        for i, layer in enumerate(self.layers):
            mask = swa_mask if layer.is_swa else full_mask
            h, c = layer(h, mask, caches[i])
            new_caches.append(c)
        return self.norm(h), new_caches


class MiMoV2ForCausalLM(nn.Module):
    def __init__(self, config: MiMoV2Config):
        super().__init__()
        self.config = config
        self.model = MiMoV2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, input_ids: mx.array, caches=None):
        h, caches = self.model(input_ids, caches)
        return self.lm_head(h), caches
