"""Naive-N0.5-Flash (model_type naive_n05_flash) in MLX — reference implementation for calibration, KL and the
JANGH runtime port.

Mirrors NaiveAI's modeling_naive_n05_flash.py (rev 0235b3b) exactly:
  * 48 layers: hybrid_layer_pattern 1 = SWA (window 128, 64 q heads / 8 kv heads, rope theta 1e4, sink bias),
    0 = DSA (64 q / 4 kv heads, rope theta 1e7, no sink) with a lightweight indexer selecting the top-2048 keys.
  * head_dim 192 (q/k), v_head_dim 128; GPT-NeoX (non-interleaved) rotary on the first int(192*0.334)=64 dims.
  * values pre-scaled by attention_value_scale (0.707); logits scaled by head_dim^-0.5; sink = extra softmax column.
  * Indexer: q = wq(x) (16 heads x 128), k = LayerNorm(wk(x)) (1 head), same rotary as the layer, per-row FP8-e4m3
    round-trip of q and k, score = sum_h relu(q_h.k) * weights_proj(x)_h * 16^-0.5.
  * MoE: DeepSeek-V3 router (fp32 sigmoid + e_score_correction_bias for selection, normalized top-8 weights,
    scaling 1.0), 256 SwiGLU experts, NO shared expert. Layer 0 is a dense SwiGLU MLP (16384).
  * RMSNorm eps 1e-5 (no +1 shift), untied lm_head.
Tensor names after sanitize() are the runtime names: experts stacked to mlp.switch_mlp.{gate,up,down}_proj.weight.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import SwitchGLU


@dataclass
class Args:
    hidden_size: int = 4096
    num_hidden_layers: int = 48
    vocab_size: int = 152576
    intermediate_size: int = 16384
    moe_intermediate_size: int = 2048
    n_routed_experts: int = 256
    num_experts_per_tok: int = 8
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    head_dim: int = 192
    v_head_dim: int = 128
    rope_theta: float = 1e7
    swa_num_attention_heads: int = 64
    swa_num_key_value_heads: int = 8
    swa_head_dim: int = 192
    swa_v_head_dim: int = 128
    swa_rope_theta: float = 1e4
    partial_rotary_factor: float = 0.334
    sliding_window: int = 128
    attention_value_scale: float | None = 0.707
    add_swa_attention_sink_bias: bool = True
    add_full_attention_sink_bias: bool = False
    layernorm_epsilon: float = 1e-5
    index_top_k: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 16
    indexer_activation_dtype: str = "fp8_e4m3"
    hybrid_layer_pattern: list = field(default_factory=list)
    moe_layer_freq: list = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: dict) -> "Args":
        kw = {k: cfg[k] for k in cls.__dataclass_fields__ if k in cfg and cfg[k] is not None}
        a = cls(**kw)
        assert len(a.hybrid_layer_pattern) == a.num_hidden_layers == len(a.moe_layer_freq)
        assert cfg.get("scoring_func", "sigmoid") == "sigmoid" and cfg.get("n_group", 1) == 1
        assert cfg.get("attention_projection_layout", "split") == "split" and cfg.get("index_n_kv_heads", 1) == 1
        assert cfg.get("n_shared_experts") in (None, 0), "reference has no shared expert"
        return a


def fp8_round(t: mx.array) -> mx.array:
    """round_indexer_fp8: per-row (last dim) absmax/448 scale, e4m3 round trip, fp32 out."""
    t = t.astype(mx.float32)
    s = mx.maximum(mx.max(mx.abs(t), axis=-1, keepdims=True), 1e-4) / 448.0
    return mx.from_fp8(mx.to_fp8(mx.clip(t / s, -448.0, 448.0)), dtype=mx.float32) * s



# ---------------------------------------------------------------------------------------------------------------
# MLX 0.32.2 BUG (found 2026-09-27): mx.fast.scaled_dot_product_attention with `sinks=` is WRONG and NON-DETERMINISTIC
# once the score matrix exceeds 8 GiB (heads * L * L_keys * 4 bytes), i.e. when MLX takes its blocked SDPA path:
# 4-9% relative error, most query rows affected, results differ call to call. Without sinks the blocked path is
# correct. Every sliding-window layer of this model uses the sink bias, so the sliding-window attention is computed
# explicitly here (reference semantics: bf16 logits, fp32 softmax with the sink as an extra column, bf16 p @ v).
SDPA_SINKS_SAFE_BYTES = 2 * 2**30


def sink_attention(q, k, v, scale, sinks, q_pos0: int, k_pos0: int, window: int | None, blk: int = 1024):
    """q (B,H,L,D) at absolute positions q_pos0.., k/v (B,KV,Lk,.) at absolute positions k_pos0..  Causal; with
    `window` a query at position p sees keys in (p - window, p]. sinks (H,) or None. Returns (B,H,L,Dv) in v.dtype."""
    B, Hh, L, D = q.shape
    KV, Lk = k.shape[1], k.shape[2]
    g = Hh // KV
    qg = q.reshape(B, KV, g, L, D)
    outs = []
    for s in range(0, L, blk):
        e = min(L, s + blk)
        lo = 0 if window is None else max(0, (q_pos0 + s) - (window - 1) - k_pos0)
        hi = min(Lk, (q_pos0 + e - 1) - k_pos0 + 1)
        ks = k[:, :, None, lo:hi]                                             # (B,KV,1,n,D)
        lg = (qg[:, :, :, s:e] @ ks.swapaxes(-1, -2)) * scale                 # (B,KV,g,b,n) in q.dtype
        qp = mx.arange(q_pos0 + s, q_pos0 + e)[:, None]; kp = mx.arange(k_pos0 + lo, k_pos0 + hi)[None, :]
        ok = qp >= kp
        if window is not None:
            ok = ok & (qp - kp < window)
        lg = mx.where(ok, lg.astype(mx.float32), -mx.inf)
        if sinks is not None:
            sk = mx.broadcast_to(sinks.astype(mx.float32).reshape(1, KV, g, 1, 1), lg.shape[:-1] + (1,))
            p = mx.softmax(mx.concatenate([lg, sk], axis=-1), axis=-1)[..., :-1]
        else:
            p = mx.softmax(lg, axis=-1)
        o = p.astype(v.dtype) @ v[:, :, None, lo:hi]                           # (B,KV,g,b,Dv)
        outs.append(o)
    o = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=3)
    return o.reshape(B, Hh, L, -1)


class LayerCache:
    """Minimal full-history cache (keys/values + indexer keys) for correctness tests and generation checks.
    The production runtime uses a rotating 128-token window for SWA layers; semantics are identical because the
    mask below is computed from absolute positions."""

    def __init__(self):
        self.k = self.v = self.ik = None
        self.offset = 0

    def update(self, k, v, ik=None):
        cat = lambda a, b: b if a is None else mx.concatenate([a, b], axis=2)
        self.k, self.v = cat(self.k, k), cat(self.v, v)
        if ik is not None:
            self.ik = cat(self.ik, ik)
        self.offset += k.shape[2]
        return self.k, self.v, self.ik


class Indexer(nn.Module):
    def __init__(self, a: Args):
        super().__init__()
        self.a = a
        self.wq = nn.Linear(a.hidden_size, a.index_n_heads * a.index_head_dim, bias=False)
        self.wk = nn.Linear(a.hidden_size, a.index_head_dim, bias=False)
        self.k_norm = nn.LayerNorm(a.index_head_dim, eps=1e-5)
        self.weights_proj = nn.Linear(a.hidden_size, a.index_n_heads, bias=False)

    def keys(self, x, rope):
        return rope(self.k_norm(self.wk(x))[:, None])                       # (B, 1, L, 128)

    def scores(self, x, rope, keys):
        a = self.a
        B, L, _ = x.shape
        q = rope(self.wq(x).reshape(B, L, a.index_n_heads, a.index_head_dim).transpose(0, 2, 1, 3))
        k = keys
        if a.indexer_activation_dtype == "fp8_e4m3":
            q, k = fp8_round(q), fp8_round(k)
        else:
            q, k = q.astype(mx.float32), k.astype(mx.float32)
        w = (self.weights_proj(x) * (a.index_n_heads ** -0.5)).astype(mx.float32)   # bf16 product, then fp32
        s = mx.maximum(q @ k.transpose(0, 1, 3, 2), 0.0)                      # (B, H, L, Lk)
        return mx.sum(s * w.transpose(0, 2, 1)[..., None], axis=1)           # (B, L, Lk)


class Attention(nn.Module):
    def __init__(self, a: Args, layer_idx: int):
        super().__init__()
        self.a = a
        self.is_swa = bool(a.hybrid_layer_pattern[layer_idx])
        p = "swa_" if self.is_swa else ""
        self.heads = getattr(a, p + "num_attention_heads")
        self.kv_heads = getattr(a, p + "num_key_value_heads")
        self.head_dim = getattr(a, p + "head_dim")
        self.v_dim = getattr(a, p + "v_head_dim")
        self.theta = a.swa_rope_theta if self.is_swa else a.rope_theta
        self.rot_dims = int(self.head_dim * a.partial_rotary_factor)
        self.q_proj = nn.Linear(a.hidden_size, self.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(a.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(a.hidden_size, self.kv_heads * self.v_dim, bias=False)
        self.o_proj = nn.Linear(self.heads * self.v_dim, a.hidden_size, bias=False)
        self.indexer = None if self.is_swa else Indexer(a)
        sink = a.add_swa_attention_sink_bias if self.is_swa else a.add_full_attention_sink_bias
        if sink:
            self.attention_sink_bias = mx.zeros((self.heads,))
        self.has_sink = sink

    def __call__(self, x, cache: LayerCache | None = None):
        a = self.a
        B, L, _ = x.shape
        off = cache.offset if cache is not None else 0
        rope = lambda t: mx.fast.rope(t, self.rot_dims, traditional=False, base=self.theta, scale=1.0, offset=off)
        q = rope(self.q_proj(x).reshape(B, L, self.heads, self.head_dim).transpose(0, 2, 1, 3))
        k = rope(self.k_proj(x).reshape(B, L, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3))
        v = self.v_proj(x).reshape(B, L, self.kv_heads, self.v_dim).transpose(0, 2, 1, 3)
        ik = self.indexer.keys(x, rope) if self.indexer is not None else None
        if cache is not None:
            k, v, ik = cache.update(k, v, ik)
        Lk = k.shape[2]
        qpos = mx.arange(off, off + L)[:, None]
        kpos = mx.arange(Lk)[None, :]
        allowed = qpos >= kpos
        if self.is_swa:
            allowed = allowed & (qpos - kpos < a.sliding_window)
        elif Lk > a.index_top_k:
            sc = self.indexer.scores(x, rope, ik)                             # (B, L, Lk)
            sc = mx.where(allowed[None], sc, -mx.inf)
            top = mx.argpartition(-sc, kth=a.index_top_k - 1, axis=-1)[..., : a.index_top_k]
            sel = mx.put_along_axis(mx.zeros(sc.shape, dtype=mx.bool_), top, mx.array(True), axis=-1)
            allowed = allowed[None] & sel                                      # (B, L, Lk)
            allowed = allowed[:, None]                                         # (B, 1, L, Lk)
        # Lk <= index_top_k: every causal key is selected -> DSA == causal attention (exact, indexer unused)
        if a.attention_value_scale is not None:
            v = v * a.attention_value_scale
        if self.is_swa or self.has_sink:
            # explicit attention (see the MLX sinks bug note above); keys are limited to the window
            o = sink_attention(q, k, v, self.head_dim ** -0.5, self.attention_sink_bias if self.has_sink else None,
                               off, 0, a.sliding_window if self.is_swa else None)
        else:
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim ** -0.5, mask=allowed)
        return self.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1))


class MLP(nn.Module):
    def __init__(self, d, i):
        super().__init__()
        self.gate_proj = nn.Linear(d, i, bias=False)
        self.up_proj = nn.Linear(d, i, bias=False)
        self.down_proj = nn.Linear(i, d, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Gate(nn.Module):
    def __init__(self, a: Args):
        super().__init__()
        self.weight = mx.zeros((a.n_routed_experts, a.hidden_size), dtype=mx.float32)
        self.e_score_correction_bias = mx.zeros((a.n_routed_experts,), dtype=mx.float32)


def route(gate: Gate, x, k: int, norm: bool, scaling: float):
    logits = x.astype(mx.float32) @ gate.weight.astype(mx.float32).T
    scores = mx.sigmoid(logits)
    choice = scores + gate.e_score_correction_bias.astype(mx.float32)
    idx = mx.argpartition(-choice, kth=k - 1, axis=-1)[..., :k]
    w = mx.take_along_axis(scores, idx, axis=-1)
    if norm:
        w = w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20)
    return idx, w * scaling


class MoE(nn.Module):
    def __init__(self, a: Args):
        super().__init__()
        self.a = a
        self.gate = Gate(a)
        self.switch_mlp = SwitchGLU(a.hidden_size, a.moe_intermediate_size, a.n_routed_experts)

    def __call__(self, x):
        a = self.a
        idx, w = route(self.gate, x, a.num_experts_per_tok, a.norm_topk_prob, a.routed_scaling_factor)
        if getattr(self.switch_mlp, "is_jangtq2", False):
            return self.switch_mlp.routed(x, idx, w).astype(x.dtype)
        y = self.switch_mlp(x, idx)                                            # (B, L, k, D)
        return mx.sum(y * w[..., None].astype(y.dtype), axis=-2).astype(x.dtype)


class DecoderLayer(nn.Module):
    def __init__(self, a: Args, i: int):
        super().__init__()
        self.self_attn = Attention(a, i)
        self.mlp = MoE(a) if a.moe_layer_freq[i] else MLP(a.hidden_size, a.intermediate_size)
        self.input_layernorm = nn.RMSNorm(a.hidden_size, eps=a.layernorm_epsilon)
        self.post_attention_layernorm = nn.RMSNorm(a.hidden_size, eps=a.layernorm_epsilon)

    def __call__(self, x, cache=None):
        x = x + self.self_attn(self.input_layernorm(x), cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class Inner(nn.Module):
    def __init__(self, a: Args):
        super().__init__()
        self.embed_tokens = nn.Embedding(a.vocab_size, a.hidden_size)
        self.layers = [DecoderLayer(a, i) for i in range(a.num_hidden_layers)]
        self.norm = nn.RMSNorm(a.hidden_size, eps=a.layernorm_epsilon)


class Model(nn.Module):
    def __init__(self, a: Args):
        super().__init__()
        self.args = a
        self.model = Inner(a)
        self.lm_head = nn.Linear(a.hidden_size, a.vocab_size, bias=False)

    def __call__(self, ids, cache=None):
        h = self.model.embed_tokens(ids)
        for i, layer in enumerate(self.model.layers):
            h = layer(h, None if cache is None else cache[i])
        return self.lm_head(self.model.norm(h))

    def make_cache(self):
        return [LayerCache() for _ in self.model.layers]


FP32_SUFFIXES = ("mlp.gate.weight", "mlp.gate.e_score_correction_bias")


def sanitize_layer(raw: dict, i: int, n_experts: int) -> dict:
    """Source tensors of layer i (full names) -> layer-relative runtime names; experts stacked."""
    pre = f"model.layers.{i}."
    out = {}
    for name in ("gate_proj", "up_proj", "down_proj"):
        ks = [f"{pre}mlp.experts.{e}.{name}.weight" for e in range(n_experts)]
        if ks[0] in raw:
            out[f"mlp.switch_mlp.{name}.weight"] = mx.stack([raw.pop(k) for k in ks])
    for k, v in raw.items():
        assert k.startswith(pre), k
        out[k[len(pre):]] = v
    return out
