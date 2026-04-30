"""Laguna MLX port — hybrid SWA+full with PER-LAYER head count + dual RoPE.

Layer L attention:
    n_heads = cfg.num_attention_heads_per_layer[L]   (48 or 64)
    n_kv    = cfg.num_key_value_heads                (8 always)
    rope    = cfg.rope_parameters[ cfg.layer_types[L] ]
              full_attention -> YaRN base 500K factor 32 partial 0.5
              sliding_attention -> default base 10K partial 1.0

Layer L MLP:
    cfg.mlp_layer_types[L] == "dense"  -> SwiGLU intermediate=8192 (layer 0)
    cfg.mlp_layer_types[L] == "sparse" -> 256 routed top-8 + 1 shared,
                                          expert dim 512, sigmoid + per-head gating
"""
from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..jangrt.loader import register_arch
from .config import LagunaConfig


class RMSNorm(nn.Module):
    def __init__(self, h, eps): super().__init__(); self.weight = mx.ones((h,)); self.eps = eps
    def __call__(self, x): return mx.fast.rms_norm(x, self.weight, self.eps)


def _swa_mask(T, win, dtype, offset=0):
    full = T + offset
    pq = mx.arange(offset, offset + T)[:, None]
    pk = mx.arange(full)[None, :]
    d = pq - pk
    keep = (d >= 0) & (d < win)
    return mx.where(keep, mx.zeros_like(d, dtype=dtype),
                    mx.full_like(d, -mx.inf, dtype=dtype))


class LagunaAttention(nn.Module):
    def __init__(self, cfg: LagunaConfig, L: int):
        super().__init__()
        self.layer_idx = L
        self.n_heads = cfg.num_attention_heads_per_layer[L]
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.kv_groups = self.n_heads // self.n_kv
        self.scale = self.head_dim ** -0.5
        self.layer_type = cfg.layer_types[L]
        rp = cfg.rope_parameters[self.layer_type]
        self.rope_base = rp["rope_theta"]
        self.partial = rp.get("partial_rotary_factor", 1.0)
        self.rope_dim = int(self.head_dim * self.partial)
        self.window = cfg.sliding_window if self.layer_type == "sliding_attention" else None

        h = cfg.hidden_size
        self.q_proj = nn.Linear(h, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(h, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(h, self.n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, h, bias=False)
        if cfg.gating:
            self.g_proj = nn.Linear(h, self.n_heads, bias=False)
        else:
            self.g_proj = None

    def _rope(self, t: mx.array, offset: int) -> mx.array:
        if self.rope_dim == self.head_dim:
            return mx.fast.rope(t, self.head_dim, traditional=False,
                                base=self.rope_base, scale=1.0, offset=offset)
        rot = t[..., :self.rope_dim]
        keep = t[..., self.rope_dim:]
        rot = mx.fast.rope(rot, self.rope_dim, traditional=False,
                           base=self.rope_base, scale=1.0, offset=offset)
        return mx.concatenate([rot, keep], axis=-1)

    def __call__(self, x, mask, cache, offset):
        B, T, H = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.n_kv, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.n_kv, self.head_dim).transpose(0, 2, 1, 3)
        q = self._rope(q, offset); k = self._rope(k, offset)
        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        new = (k, v)
        if self.kv_groups > 1:
            k = mx.repeat(k, self.kv_groups, axis=1)
            v = mx.repeat(v, self.kv_groups, axis=1)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.n_heads * self.head_dim)
        if self.g_proj is not None:
            gate = mx.sigmoid(self.g_proj(x))                          # (B,T,H_heads)
            out = out.reshape(B, T, self.n_heads, self.head_dim) * gate[..., None]
            out = out.reshape(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out), new


class DenseMLP(nn.Module):
    def __init__(self, h, inter):
        super().__init__()
        self.gate_proj = nn.Linear(h, inter, bias=False)
        self.up_proj = nn.Linear(h, inter, bias=False)
        self.down_proj = nn.Linear(inter, h, bias=False)
    def __call__(self, x): return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class LagunaMoE(nn.Module):
    def __init__(self, cfg: LagunaConfig):
        super().__init__()
        self.cfg = cfg
        self.gate = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)
        self.experts = [DenseMLP(cfg.hidden_size, cfg.moe_intermediate_size)
                        for _ in range(cfg.num_experts)]
        self.shared_expert = DenseMLP(cfg.hidden_size, cfg.shared_expert_intermediate_size)

    def __call__(self, x):
        B, T, H = x.shape
        flat = x.reshape(-1, H)
        logits = self.gate(flat).astype(mx.float32)
        scores = mx.sigmoid(logits)
        topk_idx = mx.argsort(-scores, axis=-1)[:, : self.cfg.num_experts_per_tok]
        topk_w = mx.take_along_axis(scores, topk_idx, axis=-1)
        topk_w = topk_w / (mx.sum(topk_w, axis=-1, keepdims=True) + 1e-20)
        topk_w = topk_w * self.cfg.moe_routed_scaling_factor

        out = mx.zeros_like(flat, dtype=topk_w.dtype)
        for e, expert in enumerate(self.experts):
            mask = (topk_idx == e)
            row_has = mx.any(mask, axis=-1)
            if not bool(mx.any(row_has).item()): continue
            rows = mx.where(row_has)[0]
            sel = mx.take(flat, rows, axis=0)
            y = expert(sel)
            w = mx.sum(mx.where(mask, topk_w, mx.zeros_like(topk_w)), axis=-1)
            w_sel = mx.take(w, rows, axis=0)[:, None]
            out = out.at[rows].add(y * w_sel)
        out = out + self.shared_expert(flat)
        return out.reshape(B, T, H).astype(x.dtype)


class LagunaLayer(nn.Module):
    def __init__(self, cfg: LagunaConfig, L: int):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = LagunaAttention(cfg, L)
        self.mlp = (DenseMLP(cfg.hidden_size, cfg.intermediate_size)
                    if cfg.mlp_layer_types[L] == "dense"
                    else LagunaMoE(cfg))
        self.layer_type = cfg.layer_types[L]

    def __call__(self, x, full_mask, swa_mask, cache, offset):
        mask = swa_mask if self.layer_type == "sliding_attention" else full_mask
        h, c = self.self_attn(self.input_layernorm(x), mask, cache, offset)
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, c


class LagunaForCausalLM(nn.Module):
    def __init__(self, cfg: LagunaConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [LagunaLayer(cfg, i) for i in range(cfg.num_hidden_layers)]
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def __call__(self, ids, caches=None):
        h = self.embed_tokens(ids)
        T = h.shape[1]
        offset = 0 if caches is None or caches[0] is None else caches[0][0].shape[2]
        full_mask = (mx.triu(mx.full((T, T), -mx.inf, dtype=h.dtype), k=1) if T > 1 else None)
        swa_mask = (_swa_mask(T, self.cfg.sliding_window, h.dtype, offset) if T > 1 else None)
        if caches is None: caches = [None] * len(self.layers)
        new = []
        for L, layer in enumerate(self.layers):
            h, c = layer(h, full_mask, swa_mask, caches[L], offset)
            new.append(c)
        return self.lm_head(self.norm(h)), new


@register_arch("laguna")
def _build(src: str, *, fmt: str, plan=None):
    cfg = LagunaConfig.from_json(f"{src}/config.json")
    return LagunaForCausalLM(cfg), cfg, None
