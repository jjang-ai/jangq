"""Laguna model — clean port matching mlx_lm conventions.

Architectural points (HF Laguna source):
- 40 layers; per-layer head count (full=48, SWA=64) read from config
- Hybrid attention: layer_types[i] ∈ {full_attention, sliding_attention}
- Dual RoPE: full uses YaRN scaling; SWA uses default. Both partial-rotary
  rotate `head_dim * partial_rotary_factor` dims.
- per-head q_norm + k_norm RMSNorm (head_dim)
- per-head g_proj sigmoid gate over the attention output (when cfg.gating)
- Layer 0 = dense MLP; layers 1..39 = sparse MoE
- MoE: 256 routed experts top-8 + 1 shared, sigmoid+bias top-k routing
  (DeepSeek-V3 / Qwen3.5-MoE recipe), routed scaling factor 2.5
- Token-embedding norm: standard pre-attention + pre-MLP RMSNorm

Cache convention: mlx_lm.KVCache for full attention layers, RotatingKVCache
for SWA layers. Both expose `.update_and_fetch(k, v)` and `.offset`.

Weight bridge to SwitchGLU: HF source stores experts as
`model.layers.N.mlp.experts.E.{gate,up,down}_proj.weight` (per-expert);
SwitchGLU needs them packed as `mlp.switch_mlp.{gate,up,down}_proj.weight`
shaped (num_experts, out, in). Packing happens in runtime.py before
model.update().
"""
from __future__ import annotations

import math
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .config import LagunaConfig

# SwitchGLU lives in mlx_lm.models.switch_layers (vendored across all MoE
# models we support). Falling back to a tiny copy isn't worth the size.
from mlx_lm.models.switch_layers import SwitchGLU
from mlx_lm.models.rope_utils import initialize_rope


# ─────────────────────────────────────────────────────────────────────────
# Compiled router math — keyed on top_k so different MoE blocks share the
# graph. The free-function form (no closures over `self`) plus shapeless
# tracing means prefill (T tokens) and decode (1 token) reuse the same
# compiled graph. mlx_lm uses the same pattern in
# `_hydrate_jangtq_model:_get_compiled_router_sigmoid_bias`.
# ─────────────────────────────────────────────────────────────────────────
_ROUTER_CACHE: dict[int, callable] = {}


def _compiled_sigmoid_bias_topk(k: int, logits: mx.array, e_bias: mx.array):
    """Sigmoid → topk-via-bias → renorm. Returns (inds, topk_scores)."""
    fn = _ROUTER_CACHE.get(k)
    if fn is None:
        def _router(gates_f32, e_score_bias):
            scores = mx.sigmoid(gates_f32)
            inds = mx.argpartition(-(scores + e_score_bias), kth=k - 1, axis=-1)[..., :k]
            sel = mx.take_along_axis(scores, inds, axis=-1)        # un-biased
            sel = sel / (mx.sum(sel, axis=-1, keepdims=True) + 1e-20)
            return inds, sel
        fn = mx.compile(_router)
        _ROUTER_CACHE[k] = fn
    return fn(logits, e_bias)


# ─────────────────────────────────────────────────────────────────────────
# Attention
# ─────────────────────────────────────────────────────────────────────────

class LagunaAttention(nn.Module):
    def __init__(self, cfg: LagunaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = cfg.num_attention_heads_per_layer[layer_idx]
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.kv_groups = self.n_heads // self.n_kv
        self.scale = self.head_dim ** -0.5
        self.layer_type = cfg.layer_types[layer_idx]

        # Laguna ships per-layer-type RoPE under cfg.rope_parameters:
        #   full_attention      = YaRN (rope_type=yarn, factor=32,
        #                              original_max_position_embeddings=4096,
        #                              attention_factor=1.0, partial_rotary_factor=0.5,
        #                              rope_theta=500000)
        #   sliding_attention   = default (rope_type=default,
        #                              partial_rotary_factor=1.0,
        #                              rope_theta=10000)
        # Pull partial_rotary_factor from THIS layer's dict (1.0 on SWA = full
        # rotary, 0.5 on full = half rotary). Pull rope_theta from THIS dict
        # too (different bases per layer type — 500k vs 10k).
        rp = (cfg.rope_parameters or {}).get(self.layer_type, {})
        self.rope_base = rp.get("rope_theta", 10000.0)
        self.partial = rp.get("partial_rotary_factor", cfg.partial_rotary_factor)
        self.rope_dim = int(self.head_dim * self.partial)
        self.window = cfg.sliding_window if self.layer_type == "sliding_attention" else None

        # initialize_rope dispatches on `rope_type`. Default = nn.RoPE,
        # yarn = YarnRoPE. The HF YaRN dict uses `attention_factor` for
        # the post-rotation length scaling (`mscale` in YarnRoPE's API),
        # so remap that key.
        scaling_cfg = None
        if rp.get("rope_type") and rp.get("rope_type") != "default":
            scaling_cfg = dict(rp)
            if "attention_factor" in scaling_cfg and "mscale" not in scaling_cfg:
                scaling_cfg["mscale"] = scaling_cfg.pop("attention_factor")
        self.rope = initialize_rope(
            self.rope_dim,
            base=self.rope_base,
            traditional=False,
            scaling_config=scaling_cfg,
            max_position_embeddings=cfg.max_position_embeddings,
        )

        h = cfg.hidden_size
        self.q_proj = nn.Linear(h, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(h, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(h, self.n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, h, bias=False)

        # Per-head RMSNorm on q + k AFTER projection, BEFORE rope (HF spec).
        # nn.RMSNorm(head_dim) — the (..., head_dim) trailing dim normalizes
        # each head independently when applied to (B, T, n_heads, head_dim).
        self.q_norm = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

        # Per-head sigmoid gate (cfg.gating=True). g_proj projects the input
        # to one scalar per head; multiplied into the attention output before
        # o_proj.
        self.g_proj = nn.Linear(h, self.n_heads, bias=False) if cfg.gating else None

    def _rope(self, t: mx.array, offset: int) -> mx.array:
        # Partial-rotary: apply RoPE to the first `rope_dim` of head_dim.
        if self.rope_dim == self.head_dim:
            return self.rope(t, offset=offset)
        rot = t[..., :self.rope_dim]
        keep = t[..., self.rope_dim:]
        rot = self.rope(rot, offset=offset)
        return mx.concatenate([rot, keep], axis=-1)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        B, T, _ = x.shape

        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, T, self.n_kv, self.head_dim)
        v = self.v_proj(x).reshape(B, T, self.n_kv, self.head_dim)

        # q/k norm per head — apply BEFORE rope.
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        offset = 0
        if cache is not None and hasattr(cache, "offset"):
            try:
                offset = int(cache.offset)
            except Exception:
                offset = 0

        q = self._rope(q, offset)
        k = self._rope(k, offset)

        if cache is not None and hasattr(cache, "update_and_fetch"):
            k, v = cache.update_and_fetch(k, v)

        # `mx.fast.scaled_dot_product_attention` handles GQA broadcasting
        # internally when `q.shape[1] != k.shape[1]` — the manual
        # `mx.repeat(k, kv_groups, axis=1)` we used to do here doubled
        # the KV bandwidth and caused a measurable per-step slowdown on
        # the 64-head SWA layers (256 → 64 KV tiles vs 8 → 64 broadcast).
        # Drop the repeat and let SDPA pick the optimal kernel path.
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.n_heads * self.head_dim)

        if self.g_proj is not None:
            # HF reference (modeling_laguna.py:412): softplus, not sigmoid.
            # softplus(x) = ln(1+exp(x)) ∈ [0, ∞) — unbounded gate that can
            # amplify the attention output, NOT the bounded [0,1] sigmoid I
            # had originally. Using sigmoid drove residual stream blow-up
            # (std 0.29 → 11 over 30 layers → garbage saturation).
            gate = nn.softplus(self.g_proj(x).astype(mx.float32)).astype(out.dtype)
            if gate.shape[-1] == self.n_heads:
                # per-head gating (config.gating="per-head"): one gate per head,
                # broadcast across head_dim.
                out = out.reshape(B, T, self.n_heads, self.head_dim) * gate[..., None]
                out = out.reshape(B, T, self.n_heads * self.head_dim)
            else:
                # per-element gating (config.gating="per-element", the default and
                # what Laguna-M.1 ships — see modeling_laguna.py: g_out =
                # num_heads*head_dim): g_proj emits one gate per (head, head_dim)
                # channel, so multiply elementwise into the already-flattened
                # (B, T, n_heads*head_dim) attention output.
                out = out * gate

        return self.o_proj(out)


# ─────────────────────────────────────────────────────────────────────────
# MLP variants
# ─────────────────────────────────────────────────────────────────────────

class DenseMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class LagunaMoE(nn.Module):
    """Sigmoid+bias top-k routing (DeepSeek-V3 / Qwen3.5-MoE) + shared expert.

    Score = sigmoid(gate_logits + e_score_correction_bias). Top-k indices
    selected via argpartition; weights normalized over the chosen k. Routed
    output goes through SwitchGLU (gather-style packed-expert matmul);
    shared output goes through a plain DenseMLP. Routed contribution is
    multiplied by `moe_routed_scaling_factor`.
    """
    def __init__(self, cfg: LagunaConfig):
        super().__init__()
        self.cfg = cfg
        self.num_experts = cfg.num_experts
        self.top_k = cfg.num_experts_per_tok
        self.routed_scale = cfg.moe_routed_scaling_factor

        self.gate = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)
        # Per-expert bias added to gate logits before sigmoid+top-k.
        # HF safetensors key: model.layers.N.mlp.experts.e_score_correction_bias.
        # Runtime remaps to model.layers.N.mlp.e_score_correction_bias so
        # this attribute binds.
        self.e_score_correction_bias = mx.zeros(cfg.num_experts)

        self.switch_mlp = SwitchGLU(
            cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_experts,
            bias=False,
        )
        self.shared_expert = DenseMLP(cfg.hidden_size, cfg.shared_expert_intermediate_size)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, H = x.shape
        flat = x.reshape(-1, H)

        # Sigmoid + bias topk routing (DeepSeek-V3 / Qwen3.5-MoE recipe):
        # the bias picks WHICH experts are selected, but the gating weight
        # comes from the UN-biased sigmoid scores. Mixing the bias into
        # the gating side drove a residual-stream blow-up in an earlier
        # round of this work (std 0.29 → 11 over 30 layers).
        #
        # The per-step routing math (gate matmul → sigmoid → add bias →
        # argpartition → take_along_axis → renorm) is small but runs 39×
        # per token; without `mx.compile` each op triggers a fresh graph
        # build and Metal dispatch, costing ~1 ms per layer. Compile the
        # bias+topk slice and cache it on the class (keyed on top_k so
        # different MoE blocks share the graph).
        logits = self.gate(flat).astype(mx.float32)
        inds, topk_scores = _compiled_sigmoid_bias_topk(
            self.top_k, logits, self.e_score_correction_bias.astype(mx.float32),
        )

        # SwitchGLU expects (B*T, k) indices; flat already 2-D.
        y = self.switch_mlp(flat, inds)            # (B*T, k, H)
        y = (y * topk_scores[..., None].astype(y.dtype)).sum(axis=-2)
        # Routed-scale applied to the routed-only contribution. Shared
        # expert is NOT scaled — matches HF order exactly.
        y = y * self.routed_scale + self.shared_expert(flat)
        return y.reshape(B, T, H).astype(x.dtype)


# ─────────────────────────────────────────────────────────────────────────
# Decoder layer + model
# ─────────────────────────────────────────────────────────────────────────

class LagunaLayer(nn.Module):
    def __init__(self, cfg: LagunaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = cfg.layer_types[layer_idx]
        self.is_dense = (cfg.mlp_layer_types[layer_idx] == "dense")

        self.self_attn = LagunaAttention(cfg, layer_idx)
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

        if self.is_dense:
            self.mlp = DenseMLP(cfg.hidden_size, cfg.intermediate_size)
        else:
            self.mlp = LagunaMoE(cfg)

    def __call__(self, x, mask=None, cache=None):
        h = x + self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class LagunaForCausalLM(nn.Module):
    """Top-level wrapper. Flat-attaches embed_tokens / layers / norm / lm_head
    at the root (no inner self.model). HF safetensors carry a `model.` prefix
    on these keys; runtime.py strips it before model.update().
    """
    def __init__(self, cfg: LagunaConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [LagunaLayer(cfg, i) for i in range(cfg.num_hidden_layers)]
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    @property
    def head_dim(self) -> int:
        return self.cfg.head_dim

    @property
    def n_kv_heads(self) -> int:
        return self.cfg.num_key_value_heads

    def make_cache(self):
        """Per-layer cache list. Full attention layers get a plain KVCache;
        sliding_attention layers get RotatingKVCache sized to the window.

        keep=0: the HF reference only retains attention-sink tokens when
        config.swa_attention_sink_enabled is set, and no shipped Laguna
        config (XS.2 / M.1 / S-2.1) sets it — pure sliding window. The old
        keep=4 (copied from the gemma pattern) let decode attend the first
        4 prompt tokens forever, silently diverging from the reference once
        the sequence passed the 512-token window.
        """
        from mlx_lm.models.cache import KVCache, RotatingKVCache
        caches = []
        for i in range(self.cfg.num_hidden_layers):
            if self.cfg.layer_types[i] == "sliding_attention":
                caches.append(RotatingKVCache(max_size=self.cfg.sliding_window, keep=0))
            else:
                caches.append(KVCache())
        return caches

    def __call__(self, ids, cache=None, caches=None, mask=None, **_):
        # mlx_lm convention: model(ids, cache=…) -> logits (mx.array). Caches
        # are mutated in place by the attention layers via update_and_fetch.
        # The local greedy() helper below uses the dual `caches=` name and
        # peels off the returned cache list — keep both behaviors so older
        # callers and the engine both work.
        local_call = caches is not None or cache is None
        if caches is None:
            caches = cache

        h = self.embed_tokens(ids)
        T = h.shape[1]

        if caches is None:
            # Was previously `[None] * num_layers` — that placeholder list
            # never populates KV cache during prefill (each layer's
            # `cache=None`, so attention's `cache.update_and_fetch` branch
            # never runs). Decode then re-enters with the same `[None]*N`
            # list and self-attends only to the single new token, collapsing
            # generation to a repeated token. `self.make_cache()` returns
            # real per-layer KVCache / RotatingKVCache instances, matching
            # the layer_type schedule (full vs sliding). Prefill populates
            # them, decode reuses them.
            caches = self.make_cache()

        # Per-layer-type masks (HF reference: LagunaModel.forward builds
        # separate create_causal_mask / create_sliding_window_causal_mask
        # mappings). Feeding ONE full causal mask to every layer — the old
        # behavior here — is only equivalent for prompts <= sliding_window
        # (512); past that, SWA layers wrongly attend the whole prefix.
        # mlx_lm's create_attention_mask(window_size=…) is the same banded
        # mask the gemma3 port uses; each mask is derived from a cache of
        # its OWN layer type so offsets line up.
        full_mask = None
        sliding_mask = None
        if T > 1:
            lt = self.cfg.layer_types
            full_idx = next((i for i, t in enumerate(lt) if t == "full_attention"), None)
            swa_idx = next((i for i, t in enumerate(lt) if t == "sliding_attention"), None)
            try:
                from mlx_lm.models.base import create_attention_mask
                if full_idx is not None:
                    full_mask = create_attention_mask(h, caches[full_idx])
                if swa_idx is not None:
                    sliding_mask = create_attention_mask(
                        h, caches[swa_idx], window_size=self.cfg.sliding_window)
            except Exception:
                causal = mx.triu(mx.full((T, T), -mx.inf, dtype=h.dtype), k=1)
                full_mask = causal
                if swa_idx is not None:
                    # banded causal: j > i (future) or i - j >= window (evicted)
                    rows = mx.arange(T)[:, None]
                    cols = mx.arange(T)[None, :]
                    band = mx.where(
                        (cols > rows) | ((rows - cols) >= self.cfg.sliding_window),
                        -mx.inf, 0.0).astype(h.dtype)
                    sliding_mask = band

        for layer, c in zip(self.layers, caches):
            m = full_mask if layer.layer_type == "full_attention" else sliding_mask
            h = layer(h, mask=m, cache=c)

        logits = self.lm_head(self.norm(h))
        # Engine path (cache=…, no caches=) expects bare logits.
        # Local greedy path (caches=…) expects (logits, caches).
        if cache is not None and caches is cache:
            return logits
        return logits, caches


def _build(src: str, *, fmt: str, plan=None):
    """Architecture builder used by jang_tools' load_jangtq dispatcher."""
    cfg = LagunaConfig.from_json(f"{src}/config.json")
    return LagunaForCausalLM(cfg), cfg, None
