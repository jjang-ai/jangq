"""GLM-5.3-Flash (glm5_next) text model in MLX.

Created by Jinho Jang (eric@jangq.ai) — 2026-08-29.

Faithful port of transformers `modeling_glm5_next.py` (main, 2026-08-29):

  * 45 layers: 34 KDA linear-attention (Kimi delta rule, low-rank forget/output
    gates) + 11 MLA layers (DeepSeek-V3 MLA, **pure NoPE** — qk_rope_head_dim=0,
    no rotary anywhere).
  * DSA indexer: NOT implemented. For sequences <= index_topk (2048) the top-k
    selects every key, so dense causal attention is BIT-EXACT. `__call__`
    asserts this bound; the quant pipeline (calibration replay, KL eval,
    coherence probes) never exceeds it. The serving runtime (vmlx) owns the
    sparse path.
  * mHC hyper-connections with Sinkhorn: per sublayer, fn[24,16384]/base[24]/
    scale[3] produce pre (stream collapse), post (output placement) and a 4x4
    Sinkhorn doubly-stochastic combine. Final collapse = unweighted mean.
  * MoE: 288 routed experts top-8 (sigmoid scores + e_score_correction_bias,
    n_group=1 so group masking is a no-op — asserted), norm_topk_prob,
    routed_scaling_factor 2.5, shared expert, swiglu clamp at ±10.
  * MTP layer 45 is dropped at load (same as the HF reference's
    _keys_to_ignore); the CONVERTER still carries its weights in the bundle.

KDA core (gate math differs from Ling-3.0!):
    g = lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))   # per-channel
i.e. a smooth (-5, 0) log-decay, NOT Ling's clamped -exp*softplus. The
recurrence itself (delta rule, l2-normed q/k, beta=sigmoid(b_proj)) is the
same as `jang_tools.ling3.kda`, which this module reuses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn

from jang_tools.ling3.kda import kda_chunked, kda_recurrent, kda_step, l2norm, short_conv

# Optional per-layer input recorder for calibration replays: maps
# module-path -> callback(x_rows) set by the calibration driver.
INPUT_RECORDER: dict = {}


@dataclass
class Glm5Args:
    hidden_size: int = 4096
    num_hidden_layers: int = 45
    rms_norm_eps: float = 1e-5
    vocab_size: int = 154880
    # mHC
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    # KDA
    linear_num_heads: int = 64
    linear_head_dim: int = 128
    linear_conv_kernel: int = 4
    linear_lower_bound: float = -5.0
    # MLA
    num_attention_heads: int = 64
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 256
    v_head_dim: int = 256
    index_topk: int = 2048
    # MoE
    n_routed_experts: int = 288
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 2048
    n_shared_experts: int = 1
    intermediate_size: int = 12288
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    swiglu_limit: float = 10.0
    first_k_dense_replace: int = 3
    layer_types: list = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: dict) -> "Glm5Args":
        t = cfg.get("text_config", cfg)
        la = t.get("linear_attn_config") or {}
        assert t.get("n_group", 1) == 1 and t.get("topk_group", 1) == 1, \
            "router group logic not implemented (config says n_group != 1)"
        assert (t.get("qk_rope_head_dim") or 0) == 0, "expected NoPE MLA"
        return cls(
            hidden_size=t["hidden_size"],
            num_hidden_layers=t["num_hidden_layers"],
            rms_norm_eps=t.get("rms_norm_eps", 1e-5),
            vocab_size=t["vocab_size"],
            hc_mult=t.get("hc_mult", 4),
            hc_sinkhorn_iters=t.get("hc_sinkhorn_iters", 20),
            hc_eps=t.get("hc_eps", 1e-6),
            linear_num_heads=la.get("num_heads", 64),
            linear_head_dim=la.get("head_dim", 128),
            linear_conv_kernel=la.get("short_conv_kernel_size", 4),
            linear_lower_bound=la.get("gate_lower_bound", -5.0),
            num_attention_heads=t["num_attention_heads"],
            q_lora_rank=t["q_lora_rank"],
            kv_lora_rank=t["kv_lora_rank"],
            qk_nope_head_dim=t["qk_nope_head_dim"],
            v_head_dim=t["v_head_dim"],
            index_topk=t.get("index_topk", 2048),
            n_routed_experts=t["n_routed_experts"],
            num_experts_per_tok=t["num_experts_per_tok"],
            moe_intermediate_size=t["moe_intermediate_size"],
            n_shared_experts=t.get("n_shared_experts", 1),
            intermediate_size=t["intermediate_size"],
            routed_scaling_factor=t.get("routed_scaling_factor", 2.5),
            norm_topk_prob=t.get("norm_topk_prob", True),
            swiglu_limit=t.get("swiglu_limit", 10.0),
            first_k_dense_replace=t.get("first_k_dense_replace", 3),
            layer_types=list(t["layer_types"]),
        )


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


def _record(path: str, x: mx.array):
    cb = INPUT_RECORDER.get(path)
    if cb is not None:
        cb(x)


# ---------------------------------------------------------------- mHC ------
class HyperConnection(nn.Module):
    """fn/base/scale -> (post, comb, collapsed). All mix math in fp32."""

    def __init__(self, args: Glm5Args):
        super().__init__()
        h, d = args.hc_mult, args.hidden_size
        self.hc_mult = h
        self.iters = args.hc_sinkhorn_iters
        self.eps = args.hc_eps
        self.rms_eps = args.rms_norm_eps
        self.hc_fn = mx.zeros(((2 + h) * h, h * d))
        self.hc_base = mx.zeros(((2 + h) * h,))
        self.hc_scale = mx.ones((3,))

    def __call__(self, streams: mx.array):
        # streams: [B, S, H, D]
        h = self.hc_mult
        flat = streams.reshape(*streams.shape[:2], -1).astype(mx.float32)
        flat = flat * mx.rsqrt(mx.mean(flat * flat, axis=-1, keepdims=True) + self.rms_eps)
        mix = flat @ self.hc_fn.astype(mx.float32).T
        pre_w, post_w, comb_w = mx.split(mix, [h, 2 * h], axis=-1)
        base = self.hc_base.astype(mx.float32)
        s0, s1, s2 = self.hc_scale.astype(mx.float32)

        pre = mx.sigmoid(pre_w * s0 + base[:h]) + self.eps
        post = 2.0 * mx.sigmoid(post_w * s1 + base[h:2 * h])
        comb = mx.softmax(comb_w.reshape(*comb_w.shape[:-1], h, h) * s2
                          + base[2 * h:].reshape(h, h), axis=-1) + self.eps
        comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + self.eps)
        for _ in range(self.iters - 1):
            comb = comb / (mx.sum(comb, axis=-1, keepdims=True) + self.eps)
            comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + self.eps)

        collapsed = mx.sum(pre[..., None] * streams.astype(mx.float32), axis=2)
        return post, comb, collapsed.astype(streams.dtype)


def hc_place(post: mx.array, comb: mx.array, out: mx.array, residual: mx.array) -> mx.array:
    """streams' = post⊗out + combᵀ @ residual   (all [B,S,·] shapes)."""
    dt = residual.dtype
    return (post.astype(dt)[..., None] * out[..., None, :]
            + mx.matmul(comb.astype(dt).transpose(0, 1, 3, 2), residual))


# ---------------------------------------------------------------- KDA ------
class KDAAttention(nn.Module):
    def __init__(self, args: Glm5Args):
        super().__init__()
        d, H, K = args.hidden_size, args.linear_num_heads, args.linear_head_dim
        qkv = H * K
        self.H, self.K = H, K
        self.lower_bound = args.linear_lower_bound
        self.q_proj = nn.Linear(d, qkv, bias=False)
        self.k_proj = nn.Linear(d, qkv, bias=False)
        self.v_proj = nn.Linear(d, qkv, bias=False)
        self.q_conv1d = mx.zeros((qkv, args.linear_conv_kernel))
        self.k_conv1d = mx.zeros((qkv, args.linear_conv_kernel))
        self.v_conv1d = mx.zeros((qkv, args.linear_conv_kernel))
        self.b_proj = nn.Linear(d, H, bias=False)
        self.f_a_proj = nn.Linear(d, K, bias=False)
        self.f_b_proj = nn.Linear(K, qkv, bias=False)
        self.g_a_proj = nn.Linear(d, K, bias=False)
        self.g_b_proj = nn.Linear(K, qkv, bias=False)
        self.A_log = mx.zeros((H,))
        self.dt_bias = mx.zeros((qkv,))
        self.o_norm = mx.ones((K,))
        self.o_proj = nn.Linear(qkv, d, bias=False)
        self.rms_eps = args.rms_norm_eps

    def _gate(self, x: mx.array) -> mx.array:
        # lower_bound * sigmoid(exp(A_log) * (f + dt_bias)) — smooth (-5, 0)
        B, T, _ = x.shape
        f = self.f_b_proj(self.f_a_proj(x)).astype(mx.float32) + self.dt_bias.astype(mx.float32)
        f = f.reshape(B, T, self.H, self.K)
        rate = mx.exp(self.A_log.astype(mx.float32)).reshape(1, 1, self.H, 1)
        return self.lower_bound * mx.sigmoid(rate * f)

    def __call__(self, x: mx.array, cache=None):
        B, T, _ = x.shape
        H, K = self.H, self.K
        conv_q = conv_k = conv_v = None
        state = None
        if cache is not None and cache.get("kda_state") is not None:
            conv_q, conv_k, conv_v = cache["conv"]
            state = cache["kda_state"]
        q, cq = short_conv(self.q_proj(x), self.q_conv1d, conv_q)
        k, ck = short_conv(self.k_proj(x), self.k_conv1d, conv_k)
        v, cv = short_conv(self.v_proj(x), self.v_conv1d, conv_v)
        q = l2norm(q.reshape(B, T, H, K))
        k = l2norm(k.reshape(B, T, H, K))
        v = v.reshape(B, T, H, K)
        g = self._gate(x)
        beta = mx.sigmoid(self.b_proj(x).astype(mx.float32))
        if T == 1 and state is not None:
            o, S = kda_step(q[:, 0], k[:, 0], v[:, 0], g[:, 0], beta[:, 0], state)
            o = o[:, None]
        elif T <= 64:
            o, S = kda_recurrent(q, k, v, g, beta, state)
        else:
            o, S = kda_chunked(q, k, v, g, beta, state)
        if cache is not None:
            cache["conv"] = (cq, ck, cv)
            cache["kda_state"] = S
        gate = self.g_b_proj(self.g_a_proj(x)).reshape(B, T, H, K)
        o32 = o.astype(mx.float32)
        o32 = o32 * mx.rsqrt(mx.mean(o32 * o32, axis=-1, keepdims=True) + self.rms_eps)
        o32 = self.o_norm.astype(mx.float32) * o32
        o32 = o32 * mx.sigmoid(gate.astype(mx.float32))
        return self.o_proj(o32.astype(x.dtype).reshape(B, T, H * K))


# ---------------------------------------------------------------- MLA ------
class MLAAttention(nn.Module):
    """DeepSeek-V3 MLA, pure NoPE. Dense causal attention (exact for
    T <= index_topk; asserted by the model __call__)."""

    def __init__(self, args: Glm5Args):
        super().__init__()
        d = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.qk = args.qk_nope_head_dim
        self.vd = args.v_head_dim
        self.q_a_proj = nn.Linear(d, args.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(args.q_lora_rank, args.rms_norm_eps)
        self.q_b_proj = nn.Linear(args.q_lora_rank, self.n_heads * self.qk, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(d, args.kv_lora_rank, bias=False)
        self.kv_a_layernorm = RMSNorm(args.kv_lora_rank, args.rms_norm_eps)
        self.kv_b_proj = nn.Linear(args.kv_lora_rank, self.n_heads * (self.qk + self.vd), bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.vd, d, bias=False)
        self.scale = self.qk ** -0.5

    def __call__(self, x: mx.array, cache=None):
        B, T, _ = x.shape
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.reshape(B, T, self.n_heads, self.qk).transpose(0, 2, 1, 3)
        kv = self.kv_b_proj(self.kv_a_layernorm(self.kv_a_proj_with_mqa(x)))
        kv = kv.reshape(B, T, self.n_heads, self.qk + self.vd).transpose(0, 2, 1, 3)
        k, v = mx.split(kv, [self.qk], axis=-1)
        if cache is not None:
            pk, pv = cache.get("kv", (None, None))
            if pk is not None:
                k = mx.concatenate([pk, k], axis=2)
                v = mx.concatenate([pv, v], axis=2)
            cache["kv"] = (k, v)
        # bf16 SDPA at L==1 decode is the known MLA-absorb trap; we run the
        # plain (non-absorbed) path but keep fp32 SDPA for safety.
        o = mx.fast.scaled_dot_product_attention(
            q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
            scale=self.scale, mask="causal" if T > 1 else None)
        o = o.astype(x.dtype).transpose(0, 2, 1, 3).reshape(B, T, self.n_heads * self.vd)
        return self.o_proj(o)


# ---------------------------------------------------------------- MoE ------
def _clamped_swiglu(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    gate = mx.minimum(gate, limit)
    up = mx.clip(up, -limit, limit)
    return nn.silu(gate) * up


class DenseMLP(nn.Module):
    def __init__(self, args: Glm5Args, inter: int):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, inter, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, args.hidden_size, bias=False)
        self.limit = args.swiglu_limit

    def __call__(self, x):
        return self.down_proj(_clamped_swiglu(self.gate_proj(x), self.up_proj(x), self.limit))


class ClampedSwiGLU(nn.Module):
    """SwitchGLU activation hook: silu(clamp(gate)) * clamp(up), limit ±10."""

    def __init__(self, limit: float):
        super().__init__()
        self._limit = limit

    def __call__(self, x_up, x_gate):
        return _clamped_swiglu(x_gate, x_up, self._limit)


class MoEBlock(nn.Module):
    def __init__(self, args: Glm5Args):
        super().__init__()
        E = args.n_routed_experts
        self.k = args.num_experts_per_tok
        self.norm_topk = args.norm_topk_prob
        self.scaling = args.routed_scaling_factor
        self.gate = nn.Linear(args.hidden_size, E, bias=False)
        self.e_score_correction_bias = mx.zeros((E,))
        from mlx_lm.models.switch_layers import SwitchGLU
        self.switch_mlp = SwitchGLU(args.hidden_size, args.moe_intermediate_size, E,
                                    activation=ClampedSwiGLU(args.swiglu_limit))
        self.shared_experts = DenseMLP(args, args.moe_intermediate_size * args.n_shared_experts)

    def __call__(self, x: mx.array):
        logits = x.astype(mx.float32) @ self.gate.weight.astype(mx.float32).T
        scores = mx.sigmoid(logits)
        choice = scores + self.e_score_correction_bias.astype(mx.float32)
        idx = mx.argpartition(-choice, kth=self.k - 1, axis=-1)[..., : self.k]
        w = mx.take_along_axis(scores, idx, axis=-1)
        if self.norm_topk:
            w = w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20)
        w = w * self.scaling
        routed = self.switch_mlp(x, idx)                   # [B, T, k, d]
        routed = mx.sum(routed * w[..., None].astype(routed.dtype), axis=-2)
        return routed.astype(x.dtype) + self.shared_experts(x)


# ---------------------------------------------------------------- layers ---
class DecoderLayer(nn.Module):
    def __init__(self, args: Glm5Args, layer_idx: int):
        super().__init__()
        self.is_linear = args.layer_types[layer_idx] == "linear_attention"
        self.self_attn = KDAAttention(args) if self.is_linear else MLAAttention(args)
        self.mlp = (DenseMLP(args, args.intermediate_size)
                    if layer_idx < args.first_k_dense_replace else MoEBlock(args))
        self.input_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.attn_hc = HyperConnection(args)
        self.ffn_hc = HyperConnection(args)

    def __call__(self, streams: mx.array, cache=None):
        residual = streams
        post, comb, x = self.attn_hc(streams)
        x = self.self_attn(self.input_layernorm(x), cache=cache)
        streams = hc_place(post, comb, x, residual)

        residual = streams
        post, comb, x = self.ffn_hc(streams)
        x = self.mlp(self.post_attention_layernorm(x))
        return hc_place(post, comb, x, residual)


class Glm5NextModel(nn.Module):
    def __init__(self, args: Glm5Args):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = RMSNorm(args.hidden_size, args.rms_norm_eps)

    def __call__(self, input_ids: mx.array, cache=None):
        x = self.embed_tokens(input_ids)
        # dense-attention bypass validity (exact only under index_topk)
        seen = (cache["pos"] if cache is not None else 0) + x.shape[1]
        assert seen <= self.args.index_topk, \
            f"dense DSA bypass invalid beyond {self.args.index_topk} tokens (got {seen})"
        if cache is not None:
            cache["pos"] = seen
        streams = mx.broadcast_to(x[:, :, None, :],
                                  (*x.shape[:2], self.args.hc_mult, x.shape[-1]))
        for i, layer in enumerate(self.layers):
            lc = cache["layers"][i] if cache is not None else None
            streams = layer(streams, cache=lc)
        return self.norm(mx.mean(streams, axis=2))


class Glm5NextForCausalLM(nn.Module):
    def __init__(self, args: Glm5Args):
        super().__init__()
        self.model = Glm5NextModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def make_cache(self):
        return {"pos": 0,
                "layers": [{"conv": (None, None, None), "kda_state": None}
                           if l.is_linear else {} for l in self.model.layers]}

    def __call__(self, input_ids: mx.array, cache=None):
        return self.lm_head(self.model(input_ids, cache=cache))
