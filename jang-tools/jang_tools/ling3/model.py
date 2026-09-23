"""BailingMoeV3 (Ling-3.0) in MLX.

Created by Jinho Jang (eric@jangq.ai)

A faithful port of `modeling_bailing_moe_v3.py`, built so JANG calibration
(imatrix / Hessian / AWQ) and KL evaluation can run against the bf16 source.
This is *not* the shipping runtime — vmlx-swift owns that.

Layer layout for Ling-3.0-tiny (24 layers, `layer_group_size` 4):
  * layers where ``(idx + 1) % 4 == 0`` -> gated Multi-Latent Attention (6 of them)
  * every other layer                   -> Kimi Delta Attention (18 of them)
  * layer 0 -> dense MLP; layers >= `first_k_dense_replace` -> sparse MoE
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import SwitchGLU

from jang_tools.ling3.kda import kda_gate, kda_recurrent, kda_step, l2norm, short_conv


@dataclass
class ModelArgs:
    model_type: str = "bailing_hybrid"
    hidden_size: int = 1536
    intermediate_size: int = 4608
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    head_dim: int = 128
    vocab_size: int = 157184
    rms_norm_eps: float = 1e-6
    rope_theta: float = 6000000.0
    rope_interleave: bool = True
    layer_group_size: int = 4
    first_k_dense_replace: int = 1
    # MoE
    num_experts: int = 128
    num_experts_per_tok: int = 8
    num_shared_experts: int = 1
    moe_intermediate_size: int = 512
    moe_shared_expert_intermediate_size: int = 512
    n_group: int = 8
    topk_group: int = 4
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    moe_router_enable_expert_bias: bool = True
    # MLA
    q_lora_rank: int = 256
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    qk_head_dim: int = 192
    v_head_dim: int = 128
    gated_attention_proj_granularity_type: str | None = "head_wise"
    use_qkv_bias: bool = False
    # KDA
    short_conv_kernel_size: int = 4
    kda_lower_bound: float | None = -5.0
    kda_safe_gate: bool = True
    no_kda_lora: bool = True
    tie_word_embeddings: bool = False
    rope_scaling: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "ModelArgs":
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs = {k: v for k, v in cfg.items() if k in known}
        kwargs["extra"] = {k: v for k, v in cfg.items() if k not in known}
        return cls(**kwargs)


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Gate(nn.Module):
    """`noaux_tc` sigmoid router with group-limited top-k.

    Selection uses ``sigmoid(logits) + expert_bias``; the *weights* come from the
    un-biased scores. The bias steers which experts run, never how much they count.
    Router math is fp32 throughout (`router_dtype: fp32`) — a rounding flip here
    changes which experts execute, not merely by how much.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.num_experts = args.num_experts
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.routed_scaling_factor = args.routed_scaling_factor
        self.weight = mx.zeros((args.num_experts, args.hidden_size))
        self.expert_bias = mx.zeros((args.num_experts,))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        n_tokens = x.shape[0]
        logits = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
        scores = mx.sigmoid(logits)

        scored = scores + self.expert_bias.astype(mx.float32)

        # group-limited top-k: rank groups by the sum of their top-2 experts
        per_group = self.num_experts // self.n_group
        grouped = scored.reshape(n_tokens, self.n_group, per_group)
        top2 = mx.topk(grouped, 2, axis=-1)
        group_scores = mx.sum(top2, axis=-1)                       # [T, n_group]

        group_idx = mx.argpartition(-group_scores, self.topk_group - 1, axis=-1)
        group_idx = group_idx[:, : self.topk_group]
        group_mask = mx.zeros((n_tokens, self.n_group), dtype=mx.float32)
        group_mask = mx.put_along_axis(
            group_mask, group_idx, mx.ones(group_idx.shape, dtype=mx.float32), axis=-1
        )
        score_mask = mx.repeat(group_mask, per_group, axis=-1)      # [T, E]

        masked = mx.where(score_mask > 0, scored, mx.array(-mx.inf, mx.float32))
        idx = mx.argpartition(-masked, self.top_k - 1, axis=-1)[:, : self.top_k]

        weights = mx.take_along_axis(scores, mx.stop_gradient(idx), axis=-1)
        if self.top_k > 1:
            weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
        weights = weights * self.routed_scaling_factor
        return idx, weights.astype(x.dtype), logits


class SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate = Gate(args)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        self.shared_experts = MLP(
            args.hidden_size,
            args.moe_shared_expert_intermediate_size * max(args.num_shared_experts, 1),
        )

    def __call__(self, x: mx.array) -> mx.array:
        B, T, H = x.shape
        idx, weights, _ = self.gate(x.reshape(-1, H))
        # SwitchGLU contract (mlx_lm): x [..., H] + indices [..., k] -> [..., k, H].
        # Do NOT add a singleton dim — the small-batch (unsorted, idx.size < 64)
        # fast path preserves leading dims exactly, so a spurious dim survives to
        # the weighted sum and breaks ONLY decode-sized chunks.
        # Routing indices are discrete — no gradient flows through WHICH
        # experts run (gather_mm has no VJP w.r.t. indices, by design).
        y = self.switch_mlp(x, mx.stop_gradient(idx.reshape(B, T, -1)))  # [B, T, k, H]
        y = (y * weights.reshape(B, T, -1, 1).astype(y.dtype)).sum(axis=-2)
        return y + self.shared_experts(x)


class MultiLatentAttention(nn.Module):
    """MLA with a head-wise output gate.

    The gate is easy to overlook: `g_proj` here is `[hidden, num_heads]` and scales
    each head's output by `sigmoid(g)` *before* `dense`. Dropping it silently
    degrades every attention layer.
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.num_heads = args.num_attention_heads
        self.qk_head_dim = args.qk_head_dim
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.scale = args.qk_head_dim ** -0.5

        self.q_a_proj = nn.Linear(args.hidden_size, args.q_lora_rank, bias=args.use_qkv_bias)
        self.q_a_layernorm = nn.RMSNorm(args.q_lora_rank, eps=1e-6)
        self.q_b_proj = nn.Linear(args.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            args.hidden_size, args.kv_lora_rank + self.qk_rope_head_dim, bias=args.use_qkv_bias
        )
        self.kv_a_layernorm = nn.RMSNorm(args.kv_lora_rank, eps=1e-6)
        self.kv_b_proj = nn.Linear(
            args.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.granularity = args.gated_attention_proj_granularity_type
        if self.granularity == "head_wise":
            self.g_proj = nn.Linear(args.hidden_size, self.num_heads, bias=False)
        elif self.granularity == "element_wise":
            self.g_proj = nn.Linear(args.hidden_size, self.num_heads * self.v_head_dim, bias=False)

        self.dense = nn.Linear(self.num_heads * self.v_head_dim, args.hidden_size, bias=args.use_qkv_bias)

        self.rope = nn.RoPE(
            self.qk_rope_head_dim,
            traditional=args.rope_interleave,
            base=args.rope_theta,
        )

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        B, T, _ = x.shape

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.reshape(B, T, self.num_heads, self.qk_head_dim).transpose(0, 2, 1, 3)
        q_pass, q_rot = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed = self.kv_a_proj_with_mqa(x)
        k_pass, k_rot = mx.split(compressed, [self.kv_lora_rank], axis=-1)
        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass))
        k_pass = k_pass.reshape(
            B, T, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        ).transpose(0, 2, 1, 3)
        k_pass, values = mx.split(k_pass, [self.qk_nope_head_dim], axis=-1)

        k_rot = k_rot.reshape(B, 1, T, self.qk_rope_head_dim)

        offset = cache.offset if cache is not None else 0
        q_rot = self.rope(q_rot, offset=offset)
        k_rot = self.rope(k_rot, offset=offset)
        k_rot = mx.repeat(k_rot, self.num_heads, axis=1)

        queries = mx.concatenate([q_pass, q_rot], axis=-1)
        keys = mx.concatenate([k_pass, k_rot], axis=-1)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        out = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3)                       # [B, T, H, v_head_dim]

        if self.granularity is not None:
            gate = mx.sigmoid(self.g_proj(x).astype(mx.float32)).astype(out.dtype)
            if self.granularity == "head_wise":
                out = out * gate[..., None]
            else:
                out = out * gate.reshape(B, T, self.num_heads, self.v_head_dim)

        return self.dense(out.reshape(B, T, -1))


class ShortConvWeight(nn.Module):
    """Holder for a depthwise short-conv kernel, shape ``[C, W]``.

    Not an `nn.Conv1d`: these weights are 4 numbers per channel and are a
    deliberate fp16 keep (quantizing them saves ~0 bytes and perturbs the input
    to a recurrence). Keeping them out of the Linear/Conv module types also keeps
    them out of any tooling that sweeps quantizable layers by module class.
    """

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.weight = mx.zeros((channels, kernel_size))


class KimiDeltaAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.num_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.conv_size = args.short_conv_kernel_size
        self.lower_bound = args.kda_lower_bound if args.kda_safe_gate else None
        proj = self.num_heads * self.head_dim

        self.q_proj = nn.Linear(args.hidden_size, proj, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, proj, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, proj, bias=False)

        self.q_conv1d = ShortConvWeight(proj, self.conv_size)
        self.k_conv1d = ShortConvWeight(proj, self.conv_size)
        self.v_conv1d = ShortConvWeight(proj, self.conv_size)

        self.A_log = mx.zeros((self.num_heads,))
        self.dt_bias = mx.zeros((proj,))

        self.f_proj = nn.Linear(args.hidden_size, proj, bias=False)
        self.b_proj = nn.Linear(args.hidden_size, self.num_heads, bias=False)
        self.g_proj = nn.Linear(args.hidden_size, proj, bias=False)

        self.o_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.o_proj = nn.Linear(proj, args.hidden_size, bias=False)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        conv_state = cache.conv_state if cache is not None else (None, None, None)
        rec_state = cache.rec_state if cache is not None else None

        q, cq = short_conv(self.q_proj(x), self.q_conv1d.weight, conv_state[0])
        k, ck = short_conv(self.k_proj(x), self.k_conv1d.weight, conv_state[1])
        v, cv = short_conv(self.v_proj(x), self.v_conv1d.weight, conv_state[2])

        q = l2norm(q.reshape(B, T, H, D))
        k = l2norm(k.reshape(B, T, H, D))
        v = v.reshape(B, T, H, D)

        g = kda_gate(self.f_proj(x).reshape(B, T, H, D), self.A_log, self.dt_bias, self.lower_bound)
        beta = mx.sigmoid(self.b_proj(x).astype(mx.float32))

        if T == 1 and rec_state is not None:
            o, rec_state = kda_step(q[:, 0], k[:, 0], v[:, 0], g[:, 0], beta[:, 0], rec_state)
            o = o[:, None]
        elif getattr(self, "use_chunked", False):
            # TRAINING ONLY: the WY-form recursion amplifies fp32 noise ~1e-3
            # rel (measured) — below SGD noise, above inference tolerance.
            # Also keeps the autodiff graph under Metal's ~500k-op limit.
            from jang_tools.ling3.kda import kda_chunked
            o, rec_state = kda_chunked(q, k, v, g, beta, state=rec_state)
        else:
            o, rec_state = kda_recurrent(q, k, v, g, beta, state=rec_state)

        if cache is not None:
            cache.conv_state = (cq, ck, cv)
            cache.rec_state = rec_state

        # gated RMSNorm: rmsnorm(o) * weight * sigmoid(g_proj(x))
        o = self.o_norm(o.astype(x.dtype))
        o = o * mx.sigmoid(self.g_proj(x).astype(mx.float32)).reshape(B, T, H, D).astype(o.dtype)
        return self.o_proj(o.reshape(B, T, H * D))


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        n_groups = args.num_hidden_layers // args.layer_group_size * args.layer_group_size
        self.is_full_attention = (
            (layer_idx + 1) % args.layer_group_size == 0 or layer_idx >= n_groups
        )
        self.attention = (
            MultiLatentAttention(args, layer_idx)
            if self.is_full_attention
            else KimiDeltaAttention(args, layer_idx)
        )
        self.mlp = (
            SparseMoeBlock(args)
            if layer_idx >= args.first_k_dense_replace
            else MLP(args.hidden_size, args.intermediate_size)
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        x = x + self.attention(self.input_layernorm(x), mask=mask, cache=cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class BailingMoeV3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs: mx.array, cache=None, mask=None) -> mx.array:
        h = self.word_embeddings(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        if mask is None and h.shape[1] > 1:
            mask = "causal"
        for layer, c in zip(self.layers, cache):
            # the KDA layers are recurrent — a causal mask is meaningless there
            h = layer(h, mask=mask if layer.is_full_attention else None, cache=c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = BailingMoeV3Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None, mask=None) -> mx.array:
        h = self.model(inputs, cache=cache, mask=mask)
        if self.args.tie_word_embeddings:
            return self.model.word_embeddings.as_linear(h)
        return self.lm_head(h)

    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Map HF checkpoint names onto this module tree.

        Two structural rewrites:
          * per-expert `mlp.experts.{e}.{gate,up,down}_proj.weight` are stacked into
            `mlp.switch_mlp.{...}.weight` of shape ``[E, out, in]``.
          * `*_conv1d.weight` arrives as ``[C, 1, W]``; MLX `Conv1d` wants ``[C, W, 1]``.
        """
        out: dict[str, mx.array] = {}
        experts: dict[str, dict[int, mx.array]] = {}

        for k, v in weights.items():
            if k.startswith("model.layers.") and ".mlp.experts." in k:
                head, tail = k.split(".mlp.experts.")
                eid, proj = tail.split(".", 1)
                experts.setdefault(f"{head}.mlp.switch_mlp.{proj}", {})[int(eid)] = v
                continue
            if k.endswith("_conv1d.weight") and v.ndim == 3:
                v = v.reshape(v.shape[0], -1)          # [C, 1, W] -> [C, W]
            out[k] = v

        for name, per_expert in experts.items():
            stacked = mx.stack([per_expert[i] for i in sorted(per_expert)], axis=0)
            out[name] = stacked

        return out
