"""Inkling (thinkingmachines) MLX runtime — JANG family module.

Reference: transformers@main `src/transformers/models/inkling/modeling_inkling.py`
(not present in any released transformers). Spec, traps and verification gates:
`docs/runtime/inkling-small/`.

From-scratch implementation. In particular we do NOT reuse `mlx_lm.models.cache`:
Inkling has **no RoPE**, so position information reaches attention only through
`distance = q_pos - k_pos` computed from array indices. `RotatingKVCache` is a
ring buffer whose physical slot order is rotated relative to temporal order —
every other family survives that because RoPE is baked into `k` before caching,
but here it would silently rotate the entire relative position bias. Our cache
keeps strict temporal order and tracks absolute positions.
See `docs/runtime/inkling-small/CACHE-CONTRACT.md`.

`SwitchGLU` is imported because it is the quantized packed-expert `gather_qmm`
wrapper shared by every JANG MoE family — not model or positional code.

Architecture (42 layers, 266B total / 12B active):
  - 35 sliding layers (window 512) + 7 full layers {5,11,17,23,29,35,41}
  - attention scale = 1/head_dim (NOT 1/sqrt) — q/k are per-head RMS-normed
  - content-conditioned relative position bias, banded to rel_extent
  - log attention scaling on full layers only, past 128k
  - 4 depthwise residual short convolutions per layer, fp32, state in the cache
  - 256 routed experts (top-6) + 2 router-gated shared experts
  - layers 0,1 dense MLP with a learned output global_scale
  - hidden /= 16.0 before the head; logits sliced to unpadded_vocab_size
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.switch_layers import SwitchGLU


# ─────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class ModelArgs:
    model_type: str = "inkling"

    hidden_size: int = 4096
    num_hidden_layers: int = 42
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6

    vocab_size: int = 201024
    unpadded_vocab_size: Optional[int] = 200058
    logits_mup_width_multiplier: float = 16.0
    use_embed_norm: bool = True

    # Relative position bias — there is no RoPE
    d_rel: int = 16
    rel_extent: int = 1024
    sliding_window_size: int = 512
    local_layer_ids: Optional[list] = None
    layer_types: Optional[list] = None

    # Log attention scaling (full-attention layers only)
    log_scaling_n_floor: Optional[float] = 128000.0
    log_scaling_alpha: float = 0.1

    sconv_kernel_size: int = 4

    # MoE
    n_routed_experts: int = 256
    num_experts_per_tok: int = 6
    n_shared_experts: int = 2
    intermediate_size: int = 2048          # == moe_intermediate_size
    dense_intermediate_size: int = 16384
    dense_mlp_idx: int = 2
    route_scale: float = 8.0

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.layer_types is None:
            if self.local_layer_ids is not None:
                local = set(self.local_layer_ids)
            else:
                # Reference fallback: every 6th layer is full attention.
                local = {i for i in range(self.num_hidden_layers) if (i + 1) % 6}
            self.layer_types = [
                "hybrid_sliding" if i in local else "hybrid"
                for i in range(self.num_hidden_layers)
            ]

    @property
    def moe_intermediate_size(self) -> int:
        return self.intermediate_size

    def is_sliding(self, i: int) -> bool:
        return self.layer_types[i] == "hybrid_sliding"

    def is_dense(self, i: int) -> bool:
        return i < self.dense_mlp_idx

    def rel_extent_for(self, i: int) -> int:
        """Sliding layers band the bias to the window; full layers use rel_extent."""
        return self.sliding_window_size if self.is_sliding(i) else self.rel_extent

    @classmethod
    def from_dict(cls, cfg: dict) -> "ModelArgs":
        """Accept either a bare text_config or the wrapped multimodal config."""
        if isinstance(cfg.get("text_config"), dict):
            merged: dict[str, Any] = {
                **{k: v for k, v in cfg.items() if k != "text_config"},
                **cfg["text_config"],
            }
        else:
            merged = dict(cfg)
        # HF attribute_map aliases
        for src, dst in (
            ("sliding_window", "sliding_window_size"),
            ("embedding_multiplier", "logits_mup_width_multiplier"),
            ("num_local_experts", "n_routed_experts"),
            ("conv_kernel_size", "sconv_kernel_size"),
            ("moe_intermediate_size", "intermediate_size"),
        ):
            if src in merged and dst not in merged:
                merged[dst] = merged[src]
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in merged.items() if k in known})


# ─────────────────────────────────────────────────────────────────────────
# Cache — temporal-ordered KV (windowed or full) + 4 short-conv states
# ─────────────────────────────────────────────────────────────────────────

class InklingLayerCache:
    """One layer's cache: KV in strict temporal order + 4 short-conv states.

    ``window=None`` → full attention (unbounded). ``window=N`` → only the last N
    entries are visible. The buffer over-allocates by ``step`` so the compaction
    copy is amortized instead of running on every decode step.

    Temporal order is a hard requirement here: with no RoPE the relative
    position bias is derived from ``q_pos - k_pos``, and ``k_pos`` comes from the
    slot index plus ``self.start``. A rotated buffer would rotate the bias — the
    model stays fluent for a few hundred tokens and then degrades.
    """

    def __init__(self, window: Optional[int] = None, step: int = 256):
        self.window = window
        self.step = step
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self.count = 0     # valid entries held in the buffer
        self.start = 0     # absolute position of buffer slot 0
        self.seen = 0      # total tokens ever processed by this layer
        self.conv: list = [None] * 4

    def _ensure_capacity(self, k: mx.array, v: mx.array, need: int) -> None:
        B, H, _, D = k.shape
        Dv = v.shape[-1]
        if self.keys is None:
            cap = ((need + self.step - 1) // self.step) * self.step
            self.keys = mx.zeros((B, H, cap, D), k.dtype)
            self.values = mx.zeros((B, H, cap, Dv), v.dtype)
        elif need > self.keys.shape[2]:
            extra = ((need - self.keys.shape[2] + self.step - 1) // self.step) * self.step
            self.keys = mx.concatenate(
                [self.keys, mx.zeros((B, H, extra, D), self.keys.dtype)], axis=2)
            self.values = mx.concatenate(
                [self.values, mx.zeros((B, H, extra, Dv), self.values.dtype)], axis=2)

    def _compact(self, incoming: int) -> None:
        """Drop entries a sliding layer can never reach again.

        Queries in the upcoming call sit at ``[seen, seen+T)``, and the oldest key
        any of them may attend is ``seen - window + 1``. So **window-1** prior
        entries must be retained — not ``window``. Retaining only ``window``
        *visible* entries after a multi-token chunk was a real bug: the earliest
        query in the chunk then lost keys it was still entitled to.

        Trimming is deferred until the buffer would exceed ``window-1+step`` so
        the copy is amortized over ``step`` decode steps instead of running every
        step.
        """
        if self.window is None:
            return
        keep_target = max(self.window - 1, 0)
        if self.count + incoming <= keep_target + self.step:
            return
        keep = min(keep_target, self.count)
        # A large prefill chunk leaves an oversized buffer behind. Reallocate
        # instead of holding it forever — otherwise a single 128k prefill would
        # pin a 128k-wide buffer on all 35 sliding layers and throw away the
        # entire memory advantage of the sliding design.
        oversized = self.keys is not None and self.keys.shape[2] > keep_target + 2 * self.step
        if keep > 0:
            tail_k = self.keys[:, :, self.count - keep:self.count, :]
            tail_v = self.values[:, :, self.count - keep:self.count, :]
            if oversized:
                self.keys, self.values = None, None
                self._ensure_capacity(tail_k, tail_v, keep)
            self.keys[:, :, :keep, :] = tail_k
            self.values[:, :, :keep, :] = tail_v
        elif oversized:
            self.keys, self.values = None, None
        self.start += self.count - keep
        self.count = keep

    def update(self, k: mx.array, v: mx.array):
        """Append k/v; return ``(visible_keys, visible_values, first_abs_pos)``."""
        T = k.shape[2]
        self._compact(T)
        self._ensure_capacity(k, v, self.count + T)
        self.keys[:, :, self.count:self.count + T, :] = k
        self.values[:, :, self.count:self.count + T, :] = v
        self.count += T
        self.seen += T

        # Visible span must cover every query in this call: the earliest query
        # reaches back window-1, so window-1+T entries, capped by what we hold.
        if self.window is None:
            visible = self.count
        else:
            visible = min(self.count, self.window - 1 + T)
        lo = self.count - visible
        return (
            self.keys[:, :, lo:self.count, :],
            self.values[:, :, lo:self.count, :],
            self.start + lo,
        )

    def conv_state(self, slot: int):
        return self.conv[slot]

    def set_conv_state(self, slot: int, state: mx.array) -> None:
        self.conv[slot] = state


class InklingCache:
    """Whole-model cache: one ``InklingLayerCache`` per layer."""

    def __init__(self, args: ModelArgs, step: int = 256):
        self.layers = [
            InklingLayerCache(
                window=args.sliding_window_size if args.is_sliding(i) else None,
                step=step,
            )
            for i in range(args.num_hidden_layers)
        ]

    def __getitem__(self, i: int) -> InklingLayerCache:
        return self.layers[i]

    def __len__(self) -> int:
        return len(self.layers)

    @property
    def offset(self) -> int:
        return self.layers[0].seen if self.layers else 0


# ─────────────────────────────────────────────────────────────────────────
# Primitives
# ─────────────────────────────────────────────────────────────────────────

class InklingRMSNorm(nn.Module):
    """Plain RMSNorm, fp32 variance. NO +1 shift — this is not a Gemma-style norm."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class ShortConv(nn.Module):
    """Depthwise causal Conv1d, kernel 4, no bias, RESIDUAL, computed in fp32.

    ``out = conv1d(x) + x``. The reference marks all four sconv modules
    ``_keep_in_fp32_modules_strict``, hence the fp32 compute.

    Weight layout is MLX ``(C, K, 1)``; the converter transposes the checkpoint's
    torch grouped-conv ``(C, 1, K)`` in ``ssm_layout.prepare_mlx_passthrough_tensor``.
    Explicit left-padding plus ``padding=0`` lets one code path serve both prefill
    and single-token decode. The saved state is the last ``K-1`` **pre-conv
    inputs**, never outputs.
    """

    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.weight = mx.zeros((channels, kernel_size, 1))

    def __call__(self, x: mx.array, cache: Optional[InklingLayerCache] = None,
                 slot: int = 0) -> mx.array:
        pad = self.kernel_size - 1
        xf = x.astype(mx.float32)                      # [B, T, C]
        state = cache.conv_state(slot) if cache is not None else None
        if state is not None:
            xp = mx.concatenate([state.astype(mx.float32), xf], axis=1)
        else:
            xp = mx.pad(xf, [(0, 0), (pad, 0), (0, 0)])
        if cache is not None:
            cache.set_conv_state(slot, xp[:, -pad:, :])
        y = mx.conv1d(xp, self.weight.astype(mx.float32), padding=0, groups=self.channels)
        return (y + xf).astype(x.dtype)


class RelLogits(nn.Module):
    """Content-conditioned relative position bias. Inkling has no RoPE.

    ``proj`` is a learned bank of bias-vs-distance profiles ``[d_rel, rel_extent]``;
    each token mixes them into one bias value per backward distance. Exactly zero
    outside ``0 <= distance < rel_extent``. Causality and padding stay in the mask.
    """

    def __init__(self, d_rel: int, rel_extent: int):
        super().__init__()
        self.rel_extent = rel_extent
        self.proj = mx.zeros((d_rel, rel_extent))

    def __call__(self, r: mx.array, q_pos: mx.array, k_pos: mx.array) -> mx.array:
        # r: [B, T, H, d_rel] -> rel: [B, H, T, rel_extent]
        rel = (r @ self.proj.astype(r.dtype)).transpose(0, 2, 1, 3)
        d = q_pos[:, None] - k_pos[None, :]                       # [T, kv]
        idx = mx.clip(d, 0, self.rel_extent - 1)
        B, H = rel.shape[0], rel.shape[1]
        idx_b = mx.broadcast_to(idx[None, None], (B, H) + d.shape)
        bias = mx.take_along_axis(rel, idx_b, axis=-1)
        in_band = (d >= 0) & (d < self.rel_extent)
        return mx.where(in_band[None, None], bias, mx.zeros_like(bias))


# ─────────────────────────────────────────────────────────────────────────
# Attention
# ─────────────────────────────────────────────────────────────────────────

class InklingAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_sliding = args.is_sliding(layer_idx)
        self.n_heads = args.num_attention_heads
        self.n_kv = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.d_rel = args.d_rel
        self.window = args.sliding_window_size if self.is_sliding else None
        self.rel_extent = args.rel_extent_for(layer_idx)

        # q/k are per-head RMS-normalized, so the scale is 1/d, NOT 1/sqrt(d).
        # 1/sqrt(128) would inflate every logit by 11.31x -> instant garbage.
        # Do not "fix" this line.
        self.scale = 1.0 / float(self.head_dim)

        self.log_n_floor = args.log_scaling_n_floor
        self.log_alpha = args.log_scaling_alpha

        d = args.hidden_size
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        self.r_proj = nn.Linear(d, self.n_heads * self.d_rel, bias=False)

        self.k_sconv = ShortConv(self.n_kv * self.head_dim, args.sconv_kernel_size)
        self.v_sconv = ShortConv(self.n_kv * self.head_dim, args.sconv_kernel_size)
        self.q_norm = InklingRMSNorm(self.head_dim, args.rms_norm_eps)
        self.k_norm = InklingRMSNorm(self.head_dim, args.rms_norm_eps)
        self.rel_logits_proj = RelLogits(self.d_rel, self.rel_extent)

    def __call__(self, x: mx.array, cache: Optional[InklingLayerCache] = None) -> mx.array:
        B, T, _ = x.shape

        q = self.q_proj(x)
        k = self.k_sconv(self.k_proj(x), cache, slot=0)
        v = self.v_sconv(self.v_proj(x), cache, slot=1)
        r = self.r_proj(x).reshape(B, T, self.n_heads, self.d_rel)

        q = self.q_norm(q.reshape(B, T, self.n_heads, self.head_dim)).transpose(0, 2, 1, 3)
        k = self.k_norm(k.reshape(B, T, self.n_kv, self.head_dim)).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_kv, self.head_dim).transpose(0, 2, 1, 3)

        # Absolute positions. `seen` must be read BEFORE the cache update.
        seen = cache.seen if cache is not None else 0
        q_pos = mx.arange(seen, seen + T)
        if cache is not None:
            k, v, k_start = cache.update(k, v)
        else:
            k_start = 0
        k_pos = mx.arange(k_start, k_start + k.shape[2])

        bias = self.rel_logits_proj(r, q_pos, k_pos)

        # Log attention scaling: FULL-attention layers only, in fp32, applied to
        # BOTH q and the position bias. Exactly 1.0 below n_floor, so its absence
        # is invisible in any short-context test.
        if (not self.is_sliding) and self.log_n_floor:
            tau = 1.0 + self.log_alpha * mx.log(
                mx.maximum((q_pos + 1).astype(mx.float32) / float(self.log_n_floor), 1.0)
            )
            tau = tau.reshape(1, 1, T, 1)
            q = (q.astype(mx.float32) * tau).astype(q.dtype)
            bias = (bias.astype(mx.float32) * tau).astype(bias.dtype)

        # Fold causal (+ sliding-window) validity into the same additive tensor as
        # the bias so mx.fast SDPA keeps its fast path.
        d = q_pos[:, None] - k_pos[None, :]
        valid = d >= 0
        if self.window is not None:
            valid = valid & (d < self.window)
        additive = mx.where(
            valid[None, None], bias, mx.full(bias.shape, -1e9, bias.dtype)
        )

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=additive)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, T, -1))


# ─────────────────────────────────────────────────────────────────────────
# MLP / MoE
# ─────────────────────────────────────────────────────────────────────────

class InklingDenseMLP(nn.Module):
    """Dense GLU MLP (layers 0..dense_mlp_idx-1) with a learned output scale."""

    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        self.global_scale = mx.ones((1,))

    def __call__(self, x: mx.array) -> mx.array:
        y = self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))
        return y * self.global_scale.astype(y.dtype)


class InklingRouter(nn.Module):
    """258-row router: 256 routed + 2 shared-expert rows (`shared_expert_sink`).

    Four load-bearing details:
      1. ``bias`` (e_score_correction_bias) steers SELECTION ONLY; the mixing
         weights come from the RAW logits.
      2. The normalization pool is the top_k routed AND the shared rows together
         (`norm_after_topk`), not the routed ones alone.
      3. ``route_scale`` (8.0) and a learned ``global_scale`` both multiply.
      4. Weights are a softmax over LOG-SIGMOID values — not a plain softmax over
         logits, and not a renormalized top-k of the sigmoid scores.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_routed = args.n_routed_experts
        self.n_shared = args.n_shared_experts
        self.top_k = args.num_experts_per_tok
        self.route_scale = args.route_scale
        self.weight = mx.zeros((self.n_routed + self.n_shared, args.hidden_size))
        self.bias = mx.zeros((self.n_routed,))
        self.global_scale = mx.ones((1,))

    def __call__(self, flat: mx.array):
        # Keep routing stable across BF16 matmul implementations. The expert
        # boundary margins become extremely narrow in late layers; MLX and
        # Torch BF16 kernels can otherwise select different top-6 sets even
        # when their dense outputs have cosine > 0.99999. Only this tiny
        # 4096x258 projection is promoted; attention and expert GEMMs stay BF16.
        logits = flat.astype(mx.float32) @ self.weight.astype(mx.float32).T
        routed_logits = logits[..., : self.n_routed]
        shared_logits = logits[..., self.n_routed:]

        # Selection: sigmoid(scores) + bias. The weights below deliberately do
        # NOT use this biased quantity.
        sel = mx.sigmoid(routed_logits) + self.bias
        inds = mx.argpartition(-sel, self.top_k - 1, axis=-1)[..., : self.top_k]

        tk = mx.concatenate(
            [mx.take_along_axis(routed_logits, inds, axis=-1), shared_logits], axis=-1
        )
        # logsigmoid(x) = -softplus(-x); log(sigmoid(x)) underflows at large |x|.
        logp = -mx.logaddexp(mx.zeros_like(tk), -tk)
        w = mx.exp(logp - mx.logsumexp(logp, axis=-1, keepdims=True))
        w = w * self.route_scale * self.global_scale.astype(mx.float32)
        return inds, w[..., : self.top_k], w[..., self.top_k:]


class TQSwitchGLU(nn.Module):
    """SwitchGLU over JANGTQ (Lloyd-Max codebook) experts.

    Same composition as mlx_lm's SwitchGLU -- silu(gate(x)) * up(x) -> down --
    but each leg is a `turboquant.tq_kernel.TurboQuantSwitchLinear`, which gathers
    and matmuls straight from the packed uint32 codes in a fused Metal kernel.
    Note `turboquant.linear.TurboQuantSwitchLinear` is a same-named REFERENCE
    implementation that dequantizes every active expert in a Python loop -- it
    must not be used at runtime.

    Legs can carry different widths: this build is gate/up tq2 + down tq4.
    """

    def __init__(self, hidden: int, inter: int, num_experts: int,
                 gate_bits: int = 2, up_bits: int = 2, down_bits: int = 4,
                 seed: int = 42):
        super().__init__()
        from ..turboquant.tq_kernel import TurboQuantSwitchLinear as TQSL
        self.gate_proj = TQSL(hidden, inter, num_experts, bits=gate_bits, seed=seed)
        self.up_proj = TQSL(hidden, inter, num_experts, bits=up_bits, seed=seed)
        self.down_proj = TQSL(inter, hidden, num_experts, bits=down_bits, seed=seed)

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))
        g = self.gate_proj(x, indices)
        u = self.up_proj(x, indices)
        return self.down_proj(nn.silu(g) * u, indices).squeeze(-2)


class InklingMoE(nn.Module):
    def __init__(self, args: ModelArgs, tq: Optional[dict] = None):
        super().__init__()
        self.n_shared = args.n_shared_experts
        self.gate = InklingRouter(args)
        if tq:
            self.switch_mlp = TQSwitchGLU(
                args.hidden_size, args.moe_intermediate_size, args.n_routed_experts,
                gate_bits=tq.get("gate_proj", 2), up_bits=tq.get("up_proj", 2),
                down_bits=tq.get("down_proj", 4), seed=tq.get("seed", 42))
        else:
            self.switch_mlp = SwitchGLU(
                args.hidden_size, args.moe_intermediate_size, args.n_routed_experts, bias=False
            )
        # Shared experts are a fixed-index 2-expert SwitchGLU. The reference
        # multiplies gammas onto the activated intermediate; down_proj is linear,
        # so scaling the output is mathematically identical.
        self.shared_experts = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, self.n_shared, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        B, T, H = x.shape
        flat = x.reshape(-1, H)
        inds, routed_w, gammas = self.gate(flat)

        y = self.switch_mlp(flat, inds)                           # [N, k, H]
        # The reference casts every weighted expert contribution back to the
        # activation dtype before index_add_ into a BF16 destination.  Keeping
        # the whole top-k reduction in FP32 changes the residual stream at
        # every MoE layer and is not an accuracy-preserving substitution.
        y = (y.astype(mx.float32) * routed_w[..., None]).astype(x.dtype)
        y = y.sum(axis=-2).astype(x.dtype)

        sh_inds = mx.broadcast_to(
            mx.arange(self.n_shared)[None], (flat.shape[0], self.n_shared)
        )
        s = self.shared_experts(flat, sh_inds)                    # [N, n_shared, H]
        # fp32 accumulation across shared experts, per the reference.
        s = (s.astype(mx.float32) * gammas[..., None]).sum(axis=-2).astype(x.dtype)

        return (y + s).astype(x.dtype).reshape(B, T, H)


# ─────────────────────────────────────────────────────────────────────────
# Layer / model
# ─────────────────────────────────────────────────────────────────────────

class InklingDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int, tq: Optional[dict] = None):
        super().__init__()
        self.self_attn = InklingAttention(args, layer_idx)
        self.mlp = (
            InklingDenseMLP(args.hidden_size, args.dense_intermediate_size)
            if args.is_dense(layer_idx)
            else InklingMoE(args, tq=tq)
        )
        self.input_layernorm = InklingRMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_attention_layernorm = InklingRMSNorm(args.hidden_size, args.rms_norm_eps)
        self.attn_sconv = ShortConv(args.hidden_size, args.sconv_kernel_size)
        self.mlp_sconv = ShortConv(args.hidden_size, args.sconv_kernel_size)

    def __call__(self, x: mx.array, cache: Optional[InklingLayerCache] = None) -> mx.array:
        # The sconv sits BETWEEN the sublayer and the residual add — it is not on
        # the residual stream.
        h = self.self_attn(self.input_layernorm(x), cache)
        x = x + self.attn_sconv(h, cache, slot=2)
        h = self.mlp(self.post_attention_layernorm(x))
        x = x + self.mlp_sconv(h, cache, slot=3)
        return x


class InklingModel(nn.Module):
    def __init__(self, args: ModelArgs, tq: Optional[dict] = None):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        if args.use_embed_norm:
            self.embed_norm = InklingRMSNorm(args.hidden_size, args.rms_norm_eps)
        self.layers = [InklingDecoderLayer(args, i, tq=tq)
                       for i in range(args.num_hidden_layers)]
        self.norm = InklingRMSNorm(args.hidden_size, args.rms_norm_eps)

    def __call__(self, inputs: Optional[mx.array], cache: Optional[InklingCache] = None,
                 inputs_embeds: Optional[mx.array] = None) -> mx.array:
        h = inputs_embeds if inputs_embeds is not None else self.embed_tokens(inputs)
        if self.args.use_embed_norm:
            h = self.embed_norm(h)
        for i, layer in enumerate(self.layers):
            h = layer(h, cache[i] if cache is not None else None)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs, tq: Optional[dict] = None):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.tq = tq
        self.model = InklingModel(args, tq=tq)
        # `embed` and `unembed` are separate tensors in the checkpoint, never tied.
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: Optional[mx.array], cache: Optional[InklingCache] = None,
                 inputs_embeds: Optional[mx.array] = None) -> mx.array:
        h = self.model(inputs, cache, inputs_embeds=inputs_embeds)
        # muP logit scaling — MANDATORY. Omitting it leaves every logit 16x too
        # large: the softmax saturates and temperature/top-p stop having any
        # effect, which reads as "confidently repetitive" rather than broken.
        h = h / self.args.logits_mup_width_multiplier
        logits = self.lm_head(h)
        u = self.args.unpadded_vocab_size
        if u is not None and u < logits.shape[-1]:
            # Rows u..vocab_size are untrained padding and must not be samplable.
            logits = logits[..., :u]
        return logits

    def make_cache(self, step: int = 256) -> InklingCache:
        return InklingCache(self.args, step=step)

    @property
    def layers(self):
        return self.model.layers

    # --- weight bridge ----------------------------------------------------
    @staticmethod
    def map_key(key: str) -> str:
        """Native checkpoint/bundle key -> this module tree's key.

        Shared by `sanitize()` (weight tensors) and the loader (quantization-map
        entries), so the two can never drift. Handles the `.weight`/`.scales`/
        `.biases` triple a quantized bundle stores per module.
        """
        n = key.replace("model.llm.", "model.")
        # Append a sentinel dot so every rewrite below can be written with a
        # trailing dot and match uniformly on BOTH forms we get handed:
        #   full tensor keys  `...attn.wq_du.weight`   (from safetensors)
        #   bare module bases `...attn.wq_du`          (from the quantization map)
        # Without it the bare form silently skips every dotted rule -- which is
        # how `attn_norm` first came through unrenamed.
        n = n + "."
        n = n.replace(".attn.wq_du.", ".self_attn.q_proj.")
        n = n.replace(".attn.wk_dv.", ".self_attn.k_proj.")
        n = n.replace(".attn.wv_dv.", ".self_attn.v_proj.")
        n = n.replace(".attn.wo_ud.", ".self_attn.o_proj.")
        n = n.replace(".attn.wr_du.", ".self_attn.r_proj.")
        n = n.replace(".attn_norm.", ".input_layernorm.")
        n = n.replace(".mlp_norm.", ".post_attention_layernorm.")
        n = n.replace(".attn.", ".self_attn.")      # q_norm/k_norm/sconv/rel_logits
        n = n[:-1]
        if n == "model.embed.weight":
            n = "model.embed_tokens.weight"
        elif n == "model.embed":
            n = "model.embed_tokens"
        elif n == "model.unembed.weight":
            n = "lm_head.weight"
        elif n == "model.unembed":
            n = "lm_head"
        return n

    @staticmethod
    def _split_suffix(name: str) -> tuple[str, str]:
        """Split a bundle key into (module_base, tensor_suffix)."""
        for suf in (".weight", ".scales", ".biases", ".tq_packed", ".tq_norms",
                    ".tq_bits"):
            if name.endswith(suf):
                return name[: -len(suf)], suf
        return name, ""

    @staticmethod
    def _deinterleave_w13(v: mx.array, rows_per_leg: int) -> tuple[mx.array, mx.array]:
        """Split Inkling's native row-interleaved gate/up representation.

        The checkpoint stores ``w13`` rows as ``g0, u0, g1, u1, ...`` rather
        than as two contiguous ``[gate; up]`` halves.  The row dimension is
        second-to-last for both dense and stacked expert tensors.  Affine
        packing only changes the last (input) dimension, so the identical
        transform applies to packed weights, scales, and biases.
        """
        if getattr(v, "ndim", 0) < 2 or v.shape[-2] != 2 * rows_per_leg:
            raise ValueError(
                "invalid Inkling w13 shape: expected second-to-last dimension "
                f"{2 * rows_per_leg}, got {getattr(v, 'shape', None)}"
            )
        paired = v.reshape(*v.shape[:-2], rows_per_leg, 2, v.shape[-1])
        # Materialize contiguous halves.  Leaving the even/odd views strided
        # makes gather_qmm/gather_mm take a much slower path.
        return mx.contiguous(paired[..., 0, :]), mx.contiguous(paired[..., 1, :])

    def sanitize(self, weights: dict) -> dict:
        """Map native Inkling checkpoint/bundle keys onto this module tree.

        The bundle keeps the checkpoint's own names (`wq_du`, `unembed`, fused
        `w13_weight`, ...). Renaming inside the converter has been a recurring
        source of bugs (project_convert_lm_head_vl_prefix_bug), so the remap
        lives here where it is unit-testable.
        """
        inter = self.args.moe_intermediate_size
        dense_inter = self.args.dense_intermediate_size
        out: dict = {}

        for key, v in weights.items():
            # Multimodal towers are not part of the text runtime. Note the
            # converter renames the vision tower (`model.visual.*` ->
            # `vision_tower.*` via the VL sanitizer) but leaves audio as
            # `model.audio.*`, so both spellings have to be covered.
            if key.startswith(("model.visual", "model.audio",
                               "vision_tower", "audio_tower")):
                continue
            if key.startswith("model.mtp") or ".mtp." in key:
                continue    # MTP heads have no reference implementation

            base, suf = self._split_suffix(key)
            if suf == ".tq_bits":
                continue          # width lives in jang_config, not the module tree
            # JANGTQ stores `<mod>.tq_packed` / `.tq_norms`; the module's
            # parameters are `packed` / `norms`.
            suf = {".tq_packed": ".packed", ".tq_norms": ".norms"}.get(suf, suf)
            n = self.map_key(base) + suf

            # Native fused w13 -> gate_proj / up_proj. Rows are interleaved
            # gate/up pairs, not contiguous halves. Groups run along the LAST
            # (input) axis, so de-interleaving the output-row axis is valid for
            # packed weights and scales/biases alike.
            if base.endswith("mlp.experts.w13_weight"):
                b = self.map_key(base[: -len("experts.w13_weight")])
                gate, up = Model._deinterleave_w13(v, inter)
                out[b + "switch_mlp.gate_proj" + (suf or ".weight")] = gate
                out[b + "switch_mlp.up_proj" + (suf or ".weight")] = up
                continue
            if base.endswith("mlp.experts.w2_weight"):
                b = self.map_key(base[: -len("experts.w2_weight")])
                out[b + "switch_mlp.down_proj" + (suf or ".weight")] = v
                continue
            if base.endswith("shared_experts.shared_w13_weight"):
                b = self.map_key(base[: -len("shared_w13_weight")])
                gate, up = Model._deinterleave_w13(v, inter)
                out[b + "gate_proj" + (suf or ".weight")] = gate
                out[b + "up_proj" + (suf or ".weight")] = up
                continue
            if base.endswith("shared_experts.shared_w2_weight"):
                b = self.map_key(base[: -len("shared_w2_weight")])
                out[b + "down_proj" + (suf or ".weight")] = v
                continue
            # Dense MLP fused w13 (layers 0,1) uses the same row interleave.
            if base.endswith("mlp.w13_dn"):
                b = self.map_key(base[: -len("w13_dn")])
                gate, up = Model._deinterleave_w13(v, dense_inter)
                out[b + "gate_proj" + (suf or ".weight")] = gate
                out[b + "up_proj" + (suf or ".weight")] = up
                continue
            if base.endswith("mlp.w2_md"):
                b = self.map_key(base[: -len("w2_md")])
                out[b + "down_proj" + (suf or ".weight")] = v
                continue

            # sconv: MLX wants (C, K, 1). Tolerate a bundle still in torch (C, 1, K).
            if n.endswith("_sconv.weight") and getattr(v, "ndim", 0) == 3 and v.shape[-1] != 1:
                v = v.transpose(0, 2, 1)

            out[n] = v
        return out
