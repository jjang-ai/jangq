"""dots3_note numerics core (MLX) — mirrors transformers PR #47844 exactly.

All functions are pure: they take a layer-weight dict `lw` mapping short names
to mx.arrays (already fp8-dequantized upstream). Used by both the streaming
source executor and (via adapters) the quantized-bundle runtime, so there is
exactly ONE implementation of the math.

Weight dict keys per decoder layer (checkpoint names minus the prefix):
  input_layernorm, post_attention_layernorm,
  q_a_proj, q_a_layernorm, q_b_proj,
  kv_a_proj_with_mqa, kv_a_layernorm, kv_b_proj,
  k_rope_only_layernorm, o_proj, g_proj,
  MoE:   gate_w, gate_bias(e_score f32), experts_gate, experts_up, experts_down
         (stacked [E, out, in]), shared_gate, shared_up, shared_down
  dense: mlp_gate, mlp_up, mlp_down

DSA indexer is NOT applied: for sequences <= index_topk (2048) the top-k
selects every causal position, so dense attention is mathematically identical.
Calls assert this bound; long-context indexer support is a later, separate step.
"""
from __future__ import annotations

import math

import mlx.core as mx

from .config import AttnGeom, Dots3Config


def rms_norm(x: mx.array, w: mx.array, eps: float) -> mx.array:
    return mx.fast.rms_norm(x, w, eps)


class QW:
    """Quantized weight triple in stock MLX affine/mxfp storage."""

    __slots__ = ("wq", "scales", "biases", "group_size", "bits", "mode")

    def __init__(self, wq, scales, biases, group_size, bits, mode="affine"):
        self.wq, self.scales, self.biases = wq, scales, biases
        self.group_size, self.bits, self.mode = group_size, bits, mode


def linear(x: mx.array, w) -> mx.array:
    """w stored (out, in) like the checkpoint; plain array or QW."""
    if isinstance(w, QW):
        kwargs = dict(transpose=True, group_size=w.group_size, bits=w.bits,
                      mode=w.mode)
        if w.biases is not None:
            return mx.quantized_matmul(x, w.wq, w.scales, w.biases, **kwargs)
        return mx.quantized_matmul(x, w.wq, w.scales, **kwargs)
    return x @ w.T


def _causal_mask(S: int, window: int | None, dtype,
                 past: int = 0) -> mx.array:
    """Additive mask [S, past+S]. window=None -> plain causal; else sliding
    causal (kv in [q-window+1, q], matching HF: q_idx - kv_idx < window)."""
    q = (past + mx.arange(S))[:, None]
    k = mx.arange(past + S)[None, :]
    allowed = k <= q
    if window is not None:
        allowed = mx.logical_and(allowed, (q - k) < window)
    return mx.where(allowed, mx.array(0.0, dtype=dtype),
                    mx.array(-mx.inf, dtype=dtype))


def mla_attention(x: mx.array, lw: dict, g: AttnGeom, eps: float,
                  rescale: bool, k_rope_norm: bool,
                  mask: mx.array | None = None,
                  cache: dict | None = None) -> mx.array:
    """x: [B, S, H]. MLA with decoupled interleaved rope, optional low-rank
    rescale, k-rope-only norm, sigmoid head gates. `cache` (optional) is a
    per-layer dict which accumulates materialized keys/values; sliding-window
    layers keep only the last window-1 past entries."""
    B, S, H = x.shape
    past = 0 if cache is None else cache.get("len", 0)

    q_lora = rms_norm(linear(x, lw["q_a_proj"]), lw["q_a_layernorm"], eps)
    if rescale:
        q_lora = q_lora * math.sqrt(H / g.q_lora_rank)
    q = linear(q_lora, lw["q_b_proj"])
    q = q.reshape(B, S, g.num_heads, g.qk_head_dim).transpose(0, 2, 1, 3)
    q_nope = q[..., : g.qk_nope_head_dim]
    q_pe = q[..., g.qk_nope_head_dim:]

    latent = linear(x, lw["kv_a_proj_with_mqa"])
    kv_a = latent[..., : g.kv_lora_rank]
    k_pe = latent[..., g.kv_lora_rank:]
    kv_a = rms_norm(kv_a, lw["kv_a_layernorm"], eps)
    if rescale:
        kv_a = kv_a * math.sqrt(H / g.kv_lora_rank)
    kv = linear(kv_a, lw["kv_b_proj"])
    kv = kv.reshape(B, S, g.num_heads, g.qk_nope_head_dim + g.v_head_dim)
    kv = kv.transpose(0, 2, 1, 3)
    k_nope = kv[..., : g.qk_nope_head_dim]
    v = kv[..., g.qk_nope_head_dim:]

    k_pe = k_pe.reshape(B, 1, S, g.qk_rope_head_dim)
    if k_rope_norm:
        k_pe = rms_norm(k_pe, lw["k_rope_only_layernorm"], eps)

    q_pe = mx.fast.rope(q_pe, g.qk_rope_head_dim, traditional=True,
                        base=g.rope_theta, scale=1.0, offset=past)
    k_pe = mx.fast.rope(k_pe, g.qk_rope_head_dim, traditional=True,
                        base=g.rope_theta, scale=1.0, offset=past)

    queries = mx.concatenate([q_nope, q_pe], axis=-1)
    keys = mx.concatenate(
        [k_nope, mx.broadcast_to(k_pe, (B, g.num_heads, S, g.qk_rope_head_dim))],
        axis=-1)

    if cache is not None:
        if past:
            keys = mx.concatenate([cache["k"], keys], axis=2)
            v = mx.concatenate([cache["v"], v], axis=2)
        cache["len"] = past + S
        if g.sliding_window is not None and keys.shape[2] > g.sliding_window:
            cache["k"] = keys[:, :, -(g.sliding_window - 1):]
            cache["v"] = v[:, :, -(g.sliding_window - 1):]
        else:
            cache["k"], cache["v"] = keys, v
        # NB: with a trimmed sliding cache the additive mask below is built
        # against the PHYSICAL key length, whose oldest entry is exactly
        # window-1 back — plain causal over it is equivalent.
        eff_past = keys.shape[2] - S
        mask_len_window = g.sliding_window if eff_past + S > (
            g.sliding_window or 1 << 30) else None
        if mask is None and S > 1:
            mask = _causal_mask(S, mask_len_window, mx.float32, past=eff_past)
        elif mask is None:
            mask = None
    elif mask is None:
        mask = _causal_mask(S, g.sliding_window, mx.float32)
    out = mx.fast.scaled_dot_product_attention(
        queries, keys, v, scale=g.scale, mask=mask)
    out = out.transpose(0, 2, 1, 3)                      # [B, S, heads, v]

    gate = mx.sigmoid(linear(x, lw["g_proj"]))
    if g.gate_type == "headwise":
        out = out * gate[..., None]
    else:                                                # elementwise
        out = out * gate.reshape(B, S, g.num_heads, g.v_head_dim)

    out = out.reshape(B, S, g.num_heads * g.v_head_dim)
    return linear(out, lw["o_proj"])


def route(x2d: mx.array, gate_w: mx.array, e_bias: mx.array, top_k: int,
          norm_topk: bool, rsf: float) -> tuple[mx.array, mx.array]:
    """noaux_tc with n_group=1 => plain top-k of sigmoid scores + bias for
    SELECTION; returned weights use unbiased scores. x2d: [T, H].
    Gating runs in the ACTIVATION dtype (bf16 in deployment, f32 in the
    golden gate) — matching the reference's model-dtype F.linear."""
    logits = x2d @ gate_w.astype(x2d.dtype).T
    scores = mx.sigmoid(logits).astype(mx.float32)
    choice = scores + e_bias.astype(mx.float32)[None, :]
    inds = mx.argpartition(-choice, kth=top_k - 1, axis=-1)[..., :top_k]
    w = mx.take_along_axis(scores, inds, axis=-1)
    if norm_topk:
        w = w / (w.sum(axis=-1, keepdims=True) + 1e-20)
    return inds.astype(mx.uint32), (w * rsf)


SORT_THRESHOLD = 64          # experts_apply switches to the sorted kernel here


def _gmm(x: mx.array, w, rhs_indices: mx.array,
         lhs_indices: mx.array | None = None,
         sorted_indices: bool = False) -> mx.array:
    """gather matmul for stacked experts; w plain [E,out,in] or QW.

    🚨 `sorted_indices=True` is only valid when BOTH lhs_indices and
    rhs_indices are supplied. Passing the flag with rhs_indices alone
    silently computes the WRONG result (measured 135% error) — it does not
    raise. See ops_selftest / BUILD-LOG 2026-08-15.
    """
    if sorted_indices and lhs_indices is None:
        raise ValueError("sorted_indices=True requires lhs_indices")
    if isinstance(w, QW):
        kwargs = dict(transpose=True, group_size=w.group_size, bits=w.bits,
                      mode=w.mode, rhs_indices=rhs_indices,
                      sorted_indices=sorted_indices)
        if lhs_indices is not None:
            kwargs["lhs_indices"] = lhs_indices
        if w.biases is not None:
            return mx.gather_qmm(x, w.wq, w.scales, w.biases, **kwargs)
        return mx.gather_qmm(x, w.wq, w.scales, **kwargs)
    kwargs = dict(rhs_indices=rhs_indices, sorted_indices=sorted_indices)
    if lhs_indices is not None:
        kwargs["lhs_indices"] = lhs_indices
    return mx.gather_mm(x, w.swapaxes(-1, -2), **kwargs)


def experts_apply(x2d: mx.array, inds: mx.array, weights: mx.array,
                  w_gate, w_up, w_down, sort: bool = True) -> mx.array:
    """Stacked routed experts via gather_mm/gather_qmm. x2d [T,H], inds [T,K],
    weights [T,K]; w_* stacked [E, out, in] (plain or QW). Returns [T, H]."""
    T, K = inds.shape
    flat = inds.flatten()
    xa = x2d[:, None, :]                                 # [T, 1, H]
    if sort and T >= SORT_THRESHOLD:
        # Sort by expert so the gather kernel touches each expert's weights
        # once. BOTH index arrays must be supplied and expert-sorted.
        order = mx.argsort(flat)
        inv = mx.argsort(order)
        rhs = flat[order]
        lhs = (order // K).astype(mx.uint32)
        gt = _gmm(xa, w_gate, rhs, lhs, True)
        up = _gmm(xa, w_up, rhs, lhs, True)
        h = mx.multiply(gt * mx.sigmoid(gt), up)         # silu(g)*u  [TK,1,I]
        ident = mx.arange(T * K, dtype=mx.uint32)        # h rows already sorted
        dn = _gmm(h, w_down, rhs, ident, True)
        dn = dn[inv].reshape(T, K, -1)
    else:
        lhs = mx.array(mx.arange(T * K) // K, dtype=mx.uint32)
        gt = _gmm(xa, w_gate, flat, lhs)
        up = _gmm(xa, w_up, flat, lhs)
        h = mx.multiply(gt * mx.sigmoid(gt), up)
        ident = mx.arange(T * K, dtype=mx.uint32)
        dn = _gmm(h, w_down, flat, ident)
        dn = dn.reshape(T, K, -1)
    return (dn * weights[..., None].astype(dn.dtype)).sum(axis=1)


def mlp_dense(x: mx.array, gate: mx.array, up: mx.array, down: mx.array) -> mx.array:
    g = linear(x, gate)
    return linear(mx.multiply(g * mx.sigmoid(g), linear(x, up)), down)


def moe_forward(x: mx.array, lw: dict, cfg: Dots3Config,
                capture: dict | None = None) -> mx.array:
    B, S, H = x.shape
    x2d = x.reshape(-1, H)
    inds, w = route(x2d, lw["gate_w"], lw["gate_bias"],
                    cfg.num_experts_per_tok, cfg.norm_topk_prob,
                    cfg.routed_scaling_factor)
    if capture is not None:
        capture["router_inds"] = inds
        capture["router_weights"] = w
    out = experts_apply(x2d, inds, w, lw["experts_gate"], lw["experts_up"],
                        lw["experts_down"])
    out = out + mlp_dense(x2d, lw["shared_gate"], lw["shared_up"],
                          lw["shared_down"])
    return out.reshape(B, S, H).astype(x.dtype)


def decoder_layer(x: mx.array, lw: dict, cfg: Dots3Config, layer_idx: int,
                  mask: mx.array | None = None,
                  capture: dict | None = None) -> mx.array:
    g = cfg.geom(layer_idx)
    if x.shape[1] > cfg.index_topk and not cfg.is_sliding(layer_idx):
        raise ValueError(
            f"seq {x.shape[1]} > index_topk {cfg.index_topk}: DSA indexer "
            "required; dense equivalence no longer holds")
    h = rms_norm(x, lw["input_layernorm"], cfg.rms_norm_eps)
    if capture is not None:
        capture["attn_in"] = h
    x = x + mla_attention(h, lw, g, cfg.rms_norm_eps, cfg.apply_lora_rescale,
                          cfg.k_rope_only_layernorm, mask=mask)
    h = rms_norm(x, lw["post_attention_layernorm"], cfg.rms_norm_eps)
    if capture is not None:
        capture["mlp_in"] = h
    if cfg.is_moe(layer_idx):
        x = x + moe_forward(h, lw, cfg, capture=capture)
    else:
        B, S, H = h.shape
        x = x + mlp_dense(h.reshape(-1, H), lw["mlp_gate"], lw["mlp_up"],
                          lw["mlp_down"]).reshape(B, S, H).astype(x.dtype)
    return x
