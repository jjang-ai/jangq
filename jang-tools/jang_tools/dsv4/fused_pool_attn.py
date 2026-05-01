"""A7 — fused HSA/CSA SDPA-with-pool-and-sinks attention.

Today: Python orchestrator that wraps the existing 6-step pipeline into
one function call. Bit-identical to the inline path; just easier to
swap with a Metal kernel later.

Future: a single `mx.fast.metal_kernel` that takes packed q + k_window +
k_pool + v_window + v_pool + topk_idx + sink + scale → produces the
attention output. Saves 5 of 6 dispatches per layer × 41 layers ≈ 205
dispatches/token → ~15-25% decode speedup on the long-ctx path.

Why orchestrator first:
  1. correct math NOW; can A/B vs the inline path immediately
  2. test surface to validate the future kernel against
  3. clean call-site replacement: `attention.__call__` becomes a one-liner
  4. failure mode is "back to inline" not "broken"

API contract:
    fused_pool_attention(
        q,                          # (B, n_heads, L, head_dim)
        k_window, v_window,         # (B, 1, win_len, head_dim)
        k_pool, v_pool,             # (B, 1, P, head_dim)  P may be 0
        topk_idx,                   # (B, L, K) or None — None = use full pool
        attn_sink,                  # (n_heads,) or None
        win_mask, comp_mask,        # bool masks; comp_mask may be None when topk_idx is set
        scale,                      # 1/sqrt(head_dim) typically
        sliding_window,             # 128 for DSV4
    ) -> mx.array of shape (B, n_heads, L, v_head_dim)

When topk_idx is not None and L==1, GATHER fast path materializes only K
pool rows. Otherwise builds the (win + P) concatenated cache and calls
mx.fast.scaled_dot_product_attention(sinks=attn_sink, mask=...).
"""
from __future__ import annotations
from typing import Optional

import mlx.core as mx
from mlx.core.fast import scaled_dot_product_attention as sdpa


def fused_pool_attention(
    q: mx.array,
    k_window: mx.array,
    v_window: mx.array,
    k_pool: Optional[mx.array],
    v_pool: Optional[mx.array],
    topk_idx: Optional[mx.array],
    attn_sink: Optional[mx.array],
    win_mask: Optional[mx.array],
    comp_mask: Optional[mx.array],
    scale: float,
    sliding_window: int = 128,
) -> mx.array:
    """One-call HSA / CSA / SWA attention.

    L=1 + topk → GATHER path: materialize only K pool entries.
    Otherwise → concat path: build the full (win + P) K/V and let SDPA mask.
    """
    B = q.shape[0]
    L = q.shape[2]
    head_dim = q.shape[-1]

    if k_pool is None or k_pool.shape[2] == 0:
        # SWA-only path (compress_ratio=0 layer or pool empty)
        return sdpa(q, k_window, v_window, scale=scale, mask=win_mask,
                    sinks=attn_sink)

    P = k_pool.shape[2]

    if L == 1 and topk_idx is not None:
        # GATHER fast path — materialize only top-K pool rows for the
        # single query. Saves P/K× memory.
        K = topk_idx.shape[-1]
        # k_pool: (B, 1, P, head_dim); topk_idx: (B, 1, K)
        idx = mx.broadcast_to(topk_idx[:, None, :, :, None],
                              (B, 1, 1, K, head_dim))
        k_expanded = mx.broadcast_to(k_pool[:, None, :, :, :],
                                     (B, 1, 1, P, head_dim))
        v_expanded = mx.broadcast_to(v_pool[:, None, :, :, :],
                                     (B, 1, 1, P, head_dim))
        k_gathered = mx.take_along_axis(k_expanded, idx, axis=3).reshape(B, K, head_dim)
        v_gathered = mx.take_along_axis(v_expanded, idx, axis=3).reshape(B, K, head_dim)
        full_k = mx.concatenate([k_window, k_gathered[:, None]], axis=2)
        full_v = mx.concatenate([v_window, v_gathered[:, None]], axis=2)
        # mask=None for L=1: gather already encodes selection
        return sdpa(q, full_k, full_v, scale=scale, mask=None, sinks=attn_sink)

    # S>1 prefill path or S=1 without indexer (HSA dense pool)
    full_k = mx.concatenate([k_window, k_pool], axis=2)
    full_v = mx.concatenate([v_window, v_pool], axis=2)
    if comp_mask is not None and win_mask is not None:
        full_mask = mx.concatenate([win_mask, comp_mask], axis=-1)
    elif win_mask is not None:
        # Pool present but no comp_mask: pool_visible = block_causal staircase
        # (caller must provide comp_mask; this branch should be rare)
        raise ValueError("pool present but comp_mask missing; caller bug")
    else:
        full_mask = None
    return sdpa(q, full_k, full_v, scale=scale, mask=full_mask, sinks=attn_sink)


# =====================================================================
# Future Metal kernel skeleton — to replace the orchestrator above.
# This is the SPEC; not yet executable. Follows the _hc_premix_kernel
# template. Outline:
#
#  source = """
#    // INPUT BUFFERS:
#    //   q          fp16/bf16  (B, n_heads, L, head_dim)
#    //   k_window   fp16/bf16  (B,        1, W,  head_dim)
#    //   v_window   fp16/bf16  (B,        1, W,  head_dim)
#    //   k_pool     int4 packed in uint8 (when DSV4_POOL_QUANT=1) or
#    //              fp16/bf16  (B, 1, P, head_dim)
#    //   v_pool     same as k_pool
#    //   pool_scales/biases  fp16/bf16  (when k_pool is int4)
#    //   topk_idx   uint32  (B, L, K)  or null
#    //   sink       fp32    (n_heads,)
#    //   win_mask   bool    (B, 1, L, W)  causal-windowed
#    //   comp_mask  bool    (B, 1, L, P)  block-causal-or-selected
#    //   scale      float
#    //
#    // OUTPUT:
#    //   out  fp16/bf16  (B, n_heads, L, v_head_dim)
#    //
#    // PSEUDOCODE per (batch, head, query):
#    //   acc_kv = (sink) initialization
#    //   denom  = exp(sink)
#    //   for w in 0..W:
#    //     if win_mask[w]:
#    //       s = q · k_window[w] * scale
#    //       acc_kv += exp(s) * v_window[w]
#    //       denom  += exp(s)
#    //   for p in 0..P:        // GATHER for L=1 with topk
#    //     if comp_mask[p]:
#    //       (k,v) = (dequant pool[p]) if int4 else pool[p]
#    //       s = q · k * scale
#    //       acc_kv += exp(s) * v
#    //       denom  += exp(s)
#    //   out = acc_kv / denom
#    //
#    // Threadgroup design:
#    //   1 threadgroup per (batch, head) pair. 32 threads/group.
#    //   Each thread covers a slice of the head_dim contraction; tree
#    //   reduce within group for the dot product. Sequential loop over
#    //   W + P keys (small in practice — W=128, P=top_k=512).
#    //
#    // Performance budget vs unfused:
#    //   Unfused dispatches: 1 (compressor matmul) + 1 (indexer matmul)
#    //                     + 2 (gate/score reduction) + 1 (concat-mask)
#    //                     + 1 (SDPA) = 6 per layer
#    //   Fused dispatches:  1 per layer
#    //   Saving: 5 dispatches × 41 long-ctx layers = 205/token.
#    //   At ~25us/dispatch on M3, that's 5ms/token saved → 18% throughput.
#  """
# =====================================================================


def install_into_attention_class(attn_module):
    """Monkey-patch DeepseekV4Attention to call fused_pool_attention.

    Used by `make_cache` when DSV4_FUSED_POOL_ATTN=1. Saves a forward
    rewrite; the caller still uses the standard __call__ surface.
    """
    raise NotImplementedError(
        "install_into_attention_class: pending. The orchestrator fn is "
        "ready; once we measure the fused-vs-inline equivalence test "
        "in test_fused_pool_attn.py PASS, we wire the env flag here."
    )
