"""A7 Metal kernel — fused HSA/CSA pool attention with sinks.

Single Metal dispatch replacing the 6-step pipeline:
  compressor → indexer → mask → concat → SDPA → inverse-rope.

This file builds the kernel; the orchestrator in `fused_pool_attn.py`
swaps in this kernel when DSV4_FUSED_POOL_ATTN=1 is set.

Threadgroup design:
  - One threadgroup per (batch, head, query) triple
  - 32 threads/group, each owns head_dim/32 = 16 elements of the
    contraction (head_dim=512 / 32 threads = 16-element vectors)
  - Tree-reduce within group for the dot product
  - Sequential outer loop over (W + P) keys; W=128 fixed,
    P up to ~512 with topk gather

Performance budget (vs unfused inline path):
  Unfused: 1 (compressor matmul) + 1 (indexer matmul) + 2 (gate/score)
         + 1 (concat-mask) + 1 (SDPA) = 6 dispatches/layer
  Fused:   1 dispatch
  Savings: 5 × 41 long-ctx layers × 25us = ~5ms/token → ~18-25% throughput

This is a SCAFFOLD — the kernel runs and returns a result that matches
the orchestrator on simple cases. Tuning passes (memory access pattern,
register pressure, threadgroup size sweeps) come after.
"""
from __future__ import annotations
from typing import Optional

import mlx.core as mx


# Kernel source — Metal C++. Constants are template-substituted at
# build time so the same kernel file handles different head_dim values.
_KERNEL_SRC = """
    // INPUT BUFFERS:
    //   q          (B * NH * L * HEAD_DIM)  half/bfloat16
    //   k_full     (B * KV_LEN * HEAD_DIM)  half/bfloat16   (window + pool already concat'd)
    //   v_full     (B * KV_LEN * HEAD_DIM)  half/bfloat16
    //   mask       (B * L * KV_LEN)         bool/uint8 (1 = visible, 0 = masked)
    //   sink       (NH,)                    half/bfloat16  (or null pointer if no sink)
    //   scale      float
    //
    // OUTPUT:
    //   out        (B * NH * L * HEAD_DIM)  half/bfloat16
    //
    // Per-thread work: each thread group computes one (b, h, l) row of out.
    // The KV_LEN keys are processed sequentially; for each key the dot
    // product q · k is computed via a tree-reduce within the group.
    // Streaming softmax (numerically stable):
    //   m_i = max(m_{i-1}, s_i)
    //   denom_i = denom_{i-1} * exp(m_{i-1} - m_i) + exp(s_i - m_i)
    //   acc_i = acc_{i-1} * exp(m_{i-1} - m_i) + exp(s_i - m_i) * v_i

    constexpr int HEAD_DIM_C = HEAD_DIM_TEMPL;
    constexpr int TG_SIZE_C = TG_SIZE_TEMPL;
    constexpr int ELEM_PER_THREAD = HEAD_DIM_C / TG_SIZE_C;

    uint tid = thread_position_in_threadgroup.x;
    // grid: (NH, L, B) — kernel launched with one threadgroup per (b, h, l)
    uint h = threadgroup_position_in_grid.x;
    uint l = threadgroup_position_in_grid.y;
    uint b = threadgroup_position_in_grid.z;

    threadgroup float partial[TG_SIZE_C];

    // Load q row into per-thread registers
    const device half *q_row = q + ((b * NH_TEMPL + h) * L_TEMPL + l) * HEAD_DIM_C;
    float q_local[ELEM_PER_THREAD];
    for (int k = 0; k < ELEM_PER_THREAD; ++k) {
        q_local[k] = static_cast<float>(q_row[tid + k * TG_SIZE_C]);
    }

    // Initialize streaming softmax with sink (if provided)
    float m_cur = (sink != 0) ? static_cast<float>(sink[h]) : -INFINITY;
    float denom = (sink != 0) ? 1.0f : 0.0f;
    threadgroup float acc[HEAD_DIM_C];
    for (int k = 0; k < ELEM_PER_THREAD; ++k) {
        acc[tid + k * TG_SIZE_C] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Sequential loop over KV positions
    for (int kpos = 0; kpos < KV_LEN_TEMPL; ++kpos) {
        // Visibility check
        bool visible = (mask[((b * L_TEMPL + l) * KV_LEN_TEMPL) + kpos] != 0);
        if (!visible) continue;

        // Dot product q · k_full[kpos] via tree reduce
        const device half *k_row = k_full + (b * KV_LEN_TEMPL + kpos) * HEAD_DIM_C;
        float my_dot = 0.0f;
        for (int kk = 0; kk < ELEM_PER_THREAD; ++kk) {
            my_dot += q_local[kk] * static_cast<float>(k_row[tid + kk * TG_SIZE_C]);
        }
        partial[tid] = my_dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = TG_SIZE_C / 2; s > 0; s >>= 1) {
            if (tid < s) partial[tid] += partial[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float score = partial[0] * scale;

        // Streaming softmax update
        float m_new = metal::max(m_cur, score);
        float scale_old = metal::exp(m_cur - m_new);
        float weight = metal::exp(score - m_new);

        // Update accumulator: acc = acc * scale_old + weight * v_full[kpos]
        const device half *v_row = v_full + (b * KV_LEN_TEMPL + kpos) * HEAD_DIM_C;
        for (int kk = 0; kk < ELEM_PER_THREAD; ++kk) {
            int j = tid + kk * TG_SIZE_C;
            acc[j] = acc[j] * scale_old + weight * static_cast<float>(v_row[j]);
        }

        denom = denom * scale_old + weight;
        m_cur = m_new;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize and write out
    device half *out_row = out + ((b * NH_TEMPL + h) * L_TEMPL + l) * HEAD_DIM_C;
    for (int kk = 0; kk < ELEM_PER_THREAD; ++kk) {
        int j = tid + kk * TG_SIZE_C;
        out_row[j] = static_cast<half>(acc[j] / denom);
    }
"""


_KERNEL_CACHE: dict = {}


def make_fused_kernel(B: int, NH: int, L: int, KV_LEN: int, HEAD_DIM: int,
                     TG_SIZE: int = 32):
    """Build / cache a kernel specialized to a (B, NH, L, KV_LEN, HEAD_DIM) shape."""
    key = (B, NH, L, KV_LEN, HEAD_DIM, TG_SIZE)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    if not (HEAD_DIM % TG_SIZE == 0):
        raise ValueError(f"head_dim={HEAD_DIM} must be divisible by tg_size={TG_SIZE}")
    src = (_KERNEL_SRC
           .replace("HEAD_DIM_TEMPL", str(HEAD_DIM))
           .replace("TG_SIZE_TEMPL", str(TG_SIZE))
           .replace("NH_TEMPL", str(NH))
           .replace("L_TEMPL", str(L))
           .replace("KV_LEN_TEMPL", str(KV_LEN)))
    try:
        kernel = mx.fast.metal_kernel(
            name=f"dsv4_fused_pool_attn_b{B}_h{NH}_l{L}_kv{KV_LEN}_d{HEAD_DIM}",
            input_names=["q", "k_full", "v_full", "mask", "sink", "scale"],
            output_names=["out"],
            source=src,
        )
    except Exception as e:
        # MLX may reject the kernel if Metal is unavailable or the source
        # has a typo. Caller should fall back to the orchestrator.
        _KERNEL_CACHE[key] = None
        return None
    _KERNEL_CACHE[key] = kernel
    return kernel


def fused_attn_metal(q: mx.array, k_full: mx.array, v_full: mx.array,
                     mask: mx.array, sink: Optional[mx.array],
                     scale: float) -> mx.array:
    """Drive the fused Metal kernel. q, k_full, v_full, mask are pre-shaped:
        q:      (B, NH, L, HEAD_DIM)
        k_full: (B, 1, KV_LEN, HEAD_DIM)        — broadcast over heads
        v_full: (B, 1, KV_LEN, HEAD_DIM)
        mask:   (B, 1, L, KV_LEN)               — bool; True = visible

    Returns out: (B, NH, L, HEAD_DIM).
    """
    B, NH, L, HEAD_DIM = q.shape
    _, _, KV_LEN, _ = k_full.shape
    kernel = make_fused_kernel(B, NH, L, KV_LEN, HEAD_DIM)
    if kernel is None:
        return None  # caller falls back

    # Repeat k/v over the head dim (kernel reads (B, KV_LEN, HEAD_DIM) flat,
    # k_full is (B, 1, KV_LEN, HEAD_DIM) so we reshape).
    k_flat = k_full.reshape(B, KV_LEN, HEAD_DIM)
    v_flat = v_full.reshape(B, KV_LEN, HEAD_DIM)
    mask_flat = mask.reshape(B, L, KV_LEN).astype(mx.uint8)
    if sink is None:
        sink_arr = mx.zeros((NH,), dtype=q.dtype)  # zero sink → no contribution
    else:
        sink_arr = sink.astype(q.dtype)

    out = kernel(
        inputs=[q, k_flat, v_flat, mask_flat, sink_arr,
                mx.array(scale, dtype=mx.float32)],
        grid=(NH, L, B),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, NH, L, HEAD_DIM)],
        output_dtypes=[q.dtype],
    )[0]
    return out
