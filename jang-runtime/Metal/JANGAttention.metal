/*
 * JANG Attention Kernels
 * Created by Eric Jang (eric@vmlx.net)
 *
 * Implements multi-head attention with Grouped Query Attention (GQA) support.
 * GQA: multiple query heads share the same KV head (e.g., 14 Q heads, 2 KV heads).
 *
 * For single-token decode (autoregressive):
 *   scores[h][t] = Q[h] · K_cache[kv_head][t] / sqrt(head_dim)
 *   attn[h][t] = softmax(scores[h])
 *   out[h] = sum_t(attn[h][t] * V_cache[kv_head][t])
 *
 * For prefill (multiple tokens):
 *   Uses tiled attention with causal mask.
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================
// Kernel: Single-token GQA Attention (Decode)
// ============================================================
//
// Input:
//   Q: (n_heads, head_dim) — query for current token
//   K_cache: (seq_len, n_kv_heads, head_dim) — cached keys
//   V_cache: (seq_len, n_kv_heads, head_dim) — cached values
//
// Output:
//   out: (n_heads, head_dim) — attention output
//
// Each threadgroup handles one query head.
// GQA: head h uses KV head (h * n_kv_heads / n_heads).
//

kernel void jang_attention_decode(
    device const half*   Q         [[buffer(0)]],  // (n_heads, head_dim)
    device const half*   K_cache   [[buffer(1)]],  // (max_seq, n_kv_heads, head_dim)
    device const half*   V_cache   [[buffer(2)]],  // (max_seq, n_kv_heads, head_dim)
    device half*         output    [[buffer(3)]],  // (n_heads, head_dim)
    constant uint&       n_heads   [[buffer(4)]],
    constant uint&       n_kv_heads [[buffer(5)]],
    constant uint&       head_dim  [[buffer(6)]],
    constant uint&       seq_len   [[buffer(7)]],  // current sequence length (filled positions)
    constant float&      scale     [[buffer(8)]],  // 1/sqrt(head_dim)
    uint                 head_idx  [[threadgroup_position_in_grid]],
    uint                 tid       [[thread_index_in_threadgroup]],
    uint                 simd_lane [[thread_index_in_simdgroup]],
    uint                 simd_id   [[simdgroup_index_in_threadgroup]]
) {
    if (head_idx >= n_heads) return;

    // GQA: map query head to KV head
    uint kv_head = head_idx * n_kv_heads / n_heads;

    // Pointer to this head's query vector
    device const half* q_ptr = Q + head_idx * head_dim;

    // Step 1: Compute attention scores Q·K^T for all cached positions
    // Each thread handles a subset of positions

    // Shared memory for scores and softmax
    threadgroup float scores[4096];  // max seq_len for scores
    threadgroup float shared_max[8]; // for SIMD group max reduction
    threadgroup float shared_sum[8]; // for SIMD group sum reduction

    uint threads_per_tg = 256;  // threadgroup size

    // Compute Q·K scores
    float thread_max = -INFINITY;

    for (uint pos = tid; pos < seq_len; pos += threads_per_tg) {
        // K_cache layout: (max_seq, n_kv_heads, head_dim)
        device const half* k_ptr = K_cache + pos * n_kv_heads * head_dim + kv_head * head_dim;

        // Dot product Q · K[pos]
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += float(q_ptr[d]) * float(k_ptr[d]);
        }
        dot *= scale;

        scores[pos] = dot;
        thread_max = max(thread_max, dot);
    }

    // SIMD reduction to find max score
    float group_max = simd_max(thread_max);
    if (simd_lane == 0) {
        shared_max[simd_id] = group_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final max across SIMD groups
    float global_max = -INFINITY;
    if (tid < 8) {
        global_max = shared_max[tid];
    }
    global_max = simd_max(global_max);
    // Broadcast to all threads
    if (tid == 0) {
        shared_max[0] = global_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = shared_max[0];

    // Step 2: Softmax — exp(score - max) and sum
    float thread_sum = 0.0f;

    for (uint pos = tid; pos < seq_len; pos += threads_per_tg) {
        float e = exp(scores[pos] - global_max);
        scores[pos] = e;
        thread_sum += e;
    }

    // SIMD reduction for sum
    float group_sum = simd_sum(thread_sum);
    if (simd_lane == 0) {
        shared_sum[simd_id] = group_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_sum = 0.0f;
    if (tid < 8) {
        global_sum = shared_sum[tid];
    }
    global_sum = simd_sum(global_sum);
    if (tid == 0) {
        shared_sum[0] = global_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_sum = shared_sum[0];

    // Normalize
    float inv_sum = 1.0f / global_sum;
    for (uint pos = tid; pos < seq_len; pos += threads_per_tg) {
        scores[pos] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Weighted sum of values: out = sum(attn[t] * V[t])
    // Each thread accumulates over a subset of dimensions
    for (uint d = tid; d < head_dim; d += threads_per_tg) {
        float acc = 0.0f;
        for (uint pos = 0; pos < seq_len; pos++) {
            // V_cache layout: (max_seq, n_kv_heads, head_dim)
            float v = float(V_cache[pos * n_kv_heads * head_dim + kv_head * head_dim + d]);
            acc += scores[pos] * v;
        }
        output[head_idx * head_dim + d] = half(acc);
    }
}


// ============================================================
// Kernel: Multi-token Attention (Prefill) with Causal Mask
// ============================================================
//
// For processing multiple tokens at once (prompt prefill).
// Uses causal masking: position i can only attend to positions <= i.
//
// Input:
//   Q: (seq_len, n_heads, head_dim)
//   K: (seq_len, n_kv_heads, head_dim)
//   V: (seq_len, n_kv_heads, head_dim)
//
// Output:
//   out: (seq_len, n_heads, head_dim)
//

kernel void jang_attention_prefill(
    device const half*   Q         [[buffer(0)]],
    device const half*   K         [[buffer(1)]],
    device const half*   V         [[buffer(2)]],
    device half*         output    [[buffer(3)]],
    constant uint&       n_heads   [[buffer(4)]],
    constant uint&       n_kv_heads [[buffer(5)]],
    constant uint&       head_dim  [[buffer(6)]],
    constant uint&       seq_len   [[buffer(7)]],
    constant float&      scale     [[buffer(8)]],
    uint2                tgid      [[threadgroup_position_in_grid]],
    uint                 tid       [[thread_index_in_threadgroup]]
) {
    uint query_pos = tgid.y;   // which query position
    uint head_idx = tgid.x;    // which head

    if (query_pos >= seq_len || head_idx >= n_heads) return;

    uint kv_head = head_idx * n_kv_heads / n_heads;
    uint threads_per_tg = 256;

    device const half* q_ptr = Q + query_pos * n_heads * head_dim + head_idx * head_dim;

    // Shared memory for attention scores
    threadgroup float scores[4096];
    threadgroup float shared_vals[8];

    // Compute scores for all positions <= query_pos (causal mask)
    uint valid_len = query_pos + 1;  // causal: can attend to 0..query_pos

    float thread_max = -INFINITY;
    for (uint pos = tid; pos < valid_len; pos += threads_per_tg) {
        device const half* k_ptr = K + pos * n_kv_heads * head_dim + kv_head * head_dim;

        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += float(q_ptr[d]) * float(k_ptr[d]);
        }
        dot *= scale;
        scores[pos] = dot;
        thread_max = max(thread_max, dot);
    }

    // Set masked positions to -inf
    for (uint pos = valid_len + tid; pos < seq_len; pos += threads_per_tg) {
        scores[pos] = -INFINITY;
    }

    // Max reduction
    float group_max = simd_max(thread_max);
    if (tid % 32 == 0) shared_vals[tid / 32] = group_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_max = -INFINITY;
    if (tid < 8) global_max = shared_vals[tid];
    global_max = simd_max(global_max);
    if (tid == 0) shared_vals[0] = global_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = shared_vals[0];

    // Softmax
    float thread_sum = 0.0f;
    for (uint pos = tid; pos < valid_len; pos += threads_per_tg) {
        float e = exp(scores[pos] - global_max);
        scores[pos] = e;
        thread_sum += e;
    }

    float group_sum = simd_sum(thread_sum);
    if (tid % 32 == 0) shared_vals[tid / 32] = group_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_sum = 0.0f;
    if (tid < 8) global_sum = shared_vals[tid];
    global_sum = simd_sum(global_sum);
    if (tid == 0) shared_vals[0] = global_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_sum = shared_vals[0];

    float inv_sum = 1.0f / max(global_sum, 1e-10f);
    for (uint pos = tid; pos < valid_len; pos += threads_per_tg) {
        scores[pos] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Weighted sum of V
    for (uint d = tid; d < head_dim; d += threads_per_tg) {
        float acc = 0.0f;
        for (uint pos = 0; pos < valid_len; pos++) {
            float v = float(V[pos * n_kv_heads * head_dim + kv_head * head_dim + d]);
            acc += scores[pos] * v;
        }
        output[query_pos * n_heads * head_dim + head_idx * head_dim + d] = half(acc);
    }
}
