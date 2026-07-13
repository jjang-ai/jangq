//
// JANGTQ decode-time helper kernels — RMSNorm, RoPE, SDPA, residual add,
// fp32→fp16 cast, fp16→fp16 sigmoid+dot router math.
// Created by Jinho Jang (eric@jangq.ai).
//
// These kernels exist so the Swift `JANGTQModel.forward` doesn't fall back
// to CPU helpers for the per-layer hot path. Together with the codebook
// matmul kernels and the 8-bit affine GEMV kernel, they cover every Metal
// dispatch needed for a single-token decode step at ~Python parity.
//
// All kernels are decode-only (T=1). Prefill is not yet supported.
//

#include <metal_stdlib>
using namespace metal;

// ============================================================================
//  RMSNorm (fp16 in/out, fp32 accumulation)
// ============================================================================
//
// Single-vector RMSNorm: out[i] = (x[i] / sqrt(mean(x^2) + eps)) * gamma[i]
//
// Grid: (1, 1, 1)
// Threadgroup: (256, 1, 1)
// 256 threads cooperate via simd-group reduction. dim must be ≤ 4096.
//
struct RMSNormParams {
    uint dim;
    float eps;
};

kernel void jangtq_rmsnorm(
    device const half*  x      [[buffer(0)]],
    device const half*  gamma  [[buffer(1)]],
    device       half*  out    [[buffer(2)]],
    constant RMSNormParams& p  [[buffer(3)]],
    uint tid                    [[thread_position_in_threadgroup]],
    uint tg_size                [[threads_per_threadgroup]]
) {
    threadgroup float partial[8];  // up to 256 threads / 32 lanes = 8 simd groups

    // Phase 1: each thread sums squares of its assigned elements
    float sumSq = 0.0f;
    for (uint i = tid; i < p.dim; i += tg_size) {
        float v = float(x[i]);
        sumSq += v * v;
    }
    sumSq = simd_sum(sumSq);
    if ((tid % 32u) == 0) {
        partial[tid / 32u] = sumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: thread 0 sums across simd groups
    if (tid == 0) {
        float total = 0.0f;
        uint nSimds = (tg_size + 31u) / 32u;
        for (uint i = 0; i < nSimds; i++) total += partial[i];
        partial[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = partial[0];
    float rrms = 1.0f / sqrt(total / float(p.dim) + p.eps);

    // Phase 3: each thread writes its output values
    for (uint i = tid; i < p.dim; i += tg_size) {
        float v = float(x[i]);
        float g = float(gamma[i]);
        out[i] = half(v * rrms * g);
    }
}


// ============================================================================
//  RoPE (in-place rotation, fp16)
// ============================================================================
//
// Applies rotary position embedding to a (n_heads, head_dim) tensor in place.
// `rotary_dim` may be smaller than `head_dim` for partial RoPE models; the
// full head_dim remains the row stride and only the first rotary_dim values
// in each head are rotated.
// Convention matches mlx_lm `nn.RoPE(traditional=False)`: split each head into
// real (first half of rotary_dim) + imag (second half), rotate by pos*freq.
//
// Grid: (n_heads * rotary_dim/2, 1, 1)
// One thread per (head, dim_pair).
//
struct RoPEParams {
    uint  n_heads;
    uint  head_dim;
    uint  rotary_dim;
    uint  position;
    float base;     // 10000.0 for standard RoPE
};

kernel void jangtq_rope(
    device half* qk            [[buffer(0)]],   // (n_heads * head_dim,) fp16, in-place
    constant RoPEParams& p     [[buffer(1)]],
    uint tid                    [[thread_position_in_grid]]
) {
    uint half_dim = p.rotary_dim / 2u;
    uint total_pairs = p.n_heads * half_dim;
    if (tid >= total_pairs) return;

    uint h = tid / half_dim;
    uint i = tid % half_dim;

    float freq = pow(p.base, -2.0f * float(i) / float(p.rotary_dim));
    float angle = float(p.position) * freq;
    float c = cos(angle);
    float s = sin(angle);

    uint realIdx = h * p.head_dim + i;
    uint imagIdx = realIdx + half_dim;
    float r  = float(qk[realIdx]);
    float im = float(qk[imagIdx]);
    qk[realIdx] = half(r * c - im * s);
    qk[imagIdx] = half(r * s + im * c);
}


// ============================================================================
//  SDPA — single-token decode, GQA-aware
// ============================================================================
//
// Q : (n_heads, head_dim) fp16
// K : (max_seq, n_kv_heads, head_dim) fp16  (cache, only [0..cur_len) used)
// V : same as K
// out: (n_heads, head_dim) fp16
//
// One threadgroup per query head. Threads in the group cooperate to
// compute logits over cur_len keys, softmax, then weighted sum of V.
//
// Grid: (n_heads * 32, 1, 1)        # one simd-group per head
// Threadgroup: (32, 1, 1)
//
struct SDPAParams {
    uint n_heads;
    uint n_kv_heads;
    uint head_dim;
    uint cur_len;
    uint max_seq;
    float scale;     // 1 / sqrt(head_dim)
};

kernel void jangtq_sdpa_decode(
    device const half* q             [[buffer(0)]],
    device const half* k_cache       [[buffer(1)]],
    device const half* v_cache       [[buffer(2)]],
    device       half* out           [[buffer(3)]],
    constant SDPAParams& p           [[buffer(4)]],
    uint tid                          [[thread_position_in_threadgroup]],
    uint tg                           [[threadgroup_position_in_grid]]
) {
    uint h = tg;
    if (h >= p.n_heads) return;
    uint group_size = p.n_heads / p.n_kv_heads;
    uint kv_head = h / group_size;

    // Allocate scratch for logits in threadgroup memory
    threadgroup float logits[2048];   // supports cur_len up to 2048
    threadgroup float partials[8];

    // Phase 1: each thread computes a stride of logits
    // logits[t] = (q[h] · k[t, kv_head]) * scale
    for (uint t = tid; t < p.cur_len; t += 32u) {
        float dot = 0.0f;
        uint kBase = t * p.n_kv_heads * p.head_dim + kv_head * p.head_dim;
        uint qBase = h * p.head_dim;
        for (uint d = 0; d < p.head_dim; d++) {
            dot += float(q[qBase + d]) * float(k_cache[kBase + d]);
        }
        logits[t] = dot * p.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: max over logits
    float localMax = -INFINITY;
    for (uint t = tid; t < p.cur_len; t += 32u) {
        if (logits[t] > localMax) localMax = logits[t];
    }
    localMax = simd_max(localMax);
    // tid 0 has the lane-0 result of simd_max; broadcast via threadgroup mem
    if (tid == 0) partials[0] = localMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maxLog = partials[0];

    // Phase 3: exp + sum
    for (uint t = tid; t < p.cur_len; t += 32u) {
        logits[t] = exp(logits[t] - maxLog);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float localSum = 0.0f;
    for (uint t = tid; t < p.cur_len; t += 32u) {
        localSum += logits[t];
    }
    localSum = simd_sum(localSum);
    if (tid == 0) partials[1] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sumExp = partials[1];

    // Phase 4: normalize + V dot products
    // For each output dim d in head, compute sum_t (logits[t]/sumExp) * v[t, kv_head, d]
    for (uint d = tid; d < p.head_dim; d += 32u) {
        float acc = 0.0f;
        for (uint t = 0; t < p.cur_len; t++) {
            uint vBase = t * p.n_kv_heads * p.head_dim + kv_head * p.head_dim;
            acc += (logits[t] / sumExp) * float(v_cache[vBase + d]);
        }
        out[h * p.head_dim + d] = half(acc);
    }
}


// ============================================================================
//  Per-head RMSNorm — for q_norm / k_norm in attention
// ============================================================================
//
// Each head_dim chunk normalized independently. One threadgroup per head.
// Threads in TG cooperate via simd_sum to compute the per-head sum-of-squares.
//
// Grid: (n_heads * 32, 1, 1)
// Threadgroup: (32, 1, 1)
//
struct HeadRMSNormParams {
    uint n_heads;
    uint head_dim;
    float eps;
};

kernel void jangtq_head_rmsnorm(
    device       half* qk        [[buffer(0)]],   // (n_heads * head_dim,) — in/out
    device const half* gamma     [[buffer(1)]],   // (n_heads * head_dim,)
    constant HeadRMSNormParams& p [[buffer(2)]],
    uint global_id                [[thread_position_in_grid]]
) {
    uint h = global_id / 32u;
    uint lane = global_id % 32u;
    if (h >= p.n_heads) return;

    uint base = h * p.head_dim;

    // Sum of squares across head_dim, distributed across 32 lanes
    float sumSq = 0.0f;
    for (uint i = lane; i < p.head_dim; i += 32u) {
        float v = float(qk[base + i]);
        sumSq += v * v;
    }
    sumSq = simd_sum(sumSq);
    float rrms = 1.0f / sqrt(sumSq / float(p.head_dim) + p.eps);

    // Apply normalization + gamma
    for (uint i = lane; i < p.head_dim; i += 32u) {
        float v = float(qk[base + i]);
        float g = float(gamma[base + i]);
        qk[base + i] = half(v * rrms * g);
    }
}


// ============================================================================
//  Residual add (fp16 in-place: out = a + b)
// ============================================================================
//
struct ResidualParams { uint dim; };

kernel void jangtq_residual_add(
    device const half* a   [[buffer(0)]],
    device const half* b   [[buffer(1)]],
    device       half* out [[buffer(2)]],
    constant ResidualParams& p [[buffer(3)]],
    uint tid                [[thread_position_in_grid]]
) {
    if (tid >= p.dim) return;
    out[tid] = half(float(a[tid]) + float(b[tid]));
}


// ============================================================================
//  fp32 → fp16 cast
// ============================================================================
//
struct CastParams { uint count; };

kernel void jangtq_cast_f32_to_f16(
    device const float* src [[buffer(0)]],
    device       half*  dst [[buffer(1)]],
    constant CastParams& p  [[buffer(2)]],
    uint tid                [[thread_position_in_grid]]
) {
    if (tid >= p.count) return;
    dst[tid] = half(src[tid]);
}


// ============================================================================
//  Qwen gated attention helpers
// ============================================================================
//
// Qwen3-Next/Qwen3.6 gated full attention packs q_proj per head as
// [query_head, gate_head]. Split that layout into separate contiguous query and
// gate buffers before q_norm/RoPE/SDPA.
//
struct QwenGatedQParams {
    uint n_heads;
    uint head_dim;
};

kernel void jangtq_qwen_gated_q_split(
    device const half* q_full [[buffer(0)]],
    device       half* q      [[buffer(1)]],
    device       half* gate   [[buffer(2)]],
    constant QwenGatedQParams& p [[buffer(3)]],
    uint tid                  [[thread_position_in_grid]]
) {
    uint total = p.n_heads * p.head_dim;
    if (tid >= total) return;

    uint head = tid / p.head_dim;
    uint dim = tid - head * p.head_dim;
    uint src_base = head * p.head_dim * 2u;
    q[tid] = q_full[src_base + dim];
    gate[tid] = q_full[src_base + p.head_dim + dim];
}

struct AttentionGateParams {
    uint count;
};

kernel void jangtq_apply_attention_gate(
    device       half* attn_out [[buffer(0)]],
    device const half* gate     [[buffer(1)]],
    constant AttentionGateParams& p [[buffer(2)]],
    uint tid                    [[thread_position_in_grid]]
) {
    if (tid >= p.count) return;
    float g = float(gate[tid]);
    float s = 1.0f / (1.0f + exp(-g));
    attn_out[tid] = half(float(attn_out[tid]) * s);
}
