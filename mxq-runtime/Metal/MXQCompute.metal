/*
 * MXQ Non-Quantized Compute Kernels
 * Created by Eric Jang (eric@vmlx.net)
 *
 * Supporting kernels for transformer inference:
 * RMSNorm, RoPE, Softmax, SiLU, element-wise ops.
 * These are NOT the bottleneck (quantized matmul is),
 * but they need to be correct and reasonably fast.
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================
// RMSNorm: y = x * rsqrt(mean(x²) + eps) * gamma
// ============================================================

kernel void mxq_rms_norm(
    device const half*   input   [[buffer(0)]],  // (seq_len, hidden)
    device const half*   gamma   [[buffer(1)]],  // (hidden,) — norm weights
    device half*         output  [[buffer(2)]],  // (seq_len, hidden)
    constant uint&       hidden  [[buffer(3)]],  // hidden dimension
    constant float&      eps     [[buffer(4)]],  // epsilon (1e-5 typically)
    uint2                gid     [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (col >= hidden) return;

    // Compute mean of squares for this row
    // Each thread computes the full sum (simple but correct)
    // TODO: optimize with SIMD reduction for large hidden dims
    float sum_sq = 0.0f;
    uint row_offset = row * hidden;

    for (uint i = 0; i < hidden; i++) {
        float val = float(input[row_offset + i]);
        sum_sq += val * val;
    }

    float rms = rsqrt(sum_sq / float(hidden) + eps);
    float x = float(input[row_offset + col]);
    float g = float(gamma[col]);

    output[row_offset + col] = half(x * rms * g);
}


// ============================================================
// RoPE: Rotary Position Embeddings
// ============================================================
//
// Applies rotation to Q and K vectors based on position.
// For each pair of dimensions (2i, 2i+1):
//   q_rot[2i]   = q[2i] * cos(θ) - q[2i+1] * sin(θ)
//   q_rot[2i+1] = q[2i] * sin(θ) + q[2i+1] * cos(θ)
// where θ = position / (10000 ^ (2i / dim))
//

// RoPE supports two dimension pairing modes:
//   Traditional:     pairs (2i, 2i+1) — used by LLaMA, Mistral
//   Non-traditional: pairs (i, i+half_dim) — used by Qwen, GPT-NeoX
//
// The rotation math is identical; only the index mapping differs.
// Most modern models use non-traditional (the default here).
//
// The `traditional` flag selects the mode:
//   traditional=0 → non-traditional: pairs (i, i+half_dim)
//   traditional=1 → traditional: pairs (2i, 2i+1)

kernel void mxq_rope(
    device half*         qk       [[buffer(0)]],  // Q or K tensor (seq, n_heads, head_dim)
    constant uint&       seq_len  [[buffer(1)]],
    constant uint&       n_heads  [[buffer(2)]],
    constant uint&       head_dim [[buffer(3)]],
    constant uint&       pos_offset [[buffer(4)]],  // for KV cache position
    constant float&      theta_base [[buffer(5)]],  // base frequency (10000.0)
    constant uint&       traditional [[buffer(6)]],  // 0=non-traditional, 1=traditional
    uint3                gid      [[thread_position_in_grid]]
) {
    uint pos = gid.z;         // sequence position
    uint head = gid.y;        // head index
    uint pair = gid.x;        // dimension pair index (0 to head_dim/2 - 1)

    if (pos >= seq_len || head >= n_heads || pair >= head_dim / 2) return;

    uint half_dim = head_dim / 2;

    // Compute rotation angle
    float freq = 1.0f / pow(theta_base, float(pair) / float(half_dim));
    float angle = float(pos + pos_offset) * freq;
    float cos_val = cos(angle);
    float sin_val = sin(angle);

    // Index into the QK tensor — depends on RoPE mode
    uint base_idx = pos * n_heads * head_dim + head * head_dim;
    uint idx0, idx1;

    if (traditional != 0) {
        // Traditional: pairs (0,1), (2,3), (4,5), ...
        idx0 = base_idx + pair * 2;
        idx1 = base_idx + pair * 2 + 1;
    } else {
        // Non-traditional (default): pairs (0,half), (1,half+1), ...
        idx0 = base_idx + pair;
        idx1 = base_idx + pair + half_dim;
    }

    float v0 = float(qk[idx0]);
    float v1 = float(qk[idx1]);

    qk[idx0] = half(v0 * cos_val - v1 * sin_val);
    qk[idx1] = half(v0 * sin_val + v1 * cos_val);
}


// ============================================================
// Softmax: exp(x - max(x)) / sum(exp(x - max(x)))
// ============================================================

kernel void mxq_softmax(
    device half*         data      [[buffer(0)]],  // in-place (rows, cols)
    constant uint&       n_cols    [[buffer(1)]],
    uint                 row       [[thread_position_in_grid]]
) {
    uint offset = row * n_cols;

    // Find max for numerical stability
    float max_val = -INFINITY;
    for (uint i = 0; i < n_cols; i++) {
        max_val = max(max_val, float(data[offset + i]));
    }

    // Compute exp(x - max) and sum
    float sum_exp = 0.0f;
    for (uint i = 0; i < n_cols; i++) {
        float e = exp(float(data[offset + i]) - max_val);
        data[offset + i] = half(e);  // store intermediate
        sum_exp += e;
    }

    // Normalize
    float inv_sum = 1.0f / sum_exp;
    for (uint i = 0; i < n_cols; i++) {
        data[offset + i] = half(float(data[offset + i]) * inv_sum);
    }
}


// ============================================================
// SiLU: x * sigmoid(x) = x / (1 + exp(-x))
// ============================================================

kernel void mxq_silu(
    device const half*   input   [[buffer(0)]],
    device half*         output  [[buffer(1)]],
    uint                 gid     [[thread_position_in_grid]]
) {
    float x = float(input[gid]);
    output[gid] = half(x / (1.0f + exp(-x)));
}


// ============================================================
// SiLU + Element-wise Multiply (fused for SwiGLU)
// gate_output = SiLU(gate) * up
// ============================================================

kernel void mxq_silu_mul(
    device const half*   gate    [[buffer(0)]],  // gate projection output
    device const half*   up      [[buffer(1)]],  // up projection output
    device half*         output  [[buffer(2)]],
    uint                 gid     [[thread_position_in_grid]]
) {
    float g = float(gate[gid]);
    float u = float(up[gid]);
    float silu_g = g / (1.0f + exp(-g));
    output[gid] = half(silu_g * u);
}


// ============================================================
// Element-wise Add (residual connections)
// ============================================================

kernel void mxq_add(
    device const half*   a       [[buffer(0)]],
    device const half*   b       [[buffer(1)]],
    device half*         output  [[buffer(2)]],
    uint                 gid     [[thread_position_in_grid]]
) {
    output[gid] = a[gid] + b[gid];
}


// ============================================================
// Embedding Lookup
// ============================================================

kernel void mxq_embedding(
    device const half*     embed_weights [[buffer(0)]],  // (vocab_size, hidden)
    device const uint32_t* token_ids     [[buffer(1)]],  // (seq_len,)
    device half*           output        [[buffer(2)]],  // (seq_len, hidden)
    constant uint&         hidden_dim    [[buffer(3)]],
    uint2                  gid           [[thread_position_in_grid]]
) {
    uint seq_pos = gid.y;
    uint dim = gid.x;

    if (dim >= hidden_dim) return;

    uint token_id = token_ids[seq_pos];
    output[seq_pos * hidden_dim + dim] = embed_weights[token_id * hidden_dim + dim];
}
