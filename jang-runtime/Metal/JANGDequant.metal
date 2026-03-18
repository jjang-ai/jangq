/*
 * JANG Dequantization Kernels for Apple Silicon
 * Created by Eric Jang (eric@vmlx.net)
 *
 * Core Metal compute shaders for dequantizing variable-bit-width
 * quantized weights during inference. These kernels are the
 * performance-critical path — every token generation runs through them.
 *
 * Supported bit widths: 2, 3, 4, 5, 6, 8
 * Packing: LSB-first, contiguous across byte boundaries
 * Dequantization: val = (raw_int - zero) * scale
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================
// Constants
// ============================================================

constant uint BLOCK_SIZE = 64;  // weights per quantization block

// ============================================================
// Bit Extraction Helpers
// ============================================================

// Extract a value of `bits` width from packed data at a given bit offset.
// Handles cross-byte boundaries by reading 32 bits and masking.
inline uint extract_bits(
    device const uint8_t* data,
    uint bit_offset,
    uint bits
) {
    uint byte_idx = bit_offset >> 3;       // bit_offset / 8
    uint bit_shift = bit_offset & 7;        // bit_offset % 8
    uint mask = (1u << bits) - 1u;

    // Read 4 bytes to safely handle any cross-boundary extraction
    // (max value span: 8 bits across 2 bytes, so 32-bit read is always safe)
    uint32_t raw = uint32_t(data[byte_idx]);
    if (byte_idx + 1 < 0xFFFFFFFF)  // always true, but prevents optimizer issues
        raw |= uint32_t(data[byte_idx + 1]) << 8;
    if (bits > 8 - bit_shift)
        raw |= uint32_t(data[byte_idx + 2]) << 16;

    return (raw >> bit_shift) & mask;
}

// Fast path extractors for common bit widths
inline uint extract_2bit(device const uint8_t* data, uint idx) {
    uint byte_idx = idx >> 2;       // idx / 4
    uint bit_shift = (idx & 3) << 1; // (idx % 4) * 2
    return (uint(data[byte_idx]) >> bit_shift) & 0x3;
}

inline uint extract_4bit(device const uint8_t* data, uint idx) {
    uint byte_idx = idx >> 1;       // idx / 2
    uint bit_shift = (idx & 1) << 2; // (idx % 2) * 4
    return (uint(data[byte_idx]) >> bit_shift) & 0xF;
}

inline uint extract_8bit(device const uint8_t* data, uint idx) {
    return uint(data[idx]);
}

// ============================================================
// Kernel 1: Standalone Dequantization
// ============================================================
//
// Dequantizes a weight tensor from JANG format to float16 buffer.
// Used for debugging, inspection, and non-fused code paths.
//
// Each thread dequantizes one weight value.
//

kernel void jang_dequantize(
    device const uint8_t*   qweight       [[buffer(0)]],  // packed quantized data
    device const half*      scales        [[buffer(1)]],  // per-block scale (n_blocks,)
    device const half*      zeros         [[buffer(2)]],  // per-block zero point (n_blocks,)
    device const uint8_t*   bit_map       [[buffer(3)]],  // per-block bit width (n_blocks,)
    device const uint32_t*  block_offsets [[buffer(4)]],  // byte offset per block (n_blocks,)
    device half*            output        [[buffer(5)]],  // output float16 buffer
    constant uint&          n_weights     [[buffer(6)]],  // total weight count
    uint                    gid           [[thread_position_in_grid]]
) {
    if (gid >= n_weights) return;

    // Determine which block this weight belongs to
    uint block_idx = gid / BLOCK_SIZE;
    uint in_block = gid % BLOCK_SIZE;

    // Read block metadata
    uint bits = uint(bit_map[block_idx]);
    half scale = scales[block_idx];
    half zero = zeros[block_idx];
    uint byte_offset = block_offsets[block_idx];

    // Extract quantized value
    uint raw;
    device const uint8_t* block_data = qweight + byte_offset;

    // Fast paths for common bit widths
    switch (bits) {
        case 2:
            raw = extract_2bit(block_data, in_block);
            break;
        case 4:
            raw = extract_4bit(block_data, in_block);
            break;
        case 8:
            raw = extract_8bit(block_data, in_block);
            break;
        default:
            // General case: 3, 5, 6 bit
            raw = extract_bits(block_data, in_block * bits, bits);
            break;
    }

    // Dequantize: output = (raw - zero) * scale
    output[gid] = (half(raw) - zero) * scale;
}


// ============================================================
// Kernel 2: Dequantize + GEMV (Matrix-Vector Multiply)
// ============================================================
//
// For autoregressive token generation (batch_size = 1):
//   output[j] = sum_i( dequant(W[j][i]) * x[i] )
//
// This is the MOST IMPORTANT kernel for token generation speed.
// At batch=1, inference is memory-bandwidth bound — loading the
// quantized weights IS the bottleneck. This kernel loads quantized
// data (fewer bytes than float16), dequantizes in registers, and
// accumulates the dot product.
//
// Each threadgroup computes one or more output elements.
// Threads within a SIMD group cooperate on the reduction.
//

constant uint GEMV_THREADS_PER_ROW = 256;  // threads per output row

kernel void jang_dequant_gemv(
    device const uint8_t*   qweight       [[buffer(0)]],
    device const half*      scales        [[buffer(1)]],
    device const half*      zeros         [[buffer(2)]],
    device const uint8_t*   bit_map       [[buffer(3)]],
    device const uint32_t*  block_offsets [[buffer(4)]],
    device const half*      x             [[buffer(5)]],  // input vector (K,)
    device half*            output        [[buffer(6)]],  // output vector (N,)
    constant uint&          K             [[buffer(7)]],  // input features
    constant uint&          N             [[buffer(8)]],  // output features
    uint2                   tgid          [[threadgroup_position_in_grid]],
    uint                    tid           [[thread_index_in_threadgroup]],
    uint                    simd_lane     [[thread_index_in_simdgroup]],
    uint                    simd_id       [[simdgroup_index_in_threadgroup]]
) {
    // Each threadgroup handles one output row
    uint row = tgid.x;
    if (row >= N) return;

    // Number of blocks per row
    uint blocks_per_row = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Each thread handles a stride of the input dimension
    float acc = 0.0f;  // accumulate in float32

    for (uint col = tid; col < K; col += GEMV_THREADS_PER_ROW) {
        // Which block does this weight belong to?
        uint block_in_row = col / BLOCK_SIZE;
        uint in_block = col % BLOCK_SIZE;
        uint global_block = row * blocks_per_row + block_in_row;

        // Read block metadata
        uint bits = uint(bit_map[global_block]);
        float scale = float(scales[global_block]);
        float zero = float(zeros[global_block]);
        uint byte_offset = block_offsets[global_block];

        // Extract and dequantize
        device const uint8_t* block_data = qweight + byte_offset;
        uint raw;

        switch (bits) {
            case 2: raw = extract_2bit(block_data, in_block); break;
            case 4: raw = extract_4bit(block_data, in_block); break;
            case 8: raw = extract_8bit(block_data, in_block); break;
            default: raw = extract_bits(block_data, in_block * bits, bits); break;
        }

        float w = (float(raw) - zero) * scale;
        acc += w * float(x[col]);
    }

    // SIMD reduction within each SIMD group (32 threads)
    acc = simd_sum(acc);

    // Store SIMD group partial sums in threadgroup memory
    threadgroup float partial_sums[8];  // max 8 SIMD groups per threadgroup

    if (simd_lane == 0) {
        partial_sums[simd_id] = acc;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction by first SIMD group
    if (tid == 0) {
        float total = 0.0f;
        uint n_simd_groups = (GEMV_THREADS_PER_ROW + 31) / 32;
        for (uint i = 0; i < n_simd_groups; i++) {
            total += partial_sums[i];
        }
        output[row] = half(total);
    }
}


// ============================================================
// Kernel 3: Dequantize + GEMM (Matrix-Matrix Multiply)
// ============================================================
//
// For prompt processing / prefill (batch_size > 1):
//   Output[m][n] = sum_k( dequant(W[n][k]) * X[m][k] )
//
// Tiled approach:
//   - Load a tile of X into threadgroup memory
//   - Dequantize a tile of W into threadgroup memory
//   - Compute tile product
//   - Accumulate across K dimension
//

constant uint TILE_M = 32;   // rows of output tile
constant uint TILE_N = 32;   // cols of output tile
constant uint TILE_K = 64;   // reduction tile (= BLOCK_SIZE)

kernel void jang_dequant_gemm(
    device const uint8_t*   qweight       [[buffer(0)]],
    device const half*      scales        [[buffer(1)]],
    device const half*      zeros         [[buffer(2)]],
    device const uint8_t*   bit_map       [[buffer(3)]],
    device const uint32_t*  block_offsets [[buffer(4)]],
    device const half*      X             [[buffer(5)]],  // input (M, K)
    device half*            output        [[buffer(6)]],  // output (M, N)
    constant uint&          M             [[buffer(7)]],  // batch * seq_len
    constant uint&          K             [[buffer(8)]],  // input features
    constant uint&          N             [[buffer(9)]],  // output features
    uint2                   tgid          [[threadgroup_position_in_grid]],
    uint2                   tid           [[thread_position_in_threadgroup]],
    uint                    simd_lane     [[thread_index_in_simdgroup]]
) {
    // Tile position
    uint tile_row = tgid.y * TILE_M;
    uint tile_col = tgid.x * TILE_N;

    // Shared memory for tiles
    threadgroup half x_tile[TILE_M][TILE_K];
    threadgroup half w_tile[TILE_K][TILE_N];

    // Each thread accumulates one element of the output tile
    uint local_m = tid.y;
    uint local_n = tid.x;
    float acc = 0.0f;

    uint blocks_per_row = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Iterate over K dimension in TILE_K chunks
    for (uint k_start = 0; k_start < K; k_start += TILE_K) {

        // Cooperative load of X tile
        // Each thread loads one element
        uint global_m = tile_row + local_m;
        uint global_k = k_start + local_n;
        if (global_m < M && global_k < K) {
            x_tile[local_m][local_n] = X[global_m * K + global_k];
        } else {
            x_tile[local_m][local_n] = 0;
        }

        // Also load second half if TILE_K > TILE_N
        if (TILE_K > TILE_N) {
            uint global_k2 = k_start + local_n + TILE_N;
            if (global_m < M && global_k2 < K) {
                x_tile[local_m][local_n + TILE_N] = X[global_m * K + global_k2];
            } else {
                x_tile[local_m][local_n + TILE_N] = 0;
            }
        }

        // Cooperative dequant of W tile
        // Each thread dequantizes a column of the weight tile
        uint w_col = tile_col + local_n;
        if (w_col < N) {
            uint block_idx = w_col * blocks_per_row + (k_start / BLOCK_SIZE);
            uint bits = uint(bit_map[block_idx]);
            float scale = float(scales[block_idx]);
            float zero = float(zeros[block_idx]);
            uint byte_off = block_offsets[block_idx];
            device const uint8_t* bdata = qweight + byte_off;

            for (uint w = local_m; w < TILE_K; w += TILE_M) {
                uint raw;
                switch (bits) {
                    case 2: raw = extract_2bit(bdata, w); break;
                    case 4: raw = extract_4bit(bdata, w); break;
                    case 8: raw = extract_8bit(bdata, w); break;
                    default: raw = extract_bits(bdata, w * bits, bits); break;
                }
                w_tile[w][local_n] = half((float(raw) - zero) * scale);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Tile multiply-accumulate
        for (uint k = 0; k < TILE_K; k++) {
            acc += float(x_tile[local_m][k]) * float(w_tile[k][local_n]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output
    uint out_m = tile_row + local_m;
    uint out_n = tile_col + local_n;
    if (out_m < M && out_n < N) {
        output[out_m * N + out_n] = half(acc);
    }
}
