/*
 * JANG Embedding Kernels
 * Created by Eric Jang (eric@vmlx.net)
 *
 * Specialized kernels for quantized embedding lookup.
 * Unlike GEMV (matrix × vector), embedding lookup extracts a single row
 * from the quantized weight table — much less data to read.
 *
 * For a vocab_size=151936, hidden=896, block_size=64 model at 4-bit:
 *   Each row = 14 blocks × 32 bytes = 448 bytes
 *   A single lookup reads 448 bytes vs 68MB for the full matrix
 *
 * Also handles tied embedding lm_head: the logits computation when
 * the output head shares weights with the embedding table.
 */

#include <metal_stdlib>
using namespace metal;

constant uint BLOCK_SIZE = 64;

// Bit extraction helpers (same as JANGDequant.metal)
inline uint extract_2bit(device const uint8_t* data, uint idx) {
    return (uint(data[idx >> 2]) >> ((idx & 3) << 1)) & 0x3;
}
inline uint extract_4bit(device const uint8_t* data, uint idx) {
    return (uint(data[idx >> 1]) >> ((idx & 1) << 2)) & 0xF;
}
inline uint extract_8bit(device const uint8_t* data, uint idx) {
    return uint(data[idx]);
}
inline uint extract_bits(device const uint8_t* data, uint bit_offset, uint bits) {
    uint byte_idx = bit_offset >> 3;
    uint bit_shift = bit_offset & 7;
    uint mask = (1u << bits) - 1u;
    uint32_t raw = uint32_t(data[byte_idx]);
    raw |= uint32_t(data[byte_idx + 1]) << 8;
    if (bits > 8 - bit_shift)
        raw |= uint32_t(data[byte_idx + 2]) << 16;
    return (raw >> bit_shift) & mask;
}


// ============================================================
// Kernel: Quantized Embedding Lookup
// ============================================================
//
// Extracts and dequantizes a single row from the quantized embedding table.
// Each thread handles one element of the output hidden vector.
//
// The embedding table has shape (vocab_size, hidden_dim), stored as
// blocks along the hidden dimension. For token_id T:
//   Row T starts at block index: T * blocks_per_row
//   Block b contains weights [b*64, (b+1)*64) of that row
//

kernel void jang_embedding_dequant(
    device const uint8_t*   qweight       [[buffer(0)]],
    device const half*      scales        [[buffer(1)]],
    device const half*      zeros         [[buffer(2)]],
    device const uint8_t*   bit_map       [[buffer(3)]],
    device const uint32_t*  block_offsets [[buffer(4)]],
    device half*            output        [[buffer(5)]],  // (hidden_dim,)
    constant uint&          token_id      [[buffer(6)]],
    constant uint&          hidden_dim    [[buffer(7)]],
    constant uint&          blocks_per_row [[buffer(8)]],
    uint                    gid           [[thread_position_in_grid]]
) {
    if (gid >= hidden_dim) return;

    // Which block in this row?
    uint in_row_block = gid / BLOCK_SIZE;
    uint in_block = gid % BLOCK_SIZE;

    // Global block index
    uint global_block = token_id * blocks_per_row + in_row_block;

    // Read block metadata
    uint bits = uint(bit_map[global_block]);
    half scale = scales[global_block];
    half zero = zeros[global_block];
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

    output[gid] = (half(raw) - zero) * scale;
}


// ============================================================
// Kernel: Tied Embedding LM Head (logits computation)
// ============================================================
//
// When tie_word_embeddings=true, logits are computed as:
//   logits[v] = dot(hidden, embed_row[v]) for all v in vocab
//
// This is equivalent to: logits = embed_matrix × hidden (treating embed
// as (vocab, hidden) and hidden as (hidden, 1)).
//
// We use jang_dequant_gemv for this — each threadgroup computes one
// output element (one vocab logit) by dequantizing one row of the
// embedding table and dotting it with the hidden state.
//
// This kernel is identical to jang_dequant_gemv but is here for clarity
// and future optimization (e.g., we could batch vocab rows for better
// memory coalescing since all rows use the same hidden vector).
//
// Use jang_dequant_gemv directly with:
//   N = vocab_size (output dim = number of logits)
//   K = hidden_dim (input dim = hidden state size)
//   x = hidden state from final norm
//   qweight/scales/zeros/bit_map/block_offsets = embed_tokens data
