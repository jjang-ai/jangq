# Audit: 02-llamacpp-gguf-architecture.md

Auditor: Claude Opus 4.6 (1M context)
Date: 2026-03-14
Sources: llama.cpp GitHub repository (ggml/include/ggml.h, ggml/src/ggml-common.h, ggml/src/ggml-quants.c, src/llama-quant.cpp, ggml/src/ggml-metal/ggml-metal.metal, ggml/docs/gguf.md)

---

## 1. GGUF File Format (Section 1)

### 1.1 Overview -- CORRECT

The description of GGUF as a self-contained, mmap-friendly format is accurate.

### 1.2 Header Fields

**[INCORRECT] Magic number value**
The document states: `0x47475546` in little-endian (ASCII "GGUF").

The actual bytes are `0x47 0x47 0x55 0x46`, which spell "GGUF" in ASCII. Interpreted as a uint32_t:
- Little-endian: `0x46554747`
- Big-endian: `0x47475546`

The document has the endianness labels reversed. The little-endian uint32 value is `0x46554747`, not `0x47475546`. The big-endian value in the document (`0x46554747`) is actually the little-endian value. The labels are swapped.

**CORRECT**: Version 3 is current, adds big-endian support.

**CORRECT**: n_tensors is uint64_t, n_kv is uint64_t.

### 1.2 Metadata KV Pairs

**[INCORRECT] String length prefix type**
The document implies string keys use a max of 65535 bytes. The GGUF spec defines `gguf_string_t` with a `uint64_t len` field (64-bit length), not uint32_t. The 65535-byte limit for keys is a convention, not a struct limitation. The document's claim that keys are "max 65535 bytes" is correct as a convention, but `gguf_string_t` itself uses uint64_t for the length field, which should be noted.

**CORRECT**: The 13 value types (IDs 0-12) and their mappings are accurate.

**CORRECT**: Hierarchical dot-separated namespace for keys.

**[UNCERTAIN] Required keys**
The document lists `general.quantization_version` and `general.alignment` as "required keys." These are optional metadata. Only `general.architecture` is truly required. The default alignment of 32 applies when `general.alignment` is absent.

### 1.3 Tensor Info Section

**CORRECT**: Tensor info structure with name (gguf_string_t, max 64 bytes), n_dims (uint32_t), dimensions (uint64_t[n_dims]), type (uint32_t), offset (uint64_t).

**CORRECT**: Offset is relative to tensor data section start, not file start.

**CORRECT**: Tensor naming convention (blk.N.attn_q.weight etc.) is accurate.

### 1.5 Alignment

**CORRECT**: Default alignment is 32 bytes. Formula is correct. Alignment purposes (SIMD, mmap, GPU, cache) are accurately described.

### 1.6 Memory Mapping

**CORRECT**: mmap() usage, zero-copy loading, MAP_SHARED with PROT_READ, demand paging, mlock() -- all accurately described.

**[VERIFY] Performance claims**: "100x faster on Linux" and "10x on macOS/Windows" for repeated loads are plausible but unverifiable without benchmarks. These are reasonable estimates of kernel file cache vs cold read differences.

### 1.7 Shard Support

**CORRECT**: Naming convention pattern is accurate. Each shard is self-contained. Splitting at tensor boundaries.

### 1.8 Quantization Type Enum

**CORRECT**: All ggml_type enum values verified against source:

| Type | Doc Value | Actual Value | Status |
|------|-----------|-------------|--------|
| GGML_TYPE_F32 | 0 | 0 | CORRECT |
| GGML_TYPE_F16 | 1 | 1 | CORRECT |
| GGML_TYPE_Q4_0 | 2 | 2 | CORRECT |
| GGML_TYPE_Q4_1 | 3 | 3 | CORRECT |
| GGML_TYPE_Q5_0 | 6 | 6 | CORRECT |
| GGML_TYPE_Q5_1 | 7 | 7 | CORRECT |
| GGML_TYPE_Q8_0 | 8 | 8 | CORRECT |
| GGML_TYPE_Q8_1 | 9 | 9 | CORRECT |
| GGML_TYPE_Q2_K | 10 | 10 | CORRECT |
| GGML_TYPE_Q3_K | 11 | 11 | CORRECT |
| GGML_TYPE_Q4_K | 12 | 12 | CORRECT |
| GGML_TYPE_Q5_K | 13 | 13 | CORRECT |
| GGML_TYPE_Q6_K | 14 | 14 | CORRECT |
| GGML_TYPE_Q8_K | 15 | 15 | CORRECT |
| GGML_TYPE_IQ2_XXS | 16 | 16 | CORRECT |
| GGML_TYPE_IQ2_XS | 17 | 17 | CORRECT |
| GGML_TYPE_IQ3_XXS | 18 | 18 | CORRECT |
| GGML_TYPE_IQ1_S | 19 | 19 | CORRECT |
| GGML_TYPE_IQ4_NL | 20 | 20 | CORRECT |
| GGML_TYPE_IQ3_S | 21 | 21 | CORRECT |
| GGML_TYPE_IQ2_S | 22 | 22 | CORRECT |
| GGML_TYPE_IQ4_XS | 23 | 23 | CORRECT |
| GGML_TYPE_I8 | 24 | 24 | CORRECT |
| GGML_TYPE_I16 | 25 | 25 | CORRECT |
| GGML_TYPE_I32 | 26 | 26 | CORRECT |
| GGML_TYPE_I64 | 27 | 27 | CORRECT |
| GGML_TYPE_F64 | 28 | 28 | CORRECT |
| GGML_TYPE_IQ1_M | 29 | 29 | CORRECT |
| GGML_TYPE_BF16 | 30 | 30 | CORRECT |
| GGML_TYPE_TQ1_0 | 34 | 34 | CORRECT |
| GGML_TYPE_TQ2_0 | 35 | 35 | CORRECT |
| GGML_TYPE_MXFP4 | 39 | 39 | CORRECT |
| GGML_TYPE_NVFP4 | 40 | 40 | CORRECT |

The document states "41 types" -- GGML_TYPE_COUNT = 41 but there are gaps (no 4, 5, 31-33, 36-38), so fewer than 41 actual types exist. The enum count is 41; the number of defined types is around 33.

---

## 2. Quantization Type Structs and BPW (Section 2)

### 2.3 Legacy Formats

**block_q4_0** -- CORRECT
- Struct: `ggml_half d; uint8_t qs[16]` = 18 bytes for 32 weights.
- BPW: (16 + 128) / 32 = 4.5. CORRECT.

**block_q4_1** -- CORRECT
- Struct: `{ggml_half d, m}; uint8_t qs[16]` = 20 bytes for 32 weights.
- BPW: (16 + 16 + 128) / 32 = 5.0. CORRECT.

**block_q5_0** -- CORRECT
- Struct: `ggml_half d; uint8_t qh[4]; uint8_t qs[16]` = 22 bytes for 32 weights.
- BPW: (16 + 32 + 128) / 32 = 5.5. CORRECT.

**block_q5_1** -- CORRECT
- Struct: `{ggml_half d, m}; uint8_t qh[4]; uint8_t qs[16]` = 24 bytes for 32 weights.
- BPW: (16 + 16 + 32 + 128) / 32 = 6.0. CORRECT.

**block_q8_0** -- CORRECT
- Struct: `ggml_half d; int8_t qs[32]` = 34 bytes for 32 weights.
- BPW: (16 + 256) / 32 = 8.5. CORRECT.
- The nearest_int trick with 12582912.0f (2^23 + 2^22) is verified in source. CORRECT.

### 2.4 K-Quant Family

**block_q2_K** -- CORRECT
- Struct: `uint8_t scales[16]; uint8_t qs[64]; {ggml_half d, dmin}` = 84 bytes for 256 weights.
- BPW: (128 + 512 + 32) / 256 = 2.625. CORRECT.
- Type-1 (asymmetric). CORRECT.
- 16 sub-blocks of 16 weights each. CORRECT.

**block_q3_K** -- CORRECT
- Struct: `uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; ggml_half d` = 110 bytes for 256 weights.
- BPW: (256 + 512 + 96 + 16) / 256 = 3.4375. CORRECT.
- Type-0 (symmetric). CORRECT.
- Scale encoding (6-bit packed into 12 bytes) description is accurate.

**block_q4_K** -- CORRECT
- Struct: `{ggml_half d, dmin}; uint8_t scales[12]; uint8_t qs[128]` = 144 bytes for 256 weights.
- BPW: (32 + 96 + 1024) / 256 = 4.5. CORRECT.
- Type-1 (asymmetric). CORRECT.
- 8 sub-blocks of 32 weights each. CORRECT.

**block_q5_K** -- CORRECT
- Struct: `{ggml_half d, dmin}; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]` = 176 bytes for 256 weights.
- BPW: (32 + 96 + 256 + 1024) / 256 = 5.5. CORRECT.
- Type-1. CORRECT.

**block_q6_K** -- CORRECT (struct), INCORRECT (scale formula)
- Struct: `uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; ggml_half d` = 210 bytes for 256 weights.
- BPW: (1024 + 512 + 128 + 16) / 256 = 6.5625. CORRECT.
- Type-0. CORRECT.
- 16 sub-blocks of 16 weights. CORRECT.
- **[INCORRECT] Scale formula**: The document states `actual_scale_i = d * scales[i] / 127.0`. The actual source uses `d * sc[i] * q` with NO division by 127. The formula is simply `weight = d * scale * quantized_value`. There is no normalization by 127.

**block_q8_K** -- CORRECT (struct), INCORRECT (stated total size vs calculated)
- Struct: `float d; int8_t qs[256]; int16_t bsums[16]` = 4 + 256 + 32 = 292 bytes for 256 weights.
- BPW: (32 + 2048 + 256) / 256 = 9.125. CORRECT.
- The document states "Total: 292 bytes" in the struct comment -- CORRECT.
- Purpose as intermediate format is correct.

### 2.5 Summary Table -- CORRECT

All entries verified. The table accurately summarizes block sizes, sub-block structure, scale bits, presence of min, and type classification.

### 2.6 Q4_K_M vs Q4_K_S

**[INCORRECT] Q4_K_S description**
The document states: "Q4_K_S (Small): All weight tensors use Q4_K uniformly."

In reality, Q4_K_S also promotes:
- FFN_DOWN in the first 1/8 of layers to Q5_K (for non-Falcon architectures)
- OUTPUT to Q6_K (via the general output handling logic)
- Norms remain F32

Q4_K_S is NOT uniform Q4_K across all weight tensors. It is heterogeneous, just less aggressively than Q4_K_M.

---

## 3. K-Quant Mixed Precision (Section 3)

### 3.2 use_more_bits() Heuristic

**CORRECT**: The function signature and logic match the source exactly:
```c
static bool use_more_bits(int i_layer, int n_layers) {
    return i_layer < n_layers/8 ||
           i_layer >= 7*n_layers/8 ||
           (i_layer - n_layers/8) % 3 == 2;
}
```
The description of what it selects (first 1/8, last 1/8, every 3rd in middle) is accurate.

### 3.3 Tensor Category System

**CORRECT**: The tensor_category enum exists in source with the listed categories. The sensitivity descriptions are reasonable heuristics (the source does not assign explicit sensitivity labels, but the quantization logic reflects this ranking).

**[UNCERTAIN] ATTENTION_QKV**: The document lists ATTENTION_QKV as a category. The source does include ATTENTION_QKV in the enum, and also includes ATTENTION_KV_B which the document omits.

### 3.4 Type Assignment Logic for Q4_K_M

This section has several errors:

**[INCORRECT] Item 2 -- TOKEN_EMBD**
The document claims TOKEN_EMBD gets Q2_K for Q4_K_M. The source shows TOKEN_EMBD gets Q2_K only for very low-bit ftypes (IQ2_XXS etc.). For Q4_K_M, TOKEN_EMBD retains the default Q4_K unless the model has tied embeddings (in which case it follows the OUTPUT logic and gets Q6_K).

**CORRECT: Item 3 -- OUTPUT**
OUTPUT gets Q6_K. Verified: the source has `if (new_type != GGML_TYPE_Q8_0) { new_type = GGML_TYPE_Q6_K; }` as fallback for the OUTPUT category.

**CORRECT: Item 4 -- ATTN_V**
Uses use_more_bits() to choose between Q6_K and the default Q4_K. Verified in source.

**[INCORRECT] Item 5 -- ATTN_OUTPUT**
The document claims: "if model is NOT mixture-of-experts: Q5_K (always elevated for non-MoE), else Q4_K."

The source shows NO special handling of ATTENTION_OUTPUT for Q4_K_M. There is no promotion to Q5_K for this tensor category under Q4_K_M. ATTENTION_OUTPUT retains the default Q4_K. The MoE/n_expert check for attention output does not exist in the Q4_K_M context.

**[INCORRECT] Items 7 -- ATTN_Q, ATTN_K**
The document claims these are promoted to Q5_K with use_more_bits(). The source shows NO Q4_K_M-specific handling for ATTENTION_Q or ATTENTION_K. They retain the default Q4_K.

**CORRECT: Item 6 -- FFN_DOWN**
FFN_DOWN does get promoted to Q6_K when use_more_bits() returns true. For Falcon architecture, the logic differs (Q6_K for early layers, Q5_K for use_more_bits layers, Q4_K otherwise).

**CORRECT: Item 8 -- FFN_GATE, FFN_UP**
These remain at Q4_K with no promotion. Verified.

**Corrected Q4_K_M assignment summary:**
```
NORMS/BIASES   --> F32
TOKEN_EMBD     --> Q4_K (default), or Q6_K if tied to output
OUTPUT         --> Q6_K
ATTN_V         --> Q6_K if use_more_bits(), else Q4_K
ATTN_OUTPUT    --> Q4_K (no special promotion)
ATTN_Q, ATTN_K --> Q4_K (no special promotion)
FFN_DOWN       --> Q6_K if use_more_bits(), else Q4_K
FFN_GATE       --> Q4_K
FFN_UP         --> Q4_K
```

### 3.5 Variant Differences

**[INCORRECT] Q4_K_S description** (repeated from 2.6)
Document: "Uniform Q4_K across all weight tensors. Only norms (F32) and output (Q6_K) differ."
Reality: Q4_K_S also promotes early FFN_DOWN layers (first 1/8) to Q5_K.

**[UNCERTAIN] Q3_K_M details**
Document claims Q3_K_M uses `n_layers/16` cutoff instead of `n_layers/8`. Source confirms Q3_K_M promotes FFN_DOWN: early layers (< n_layer/16) get Q5_K; layers matching use_more_bits or non-Falcon get Q4_K; otherwise Q3_K. The document's characterization is approximately correct but simplified.

---

## 4. Importance Quantization (Section 4)

### 4.2 IQ Type Catalog

**[INCORRECT] IQ2_XXS storage structure**
Document claims: `d(2) + qs(32) = 34B` giving 34 bytes.
But qs is uint16_t[32] = 64 bytes. Total: 2 + 64 = 66 bytes. The document's SIZE column says 66B, but the storage structure description `d(2) + qs(32)` is misleading -- it should say `d(2) + qs(64)` since qs is 32 uint16_t values = 64 bytes.

**[UNCERTAIN] IQ2_M**
Document lists IQ2_M at ~2.7 bpw. There is no GGML_TYPE_IQ2_M in the ggml_type enum. IQ2_M may not exist as a distinct type -- it could be a mixed-precision profile using IQ2_S and other types. This needs verification.

**CORRECT**: IQ1_S (50 bytes), IQ1_M (56 bytes), IQ2_XXS (66 bytes), IQ2_XS (74 bytes), IQ2_S (82 bytes), IQ3_XXS (98 bytes), IQ3_S (110 bytes), IQ4_NL (18 bytes), IQ4_XS (136 bytes) -- all struct sizes match source.

### 4.3 E8 Lattice and Codebook

**CORRECT (mostly)**: The description of E8 lattice properties (densest sphere packing in 8D, optimal quantization properties) is accurate.

**[INCORRECT] IQ2_XXS codebook size**
Document states: "The low 9 bits index into iq2xxs_grid (512 entries)."
Source confirms: iq2xxs_grid has 256 entries (not 512). The 512-entry grid belongs to IQ2_XS. The IQ2_XXS uses 8-bit indexing into a 256-entry grid.

**Corrected codebook sizes:**
- IQ2_XXS: 256-entry grid (8-bit index)
- IQ2_XS: 512-entry grid (9-bit index)
- IQ2_S: 1024-entry grid (10-bit index) -- CORRECT as stated

**[INCORRECT] IQ2_XXS encoding detail (Section 4.6)**
Document states: "Each uint16_t in qs encodes a codebook index and sign bits for a group of 8 weights. The low 9 bits index into iq2xxs_grid (512 entries)."

The actual encoding processes pairs of uint32_t (8 bytes) at a time to produce 32 weights in 4 sub-groups of 8. Individual byte values from the uint32_t pair are used as 8-bit grid indices into the 256-entry grid. Sign bits and sub-block scales are packed into the second uint32_t. The 9-bit / 512-entry claim is wrong for IQ2_XXS.

### 4.4 imatrix Computation

**CORRECT**: The description of importance matrix computation (squared activations, per-tensor statistics, calibration process) is accurate.

**[UNCERTAIN] imatrix mathematical representation**
Document: "Entry i represents the importance of row i." For a weight matrix W of shape (N, M), the imatrix is described as having N entries representing the diagonal of `<a * a^T>`. This is a reasonable description of the Fisher information approximation used, though the exact implementation details may vary.

### 4.5 imatrix Usage During Quantization

**CORRECT**: The importance-weighted error formula is verified in source:
```c
weight[l] = qw[l] * sqrtf(sigma2 + x[16*j + l]*x[16*j + l])
```
This matches the document's description.

### 4.6 IQ Type Deep Dives

**IQ2_XXS** -- see codebook size errors noted above.

**IQ2_XS** -- CORRECT struct (74 bytes, 512-entry E8 lattice grid).

**IQ2_S** -- CORRECT struct (82 bytes, 1024-entry grid, 10-bit index from qs + qh).

**IQ1_S** -- CORRECT struct (50 bytes). CORRECT description.

**IQ1_M** -- CORRECT struct (56 bytes). Elimination of explicit FP16 d field is accurately described.

**IQ3_XXS** -- CORRECT struct (98 bytes).

**IQ3_S** -- CORRECT struct (110 bytes). Sign bit separation accurately described.

**IQ4_NL** -- CORRECT struct (18 bytes, block size 32). Non-linear lookup table description is accurate.

**IQ4_XS** -- CORRECT struct (136 bytes for 256 weights).

### 4.9 Ternary Formats

**[INCORRECT] TQ1_0 struct size**
Document states: `uint8_t qs[51]` giving 57 bytes total.
Source formula: `qs[(QK_K - 4*QK_K/64) / 5]` = `qs[(256 - 16) / 5]` = `qs[48]`.
With qh[4] and d(2): 48 + 4 + 2 = **54 bytes**, not 57.
The qs array is 48 bytes, not 51.

**[INCORRECT] TQ1_0 bpw**
Document claims 1.6875 bpw. With 54 bytes for 256 weights: 54 * 8 / 256 = 1.6875. Despite the incorrect struct size in the document, the bpw happens to be stated correctly (because the document used 57 * 8 / 256 = 1.78125... wait, that does not equal 1.6875 either). Let me recalculate:
- Document's claim: 57 bytes, 1.6875 bpw. Check: 57*8/256 = 1.78125. This is INCONSISTENT.
- Actual: 54 bytes, bpw = 54*8/256 = 1.6875.
So the bpw of 1.6875 is correct, but it corresponds to 54 bytes, not 57. The struct definition in the document (qs[51]) is wrong.

**TQ2_0** -- CORRECT
- Struct: `uint8_t qs[64]; ggml_half d` = 66 bytes for 256 weights.
- BPW: 66 * 8 / 256 = 2.0625. CORRECT.

---

## 5. Metal Backend (Section 5)

### 5.1 Overview

**CORRECT**: Metal backend for Apple Silicon, MSL kernels, source file locations accurate.

### 5.2 Compute Pipeline Setup

**CORRECT**: MTLCreateSystemDefaultDevice, shader compilation modes (embedded vs runtime), PSO caching, function constants. All verified in source.

### 5.3 Dequantize-and-Multiply Pattern

**CORRECT**: The pattern of dequantizing into float4x4 registers during multiply-accumulate is verified in source. The fusion concept (no full materialization of dequantized weights in memory) is accurate.

### 5.4 Thread Dispatch

**CORRECT**: N_SIMDWIDTH = 32 is defined in the Metal shader source. simd_sum(), simd_max(), simd_shuffle() are used. The hierarchical reduction pattern (SIMD-level then threadgroup-level) is verified.

**[UNCERTAIN] simd_prefix_inclusive_sum()**: Not specifically verified, but it is a standard Metal SIMD intrinsic and likely used.

### 5.5 Memory Management

**CORRECT**: Shared and Private storage modes are accurate for Metal. The residency management description for macOS 15.0+ is plausible but not verified against the exact source.

### 5.7 Performance Characteristics

**[VERIFY] All benchmark numbers in the performance tables**
These are point-in-time measurements that change with every llama.cpp release, driver update, and macOS version. The general relationships are plausible:
- Token generation being memory-bandwidth-bound: CORRECT principle
- Prompt processing being compute-bound: CORRECT principle
- Efficiency decreasing with more GPU cores: CORRECT trend (diminishing returns)
- IQ types being slower than K-quants due to codebook lookups: CORRECT principle

The specific numbers (e.g., M1 at 14.15 t/s, M4 Max at 83.06 t/s) should be treated as illustrative, not authoritative.

---

## 6. Quality Preservation Techniques (Section 6)

### 6.5 Perplexity Measurements

**[VERIFY] All perplexity numbers**
Every benchmark number in the perplexity tables should be treated as approximate. These depend on:
- Exact llama.cpp version used
- Calibration data (WikiText-2 version)
- Model weights version
- Context length settings

The relative ordering (Q6_K better than Q4_K_M better than Q3_K_M etc.) is stable and reliable. The absolute numbers may differ from current measurements.

**[INCORRECT] Q3_K_M bpw in Llama 3.1 table**
The Llama 3.1 8B perplexity table lists Q3_K_M at 4.00 bpw. The raw Q3_K block format is 3.4375 bpw, but the mixed-precision Q3_K_M profile (which promotes some tensors to Q4_K and Q5_K) has a higher effective bpw. The 4.00 figure appears to be the effective bpw after mixed-precision promotion, which is different from the 3.4375 stated in the summary table (which is the raw Q3_K block bpw). This discrepancy should be clarified -- the summary table gives the base block format bpw, while the benchmark table gives the effective model-wide bpw.

### 6.6 Task-Specific Quantization Impact

**[VERIFY]** The benchmark scores for GSM8K, HellaSwag, MMLU, TruthfulQA, and IFEval are plausible but should be verified against the specific study referenced.

---

## 7. Architecture of llama.cpp Inference (Section 7)

### 7.1 Computational Graph

**CORRECT**: ggml_cgraph structure, lazy graph construction, architecture-specific builders are accurately described.

### 7.4 KV Cache

**CORRECT**: Ring buffer architecture, cell tracking with bitsets, sequence operations, sliding window attention, KV cache quantization options are all accurately described.

### 7.5-7.8 Batch Processing, Token Generation Loop

**CORRECT**: The batch structure, processing pipeline, token generation loop, unified vs multi-stream modes, and position shift mechanism are all accurately described.

---

## Summary of Errors

### INCORRECT (verified wrong):

1. **Magic number endianness labels swapped** (Section 1.2) -- LE value is 0x46554747, not 0x47475546
2. **Q6_K scale formula** (Section 2.4) -- No /127.0 division. Formula is `d * scales[i] * q`
3. **Q4_K_M: TOKEN_EMBD assignment** (Section 3.4) -- Gets Q4_K (default), not Q2_K
4. **Q4_K_M: ATTN_OUTPUT assignment** (Section 3.4) -- Stays Q4_K, not promoted to Q5_K
5. **Q4_K_M: ATTN_Q/ATTN_K assignment** (Section 3.4) -- Stay Q4_K, not promoted to Q5_K via use_more_bits
6. **Q4_K_S described as "uniform Q4_K"** (Sections 2.6, 3.5) -- Also promotes early FFN_DOWN to Q5_K and OUTPUT to Q6_K
7. **IQ2_XXS codebook size** (Sections 4.3, 4.6) -- 256 entries with 8-bit index, not 512 with 9-bit
8. **TQ1_0 struct** (Section 4.9) -- qs array is 48 bytes (not 51), total is 54 bytes (not 57)

### UNCERTAIN (cannot fully verify or partially correct):

1. **general.quantization_version listed as "required"** -- It is optional metadata
2. **IQ2_M as a type** -- No GGML_TYPE_IQ2_M in enum; may be a mixed-precision profile name
3. **Q3_K_M n_layers/16 cutoff detail** -- Approximately correct but simplified
4. **imatrix mathematical representation** -- Reasonable but implementation may differ in details

### VERIFY (benchmark/performance data that changes over time):

1. All perplexity numbers in Section 6.5 tables
2. All Metal performance numbers in Section 5.7 tables
3. All task-specific benchmark scores in Section 6.6
4. mmap performance improvement estimates (100x Linux, 10x macOS)
5. IQ vs K-quant speed comparison numbers (Section 5.7)

### Confident Correct:

1. All struct definitions for Q2_K through Q6_K, Q4_0 through Q8_0
2. All bits-per-weight calculations
3. All ggml_type enum values
4. use_more_bits() function definition
5. GGUF file structure (header, metadata, tensor info, alignment, tensor data)
6. Memory mapping description
7. Metal backend architecture (SIMD width, reduction patterns, dequantize-multiply fusion)
8. KV cache architecture
9. Computational graph and inference pipeline
10. IQ1_S, IQ1_M, IQ2_S, IQ2_XS, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS struct definitions and sizes
