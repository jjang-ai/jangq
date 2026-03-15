# llama.cpp and GGUF: How They Preserve Quality at Small Bits

## Table of Contents

1. [GGUF File Format](#1-gguf-file-format)
2. [Quantization Types in GGUF (the K-Quant System)](#2-quantization-types-in-gguf-the-k-quant-system)
3. [K-Quant Mixed Precision -- How llama.cpp Decides Which Layers Get More Bits](#3-k-quant-mixed-precision)
4. [Importance Quantization (IQ) -- llama.cpp's Frontier Innovation](#4-importance-quantization-iq)
5. [Metal Backend in llama.cpp](#5-metal-backend-in-llamacpp)
6. [Quality Preservation Techniques](#6-quality-preservation-techniques)
7. [Architecture of llama.cpp Inference](#7-architecture-of-llamacpp-inference)

---

## 1. GGUF File Format

### 1.1 Overview and Design Philosophy

GGUF (GGML Universal File Format) is the binary serialization format used by llama.cpp for storing large language models. Its core design principle is self-containment: all information needed to load and run a model -- architecture parameters, tokenizer vocabulary, quantization metadata, and tensor weights -- is encoded within the file itself. No external configuration files or sidecar metadata are required.

The format is explicitly designed for memory-mapped (mmap) access, enabling zero-copy model loading where tensor data on disk is mapped directly into virtual address space without intermediate read/copy steps.

### 1.2 File Structure

A GGUF file is organized into four sequential sections:

```
+------------------+
|      Header      |   Fixed-size header: magic, version, counts
+------------------+
|  Metadata KV     |   Variable-length key-value pairs
+------------------+
|  Tensor Info      |   Per-tensor metadata (name, shape, type, offset)
+------------------+
|   [padding]       |   Zero-pad to alignment boundary
+------------------+
|  Tensor Data      |   Raw binary tensor weights (bulk of the file)
+------------------+
```

#### Header Fields

| Field | Type | Description |
|-------|------|-------------|
| magic | uint32_t | `0x47475546` in little-endian (ASCII "GGUF") or `0x46554747` in big-endian |
| version | uint32_t | Format version; current is 3 (introduced big-endian support) |
| n_tensors | uint64_t | Total number of tensors stored in the file |
| n_kv | uint64_t | Number of metadata key-value pairs |

The magic number is the first 4 bytes readers check. Version 1 was the initial release, version 2 added certain metadata conventions, and version 3 (current) added big-endian support. Little-endian is the default and overwhelmingly common encoding.

#### Metadata Key-Value Pairs

Metadata entries follow the header and use a structured key-value encoding:

```
gguf_kv_t:
  key:    gguf_string_t        (UTF-8, max 65535 bytes, lower_snake_case with dot separators)
  type:   uint32_t             (value type enum)
  value:  <type-dependent>     (actual data)
```

Supported value types (13 total):

| Type ID | Type | Size |
|---------|------|------|
| 0 | uint8 | 1 byte |
| 1 | int8 | 1 byte |
| 2 | uint16 | 2 bytes |
| 3 | int16 | 2 bytes |
| 4 | uint32 | 4 bytes |
| 5 | int32 | 4 bytes |
| 6 | float32 | 4 bytes |
| 7 | bool | 1 byte |
| 8 | string | length-prefixed UTF-8 |
| 9 | array | type + count + elements |
| 10 | uint64 | 8 bytes |
| 11 | int64 | 8 bytes |
| 12 | float64 | 8 bytes |

Keys use a hierarchical dot-separated namespace. Required keys include:

- `general.architecture` -- model family (e.g., "llama", "qwen2", "phi3")
- `general.quantization_version` -- version of quantization scheme applied
- `general.alignment` -- global alignment value in bytes (default: 32)

Architecture-specific keys encode hyperparameters:

```
llama.context_length          = 4096       (uint32)
llama.embedding_length        = 4096       (uint32)
llama.block_count             = 32         (uint32)
llama.feed_forward_length     = 11008      (uint32)
llama.attention.head_count    = 32         (uint32)
llama.attention.head_count_kv = 32         (uint32)
llama.rope.freq_base          = 10000.0    (float32)
```

Tokenizer data is also stored as metadata:

```
tokenizer.ggml.model          = "llama"    (string)
tokenizer.ggml.tokens         = [...]      (array of strings)
tokenizer.ggml.scores         = [...]      (array of float32)
tokenizer.ggml.token_type     = [...]      (array of int32)
tokenizer.ggml.bos_token_id   = 1          (uint32)
tokenizer.ggml.eos_token_id   = 2          (uint32)
```

### 1.3 Tensor Info Section

Each tensor is described by a `gguf_tensor_info_t` entry:

```
gguf_tensor_info_t:
  name:       gguf_string_t     (UTF-8, max 64 bytes)
  n_dims:     uint32_t          (number of dimensions, max 4)
  dimensions: uint64_t[n_dims]  (size of each dimension)
  type:       uint32_t          (ggml_type enum)
  offset:     uint64_t          (byte offset into tensor data section)
```

The tensor name follows a naming convention that encodes the layer and component:

```
blk.0.attn_q.weight           # Layer 0 attention query weight
blk.0.attn_k.weight           # Layer 0 attention key weight
blk.0.attn_v.weight           # Layer 0 attention value weight
blk.0.attn_output.weight      # Layer 0 attention output projection
blk.0.ffn_gate.weight         # Layer 0 FFN gate (w1)
blk.0.ffn_up.weight           # Layer 0 FFN up projection (w3)
blk.0.ffn_down.weight         # Layer 0 FFN down projection (w2)
blk.0.attn_norm.weight        # Layer 0 attention RMS norm
blk.0.ffn_norm.weight         # Layer 0 FFN RMS norm
token_embd.weight             # Token embeddings
output_norm.weight            # Final RMS norm
output.weight                 # Output projection (LM head)
```

The `type` field specifies the quantization format used for that specific tensor. In mixed-precision quantization schemes (like Q4_K_M), different tensors can have different types -- this is how llama.cpp achieves per-layer bit allocation.

The `offset` field is relative to the start of the tensor data section (not the file start), making it simpler for writers to compute. Readers translate this to an absolute file offset for mmap by adding the size of all preceding sections plus alignment padding.

### 1.4 Tensor Data Section

Following the tensor info entries and alignment padding, the tensor data section contains the raw binary weight data for all tensors concatenated sequentially. This section constitutes the vast majority of the file size (typically 95%+ for quantized models).

Each tensor's data begins at an offset aligned to the global alignment boundary. Between tensors, zero-byte padding is inserted as needed.

### 1.5 Alignment Requirements

GGUF enforces 32-byte alignment by default (configurable via `general.alignment`). The alignment formula is:

```
aligned_offset = offset + (ALIGNMENT - (offset % ALIGNMENT)) % ALIGNMENT
```

Alignment serves multiple purposes:

1. **SIMD compatibility**: 32-byte alignment matches AVX2 register width (256 bits), enabling aligned vector loads without performance penalties
2. **mmap compatibility**: While mmap itself requires only page alignment (4096 bytes), having tensor data at 32-byte boundaries within a page ensures efficient SIMD access after mapping
3. **GPU requirements**: Metal and CUDA backends benefit from aligned data for efficient buffer binding
4. **Cache line efficiency**: Prevents quantized values from straddling 64-byte cache line boundaries on x86, avoiding unnecessary cache line loads

Padding bytes are always zero (`0x00`). Alignment applies at:
- The boundary between tensor info and tensor data
- Between individual tensor data blocks

### 1.6 Memory Mapping: Zero-Copy Loading

llama.cpp's model loading uses `mmap()` (or `CreateFileMapping()`/`MapViewOfFile()` on Windows) to map the GGUF file directly into virtual address space:

```c
// Simplified mmap loading flow:
fd = open("model.gguf", O_RDONLY);
file_size = lseek(fd, 0, SEEK_END);
addr = mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);

// Tensor pointers are direct offsets into the mapped region:
tensor->data = addr + header_size + tensor_info->offset;
```

Key properties of mmap-based loading:

- **Zero-copy**: The OS creates page table entries mapping virtual addresses to file pages. No `read()` system call copies data to heap memory. Tensor pointers point directly into the mapped file region.
- **Demand paging**: Pages are loaded from disk only when accessed. After a cold boot, the first inference pass triggers page faults that load weight data. Subsequent accesses hit the OS file cache, achieving near-RAM speed.
- **Shared mapping**: Using `MAP_SHARED` with `PROT_READ`, multiple inference processes can share the same physical pages of model weights. Running two llama.cpp instances with the same model does not double RAM usage for weights.
- **Memory locking**: `mlock()` can pin model pages in RAM to prevent the OS from evicting them under memory pressure, ensuring consistent inference latency.

The GGUF format's 32-byte tensor alignment ensures that when the file is memory-mapped, tensor data boundaries fall on SIMD-friendly addresses. The on-disk layout exactly matches the in-memory layout expected by inference kernels -- no data transformation, reordering, or copying is needed after mapping.

Performance impact: On Linux, mmap-based loading can be up to 100x faster than traditional file I/O for repeated loads (subsequent loads hit kernel file cache). On macOS/Windows, the improvement is approximately 10x due to different file cache behaviors. Additionally, models that exceed physical RAM can still be loaded via mmap -- the OS will page in/out portions as needed, at the cost of inference speed.

### 1.7 Shard Support for Large Models

Models too large for a single file (or exceeding filesystem limits) can be split across multiple GGUF shards. The naming convention is:

```
model-00001-of-00005.gguf
model-00002-of-00005.gguf
model-00003-of-00005.gguf
model-00004-of-00005.gguf
model-00005-of-00005.gguf
```

Pattern: `{basename}-{shard:05d}-of-{total:05d}.gguf`

Each shard is a valid, self-contained GGUF file with its own header, metadata, tensor info, and tensor data sections. The splitting is performed at tensor boundaries -- a single tensor is never split across shards.

The `gguf-split` tool handles splitting and merging:

```bash
# Split into shards of max N tensors each
llama-gguf-split --split --split-max-tensors 128 model.gguf output-prefix

# Merge shards back
llama-gguf-split --merge model-00001-of-00005.gguf merged.gguf
```

When loading, `llama_load_model_from_file()` detects multi-shard models automatically by examining the filename pattern. It opens all shard files and loads tensors from the appropriate shard based on tensor-to-shard mappings maintained in metadata.

### 1.8 Supported Quantization Types in GGUF

The `type` field in tensor info references the ggml_type enum, which defines 41 types:

| Enum | Value | Category |
|------|-------|----------|
| GGML_TYPE_F32 | 0 | Full precision |
| GGML_TYPE_F16 | 1 | Half precision |
| GGML_TYPE_Q4_0 | 2 | Legacy 4-bit |
| GGML_TYPE_Q4_1 | 3 | Legacy 4-bit |
| GGML_TYPE_Q5_0 | 6 | Legacy 5-bit |
| GGML_TYPE_Q5_1 | 7 | Legacy 5-bit |
| GGML_TYPE_Q8_0 | 8 | 8-bit |
| GGML_TYPE_Q8_1 | 9 | 8-bit |
| GGML_TYPE_Q2_K | 10 | K-quant 2-bit |
| GGML_TYPE_Q3_K | 11 | K-quant 3-bit |
| GGML_TYPE_Q4_K | 12 | K-quant 4-bit |
| GGML_TYPE_Q5_K | 13 | K-quant 5-bit |
| GGML_TYPE_Q6_K | 14 | K-quant 6-bit |
| GGML_TYPE_Q8_K | 15 | K-quant 8-bit |
| GGML_TYPE_IQ2_XXS | 16 | Importance 2-bit |
| GGML_TYPE_IQ2_XS | 17 | Importance 2-bit |
| GGML_TYPE_IQ3_XXS | 18 | Importance 3-bit |
| GGML_TYPE_IQ1_S | 19 | Importance 1-bit |
| GGML_TYPE_IQ4_NL | 20 | Importance 4-bit NL |
| GGML_TYPE_IQ3_S | 21 | Importance 3-bit |
| GGML_TYPE_IQ2_S | 22 | Importance 2-bit |
| GGML_TYPE_IQ4_XS | 23 | Importance 4-bit |
| GGML_TYPE_I8 | 24 | Integer 8 |
| GGML_TYPE_I16 | 25 | Integer 16 |
| GGML_TYPE_I32 | 26 | Integer 32 |
| GGML_TYPE_I64 | 27 | Integer 64 |
| GGML_TYPE_F64 | 28 | Double precision |
| GGML_TYPE_IQ1_M | 29 | Importance 1-bit M |
| GGML_TYPE_BF16 | 30 | Brain float 16 |
| GGML_TYPE_TQ1_0 | 34 | Ternary 1-bit |
| GGML_TYPE_TQ2_0 | 35 | Ternary 2-bit |
| GGML_TYPE_MXFP4 | 39 | Microsoft MXFP4 |
| GGML_TYPE_NVFP4 | 40 | NVIDIA NVFP4 |

---

## 2. Quantization Types in GGUF (the K-Quant System)

### 2.1 Evolution of Quantization in llama.cpp

The quantization system evolved through several generations:

1. **Legacy formats (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0)**: Simple per-block linear quantization with fixed block size of 32 weights. One scale factor (and optionally one offset/minimum) per block.

2. **K-quant family (Q2_K through Q6_K)**: Introduced super-blocks of 256 weights with hierarchical scaling -- sub-block scales are themselves quantized. This "double quantization" reduces metadata overhead while preserving local adaptation.

3. **IQ family (IQ1_S through IQ4_XS)**: Importance-weighted quantization using codebook-based (non-linear) value assignment, E8 lattice structures, and calibration-driven optimization. These achieve the best quality at extremely low bit widths.

4. **Ternary formats (TQ1_0, TQ2_0)**: Ternary (-1, 0, +1) quantization for extreme compression.

5. **Vendor-specific formats (MXFP4, NVFP4)**: Microsoft and NVIDIA float4 formats for hardware-accelerated inference on specific platforms.

### 2.2 Fundamental Concepts: Type-0 vs Type-1

All block-based quantization in llama.cpp follows one of two mathematical models:

**Type-0 (symmetric, scale-only)**:
```
weight_i = scale * q_i
```
where `q_i` is a signed integer quantized value and `scale` is a floating-point scalar per block. The zero point is implicitly at zero. Used by Q4_0, Q5_0, Q8_0, Q3_K, Q6_K.

**Type-1 (asymmetric, scale + minimum)**:
```
weight_i = scale * q_i + minimum
```
where `q_i` is an unsigned integer, `scale` controls the step size, and `minimum` provides an offset. This handles asymmetric weight distributions more effectively. Used by Q4_1, Q5_1, Q2_K, Q4_K, Q5_K.

### 2.3 Legacy Formats (Block Size = 32)

#### Q4_0 -- 4-bit Symmetric (4.5 bpw)

```c
typedef struct {
    ggml_half d;        // scale factor (FP16, 2 bytes)
    uint8_t qs[16];     // 32 x 4-bit values packed into 16 bytes
} block_q4_0;           // Total: 18 bytes for 32 weights
```

- **Bits per weight**: (16 + 128) / 32 = 4.5 bpw
- **Quantization**: Find max absolute value in block. Scale = max / 7. Each weight is rounded to nearest integer in [-8, 7]. Two 4-bit values are packed per byte (low nibble + high nibble).
- **Dequantization**: `weight = d * (q - 8)` where q is the unsigned 4-bit value
- **Quality**: Crude but fast. Loses significant information for non-symmetric distributions.

#### Q4_1 -- 4-bit Asymmetric (5.0 bpw)

```c
typedef struct {
    union { struct { ggml_half d, m; }; ggml_half2 dm; };
    uint8_t qs[16];     // 32 x 4-bit values packed into 16 bytes
} block_q4_1;           // Total: 20 bytes for 32 weights
```

- **Bits per weight**: (16 + 16 + 128) / 32 = 5.0 bpw
- **Quantization**: Computes both scale (d) and minimum (m). `q = round((x - min) / d)` for q in [0, 15].
- **Dequantization**: `weight = d * q + m`
- **Quality**: Better for asymmetric distributions but larger than Q4_0. Eventually supplanted by K-quants.

#### Q5_0 -- 5-bit Symmetric (5.5 bpw)

```c
typedef struct {
    ggml_half d;        // scale (2 bytes)
    uint8_t qh[4];      // 32 high bits, 1 per weight (4 bytes)
    uint8_t qs[16];     // 32 x low 4 bits packed (16 bytes)
} block_q5_0;           // Total: 22 bytes for 32 weights
```

- **Bits per weight**: (16 + 32 + 128) / 32 = 5.5 bpw
- **Encoding**: The 5th bit for each weight is stored separately in `qh`. The low 4 bits are packed as in Q4_0. To reconstruct: combine `qs[i/2] nibble` with `qh[i/32] bit` to get a 5-bit value.
- **Quality**: Noticeable improvement over Q4_0 with moderate size increase.

#### Q5_1 -- 5-bit Asymmetric (6.0 bpw)

```c
typedef struct {
    union { struct { ggml_half d, m; }; ggml_half2 dm; };
    uint8_t qh[4];      // high bits (4 bytes)
    uint8_t qs[16];     // low 4 bits (16 bytes)
} block_q5_1;           // Total: 24 bytes for 32 weights
```

- **Bits per weight**: (16 + 16 + 32 + 128) / 32 = 6.0 bpw
- **Quality**: Best legacy format short of Q8_0, but K-quants offer better quality at similar sizes.

#### Q8_0 -- 8-bit Symmetric (8.5 bpw)

```c
typedef struct {
    ggml_half d;        // scale (2 bytes)
    int8_t qs[32];      // 32 x 8-bit signed values (32 bytes)
} block_q8_0;           // Total: 34 bytes for 32 weights
```

- **Bits per weight**: (16 + 256) / 32 = 8.5 bpw
- **Quantization**: `q = round(x / d)` where `d = max_abs / 127`. Uses a bit-manipulation trick for rounding: adding the float `12582912.0f` (2^23 + 2^22) shifts the value into the integer portion of an IEEE 754 float, then extracting the low bits gives the rounded integer.
- **Quality**: Nearly lossless for most models (+0.0004 perplexity vs FP16). Used as an intermediate format during quantization and for KV cache storage.

### 2.4 K-Quant Family (Super-Block Size = 256)

The K-quant system, introduced by contributor Kawrakow, uses super-blocks of QK_K = 256 weights. The key innovation is hierarchical quantization: sub-block scales and minimums are themselves quantized, and a super-block-level scale factor controls the global range.

The constant `K_SCALE_SIZE = 12` defines the storage used for quantized scales in Q3_K, Q4_K, and Q5_K.

#### Q2_K -- 2-bit with 4-bit Scales (2.625 bpw)

```c
typedef struct {
    uint8_t scales[16];     // 4-bit quantized scales and minimums (16 bytes)
    uint8_t qs[64];         // 256 x 2-bit values packed (64 bytes)
    union {
        struct { ggml_half d, dmin; };
        ggml_half2 dm;
    };                      // super-block scale + min scale (4 bytes)
} block_q2_K;               // Total: 84 bytes for 256 weights
```

- **Bits per weight**: (128 + 512 + 32) / 256 = 2.625 bpw
- **Type**: Type-1 (asymmetric: `weight = scale * q + min`)
- **Structure**:
  - The 256 weights are divided into 16 sub-blocks of 16 weights each
  - Each sub-block has its own scale and minimum, quantized to 4 bits each
  - The 16 bytes of `scales[]` store 16 x 4-bit scale values interleaved with 16 x 4-bit minimum values
  - `d` is the FP16 super-block scale for reconstructing sub-block scales: `actual_scale = d * scales_q4`
  - `dmin` is the FP16 super-block scale for minimums: `actual_min = dmin * mins_q4`
  - Each weight is stored as a 2-bit unsigned integer (values 0-3)
- **Dequantization**:
  ```
  sub_scale = d * (scales[block_idx] & 0xF)
  sub_min   = dmin * (scales[block_idx] >> 4)
  weight    = sub_scale * q_2bit + sub_min
  ```
- **Quality**: Significant quality loss at 2.625 bpw. Perplexity increase of +0.6717 on 7B LLaMA-2. Usable for very large models where memory is the binding constraint.

#### Q3_K -- 3-bit with 6-bit Scales (3.4375 bpw)

```c
typedef struct {
    uint8_t hmask[32];      // high bit mask for 256 weights (32 bytes)
    uint8_t qs[64];         // 256 x low 2 bits packed (64 bytes)
    uint8_t scales[12];     // 16 x 6-bit quantized scales (12 bytes)
    ggml_half d;            // super-block scale (2 bytes)
} block_q3_K;               // Total: 110 bytes for 256 weights
```

- **Bits per weight**: (256 + 512 + 96 + 16) / 256 = 3.4375 bpw
- **Type**: Type-0 (symmetric: `weight = d * scale * q_signed`)
- **Structure**:
  - 16 sub-blocks of 16 weights
  - Each weight has 3 bits: 2 low bits packed in `qs[]`, 1 high bit in `hmask[]`
  - Sub-block scales are 6 bits each, packed into 12 bytes using a non-trivial packing scheme
  - The 12-byte `scales[]` array encodes 16 x 6-bit values: the low 4 bits of each scale are stored in the first 8 bytes; the upper 2 bits are packed into the remaining 4 bytes
  - `d` is the single FP16 super-block scale
- **Scale encoding detail**: For 16 sub-blocks with 6-bit scales packed into 12 bytes:
  - Bytes 0-7: Low 4 bits of scales 0-15 (two 4-bit values per byte)
  - Bytes 8-11: High 2 bits of scales 0-15 (four 2-bit values per byte)
- **Quality**: +0.2437 ppl for Q3_K_M on 7B models. Adequate for prototyping, not production.

#### Q4_K -- 4-bit with 6-bit Scales (4.5 bpw)

```c
typedef struct {
    union {
        struct { ggml_half d, dmin; };
        ggml_half2 dm;
    };                      // super-block scale + min scale (4 bytes)
    uint8_t scales[12];     // 8 x 6-bit scales + 8 x 6-bit mins (12 bytes)
    uint8_t qs[128];        // 256 x 4-bit values packed (128 bytes)
} block_q4_K;               // Total: 144 bytes for 256 weights
```

- **Bits per weight**: (32 + 96 + 1024) / 256 = 4.5 bpw
- **Type**: Type-1 (asymmetric)
- **Structure**:
  - 8 sub-blocks of 32 weights each
  - Each sub-block has a 6-bit scale and a 6-bit minimum
  - The `scales[12]` array stores 8 scales and 8 minimums using the same 6-bit packing as Q3_K
  - Each weight is stored as a 4-bit unsigned integer (values 0-15)
  - `d` and `dmin` are super-block scales for reconstructing actual sub-block parameters
- **Dequantization**:
  ```
  actual_scale = d * scale_6bit
  actual_min   = dmin * min_6bit
  weight       = actual_scale * q_4bit + actual_min
  ```
- **Quality**: The workhorse quantization. Q4_K_M achieves +0.0535 ppl on 7B models -- the recommended default for most use cases.

#### Q5_K -- 5-bit with 6-bit Scales (5.5 bpw)

```c
typedef struct {
    union {
        struct { ggml_half d, dmin; };
        ggml_half2 dm;
    };                      // super-block scale + min scale (4 bytes)
    uint8_t scales[12];     // 6-bit quantized scales and mins (12 bytes)
    uint8_t qh[32];         // 256 high bits, 1 per weight (32 bytes)
    uint8_t qs[128];        // 256 x low 4 bits packed (128 bytes)
} block_q5_K;               // Total: 176 bytes for 256 weights
```

- **Bits per weight**: (32 + 96 + 256 + 1024) / 256 = 5.5 bpw
- **Type**: Type-1 (asymmetric)
- **Structure**: Same sub-block organization as Q4_K (8 sub-blocks of 32). The 5th bit per weight is stored in `qh[]`, with low 4 bits in `qs[]`. Same 6-bit scale/min packing.
- **Quality**: +0.0142 ppl on 7B (Q5_K_M). Near-transparent quality loss. Recommended when memory allows.

#### Q6_K -- 6-bit with 8-bit Scales (6.5625 bpw)

```c
typedef struct {
    uint8_t ql[128];        // 256 x lower 4 bits (128 bytes)
    uint8_t qh[64];         // 256 x upper 2 bits packed (64 bytes)
    int8_t  scales[16];     // 16 x 8-bit signed scales (16 bytes)
    ggml_half d;            // super-block scale (2 bytes)
} block_q6_K;               // Total: 210 bytes for 256 weights
```

- **Bits per weight**: (1024 + 512 + 128 + 16) / 256 = 6.5625 bpw
- **Type**: Type-0 (symmetric)
- **Structure**:
  - 16 sub-blocks of 16 weights
  - Each weight has 6 bits: 4 low bits in `ql[]`, 2 high bits in `qh[]`
  - Sub-block scales are full 8-bit signed integers (no quantization of scales)
  - Single FP16 super-block scale `d`
  - Scale formula: `actual_scale_i = d * scales[i] / 127.0`
- **Quality**: +0.0008 ppl on 7B. Essentially lossless for practical purposes. Used as the "premium" type for critical tensors in mixed-precision schemes.

#### Q8_K -- 8-bit K-quant (8.5 bpw)

```c
typedef struct {
    float d;                // scale (4 bytes, full float)
    int8_t qs[256];         // 256 x 8-bit values (256 bytes)
    int16_t bsums[16];      // 16 x block sums (32 bytes)
} block_q8_K;               // Total: 292 bytes for 256 weights
```

- **Bits per weight**: (32 + 2048 + 256) / 256 = 9.125 bpw
- **Purpose**: Internal format used during quantization as an intermediate representation. Not typically stored in GGUF files. The `bsums` array stores pre-computed sums of 16-element sub-blocks, accelerating dot product computation by enabling partial-sum accumulation.

### 2.5 Summary Table: All K-Quant Types

| Type | Bits/Weight | Block Size | Sub-blocks | Scale Bits | Has Min | Type |
|------|-------------|------------|------------|------------|---------|------|
| Q2_K | 2.625 | 256 | 16 x 16 | 4-bit | Yes | Type-1 |
| Q3_K | 3.4375 | 256 | 16 x 16 | 6-bit | No | Type-0 |
| Q4_K | 4.5 | 256 | 8 x 32 | 6-bit | Yes | Type-1 |
| Q5_K | 5.5 | 256 | 8 x 32 | 6-bit | Yes | Type-1 |
| Q6_K | 6.5625 | 256 | 16 x 16 | 8-bit | No | Type-0 |
| Q8_K | 9.125 | 256 | 16 x 16 | float32 | No | Type-0 |

### 2.6 Q4_K_M vs Q4_K_S -- What "M" and "S" Mean

The `_S`, `_M`, and `_L` suffixes denote mixed-precision profiles, not different quantization algorithms:

- **Q4_K_S (Small)**: All weight tensors use Q4_K uniformly. Smallest file size for the 4-bit tier. Minimal per-layer differentiation.
- **Q4_K_M (Medium)**: Selectively promotes critical tensors to Q5_K or Q6_K while keeping most at Q4_K. The "medium" represents a balance between size and quality.
- **Q4_K_L (Large)**: Even more tensors promoted to higher precision. Largest file in the 4-bit tier.

The actual quantization type per tensor is determined at quantization time by `llama_tensor_get_type_impl()`, which examines the tensor's role in the model architecture and assigns types according to heuristic rules (detailed in Section 3).

Example: For a 7B LLaMA in Q4_K_M:
- Most tensors: Q4_K (4.5 bpw)
- Attention V projections in important layers: Q6_K (6.5625 bpw)
- Attention output projections: Q5_K (5.5 bpw)
- FFN down projections in important layers: Q6_K
- Result: ~4.89 bpw effective (vs 4.5 for uniform Q4_K_S, vs 4.67 for Q4_K_S on 8B)

Measured sizes on Llama 3.1-8B:
| Variant | Effective bpw | Size (GiB) |
|---------|---------------|-------------|
| Q4_K_S | 4.6672 | 4.36 |
| Q4_K_M | 4.8944 | 4.58 |

---

## 3. K-Quant Mixed Precision

### 3.1 The Core Insight: Not All Layers Are Equal

Different tensors in a transformer contribute unevenly to output quality. Quantizing all tensors identically wastes bits on unimportant tensors while starving critical ones. K-quant mixed precision exploits this by allocating more bits to sensitive tensors and fewer to tolerant ones, achieving better quality at the same average bit width.

### 3.2 The `use_more_bits()` Heuristic

The central function that determines which layers receive preferential treatment:

```c
static bool use_more_bits(int i_layer, int n_layers) {
    return i_layer < n_layers/8 ||
           i_layer >= 7*n_layers/8 ||
           (i_layer - n_layers/8) % 3 == 2;
}
```

This selects:
1. **First 1/8 of layers**: These process raw token embeddings. Errors here propagate through all subsequent layers, making them disproportionately impactful.
2. **Last 1/8 of layers**: These directly shape the output distribution. Degradation here directly affects token probabilities.
3. **Every 3rd layer in the middle 6/8**: Periodic "anchor" layers in the middle of the network that stabilize representation quality.

For a 32-layer model, this selects layers: 0-3 (first 4), 28-31 (last 4), and layers 6, 9, 12, 15, 18, 21, 24 (every 3rd in middle).

### 3.3 Tensor Category System

The function `tensor_get_category()` classifies each tensor by its architectural role:

| Category | Tensor Pattern | Sensitivity |
|----------|---------------|-------------|
| TOKEN_EMBD | `token_embd.weight` | Medium |
| OUTPUT | `output.weight` | High |
| ATTN_Q | `attn_q.weight` | Medium |
| ATTN_K | `attn_k.weight` | Medium |
| ATTN_V | `attn_v.weight` | Very High |
| ATTN_OUTPUT | `attn_output.weight` | High |
| ATTN_QKV (fused) | `attn_qkv.weight` | High |
| FFN_GATE | `ffn_gate.weight` | Medium-Low |
| FFN_UP | `ffn_up.weight` | Medium-Low |
| FFN_DOWN | `ffn_down.weight` | High |
| NORM | `*_norm.weight` | N/A (always F32) |

### 3.4 Type Assignment Logic in `llama_tensor_get_type_impl()`

The complete assignment logic for Q4_K_M (reconstructed from source):

```
For each tensor in the model:

1. NORMS and BIASES  --> always F32 (never quantized)

2. TOKEN_EMBD        --> Q2_K (embedding tables are large but tolerant)
                         For some architectures: Q4_K or IQ3_S

3. OUTPUT            --> Q6_K (output projection is critical for token probabilities)

4. ATTN_V:
   if use_more_bits(layer, n_layers):
       --> Q6_K      (6.5625 bpw for critical layers)
   else:
       --> Q4_K      (4.5 bpw for standard layers)

5. ATTN_OUTPUT:
   if model is NOT mixture-of-experts:
       --> Q5_K      (5.5 bpw -- always elevated for non-MoE)
   else:
       --> Q4_K      (MoE models have many more parameters, economize)

6. FFN_DOWN:
   if use_more_bits(layer, n_layers):
       --> Q6_K      (critical FFN layers get maximum treatment)
   elif layer < n_layers / 8:
       --> Q5_K      (early FFN layers get intermediate boost)
   else:
       --> Q4_K      (most FFN down projections at base precision)

7. ATTN_Q, ATTN_K, ATTN_QKV:
   if use_more_bits(layer, n_layers):
       --> Q5_K
   else:
       --> Q4_K

8. FFN_GATE, FFN_UP:
       --> Q4_K      (these are the least sensitive -- always base precision)

9. ALL OTHER:
       --> Q4_K      (default)
```

### 3.5 Variant Differences: S vs M vs L

The variants adjust the aggressiveness of bit promotion:

**Q3_K_S**:
- All tensors at Q3_K except output (Q6_K) and token embeddings
- Minimal promotion; smallest size in 3-bit tier

**Q3_K_M**:
- Uses `n_layers/16` cutoff instead of `n_layers/8` for the early-layer threshold
- Attention V and FFN down in select layers get Q4_K
- Attention output gets Q4_K

**Q3_K_L**:
- More layers promoted to Q4_K and Q5_K
- Attention V in important layers gets Q5_K

**Q4_K_S**:
- Uniform Q4_K across all weight tensors
- Only norms (F32) and output (Q6_K) differ

**Q4_K_M**:
- Promotes attention V to Q6_K in important layers
- Attention output to Q5_K
- FFN down to Q6_K in critical layers

**Architecture-specific adjustments**: Different model architectures (Falcon, MPT, etc.) may use different thresholds or type assignments based on their unique sensitivity patterns.

### 3.6 Why This Simple Heuristic Works

The `use_more_bits()` heuristic approximates what an ideal per-tensor sensitivity analysis would recommend, because:

1. **First layers matter most for information loss**: Early transformer layers extract basic features from token embeddings. Quantization errors in these layers corrupt the representation before all subsequent layers process it, creating compounding errors.

2. **Last layers directly control output**: The final layers shape the logit distribution. Even small quantization errors translate directly into changed token probabilities, affecting generation quality.

3. **Periodic anchoring prevents drift**: In deep transformers, quantization error can accumulate across layers. Having occasional high-precision "anchor" layers resets the error accumulation, preventing quality degradation from cascading.

4. **Attention V is the most sensitive projection**: The value projection directly determines what information the attention mechanism retrieves. Corrupting V means the model retrieves incorrect information. K and Q control where to attend (which is more robust to noise), but V controls what is retrieved (which must be precise).

5. **FFN down combines features**: The down projection in the FFN sub-layer reduces from hidden dimension back to model dimension. It acts as a bottleneck that aggregates feature information; errors here lose information irreversibly.

This heuristic achieves 80-90% of the quality benefit of full sensitivity analysis (which would require running calibration data through every possible quantization configuration) at zero computational cost.

---

## 4. Importance Quantization (IQ)

### 4.1 Overview

Importance Quantization (IQ) represents the most advanced quantization technique in llama.cpp. While K-quants use uniform (linear) quantization within each sub-block, IQ types use non-linear quantization with pre-computed codebooks, importance-weighted optimization, and lattice-based value selection. This allows IQ types to achieve significantly better quality at extremely low bit widths (1-3 bits per weight).

The IQ family was developed primarily by contributor ikawrakow, drawing inspiration from the QuIP# (Quantization with Incoherence Processing) research paper.

### 4.2 IQ Type Catalog

| Type | Effective bpw | Block Size | Storage Structure | Key Technique |
|------|---------------|------------|-------------------|---------------|
| IQ1_S | 1.5625 | 256 | `d(2) + qs(32) + qh(16)` = 50B | Ternary-like with grid |
| IQ1_M | 1.75 | 256 | `qs(32) + qh(16) + scales(8)` = 56B | Grid index + 3-bit scales |
| IQ2_XXS | 2.0625 | 256 | `d(2) + qs(64)` = 66B | Packed grid indices + signs |
| IQ2_XS | 2.3125 | 256 | `d(2) + qs(64) + scales(8)` = 74B | 512-point E8 lattice |
| IQ2_S | 2.5625 | 256 | `d(2) + qs(64) + qh(8) + scales(8)` = 82B | 1024-point E8 lattice |
| IQ2_M | ~2.7 | 256 | Based on IQ2_S structure | Enhanced E8 selection |
| IQ3_XXS | 3.0625 | 256 | `d(2) + qs(96)` = 98B | Compact 3-bit with grid |
| IQ3_S | 3.4375 | 256 | `d(2) + qs(64) + qh(8) + signs(32) + scales(4)` = 110B | E8 grid + explicit signs |
| IQ4_NL | 4.5 | 32 | `d(2) + qs(16)` = 18B | Non-linear mapping table |
| IQ4_XS | 4.25 | 256 | `d(2) + scales_h(2) + scales_l(4) + qs(128)` = 136B | Super-block with IQ4_NL sub-blocks |

### 4.3 The E8 Lattice and Codebook Approach

The central innovation of IQ quantization is replacing linear quantization grids with pre-computed codebooks derived from the E8 (Gosset) lattice.

**What is the E8 lattice?**
The E8 lattice is an 8-dimensional lattice that achieves the densest possible sphere packing in 8 dimensions. Its structure provides optimal quantization properties -- points in the E8 lattice are maximally spread in 8D space, minimizing expected quantization error for random vectors.

**How llama.cpp uses it:**

1. **Codebook construction**: A set of 256 or 512 points from the E8 lattice are pre-selected to serve as a codebook. These points are stored as static lookup tables in the source code (arrays like `iq2xxs_grid`, `iq2xs_grid`, `iq3s_grid`).

2. **Vector quantization**: Instead of quantizing each weight independently, IQ types quantize groups of 4 or 8 weights together as a vector. The quantizer finds the nearest codebook entry (E8 lattice point) to each weight vector.

3. **Sign handling**: Because E8 lattice points have a specific sign parity property (even number of negative components), signs are handled separately. A sign bitmask indicates which components should be negated.

4. **Index storage**: The GGUF file stores the codebook index (typically 8-9 bits) and sign bits for each group of weights, rather than the quantized values directly.

**Dequantization** at inference time:
```
1. Read codebook index from qs[]
2. Look up E8 lattice point in static grid table
3. Apply sign bitmask
4. Multiply by per-block scale factor d
5. Result is the dequantized weight vector
```

The codebook approach means that quantized values are non-uniformly spaced -- they cluster where the E8 lattice has good coverage, providing finer granularity where neural network weights are statistically more likely to land.

### 4.4 How Importance Matrices (imatrix) Are Computed

The importance matrix captures which weight dimensions have the greatest impact on model output. It is computed by running calibration data through the model and collecting activation statistics.

**Computation process:**

1. **Run calibration text** through the model using `llama-imatrix`:
   ```bash
   llama-imatrix -m model-f16.gguf -f calibration.txt -o imatrix.dat \
       --chunk 512 -ngl 99
   ```

2. **Collect squared activations**: For each tensor with N x M weights, the imatrix stores N values. Each value is the sum of squared activations `sum(a_i^2)` across all calibration tokens for dimension i. This represents how much each input dimension activates during real inference.

3. **Statistical computation**: The tool computes per-tensor statistics including:
   - Sum of squared activations: `Sigma(Act_i^2)` for each dimension
   - Entropy and normalized entropy
   - Z-score distribution
   - Cosine similarity between layers

4. **Output format**: Modern versions output GGUF format by default (legacy binary `.dat` format still supported). The imatrix contains one entry per input dimension per tensor.

**What the imatrix represents mathematically:**

For a weight matrix W of shape (N, M), the imatrix has N entries. Entry i represents the importance of row i -- if changing weight W[i,j] by a small amount causes a large change in output, then imatrix[i] is large. This is approximated by the diagonal of the expected outer product of activations: `<a * a^T>_diagonal`.

**Calibration data selection:**

- Should be representative of the target domain (general text for general-purpose models)
- Using random/garbage tokens risks corrupting attention patterns
- For language-specific deployment, calibration data in that language improves results
- Typical calibration uses Wikipedia articles or a mix of web text
- Text is processed in chunks (default 512 tokens per chunk)
- Typically 100-500 chunks are sufficient for stable estimates

### 4.5 How the imatrix Is Used During Quantization

When quantizing with an imatrix, the importance weights modify the error metric used during codebook selection:

```c
// Without imatrix: minimize unweighted MSE
error = sum((original[i] - quantized[i])^2)

// With imatrix: minimize importance-weighted MSE
error = sum(importance[i] * (original[i] - quantized[i])^2)
```

This means:
- High-importance dimensions tolerate less quantization error
- Low-importance dimensions can accept more error
- The quantizer allocates representational precision proportional to dimension importance
- For E8 codebook selection, the imatrix weights influence which lattice point is chosen as "nearest"

The quantization function receives importance weights through the `quant_weights` parameter:

```c
// Importance-weighted quantization (simplified):
weight[j] = qw[j] * sqrt(sigma2 + x[j]*x[j])
```

where `qw[j]` is the importance weight and `sigma2` is the variance across the row. This focuses precision on dimensions that are both important (high imatrix value) and large in magnitude.

### 4.6 Specific IQ Type Deep Dives

#### IQ2_XXS (2.0625 bpw)

The most compact 2-bit format. Structure:

```c
typedef struct {
    ggml_half d;            // super-block scale (2 bytes)
    uint16_t qs[32];        // grid indices + sign info (64 bytes)
} block_iq2_xxs;            // Total: 66 bytes for 256 weights
```

Each `uint16_t` in `qs` encodes a codebook index and sign bits for a group of 8 weights. The low 9 bits index into `iq2xxs_grid` (512 entries), and remaining bits encode signs and sub-block scales.

Dequantization:
```c
// For each group of 8 weights:
grid_index = qs[i] & 511;          // 9-bit index
grid_values = iq2xxs_grid[grid_index]; // 8 pre-computed values
// Apply signs and scale
```

#### IQ2_XS (2.3125 bpw)

Adds per-sub-block scale refinement:

```c
typedef struct {
    ggml_half d;            // super-block scale (2 bytes)
    uint16_t qs[32];        // grid indices (64 bytes)
    uint8_t scales[8];      // sub-block scales (8 bytes)
} block_iq2_xs;             // Total: 74 bytes for 256 weights
```

Uses 512 E8 lattice points. The `scales` array provides 4-bit scale adjustments for groups of 32 weights, allowing local adaptation beyond the super-block scale.

#### IQ2_S (2.5625 bpw)

Uses 1024 E8 lattice points (doubled codebook):

```c
typedef struct {
    ggml_half d;            // super-block scale (2 bytes)
    uint8_t qs[64];         // grid indices (64 bytes)
    uint8_t qh[8];          // high bits for grid indices (8 bytes)
    uint8_t scales[8];      // sub-block scales (8 bytes)
} block_iq2_s;              // Total: 82 bytes for 256 weights
```

The 10-bit codebook index (from `qs` + `qh`) addresses 1024 E8 lattice points, doubling the codebook resolution compared to IQ2_XS. This additional precision explains why IQ2_S outperforms Q2_K despite having similar bits per weight.

#### IQ1_S (1.5625 bpw) and IQ1_M (1.75 bpw)

The most extreme compression formats:

```c
typedef struct {         // IQ1_S
    ggml_half d;         // super-block scale (2 bytes)
    uint8_t qs[32];      // grid indices (32 bytes)
    uint16_t qh[8];      // high bits + shift (16 bytes)
} block_iq1_s;           // Total: 50 bytes for 256 weights

typedef struct {         // IQ1_M
    uint8_t qs[32];      // grid index low (32 bytes)
    uint8_t qh[16];      // grid index high + shift (16 bytes)
    uint8_t scales[8];   // 3-bit block scales (8 bytes)
} block_iq1_m;           // Total: 56 bytes for 256 weights
```

IQ1_M eliminates the explicit FP16 `d` field and instead distributes scale information across the `scales` array with 3-bit per-block scales, squeezing more quality from minimal bits.

At 1.56-1.75 bpw, these formats produce models that are 8-10x smaller than FP16. Quality is significantly degraded but remains functional for very large models (70B+) where the large parameter count compensates for per-parameter information loss.

#### IQ3_XXS (3.0625 bpw) and IQ3_S (3.4375 bpw)

```c
typedef struct {         // IQ3_XXS
    ggml_half d;         // super-block scale (2 bytes)
    uint8_t qs[96];      // packed grid indices + signs (96 bytes)
} block_iq3_xxs;         // Total: 98 bytes for 256 weights

typedef struct {         // IQ3_S
    ggml_half d;         // super-block scale (2 bytes)
    uint8_t qs[64];      // grid indices (64 bytes)
    uint8_t qh[8];       // high bits (8 bytes)
    uint8_t signs[32];   // explicit sign bits (32 bytes)
    uint8_t scales[4];   // sub-block scales (4 bytes)
} block_iq3_s;           // Total: 110 bytes for 256 weights
```

IQ3_S separates sign bits explicitly (`signs[32]` = 256 bits for 256 weights), making dequantization slightly faster at the cost of 0.4 bpw. The `scales[4]` provides 4 x 8-bit sub-block scales (QK_K/64 = 4 sub-blocks).

#### IQ4_NL (4.5 bpw) -- Non-Linear 4-bit

```c
typedef struct {
    ggml_half d;         // scale (2 bytes)
    uint8_t qs[16];      // 32 x 4-bit indices (16 bytes)
} block_iq4_nl;          // Total: 18 bytes for 32 weights
```

Unlike other IQ types, IQ4_NL uses a small block size of 32 (like legacy formats) and a 16-entry non-linear lookup table instead of the E8 lattice. Each 4-bit value indexes into a table of optimized float values that are non-uniformly spaced to match typical weight distributions.

This makes IQ4_NL fast (comparable to Q4_0) while providing quality improvements from non-linear value spacing. It does not require an imatrix.

#### IQ4_XS (4.25 bpw) -- Super-block IQ4

```c
typedef struct {
    ggml_half d;            // super-block scale (2 bytes)
    uint16_t scales_h;      // high bits of scales (2 bytes)
    uint8_t scales_l[4];    // low bits of scales (4 bytes)
    uint8_t qs[128];        // 256 x 4-bit values (128 bytes)
} block_iq4_xs;             // Total: 136 bytes for 256 weights
```

Applies the IQ4_NL non-linear mapping at super-block scale with additional per-sub-block scale refinement. The `scales_h` and `scales_l` together encode sub-block scale adjustments.

### 4.7 Why IQ2_M at ~2.7 bpw Significantly Outperforms Q2_K at 2.625 bpw

Despite having similar bits per weight, IQ2_M dramatically outperforms Q2_K because of fundamental algorithmic differences:

1. **Non-uniform vs uniform quantization**: Q2_K maps weights to 4 uniformly spaced values (0, 1, 2, 3) within each sub-block range. IQ2_M maps groups of 8 weights to E8 lattice points -- each point is an 8-dimensional vector carefully positioned to minimize quantization error for typical weight distributions.

2. **Correlated vs independent quantization**: Q2_K quantizes each weight independently. IQ2_M quantizes vectors of 8 weights jointly, exploiting correlations between adjacent weights to find a better joint representation.

3. **Optimized codebook vs linear grid**: The E8 lattice codebook is information-theoretically close to optimal for quantizing Gaussian-like distributions. The 4 levels of Q2_K are far from optimal, especially for the heavy-tailed distributions typical of neural network weights.

4. **Importance weighting**: IQ2_M uses the importance matrix to focus precision on dimensions that matter most for output quality. Q2_K treats all dimensions equally.

5. **Superior scale encoding**: IQ2_M's scale system (combining super-block FP16 scale with per-sub-block adjustments) provides finer local adaptation than Q2_K's 4-bit sub-block scales.

The net effect: IQ2_M at 2.7 bpw typically achieves perplexity comparable to Q3_K at 3.4 bpw -- a full bit less for similar quality.

### 4.8 Trellis Quantization

Trellis-coded quantization (TCQ) is a newer technique being explored for integration into llama.cpp, based on the QTIP (Quantization with Trellises and Incoherence Processing) research. TCQ replaces the vector quantizer with a trellis quantizer that models sequential dependencies between quantization decisions.

In a trellis quantizer:
- Quantization decisions are modeled as paths through a state machine (trellis)
- Each state transition produces a quantized value
- The Viterbi algorithm finds the optimal path (sequence of quantized values) that minimizes total distortion
- This captures sequential correlations that independent vector quantization misses

The connection to llama.cpp: since llama.cpp's IQ vector quantizer is based on QuIP#'s E8 vector quantizer, QTIP's trellis quantizer can potentially replace it. The trellis approach achieves state-of-the-art compression quality by combining incoherence processing (random rotations to spread weight information uniformly) with the sequential optimization of trellis coding.

### 4.9 Ternary Quantization Formats

#### TQ1_0 (1.6875 bpw)

```c
typedef struct {
    uint8_t qs[51];      // ternary values, 5 per byte via base-3 (51 bytes)
    uint8_t qh[4];       // overflow values, 4 per byte (4 bytes)
    ggml_half d;         // scale (2 bytes)
} block_tq1_0;           // Total: 57 bytes for 256 weights
```

Encodes values as ternary (-1, 0, +1). Five ternary digits pack into one byte using base-3 encoding: `byte = t0 + 3*t1 + 9*t2 + 27*t3 + 81*t4` (max value = 242, fits in uint8). The encoding uses ceiling division: `q = ((uint16_t)q * 256 + 242) / 243`.

#### TQ2_0 (2.0625 bpw)

```c
typedef struct {
    uint8_t qs[64];      // 2 bits per element (64 bytes)
    ggml_half d;         // scale (2 bytes)
} block_tq2_0;           // Total: 66 bytes for 256 weights
```

Four ternary values per byte using simple 2-bit encoding. Less compact than TQ1_0's base-3 packing but simpler to decode.

---

## 5. Metal Backend in llama.cpp

### 5.1 Overview

The Metal backend provides GPU acceleration for llama.cpp on Apple Silicon (M1/M2/M3/M4) using Apple's Metal framework. It implements all critical tensor operations as Metal Shading Language (MSL) kernels, with specialized implementations for each quantization format.

Key source files:
- `ggml/src/ggml-metal/ggml-metal.m` -- Objective-C host code: device management, pipeline compilation, command buffer dispatch
- `ggml/src/ggml-metal/ggml-metal.metal` -- MSL shader code: kernel implementations for all operations

### 5.2 Compute Pipeline Setup

The Metal backend initializes through several stages:

1. **Device detection**: Queries `MTLCreateSystemDefaultDevice()` to identify the GPU, its capabilities (GPU family, feature set), and available memory.

2. **Shader compilation**: The Metal shader library (`ggml-metal.metal`) is compiled either:
   - At build time (embedded library via `GGML_METAL_EMBED_LIBRARY=ON`) -- faster startup
   - At runtime from source -- more flexible for development

3. **Pipeline state objects (PSOs)**: Each kernel function is compiled into a `MTLComputePipelineState` with specific function constants. PSOs are cached in a hash map to avoid recompilation.

4. **Function constants**: Metal function constants enable compile-time specialization. For example:
   ```metal
   constant int nsg [[function_constant(0)]];   // number of SIMD groups
   constant int nr0 [[function_constant(1)]];   // number of rows processed
   ```
   This allows the compiler to optimize each kernel variant for specific configurations.

5. **Pipeline naming**: Cached PSOs use descriptive names encoding their configuration:
   ```
   kernel_mul_mv_q4_0_f32_nsg=2_nr0=4
   kernel_mul_mv_q4_K_f32_nsg=2_nr0=4
   kernel_mul_mv_iq4_xs_f32_nsg=2_nr0=2
   ```

### 5.3 The Dequantize-and-Multiply Pattern

The most performance-critical operation in LLM inference is matrix-vector multiplication (for token generation) and matrix-matrix multiplication (for prompt processing). The Metal backend fuses dequantization with multiplication to minimize memory traffic.

**Architecture:**

Each quantization format has a corresponding `dequantize_*` function that converts a quantized block into a 4x4 float matrix:

```metal
// Template pattern for dequantization (simplified):
void dequantize_q4_0(device const block_q4_0 * src, short il, thread float4x4 & reg) {
    float d = src->d;
    // Extract 4-bit values, subtract 8 for signed, multiply by scale
    for (int i = 0; i < 4; i++) {
        uint8_t byte = src->qs[il * 4 + i];
        reg[i] = float4(
            d * ((byte & 0xF) - 8),
            d * ((byte >> 4) - 8),
            ...
        );
    }
}
```

For IQ types with codebook lookups:

```metal
void dequantize_iq2_xs(device const block_iq2_xs * src, short il, thread float4x4 & reg) {
    // Look up E8 lattice point from codebook
    uint16_t idx = src->qs[il];
    uint32_t grid_val = iq2xs_grid[idx & 511];
    // Apply signs and scale
    float d = src->d * (src->scales[il/4] & 0xF);
    // Unpack grid values and apply to reg
    ...
}
```

The matrix-vector multiplication kernel then:

1. Loads quantized blocks from device memory
2. Calls the dequantize function to produce float4x4 tiles
3. Accumulates dot products in registers
4. Performs SIMD reduction across the threadgroup
5. Writes the result

This fusion means quantized weights are never fully materialized in memory -- they are dequantized directly into registers during the multiply-accumulate loop, saving memory bandwidth.

### 5.4 Thread Dispatch: Grid and Threadgroup Sizes

The Metal backend carefully tunes dispatch parameters for Apple GPU architecture:

**SIMD Width**: All Apple GPUs use 32 threads per SIMD group (warp). This is the fundamental unit of parallel execution.

**Threadgroup organization**: Kernels specify threadgroup sizes based on the operation:

- **Matrix-vector multiplication** (`kernel_mul_mv_*`):
  - Threadgroup: typically 32-256 threads
  - Each SIMD group processes multiple rows of the matrix
  - `nsg` (number of SIMD groups) controls how many SIMD groups per threadgroup (typically 2-4)
  - `nr0` controls how many rows each SIMD group processes (typically 2-8)

- **Matrix-matrix multiplication** (`kernel_mul_mm_*`):
  - Threadgroup: typically 32x32 or similar 2D configurations
  - Uses shared memory (threadgroup memory) for tiling

- **Flash attention**:
  - Supports masking and sliding window
  - Threadgroup size tuned for attention head dimensions

**Grid dispatch**: The grid maps to tensor dimensions:
```
grid_x = number of output rows (or row groups)
grid_y = batch dimension
grid_z = higher dimensions
```

**SIMD intrinsics used**:
- `simd_sum()` -- parallel reduction across 32 threads (requires Apple7+ GPU family)
- `simd_max()` -- parallel maximum
- `simd_prefix_inclusive_sum()` -- scan operation
- `simd_shuffle()` -- data exchange within SIMD group

**Hierarchical reduction pattern**:
```metal
// Level 1: SIMD-level reduction (hardware-accelerated)
float local_sum = simd_sum(thread_partial);

// Level 2: Cross-SIMD reduction via threadgroup memory (if needed)
if (tptg.x > N_SIMDWIDTH) {
    threadgroup_memory[sgid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        float total = 0;
        for (int i = 0; i < nsg; i++) total += threadgroup_memory[i];
    }
}
```

### 5.5 Memory Management

**Buffer allocation strategies:**

| Strategy | Metal Mode | Use Case |
|----------|-----------|----------|
| Shared | `MTLResourceStorageModeShared` | Small tensors, host-device transfers, KV cache updates |
| Private | `MTLResourceStorageModePrivate` | Large model weights (GPU-only, highest bandwidth) |

For model weights (which are read-only during inference), private storage mode is optimal because:
- The GPU can access them at full memory bandwidth without CPU cache coherency overhead
- Weights loaded via mmap can be transferred to private buffers once

**Residency management**: On macOS 15.0+ and iOS 18.0+, the backend uses `MTLResidencySet` to keep GPU memory resident and prevent the OS from paging it out. A background thread executes residency requests every 500ms to maintain GPU memory pressure awareness.

### 5.6 KV Cache on Metal

The KV cache stores attention key and value tensors from previously processed tokens. On Metal:

- KV cache tensors are allocated as Metal buffers (typically shared mode for CPU writability)
- Cache type can be quantized: Q8_0 for ~2x savings, Q4_0 for ~4x savings (with quality tradeoffs)
- Flash attention kernels read directly from KV cache buffers
- The backend tracks which cache regions are accessed for each operation to determine concurrency feasibility
- Cache updates write new K/V vectors using `ggml_set_1d()` for contiguous slots or `ggml_set_rows()` for scattered slots

### 5.7 Performance Characteristics on Apple Silicon

Performance is primarily bound by memory bandwidth for token generation (auto-regressive, batch size 1) and by compute for prompt processing (batch size > 1).

**Token Generation (TG) -- Memory Bandwidth Bound:**

| Chip | GPU Cores | Bandwidth | Q4_0 TG (7B) | Efficiency |
|------|-----------|-----------|---------------|------------|
| M1 | 8 | 68 GB/s | 14.15 t/s | ~85% BW utilization |
| M1 Pro | 16 | 200 GB/s | 36.41 t/s | ~75% |
| M1 Max | 32 | 400 GB/s | 61.19 t/s | ~63% |
| M1 Ultra | 64 | 800 GB/s | 83.73 t/s | ~43% |
| M2 | 10 | 100 GB/s | 21.91 t/s | ~89% |
| M2 Pro | 19 | 200 GB/s | 38.86 t/s | ~79% |
| M2 Max | 38 | 400 GB/s | 65.95 t/s | ~67% |
| M2 Ultra | 76 | 800 GB/s | 94.27 t/s | ~48% |
| M3 Pro | 18 | 150 GB/s | 30.74 t/s | ~83% |
| M3 Max | 40 | 400 GB/s | 66.31 t/s | ~67% |
| M4 | 10 | 120 GB/s | 24.11 t/s | ~82% |
| M4 Max | 40 | 546 GB/s | 83.06 t/s | ~62% |

Token generation speed scales with memory bandwidth because each token requires reading the entire model from memory once. A Q4_0 7B model is ~3.8 GB; at 68 GB/s (M1), reading 3.8 GB takes ~56ms, allowing ~18 tokens/second theoretically. The ~80% efficiency accounts for computation overhead and memory access patterns.

**Prompt Processing (PP) -- Compute Bound:**

| Chip | GPU Cores | Q4_0 PP (7B) t/s |
|------|-----------|-------------------|
| M1 | 8 | 117.96 |
| M1 Max | 32 | 530.06 |
| M2 Max | 38 | 671.31 |
| M3 Max | 40 | 759.70 |
| M4 Max | 40 | 885.68 |

PP scales nearly linearly with GPU core count because matrix-matrix multiplication is compute-bound with high arithmetic intensity.

**IQ quant performance considerations**:

IQ quantization types (especially IQ2_XS) are slower on Apple Silicon than K-quants at equivalent bit widths because:
- Codebook lookups require scattered memory access (random indexing into lookup tables)
- Apple GPUs are optimized for sequential, predictable memory access patterns
- The E8 codebook fits in L1 cache but cannot reside in the GPU register file
- Comparison: Q4_K achieves ~63 t/s on M2 Max vs ~50 t/s for IQ2_XS -- a 20% penalty despite IQ2_XS using fewer bits

However, M4 generation shows significant improvement: M4 Max achieves 17.42 t/s with IQ4_XS, matching Q4_K_M performance, suggesting Apple is improving scattered memory access patterns in newer GPU architectures.

NVIDIA GPUs show much less IQ overhead: an RTX 4080 achieves 175 t/s for IQ2_XS vs 50 t/s on M2 Max -- a 3.5x advantage that reflects NVIDIA's superior handling of codebook-style access patterns.

---

## 6. Quality Preservation Techniques

### 6.1 Why Naive Uniform Quantization Fails at 2-bit

At 2 bits per weight, a uniform linear quantizer provides only 4 distinct values per weight. Consider what this means:

- A block of 32 weights, each with 4 possible values, can represent only 4^32 ~ 1.8 x 10^19 distinct weight configurations
- The same block in FP16 has (2^16)^32 ~ 10^154 possible configurations
- This represents a compression ratio of about 10^135:1 in information content

For the model to remain functional, the 4 values chosen for each block must be extraordinarily well-placed. The problems with naive 2-bit:

1. **Uniform spacing wastes levels**: Neural network weights follow approximately Gaussian distributions with heavy tails. Uniform quantization places levels at equal intervals, but most weights cluster near zero. The outer levels (representing tail values) are rarely used, while the inner levels (near zero) lack resolution where most weights exist.

2. **No inter-weight correlation exploitation**: Each weight is quantized independently, ignoring statistical correlations between adjacent weights in the same layer. In practice, weight vectors have structured patterns that could be exploited for more efficient encoding.

3. **Scale overhead dominates**: At 2 bits per weight, the FP16 scale factor (16 bits) shared across 32 weights adds 0.5 bpw overhead, making the effective rate 2.5 bpw -- a 25% overhead just for the scale.

4. **Outlier destruction**: Neural network weights often have rare but important outliers. With only 4 levels, outliers are either truncated (losing critical information) or force the scale to expand (reducing resolution for all other weights).

### 6.2 How K-Quants Improve on This

K-quants address these problems through hierarchical scaling:

**Super-block structure**: By using 256-weight super-blocks with 16 sub-blocks, K-quants amortize scale overhead. In Q2_K:
- 256 weights x 2 bits = 512 bits for values
- 16 sub-blocks x 8 bits (4-bit scale + 4-bit min) = 128 bits for local scales
- 32 bits for super-block scales
- Total: 672 bits for 256 weights = 2.625 bpw

The double-quantization (scales themselves are quantized) reduces scale overhead from what would be 16 bits x 16 sub-blocks = 256 bits (1.0 bpw) to just 128 + 32 = 160 bits (0.625 bpw).

**Asymmetric quantization**: Type-1 K-quants store both scale and minimum per sub-block, handling asymmetric weight distributions (which are common in FFN layers). This effectively shifts the quantization grid to center on the actual weight distribution rather than zero.

### 6.3 How IQ Types Achieve Usable 2-bit

IQ types employ four advanced techniques that together produce dramatically better quality at ultra-low bit widths:

#### 6.3.1 Non-Uniform Value Spacing (Optimized Codebooks)

Instead of 4 uniformly-spaced values, IQ types use codebook entries derived from the E8 lattice. These codebook values are:
- Non-uniformly spaced, clustering where weights are statistically most common
- Selected from a mathematically optimal lattice that minimizes expected quantization error
- Pre-computed and stored as static lookup tables -- no per-model optimization needed

The E8 lattice achieves the densest sphere packing in 8 dimensions, meaning its Voronoi cells (regions of points closest to each lattice point) have minimal average distance to a random point. This makes it near-optimal for vector quantization.

#### 6.3.2 Super-Block Scaling (Scales Have Their Own Scales)

Like K-quants, IQ types use hierarchical scaling:
- A global FP16 scale per super-block (256 weights)
- Optional sub-block scale adjustments (4-bit in IQ2_XS, 3-bit in IQ1_M)
- The sub-block scales allow local adaptation to weight magnitude variations within the super-block

#### 6.3.3 Importance-Aware Bit Allocation

The imatrix transforms quantization from a uniform operation to an importance-weighted one:
- Dimensions with high activation variance receive more careful codebook assignment
- The quantizer's error metric is importance-weighted, ensuring critical dimensions are preserved
- This can be thought of as a soft form of mixed-precision: while all dimensions get the same number of bits, the codebook selection process prioritizes accuracy for important dimensions

#### 6.3.4 Lattice-Based Value Selection (Vector Quantization)

The fundamental shift from scalar to vector quantization:
- Q2_K quantizes each weight independently: 1 weight -> 1 of 4 values
- IQ2_XS quantizes 8 weights jointly: 8 weights -> 1 of 512 E8 lattice points (each point is an 8-dimensional vector)
- The 512-entry codebook can represent 512 distinct 8-dimensional patterns
- This exploits correlations between adjacent weights for more efficient encoding
- The number of representable configurations per group: 512 (IQ) vs 4^8 = 65536 (Q2_K), but IQ's 512 configurations are optimally placed in 8D space while Q2_K's 65536 are uniformly distributed on a rectilinear grid -- most of those grid points are far from any realistic weight vector

### 6.4 The Role of Calibration Data

Calibration data serves two purposes in IQ quantization:

1. **imatrix computation**: Identifies which weight dimensions most strongly affect model output, enabling importance-weighted quantization
2. **Implicit distribution learning**: The imatrix captures the statistical properties of real activations, so the quantizer can focus precision where it matters

Quality of calibration data directly impacts quantized model quality:
- Domain-matched calibration (e.g., code-heavy text for code models) produces the best results for that domain
- General calibration (Wikipedia, web text) produces broadly good results
- Random/garbage calibration can actively hurt quality by corrupting attention patterns

### 6.5 Perplexity Measurements Across Quantization Types

Perplexity on WikiText-2 for LLaMA 2 7B (lower is better, FP16 baseline = 5.8):

| Type | bpw | Perplexity | Delta |
|------|-----|-----------|-------|
| F16 | 16.0 | 5.80 | -- |
| Q8_0 | 8.5 | 5.80 | +0.00 |
| Q6_K | 6.56 | 5.80 | +0.00 |
| Q5_K_M | 5.70 | 5.81 | +0.01 |
| Q5_K_S | 5.57 | 5.84 | +0.04 |
| Q4_K_M | 4.89 | 5.85 | +0.05 |
| Q4_K_S | 4.67 | 5.90 | +0.10 |
| Q3_K_L | 3.89 | 5.98 | +0.18 |
| Q3_K_M | 3.44 | 6.04 | +0.24 |
| Q3_K_S | 3.64 | 6.36 | +0.56 |
| Q2_K | 2.63 | 6.47 | +0.67 |
| IQ2_M | ~2.7 | ~6.10 | ~+0.30 |
| IQ2_XS | 2.31 | ~6.30 | ~+0.50 |

Perplexity on WikiText-2 for Llama 3.1 8B (FP16 baseline = 7.32):

| Type | bpw | Perplexity |
|------|-----|-----------|
| F16 | 16.0 | 7.32 |
| Q8_0 | 8.50 | 7.33 |
| Q6_K | 6.56 | 7.35 |
| Q5_K_M | 5.70 | 7.40 |
| Q5_K_S | 5.57 | 7.43 |
| Q4_K_M | 4.89 | 7.56 |
| Q4_K_S | 4.67 | 7.62 |
| Q4_0 | 4.50 | 7.74 |
| Q3_K_M | 4.00 | 7.96 |
| Q3_K_S | 3.64 | 8.96 |

Key observation: Q4_K_M vs Q4_0 -- at similar bit widths (4.89 vs 4.50 bpw), Q4_K_M achieves 7.56 vs 7.74 perplexity. The K-quant's hierarchical scaling and mixed precision provide a measurable advantage even at 4-bit.

### 6.6 Task-Specific Quantization Impact

A 2025/2026 systematic study on Llama 3.1-8B-Instruct found that quantization impact is task-dependent:

| Task | Most Sensitive? | Q5_K_M Score | Q3_K_S Score | FP16 Score |
|------|----------------|-------------|-------------|------------|
| GSM8K (math) | Yes | ~68% | ~58% | ~69% |
| HellaSwag (common sense) | No | ~82% | ~81% | ~82% |
| MMLU (knowledge) | Moderate | ~68% | ~64% | ~68% |
| TruthfulQA | Moderate | ~56% | ~53% | ~55% |
| IFEval (instruction) | Low | ~75% | ~73% | ~75% |

Mathematical reasoning (GSM8K) is most sensitive to quantization because precise numerical relationships encoded in weights are easily corrupted. Common-sense reasoning (HellaSwag) is robust because the required knowledge is broadly distributed across many weights.

---

## 7. Architecture of llama.cpp Inference

### 7.1 The Computational Graph (ggml_cgraph)

All computation in llama.cpp is expressed as a directed acyclic graph (DAG) of tensor operations. The `ggml_cgraph` structure represents this graph:

```c
struct ggml_cgraph {
    int n_nodes;           // number of operation nodes
    int n_leafs;           // number of leaf tensors (inputs/weights)
    struct ggml_tensor ** nodes;   // operation nodes in topological order
    struct ggml_tensor ** grads;   // gradient tensors (for training)
    struct ggml_tensor ** leafs;   // input/weight tensors
};
```

**Graph construction** is lazy -- when you call operations like `ggml_mul_mat()`, they do not execute immediately. Instead, they record the operation in the graph as a new node with references to input tensors. The full graph is built during the "build" phase and executed during the "compute" phase.

**Architecture-specific builders**: Each model architecture has a dedicated graph builder:

```c
// Simplified architecture dispatch:
switch (model.arch) {
    case LLM_ARCH_LLAMA:   build_llama(ctx, batch);   break;
    case LLM_ARCH_QWEN2:   build_qwen2(ctx, batch);   break;
    case LLM_ARCH_PHI3:    build_phi3(ctx, batch);     break;
    case LLM_ARCH_GEMMA:   build_gemma(ctx, batch);    break;
    // ... 80+ architectures supported
}
```

Each builder constructs the specific sequence of operations for that architecture:
- Embedding lookup
- For each layer: attention (Q/K/V projections, RoPE, attention computation, output projection) + FFN (gate, up, down projections with activation function) + residual connections + normalization
- Final norm + output projection

### 7.2 Operation Scheduling on Metal

The backend scheduler (`ggml_backend_sched`) dispatches graph nodes to available hardware backends:

```
1. Walk graph in topological order
2. For each node, determine best backend:
   - Matrix multiplication -> GPU (Metal/CUDA)
   - Element-wise operations -> GPU
   - Control flow, special ops -> CPU
3. Insert copy operations where data crosses backend boundaries
4. Build per-backend subgraphs
5. Execute subgraphs on respective backends
```

**Multi-backend coordination**: In GPU-offloaded configurations (the `-ngl` flag), some layers run on GPU and others on CPU. The scheduler:
- Tracks tensor locations (which backend "owns" each tensor)
- Automatically inserts data transfer operations between CPU and GPU
- Can overlap data transfer with computation when the hardware supports it

**Metal-specific scheduling**:
- Multiple command buffers are used for parallel encoding across CPU threads
- Operation fusion combines compatible kernels to reduce launch overhead
- The backend maintains a "dirty bit" to detect when the compute graph changes (which is rare during steady-state generation)

### 7.3 Memory Management for Large Models

llama.cpp manages several distinct memory regions:

**Model weights** (`llama_model`):
- Loaded via mmap from the GGUF file
- For GPU-offloaded layers, weights are copied to Metal private buffers
- The `-ngl N` flag controls how many layers are offloaded to GPU
- When the model exceeds GPU memory, remaining layers stay on CPU (accessed via mmap)

**Context memory** (`llama_context`):
- Allocated from GGML memory arenas (contiguous allocations for efficiency)
- Contains scratch buffers for intermediate activations
- Sized based on context length and batch size

**KV cache**:
- Pre-allocated at context creation time
- Size = `n_layers * (n_kv * n_embd_k_gqa + n_kv * n_embd_v_gqa) * type_size`
- For a 7B model with 4096 context in FP16: ~2 GB
- With Q8_0 KV cache: ~1 GB
- With Q4_0 KV cache: ~0.5 GB

### 7.4 KV Cache Implementation

The KV cache is the primary stateful component during inference, storing attention key and value tensors from all previously processed tokens.

**Ring buffer architecture**:

```
Layer 0 K: [token_0_k | token_1_k | token_2_k | ... | token_N_k | empty ... ]
Layer 0 V: [token_0_v | token_1_v | token_2_v | ... | token_N_v | empty ... ]
Layer 1 K: [token_0_k | token_1_k | ...]
Layer 1 V: [token_0_v | token_1_v | ...]
...
```

Each layer maintains separate K and V tensors with shapes:
- K: `[n_embd_k_gqa, kv_size, n_stream]`
- V: `[n_embd_v_gqa, kv_size, n_stream]` (transposed for flash attention) or `[kv_size, n_embd_v_gqa, n_stream]` (non-transposed)

**Cell tracking**:
- Each KV cache cell represents a single token position
- Cells track which sequences occupy them using bitsets
- This enables prefix sharing: if two conversations share a common system prompt, the KV cache entries for the shared prefix are shared (unified mode)

**Sequence operations**:
- `seq_rm(seq_id, pos_start, pos_end)` -- Remove tokens in a position range; cells become empty when all sequences depart
- `seq_cp(src_seq, dst_seq)` -- Copy sequence metadata (unified mode) or actual tensor data (multi-stream mode)
- `seq_add(seq_id, delta)` -- Shift positions forward/backward for RoPE adjustments

**Slot allocation**: When processing a new batch, the cache allocator finds contiguous (or scattered) available cells for new token positions. The allocation algorithm handles both cases, falling back to scattered allocation with `ggml_set_rows()` when contiguous space is unavailable.

**Sliding window attention (SWA)**: For models like Mistral that use windowed attention, cells outside the attention window can be reused. A cell is recyclable when it contains exactly one sequence and its position falls outside the current window.

**KV cache quantization**: The cache type can be set independently for K and V:

```bash
llama-server --cache-type-k q8_0 --cache-type-v q8_0   # ~2x memory savings
llama-server --cache-type-k q4_0 --cache-type-v q4_0   # ~4x savings, quality risk
```

Q8_0 KV cache is generally safe with minimal quality impact. Q4_0 can cause visible degradation, especially with smaller models or aggressive quantization on the weights themselves.

### 7.5 Batch Processing

**Batch structure**: A batch contains multiple token positions to process in parallel:

```c
struct llama_batch {
    int32_t n_tokens;       // number of tokens in this batch
    llama_token * token;    // token IDs
    float ** embd;          // or pre-computed embeddings
    llama_pos * pos;        // position of each token
    int32_t * n_seq_id;     // number of sequences per token
    llama_seq_id ** seq_id; // sequence IDs
    int8_t * logits;        // which tokens need logit output
};
```

**Processing pipeline**:

1. **init_batch()**: Splits large batches into micro-batches (ubatches) that fit in available memory. Determines slot requirements.

2. **Slot allocation**: Finds KV cache cells for each token position. Pending memory operations (shifts, copies) are queued.

3. **Graph building**: Constructs the compute graph referencing allocated KV cache slots. Input tensors include:
   - `self_k_idxs`: Cell indices for writing K vectors `[n_tokens]`
   - `self_v_idxs`: V write indices
   - `self_kq_mask`: Attention mask `[n_kv, n_tokens]` combining causal, SWA, and sequence masking
   - `self_k_shift`: RoPE shift amounts from accumulated position shifts

4. **Execution**: `ggml_backend_sched_graph_compute()` executes the graph.

5. **Cache update**: New K/V vectors are written to allocated slots. For contiguous slots, `ggml_set_1d()` is used; for non-contiguous, `ggml_set_rows()`.

6. **Cleanup**: Expired sequence data is removed.

### 7.6 Token Generation Loop

The high-level inference loop:

```
Input: prompt text

1. TOKENIZE
   text --> token_ids[]  (using model's tokenizer from GGUF metadata)

2. PROMPT PROCESSING (prefill)
   batch = { token_ids, positions[0..N-1], seq_id=0 }
   llama_decode(batch)
   --> Process all prompt tokens in parallel (batch size = prompt length)
   --> Fills KV cache with prompt representations
   --> Returns logits for last token only

3. TOKEN GENERATION (auto-regressive)
   loop:
     a. SAMPLE next token from logits:
        - Apply temperature scaling
        - Apply top-k filtering
        - Apply top-p (nucleus) filtering
        - Apply grammar constraints (if enabled)
        - Sample from resulting distribution
        --> next_token_id

     b. CHECK stopping conditions:
        - Is next_token_id == EOS token?
        - Have we reached max_tokens?
        - Does grammar reject this token?
        If yes: break

     c. DECODE next token:
        batch = { [next_token_id], position=[pos++], seq_id=0 }
        llama_decode(batch)
        --> Process single token (batch size = 1)
        --> Updates KV cache with new token
        --> Returns logits for next prediction

     d. OUTPUT token to user (streaming)

4. Return generated text
```

**Performance characteristics**:
- Prompt processing: compute-bound, benefits from large batch sizes (matrix-matrix multiplication)
- Token generation: memory-bandwidth-bound, batch size 1 (matrix-vector multiplication)
- The transition from prefill to generation is the key performance boundary

### 7.7 Unified vs Multi-Stream Modes

**Unified mode** (`n_stream = 1`):
- All sequences map to stream 0
- KV cache cells are shared across sequences via bitsets
- Enables prefix sharing (e.g., shared system prompts)
- Standard for single-user or shared-prefix scenarios
- Most memory-efficient

**Multi-stream mode** (`n_stream = n_seq_max`):
- Each sequence has a dedicated stream with isolated KV cache
- No cross-sequence sharing
- Enables parallel inference with full isolation
- Higher memory usage but simpler management

### 7.8 Memory Optimization: Position Shifts

When sequences are shifted (e.g., for sliding window or context extension), positions are updated without recomputing attention:

```
1. Accumulate shift amounts per cell
2. Store shifts in self_k_shift tensor
3. During graph execution, apply RoPE correction:
   k_shifted = rotate(k_original, shift_amount)
```

Only cells with non-zero shifts are processed, avoiding full cache recomputation. This is critical for long-context inference where maintaining the full KV cache would exceed memory.

---

## Appendix A: Quantization Selection Guide

| Use Case | Recommended Type | bpw | Notes |
|----------|-----------------|-----|-------|
| Maximum quality | Q6_K | 6.56 | Essentially lossless |
| Quality priority | Q5_K_M | 5.70 | Near-transparent loss |
| Balanced (default) | Q4_K_M | 4.89 | Best quality/size ratio |
| Memory-constrained | Q4_K_S | 4.67 | Slight quality reduction |
| Aggressive compression | Q3_K_M | 4.00 | Noticeable quality loss |
| Extreme compression (with imatrix) | IQ2_M | ~2.7 | Requires calibration data |
| Maximum compression (large models) | IQ1_M | 1.75 | Only viable for 70B+ models |
| KV cache reduction | Q8_0 cache | -- | Safe default for KV cache |
| Near-lossless | Q8_0 | 8.50 | +0.0004 ppl; 2x smaller than FP16 |

## Appendix B: Bits Per Weight Calculation Reference

General formula for K-quants:

```
bpw = (value_bits * QK_K + scale_bits_total + superblock_scale_bits) / QK_K
```

Worked example for Q2_K:
```
value_bits      = 256 weights * 2 bits = 512 bits
scale_bits      = 16 sub-blocks * 4-bit scale = 64 bits
min_bits        = 16 sub-blocks * 4-bit min   = 64 bits
superblock_d    = 16 bits (FP16)
superblock_dmin = 16 bits (FP16)
                ─────────────────
total           = 512 + 64 + 64 + 16 + 16 = 672 bits
bpw             = 672 / 256 = 2.625
```

Worked example for Q4_K:
```
value_bits      = 256 weights * 4 bits = 1024 bits
scale_bits      = 8 sub-blocks * 6-bit scale = 48 bits
min_bits        = 8 sub-blocks * 6-bit min   = 48 bits
superblock_d    = 16 bits
superblock_dmin = 16 bits
                ─────────────────
total           = 1024 + 48 + 48 + 16 + 16 = 1152 bits
bpw             = 1152 / 256 = 4.5
```

## Appendix C: Key Source File Map

| File | Purpose |
|------|---------|
| `ggml/src/ggml-common.h` | Block structure definitions for all quant types |
| `ggml/src/ggml-quants.c` | Quantize/dequantize implementations (CPU) |
| `ggml/src/ggml-quants.h` | Quantization function declarations |
| `ggml/include/ggml.h` | Core type definitions, ggml_type enum |
| `ggml/src/ggml-metal/ggml-metal.m` | Metal backend host code |
| `ggml/src/ggml-metal/ggml-metal.metal` | Metal shader kernels |
| `src/llama-quant.cpp` | Model-level quantization logic, mixed precision |
| `src/llama-model-loader.cpp` | GGUF file parsing and model loading |
| `src/llama-kv-cache.cpp` | KV cache implementation |
| `src/llama-kv-cells.h` | KV cache cell management |
| `src/llama-context.cpp` | Inference context management |
| `src/llama-batch.cpp` | Batch processing pipeline |
| `tools/quantize/` | Quantization CLI tool |
| `tools/imatrix/` | Importance matrix computation tool |
| `docs/gguf.md` | GGUF format specification |
