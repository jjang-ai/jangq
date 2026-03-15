# Experiment 019: llama.cpp Metal Backend Analysis

**Date**: 2026-03-14
**Author**: Eric Jang (eric@vmlx.net)
**Status**: REFERENCE DOCUMENT

## Purpose

Detailed analysis of how llama.cpp implements its Metal-based forward pass for
transformer inference. This document serves as a reference for MXQ's own Metal
inference pipeline, identifying patterns to adopt and pitfalls to avoid.

---

## 1. Metal Kernel Dispatch Pattern

### Command Buffer Architecture

llama.cpp uses a configurable number of command buffers controlled by `n_cb`
(environment variable `GGML_METAL_N_CB`). The default is 1 on integrated GPUs
(Apple Silicon) and 2 on discrete GPUs.

The core graph compute function is `ggml_metal_graph_compute()`, which:

1. Walks every node in the `ggml_cgraph` DAG sequentially
2. For each node, calls the operation encoder (`ggml_metal_encode_node`) to
   encode it into a Metal compute command encoder
3. Tracks memory read/write ranges for each operation
4. When a memory conflict is detected (current node writes to a range previously
   read or written by an earlier node in the same encoder), inserts a
   `memoryBarrier` via `ggml_metal_encoder_memory_barrier()`
5. After all nodes are encoded, commits the command buffer and calls
   `waitUntilCompleted`

**Key insight**: llama.cpp encodes the **entire forward pass** (all layers, all
operations) into a single command buffer (or a small number of command buffers
when `n_cb > 1`). It does NOT use one command buffer per layer or per operation.
This minimizes CPU-GPU synchronization overhead.

### Concurrent Operation Execution

The Metal backend analyzes memory dependencies between operations to determine
which can execute concurrently on the GPU. The memory range tracking system:

- Records the source and destination buffer ranges for each encoded operation
- When the next operation's write range overlaps any previous operation's
  read or write range, a memory barrier is inserted
- When there is no overlap, operations can execute concurrently on the GPU
  (Metal's command encoder allows this automatically within a single encoder)

### Operation Fusion

The backend supports fusing compatible operations to reduce kernel dispatch
overhead. Compatible operations that share the same pipeline state and have
compatible memory access patterns can be encoded together, reducing the total
number of dispatch calls.

### Pipeline State Caching

Metal compute pipeline states are compiled from shader functions with
function constants for operation-specific specialization. Pipelines are cached
in a hash map keyed by pipeline name (which encodes operation parameters, e.g.,
`kernel_unary_f32_f32_op=100_cnt=1` for TANH). This avoids expensive
recompilation for repeated operation types.

**Key source files**:
- `ggml/src/ggml-metal/ggml-metal.m` (graph compute, context, command buffer management)
- `ggml/src/ggml-metal/ggml-metal-ops.cpp` (operation encoding, lines 113-480)
- `ggml/src/ggml-metal/ggml-metal-impl.h` (constants, structures, pipeline configs)

---

## 2. Quantized MatMul Dispatch

### Two Kernel Families

llama.cpp provides two distinct matrix multiplication kernel families on Metal:

| Kernel | Use Case | When Selected |
|--------|----------|---------------|
| `kernel_mul_mv_*` | Matrix-vector multiply | ne11 == 1 (single-token decode) |
| `kernel_mul_mm_*` | Matrix-matrix multiply | ne11 > 1 (prefill / batched) |

The selection is based on `ne11` (the second dimension of the second operand),
which corresponds to the number of tokens being processed simultaneously.

### kernel_mul_mv (Matrix-Vector)

Used during autoregressive token generation (the common case for inference).

**Naming convention**: `kernel_mul_mv_{src_type}_{dst_type}` e.g.,
`kernel_mul_mv_q4_K_f32` processes Q4_K quantized weights with f32 output.

**Thread dispatch**:
```
dispatchThreadgroups: MTLSizeMake(ne01, ny, ne12*ne13)
threadsPerThreadgroup: MTLSizeMake(nth0, nth1, 1)
```

Where:
- `ne01` = number of output rows (output dimension of the weight matrix)
- `ny` = number of row groups processed per threadgroup
- `ne12*ne13` = batch/head dimensions
- `nth0, nth1` = threads per threadgroup (typically 32 for SIMD width)

**SIMD group configuration** (from `ggml-metal-impl.h`):

| Quant Type | N_R0 (rows/simdgroup) | N_SG (simdgroups/threadgroup) |
|------------|----------------------|------------------------------|
| Q4_0, Q4_1, Q5_0, Q5_1 | 4 | 2 |
| Q8_0 | 2 | 4 |
| Q2_K, Q3_K, Q4_K, Q6_K | 2 | 2 |
| Q5_K | 1 | 2 |
| IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S | 4 | 2 |
| IQ4_NL, IQ4_XS | 2 | 2 |
| MXFP4 | 2 | 2 |

Each SIMD group processes N_R0 rows of the weight matrix. Each threadgroup
contains N_SG SIMD groups. So a single threadgroup processes N_R0 * N_SG rows.

**Dequantization**: Performed on-the-fly inside the kernel. The kernel reads
compact quantized blocks, extracts scale/zero/quant values, reconstructs
float16 or float32 values, and immediately multiplies with the input vector.
Results are accumulated using `simd_sum` for SIMD-level reduction.

### kernel_mul_mm (Matrix-Matrix)

Used during prompt processing (prefill) when multiple tokens are processed
simultaneously.

Uses a tiled approach with threadgroup shared memory:
- Loads tiles of both input and weight matrices into threadgroup memory
- Performs multiply-accumulate within the tile
- Uses SIMD-level operations for efficient reduction
- Supports all the same quant types as mul_mv

### On-the-Fly Dequantization Pattern

For K-quants (Q2_K through Q6_K), the dequantization follows a hierarchical
super-block structure:

```
Super-block (QK_K = 256 values):
  - 1x FP16 super-scale (d)     -- scales the block-level scales
  - 1x FP16 super-minimum (dmin) -- scales the block-level minimums
  - 8x blocks of 32 values each:
      - quantized scale (6-bit)
      - quantized minimum (6-bit)
      - 32 quantized weight values (N bits each)

Dequantization formula:
  weight = d * block_scale * quant_value - dmin * block_minimum
```

This double-quantization technique (quantizing the scales themselves) saves
storage while maintaining good accuracy.

---

## 3. Attention Implementation in Metal

### Flash Attention (Fused Kernel)

llama.cpp implements Flash Attention as the `GGML_OP_FLASH_ATTN_EXT` operation,
introduced in PR #5021 (merged April 2024). This replaces the unfused approach
of separate Q*K^T, softmax, and *V operations.

**API**:
```c
res = ggml_flash_attn_ext(ctx, q, k, v, kq_mask, kq_scale, kq_log2e, alibi_m);
```

### Two Metal Kernel Variants

| Kernel | Use Case | Configuration |
|--------|----------|---------------|
| `kernel_flash_attn_ext` | Batched (prefill) | 8 queries per simdgroup, 64 keys per simdgroup |
| `kernel_flash_attn_ext_vec` | Single query (decode) | 1 query per simdgroup, 32 keys per simdgroup |

Selection: When Q sequence length <= 1, the vector variant is used.

### Tiling Strategy

The Flash Attention kernel uses the classic IO-aware tiling approach:

- **Br (query tile)**: 16-32 elements
- **Bc (KV tile)**: 32-64 elements
- **d_head**: 64-128 (determined by model architecture)

Critical optimization: The attention score matrix S = Q * K^T (shape Br x Bc)
**never needs to be written to global memory**. It stays in registers/threadgroup
memory, is scaled, masked, softmaxed, and immediately multiplied with V, all
within the kernel.

### Key Optimizations Applied

1. **Results stay in registers**: Intermediate attention scores remain in SIMD
   registers, significantly reducing SRAM usage
2. **Efficient -INF skipping**: Masked positions (future tokens) are skipped
   efficiently rather than computing through them
3. **No simdgroup barrier in hot loop**: The inner loop avoids expensive
   synchronization barriers
4. **Parallel KV reduction**: Parallelizes across KV sequence dimension
5. **Parallel head reduction**: Parallelizes across attention heads
6. **Split-K variant**: For very long sequences, divides the KV sequence
   across multiple workgroups for higher parallelism

### Accumulator Precision

Two precision options:
- **FP32 accumulation**: Full precision, wider dynamic range (default)
- **FP16 accumulation**: 50% less register pressure, higher occupancy, but
  reduced precision

### KV Cache Interaction

**Storage**: The KV cache stores K and V tensors in contiguous buffers per
layer. V is stored **non-transposed** (this changed with Flash Attention -- the
old unfused attention stored V transposed).

**Padding**: KV cache `n_kv` is padded to multiples of 128 (increased from 32)
to support larger Flash Attention tile sizes.

**Quantized KV cache**: Flash Attention supports on-the-fly dequantization of
quantized KV cache values (q4_0, q5_0, q8_0, and k-quant formats). The
dequantization functions are included via preprocessor conditionals. However,
there is a known performance issue: quantized KV cache with Flash Attention can
be ~33% slower than unquantized, as both prompt processing and token generation
degrade.

### KQ Mask

The attention mask is stored as F16 and padded to 32 elements. Input buffers
are pre-cleared with zeros to avoid NaNs in padding regions. ALiBi positional
biases are applied as mask additions rather than having kernel-specific handling.

### Minimum Batch Size

Minimum `n_batch` is 32 (prevents out-of-bounds kernel access in Flash
Attention tiles). This is a constraint that affects how tokens are batched.

---

## 4. Forward Pass Structure

### GGML Computation Graph System

llama.cpp's forward pass is built on the ggml library's computation graph
(DAG) model:

1. **Graph Construction**: The model builder creates a `ggml_cgraph` DAG
   defining all operations. Operations are nodes; tensor dependencies are edges.
   The graph is built once and can be reused.

2. **Scheduling**: `ggml_backend_sched_graph_compute()` receives the graph.
   For multi-backend setups (GPU + CPU), the scheduler splits the graph into
   sub-graphs, assigning each operation to the appropriate backend. Operations
   not supported on GPU fall back to CPU automatically.

3. **Graph Allocation**: `ggml_gallocr` (the graph allocator) handles
   intermediate buffer allocation with lifetime analysis -- it determines when
   intermediate results can be freed and reuses memory accordingly.

4. **Execution**: The Metal backend's `ggml_metal_graph_compute()` walks every
   node in topological order and encodes each into the Metal command encoder.
   After all nodes are encoded, the command buffer is committed and the CPU
   waits for GPU completion.

### Intermediate Buffer Management

**ggml's approach** (what llama.cpp does):
- Tensors have metadata (shape, type, strides) separate from data
- Data allocation is deferred to backends
- The graph allocator performs lifetime analysis on intermediate tensors
- Intermediate buffers are automatically reused when their values are no longer
  needed by any downstream operation
- This is transparent to the operation encoding -- each operation just receives
  pointers to its input and output buffers

**MXQ's current approach** (from MXQInference.swift):
- Pre-allocated named buffers (hiddenBuffer, normBuffer, qBuffer, etc.)
- Manual buffer reuse (normBuffer is reused as temp for O projection output)
- No automatic lifetime analysis
- Simple but works for the current fixed architecture

### Memory Modes

llama.cpp's Metal backend uses two memory modes:
- **StorageModeShared**: CPU-accessible, used for smaller tensors and
  intermediate results
- **StorageModePrivate**: GPU-only, used for large weight tensors (faster
  GPU access)

On macOS 15+, `MTLResidencySet` keeps GPU memory resident to prevent OS paging.
A background thread maintains residency every 500ms (configurable via
`GGML_METAL_RESIDENCY_KEEP_ALIVE_S`).

---

## 5. Importance Matrix (imatrix) Computation

### Overview

The `llama-imatrix` tool computes per-channel importance statistics for each
weight tensor by running calibration data through the model and observing
activation patterns. These statistics guide quantization decisions.

### Mathematical Foundation

The importance matrix uses diagonal elements of the activation expectation:

```
imatrix[j] = E[a_j^2] = (1/N) * sum(a_j^2)  over N calibration tokens
```

For a tensor with shape (N_out x N_in), the imatrix has N_in entries.
Each entry is a weight in a weighted RMSE minimization across N_out model
weights during quantization.

### Data Collection Mechanism

**Callback hook**: The ImatrixCollector registers a callback on `GGML_OP_MUL_MAT`
operations. During forward passes on calibration data, whenever a matrix
multiplication executes, the callback:

1. Identifies the weight tensor by name (e.g., `blk.0.ffn_down.weight`)
2. Reads the activation tensor (src1) from the backend
3. Accumulates squared activation values per channel:
   ```c
   e.values[j] += x[j] * x[j];
   e.counts[j]++;
   ```
4. Only triggers when batch size >= 16 tokens (ignores tiny batches)
5. Handles both standard (`MUL_MAT`) and mixture-of-experts (`MUL_MAT_ID`)

**Filtering**: Data is collected only for weight matrices matching patterns like
`blk.*.weight`. The output.weight tensor is excluded by default (controlled by
`-ow` flag) because imatrix-guided quantization of the output projection
typically hurts quality.

### Statistics Stored

The saved imatrix file (GGUF format) contains per weight tensor:
- Sum of squared activations per channel: `sum(a_j^2)`
- Count of observations per channel
- Number of forward pass calls

### How imatrix Guides Quantization

During quantization, the importance weights modify the block-wise RMSE
minimization:

```c
weight[j] = qw[j] * sqrt(sigma2 + imatrix[j])
```

Where:
- `qw[j]` = base quantization weight
- `sigma2` = mean of all squared activation values in the vector (global variance)
- `imatrix[j]` = per-channel squared activation average

This means channels with higher activation magnitudes get higher weight in the
quantization optimization, causing the quantizer to preserve those channels more
precisely. The `sigma2` term acts as regularization, preventing zero weights
and smoothing extreme values.

### Calibration Data Considerations

Empirical findings from the llama.cpp community:
- **Context length**: 128-256 tokens generalizes better than full context
- **Data diversity**: Mixed data (wiki, code, conversation) outperforms
  domain-specific data
- **Near-random data**: Surprisingly effective for out-of-domain evaluation
- **Token volume**: 8,000-50,000 tokens typical for 7B models
- **Compute cost**: ~10 min for 50K tokens on a 32-core CPU for 7B models

---

## 6. Key Patterns MXQ Should Adopt

### Pattern 1: Single Command Buffer for Entire Forward Pass

**What llama.cpp does**: Encodes ALL operations (all layers, all ops) into one
command buffer, then commits and waits once.

**What MXQ currently does**: Creates a single command buffer per `forward()`
call, which is correct. However, MXQ should verify it is not inadvertently
creating multiple command buffers or calling `waitUntilCompleted` between layers.

**Why it matters**: Each `commit` + `waitUntilCompleted` cycle has significant
CPU-GPU synchronization overhead. Encoding everything into one command buffer
allows the GPU to pipeline operations continuously.

### Pattern 2: Memory Barrier Instead of Separate Encoders

**What llama.cpp does**: Uses `memoryBarrier()` within a single compute
command encoder when operations have memory dependencies. Does NOT create a
new encoder for each operation.

**What MXQ should do**: Within the forward pass, when buffer reuse creates
read-after-write hazards (e.g., normBuffer reused for O projection output),
insert `memoryBarrier(scope: .buffers)` rather than ending and beginning a
new encoder.

### Pattern 3: Distinct MatVec vs MatMat Kernels

**What llama.cpp does**: Separate `kernel_mul_mv` (decode) and `kernel_mul_mm`
(prefill) kernels with different tiling strategies and thread dispatch configs.

**What MXQ has**: `mxq_dequant_gemv` and `mxq_dequant_gemm` -- this is already
the right split. But MXQ should ensure the GEMM kernel uses tiled shared-memory
accumulation (as llama.cpp's `kernel_mul_mm` does) rather than naive per-thread
accumulation.

### Pattern 4: SIMD Group Configuration Per Quant Type

**What llama.cpp does**: Different N_R0 (rows per simdgroup) and N_SG
(simdgroups per threadgroup) for each quant type, tuned for optimal occupancy.

**What MXQ should do**: Since MXQ supports variable bit widths (2-8 bit),
the GEMV kernel should tune its thread dispatch based on the bit width.
Lower bit widths (2-3 bit) can process more rows per simdgroup because
each weight uses fewer bytes. Higher bit widths (6-8 bit) should process
fewer rows per simdgroup. Suggested starting configuration:

| MXQ Bit Width | N_R0 | N_SG | Rationale |
|---------------|------|------|-----------|
| 2-3 bit | 4 | 2 | Small weights, more rows fit in registers |
| 4 bit | 4 | 2 | Matches llama.cpp Q4_0 |
| 5-6 bit | 2 | 2 | Matches llama.cpp Q5_K/Q6_K |
| 8 bit | 2 | 4 | Matches llama.cpp Q8_0 |

### Pattern 5: Flash Attention with Tiling

**What llama.cpp does**: Fused Flash Attention kernel that keeps attention
scores in registers, never writing the N*N attention matrix to memory.

**What MXQ currently does**: Based on experiment 014, the attention kernel
is a primary suspect for correctness issues. MXQ should implement a tiled
Flash Attention kernel similar to llama.cpp's approach:

1. Load a tile of Q (Br rows)
2. For each KV tile (Bc columns):
   a. Load K tile, compute S = Q * K^T (in registers)
   b. Apply scale (1/sqrt(head_dim)) and mask
   c. Compute local softmax (online softmax algorithm)
   d. Load V tile, accumulate O += softmax(S) * V
3. Write final O tile to output

Critical details:
- Use the vector variant for single-token decode (Q length == 1)
- Pad KV cache to multiples of 32 or 64 for tile alignment
- Pre-clear input buffers to avoid NaN from padding
- Store V non-transposed in the KV cache

### Pattern 6: Hierarchical Block Quantization Structure

**What llama.cpp does**: K-quants use super-blocks of 256 values with
double-quantization (scales are themselves quantized), giving better
compression without proportional accuracy loss.

**What MXQ should consider**: MXQ's per-block (64 values) scale/zero is
simpler. If quality issues arise at low bit widths, consider a hierarchical
approach where blocks share a higher-precision super-scale, similar to K-quants.

### Pattern 7: Importance Matrix for Quality Optimization

**What llama.cpp does**: Collects squared activation statistics during
calibration forward passes, then uses these to weight the quantization
optimization per channel.

**What MXQ should do**: Implement an imatrix collection pass:
1. Run the FP16 model on calibration data
2. At each linear layer, accumulate `sum(activation[j]^2)` per input channel
3. During quantization, use these weights to prioritize high-activation channels
   when computing optimal scales/zeros for each block

This is especially important for MXQ because variable-bit-width assignment
could use the imatrix to determine which layers/blocks get more bits:
- High-importance blocks (large sum of squared activations) get more bits
- Low-importance blocks get fewer bits
- This is the core value proposition of mixed-precision quantization

### Pattern 8: StorageModePrivate for Weight Buffers

**What llama.cpp does**: Uses `MTLResourceStorageModePrivate` for large
weight tensors (GPU-only, no CPU access needed after initial upload).

**What MXQ currently does**: Uses `MTLResourceStorageModeShared` for
everything (from MXQMetalDevice.swift).

**What MXQ should do**: After loading quantized weights via mmap, copy them
to Private-mode buffers for GPU inference. This gives the Metal driver more
freedom to optimize memory placement and can improve bandwidth.

### Pattern 9: Residency Set Management

**What llama.cpp does**: On macOS 15+, uses `MTLResidencySet` to prevent
the OS from paging GPU memory during inference.

**What MXQ should do**: For large models that consume most of GPU memory,
adopt the residency set pattern to prevent memory pressure from evicting
weight buffers mid-inference.

---

## Source Files in llama.cpp

| File | Purpose |
|------|---------|
| `ggml/src/ggml-metal/ggml-metal.m` | Metal context, command buffer, graph compute |
| `ggml/src/ggml-metal/ggml-metal-ops.cpp` | Operation encoding (encode_node) |
| `ggml/src/ggml-metal/ggml-metal-impl.h` | Constants, threadgroup configs, structures |
| `ggml/src/ggml-metal/ggml-metal.metal` | All Metal shader kernels |
| `ggml/src/ggml-metal/ggml-metal-device.cpp` | Device detection, GPU family |
| `tools/imatrix/imatrix.cpp` | ImatrixCollector, calibration data collection |
| `tools/quantize/` | Quantization engine using imatrix data |

---

## References

- [llama.cpp Metal Backend - DeepWiki](https://deepwiki.com/ggml-org/llama.cpp/5.2-http-server)
- [Flash Attention PR #5021](https://github.com/ggml-org/llama.cpp/pull/5021)
- [Flash Attention Optimizations - DeepWiki](https://deepwiki.com/ggml-org/llama.cpp/7.4-flash-attention-and-optimizations)
- [Metal Matrix Multiplication Discussion #5197](https://github.com/ggml-org/llama.cpp/issues/5197)
- [Metal Kernel Thread Dispatch Issue #6089](https://github.com/ggml-org/llama.cpp/issues/6089)
- [Importance Matrix Calculations Discussion #5006](https://github.com/ggml-org/llama.cpp/discussions/5006)
- [imatrix Overfitting Discussion #5263](https://github.com/ggml-org/llama.cpp/discussions/5263)
- [imatrix Tool README](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md)
- [Quantization README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
- [Introduction to ggml - HuggingFace](https://huggingface.co/blog/introduction-to-ggml)
- [llama.cpp GitHub Repository](https://github.com/ggml-org/llama.cpp)
