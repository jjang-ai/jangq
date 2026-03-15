# MLXQ — Mixed-Precision Importance Quantization for Apple Silicon

> The GGUF of Apple Silicon. Open format, open runtime, anyone can use it.
> Like GGUF is to llama.cpp, MXQ is to the MXQ runtime.

---

## What MXQ Is

MXQ is three things:

1. **A quantization method** — importance-aware mixed-precision quantization that allocates more bits to weights that matter and fewer to weights that don't
2. **A file format** — `.mxq` files that store mixed-precision weights with all metadata needed for inference
3. **An inference runtime** — a Swift + Metal engine built from scratch for Apple Silicon, like llama.cpp but Metal-first

MXQ produces 2-3 bit models that match the quality of standard 4-bit uniform quantization, enabling larger models on less RAM. The format is open — anyone can build tools that read and write MXQ files.

**Format**: `.mxq` files (safetensors-based with MXQ metadata)
**Runtime**: Swift + Metal native binary (`mxq run`, `mxq serve`)
**Tooling**: Python package for calibration and quantization (`pip install mxq-tools`)
**Licensing**: Open format, open source
**Distribution**: Published on HuggingFace under dealignai org

### Why it matters

| Approach | 70B model RAM | Quality at 2.5-bit avg | Tokens/sec (M4 Max) |
|----------|--------------|----------------------|---------------------|
| MLX uniform 4-bit | ~40 GB | N/A (no 2.5-bit option) | ~15 t/s |
| MLX uniform 2-bit | ~20 GB | Garbage — unusable | ~31 t/s |
| GGUF Q2_K (llama.cpp) | ~22 GB | Mediocre — K-quant helps but rough | ~25 t/s |
| **MXQ 2.5-bit** | **~22 GB** | **Matches uniform 4-bit** | **~28 t/s** |

MXQ lets users run 70B+ models on 32-64GB Macs at quality levels that currently require 128GB+, at nearly 2x the speed of 4-bit.

### The vision

Make high-intelligence models accessible to everyone with an Apple Silicon Mac. A 70B model on a 32GB MacBook Air. A 120B model on a 64GB Mac Studio. A 400B model on a 192GB Mac Pro. Not dumbed-down — full quality, through mathematically optimal compression.

---

## Architecture Overview

MXQ has two major components: the **offline tooling** (Python) that creates `.mxq` files, and the **runtime** (Swift + Metal) that runs them.

```
MXQ Offline Pipeline (Python):

  [Calibrate] --> [Score] --> [Allocate] --> [Quantize] --> [Pack .mxq]
   (imatrix)    (importance)  (bits/block)  (mixed-prec)   (safetensors)

                              ↓

MLXQ Runtime (Swift + Metal):

  [Load .mxq] --> [Metal Kernels] --> [Inference] --> [Output]
  (mmap, zero-copy)  (dequant+matmul)   (transformer)   (tokens)
```

### Why this split

- **Offline tooling in Python**: Calibration requires running forward passes through full-precision models. Python has the ecosystem for this (HuggingFace, PyTorch, MLX). The offline pipeline runs once per model — speed isn't critical.
- **Runtime in Swift + Metal**: Inference runs millions of times. Every microsecond matters. Swift has first-class Metal integration. No other inference engine is built Swift-first for Apple Silicon — this is genuinely novel.

---

## Phase 1: MXQ File Format Specification (Week 1)

### Goal
Define the `.mxq` file format completely before building anything. The format is the contract between the offline tooling and the runtime. Get it right first.

### Format specification

```
model-name-MXQ-2.5bit/
  config.json                              # Standard HuggingFace model config
  tokenizer.json                           # Standard HuggingFace tokenizer
  tokenizer_config.json
  special_tokens_map.json
  mxq_config.json                          # MXQ-specific metadata
  mxq_imatrix.safetensors                  # Importance matrix (reproducibility)
  model-00001-of-00002.mxq.safetensors     # Quantized weights shard 1
  model-00002-of-00002.mxq.safetensors     # Quantized weights shard 2
  model.mxq.index.json                     # Shard index (tensor → shard mapping)
```

### mxq_config.json

```json
{
  "format": "mxq",
  "format_version": "1.0",
  "quantization": {
    "method": "mxq-importance",
    "target_bits": 2.5,
    "actual_bits": 2.51,
    "block_size": 64,
    "calibration_dataset": "mxq-calib-v1",
    "calibration_samples": 512,
    "scoring_method": "awq+hessian",
    "bit_widths_used": [2, 3, 4, 6, 8],
    "quantization_scheme": "asymmetric"
  },
  "layer_allocation": {
    "embed_tokens": { "bits": 4, "note": "vocabulary representation — protected" },
    "lm_head": { "bits": 6, "note": "output logits — protected" },
    "layers.0-1": { "avg_bits": 4.2, "note": "first layers — error propagation protection" },
    "layers.2-77": { "avg_bits": 2.3, "note": "middle layers — aggressively compressed" },
    "layers.78-79": { "avg_bits": 4.0, "note": "last layers — output quality protection" },
    "attention.q_proj": { "avg_bits": 3.8 },
    "attention.k_proj": { "avg_bits": 3.5 },
    "attention.v_proj": { "avg_bits": 3.2 },
    "attention.o_proj": { "avg_bits": 3.0 },
    "mlp.gate_proj": { "avg_bits": 2.1 },
    "mlp.up_proj": { "avg_bits": 2.1 },
    "mlp.down_proj": { "avg_bits": 2.3 }
  },
  "source_model": {
    "name": "Qwen/Qwen3.5-72B",
    "dtype": "bfloat16",
    "parameters": "72B",
    "sha256": "abc123..."
  },
  "quality_metrics": {
    "perplexity_bf16": 5.21,
    "perplexity_mxq": 5.38,
    "perplexity_uniform_4bit": 5.42,
    "perplexity_uniform_2bit": 12.7,
    "mmlu_mxq": 0.82,
    "humaneval_mxq": 0.71
  },
  "runtime": {
    "recommended_memory_gb": 24,
    "kv_cache_memory_32k_fp16_gb": 8.4,
    "total_memory_32k_gb": 32.4
  }
}
```

### Weight storage in safetensors

Each weight tensor is stored with companion metadata tensors:

```
layers.N.self_attn.q_proj.qweight          # Packed quantized weight data (uint8 blob)
layers.N.self_attn.q_proj.scales           # Per-block scale factors (float16)
layers.N.self_attn.q_proj.zeros            # Per-block zero points (float16)
layers.N.self_attn.q_proj.bit_map          # Per-block bit width (uint8: 2,3,4,5,6,8)
layers.N.self_attn.q_proj.block_offsets    # Byte offset of each block in qweight (uint32)
```

The `block_offsets` array is critical for variable bit-width: since blocks have different numbers of bits, you can't compute byte offsets with simple arithmetic. The offset array gives O(1) random access to any block.

### Bit packing

Each block of weights (default 64 weights) is packed at its assigned bit width:

| Bit width | Values per byte | Bytes per block (64 weights) | Mask |
|-----------|----------------|------------------------------|------|
| 2 | 4 | 16 | 0x03 |
| 3 | 2.67 | 24 | 0x07 |
| 4 | 2 | 32 | 0x0F |
| 5 | 1.6 | 40 | 0x1F |
| 6 | 1.33 | 48 | 0x3F |
| 8 | 1 | 64 | 0xFF |

For non-byte-aligned bit widths (3, 5, 6): bits are packed contiguously across byte boundaries. The dequant kernel handles cross-byte extraction via bit shifting.

### Deliverables
- `FORMAT.md` — complete format specification (the public standard)
- Reference Python reader/writer for validation
- Format validation tool: `mxq-tools validate model-dir/`

---

## Phase 2: Calibration Engine (Week 2-3)

### Goal
Run calibration data through a full-precision model, producing an importance matrix that tells the quantizer which weights matter most.

### The math

**Why calibration matters**: At 2-bit, you have only 4 quantization levels per weight. If you allocate bits uniformly, critical weights get the same 4 levels as throwaway weights. Calibration identifies which weights are critical so the allocator can give them more bits.

**Importance scoring combines two signals:**

1. **Activation-aware scoring (AWQ-style)**
   ```
   importance(W_ij) = ||X_j||₂ × |W_ij|
   ```
   Where `X_j` is the activation vector for input channel j across all calibration tokens. Weights that process large activations AND are large themselves are most important.

   Per-block importance: mean of per-weight importance within the block.

2. **Hessian diagonal scoring**
   ```
   importance(W_ij) = W_ij² × H_jj
   ```
   Where `H_jj = Σ_t X_{t,j}²` (sum of squared activations for channel j across all tokens).
   This measures how much the loss would increase if we perturbed weight W_ij.

3. **Combined score**
   ```
   final_score = α × activation_score + (1-α) × hessian_score
   ```
   Where α is tuned on a validation set (typically α ≈ 0.7 — activation score is cheaper and nearly as good).

### Calibration dataset

Curate a diverse dataset (~512 samples, 2048 tokens each):

| Category | % | Purpose |
|----------|---|---------|
| General text (web, books) | 25% | Baseline language modeling |
| Code (Python, JS, C++, Swift, Rust) | 20% | Preserve coding ability |
| Multi-turn conversation | 15% | Chat quality |
| Reasoning (math, logic, CoT) | 15% | Preserve reasoning chains |
| Multilingual (ZH, JA, KO, ES, DE) | 15% | Multilingual capability |
| Long context (8K-32K tokens) | 10% | Long-range dependency preservation |

Store as: `datasets/calibration_v1.jsonl`

### Activation collection

```python
# Hook into every Linear layer
def collect_activations(model, calibration_data):
    stats = {}  # layer_name → running statistics

    def hook_fn(name):
        def hook(module, input, output):
            x = input[0].float()  # (batch, seq, hidden)
            # Welford's online algorithm for numerical stability
            channel_sq_sum = x.pow(2).sum(dim=(0, 1))  # (hidden,)
            stats[name].update(channel_sq_sum, x.shape[0] * x.shape[1])
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            stats[name] = WelfordAccumulator(module.in_features)
            module.register_forward_hook(hook_fn(name))

    # Forward pass over all calibration samples
    for batch in calibration_data:
        with torch.no_grad():
            model(batch)

    return stats
```

### Output: importance matrix

Saved as safetensors file:
```
layers.0.self_attn.q_proj.importance    # shape: (n_blocks,) float32
layers.0.self_attn.q_proj.act_norms     # shape: (in_features,) float32
layers.0.self_attn.k_proj.importance    # ...
...
```

### Tasks
- `mxq-tools/calibrate.py` — calibration runner
- `mxq-tools/hooks.py` — activation collection hooks
- `mxq-tools/score.py` — importance scoring (AWQ + Hessian)
- `mxq-tools/datasets/` — calibration dataset curation

### CLI
```bash
mxq-tools calibrate \
  --model Qwen/Qwen3.5-72B \
  --dataset calibration_v1.jsonl \
  --output imatrix.safetensors \
  --samples 512 \
  --seq-len 2048 \
  --scoring awq+hessian
```

---

## Phase 3: Bit Allocation and Quantization (Week 3-5)

### Goal
Given an importance matrix and a target average bit width, decide how many bits each weight block gets, then quantize with optimal rounding.

### Bit allocation algorithm

**The mathematical foundation: rate-distortion optimization**

For N blocks with importance scores s_1, ..., s_N and weight variances σ²_1, ..., σ²_N:

```
Minimize:  Σᵢ sᵢ × D(bᵢ)          (total importance-weighted distortion)
Subject to: (1/N) × Σᵢ bᵢ = B_target  (average bits = target)
            bᵢ ∈ {2, 3, 4, 5, 6, 8}   (allowed bit widths)
```

Where D(b) is the quantization distortion at b bits:
```
D(b) ≈ σ² × Δ²/12,  where Δ = range / 2^b
```

**Practical algorithm (greedy, fast):**

```python
def allocate_bits(importance_scores, target_bits, n_blocks):
    # Initialize all blocks at minimum
    bits = [2] * n_blocks

    # Apply layer-type priors (minimum bit floors)
    for i, block in enumerate(blocks):
        if block.layer_type == 'embed_tokens':   bits[i] = max(bits[i], 4)
        if block.layer_type == 'lm_head':        bits[i] = max(bits[i], 6)
        if block.layer_type == 'q_proj':          bits[i] = max(bits[i], 3)
        if block.layer_type == 'k_proj':          bits[i] = max(bits[i], 3)
        if block.is_first_or_last_2_layers:       bits[i] += 1

    # Compute remaining bit budget
    current_avg = mean(bits)
    budget = (target_bits - current_avg) * n_blocks

    # Sort blocks by importance (descending)
    priority = argsort(importance_scores, descending=True)

    # Upgrade most important blocks until budget exhausted
    for idx in priority:
        if budget <= 0:
            break
        if bits[idx] < 8:  # max bit width
            next_bits = next_allowed(bits[idx])  # 2→3→4→5→6→8
            cost = next_bits - bits[idx]
            if cost <= budget:
                bits[idx] = next_bits
                budget -= cost

    return bits
```

**Advanced algorithm (dynamic programming, optimal):**

```python
def allocate_bits_optimal(importance, target_bits, n_blocks):
    # For each block, precompute distortion at each bit width
    # distortion[i][b] = importance[i] * quantization_error(block_weights[i], b)

    # Solve knapsack: minimize total distortion subject to bit budget
    # This is a bounded knapsack problem — solvable in O(n_blocks × bit_budget)

    # DP state: dp[i][b] = min distortion using first i blocks with total bits = b
    ...
```

### Per-block quantization

For each block at its allocated bit width:

```python
def quantize_block(weights, bits):
    """Asymmetric per-block quantization with optimal scaling."""
    n_levels = 2 ** bits

    # Find optimal scale and zero-point (MSE minimization)
    w_min, w_max = weights.min(), weights.max()
    scale = (w_max - w_min) / (n_levels - 1)
    zero_point = round(-w_min / scale)

    # Quantize
    quantized = torch.clamp(torch.round(weights / scale + zero_point), 0, n_levels - 1)

    # Pack into bytes
    packed = pack_bits(quantized.to(torch.uint8), bits)

    return packed, scale, zero_point
```

**GPTQ-style optimal rounding (for higher quality):**

Instead of simple round-to-nearest, use Hessian-guided error compensation:

```python
def quantize_block_gptq(weights, bits, hessian_diag):
    """Quantize with error compensation — each weight's rounding error
    is compensated by adjusting subsequent weights."""
    W = weights.clone()
    Q = torch.zeros_like(W)

    for col in range(W.shape[1]):
        w = W[:, col]
        q = quantize_scalar(w, bits)
        error = w - dequantize(q, bits)
        Q[:, col] = q

        # Compensate remaining weights for this error
        if col + 1 < W.shape[1]:
            W[:, col+1:] -= error.unsqueeze(1) * (hessian_row / hessian_diag[col])

    return Q
```

### Bit allocation profiles (presets)

| Profile | Avg bits | Effective bits (with overhead) | Target use case |
|---------|----------|-------------------------------|----------------|
| MXQ-2 | 2.0-2.4 | 2.5-2.9 | Maximum compression: 70B on 32GB, 120B on 64GB |
| MXQ-2.5 | 2.5 | 3.0 | Sweet spot — matches uniform 4-bit quality |
| MXQ-3 | 3.0 | 3.5 | High quality with significant compression |
| MXQ-4 | 4.0 | 4.5 | Best quality with smart allocation (better than uniform 4-bit) |
| MXQ-6 | 6.0 | 6.5 | Near-lossless |

### Quality validation

After quantization, evaluate:

```bash
mxq-tools evaluate \
  --model ./Qwen3.5-72B-MXQ-2.5bit \
  --baseline Qwen/Qwen3.5-72B \
  --datasets wikitext,c4,lambada \
  --tasks mmlu,humaneval,gsm8k
```

Report includes:
- Perplexity delta vs full precision and uniform 4-bit
- Per-layer quantization error (Frobenius norm)
- Bit allocation histogram
- Task-specific accuracy

### Tasks
- `mxq-tools/allocate.py` — bit allocation (greedy + optimal DP)
- `mxq-tools/quantize.py` — per-block quantization with GPTQ rounding
- `mxq-tools/pack.py` — bit packing into .mxq safetensors
- `mxq-tools/evaluate.py` — quality evaluation suite
- `mxq-tools/format/writer.py` — write .mxq directory
- `mxq-tools/format/reader.py` — read .mxq directory

### CLI
```bash
# Quantize with importance matrix
mxq-tools quantize \
  --model Qwen/Qwen3.5-72B \
  --imatrix imatrix.safetensors \
  --bits 2.5 \
  --output ./Qwen3.5-72B-MXQ-2.5bit

# One-shot: calibrate + quantize + evaluate
mxq-tools convert \
  --model Qwen/Qwen3.5-72B \
  --bits 2.5 \
  --output ./Qwen3.5-72B-MXQ-2.5bit \
  --evaluate
```

---

## Phase 4: Metal Dequantization Kernels (Week 5-8)

### Goal
Write high-performance Metal GPU kernels that dequantize MXQ weights during inference. This is the hardest and most critical phase — the performance of MXQ inference depends entirely on these kernels.

### Why custom kernels from scratch

We are NOT using MLX's kernel infrastructure. We are writing our own Metal compute pipeline from scratch because:

1. MLX only supports uniform bit widths — no variable bit-width per block
2. We want total control over memory layout, threadgroup sizing, and dispatch strategy
3. We can optimize for specific Apple GPU generations (M3, M4, M5)
4. No dependency on MLX's internals — if MLX changes, we don't break
5. This is what makes MXQ a real runtime, not a library on top of someone else's runtime

### The core challenge: variable bit-width dequantization

In uniform quantization, every weight uses the same number of bits. The byte offset of weight i is trivially `i * bits / 8`. In MXQ, each block uses a DIFFERENT number of bits. You need the `block_offsets` array to find where each block starts.

### Kernel 1: mxq_dequant_matmul (fused dequantize + matrix multiply)

This kernel does 90%+ of all compute. For every linear layer: load quantized weights, dequantize on-the-fly, multiply with input activations, produce output.

```metal
#include <metal_stdlib>
using namespace metal;

// Block size (weights per block)
constant uint BLOCK_SIZE = 64;

// Tile sizes for matmul
constant uint TILE_M = 32;  // output rows per tile
constant uint TILE_N = 32;  // output cols per tile
constant uint TILE_K = 64;  // reduction dimension per tile (= BLOCK_SIZE)

kernel void mxq_dequant_matmul(
    // Quantized weight data
    device const uint8_t*  qweight       [[buffer(0)]],  // packed bits
    device const half*     scales        [[buffer(1)]],  // per-block scale
    device const half*     zeros         [[buffer(2)]],  // per-block zero point
    device const uint8_t*  bit_map       [[buffer(3)]],  // per-block bit width
    device const uint32_t* block_offsets [[buffer(4)]],  // byte offset per block

    // Input activations
    device const half*     x             [[buffer(5)]],  // (M, K)

    // Output
    device half*           output        [[buffer(6)]],  // (M, N)

    // Dimensions
    constant uint&         M             [[buffer(7)]],  // batch * seq_len
    constant uint&         K             [[buffer(8)]],  // input features
    constant uint&         N             [[buffer(9)]],  // output features

    // Thread indexing
    uint3 tgid  [[threadgroup_position_in_grid]],
    uint3 tid   [[thread_position_in_threadgroup]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_id   [[simdgroup_index_in_threadgroup]]
) {
    // Each threadgroup computes a TILE_M × TILE_N tile of the output
    uint row_start = tgid.y * TILE_M;
    uint col_start = tgid.x * TILE_N;

    // Accumulator in float32 for numerical stability
    float acc[TILE_M / 32][TILE_N / 32];  // per-thread accumulation

    // Threadgroup shared memory for input tile and dequantized weight tile
    threadgroup half x_tile[TILE_M][TILE_K];
    threadgroup half w_tile[TILE_K][TILE_N];

    // Iterate over K dimension in TILE_K chunks
    uint n_blocks_per_col = K / BLOCK_SIZE;

    for (uint k_start = 0; k_start < K; k_start += TILE_K) {
        // Load input tile into shared memory (cooperative load)
        // ... (standard tiled matmul input loading)

        // Dequantize weight tile into shared memory
        // Each thread dequantizes a portion of the weight blocks
        uint block_col_start = col_start;
        uint block_k = k_start / BLOCK_SIZE;

        for (uint local_n = tid.x; local_n < TILE_N; local_n += 32) {
            uint global_col = col_start + local_n;
            if (global_col >= N) continue;

            uint block_idx = global_col * n_blocks_per_col + block_k;
            uint8_t bits = bit_map[block_idx];
            half scale = scales[block_idx];
            half zero = zeros[block_idx];
            uint32_t byte_off = block_offsets[block_idx];

            // Dequantize BLOCK_SIZE weights from this block
            for (uint w = tid.y; w < BLOCK_SIZE; w += 32) {
                uint bit_offset = w * bits;
                uint byte_idx = byte_off + bit_offset / 8;
                uint bit_shift = bit_offset % 8;
                uint mask = (1u << bits) - 1u;

                // Extract value (handles cross-byte boundary)
                uint16_t raw_bytes = *(device const uint16_t*)(qweight + byte_idx);
                uint raw = (raw_bytes >> bit_shift) & mask;

                // Dequantize: val = (raw - zero) * scale
                w_tile[w][local_n] = half(float(raw) * float(scale) + float(zero));
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Standard tiled matmul on the dequantized tile
        // Using simdgroup_matrix for hardware-accelerated 8x8 matmul
        // ... (Apple's simdgroup matrix multiply operations)

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output tile
    // ...
}
```

### Kernel 2: mxq_dequant_standalone

Dequantize to a float16 buffer without matmul. Used for debugging and inspection.

### Kernel 3: mxq_dequant_batched

Batched version for prefill (processing multiple tokens at once). Same as Kernel 1 but with larger TILE_M.

### Kernel 4: Non-quantized compute kernels

The runtime also needs kernels for operations that don't involve quantized weights:

```
mxq_rmsnorm          — RMSNorm: y = x * rsqrt(mean(x²) + ε) * γ
mxq_rope             — Rotary Position Embeddings
mxq_softmax          — Numerically stable softmax
mxq_silu_mul         — SiLU(x) * y (for SwiGLU)
mxq_add              — Residual connections
mxq_embedding_lookup — Token → embedding vector
mxq_rms_norm_rope    — Fused RMSNorm + RoPE (optimization)
```

### Performance optimization strategy

1. **Memory coalescing**: adjacent threads read adjacent bytes from qweight
2. **Threadgroup sizing**: tune for each Apple GPU (M3: 10 cores, M4: up to 40 cores for Max)
3. **SIMD-group matrix ops**: use `simdgroup_float8x8` for hardware-accelerated small matmul
4. **Double buffering**: prefetch next K-tile while computing current
5. **Block sorting**: sort blocks by bit width at load time so threads in a SIMD group process same-width blocks (avoids branch divergence)
6. **Dispatch strategy for variable bit-width**:
   - Option A: single kernel with runtime bit_width lookup (simplest)
   - Option B: separate kernels per bit width, dispatch in groups (best throughput)
   - Option C: sort blocks by bit width, dispatch sorted ranges (good balance)
   - Start with A, optimize to C if benchmarks show branch divergence costs

### Performance targets
- Decode throughput (batch=1): within 95% of theoretical bandwidth limit
- Prefill throughput: within 80% of peak GPU TFLOPS
- Kernel overhead vs uniform 4-bit: less than 5%
- Memory overhead: block_offsets + bit_map < 1% of total model size

### Tasks
- `mxq-runtime/Metal/MXQDequantMatmul.metal` — fused dequant+matmul
- `mxq-runtime/Metal/MXQDequant.metal` — standalone dequant
- `mxq-runtime/Metal/MXQCompute.metal` — non-quantized kernels (RMSNorm, RoPE, etc.)
- `mxq-runtime/Metal/MXQKernelBench.swift` — kernel benchmarks
- `mxq-runtime/Sources/MXQMetal/` — Swift Metal pipeline management

---

## Phase 5: Swift Inference Runtime (Week 7-10)

### Goal
Build a complete inference runtime in Swift that loads MXQ models and runs them on Apple Silicon GPUs. This is the equivalent of llama.cpp but Swift-native and Metal-first.

### Architecture

```
mxq-runtime/
  Package.swift
  Sources/
    MXQ/                          # Main library
      MXQModel.swift              # Model abstraction
      MXQConfig.swift             # Config parsing
      MXQLoader.swift             # Model loading (mmap + Metal buffers)
      MXQTokenizer.swift          # BPE tokenizer
      MXQSampler.swift            # Sampling strategies
      MXQKVCache.swift            # KV cache management
      MXQInference.swift          # Forward pass orchestration

    MXQMetal/                     # Metal integration
      MXQMetalDevice.swift        # GPU device management
      MXQMetalPipeline.swift      # Compute pipeline setup
      MXQMetalBuffers.swift       # Buffer allocation
      MXQMetalDispatch.swift      # Kernel dispatch

    MXQArchitectures/             # Model architecture implementations
      LlamaModel.swift            # Llama 3/4 family
      QwenModel.swift             # Qwen 2/3 family
      MistralModel.swift          # Mistral family
      GemmaModel.swift            # Gemma 2/3 family
      PhiModel.swift              # Phi 3/4 family
      DeepSeekModel.swift         # DeepSeek V2/V3
      ModelProtocol.swift         # Protocol all architectures implement

    MXQCLI/                       # CLI binary
      main.swift
      RunCommand.swift            # mxq run
      ServeCommand.swift          # mxq serve (OpenAI-compatible API)
      InfoCommand.swift           # mxq info
      BenchCommand.swift          # mxq bench

  Metal/                          # Metal shader sources
    MXQDequantMatmul.metal
    MXQDequant.metal
    MXQCompute.metal
```

### Model loading (zero-copy via unified memory)

```swift
class MXQLoader {
    func load(path: URL) throws -> MXQModel {
        // 1. Parse mxq_config.json
        let config = try MXQConfig(contentsOf: path.appending("mxq_config.json"))

        // 2. Memory-map safetensors files (zero-copy)
        let shards = try loadShards(path: path, index: config.shardIndex)

        // 3. Create MTLBuffers from mmap'd memory (NO COPY — unified memory)
        let device = MTLCreateSystemDefaultDevice()!
        var buffers: [String: MTLBuffer] = [:]

        for (name, tensorInfo) in shards.tensors {
            // makeBuffer(bytesNoCopy:) wraps existing memory as GPU-accessible
            // This is the Apple Silicon advantage: CPU memory IS GPU memory
            let buffer = device.makeBuffer(
                bytesNoCopy: tensorInfo.dataPointer,
                length: tensorInfo.byteCount,
                options: .storageModeShared  // accessible by CPU and GPU
            )
            buffers[name] = buffer
        }

        // 4. Determine architecture and instantiate
        let arch = try detectArchitecture(config: config)
        return try arch.init(config: config, buffers: buffers, device: device)
    }
}
```

### Forward pass

```swift
protocol MXQModelArchitecture {
    func forward(tokens: [Int], kvCache: inout MXQKVCache,
                 commandBuffer: MTLCommandBuffer) -> MTLBuffer  // logits
}

// Example: Llama architecture
class LlamaModel: MXQModelArchitecture {
    func forward(tokens: [Int], kvCache: inout MXQKVCache,
                 commandBuffer: MTLCommandBuffer) -> MTLBuffer {
        var hidden = embedTokens(tokens)  // embedding lookup

        for layer in 0..<config.numLayers {
            // Attention block
            let normed = rmsNorm(hidden, weights: layer.inputNorm)
            let qkv = mxqMatmul(normed, qWeight: layer.qProj,
                                         kWeight: layer.kProj,
                                         vWeight: layer.vProj)
            let (q, k, v) = splitQKV(qkv)
            let (qRoped, kRoped) = applyRoPE(q, k, position: kvCache.position)
            kvCache.append(key: kRoped, value: v, layer: layer.index)
            let attnOut = attention(qRoped, kvCache: kvCache, layer: layer.index)
            let attnProj = mxqMatmul(attnOut, weight: layer.oProj)
            hidden = hidden + attnProj  // residual

            // MLP block
            let normed2 = rmsNorm(hidden, weights: layer.postNorm)
            let gate = mxqMatmul(normed2, weight: layer.gateProj)
            let up = mxqMatmul(normed2, weight: layer.upProj)
            let mlpOut = siluMul(gate, up)
            let down = mxqMatmul(mlpOut, weight: layer.downProj)
            hidden = hidden + down  // residual
        }

        let finalNorm = rmsNorm(hidden, weights: model.finalNorm)
        let logits = mxqMatmul(finalNorm, weight: model.lmHead)
        return logits
    }
}
```

### KV Cache

```swift
class MXQKVCache {
    var keys: [[MTLBuffer]]     // [layer][position] → key vector
    var values: [[MTLBuffer]]   // [layer][position] → value vector
    var position: Int = 0
    let maxSeqLen: Int

    // Pre-allocate full cache at startup (avoids runtime allocation)
    init(config: MXQConfig, device: MTLDevice, maxSeqLen: Int) {
        let kvHeads = config.numKVHeads
        let headDim = config.headDim
        let bytesPerPos = kvHeads * headDim * 2  // fp16

        for layer in 0..<config.numLayers {
            let keyBuf = device.makeBuffer(length: maxSeqLen * bytesPerPos)!
            let valBuf = device.makeBuffer(length: maxSeqLen * bytesPerPos)!
            keys.append(keyBuf)
            values.append(valBuf)
        }
    }
}
```

### CLI

```bash
# Interactive chat
mxq run dealignai/Qwen3-72B-MXQ-2.5bit --interactive

# Single prompt
mxq run model-dir/ --prompt "Explain quantum computing" --temp 0.7

# OpenAI-compatible API server
mxq serve dealignai/Qwen3-72B-MXQ-2.5bit --port 8080

# Model info
mxq info model-dir/   # shows bit allocation, quality metrics, memory requirements

# Benchmark
mxq bench model-dir/  # measures tokens/sec, prefill speed, memory usage
```

### Supported architectures (priority order)

1. Llama (3.x, 4.x) — most popular
2. Qwen (2.5, 3, 3.5) — best multilingual
3. Gemma (2, 3) — Google's best small models
4. Mistral / Mixtral — including MoE
5. Phi (3, 4) — best tiny models
6. DeepSeek (V2, V3) — MoE + MLA architecture
7. Command R — Cohere models

### Tasks
- Swift package setup with Metal shader compilation
- Model loader with mmap + MTLBuffer
- Tokenizer (BPE implementation)
- Forward pass for Llama architecture (first target)
- KV cache
- Sampling (top-k, top-p, min-p, temperature, repetition penalty)
- CLI with ArgumentParser
- OpenAI-compatible HTTP server (swift-nio based)

---

## Phase 6: Python Bindings and Tooling Package (Week 10-11)

### Goal
Ship two Python packages:
1. `mxq-tools` — offline calibration, quantization, evaluation (pure Python)
2. `mxq` — Python bindings for the Swift runtime (for users who want Python API)

### mxq-tools (offline tooling)

```bash
pip install mxq-tools
```

```python
import mxq_tools

# Calibrate
imatrix = mxq_tools.calibrate(
    model="Qwen/Qwen3.5-72B",
    dataset="calibration_v1.jsonl",
    samples=512,
    scoring="awq+hessian"
)
imatrix.save("imatrix.safetensors")

# Quantize
mxq_tools.quantize(
    model="Qwen/Qwen3.5-72B",
    imatrix="imatrix.safetensors",
    bits=2.5,
    output="./Qwen3.5-72B-MXQ-2.5bit"
)

# Evaluate
results = mxq_tools.evaluate(
    model="./Qwen3.5-72B-MXQ-2.5bit",
    baseline="Qwen/Qwen3.5-72B",
    datasets=["wikitext", "c4"]
)
print(results.perplexity_delta)

# Inspect
info = mxq_tools.inspect("./Qwen3.5-72B-MXQ-2.5bit")
print(info.bit_allocation_histogram)
print(info.layer_summary)
```

### mxq (Python bindings for runtime)

```bash
pip install mxq
```

```python
import mxq

# Load and run
model = mxq.load("dealignai/Qwen3-72B-MXQ-2.5bit")
output = model.generate("Hello world", max_tokens=100, temperature=0.7)
print(output)

# Streaming
for token in model.stream("Explain quantum computing"):
    print(token, end="", flush=True)
```

### Dependencies
- `mxq-tools`: Python 3.11+, torch or mlx-lm, safetensors, numpy, tqdm, huggingface_hub
- `mxq`: Python 3.11+, macOS 15+ (wraps Swift runtime via ctypes/cffi)

---

## Phase 7: Model Publishing and Validation (Week 11-12)

### Goal
Quantize flagship models, validate quality rigorously, publish on HuggingFace.

### Priority models

| Model | Params | MXQ-2.5 size | MXQ-3 size | Target hardware |
|-------|--------|-------------|-----------|----------------|
| Qwen3.5-72B | 72B | ~22 GB | ~27 GB | 32GB MacBook Pro/Air |
| Llama-4-Scout-109B | 109B | ~34 GB | ~41 GB | 64GB Mac Studio |
| Gemma-3-27B | 27B | ~8.5 GB | ~10 GB | 16GB MacBook Air |
| Phi-4-14B | 14B | ~4.4 GB | ~5.3 GB | 8GB base Mac |
| DeepSeek-V3-671B | 671B | ~210 GB | ~252 GB | 512GB Mac Studio Ultra |
| Nemotron-H-120B | 120B | ~38 GB | ~45 GB | 96GB Mac Studio |
| Qwen3.5-VL-72B | 72B | ~22 GB | ~27 GB | 32GB + vision |

### HuggingFace naming convention

```
dealignai/Qwen3.5-72B-MXQ-2.5bit
dealignai/Qwen3.5-72B-MXQ-3bit
dealignai/Qwen3.5-72B-MXQ-4bit
dealignai/Gemma-3-27B-MXQ-2.5bit
```

### Quality validation checklist

For each model before publishing:
- [ ] Perplexity within 5% of uniform 4-bit (for MXQ-2.5)
- [ ] Perplexity within 2% of uniform 4-bit (for MXQ-3)
- [ ] MMLU score within 3% of full precision
- [ ] HumanEval score within 5% of full precision
- [ ] No degenerate outputs on 100 diverse test prompts
- [ ] Code generation quality maintained
- [ ] Multilingual quality maintained (EN, ZH, JA, KO, ES)
- [ ] Long context coherence at 32K+ tokens
- [ ] Correct output on chain-of-thought reasoning tasks
- [ ] Model card with full quality metrics, bit allocation breakdown, and comparison charts

---

## Phase 8: REAP — Pruning + Quantization (Future)

### Goal
Combine structured pruning with MXQ quantization for even more extreme compression. REAP (Realistic Estimation And Pruning) removes entire weight blocks before quantization, then MXQ quantizes what remains.

### Why pruning + quantization

The two techniques handle different types of redundancy:
- **Pruning**: removes weights that don't matter at all (set to zero, don't store)
- **Quantization**: compresses weights that matter but don't need full precision

Combined compression:
```
effective_bits = (1 - sparsity) × bits_per_weight
```

Example: 40% pruning + 2.5-bit MXQ = 1.5 effective bits per weight
- 70B model at 1.5 bits ≈ 13 GB — runs on 16GB MacBook Air

### Pipeline

```
[Calibrate] → [Score importance] → [Prune] → [Re-calibrate] → [Allocate bits] → [Quantize] → [Pack .mxq]
```

1. Score all weight blocks using importance metrics
2. Prune blocks below threshold (set to zero, mark in bitmap)
3. Re-calibrate on remaining weights (pruning changes optimal bit allocation)
4. Allocate bits to surviving blocks
5. Quantize surviving blocks
6. Store with pruning bitmap in .mxq format

### Format extension

```json
{
  "compression": {
    "method": "mxq+reap",
    "sparsity": 0.40,
    "target_bits": 2.5,
    "effective_bits": 1.5,
    "pruning_granularity": "block"
  }
}
```

Additional tensor per weight:
```
layers.N.self_attn.q_proj.prune_mask    # uint8 bitmap: 1=kept, 0=pruned
```

Metal kernel skips pruned blocks entirely (no memory load, no compute).

---

## File Structure (Complete Project)

```
mxq/
  MLXQ-PLAN.md                          # This document
  FORMAT.md                            # Public format specification
  LICENSE                              # Open source license

  research/                            # Research notes and references
    01-quantization-fundamentals.md
    02-llamacpp-gguf-architecture.md
    03-importance-aware-quantization-methods.md
    04-extreme-quantization-2bit.md
    05-apple-metal-gpu-computing.md
    06-pruning-and-quantization.md
    07-matrix-mathematics-for-inference.md
    08-calibration-and-importance-scoring.md
    09-swift-inference-runtime-design.md

  mxq-tools/                           # Python: offline calibration & quantization
    pyproject.toml
    mxq_tools/
      __init__.py
      __main__.py                      # CLI entry point
      calibrate.py                     # Calibration engine
      hooks.py                         # Activation collection hooks
      score.py                         # Importance scoring
      allocate.py                      # Bit allocation algorithm
      quantize.py                      # Per-block quantization
      pack.py                          # Bit packing
      evaluate.py                      # Quality evaluation
      format/
        __init__.py
        spec.py                        # Format constants
        writer.py                      # Write .mxq files
        reader.py                      # Read .mxq files
        validator.py                   # Validate .mxq format
      datasets/
        calibration_v1.jsonl
    tests/
      test_quantize.py
      test_allocate.py
      test_format.py

  mxq-runtime/                         # Swift + Metal: inference runtime
    Package.swift
    Sources/
      MXQ/
        MXQModel.swift
        MXQConfig.swift
        MXQLoader.swift
        MXQTokenizer.swift
        MXQSampler.swift
        MXQKVCache.swift
        MXQInference.swift
      MXQMetal/
        MXQMetalDevice.swift
        MXQMetalPipeline.swift
        MXQMetalBuffers.swift
        MXQMetalDispatch.swift
      MXQArchitectures/
        ModelProtocol.swift
        LlamaModel.swift
        QwenModel.swift
        GemmaModel.swift
        MistralModel.swift
        PhiModel.swift
      MXQCLI/
        main.swift
        RunCommand.swift
        ServeCommand.swift
        InfoCommand.swift
        BenchCommand.swift
    Metal/
      MXQDequantMatmul.metal
      MXQDequant.metal
      MXQCompute.metal
    Tests/
      MXQTests/
        KernelTests.swift
        LoaderTests.swift
        InferenceTests.swift
    Benchmarks/
      DequantBench.swift
      MatmulBench.swift

  mxq-python/                          # Python bindings for Swift runtime
    pyproject.toml
    mxq/
      __init__.py
      model.py
      runtime.py                       # ctypes/cffi wrapper around Swift lib
```

---

## How MXQ Compares

| | MXQ | GGUF (llama.cpp) | MLX uniform | AWQ | EXL2 |
|---|-----|-------------------|-------------|-----|------|
| Target hardware | Apple Silicon | CPU + CUDA + Metal | Apple Silicon | CUDA | CUDA |
| Runtime | Swift + Metal | C++ + backends | Python + Metal | Various | ExLlamaV2 |
| Mixed precision | Yes, per-block | Yes, per-layer (K-quant) | No, uniform | No, uniform | Yes, per-block |
| Importance-aware | Yes, calibrated | Yes, imatrix (IQ types) | No | Yes, activation | Yes, calibrated |
| 2-bit usable | Yes | Mediocre (Q2_K), OK (IQ2) | No | No | Yes |
| Metal-first | Yes, designed for | No, bolted on | Partial (MLX) | No | No |
| Unified memory | Yes, mmap zero-copy | Yes, mmap | Yes | No | No |
| Open format | Yes | Yes | N/A (uses safetensors) | N/A | No (ExLlama only) |
| Pruning support | Planned (REAP) | No | No | No | No |

---

## Dependencies

### mxq-tools (Python, offline)
- Python 3.11+
- torch >= 2.0 or mlx >= 0.22 (for model loading and forward passes)
- safetensors
- numpy
- tqdm
- huggingface_hub
- datasets (for calibration data)

### mxq-runtime (Swift, inference)
- macOS 15+ (Sequoia) / macOS 26 (Tahoe) for latest Metal features
- Xcode 16+ / Swift 6+
- Metal 3 (M1+) — Metal 4 (M4+) for latest optimizations
- swift-argument-parser (CLI)
- swift-nio (HTTP server for `mxq serve`)

---

## Success Criteria

1. MXQ-2.5bit perplexity within 5% of uniform 4-bit on Wikitext
2. MXQ-3bit perplexity within 2% of uniform 4-bit
3. 70B model runs on 32GB Mac via MXQ-2.5 at 20+ tokens/sec
4. Metal dequant kernel adds less than 5% overhead vs uniform 4-bit throughput
5. `mxq run` loads a model in under 5 seconds (mmap, zero-copy)
6. At least 5 flagship models published on dealignai HuggingFace
7. Format spec is public and documented — anyone can implement a loader
8. Python tooling works end-to-end: `mxq-tools convert` produces a working .mxq model
9. Community can download and run MXQ models with zero configuration
