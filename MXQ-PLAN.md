# MXQ — Mixed-Precision Importance Quantization for MLX

> Proprietary quantization format for Apple Silicon. Open for use, exclusive to vMLX Engine.
> Like EXL2 is to ExLlamaV2, MXQ is to vMLX.

---

## What MXQ Is

MXQ is an importance-aware mixed-precision quantization method and file format for MLX models on Apple Silicon. It allocates more bits to weights that matter and fewer to weights that don't — producing 2-3 bit models that match the quality of standard 4-bit uniform quantization, enabling larger models on less RAM.

**Format**: `.mxq` files (safetensors-based with MXQ metadata headers)
**Licensing**: Proprietary format, open for community use. Anyone can download and run MXQ models, but only vMLX Engine can load them.
**Distribution**: Published on HuggingFace under dealignai org.

### Why it matters

| Approach | 70B model RAM needed | Quality at 2.5-bit avg |
|----------|---------------------|----------------------|
| MLX uniform 4-bit | ~40 GB | N/A (no 2.5-bit) |
| MLX uniform 2-bit | ~20 GB | Garbage — unusable |
| GGUF Q2_K (llama.cpp) | ~22 GB | Mediocre — K-quant helps but still rough |
| **MXQ 2.5-bit** | **~22 GB** | **Matches uniform 4-bit** — importance-aware |

MXQ lets users run 70B+ models on 32-64GB Macs at quality levels that currently require 128GB+.

---

## Architecture Overview

```
MXQ Pipeline:

  [Calibrate] --> [Score] --> [Allocate] --> [Quantize] --> [Pack .mxq]
   (imatrix)    (importance)  (bits/block)  (mixed-prec)

                              |
                              v

  vMLX Engine Runtime:
  [MXQ Loader] --> [Metal Dequant Kernels] --> [Inference]
  (parse format)   (variable bit-width)
```

---

## Phase 1: Calibration Engine (Week 1-2)

### Goal
Run a calibration dataset through the full-precision model, collecting per-layer activation statistics that tell us which weights are most important.

### Tasks

1. **Build calibration runner**
   - Input: full-precision MLX model (bf16/f16) + calibration dataset
   - Process: forward pass through the model, recording activation magnitudes per layer and per weight block (block size = 32 or 64 weights, matching MLX quantization granularity)
   - Output: importance matrix (imatrix) — a JSON/safetensors file mapping every weight block to its importance score
   - File: `mxq/calibrate.py`

2. **Calibration dataset**
   - Curate a diverse dataset (~1000 samples):
     - Code (Python, Swift, JS, Rust)
     - Conversation (multi-turn chat)
     - Reasoning (math, logic, chain-of-thought)
     - Multilingual (EN, ZH, JA, KO, ES)
     - Long context (4K-32K token samples)
   - Store as: `mxq/datasets/calibration_v1.jsonl`
   - This dataset directly affects quantization quality — it's part of the secret sauce

3. **Activation collection**
   - Hook into every Linear layer in the model
   - For each forward pass, record:
     - Input activation magnitudes per channel
     - Output sensitivity: how much each weight block affects the output
   - Aggregate across all calibration samples (mean + variance)
   - File: `mxq/hooks.py`

4. **Importance scoring**
   - Two scoring methods, combined:
     - **Activation-aware (AWQ-style)**: importance = mean(activations) * weight_magnitude
       - Weights that process large activations are more important
     - **Sensitivity-based**: perturb each block, measure output KL-divergence
       - Weights where small changes cause large output shifts get more bits
   - Final score: weighted combination of both methods
   - File: `mxq/score.py`

### Deliverables
- `mxq calibrate --model [path] --dataset [path] --output [imatrix.safetensors]`
- Produces an importance matrix file that Phase 2 consumes

---

## Phase 2: Bit Allocation and Quantization (Week 3-4)

### Goal
Given an importance matrix and a target average bit width, decide how many bits each weight block gets, then quantize.

### Tasks

1. **Bit allocation algorithm**
   - Input: importance matrix + target average bits (e.g., 2.5, 3.0, 3.5, 4.0)
   - Constraint: total bits must average to target across all weights
   - Available bit widths per block: 2, 3, 4, 5, 6, 8
   - Algorithm:
     - Sort all blocks by importance score (ascending)
     - Start with all blocks at minimum bits (2)
     - While average_bits is less than target: upgrade the most important under-allocated block by 1 bit
   - Refinements:
     - **Layer-type priors**: attention Q/K/V start at 4-bit minimum, lm_head at 6-bit minimum
     - **First/last layer protection**: first 2 and last 2 transformer layers get +1 bit bonus
     - **Embedding protection**: embedding layer gets 4-bit minimum
   - File: `mxq/allocate.py`

2. **Per-block quantization**
   - For each block, quantize to its allocated bit width:
     - 2-bit: 4 values per byte, absmax scaling per block
     - 3-bit: packed 3-bit with per-block scale + zero-point
     - 4-bit: standard MLX 4-bit (compatible path)
     - 5-bit: packed 5-bit with per-block scale
     - 6-bit: packed 6-bit with per-block scale
     - 8-bit: standard MLX 8-bit (compatible path)
   - Each block stores: quantized weights + scale + zero_point + bit_width
   - File: `mxq/quantize.py`

3. **Quality validation**
   - After quantization, run perplexity evaluation on a held-out dataset
   - Compare against:
     - Full precision (bf16) — baseline
     - Uniform 4-bit MLX — current standard
     - Uniform 2-bit MLX — worst case
   - Report: perplexity delta, per-layer error distribution
   - File: `mxq/evaluate.py`

### Bit allocation profiles (presets)

| Profile | Avg bits | Target use case |
|---------|----------|----------------|
| MXQ-2 | 2.0-2.4 | Maximum compression, 70B+ on 32GB |
| MXQ-2.5 | 2.5 | Sweet spot — matches uniform 4-bit quality |
| MXQ-3 | 3.0 | High quality, significant compression |
| MXQ-4 | 4.0 | Best quality with smart allocation |
| MXQ-6 | 6.0 | Near-lossless with minor savings |

### Deliverables
- `mxq quantize --model [path] --imatrix [path] --bits 2.5 --output [model.mxq]`
- Produces a quantized model in MXQ format

---

## Phase 3: MXQ File Format (Week 4-5)

### Goal
Define the .mxq file format that stores mixed-precision weights with all metadata needed for inference.

### Format specification

```
model-name-MXQ-2.5bit/
  config.json                 # Standard HF model config
  tokenizer.json              # Standard HF tokenizer
  tokenizer_config.json
  special_tokens_map.json
  mxq_config.json             # MXQ-specific metadata (see below)
  mxq_imatrix.safetensors     # Importance matrix used (reproducibility)
  model-00001-of-00002.mxq.safetensors   # Quantized weights shard 1
  model-00002-of-00002.mxq.safetensors   # Quantized weights shard 2
  model.mxq.index.json        # Shard index
```

### mxq_config.json structure

```json
{
  "format": "mxq",
  "format_version": "1.0",
  "engine": "vmlx",
  "engine_min_version": "1.3.0",
  "quantization": {
    "method": "mxq-importance",
    "target_bits": 2.5,
    "actual_bits": 2.51,
    "block_size": 64,
    "calibration_dataset": "mxq-calib-v1",
    "scoring_method": "awq+sensitivity",
    "bit_widths_used": [2, 3, 4, 6, 8]
  },
  "layer_allocation": {
    "embed_tokens": 4,
    "lm_head": 6,
    "layers.0-1": "avg 4.2",
    "layers.2-29": "avg 2.3",
    "layers.30-31": "avg 4.0",
    "attention.q_proj": "avg 3.8",
    "attention.k_proj": "avg 3.5",
    "attention.v_proj": "avg 3.2",
    "mlp.gate_proj": "avg 2.1",
    "mlp.up_proj": "avg 2.1",
    "mlp.down_proj": "avg 2.3"
  },
  "source_model": {
    "name": "Qwen/Qwen3.5-72B",
    "dtype": "bfloat16",
    "parameters": "72B"
  },
  "quality_metrics": {
    "perplexity_bf16": 5.21,
    "perplexity_mxq": 5.38,
    "perplexity_uniform_4bit": 5.42,
    "perplexity_uniform_2bit": 12.7
  }
}
```

### Weight storage in safetensors

Each weight tensor is stored with companion metadata tensors:
- `layers.N.self_attn.q_proj.weight` — quantized weight data (packed bits)
- `layers.N.self_attn.q_proj.weight_scales` — per-block scale factors (float16)
- `layers.N.self_attn.q_proj.weight_zeros` — per-block zero points (float16)
- `layers.N.self_attn.q_proj.weight_bits` — per-block bit width (uint8)

### Tasks
- Define format spec document: `mxq/FORMAT.md`
- Build MXQ writer: `mxq/format/writer.py`
- Build MXQ reader: `mxq/format/reader.py`
- Build MXQ inspector (CLI): `mxq inspect [model.mxq]` — shows bit allocation, quality metrics, layer breakdown

---

## Phase 4: Metal Dequantization Kernels (Week 5-7)

### Goal
Write high-performance Metal GPU kernels that dequantize MXQ weights during inference. This is the hardest and most critical phase.

### Why custom kernels

MLX's built-in quantization only handles uniform bit widths (all 4-bit or all 8-bit). MXQ needs a kernel that:
1. Reads the bit_width for each block
2. Unpacks the correct number of bits
3. Applies the correct scale and zero point
4. Outputs float16/bfloat16 for matrix multiplication
5. Does all of this at full Metal throughput

### Kernel architecture (pseudocode)

```
kernel mxq_dequantize:
  inputs: packed_weights, scales, zeros, bit_widths
  output: dequantized float16 weights

  block_idx = thread_id / BLOCK_SIZE
  in_block = thread_id % BLOCK_SIZE
  bits = bit_widths[block_idx]

  // Calculate byte offset based on variable bit packing
  bit_offset = block_start_bits[block_idx] + in_block * bits
  byte_offset = bit_offset / 8
  bit_shift = bit_offset % 8

  // Extract value (handles cross-byte boundaries)
  mask = (1 << bits) - 1
  raw = (read_uint16(packed_weights + byte_offset) >> bit_shift) & mask

  // Dequantize
  output[thread_id] = float16(raw) * scales[block_idx] + zeros[block_idx]
```

### Kernels to implement

1. **mxq_dequant_matmul** — fused dequantize + matrix multiply (most important, used for every linear layer)
2. **mxq_dequant_standalone** — dequantize to float16 buffer (for debugging / non-fused paths)
3. **mxq_dequant_batched** — batched version for continuous batching compatibility
4. **mxq_dequant_kv** — variant that works with the KV cache quantization path

### Performance targets
- Dequant throughput: must not bottleneck generation speed
- Target: less than 5% overhead vs uniform 4-bit MLX quantization
- Memory bandwidth is the constraint — design for coalesced reads

### Tasks
- `mxq/metal/mxq_dequant.metal` — core Metal shaders
- `mxq/metal/mxq_matmul.metal` — fused dequant+matmul kernel
- `mxq/kernels.py` — Python bindings via MLX custom ops
- Benchmark suite: `mxq/bench/kernel_bench.py`

---

## Phase 5: vMLX Engine Integration (Week 7-8)

### Goal
Make vMLX Engine natively load and run MXQ models with zero user configuration.

### Tasks

1. **MXQ model loader**
   - Detect MXQ format via `mxq_config.json` presence
   - Load mixed-precision weights using MXQ reader
   - Initialize Metal dequant kernels for each layer based on its bit width
   - Register MXQ model in the model architecture auto-detection (50+ architectures)
   - File: `engine/mxq_loader.py`

2. **Integration with existing caching stack**
   - MXQ models must work with all 5 caching layers:
     - Prefix cache: operates on activations, not weights — no changes needed
     - Paged KV cache: same — no changes needed
     - KV cache quantization: independent of weight quantization — no changes needed
     - Continuous batching: uses batched dequant kernel
     - Disk cache: same — no changes needed
   - MXQ models must work with:
     - Tool calling (14 parsers): operates at token level — no changes needed
     - Reasoning parsers: same — no changes needed
     - Speculative decoding: needs testing with MXQ draft model
     - VLM support: needs MXQ quantization of vision encoder

3. **API surface**
   - No API changes needed — MXQ models serve the same way as any other model
   - New info in model response: quantization type "mxq-2.5bit"
   - MXQ metadata visible in admin dashboard

4. **MLX Studio UI**
   - Model browser shows MXQ badge on MXQ models
   - Settings panel shows bit allocation breakdown
   - Quality metrics displayed (perplexity vs baseline)

---

## Phase 6: Quantization CLI Tool (Week 8-9)

### Goal
Ship a CLI tool that lets power users quantize their own models to MXQ format.

### Commands

```bash
# Step 1: Calibrate (generates importance matrix)
mxq calibrate \
  --model mlx-community/Qwen3.5-72B-bf16 \
  --dataset mxq-calib-v1 \
  --output ./imatrix.safetensors

# Step 2: Quantize (produces MXQ model)
mxq quantize \
  --model mlx-community/Qwen3.5-72B-bf16 \
  --imatrix ./imatrix.safetensors \
  --bits 2.5 \
  --output ./Qwen3.5-72B-MXQ-2.5bit

# Step 3: Evaluate (quality check)
mxq evaluate \
  --model ./Qwen3.5-72B-MXQ-2.5bit \
  --baseline mlx-community/Qwen3.5-72B-bf16 \
  --dataset wikitext

# Step 4: Inspect (show bit allocation)
mxq inspect ./Qwen3.5-72B-MXQ-2.5bit

# One-shot (calibrate + quantize + evaluate)
mxq convert \
  --model mlx-community/Qwen3.5-72B-bf16 \
  --bits 2.5 \
  --output ./Qwen3.5-72B-MXQ-2.5bit \
  --evaluate
```

### Package
- PyPI package: `pip install mxq`
- Requires: mlx, mlx-lm, safetensors, numpy
- CLI entry point: `mxq`

---

## Phase 7: Model Publishing and Validation (Week 9-10)

### Goal
Quantize flagship models, validate quality, publish on HuggingFace.

### Priority models to quantize

| Model | Params | MXQ-2.5 size | MXQ-3 size | Notes |
|-------|--------|-------------|-----------|-------|
| Qwen3.5-72B | 72B | ~22 GB | ~27 GB | Flagship — runs on 32GB Mac |
| Llama-4-Scout-109B | 109B | ~34 GB | ~41 GB | Fits on 64GB Mac |
| DeepSeek-V3-671B | 671B | ~210 GB | ~252 GB | Fits on 512GB Mac Studio |
| Nemotron-H-120B | 120B | ~38 GB | ~45 GB | Hybrid SSM — vMLX exclusive |
| Qwen3.5-VL-72B | 72B | ~22 GB | ~27 GB | Vision + MXQ |
| Gemma-3-27B | 27B | ~8.5 GB | ~10 GB | Runs on 16GB Mac |
| Phi-4-14B | 14B | ~4.4 GB | ~5.3 GB | Runs on 8GB Mac |

### Naming convention on HuggingFace

```
dealignai/Qwen3.5-72B-MXQ-2.5bit
dealignai/Qwen3.5-72B-MXQ-3bit
dealignai/Qwen3.5-72B-MXQ-4bit
dealignai/Nemotron-H-120B-MXQ-2.5bit-CRACK  (abliterated + MXQ)
```

### Quality validation checklist

For each model, before publishing:
- [ ] Perplexity within 5% of uniform 4-bit (for MXQ-2.5)
- [ ] Perplexity within 2% of uniform 4-bit (for MXQ-3)
- [ ] No degenerate outputs on 100 test prompts
- [ ] Tool calling works correctly (test with 14 parsers)
- [ ] Reasoning works correctly (test with 4 reasoning parsers)
- [ ] Code generation quality maintained (HumanEval benchmark)
- [ ] Multilingual quality maintained (test EN/ZH/JA/KO/ES)
- [ ] Long context coherence at 32K+ tokens
- [ ] Works with all 5 caching layers in vMLX
- [ ] Works with speculative decoding

---

## File Structure

```
mxq/
  MXQ-PLAN.md                 # This document
  FORMAT.md                   # Format specification
  setup.py
  pyproject.toml
  README.md
  mxq/
    __init__.py
    __main__.py                # CLI entry point
    calibrate.py               # Phase 1: calibration engine
    hooks.py                   # Phase 1: activation hooks
    score.py                   # Phase 1: importance scoring
    allocate.py                # Phase 2: bit allocation
    quantize.py                # Phase 2: per-block quantization
    evaluate.py                # Phase 2: quality evaluation
    format/
      __init__.py
      spec.py                  # Format constants and version
      writer.py                # Phase 3: write .mxq files
      reader.py                # Phase 3: read .mxq files
      inspector.py             # Phase 3: inspect command
    metal/
      mxq_dequant.metal        # Phase 4: core dequant kernel
      mxq_matmul.metal         # Phase 4: fused dequant+matmul
      build.py                 # Phase 4: kernel compilation
    kernels.py                 # Phase 4: Python bindings
    datasets/
      calibration_v1.jsonl
    bench/
      kernel_bench.py
      quality_bench.py
  engine_integration/
    mxq_loader.py              # Phase 5: vMLX Engine loader
    mxq_model.py               # Phase 5: model wrapper
```

---

## Competitive Positioning

### What to say on vmlx.net and mlx.studio

> **MXQ — Importance-Aware Quantization for Apple Silicon**
> The only quantization format built specifically for MLX and Apple Silicon.
> Run 70B models on 32GB Macs at quality levels that match standard 4-bit.
> MXQ models are exclusive to vMLX Engine — no other app can load them.

### How MXQ compares to other quantization methods

| | MXQ | GGUF K-quant | AWQ | GPTQ | EXL2 |
|---|-----|-------------|-----|------|------|
| Target hardware | Apple Silicon | CPU/CUDA | CUDA | CUDA | CUDA |
| Framework | MLX | llama.cpp | Various | Various | ExLlamaV2 |
| Mixed precision | Yes per-block | Yes per-layer | No uniform | No uniform | Yes per-block |
| Importance-aware | Yes calibrated | Yes imatrix | Yes activation | Yes Hessian | Yes calibrated |
| 2-bit usable | Yes | Mediocre | No | No | Yes |
| Metal optimized | Yes native | No | No | No | No |
| Unified memory | Yes designed for | No | No | No | No |

### The moat

1. **MXQ models only run on vMLX** — engine-exclusive format
2. **dealignai becomes the HuggingFace destination** for MXQ models
3. **Nobody can copy it quickly** — requires Metal kernel expertise + calibration infrastructure
4. **oMLX/LM Studio/Inferencer are stuck with uniform quantization** — their models are bigger and worse quality at the same bit width
5. **Compounds with existing advantages** — MXQ + 5-layer caching + hybrid SSM + agentic tools = unmatched

---

## Dependencies

- Python 3.11+
- mlx >= 0.22
- mlx-lm >= 0.20
- safetensors
- numpy
- tqdm
- Xcode Command Line Tools (for Metal shader compilation)
- macOS 26+ (Tahoe) with Metal 4

---

## Timeline

| Phase | Description | Weeks | Depends on |
|-------|------------|-------|------------|
| 1 | Calibration engine | 1-2 | — |
| 2 | Bit allocation and quantization | 3-4 | Phase 1 |
| 3 | MXQ file format | 4-5 | Phase 2 |
| 4 | Metal dequant kernels | 5-7 | Phase 3 |
| 5 | vMLX Engine integration | 7-8 | Phase 4 |
| 6 | CLI tool and packaging | 8-9 | Phase 5 |
| 7 | Model publishing | 9-10 | Phase 6 |

**Total: ~10 weeks to first MXQ model on HuggingFace.**

---

## Success Criteria

1. MXQ-2.5bit perplexity within 5% of uniform 4-bit on Wikitext
2. MXQ-3bit perplexity within 2% of uniform 4-bit
3. 70B model runs on 32GB Mac via MXQ-2.5
4. Metal dequant kernel adds less than 5% overhead vs uniform 4-bit
5. Full compatibility with vMLX 5-layer caching stack
6. At least 5 flagship models published on dealignai
7. oMLX/LM Studio/Inferencer cannot load MXQ models
