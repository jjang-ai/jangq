# JANG — Jang Adaptive N-bit Grading

> The GGUF equivalent for MLX. Open format + tools + inference via MLX native kernels.
> Created by Jinho Jang (eric@jangq.ai)

---

## What JANG Is

JANG is three things:

1. **A quantization method** — tier-based mixed-precision that classifies tensors by sensitivity and assigns bits per tier
2. **A file format** — `.jang.safetensors` files with per-tensor quantized weights and metadata
3. **An inference path** — repacks JANG weights into MLX native quantized format for full-speed `quantized_matmul` and `gather_qmm`

JANG produces models that stay **quantized in GPU memory** (like GGUF) and dequantize on-the-fly during the forward pass using MLX's Metal kernels. No float16 expansion.

**Format**: `.jang.safetensors` (standard safetensors with JANG metadata)
**Tools**: Python package (`pip install jang-tools`) for quantization
**Inference**: MLX native — works with vMLX, MLX Studio, any OpenAI-compatible MLX server
**Distribution**: HuggingFace org [JANGQ-AI](https://huggingface.co/JANGQ-AI)
**Source**: GitHub [jjang-ai/jangq](https://github.com/jjang-ai/jangq)

### Why it matters

JANG excels on **MoE/hybrid models** where expert MLP is 94-98% of parameters. Protecting the other 2-6% (attention) at 8-bit costs almost nothing but makes the difference between coherent and broken output.

| Method | 122B MoE Size | GPU | MMLU |
|--------|--------------|-----|------|
| **JANG_1L (2.24b)** | **51 GB** | **46 GB** | **73.0%** |
| MLX mixed_2_6 | 44 GB | 45 GB | 46.0% |
| MLX uniform 2-bit | 36 GB | 36 GB | 56.0% |

### The vision

Make high-intelligence MoE models accessible to everyone with Apple Silicon. A 122B model on a 64GB Mac. A 35B model on a 16GB MacBook. Not dumbed-down — coherent output through smart bit allocation.

---

## Architecture

### 3-Tier System

Every weight tensor is classified into a sensitivity tier:

```
CRITICAL  (full softmax attention, output head, MLA)  → 6-8 bit
IMPORTANT (embeddings, routers, linear attention)      → 4-8 bit
COMPRESS  (MLP, MoE experts, FFN)                      → 2-4 bit
```

### Profile Strategy

At 3-bit+, IMPORTANT = COMPRESS (only CRITICAL boosted). This gives ~2% overhead on MoE.
At 2-bit, IMPORTANT also gets protection (linear attention breaks at 2-bit).

### Quantization Backend

Uses `mx.quantize()` directly from MLX for the quantization step. This ensures JANG-quantized tensors are byte-identical to MLX on the COMPRESS tier. The per-tensor MSE matches MLX's native quantization.

Previously used custom RTN which was 0.36% worse per tensor, compounding to ~4.5% MMLU gap across thousands of tensors.

### Inference Path

JANG weights are repacked from JANG uint8 → MLX uint32 format at load time, then use:
- `quantized_matmul` for standard linear layers
- `gather_qmm` for MoE expert layers via `QuantizedSwitchLinear`

Models stay quantized in GPU memory. Memory footprint ≈ file size on disk.

---

## Profiles

```python
JANG_PROFILES = {
    # 2-bit: IMPORTANT needs protection (linear attention breaks)
    "JANG_2S": (6, 4, 2),   # Tightest
    "JANG_2M": (8, 4, 2),   # Balanced
    "JANG_2L": (8, 6, 2),   # Best quality (proven: 73% MMLU on 122B)
    "JANG_1L": (8, 8, 2),   # Maximum quality (proven: 6/6 free-form)

    # 3-bit+: IMPORTANT = COMPRESS, only CRITICAL boosted
    "JANG_3S": (6, 3, 3),   # Attention 6-bit
    "JANG_3M": (8, 3, 3),   # Attention 8-bit, ~2% overhead
    "JANG_3L": (8, 4, 3),   # Attention 8-bit, embeddings 4-bit

    # 4-bit: THE STANDARD for MoE models
    "JANG_4S": (6, 4, 4),   # Attention 6-bit
    "JANG_4M": (8, 4, 4),   # Attention 8-bit, rest 4-bit (~2% overhead on MoE)
    "JANG_4L": (8, 6, 4),   # Also boost embeddings

    # 6-bit: near-lossless
    "JANG_6M": (8, 6, 6),
}

BIT_TO_PROFILE = {
    1: "JANG_1L",   # Extreme 2-bit + max protection
    2: "JANG_2L",   # 2-bit + protected attention
    3: "JANG_3M",   # 3-bit + 8-bit attention
    4: "JANG_4M",   # 4-bit + 8-bit attention — THE DEFAULT
    5: "JANG_6M",   # Near-lossless
    6: "JANG_6M",
}
```

### Why JANG_4M is the standard

JANG_4M (8, 4, 4) gives the same 4-bit MLP as MLX uniform, but attention at 8-bit. On MoE models where attention is only 2-6% of params, this adds ~2% overhead for significantly better quality. The quantization uses `mx.quantize()` so COMPRESS tensors are byte-identical to MLX.

---

## Format: v1.1

### Directory Layout

```
model-name-JANG_2L/
  config.json                              # HuggingFace config (unmodified)
  tokenizer.json                           # HuggingFace tokenizer (unmodified)
  tokenizer_config.json
  special_tokens_map.json
  jang_config.json                         # JANG metadata (profile, bits, source model)
  model-00001-of-NNNNN.jang.safetensors    # Quantized weight shards
  model.jang.index.json                    # Shard index
```

### Per-Tensor Storage (v1.1)

Each quantized tensor has these companions:

```
{name}.qweight    # uint8 — packed quantized data
{name}.scales     # float16 — per-block scale factors
{name}.zeros      # float16 — per-block zero points
{name}.bits       # uint8[1] — single value: bit width for entire tensor
{name}.shape      # int64 — original weight shape
```

**v1.1 change**: Eliminated per-block `bit_map` and `block_offsets`. Since JANG uses tier-based allocation (every block in a tensor gets the same bits), a single `.bits` value per tensor is sufficient. This reduced format overhead by 18%.

### Dequantization

```
bits = tensor.bits[0]           # single value for entire tensor
scale = scales[block_idx]
zero = zeros[block_idx]
raw = unpack(qweight, block_idx, bits)
dequantized = (raw - zero) * scale
```

See [FORMAT.md](FORMAT.md) for complete specification.

---

## Supported Architectures

| Architecture | Examples | Key Tensors |
|-------------|----------|-------------|
| Dense Transformer | Llama, Qwen, Gemma, Phi, Mistral | q/k/v/o_proj → CRITICAL |
| Mixture of Experts | Mixtral, Qwen3.5 MoE, DeepSeek, MiniMax | expert MLP → COMPRESS, routers → IMPORTANT |
| Hybrid SSM + Attention | Jamba, Zamba, Nemotron-H | SSM state → CRITICAL, SSM proj → COMPRESS |
| Linear Attention | Qwen3.5 GatedDeltaNet | in_proj_* → IMPORTANT (breaks at 2-bit) |
| Multi-head Latent Attention | DeepSeek-V3/R1 | kv_a/b_proj → CRITICAL |
| Vision-Language | Qwen-VL, LLaVA, Pixtral | visual.merger → IMPORTANT |
| Pure SSM | Mamba, Mamba2 | A_log, D → CRITICAL |
| FP8 Source Models | MiniMax-M2.5, DeepSeek FP8 | FP8 E4M3 → float32 → quantize |

---

## MoE vs Dense Finding

**JANG is designed for MoE/hybrid models.** On dense models, MLX uniform quantization is recommended.

- **MoE**: expert MLP is 94-98% of params → protecting attention costs ~2% overhead
- **Dense**: attention is ~12% of params → protecting it costs ~6-40% overhead → not worth it

Results:
- 122B MoE: JANG 73% vs MLX 46% MMLU (+27 points)
- 35B MoE: JANG 5/6 vs MLX 0/6 free-form
- 27B Dense: JANG 32.5% vs MLX 78% MMLU (JANG loses on dense)

---

## Project Structure

```
jang/
  JANG-PLAN.md              # This document
  FORMAT.md                  # Public format specification
  LICENSE

  jang-tools/                # Python: quantization toolkit
    pyproject.toml
    jang_tools/
      __init__.py
      __main__.py            # CLI: jang convert
      convert.py             # End-to-end pipeline
      allocate.py            # Tier-based bit allocation
      quantize.py            # Per-tensor quantization (uses mx.quantize)
      pack.py                # Bit packing (vectorized)
      calibrate.py           # Weight-based importance scoring
      architectures.py       # Architecture detection (all families)
      fp8.py                 # FP8 E4M3 dequantization
      loader.py              # MLX inference loader (repack + load)
      format/
        spec.py              # Format constants (v1.1)
        writer.py            # Write .jang files
        reader.py            # Read .jang files
    tests/

  research/                  # Research notes, experiments, results
    JANG-RESULTS.md          # Empirical evidence
    experiments/             # 48+ documented experiments
```

---

## Key Technical Decisions

1. **MLX native kernels over custom Metal**: MLX's `quantized_matmul` and `gather_qmm` are heavily optimized. Writing custom kernels would be slower and harder to maintain.

2. **Tier-based over calibration-based allocation**: Calibration (AWQ, GPTQ, Hessian) adds complexity and requires forward passes. Tier classification by tensor name is simpler, works on any architecture, and produces equivalent or better results on MoE models.

3. **mx.quantize() over custom RTN**: MLX's native quantization is slightly better per-tensor (0.36% MSE), which compounds across thousands of tensors. Using it directly makes JANG COMPRESS tensors byte-identical to MLX uniform.

4. **Single .bits per tensor over per-block bit_map**: Tier-based allocation gives every block in a tensor the same bits. Eliminating per-block metadata saves 18% overhead.

5. **MoE positioning over universal**: JANG's overhead is negligible on MoE (2-6% attention) but significant on dense (12%+ attention). Honest positioning prevents bad user experience.

---

## CLI

```bash
# Simple: pick 1-8 for target bits
jang convert path/to/model -p 2

# Specific profile
jang convert path/to/model -p JANG_1L

# From HuggingFace
jang convert Qwen/Qwen3.5-35B-A3B -p 2
```

---

## Results Summary

| Model | Profile | Score | vs MLX |
|-------|---------|-------|--------|
| Qwen3.5-122B-A10B | JANG_1L | 73% MMLU, 6/6 free-form | +27 MMLU over mixed_2_6 |
| Qwen3.5-35B-A3B | JANG_2L | 5/6 free-form | MLX 0/6 |
| 7 small models (1-7B) | various | 65 wins, 0 losses | at degradation boundary |

---

## Dependencies

- Python 3.11+
- numpy, safetensors, tqdm, huggingface_hub
- mlx, mlx-lm (for inference and mx.quantize backend)
- torch (optional, for FP8 source models)
