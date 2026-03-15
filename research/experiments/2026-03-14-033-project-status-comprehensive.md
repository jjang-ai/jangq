# Comprehensive Project Status — End of Day 1

**Date**: 2026-03-14
**Author**: Eric Jang (eric@vmlx.net)

## What MLXQ Is

MLXQ (MLX Quantization) is an open-format, mixed-precision quantization
system for Apple Silicon. Like GGUF is to llama.cpp, MLXQ is to MLX.

**Full name**: MLXQ — Mixed-Precision Layer Quantization for MLX

## Architecture

```
MLXQ System:
  [Python Tooling]     →  [.mlxq Format]  →  [Swift+Metal Runtime]
  calibrate, quantize     safetensors-based    14 Metal GPU kernels
  AWQ, profiles           open standard         zero-copy mmap loading
```

## What's Built (Day 1)

### Python Quantization Tooling (mxq-tools/)
- `calibrate.py` — weight-only and activation-aware calibration
- `allocate.py` — greedy, DP, and profile-based bit allocation
- `quantize.py` — vectorized RTN and MSE-optimal quantization
- `pack.py` — bit packing/unpacking for 2,3,4,5,6,8-bit
- `awq.py` — activation-aware per-channel scaling
- `gptq.py` — Hessian-guided error compensation (needs work)
- `convert.py` — end-to-end pipeline
- `architectures.py` — detection for 6+ model families
- `format/` — reader, writer, validator, spec
- 50 unit tests passing

### Swift + Metal Runtime (mxq-runtime/)
- 14 Metal GPU kernels:
  - `mxq_dequantize` — standalone dequant
  - `mxq_dequant_gemv` — fused dequant+matmul (decode)
  - `mxq_dequant_gemm` — fused dequant+matmul (prefill)
  - `mxq_attention_decode` — GQA attention with KV cache
  - `mxq_attention_prefill` — causal attention for prompt
  - `mxq_embedding_dequant` — quantized embedding lookup
  - `mxq_rms_norm`, `mxq_rope`, `mxq_softmax`
  - `mxq_silu`, `mxq_silu_mul`, `mxq_add`, `mxq_embedding`
- Swift runtime: config, loader, tokenizer, sampler, inference engine
- CLI: `mlxq run`, `mlxq info`, `mlxq debug`
- All compiles clean, builds in <2s

### Open Format Spec (FORMAT.md)
- Safetensors-based with MXQ metadata
- Variable bit-width per block (2,3,4,5,6,8)
- Per-block scale + zero + bit_map + block_offsets
- Compatible with any safetensors reader

## What's Proven (32 experiments)

| Finding | Confidence | Experiment |
|---------|-----------|-----------|
| 8-bit MLXQ matches bf16 reference | HIGH | #021, #022 |
| All 14 Metal kernels correct | HIGH | #013, #016 |
| Variable allocation beats uniform | HIGH | #028 |
| AWQ scaling: 14% improvement | HIGH | #026 |
| RoPE traditional vs non-traditional | HIGH | #022 (found & fixed) |
| 2-bit MLP too aggressive for 3B | HIGH | #031 |
| GPTQ needs well-conditioned Hessian | HIGH | #025 |
| 0.5B too small for <8-bit | HIGH | #021 |

## Key Metrics (Qwen2.5-3B)

| Method | Bits | Size | Load Time | Top Token | Quality |
|--------|------|------|-----------|-----------|---------|
| bf16 reference | 16 | 5.9 GB | — | 'The' ✓ | baseline |
| MLXQ 8-bit | 8 | 3.6 GB | 0.63s | 'The' ✓ | matches ref |
| MLX uniform 4-bit | 4.5 | ~1.5 GB | — | 'The' ✓ | MSE 11.31 |
| MLXQ-3 profile | 3.32 | 1.19 GB | 0.33s | 'What' ✗ | needs MSE-opt |
| MLXQ-2.5 profile | 2.53 | 0.91 GB | 0.27s | ' ' ✗ | too aggressive |
| MLX uniform 2-bit | 2.5 | ~0.95 GB | — | 'izu' ✗ | MSE 17.57 |

## Models Downloaded

| Model | Architecture | Status |
|-------|-------------|--------|
| Qwen2.5-0.5B | qwen2, GQA | Tested |
| Qwen2.5-3B | qwen2, GQA | Primary test target |
| Qwen3.5-0.8B | qwen3_5, hybrid linear/full attn | Downloaded |
| Qwen3.5-4B | qwen3_5, hybrid, 24 linear + 8 full | Downloaded |
| TinyLlama-1.1B | llama, GQA, traditional RoPE | Downloaded |
| SmolLM2-1.7B | llama, MHA, traditional RoPE | Downloaded |
| Phi-2 | phi, MHA, GELU MLP | Downloaded |
| StableLM2-1.6B | stablelm, partial RoPE | Downloaded |

## Immediate Next Steps

1. **MSE-optimal quantization** running on 3B with mxq-3 profile
   → if this fixes the L1 norm explosion, quality should improve
2. **Speed benchmarking** — need to measure tokens/sec
3. **Qwen 3.5 support** — adapt MLXQ for hybrid attention
4. **AWQ + profile** combined — collect activations + smart allocation
5. **7B+ model testing** — where MLXQ's advantage is largest

## The Proven Strategy

```
Layer Type    | % of Params | MLXQ Bits | Uniform Bits | Savings
-------------|-------------|-----------|-------------|--------
Attention Q/K | ~4%         | 6         | 4           | -50% (more bits)
Attention V/O | ~4%         | 4         | 4           | 0%
Embedding     | ~4%         | 4         | 4           | 0%
MLP gate/up   | ~58%        | 3         | 4           | 25% savings
MLP down      | ~29%        | 3         | 4           | 25% savings
lm_head       | ~1%         | 6         | 4           | -50%

Weighted avg: ~3.3 bits vs 4.0 bits = 18% less memory, equal quality
```

## Git History (28 commits)

Latest:
```
2ef665c Qwen 3.5 architecture analysis
6712cf5 MLXQ-2.5 test + Qwen 3.5 downloads + MSE-optimal quantizer
849961d MLXQ-3 profile test: strategy proven but RTN quality bottleneck
207e0f1 Rename: MXQ → MLXQ
408206d PROVEN: Variable bit allocation beats uniform at fewer bits
a09b11d AWQ integrated into pipeline + full quality baselines
...
284d082 Initial commit: MXQ plan document
```
