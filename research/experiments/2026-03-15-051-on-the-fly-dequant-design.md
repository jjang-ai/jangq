# Experiment 051: On-the-Fly Dequantization Design

**Date**: 2026-03-15
**Author**: Jinho Jang (eric@jangq.ai)
**Status**: DESIGN PHASE

## The Problem

JANG dequant-at-load expands models to full float16 size in memory:
- MiniMax 230B at 2-bit: 60 GB on disk → 460 GB float16 in RAM (doesn't fit)
- The entire point of JANG is to run large models in small memory, like GGUF

## Constraint: MLX Per-Tensor Quant Speed Regression

From CRACK research (Finding #75):
- Uniform Q_BITS for all tensors: ~38-50 tok/s
- Per-tensor quantization config: ~9.4 tok/s (60% regression)
- Cause: mlx_lm uses slow `class_predicate` loading path when per-tensor overrides exist

This means we CANNOT use different bit widths per layer through mlx_lm's standard path
without massive speed loss.

## Options

### Option A: Uniform bits, smarter calibration
- All layers at same bit width (e.g., 2-bit)
- Use MSE-optimal clipping on CRITICAL tensors, RTN on COMPRESS
- Same storage format as MLX native quantization
- Pro: Full mlx_lm speed, simple implementation
- Con: No actual bit-width differentiation — loses JANG's core advantage

### Option B: Two-tier MLX quantization
- CRITICAL/IMPORTANT at 4-bit, COMPRESS at 2-bit
- Use `nn.quantize()` with class_predicate for just 2 bit widths
- Test if 2-tier causes the same speed regression as per-tensor
- Pro: Retains JANG's core advantage (protect attention with more bits)
- Con: May still trigger slow path; needs testing

### Option C: Custom MLX kernel
- Write a Metal kernel that handles variable bit-width per group
- Replace `quantized_matmul` for JANG layers
- Pro: Full control, no compromises
- Con: Significant engineering effort, maintenance burden

### Option D: Repack to MLX native format per-tier
- Each tensor gets repacked to MLX's uint32 format
- Use group_size=64 (matches JANG block_size)
- Scale/bias conversion: mlx_bias = -jang_scale * jang_zero
- Two QuantizedLinear configs: one for 2-bit layers, one for 6/8-bit layers
- Test if `quantized_matmul` handles mixed configs without regression
- Pro: Reuses MLX Metal kernels, model stays quantized
- Con: Need to verify speed with mixed configs

## Key Insight from CRACK Research

The speed regression happens because of `class_predicate` in the LOADING path, not in
the inference path. If we can load the model correctly (bypassing class_predicate) and
set up QuantizedLinear layers directly, the actual `quantized_matmul` calls may be fast
regardless of per-tensor bit differences.

## Recommended Approach: Option D

1. Repack JANG tensors → MLX uint32 format at load time (one-time, fast)
2. Create QuantizedLinear layers with per-tensor bits/group_size
3. Load directly into the model (bypass nn.quantize() and class_predicate)
4. Test speed — if quantized_matmul is fast regardless of per-layer bits, we win

## JANG → MLX Format Mapping (Verified)

```
JANG qweight (uint8 LSB) → view as uint32 (same bytes, different dtype)
JANG scale → mlx scale (same)
JANG zero → mlx bias = -scale * zero
JANG block_size=64 → mlx group_size=64
```

Verified: `mx.dequantize()` produces identical results with repacked data.

## Architecture Notes from CRACK (Qwen 3.5)

- All Qwen 3.5: `full_attention_interval=4` (FA at L3, L7, L11, ...)
- 122B: 48 layers, 12 FA layers, 36 GDN layers, 256 experts (top-8)
- 397B: 60 layers, 15 FA layers, 45 GDN layers, 512 experts (top-10)
- group_size=64 is standard for MLX quantization
- head_dim=256 for all Qwen 3.5 models
