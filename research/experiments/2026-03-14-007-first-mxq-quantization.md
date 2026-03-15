# Experiment 007: First MXQ Model Quantization

**Date**: 2026-03-14
**Author**: Eric Jang (eric@vmlx.net)
**Status**: PASS — first MXQ model ever created

## Setup

- **Source model**: Qwen/Qwen2.5-0.5B (494M parameters, bfloat16)
- **Target**: MXQ-2.5bit
- **Block size**: 64
- **Calibration**: weight-only (no forward pass)
- **Quantization**: RTN (round-to-nearest), vectorized
- **Hardware**: Apple Silicon Mac (Python 3.14.2)

## Results

| Metric | Value |
|--------|-------|
| Total weights | 493,961,216 |
| Total blocks | 7,718,144 |
| Target bits | 2.5 |
| Actual bits | 2.76 |
| Output size | 0.16 GB |
| Source size | ~1 GB (bf16) |
| Compression ratio | ~6.3x |
| Quantization time | 74.2 seconds |

### Bit Allocation Breakdown

| Bit Width | Blocks | Percentage |
|-----------|--------|------------|
| 2-bit | 4,085,760 | 52.9% |
| 3-bit | 1,390,592 | 18.0% |
| 4-bit | 2,241,792 | 29.0% |

### Architecture Detection

- Model type: qwen2 (standard transformer)
- Attention: GQA (Grouped Query Attention)
- Correctly identified 169 weight tensors for quantization
- Correctly skipped: RMSNorm weights, bias terms

### Layer Type Allocation

- embed_tokens: 4-bit minimum enforced
- lm_head: not found as separate tensor (weight-tied with embed)
- attention Q/K: 3-bit minimum enforced
- attention V/O: 3-bit minimum enforced
- MLP gate/up: 2-bit allowed (most compressible)
- MLP down: 2-bit allowed

## Observations

1. **Actual bits (2.76) exceeded target (2.5)**: This is because layer-type priors
   enforce minimum bit floors (attention=3, embed=4). With 29% of blocks at 4-bit
   and 18% at 3-bit, the average is pulled up. The greedy allocator respected the
   target but layer priors dominated.

2. **bfloat16 handling**: Required custom raw byte reading — numpy doesn't support
   bfloat16. Implemented manual conversion: read uint16 → shift left 16 bits →
   reinterpret as float32. This worked correctly.

3. **Vectorized quantization**: The initial per-block Python loop was too slow
   (7.7M blocks × 100-point MSE search = stuck). Rewrote to group blocks by bit
   width and quantize entire groups with numpy broadcasting. Result: 169 tensors
   in 64 seconds.

4. **Format roundtrip**: The output .mxq model loads correctly via the reader
   and produces a valid summary via `mxq-tools inspect`.

## Issues Found

1. Packing loop is still per-block (not vectorized) — this is the remaining bottleneck
2. Target bits overshoot due to layer priors — allocator needs budget-aware prior application
3. No quality evaluation yet (no dequant → perplexity comparison)

## Next Steps

- Run dequantization and compare reconstruction error vs uniform quantization
- Run activation-aware calibration for better importance scoring
- Quantize Qwen2.5-3B for a more realistic test
- Begin Metal kernel development
