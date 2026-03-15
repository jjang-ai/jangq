# Experiment 027: Full Quality Baselines — MXQ vs MLX on Qwen2.5-3B

**Date**: 2026-03-14
**Author**: Eric Jang (eric@vmlx.net)
**Status**: BASELINES ESTABLISHED

## Quality Comparison Table

| Method | Bits (eff) | Top Token | Correct? | Top Logit | vs Reference |
|--------|-----------|-----------|----------|-----------|-------------|
| bf16 reference | 16 | 'The' | YES | 15.38 | baseline |
| **MXQ 8-bit** | **8** | **'The'** | **YES** | **15.38** | **matches** |
| MLX uniform 4-bit | 4.5 | 'The' | YES | 11.31 | MSE 11.31 |
| MXQ 4-bit (RTN) | 4.0 | 'How' | NO | 12.04 | broken |
| MLX uniform 2-bit | 2.5 | 'izu' | NO | 8.56 | MSE 17.57 |

## The Challenge

MXQ at 2.5-bit average must beat MLX uniform 4-bit (MSE < 11.31).

Current status:
- MXQ 4-bit RTN: WORSE than MLX 4-bit (wrong token)
- MXQ 8-bit: matches reference perfectly

## Why MXQ 4-bit RTN Fails While MLX 4-bit Works

1. **MLX uses effective 4.5 bits** (group_size=64 adds scale+bias overhead).
   MXQ uses exactly 4.0 bits average with some blocks at 2-3 bit.
   Those low-bit blocks degrade quality disproportionately.

2. **MLX's quantized matmul is fused and optimized**. Our GEMV dequants
   to float16 per element — more truncation error in the accumulation.

3. **AWQ scaling not yet applied at runtime**. The AWQ-quantized model
   can't be loaded by the runtime (needs AWQ inverse in the kernel).

## Path Forward

1. **AWQ integration in runtime**: modify dequant kernel to apply
   AWQ inverse after dequantization. This should improve 4-bit quality
   by ~14% (verified on single layer).

2. **Cross-layer variable bit allocation**: give MLP layers 2-3 bits
   and attention 4-6 bits. Keep average at 4.0 but distribute smartly.

3. **Better calibration**: more diverse data, longer texts.

4. **For the paper**: focus on larger models (7B+) where the quality
   gap between 2.5-bit and 4-bit is much smaller because larger
   models have more redundancy.

## Key Metric for Paper

The paper claim "MXQ-2.5bit matches uniform 4-bit" requires:
- Perplexity measurement on a standard benchmark (wikitext-2)
- On a model of at least 7B parameters
- With GPTQ + AWQ + importance-based allocation

This is the standard that EXL2 meets and what MXQ must also meet.
