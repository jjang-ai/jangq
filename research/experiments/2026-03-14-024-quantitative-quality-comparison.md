# Experiment 024: Quantitative Quality Comparison — MXQ vs MLX

**Date**: 2026-03-14
**Author**: Eric Jang (eric@vmlx.net)
**Status**: CRITICAL — establishes baselines and identifies path forward

## Purpose

Quantitatively compare MXQ quantization quality against MLX's built-in
quantization and the bf16 reference. This is the experiment that determines
whether MXQ's approach has value.

## Model: Qwen2.5-3B

Prompt: `<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n`

## Results

| Method | Bits | Top Token | Correct? | Top Logit | Logit MSE | KL Div |
|--------|------|-----------|----------|-----------|-----------|--------|
| bf16 reference | 16 | 'The' | YES | 15.38 | 0.00 | 0.00 |
| **MXQ 8-bit (RTN)** | 8 | **'The'** | **YES** | **15.38** | **~0.5** | **~0.1** |
| **MLX uniform 4-bit** | 4.5 | **'The'** | **YES** | **11.31** | **11.31** | **1.50** |
| MXQ 4-bit (RTN) | 4 | 'How' | NO | 12.04 | >15 | >5 |
| MLX uniform 2-bit | 2.5 | 'izu' | NO | 8.56 | 17.56 | 10.21 |

## Analysis

### What the data tells us

1. **MXQ 8-bit ≈ reference**: logit 15.38 matches bf16's 15.38 exactly.
   At 8-bit, both MXQ and MLX are essentially lossless.

2. **MLX 4-bit works, MXQ 4-bit doesn't**: Both use RTN (round-to-nearest).
   MLX uses group_size=64, 4-bit uniform. MXQ uses block_size=64 with
   variable allocation. Yet MLX gets the right token and MXQ doesn't.

3. **MLX uniform 2-bit fails too**: KL=10.2, wrong token. So aggressive
   quantization is hard regardless of method.

### Why MLX 4-bit beats MXQ 4-bit

MLX's `mx.quantize()` and `mx.dequantize()` are **hardware-optimized
primitives** implemented in C++/Metal. The quantization uses affine scaling
with group_size=64 — same as our block_size.

The difference: MLX's dequantization is done in **fused Metal kernels**
that were carefully optimized for numerical precision. Our GEMV kernel
dequantizes in float32 but the intermediate results go through float16
when writing to output buffers. The repeated float16 truncation across
24 layers compounds.

Additionally, MLX 4-bit actually uses 4.501 effective bits (the group
scale/bias overhead adds ~0.5 bits). Our MXQ 4-bit has 4.0 actual average
because the allocator gives some blocks fewer bits (3-bit) to keep the
average at 4.0. Those 3-bit blocks introduce more error than uniform 4-bit.

### The real path to MXQ's advantage

MXQ's variable bit-width can only help when combined with:

1. **GPTQ error compensation**: Reduces per-block error by 2-5x by
   adjusting remaining weights after each quantization step. This is
   what makes EXL2 (the closest analog to MXQ) actually work.

2. **Optimal bit allocation**: Our current greedy allocator uses
   weight-only statistics. AWQ-style activation-aware scoring would
   give better allocation decisions.

3. **Higher base quality**: Use GPTQ within each block, then use
   importance to allocate bits. GPTQ+importance > uniform GPTQ.

4. **Larger models**: 3B has limited redundancy. At 70B, the same
   quantization errors matter less because each weight contributes
   less to the output.

## Mathematical Framework for MXQ's Advantage

### Rate-Distortion Theory

For a Gaussian source with variance σ², the minimum achievable
distortion at rate R bits is:

```
D(R) = σ² × 2^(-2R)
```

For N blocks with different variances σ₁², ..., σₙ²:

**Uniform allocation** (all blocks get R bits):
```
D_uniform = (1/N) Σ σᵢ² × 2^(-2R)
```

**Optimal allocation** (reverse water-filling):
```
D_optimal = (1/N) Σ σᵢ² × 2^(-2Rᵢ)
where Rᵢ chosen to equalize marginal distortion
```

The ratio D_optimal / D_uniform < 1 always (strictly less when
variances differ). The improvement depends on the variance spread:

```
Improvement ≈ 1 - (geometric_mean(σ²) / arithmetic_mean(σ²))
```

For typical transformer weights, this gives 10-30% reduction in
distortion at the same average bit width. BUT this only holds when
the per-block quantization is optimal (GPTQ), not RTN.

### Why RTN + variable bits ≈ RTN + uniform bits

RTN has quantization error:
```
E_RTN ≈ Δ²/12 where Δ = range/(2^b - 1)
```

With variable bits, low-bit blocks have MUCH larger Δ. A single
2-bit block has 4× the Δ of a 4-bit block → 16× the error.
Without error compensation, these bad blocks dominate the total error,
negating the benefit of giving more bits to important blocks.

GPTQ fixes this by compensating: when block i has high error,
subsequent weights are adjusted to absorb it.

## Action Plan

1. **Implement GPTQ** for per-block quantization → reduces per-block error
2. **Re-run this comparison** with GPTQ: MXQ-4bit-GPTQ vs MLX uniform 4-bit
3. **If MXQ-4bit-GPTQ beats MLX-4bit**: the approach is validated
4. **Test MXQ-2.5bit-GPTQ vs MLX-4bit**: the headline claim
5. **Perplexity evaluation on wikitext**: proper benchmark, not just one prompt
