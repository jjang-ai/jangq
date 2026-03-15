# Experiment 025: GPTQ Mathematical Verification

**Date**: 2026-03-14
**Author**: Eric Jang (eric@vmlx.net)
**Status**: VERIFIED — GPTQ provides 1.2-1.3x improvement over RTN

## Purpose

Verify the GPTQ algorithm produces lower output error than RTN,
establishing the mathematical foundation for MXQ's quality improvement.

## GPTQ Algorithm

For weight matrix W (out × in) with Hessian H = X^T X / n_samples:

```
1. Compute H_inv = (H + λI)^{-1}   (λ = damping for stability)
2. For each column j:
   a. q_j = quantize(W[:, j], bits)         — quantize column
   b. δ_j = W[:, j] - dequant(q_j)          — quantization error
   c. W[:, j+1:] -= δ_j ⊗ H_inv[j, j+1:] / H_inv[j, j]  — compensate
```

Step (c) is the key: it adjusts remaining unquantized weights to minimize
the **output error** (Y = X @ W^T), not just the **weight error**.

## Results

### Small example (4×4, 2 samples)
GPTQ 0.8x RTN — WORSE (too few samples for stable Hessian)

### Realistic example (256×256, 128 samples)

| Bits | RTN MSE | GPTQ MSE | GPTQ Improvement |
|------|---------|----------|-----------------|
| 2 | 0.03080 | 0.02541 | 1.2x |
| 3 | 0.00566 | 0.00435 | 1.3x |
| 4 | 0.00123 | 0.00093 | 1.3x |

### Why only 1.2-1.3x (paper claims 2-5x)

My simplified implementation lacks:
1. **Act-order**: quantize most-sensitive columns first (by H diagonal)
2. **Full Cholesky**: proper inverse computation
3. **Cross-block compensation**: my block_size=128 limits compensation range
4. **Optimal damping**: λ should be tuned per-matrix

## Mathematical Analysis

### Per-column error compensation

The GPTQ update rule:
```
W[:, j+1:] -= δ_j ⊗ H_inv[j, j+1:] / H_inv[j, j]
```

is derived from minimizing:
```
ΔL = (W[:, j] - Q(W[:, j]))^T × H × (W[:, j] - Q(W[:, j]))
```

by finding the optimal compensation for remaining weights. This is
the OBS (Optimal Brain Surgeon) framework applied column-by-column.

### Why compensation helps more at lower bits

At 4-bit: quantization error δ is small → compensation is a small adjustment
At 2-bit: quantization error δ is large → compensation is a large adjustment
→ GPTQ's advantage grows as bit width decreases

### Combined with variable bit allocation

MXQ + GPTQ: for each block at its allocated bit width (2, 3, 4, ...),
apply GPTQ error compensation WITHIN the block. This gives:

1. Smart bit allocation (importance-based) — allocates bits where they matter
2. Optimal rounding (GPTQ) — minimizes error at each bit width
3. Error compensation — propagates corrections across weights

The combination should produce strictly better quality than either alone.

## Next Steps

1. Implement full GPTQ with act-order and Cholesky
2. Integrate GPTQ into MXQ's per-block quantization
3. Re-run the comparison: MXQ-4bit-GPTQ vs MLX-uniform-4bit
4. Test MXQ-2.5bit-GPTQ vs MLX-uniform-4bit (the headline claim)
