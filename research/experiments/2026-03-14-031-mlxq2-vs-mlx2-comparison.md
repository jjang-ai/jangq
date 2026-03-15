# Experiment 031: MLXQ-2.5 vs MLX Uniform 2-bit

**Date**: 2026-03-14
**Author**: Eric Jang (eric@vmlx.net)
**Status**: BOTH FAIL — 2-bit on 3B model is too aggressive

## Results

| Metric | MLXQ-2.5 (MLP=2, attn=6) | MLX Uniform 2-bit |
|--------|--------------------------|-------------------|
| Avg bits | 2.53 | 2.50 |
| Model size | 0.91 GB | ~0.95 GB |
| Load time | 0.27s | N/A |
| GPU memory | 1.30 GB | N/A |
| Top token | ' ' (space) | 'izu' |
| Output quality | Garbage | Garbage |
| L1 norm | **714** (explosion!) | N/A |

## Analysis

Both approaches fail at ~2.5 bits on a 3B model. The L1 norm explosion
(13.5 → 714 between layer 0 and 1) shows that 2-bit MLP blocks
introduce catastrophic errors even with MSE-optimal clipping.

### Why 2-bit MLP fails on 3B

MLP weights in Qwen2.5-3B have:
- gate_proj: (2048, 11008) — 22.5M weights per layer
- up_proj: (2048, 11008) — 22.5M weights per layer
- down_proj: (11008, 2048) — 22.5M weights per layer

At 2-bit, each block of 64 weights gets only 4 quantization levels.
The dynamic range of MLP weights is [-0.7, 0.7], so the step size is
Δ = 1.4 / 3 ≈ 0.47. Any weight value can only be represented within
±0.23 of its true value. This is a ~33% relative error per weight.

When this feeds into the next layer's attention (at 6-bit precision),
the attention scores are computed on corrupted values, causing the
entire layer to produce wrong attention patterns.

### What would work at ~2.5 bits

1. **Larger models (7B+)**: more redundancy per weight → tolerates more error
2. **GPTQ error compensation**: reduces effective error by propagating corrections
3. **MLP at 3-bit, attention at 4-bit**: avg ~3.1 bits, tested as experiment 028
   where it BEATS uniform 4-bit
4. **NF2 (NormalFloat 2-bit)**: non-uniform quantization levels optimized for
   Gaussian weight distributions. 4 values at {-0.80, -0.27, 0.27, 0.80}
   instead of uniformly spaced {0, 0.33, 0.67, 1.0}

## Conclusion

The MLXQ-2.5 profile is too aggressive for 3B models with RTN quantization.
The mxq-3 profile (MLP=3, attn=6) at ~3.3 bits is the sweet spot for 3B.
True 2.5-bit quality requires either larger models or advanced quantization
(GPTQ + NF2).

## Speed Notes

Model load time: 0.27s (mmap zero-copy — this is excellent)
GPU memory: 1.30 GB for a 3B model at 2.5 bits
Speed benchmarking not yet implemented — need to measure tokens/sec.
