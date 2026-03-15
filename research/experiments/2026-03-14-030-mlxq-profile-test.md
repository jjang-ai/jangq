# Experiment 030: MLXQ-3 Profile on Qwen2.5-3B

**Date**: 2026-03-14
**Author**: Eric Jang (eric@vmlx.net)
**Status**: PARTIAL — profile works but quality needs improvement

## Setup

- Profile: mxq-3 (MLP=3bit, attention=6bit, embed=4bit, lm_head=6bit)
- Model: Qwen2.5-3B
- Quantization: RTN (no AWQ, no GPTQ)

## Results

| Metric | MLXQ-3 Profile | MLXQ 8-bit | MLX Uniform 4-bit |
|--------|---------------|-----------|-------------------|
| Avg bits | 3.32 | 8.0 | 4.5 |
| Size | 1.19 GB | 3.6 GB | ~1.5 GB |
| Load time | 0.33s | 0.63s | N/A (MLX) |
| GPU memory | 1.59 GB | 3.3 GB | N/A |
| Top token | 'What' | 'The' ✓ | 'The' ✓ |
| Correct? | NO | YES | YES |

### Bit allocation
- 3-bit: 78.9% (MLP layers)
- 4-bit: 15.6% (embeddings, V/O projections)
- 6-bit: 5.5% (attention Q/K projections)

### Layer norm analysis
- L0 norm: 14.40 (reference ~14.7 — close)
- L1 norm: **63.67** (reference ~19.6 — 3x too high!)
- L2 norm: 67.00 (reference ~20.4 — still exploding)

## Analysis

The L1 norm explosion (14 → 64) indicates layer 0's MLP at 3-bit
introduces massive error that compounds immediately. The reference
model stays at ~20 through layer 1.

This is because our RTN 3-bit quantization on MLP layers (which have
the largest weight values and highest dynamic range) loses too much
information without error compensation.

## Comparison with Experiment 028

In experiment 028, we proved MLP=3/attn=6 beats uniform 4-bit when
using **MLX's quantizer**. But MLX uses group_size=64 with affine
scaling (effectively 3.5 bits with scale overhead). Our RTN quantizer
at true 3-bit is more aggressive and lacks the same precision.

## What Would Fix This

1. **AWQ scaling** — would reduce the dynamic range of important
   channels before quantization, reducing error
2. **GPTQ within each block** — error compensation would distribute
   the quantization error more evenly
3. **Using 3-bit with group_size overhead** — our format already
   includes per-block scale/zero, so effective bits are ~3.5, but
   the scale computation could be improved (MSE-optimal instead of RTN)

## Key Takeaway

The variable allocation STRATEGY is correct (proven by experiment 028
with MLX quantizer). The implementation quality of our per-block
quantization needs improvement. The bottleneck is RTN quality at 3-bit,
not the allocation strategy itself.
