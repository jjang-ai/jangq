# Experiment 020: Per-Layer Hidden State Comparison

**Date**: 2026-03-14
**Author**: Eric Jang (eric@vmlx.net)
**Status**: IMPORTANT — quantization error compounds across layers

## Setup

Both models process same 26-token prompt with causal attention.
MLX uses bf16 original weights. MXQ uses our 2.76-bit quantized weights.

## Per-Layer Hidden State Norms (last position)

| Layer | MXQ Norm | MLX Norm | Ratio | Diverging? |
|-------|----------|----------|-------|------------|
| L00 | 3.89 | 3.67 | 1.06 | Slight |
| L01 | 6.64 | 6.93 | 0.96 | Slight |
| L02 | 7.42 | 7.32 | 1.01 | OK |
| L03 | 9.07 | 8.52 | 1.06 | Slight |
| L04 | 10.43 | 9.36 | 1.11 | Growing |
| L05 | 10.88 | 10.04 | 1.08 | |
| L10 | 17.15 | 14.52 | 1.18 | Significant |
| L15 | 20.25 | 20.45 | 0.99 | Recovered? |
| L20 | 44.31 | 50.79 | 0.87 | Significant |
| L23 | 72.28 | 53.08 | 1.36 | Large |

## Final Logits

| | Top Token | Top Logit | Logits[:4] |
|---|-----------|-----------|-----------|
| MLX (bf16) | 17 ('2') | 14.31 | [2.84, 2.44, 4.56, 8.63] |
| MXQ (2.76bit) | 251 ('') | 13.23 | [7.18, 5.40, 3.49, 3.99] |

## Analysis

1. **Quantization error compounds**: Starting at ~6% difference at L0,
   growing to ~36% by L23. This is expected for aggressive quantization.

2. **The norms are in the same ballpark**: Not off by orders of magnitude.
   This suggests the forward pass structure is correct but the quantization
   quality at 2.76 bits is degrading the 0.5B model too much.

3. **0.5B models are particularly sensitive**: Small models have less
   redundancy — each weight matters more. The same quantization level
   that works for 70B might destroy a 0.5B model.

## Possible Causes

1. **Quantization quality too low for 0.5B**: 2.76-bit average on a 0.5B
   model might be below the quality threshold. Need to test with higher
   bits (e.g., MXQ-4 or MXQ-6).

2. **Weight-only calibration inadequate**: Our calibration used weight
   statistics only, not activation-aware scoring. Better calibration
   would allocate bits more optimally.

3. **RTN quantization vs GPTQ**: We used simple round-to-nearest.
   GPTQ-style error compensation would reduce per-layer errors.

4. **Forward pass bug still possible**: Though norms track reasonably,
   the value signs differ significantly at some layers.

## Next Steps

1. Test MXQ-4bit and MXQ-6bit to see if higher bits fix the output
2. Test with activation-aware calibration
3. Try GPTQ-style quantization
4. Rule out forward pass bugs by testing with bf16 weights through MXQ runtime
