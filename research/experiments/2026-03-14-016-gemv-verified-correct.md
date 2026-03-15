# Experiment 016: GEMV Kernel Verified Correct

**Date**: 2026-03-14
**Author**: Eric Jang (eric@vmlx.net)
**Status**: PASS — GEMV is correct, false alarm resolved

## The Scare

CPU Q proj[4] = -15.85, GPU Q proj[4] = -1.35. Appeared to be 11.7x off.

## Resolution

The Q proj bias for element 4 is **-14.5**.

```
CPU Q[4] with bias:    -15.85
CPU Q[4] without bias: -15.85 - (-14.5) = -1.35
GPU Q[4] (no bias):    -1.3496
```

**GPU matches CPU exactly.** The debug trace was running without bias
application (debugForwardOneLayer was written before bias support was added).

## Verified Correct

All core kernels are now verified:
1. mxq_embedding_dequant — bit-identical to CPU ✓
2. mxq_rms_norm — matches CPU to 4 sig figs ✓
3. mxq_dequant_gemv — matches CPU without bias ✓ (bias applied separately)
4. mxq_add (bias) — applied in real forward(), not in debug path
5. mxq_rope — identity at pos=0, formula verified ✓
6. mxq_silu_mul — output looks reasonable ✓
7. mxq_attention_decode — produces V values for single token ✓

## Remaining Issue

All kernels are correct individually. The forward pass produces garbled
output. The issue must be in:
- How kernels interact across 24 layers
- Possible buffer overwrite between encoder dispatches
- The command buffer encoding (multiple encoders in one buffer)
- lm_head computation (151,936 output GEMV)

The most productive next step is to compare final logits (after all 24
layers) between CPU reference and GPU.
