# Experiment 012: Inference with Real Embedding Dequant

**Date**: 2026-03-14
**Author**: Eric Jang (eric@vmlx.net)
**Status**: PROGRESS — embedding works, output incoherent

## Setup
- Same as experiment 011 but with real kernel dispatch
- Embedding dequant: `mxq_embedding_dequant` kernel
- Attention: `mxq_attention_decode` kernel with GQA
- LM head: tied embeddings via `mxq_dequant_gemv` on embed_tokens

## Results

### Output
```
ascusirossão=-=-igadb/aycassertziel,DB_pemb novità LoginActivity抛弃irit鳞 potràensoepad lou
```

### Analysis
1. **Embedding dequant IS working**: output changed from constant `.<;(E"/E*-`
   (zeros → same logits → same token) to varied, multilingual tokens. This
   means the embedding lookup is producing real hidden states.

2. **Forward pass produces non-trivial logits**: different tokens selected
   (Portuguese, Italian, Chinese, English, code). The logits distribution
   is not degenerate.

3. **But output is incoherent**: the model isn't generating sensible text.
   This could be caused by:

   a. **GEMV kernel bug**: the dequant+matmul might have a dimensional
      or indexing error (most likely — this is the hardest kernel)

   b. **Attention bug**: GQA head mapping, KV cache indexing, or the
      softmax reduction might be wrong

   c. **Weight layout mismatch**: our Python quantizer packs weights
      row-major but the GEMV kernel might expect column-major

   d. **RoPE theta**: should be 1,000,000 for Qwen — verify it's
      being passed correctly

   e. **Norm epsilon**: should be 1e-6 for Qwen — verify

## Debugging Plan

1. **Validate GEMV correctness**: dequant a small weight block on CPU,
   compare with GPU kernel output
2. **Check weight layout**: verify the block indexing in GEMV matches
   the Python packing order
3. **Verify attention**: check Q·K^T scores make sense
4. **Verify RoPE params**: check theta_base in actual kernel dispatch

## Significance

The pipeline is functionally complete — every kernel fires, data flows
through the full transformer. This is a correctness debugging phase,
not a structural issue. The architecture works.
