# Experiment 014: Attention Kernel Debugging

**Date**: 2026-03-14
**Author**: Eric Jang (eric@vmlx.net)
**Status**: IN PROGRESS

## Context

Experiments 011-013 established:
- Python quantization math: CORRECT (MSE = 0.0000022)
- GPU embedding dequant: BIT-IDENTICAL to CPU
- GPU RMSNorm: correct normalized output
- GPU dequant GEMV: meaningful projections
- Full pipeline runs without crashes
- Output is incoherent: garbage tokens

## Hypothesis

The attention kernel or forward pass buffer management is the issue.
Possible causes ranked by likelihood:

1. **Attention KV cache not being filled correctly** — if the K/V
   values aren't stored at the right offsets, attention reads garbage
2. **GQA head mapping wrong** — 14 Q heads / 2 KV heads = each KV head
   serves 7 Q heads. If the mapping math is wrong, heads read wrong KV
3. **Attention scale incorrect** — should be 1/sqrt(64) = 0.125
4. **Softmax in attention kernel** — SIMD reduction across threadgroup
   might have race conditions
5. **Forward pass buffer reuse** — normBuffer used as temp in multiple
   places, might be overwritten before being read

## Findings So Far

### Verified CORRECT:
- Embedding dequant: GPU = CPU (bit-identical)
- RMSNorm: reasonable output
- Dequant GEMV: meaningful projections
- Tokenizer: 26 tokens match HuggingFace reference EXACTLY
- Q/K/V biases: now loaded and applied (Qwen has attention biases)
- System prompt: now includes default "You are a helpful assistant."

### Still Wrong:
- Output is garbled multilingual text despite all above being correct
- Single-layer debug trace shows reasonable values at every step
- Issue must be in multi-layer accumulation or attention across positions

### Remaining Suspects:
1. Attention kernel: does it handle the growing KV cache correctly across 26 prefill steps?
2. Forward pass: is the buffer reuse causing data corruption between steps?
3. RoPE: position encoding might be computed wrong after position 0
4. GEMV for lm_head: with 151,936 output rows, is there a dispatch issue?

## Method

Step 1: Dump K/V values after first token's KV cache store
Step 2: Dump attention output after first attention computation
Step 3: Compare attention output with CPU reference
Step 4: If attention is wrong, isolate which sub-step fails
