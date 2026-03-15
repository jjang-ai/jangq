# Experiment 011: First MXQ Inference Attempt

**Date**: 2026-03-14
**Author**: Eric Jang (eric@vmlx.net)
**Status**: PARTIAL PASS — pipeline runs, output is garbage (expected)

## Setup

- **Model**: Qwen2.5-0.5B-MXQ-2.5bit (our quantized model)
- **Hardware**: Apple M4 Max, 107 GB unified memory
- **Command**: `mxq run model/ --prompt "What is 2+2?" --max-tokens 10`

## Results

### What worked
- Metal device initialization: Apple M4 Max detected correctly
- Model loading: 0.06 seconds (safetensors → Metal buffers)
- GPU memory: 210.7 MB loaded
- Tokenizer: loaded, encoded 15 tokens for chat prompt
- Inference engine: KV cache allocated (24 layers × 2048 positions = 24 MB)
- Forward pass: 15 prefill + 10 decode tokens executed without crash
- Metal kernel dispatch: all 10+ kernel types dispatched successfully

### What didn't work
- Output: `.<;(E"/E*-` (garbage)
- Root cause: `dispatchEmbedding` is a placeholder returning zeros
  - Hidden state starts as all zeros → all subsequent computations are on zeros
  - Even with correct dequant kernels, zero input → garbage output

### Performance (preliminary, with placeholder embedding)
- Model load: 0.06 seconds (mmap zero-copy works)
- 25 forward passes: completed (no timing instrumentation yet)

## Attempt History

1. First attempt: `Error: Tensor not found: model.embed_tokens.weight`
   - Cause: loader tried to load embedding as float16, but it's quantized
   - Fix: load embedding as MXQWeight instead of float16 tensor

2. Second attempt: runs end-to-end, garbage output
   - Cause: embedding dequant placeholder returns zeros
   - Fix needed: implement real embedding dequant kernel dispatch

## Next Steps

1. Wire up `mxq_embedding_dequant` kernel in the inference engine
2. Verify dequant GEMV produces correct output on real weights
3. Verify attention kernel handles GQA correctly
4. Once embedding works, dequant quality becomes testable

## Significance

This is the first time the full MXQ pipeline runs end-to-end:
Python quantize → .mxq format → Swift loader → Metal kernels → token output

The 0.06s model load time is excellent — mmap zero-copy on Apple Silicon
unified memory means the weights go directly from disk to GPU-accessible
memory without any CPU copies.
