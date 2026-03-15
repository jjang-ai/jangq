# Experiment 017: First Token Comparison — MXQ vs Reference

**Date**: 2026-03-14
**Author**: Eric Jang (eric@vmlx.net)
**Status**: MISMATCH — investigating multi-layer forward pass

## Setup

- **Prompt**: `<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n`
- **Token IDs**: [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30, 151645, 198, 151644, 77091, 198] (26 tokens, verified identical)

## Results

| | Reference (MLX bf16) | MXQ Runtime |
|---|---|---|
| First token | `2` (id 17) | `pecting` (wrong) |
| Top logit | 14.3125 | unknown |
| Logits[0:8] | [2.84, 2.44, 4.56, 8.63, 3.98, 1.92, 1.77, 6.03] | unknown |

## Verified Correct
- Token IDs: identical (26 tokens)
- All individual Metal kernels: verified (embed, norm, GEMV, RoPE, attention, SiLU)
- Bias application: implemented
- Block metadata: correct on GPU

## Remaining Bug Location

The issue is in the **multi-layer forward pass composition**. Individual
kernels are correct, but something goes wrong when 24 layers of them
chain together across 26 prefill tokens.

Possible causes:
1. Command buffer dispatch ordering within a single buffer
2. Buffer aliasing between kernel dispatches
3. Attention kernel behavior with growing KV cache (positions 1-25)
4. Float16 accumulation error compounding across 24 layers
5. A bug in how the forward pass handles the prefill loop (26 tokens processed one at a time)

## Next Steps
- Dump final logits from GPU and compare with reference
- Run with fewer layers (e.g., 1 layer) to see if issue is per-layer or cumulative
- Check if issue appears at position 0 (first token) or only after multiple tokens
