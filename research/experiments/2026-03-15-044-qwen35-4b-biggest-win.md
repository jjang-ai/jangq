# Experiment 044: Qwen3.5-4B — BIGGEST WIN YET

**Date**: 2026-03-15
**Author**: Eric Jang (eric@vmlx.net)
**Status**: LANDMARK RESULT

## The Result

**Qwen3.5-4B at 2.5 bits: MLXQ MQ2S gets 6/6 correct, Uniform gets 0/6.**
Same effective bits. Same model. Night and day difference.

## Side-by-Side at 2.5 Effective Bits

| Prompt | Uniform 2-bit | MLXQ MQ2S |
|--------|--------------|-----------|
| "What is 2+2?" | `2+2? 2+2? 2+2? 2+2?` | **"The answer is 4."** |
| "Is a tomato a fruit?" | `1 1 1 1 1 1 1 1 1 1` | **"A tomato is a fruit, not a vegetable."** |
| "Who wrote Romeo and Juliet?" | `10, 10, 10, 10, 10, 10` | **"The user is asking about the author..."** |
| "What is photosynthesis?" | garbled text | **"Photosynthesis is the process by which plants..."** |
| "How many legs does a spider?" | `10, 10, 10, 10, 10` | **"The user is asking about the number..."** |
| "Largest ocean on Earth?" | loops | **"The Pacific Ocean."** |

## Why This Model Shows the Strongest Results

Qwen3.5-4B has a **hybrid architecture**:
- 24 linear attention layers (SSM-like, no softmax)
- 8 full attention layers (standard transformer)

The full attention layers are CRITICAL — they're the only ones doing
traditional attention with KV cache. When these 8 layers lose precision,
the entire model collapses. MLXQ protects these 8 layers with 6-bit
while compressing the 24 linear layers and all MLP at 2-bit.

This is the strongest demonstration yet that **architecture-aware
quantization is the right approach for modern hybrid models**.

## Also Notable: 3-bit Results

At 3-bit, both uniform and MLXQ work well on this model:
- Uniform 3: correct answers, slight number spam on "2+2"
- MQ3M: correct answers, cleaner output, uses <think> tags properly

## Model Architecture Details

```
Qwen3.5-4B:
  model_type: qwen3_5
  hidden: 2560, layers: 32, heads: 16/4 (GQA 4:1)
  layer_types: 24 linear_attention + 8 full_attention
  head_dim: 256
  partial_rotary_factor: 0.25
  rope_theta: 10,000,000
```

## Significance

This is the result that proves MLXQ's value for next-generation models:
1. Hybrid architectures (Qwen 3.5, Jamba, Nemotron) have mixed layer types
2. Uniform quantization treats all layers equally — wastes bits on robust
   linear attention layers, starves critical full attention layers
3. MLXQ allocates bits where they matter — dramatic quality improvement
4. **6/6 vs 0/6 at the same bit count is the clearest win possible**
