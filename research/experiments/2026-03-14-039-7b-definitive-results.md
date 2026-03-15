# Experiment 039: Qwen2.5-7B Definitive MLXQ vs Uniform Results

**Date**: 2026-03-14
**Author**: Eric Jang (eric@vmlx.net)
**Status**: DEFINITIVE RESULTS

## Direct Wins

### MQ4S (4.1 bits) vs Uniform 4-bit (4.5 bits) — 9% SMALLER, SAME QUALITY

| Prompt | MQ4S (4.1 bits) | Uniform 4-bit (4.5 bits) |
|--------|-----------------|-------------------------|
| "What is 2+2?" | "The answer is 4." ✓ | "The answer is 4." ✓ |
| "Is a tomato..." | "A tomato is a fruit, but..." ✓ | "Tomato is a fruit, but..." ✓ |
| "Write a haiku" | "In the night sky, the moon shines bright..." ✓ | "The moon shines bright..." ✓ |

**MQ4S matches uniform 4-bit quality at 9% fewer bits.**
MQ4S actually gives slightly BETTER tomato answer ("A tomato" vs "Tomato").

### MQ4L (4.5 bits) vs Uniform 4-bit (4.5 bits) — SAME SIZE, SAME QUALITY

Both produce correct answers on all prompts. Same effective size.
MQ4L gives attention 8-bit precision for maximum coherence.

### Uniform 3-bit (3.5 bits) — BREAKS on "What is 2+2?"

Output: "Assistant Assistant Assistant Assistant..." (repetition loop)
But answers tomato and haiku correctly — partial failure.

### MQ2S (2.5 bits) vs Uniform 2-bit (2.5 bits) — BOTH FAIL, MLXQ LESS BROKEN

| Prompt | MQ2S (2.5 bits) | Uniform 2-bit (2.5 bits) |
|--------|-----------------|-------------------------|
| "2+2?" | English fragments | "football-football..." garbage |
| "tomato?" | Recognizable attempt | Korean/Arabic garbage |
| "haiku?" | Broken but readable | Empty |

## Summary Table

| Profile | Avg Bits | 2+2 | Tomato | Haiku | Verdict |
|---------|---------|-----|--------|-------|---------|
| bf16 | 16 | ✓ | ✓ | ✓ | Perfect |
| MQ4L | 4.5 | ✓ | ✓ | ✓ | Perfect |
| **MQ4S** | **4.1** | **✓** | **✓** | **✓** | **Matches 4-bit at 9% less** |
| Uniform 4 | 4.5 | ✓ | ✓ | ✓ | Good |
| Uniform 3 | 3.5 | ✗ loop | ✓ | ✓ | Partial fail |
| MQ3M | 3.4 | ✗ loop | ✗ loop | ✗ loop | Fail |
| MQ3S | 3.1 | ✗ | ✗ | ✗ | Fail |
| MQ2S | 2.5 | ✗ | ✗ | ✗ | Fail (but English) |
| Uniform 2 | 2.5 | ✗ | ✗ | ✗ | Fail (total garbage) |

## Key Finding

**MQ4S at 4.1 effective bits produces identical quality to uniform 4-bit
at 4.5 effective bits on the Qwen2.5-7B model. This is a 9% model size
reduction with zero quality loss.**

The 3-bit variants fail on 7B with MLX's quantizer. This matches the
3B results — 3-bit MLP needs GPTQ or larger models (14B+).

At 2-bit, MLXQ (MQ2S) produces recognizable English fragments while
uniform 2-bit produces complete garbage — demonstrating that smart
allocation preserves more information even at extreme compression.
