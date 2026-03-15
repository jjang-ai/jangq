# Experiment 036: DIRECT Comparison — Real Text Output

**Date**: 2026-03-14
**Author**: Eric Jang (eric@vmlx.net)
**Status**: CRITICAL RESULT — honest assessment

## Setup

Model: Qwen2.5-3B
3 prompts, max 60 tokens each
All use MLX's own quantizer (eliminating implementation artifacts)
All use same chat template and tokenizer

## Results

### "What is 2+2?"

| Method | Bits | Output |
|--------|------|--------|
| bf16 | 16 | "The answer is 4. Is there anything else I can help you with?" |
| **Uniform 4-bit** | **4.50** | **"The answer to the question is 4. Is there anything else..."** |
| MLXQ MLP=3/attn=6 | 3.37 | "What 2+2+2+2+2+2+2+..." (repetition loop) |
| MLXQ MLP=3/attn=4 | 3.12 | "What is 2+2+2+2+2+..." (repetition loop) |
| Uniform 3-bit | 3.14 | Garbage Unicode characters |

### "Write a Python function to reverse a string."

| Method | Bits | Output |
|--------|------|--------|
| bf16 | 16 | "Sure, here's a Python function that reverses a string: ```python..." |
| **Uniform 4-bit** | **4.50** | **"Here's a Python function that reverses a string: ```python..."** |
| MLXQ MLP=3/attn=4 | 3.12 | "Write a Python function to reverse a string._snap..." (echo) |
| MLXQ MLP=3/attn=6 | 3.37 | "_snap a string a string a string..." (repetition) |
| Uniform 3-bit | 3.14 | "Write a Python: A B I I I I..." (garbage) |

### "Explain gravity in one sentence."

| Method | Bits | Output |
|--------|------|--------|
| bf16 | 16 | "Gravity is the force that attracts two objects with mass towards each other." |
| **Uniform 4-bit** | **4.50** | **"gravity is the force that pulls objects towards the center..."** |
| MLXQ MLP=3/attn=4 | 3.12 | "Explain gravity in one sentence.‗" (echo + stop) |
| MLXQ MLP=3/attn=6 | 3.37 | "izzy_snap ney ney ney..." (garbage loop) |
| Uniform 3-bit | 3.14 | "bbw bb bb āā..." (garbage) |

## Honest Analysis

### What works
- **bf16**: Perfect output
- **Uniform 4-bit**: Excellent — correct answers, good quality

### What doesn't work (at ~3 bits on 3B)
- **ALL 3-bit methods produce garbage** on this 3B model
- Uniform 3-bit: worst (garbage characters)
- MLXQ MLP=3/attn=4: echoes prompt, stops
- MLXQ MLP=3/attn=6: slightly better (knows some words) but still fails

### MLXQ vs Uniform at same bits
- At ~3.1 bits: MLXQ (MLP=3/attn=4) > Uniform 3-bit
  - MLXQ echoes the prompt (knows English)
  - Uniform 3-bit produces Unicode garbage
  - **MLXQ is BETTER at the same bit count, but both fail**

### Why 3-bit fails on 3B

Qwen2.5-3B has 3B parameters. At 3-bit, each weight has only 8 levels.
The model doesn't have enough redundancy to tolerate this level of
compression. The same quantization would likely work on 7B+ models
because larger models have more redundancy per weight.

## The Real Comparison That Matters

The comparison should be:
1. **MLXQ at 3.37 bits vs Uniform at 3.14 bits** — MLXQ wins (less garbage)
2. **MLXQ at ~4 bits vs Uniform at 4 bits** — need to test this
3. **On 7B+ models** — where 3-bit actually becomes viable

## Key Takeaway for the Paper

On Qwen2.5-3B:
- 4-bit is the minimum for usable output
- At 3-bit, MLXQ allocation (MLP=3/attn=4) produces more coherent
  output than uniform 3-bit (echoes prompt vs garbage)
- The real value of MLXQ is proven on the MSE metric (experiment 028)
  even though both methods fail on text generation at 3-bit on 3B

**The paper should test on 7B+ where the quality gap is actionable.**
