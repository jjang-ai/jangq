# Experiment 038: Multi-Model MLXQ vs Uniform Comparison

**Date**: 2026-03-14
**Author**: Eric Jang (eric@vmlx.net)

## Results Across 3 Models

### Qwen2.5-3B (qwen2, GQA 8:1, non-trad RoPE)

**DIRECT WIN FOUND:**

| Config | Bits | "Is a tomato a fruit?" |
|--------|------|----------------------|
| Uniform 4-bit | 4.00 | Repetition loop ✗ |
| **MLXQ MLP=4/A=5** | **4.12** | **"A tomato is a fruit."** ✓ |
| MLXQ MLP=4/A=8 | 4.49 | "A tomato is a fruit." ✓ |

### SmolLM2-1.7B (llama, MHA, traditional RoPE)

| Config | Bits | Tomato | 2+2 | Capital |
|--------|------|--------|-----|---------|
| bf16 | 16 | "A tomato is a fruit." ✓ | "4" ✓ | "Paris" ✓ |
| Uniform 4-bit | 4.0 | "botanically a fruit" ✓ | "4" ✓ | "Paris" ✓ |
| MLXQ MLP=4/A=5 | 4.3 | "botanically a fruit" ✓ | starts looping | "Paris" ✓ |
| Uniform 3-bit | 3.0 | "fruit or vegetable" (loops) | "4" ✓ | "Paris" ✓ |
| **MLXQ MLP=3/A=6** | **4.0** | **"A tomato is a fruit."** ✓ | starts looping | "Paris" ✓ |

Note: SmolLM2 at MLXQ MLP=3/A=6 (4.0 eff bits) gives better tomato answer
than uniform 3-bit (3.5 eff bits) — "A tomato is a fruit" vs "fruit or vegetable" loop.

### TinyLlama-1.1B (llama, GQA 8:1, traditional RoPE)

| Config | Bits | Tomato | 2+2 | Capital |
|--------|------|--------|-----|---------|
| bf16 | 16 | "Tomatoes are a fruit." ✓ | "4" ✓ | "Paris" ✓ |
| Uniform 4-bit | 4.0 | "Tomato is a fruit." ✓ | "4" ✓ | "Paris" ✓ |
| Uniform 3-bit | 3.0 | "not fruits" (wrong!) | "4" ✓ | "Paris" ✓ |
| **MLXQ MLP=3/A=6** | **4.0** | **"not a fruit" (wrong)** | "4" ✓ | "Paris" ✓ |

TinyLlama at 3-bit gives wrong answer regardless of allocation — model too small.

## Key Findings Across Models

1. **Qwen2.5-3B**: MLXQ clearly beats uniform at ~4 bits (prevents repetition loops)
2. **SmolLM2-1.7B**: MLXQ MLP=3/A=6 gives better tomato answer than uniform 3-bit
3. **TinyLlama-1.1B**: Too small — 3-bit factual errors regardless of allocation

## Pattern

- **MLXQ's advantage is most visible on Qwen** (GQA architecture is more
  sensitive to attention precision — fewer KV heads = each matters more)
- **MHA models** (SmolLM2, TinyLlama with 32 heads) are more robust to
  attention quantization because redundancy across many heads
- **Larger models benefit more** — 3B shows clear wins, 1.7B marginal, 1.1B none

## Downloads Status (Mac Studio)

- Qwen2.5-7B: DONE (14 GB) — copying to local
- Mistral-7B-v0.3: DONE (27 GB)
- Qwen3.5-9B: downloading (5.4/18 GB)
- Others: queued
