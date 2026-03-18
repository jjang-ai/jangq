# Experiment 054: 122B and MiniMax Quantized-In-Memory Inference

**Date**: 2026-03-15
**Author**: Jinho Jang (eric@jangq.ai)
**Status**: IN PROGRESS

## Qwen3.5-122B-A10B JANG_2L — WORKING

- **Parameters**: 122B total, 10B active (MoE 256 experts, top-8)
- **Profile**: JANG_2L (CRITICAL=8b, IMPORTANT=6b, COMPRESS=2b)
- **Average bits**: 2.19
- **GPU memory**: 45.3 GB (quantized in memory)
- **Speed**: 38-49 tok/s on M4 Max 128 GB
- **Load time**: ~10s (repack 50 GB JANG → MLX quantized)

### Results

| Prompt | Speed | Response | Score |
|--------|-------|----------|-------|
| What is 2+2? | 38.2 tok/s | "2+2 is 4." (correct, then repeats) | PARTIAL |
| What is photosynthesis? | 49.5 tok/s | "process by which green plants, algae, and some bacteria convert light energy, usually from the sun, into chemical energy in the form of glucose" | **PERFECT** |
| Three planets larger | 49.0 tok/s | Uses `<think>` tags, lists Jupiter and Saturn with details | **PERFECT** |
| Capital of France | 49.2 tok/s | "Paris" then coherently answers follow-up questions | **PERFECT** |

### Notable Features
- Proper `<think>` tag usage (reasoning capability preserved at 2.19 bits)
- Markdown formatting (**bold** text)
- Multi-question answering (continues coherently after first answer)
- 49 tok/s on a 122B model is excellent — memory bandwidth bound

## MiniMax-M2.5 JANG_2S — Testing on Mac Studio

- **Parameters**: 230B total, 10B active (MoE 256 experts, top-8)
- **Profile**: JANG_2S (CRITICAL=6b, IMPORTANT=4b, COMPRESS=2b)
- **Average bits**: 2.06
- **Expected GPU memory**: ~60 GB (fits in Mac Studio 192 GB)
- Loading on Mac Studio now...

## Summary So Far

| Model | Bits | GPU Memory | Speed | Score |
|-------|------|-----------|-------|-------|
| Qwen3.5-35B-A3B | 2.28 | 13.3 GB | 90-106 tok/s | 4/6 |
| **Qwen3.5-122B-A10B** | **2.19** | **45.3 GB** | **38-49 tok/s** | **3/4** |
| MiniMax-M2.5 | 2.06 | ~60 GB | TBD | TBD |

All models running quantized in memory using MLX native Metal kernels.
No float16 expansion. JANG = GGUF for MLX.
