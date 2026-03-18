# Experiment 052: Quantized-In-Memory Inference — The GGUF Breakthrough

**Date**: 2026-03-15
**Author**: Jinho Jang (eric@jangq.ai)
**Status**: WORKING

## The Problem

JANG dequant-at-load expanded models to full float16 in RAM:
- MiniMax 230B at 2-bit: 60 GB on disk → 460 GB float16 = doesn't fit anywhere
- This defeated the entire purpose of quantization

## The Solution: Repack JANG → MLX Quantized Format

Instead of dequantizing to float16, repack JANG weights into MLX's native
quantized format (uint32 packed + float32 scales/biases). The model stays
compressed in GPU memory and dequantizes on-the-fly during `quantized_matmul`.

### Key Discovery: MLX Per-Layer Mixed Bits Has NO Speed Regression

The CRACK research (Finding #75) showed 60% speed regression with per-tensor
quantization via `class_predicate`. But testing showed the regression is in the
LOADING path only — `quantized_matmul` itself runs at full speed regardless of
per-layer bit differences.

Benchmark (2048×2048 layers):
```
Mixed bits (2+8): 0.2ms per forward (5490 fwd/s)
Uniform 2-bit:    0.2ms per forward (5731 fwd/s)
Uniform 8-bit:    0.2ms per forward (6207 fwd/s)
```

No regression. This means JANG can use different bits per tensor (CRITICAL=8,
COMPRESS=2) without any inference speed penalty.

### Format Mapping (Verified)

```
JANG qweight (uint8 LSB-first) → view as uint32 (same bytes)
JANG scale → MLX scale (same value)
JANG zero → MLX bias = -scale * zero
JANG block_size=64 → MLX group_size=64
```

All MLX bit widths (2, 3, 4, 5, 6, 8) supported in quantized_matmul.

## Results

### Qwen2.5-0.5B JANG_2S — First Quantized-In-Memory Test
- Load time: **0.1s** (repack only, no dequant)
- GPU memory: **0.23 GB** (quantized, not 1 GB float16)
- Generates text (quality is expected garbage at 0.5B 2.9-bit)

### Memory Comparison

| Model | Dequant-at-load | Quantized-in-memory |
|-------|----------------|-------------------|
| Qwen2.5-0.5B 2.9b | ~1 GB | **0.23 GB** |
| Qwen3.5-35B 2.3b | ~67 GB | **~10 GB** |
| Qwen3.5-122B 2.2b | ~244 GB | **~32 GB** |
| MiniMax 230B 2.1b | ~460 GB (OOM) | **~60 GB** |

### Implementation

The loader (`jang_loader.py`) now:
1. Reads JANG shards (uint8 packed weights, float16 scales/zeros)
2. Views uint8 as uint32 (zero-copy, same bytes)
3. Converts JANG zeros → MLX biases (-scale * zero)
4. Sets up MLX quantization config (group_size, bits)
5. Creates QuantizedLinear/QuantizedEmbedding layers at default bits
6. Loads repacked weights — auto-fixes bits per layer from weight shape
7. Model runs with `quantized_matmul` at full Metal speed

## LANDMARK: 35B MoE at 2.28 bits, 13.3 GB, Coherent

```
GPU: 13.3 GB
> What is 2+2?
  2+2 equals 4. This is a simple addition problem where we combine two
  units with two more units to get a total of four units.
```

- Load time: 3.3 seconds (repack JANG → MLX quantized, no dequant)
- Memory: 13.3 GB (vs 67 GB float16 = 5x compression in memory)
- Speed: Full MLX Metal `quantized_matmul` + `gather_qmm` — native speed
- Output: Correct, coherent, extended explanation

## Full 6-Prompt Test — 35B JANG_2L at 2.28 bits

| Prompt | Speed | Response | Correct? |
|--------|-------|----------|----------|
| What is 2+2? | 90 tok/s | "2+2 equals 4. This is a simple addition problem..." | YES |
| Is a tomato a fruit? | 99.8 tok/s | "A. A. A. A..." loops | NO |
| What is photosynthesis? | 94.4 tok/s | "process by which plants, algae, and some bacteria convert light energy into chemical energy..." | YES |
| Three planets larger | 100.3 tok/s | "**Jupiter**, **Saturn**, and **Uranus**" | YES |
| Romeo and Juliet | 103.7 tok/s | "William Shakespeare" (then format loops) | PARTIAL |
| Capital of France | 105.6 tok/s | "**Paris**. It is also the country's most populous city..." | YES |

**4/6 correct at 90-106 tok/s, 13.3 GB memory. Full MLX native Metal speed.**

## Models Ready

| Model | Profile | Bits | Disk | Memory |
|-------|---------|------|------|--------|
| Qwen3.5-35B-JANG_2L_v2 | JANG_2L | 2.28 | 15 GB | ~10 GB |
| Qwen3.5-35B-JANG_3M | JANG_3M | 3.10 | 18 GB | ~13 GB |
| Qwen3.5-122B-JANG_2L | JANG_2L | 2.19 | 50 GB | ~32 GB |
| MiniMax-M2.5-JANG_2S | JANG_2S | 2.06 | 88 GB | ~60 GB |

## Files Modified

- `vmlx_engine/utils/jang_loader.py` — Complete rewrite:
  - `_repack_jang_to_mlx()` replaces `_dequantize_jang_weights()`
  - `_fix_quantized_bits()` — adjusts per-layer bit widths
  - Vectorized `_unpack_bits()` for 3/5/6-bit
  - Sets up `quantization` config for QuantizedLinear creation
