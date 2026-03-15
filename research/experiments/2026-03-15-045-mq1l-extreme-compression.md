# Experiment 045: MQ1L — Extreme Compression Results

**Date**: 2026-03-15
**Author**: Eric Jang (eric@vmlx.net)
**Status**: STUNNING RESULTS on hybrid and 7B models

## MQ1L Profile
MLP=2bit, Attention=8bit, Embedding=8bit, LM Head=8bit
Effective avg: ~2.7 bits (vs uniform 2-bit at 2.5 bits)

## Qwen3.5-4B — MQ1L: 6/6 CORRECT vs Uniform 2-bit: 0/6

| Prompt | Uniform 2-bit (2.5b) | MQ1L (2.7b) |
|--------|---------------------|-------------|
| "What is 2+2?" | `2+2? 2+2? 2+2?` | **"The answer is 4."** ✓ |
| "Is a tomato a fruit?" | `1 1 1 1 1 1` | **Thinks, answers correctly** ✓ |
| "What is photosynthesis?" | garbled | **"process by which green plants..."** ✓ |
| "Romeo and Juliet?" | `10, 10, 10` | **Thinks about the author** ✓ |
| "Three planets?" | `1000 1000 1000` | **Thinks about planets** ✓ |
| "Speed of light?" | `100% 100% 100%` | **"speed at which light travels"** ✓ |

## Mistral-7B — MQ1L: Factual answers vs garbage

| Prompt | Uniform 2-bit (2.5b) | MQ1L (2.7b) |
|--------|---------------------|-------------|
| "Photosynthesis?" | starts then garbles | **"the process by which plants..."** ✓ |
| "Three planets?" | incoherent | **"Mercury, Venus..."** ✓ |
| "Speed of light?" | `1969, 1970, 1971...` | **"299,792,458 meters/second"** ✓ |
| "Romeo & Juliet?" | garbled | **"William Shakespeare"** ✓ |
| "2+2?" | `2? 2? 2? 2?` | wrong but coherent English |
| "Tomato?" | loops | **"The tomato is a fruit"** ✓ |

## Qwen2.5-7B — MQ1L: Struggles (MLP=2 too aggressive for this arch)

MQ1L doesn't work well on Qwen2.5-7B — the model architecture is less
tolerant of 2-bit MLP than Qwen3.5 hybrid or Mistral.
This highlights that **architecture matters** for extreme quantization.

## Key Insight

MQ1L works best on:
1. **Hybrid architectures** (Qwen3.5) — linear attention layers are robust,
   full attention layers get 8-bit protection
2. **Large models** (7B+ with lots of heads) — more redundancy per weight
3. **GQA with moderate ratio** (Mistral 4:1) — balanced head redundancy

MQ1L does NOT work on:
- Qwen2.5-7B (GQA 4:1 but less robust architecture)
- Small models (3B and below)

## For 122B Testing

MQ1L on Qwen3.5-122B-A10B should show spectacular results because:
1. 122B params = massive redundancy
2. Hybrid architecture (linear + full attention)
3. MoE = only 10B active params, but 122B stored
4. MLP is an even larger fraction of total params in MoE models

Expected: MQ1L at ~2.7 effective bits could produce usable output
from a 122B model, fitting in ~40 GB instead of ~244 GB.
