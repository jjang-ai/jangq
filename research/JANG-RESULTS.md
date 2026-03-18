# JANG Results — Empirical Evidence

> Created by Jinho Jang (eric@jangq.ai)
> Date: 2026-03-14, updated 2026-03-17

## What is JANG?

JANG is an adaptive mixed-precision quantization format + runtime for Apple Silicon.
Instead of quantizing all weights at the same bit width (like MLX uniform quantization),
JANG classifies every tensor into a sensitivity tier and assigns bits accordingly:

- **CRITICAL** (full attention, MoE routers, output head, Mamba A/D): 6-8 bit
- **IMPORTANT** (embeddings, linear attention/DeltaNet, shared experts): 4-6 bit
- **COMPRESS** (MLP/FFN, MoE experts, vision FFN): 2-3 bit

Works for any architecture: dense transformers, MoE, hybrid SSM (Mamba), MLA (DeepSeek), GatedDeltaNet, VLM, FP8 source (MiniMax/DeepSeek).

## 200-Question MMLU Results

All benchmarks: 200 questions, 20 per subject × 10 subjects.
Chat template with `enable_thinking=False`, `temp=0.0`.

### MoE at 4-bit: JANG_4K beats MLX

| Model | JANG_4K | MLX 4-bit | JANG Size | MLX Size |
|-------|---------|-----------|-----------|----------|
| Qwen3.5-122B | **86%** | 85% | 69 GB | 64 GB |
| Qwen3.5-35B | **77.5%** | 75.5% | **16.7 GB** | 18 GB |

### MoE at 2-bit: JANG dominates

| Model | JANG_2S | MLX 2-bit | JANG Size | MLX Size |
|-------|---------|-----------|-----------|----------|
| Qwen3.5-122B | **79%** | 56.5% | 38 GB | 36 GB |
| Qwen3.5-35B | **65.5%** | ~20% | 12 GB | 10 GB |

### MiniMax: JANG is the ONLY working option

| Model | JANG_2L | MLX 4-bit | MLX 3-bit | MLX 2-bit |
|-------|---------|-----------|-----------|-----------|
| MiniMax-M2.5 | **74%** | 26.5% | 24.5% | 25% |

MLX is broken on MiniMax at ALL bit levels (~25% = random). JANG scores 74%.

### Dense/Hybrid: Full Comparison

#### Qwen3.5-4B (hybrid SSM + full attention, VLM)

| Model | MMLU | Size | GPU |
|-------|------|------|-----|
| **JANG_2S** (2.95-bit) | **28.5%** | 2.4 GB | 1.7 GB |
| JANG_3K (2.97-bit) | 29.5% | 2.4 GB | 1.7 GB |
| JANG_4K (3.97-bit) | 62.5% | 2.9 GB | 2.2 GB |
| MLX 2-bit | 12.5% | 1.2 GB | 1.2 GB |
| MLX 3-bit | 48.5% | 1.7 GB | 1.7 GB |
| **MLX 4-bit** | **67.0%** | 2.2 GB | 2.2 GB |

Per-subject breakdown (out of 20):

| Subject | JANG_2S | JANG_3K | JANG_4K | MLX 2 | MLX 3 | MLX 4 |
|---------|---------|---------|---------|-------|-------|-------|
| Abstract Algebra | 6 | 5 | 4 | 3 | 3 | 6 |
| Anatomy | 7 | 6 | 15 | 1 | 11 | 15 |
| Astronomy | 4 | 7 | 18 | 1 | 13 | 17 |
| College CS | 8 | 7 | 7 | 1 | 7 | 11 |
| College Physics | 7 | 8 | 13 | 7 | 8 | 13 |
| HS Biology | 6 | 6 | 17 | 5 | 13 | 17 |
| HS Chemistry | 6 | 5 | 11 | 3 | 7 | 12 |
| HS Mathematics | 5 | 5 | 8 | 1 | 6 | 9 |
| Logical Fallacies | 4 | 4 | 15 | 1 | 13 | 17 |
| World Religions | 4 | 6 | 17 | 2 | 16 | 17 |
| **Total** | **57** | **59** | **125** | **25** | **97** | **134** |

#### Qwen3.5-9B (hybrid SSM + full attention, VLM)

| Model | MMLU | Size | GPU |
|-------|------|------|-----|
| **JANG_2S** (3.16-bit) | **25.5%** | 4.8 GB | 3.8 GB |
| JANG_3K (2.91-bit) | 25.5% | 4.5 GB | 3.6 GB |
| JANG_4K (3.91-bit) | 70.5% | 5.6 GB | 4.6 GB |
| MLX 2-bit | 22.0% | 2.6 GB | 2.6 GB |
| MLX 3-bit | 64.0% | 3.7 GB | 3.6 GB |
| **MLX 4-bit** | **72.5%** | 4.7 GB | 4.7 GB |

Per-subject breakdown (out of 20):

| Subject | JANG_2S | JANG_3K | JANG_4K | MLX 2 | MLX 3 | MLX 4 |
|---------|---------|---------|---------|-------|-------|-------|
| Abstract Algebra | 3 | 4 | 11 | 4 | 8 | 11 |
| Anatomy | 4 | 6 | 16 | 6 | 13 | 15 |
| Astronomy | 4 | 3 | 18 | 5 | 16 | 20 |
| College CS | 6 | 8 | 12 | 7 | 10 | 13 |
| College Physics | 6 | 6 | 13 | 6 | 12 | 13 |
| HS Biology | 5 | 6 | 17 | 4 | 19 | 18 |
| HS Chemistry | 7 | 4 | 16 | 4 | 15 | 14 |
| HS Mathematics | 7 | 6 | 7 | 2 | 5 | 9 |
| Logical Fallacies | 4 | 1 | 16 | 3 | 16 | 16 |
| World Religions | 5 | 7 | 15 | 3 | 14 | 16 |
| **Total** | **51** | **51** | **141** | **44** | **128** | **145** |

## Key Findings

### Where JANG wins:
- **2-bit on ALL models**: JANG_2S always beats MLX 2-bit (+16 on 4B, +3.5 on 9B, +22.5 on 122B, +45 on 35B)
- **MoE at all bit levels**: JANG_4K beats MLX 4-bit on 122B (+1%) and 35B (+2%)
- **MiniMax**: JANG is the ONLY working option (MLX broken at ALL bit levels)
- **Extreme compression (2-bit MoE)**: JANG_2S 79% vs MLX 56.5% on 122B

### Where MLX uniform wins:
- **Dense/hybrid at 3-bit**: MLX 3-bit beats JANG_3K (48.5% vs 29.5% on 4B, 64% vs 25.5% on 9B)
- **Dense/hybrid at 4-bit**: MLX 4-bit beats JANG_4K (67% vs 62.5% on 4B, 72.5% vs 70.5% on 9B)

### Why:
Dense models have 15-25% CRITICAL parameters (attention). Boosting CRITICAL to 8-bit and downgrading COMPRESS to compensate hurts dense models because the COMPRESS tier covers a smaller fraction. On MoE models, CRITICAL is <5% of params, so the boost is nearly free.

**JANG's value: 2-bit (where uniform fails) + MoE (where CRITICAL is <5%) + broken architectures (MiniMax).**

## VLM Support

All Qwen3.5 models are VLMs (Qwen3_5ForConditionalGeneration). JANG preserves all 297-333 vision tensors. VLM inference works with `load_jang_vlm_model()` via mlx-vlm.

Tested: 4B and 9B JANG models process images correctly through the vision encoder.

## Methodology

- 200 questions: 20 per subject × 10 subjects
- Subjects: abstract_algebra, anatomy, astronomy, college_computer_science, college_physics, high_school_biology, high_school_chemistry, high_school_mathematics, logical_fallacies, world_religions
- Chat template with `enable_thinking=False`
- Temperature: 0.0 (greedy)
- Max tokens: 20 per question
- MMLU dataset: cais/mmlu (HuggingFace, test split)
- MacBook Pro M4 Max 128 GB (4B/9B), Mac Studio M4 Ultra 256 GB (35B/122B/MiniMax)
