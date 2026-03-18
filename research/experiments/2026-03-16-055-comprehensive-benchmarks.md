# Experiment 055: Comprehensive MMLU + HumanEval Benchmarks

**Date**: 2026-03-16
**Author**: Jinho Jang (eric@jangq.ai)
**Status**: IN PROGRESS

## Test Setup

- **Hardware**: M4 Max 128 GB (MacBook Pro)
- **MMLU**: 50 questions (10 subjects x 5), thinking disabled via `enable_thinking=False`, temp=0.0
- **HumanEval**: 20 problems, thinking disabled, temp=0.0, max_tokens=512
- **Quantization backend**: mx.quantize() with direct biases (no precision loss)

## THE KEY RESULTS

### 122B MoE at 2-bit — JANG_2S is the sweet spot

| Model | MMLU | HumanEval | Size | GPU | vs MLX 2-bit |
|-------|------|-----------|------|-----|-------------|
| **JANG_2S (6,4,2)** | **84%** | **pending** | **38 GB** | **44.3 GB** | **+28 pts, +6% size** |
| JANG_2M (8,4,2) | 84% | 90% | 42 GB | 44.7 GB | +28 pts, +17% size |
| MLX mixed_2_6 | 46% | — | 44 GB | 45 GB | -10 pts, +22% size |
| MLX 2-bit | 56% | — | 36 GB | 36 GB | baseline |

**JANG_2S: +28 MMLU at only 6% bigger. Smaller than MLX mixed_2_6.**

### 35B MoE at 4-bit — Budget-neutral beats MLX

| Model | MMLU | HumanEval | Size | GPU | vs MLX 4-bit |
|-------|------|-----------|------|-----|-------------|
| **JANG budget-4** | **84%** | **90%** | **16.7 GB** | **20.1 GB** | **+2 pts, -7% size** |
| MLX 4-bit | 82% | 95% | 18 GB | 18.2 GB | baseline |

**JANG budget-4: +2 MMLU, smaller file size. Same HumanEval within noise.**

### 35B MoE at 2-bit — JANG_2S dominates

| Model | MMLU | Size | GPU | vs MLX mixed |
|-------|------|------|-----|-------------|
| **JANG_2S (6,4,2)** | **62%** | **12 GB** | **12.8 GB** | **+28 pts, -8% size** |
| JANG_2L (8,6,2) | 56% | 13 GB | 13.3 GB | +22 pts, same size |
| MLX mixed_2_6 | 34% | 13 GB | 12.8 GB | baseline |

**JANG_2S: 62% vs 34% at smaller size. JANG_2S beats JANG_2L despite less attention boost.**

## Full 35B Comparison Table

| Model | Profile/Method | Avg Bits | MMLU | HumanEval | Size |
|-------|---------------|----------|------|-----------|------|
| MLX 4-bit | uniform | 4.0 | 82% | 95% | 18 GB |
| **JANG budget-4** | **budget** | **3.99** | **84%** | **90%** | **16.7 GB** |
| JANG_4S | (6,4,4) | 4.04 | 82% | — | 17 GB |
| JANG_4M v2 | (8,4,4) | 4.09 | 80% | 90% | 20 GB |
| JANG_4L | (8,6,4) | 4.17 | 82% | — | 21 GB |
| **JANG_2S** | **(6,4,2)** | **2.17** | **62%** | **—** | **12 GB** |
| JANG_2L v2 | (8,6,2) | 2.28 | 56% | — | 13 GB |
| MLX mixed_2_6 | varies | ~2.5 | 34% | 0% | 13 GB |

## Discovery: Budget-Neutral Allocation

The breakthrough approach: redistribute bits within the same total budget instead of adding bits.

**Old (profile-based)**: JANG_4M (8,4,4) = attention at 8-bit + MLP at 4-bit = BIGGER than MLX, no quality gain.

**New (budget-neutral)**: Target 4.0 avg bits. Boost attention 4->5-bit, compensate 3% of MLP 4->3-bit. Net: same/smaller size, +2 MMLU.

```
Budget-4 allocation on 35B MoE:
  94.8% of blocks: 4-bit (expert MLP — same as MLX)
   2.2% of blocks: 5-bit (attention — boosted)
   3.0% of blocks: 3-bit (least-important MLP — compensating)
  Average: 3.99 bits = smaller than MLX uniform 4-bit
```

## Discovery: JANG_2S beats JANG_2L

On 35B: JANG_2S (6,4,2) scored 62% vs JANG_2L (8,6,2) at 56%. The tighter profile wins because:
- Less MLP disruption (IMPORTANT at 4-bit vs 6-bit = less overhead)
- 6-bit attention is sufficient (8-bit doesn't add measurable quality)
- Smaller size = less overhead everywhere

On 122B: Both score 84% but JANG_2S is 4 GB smaller (38 vs 42 GB).

## Discovery: Thinking Mode Matters for Benchmarks

With `enable_thinking=True`: model generates long reasoning chains, often runs out of tokens.
With `enable_thinking=False`: model answers directly, much faster and more accurate on MMLU.

Anatomy test (5 questions): Thinking ON = 2/5, Thinking OFF = 5/5.

**All benchmarks use thinking OFF for fair comparison.**

## Pipeline Verification

JANG_4S (6,4,4) scored 82% = identical to MLX 4-bit uniform. Proves:
- uint8 -> uint32 repack is lossless
- Direct biases storage is lossless
- Loader correctly reconstructs QuantizedLinear layers
- Any quality difference is from bit allocation, not pipeline

## Bug Fixes Applied

1. **Direct biases**: Store MLX biases directly instead of zeros round-trip (eliminated precision loss)
2. **enable_thinking=False**: Use chat template to disable thinking for benchmarks
3. **shared_expert singular**: Fixed tier rule matching
4. **layer_types detection**: Fixed Qwen3.5 hybrid detection
5. **Vision tensor ValueError**: Fallback to RTN for non-64-divisible dims
6. **Budget-neutral allocator**: Per-tensor (fast) instead of per-block (561M blocks = hours)

## Models on HuggingFace

- [JANGQ-AI/Qwen3.5-122B-A10B-JANG_1L](https://huggingface.co/JANGQ-AI/Qwen3.5-122B-A10B-JANG_1L) — 73% MMLU, 51 GB
- [JANGQ-AI/Qwen3.5-35B-A3B-JANG-4bit](https://huggingface.co/JANGQ-AI/Qwen3.5-35B-A3B-JANG-4bit) — 84% MMLU, 16.7 GB (NEW)

## Ongoing

- 122B JANG_2S HumanEval — running
- MiniMax-M2.5 JANG_2L v2 — done (89 GB), needs MMLU testing on Mac Studio
- Qwen3.5-397B-A17B — downloading (751 GB / ~800 GB)
- 122B JANG_2S HuggingFace upload — queued
- 35B JANG_2S HuggingFace upload — queued

## Small/Hybrid Model Results (Non-MoE)

| Model | JANG_2S | MLX 2-bit | JANG Size | MLX Size | JANG GPU | Improvement |
|-------|---------|-----------|-----------|----------|----------|-------------|
| Qwen3.5-4B | 28% | 14% | 1.6 GB | 1.3 GB | 1.9 GB | +14 pts (2x) |
| Qwen3.5-9B | 36% | 18% | 3.5 GB | 2.6 GB | 4.3 GB | +18 pts (2x) |

Both are hybrid (GatedDeltaNet + full attention) but NOT MoE. JANG_2S on these has 2.9-3.1 avg bits
(vs 2.0 for MLX) because attention is ~30% of params on dense models. Still, JANG doubles
the MMLU score — the attention protection prevents the catastrophic failure mode at 2-bit.

### Mac Neo (8 GB) Recommendations

For users with 8 GB Apple Silicon:
- **Qwen3.5-9B JANG_2S** (4.3 GB GPU): 36% MMLU — actual knowledge, fits with KV cache
- **Qwen3.5-4B JANG_2S** (1.9 GB GPU): 28% MMLU — smallest coherent model
- MLX 2-bit alternatives: 14-18% MMLU (barely above random)

### Complete Results Across All Models

| Model | Type | JANG_2S MMLU | MLX 2-bit MMLU | JANG Size | MLX Size |
|-------|------|-------------|----------------|-----------|----------|
| Qwen3.5-4B | hybrid | 28% | 14% | 1.6 GB | 1.3 GB |
| Qwen3.5-9B | hybrid | 36% | 18% | 3.5 GB | 2.6 GB |
| Qwen3.5-35B-A3B | MoE | 62% | ~20% | 12 GB | 10 GB |
| Qwen3.5-122B-A10B | MoE | 84% | 56% | 38 GB | 36 GB |

JANG doubles MLX 2-bit on every model. Improvement is largest on MoE (3x on 35B).

## 122B JANG_4K — The Flagship Result

| Model | MMLU | HumanEval | Size | GPU |
|-------|------|-----------|------|-----|
| **JANG_4K** | **94%** | **95%** | **69 GB** | **71 GB** |
| MLX 4-bit | 90% | — | 64 GB | 64 GB |
| JANG_2S | 84% | 90% | 38 GB | 44 GB |
| MLX 2-bit | 56% | — | 36 GB | 36 GB |

JANG_4K beats MLX 4-bit by +4 MMLU (94% vs 90%) at only 5 GB larger (69 vs 64 GB).
The K-quant redistribution on MoE gives near-perfect scores.

## Non-MoE K-quant Results (JANG_4K does NOT help)

| Model | Type | JANG_4K | MLX 4-bit | Verdict |
|-------|------|---------|-----------|---------|
| 4B | hybrid | 70% (2.4 GB) | 72% (2.2 GB) | MLX wins |
| 9B | hybrid | 80% (5.2 GB) | 82% (4.7 GB) | MLX wins |
| 35B MoE | MoE | 84% (16.7 GB) | 82% (18 GB) | **JANG wins** |
| 122B MoE | MoE | 94% (69 GB) | 90% (64 GB) | **JANG wins** |

Confirmed: JANG_4K only beats MLX on MoE models. Non-MoE models should use MLX 4-bit.

## VL Status

Code fixed (skip quantizing vision, rename tensors for mlx-vlm) but current models
have quantized vision. VL requires reconversion with updated code. All current HF
models are text-only.

## Final Model Lineup on HuggingFace

| Repo | MMLU | HumanEval | Size | Status |
|------|------|-----------|------|--------|
| Qwen3.5-122B-A10B-JANG_4K | 94% | 95% | 69 GB | Uploading |
| Qwen3.5-122B-A10B-JANG_2S | 84% | 90% | 38 GB | Live |
| Qwen3.5-122B-A10B-JANG_1L | 73% | — | 51 GB | Live (old) |
| Qwen3.5-35B-A3B-JANG_4K | 84% | 90% | 16.7 GB | Live |
| Qwen3.5-35B-A3B-JANG_2S | 62% | — | 12 GB | Live |

## 200-Question MMLU Results (Proper Benchmark)

Previous 50q results were inflated. 200q (20 per subject) is the real benchmark.

### 122B Results at 200q

| Subject | JANG_4K (69 GB) | JANG_2S (38 GB) |
|---------|----------------|----------------|
| abstract_algebra | 16/20 | 9/20 |
| anatomy | 19/20 | 18/20 |
| astronomy | 19/20 | 20/20 |
| college_computer_science | 15/20 | 14/20 |
| college_physics | 14/20 | 15/20 |
| high_school_biology | 19/20 | 19/20 |
| high_school_chemistry | 18/20 | 18/20 |
| high_school_mathematics | 14/20 | 11/20 |
| logical_fallacies | 19/20 | 16/20 |
| world_religions | 19/20 | 18/20 |
| **TOTAL** | **172/200 (86%)** | **158/200 (79%)** |

### 50q vs 200q Comparison (Inflation Check)

| Model | 50q Score | 200q Score | Difference |
|-------|-----------|-----------|------------|
| 122B JANG_4K | 94% | 86% | -8 pts (inflated) |
| 122B JANG_2S | 84% | 79% | -5 pts (inflated) |

50q tests are unreliable — 5-8 point inflation from small sample size.
All HF model cards will be updated with 200q scores.

### Pending 200q Tests

- 122B MLX 4-bit: converting
- 122B MLX 2-bit: converting
- 35B JANG_4K: queued
- 35B JANG_2S: queued
- 35B MLX 4-bit: converting
- 35B MLX mixed_2_6: TBD

## Complete 200q Per-Subject Results

### 122B JANG_4K vs MLX 4-bit (200q)

| Subject | JANG_4K (69 GB) | MLX 4-bit (64 GB) |
|---------|----------------|-------------------|
| abstract_algebra | **16/20** | 15/20 |
| anatomy | **19/20** | 18/20 |
| astronomy | 19/20 | 19/20 |
| college_computer_science | 15/20 | 15/20 |
| college_physics | 14/20 | 14/20 |
| high_school_biology | 19/20 | 19/20 |
| high_school_chemistry | 18/20 | 18/20 |
| high_school_mathematics | 14/20 | 14/20 |
| logical_fallacies | 19/20 | 19/20 |
| world_religions | 19/20 | 19/20 |
| **TOTAL** | **172/200 (86%)** | **170/200 (85%)** |

### 122B JANG_2S vs MLX 2-bit (200q)

| Subject | JANG_2S (38 GB) | MLX 2-bit (36 GB) |
|---------|----------------|-------------------|
| abstract_algebra | 9/20 | 9/20 |
| anatomy | **18/20** | 11/20 |
| astronomy | **20/20** | 16/20 |
| college_computer_science | **14/20** | 8/20 |
| college_physics | **15/20** | 10/20 |
| high_school_biology | **19/20** | 15/20 |
| high_school_chemistry | **18/20** | 13/20 |
| high_school_mathematics | **11/20** | 4/20 |
| logical_fallacies | **16/20** | 13/20 |
| world_religions | **18/20** | 14/20 |
| **TOTAL** | **158/200 (79%)** | **113/200 (56.5%)** |

JANG_2S wins 9 out of 10 subjects. Ties on abstract_algebra.

### 35B JANG_4K vs MLX 4-bit (200q)

| Subject | JANG_4K (16.7 GB) | MLX 4-bit (18 GB) |
|---------|------------------|-------------------|
| abstract_algebra | **12/20** | 10/20 |
| anatomy | **17/20** | 16/20 |
| astronomy | 18/20 | 18/20 |
| college_computer_science | 14/20 | **15/20** |
| college_physics | **14/20** | 13/20 |
| high_school_biology | 18/20 | 18/20 |
| high_school_chemistry | 17/20 | 17/20 |
| high_school_mathematics | **10/20** | 8/20 |
| logical_fallacies | 18/20 | **19/20** |
| world_religions | 17/20 | 17/20 |
| **TOTAL** | **155/200 (77.5%)** | **151/200 (75.5%)** |

JANG_4K wins 4 subjects, loses 2, ties 4. Net +4 questions.

### 35B JANG_2S (200q, no MLX 200q baseline yet)

| Subject | JANG_2S (12 GB) |
|---------|----------------|
| abstract_algebra | 8/20 |
| anatomy | 14/20 |
| astronomy | 19/20 |
| college_computer_science | 14/20 |
| college_physics | 11/20 |
| high_school_biology | 16/20 |
| high_school_chemistry | 14/20 |
| high_school_mathematics | 5/20 |
| logical_fallacies | 14/20 |
| world_religions | 16/20 |
| **TOTAL** | **131/200 (65.5%)** |

MLX mixed_2_6 scored 34% at 50q (200q not tested).

### 122B JANG_1L (original 200q from earlier session)

| Subject | JANG_1L (51 GB) | MLX mixed_2_6 (44 GB) | MLX 2-bit (36 GB) |
|---------|----------------|----------------------|-------------------|
| abstract_algebra | 9/20 | 7/20 | — |
| anatomy | 15/20 | 9/20 | — |
| astronomy | 17/20 | 12/20 | — |
| college_computer_science | 16/20 | 8/20 | — |
| college_physics | 11/20 | 4/20 | — |
| high_school_biology | 18/20 | 12/20 | — |
| high_school_chemistry | 19/20 | 12/20 | — |
| high_school_mathematics | 7/20 | 3/20 | — |
| logical_fallacies | 17/20 | 11/20 | — |
| world_religions | 17/20 | 14/20 | — |
| **TOTAL** | **146/200 (73%)** | **92/200 (46%)** | **56.5%** |

## Summary Table (All 200q)

| Model | Profile | MMLU 200q | Size | GPU | vs MLX |
|-------|---------|----------|------|-----|--------|
| 122B | JANG_4K | 86.0% | 69 GB | 71 GB | +1 vs MLX 4-bit (85%) |
| 122B | JANG_2S | 79.0% | 38 GB | 44 GB | +22.5 vs MLX 2-bit (56.5%) |
| 122B | JANG_1L | 73.0% | 51 GB | 46 GB | +27 vs MLX mixed (46%) |
| 35B | JANG_4K | 77.5% | 16.7 GB | 20.1 GB | +2 vs MLX 4-bit (75.5%) |
| 35B | JANG_2S | 65.5% | 12 GB | 12.8 GB | +31.5 vs MLX mixed (34% at 50q) |

## MiniMax-M2.5 Results (200q MMLU)

### MiniMax 2-bit comparison

| Model | MMLU (200q) | Size | GPU | Speed |
|-------|-----------|------|-----|-------|
| **JANG_2L** | **74.0%** | **89 GB** | **82.5 GB** | **0.9s/q** |
| MLX 2-bit | 25.0% | 67 GB | 66.6 GB | 2.1s/q |

JANG triples MLX score (74% vs 25%) and is 2x faster.

### MiniMax JANG_2L Per-Subject (200q)

| Subject | JANG_2L | MLX 2-bit |
|---------|---------|-----------|
| abstract_algebra | **10/20** | 5/20 |
| anatomy | **15/20** | 5/20 |
| astronomy | **20/20** | 4/20 |
| college_computer_science | **13/20** | 6/20 |
| college_physics | **13/20** | 6/20 |
| high_school_biology | **18/20** | 6/20 |
| high_school_chemistry | **18/20** | 5/20 |
| high_school_mathematics | **8/20** | 3/20 |
| logical_fallacies | **18/20** | 5/20 |
| world_religions | **15/20** | 5/20 |
| **TOTAL** | **148/200 (74%)** | **50/200 (25%)** |

JANG wins all 10 subjects.

### MiniMax MLX 4-bit Per-Subject (200q)

| Subject | MLX 4-bit (120 GB) | MLX 2-bit (67 GB) |
|---------|-------------------|-------------------|
| abstract_algebra | 3/20 | 5/20 |
| anatomy | 7/20 | 5/20 |
| astronomy | 7/20 | 4/20 |
| college_computer_science | 4/20 | 6/20 |
| college_physics | 8/20 | 6/20 |
| high_school_biology | 4/20 | 6/20 |
| high_school_chemistry | 4/20 | 5/20 |
| high_school_mathematics | 6/20 | 3/20 |
| logical_fallacies | 5/20 | 5/20 |
| world_religions | 5/20 | 5/20 |
| **TOTAL** | **53/200 (26.5%)** | **50/200 (25%)** |

MLX is completely broken on MiniMax at ALL bit levels. 25-26% = random guessing.
JANG is the ONLY way to run MiniMax quantized on Apple Silicon.

### MiniMax MLX 3-bit (200q) — Also Broken

| Subject | JANG_2L | MLX 4-bit | MLX 3-bit | MLX 2-bit |
|---------|---------|-----------|-----------|-----------|
| abstract_algebra | **10/20** | 3/20 | 2/20 | 5/20 |
| anatomy | **15/20** | 7/20 | 5/20 | 5/20 |
| astronomy | **20/20** | 7/20 | 6/20 | 4/20 |
| college_computer_science | **13/20** | 4/20 | 5/20 | 6/20 |
| college_physics | **13/20** | 8/20 | 6/20 | 6/20 |
| high_school_biology | **18/20** | 4/20 | 5/20 | 6/20 |
| high_school_chemistry | **18/20** | 4/20 | 5/20 | 5/20 |
| high_school_mathematics | **8/20** | 6/20 | 6/20 | 3/20 |
| logical_fallacies | **18/20** | 5/20 | 4/20 | 5/20 |
| world_religions | **15/20** | 5/20 | 5/20 | 5/20 |
| **TOTAL** | **148/200 (74%)** | **53/200 (26.5%)** | **49/200 (24.5%)** | **50/200 (25%)** |

MLX produces garbage on MiniMax at ALL bit levels (2/3/4-bit all ~25%).
Root cause: MLX generates meta-commentary ("The user asks...") instead of direct answers.
JANG is the only working quantized MiniMax on Apple Silicon.
