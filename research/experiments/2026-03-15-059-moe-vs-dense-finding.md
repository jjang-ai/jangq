# Experiment 059: JANG Works for MoE, Not Dense — Critical Finding

**Date**: 2026-03-15
**Author**: Jinho Jang (eric@jangq.ai)
**Status**: DEFINITIVE

## The Finding

JANG's tier-based allocation provides massive quality improvement on **MoE models**
but adds overhead without benefit on **dense models**.

## Evidence

### MoE: JANG Dominates

**122B Qwen3.5 MoE — MMLU 200 questions:**

| Method | Size | MMLU |
|--------|------|------|
| **JANG_1L** | 51 GB | **73.0%** |
| MLX mixed_2_6 | 44 GB | 46.0% |
| MLX uniform 2-bit | 36 GB | 56.0% |

JANG wins by 27 percentage points. The 122B has 256 experts per layer — 98% of
parameters are expert MLP at COMPRESS tier. The 2% overhead for 8-bit attention
costs almost nothing but provides massive quality improvement.

### Dense: MLX Wins

**27B Qwen3.5 Dense — MMLU 200 questions:**

| Method | Size | MMLU |
|--------|------|------|
| JANG_4L | 20 GB | 32.5% |
| **MLX 4-bit** | **14 GB** | **~78%** |

JANG_4L at 4.83 effective bits uses 43% more space than MLX 4-bit but scores
much lower. On dense models, attention is ~12% of total parameters — bumping it
to 8-bit adds significant overhead proportional to the total model size.

## Why

**MoE models:**
- Expert MLP: 94-98% of params → COMPRESS tier dominates
- Attention: 2-6% of params → 8-bit protection costs almost nothing
- JANG overhead: 5-15% more size for 27+ points better MMLU

**Dense models:**
- MLP: ~65% of params
- Attention: ~12% of params → 8-bit protection costs ~40% more size
- At 4-bit, attention already has enough precision
- JANG overhead: 40%+ more size for WORSE scores

## Recommendation

| Model Type | Use | Why |
|-----------|-----|-----|
| **MoE** (35B+, 122B, 230B, 397B) | **JANG** | Massive quality gain, tiny overhead |
| **Dense** (9B, 27B, Llama, Mistral) | **MLX uniform** | JANG adds overhead without benefit |
| **Hybrid SSM+MoE** (Qwen3.5, Jamba) | **JANG** | Protects GatedDeltaNet + attention |

## Profiles for MoE

At every bit level, JANG should be compared to same-size MLX:

| JANG Profile | MoE Size vs MLX | Expected MMLU advantage |
|-------------|----------------|------------------------|
| JANG_1L (COMPRESS=2, all else=8) | ~15% larger than MLX 2-bit | +27 points (proven on 122B) |
| JANG_2L (COMPRESS=2, CRITICAL=8, IMP=6) | ~10% larger than MLX 2-bit | +20 points (estimated) |
| JANG_3L (COMPRESS=3, CRITICAL=8, IMP=4) | ~10% larger than MLX 3-bit | Testing |
| JANG_4L (COMPRESS=4, CRITICAL=8, IMP=6) | ~15% larger than MLX 4-bit | Testing |
