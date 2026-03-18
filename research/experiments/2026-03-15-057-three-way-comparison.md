# Experiment 057: JANG vs MLX mixed_2_6 vs MLX Uniform — Honest Three-Way

**Date**: 2026-03-15
**Author**: Jinho Jang (eric@jangq.ai)
**Status**: IN PROGRESS (35B done, 122B converting)

## Why This Test Matters

MLX already supports mixed-precision quantization via `quant_predicate="mixed_2_6"`.
Our earlier benchmarks compared JANG only against uniform 2-bit — that wasn't the
fairest comparison. This test includes MLX's best mixed mode.

## MLX mixed_2_6 Strategy (from llama.cpp Q4_K_M)

- `v_proj` + `down_proj` at 6-bit in first/last 1/8 of layers + every 3rd layer
- `lm_head` at 6-bit
- Everything else at 2-bit (q_proj, k_proj, o_proj, gate_proj, up_proj)

## JANG_2L Strategy

- ALL q/k/v/o_proj at 8-bit (CRITICAL tier)
- ALL linear attention, embeddings, routers at 6-bit (IMPORTANT tier)
- ALL MLP/experts at 2-bit (COMPRESS tier)

## Key Difference

MLX mixed protects SOME layers SOME of the time (v_proj + down_proj in select layers).
JANG protects ALL attention ALL the time. JANG also knows about MoE expert routing,
linear attention (GatedDeltaNet), and MLA — MLX mixed doesn't.

## 35B Results (Qwen3.5-35B-A3B, temp=0.0)

| Metric | JANG_2L | MLX mixed_2_6 | MLX uniform 2-bit |
|--------|---------|---------------|-------------------|
| Disk size | 15 GB | 13 GB | 10 GB |
| GPU memory | 13.3 GB | 12.8 GB | 10.1 GB |
| Speed | 100 tok/s | 120 tok/s | 128 tok/s |

| Prompt | JANG_2L | MLX mixed_2_6 | MLX uniform |
|--------|---------|---------------|-------------|
| 2+2 | **"2+2 equals 4" ✅** | "2+2=4" then loops | Number spam |
| Tomato | Loops ❌ | Partial reasoning ⚠️ | Garbage |
| Photosynthesis | **"convert light energy" ✅** | "I cannot respond" ❌ | "6 6 6" garbage |
| Planets | **"Jupiter, Saturn, Uranus" ✅** | "Antina" loops ❌ | Number spam |
| Romeo | "Shakespeare" ⚠️ | Contradicts itself ❌ | Garbage |
| France | **"Paris" with details ✅** | Never answers ❌ | "Paris" partial |

| Method | Correct | Partial | Broken |
|--------|---------|---------|--------|
| **JANG_2L** | **4** | 1 | 1 |
| MLX mixed_2_6 | 0 | 1 | 5 |
| MLX uniform | 0 | 1 | 5 |

**JANG_2L: 4/6. MLX mixed_2_6: 0/6. MLX uniform: 0/6.**

MLX mixed_2_6 is barely better than uniform on this model — because Qwen3.5-35B
has 30 GatedDeltaNet linear attention layers that MLX mixed doesn't know about.
JANG's architecture-aware tier system correctly identifies and protects them.

## 122B Results

*Converting MLX mixed_2_6 for 122B — testing next...*
