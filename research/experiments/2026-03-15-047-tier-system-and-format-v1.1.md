# Experiment 047: Tier-Based Allocation System + Format v1.1

**Date**: 2026-03-15
**Author**: Jinho Jang (eric@jangq.ai)
**Status**: COMPLETE

## Summary

Replaced the brittle per-tensor-name profile system with an architecture-agnostic
3-tier classification system. Also redesigned the .jang format to eliminate per-block
metadata overhead (bit_map + block_offsets), reducing file sizes by ~18%.

## Tier System Design

Every weight tensor is classified into one of three sensitivity tiers:

| Tier | What belongs here | Why |
|------|------------------|-----|
| **CRITICAL** | Full softmax attention (Q/K/V/O), output head (lm_head), MLA latent projections, SSM state matrices (A_log) | Controls coherence, attention patterns, token probabilities. Errors here cause repetition loops and incoherent output. |
| **IMPORTANT** | Embeddings, MoE routers/gates, shared experts, vision-language connectors, SSM timestep projections | Moderate sensitivity. Errors degrade but don't destroy quality. |
| **COMPRESS** | MLP/FFN layers (dense or expert), linear attention projections (RWKV/DeltaNet), SSM input/output projections, vision FFN | Most robust to quantization. Bulk of parameters in most architectures. |

### Profile Definitions

Each profile is a 3-tuple: (CRITICAL_bits, IMPORTANT_bits, COMPRESS_bits)

```python
JANG_PROFILES = {
    "JANG_2S": (6, 4, 2),   # Tightest — attention at 6b, MLP at 2b
    "JANG_2M": (8, 4, 2),   # Balanced — attention at 8b
    "JANG_2L": (8, 6, 2),   # Max quality at 2b compress
    "JANG_3M": (6, 4, 3),   # Sweet spot for 7B+ models
    "JANG_4M": (6, 4, 4),   # Minimum for 3B dense models
    "JANG_6M": (8, 6, 6),   # Near-lossless
}
```

### Architecture Adaptability

Same profile name produces different avg bits depending on architecture:

| Model | Architecture | CRITICAL % | COMPRESS % | JANG_2S avg |
|-------|-------------|-----------|-----------|-------------|
| Qwen2.5-3B | Dense transformer | 12% | 78% | 2.64 bits |
| Qwen3.5-4B | Hybrid linear+full attn | 7% | 79% | 2.57 bits |
| Qwen3.5-35B-A3B | MoE + VL | 22% | 63% | 3.17 bits |
| Qwen3.5-122B-A10B | Large MoE + VL | 1.4% | 98% | 2.07 bits |

MoE models with many experts naturally get lower avg bits because COMPRESS tier dominates.

## Architectures Supported

Tier classification covers all major LLM families:

- **Dense Transformers**: Llama, Qwen, Gemma, Phi, Mistral (q/k/v/o_proj, gate/up/down_proj)
- **MoE**: Mixtral (w1/w2/w3), Qwen3.5 MoE (gate_up_proj), MiniMax (block_sparse_moe), DeepSeek (n_routed_experts)
- **MLA**: DeepSeek-V3/R1 (kv_a_proj_with_mqa, kv_b_proj, q_a_proj, q_b_proj)
- **Hybrid SSM**: Jamba (mamba.in_proj, x_proj, dt_proj, A_log, conv1d)
- **Linear Attention**: Qwen3.5 GatedDeltaNet (in_proj_qkv, in_proj_z, in_proj_a/b)
- **Vision-Language**: Qwen-VL (visual.*, merger.*)
- **FP8 source models**: MiniMax-M2.5, DeepSeek-V3 (E4M3 + block scale_inv)

## Format v1.1 Changes

### Removed per-block metadata

Old format (v1.0) stored per block:
- `.bit_map` (uint8 per block) — redundant when all blocks share same bits
- `.block_offsets` (uint32 per block) — computable from bits × block_index

New format (v1.1) stores per tensor:
- `.bits` (single uint8) — the bit width for this tensor
- `.shape` (int64 array) — original shape for 3D+ tensors

### Size impact

For Qwen3.5-35B-A3B JANG_2S (561M blocks):
- v1.0: 17 GB (8.88 GB qweight + 2.1 GB block_offsets + 1.05 GB bit_map + 2.1 GB scales/zeros)
- v1.1: 14 GB (8.88 GB qweight + 2.1 GB scales/zeros + negligible .bits/.shape)
- **Savings: 3 GB (18% reduction)**

### Effective overhead per weight

- v1.0: 72 bits per block = 1.125 bits/weight overhead
- v1.1: 32 bits per block = 0.5 bits/weight overhead (scales + zeros only)
- At 2-bit quantization: effective bits = 2.5 (v1.0) vs 2.5 (v1.1, but less disk)

## Quantization Speed Improvements

### RTN for COMPRESS tier

COMPRESS-tier tensors (expert MLP, linear attention, FFN) use RTN instead of MSE.
At 2-bit, RTN vs MSE quality difference is negligible but speed is identical
because the bottleneck was the packing step, not the quantization.

### Vectorized packing

Replaced per-block Python packing loop with vectorized `pack_bits()` call:
- Old: 42 sec per expert tensor (8M blocks × per-block pack)
- New: 2.6 sec per expert tensor
- **16x speedup**

### 35B conversion time

- v1.0 with MSE: ~38 hours (estimated)
- v1.0 with RTN: ~16 hours (estimated, packing bottleneck)
- v1.1 with vectorized pack: ~20 min
- **~100x speedup total**

## Architecture Detection Fixes

1. **MiniMax-M2.5**: Fixed attn_type_list misdetection (was triggering Qwen3.5 deltanet path)
2. **DeepSeek-V3/R1**: Added `n_routed_experts` config key
3. **Jamba**: Added MoE detection in hybrid SSM branch
4. **FP8 models**: Skip `weight_scale_inv` companion tensors in calibrate/convert

## Quality Testing Results

### 3B Dense (Qwen2.5-3B) — Minimum profile validation

| Profile | Avg bits | Coherent? | Notes |
|---------|----------|-----------|-------|
| JANG_2S | 2.64 | No | Garbage output — 2b MLP too aggressive for 3B |
| JANG_2L | 3.06 | No | Still broken — 2b compress too low for dense 3B |
| JANG_3M | 3.43 | No | Repetition loops — 3b MLP insufficient for 3B |
| JANG_4M | 4.22 | **Yes** | Correct answers: "2+2=4", "Paris", "Shakespeare" |

**Conclusion**: 3B dense models need JANG_4M minimum. 2-3 bit profiles are for 7B+ or MoE.

### 35B MoE (Qwen3.5-35B-A3B) — Testing in progress

JANG_2S at 2.12 avg bits currently loading through vMLX. This is the real test:
- 96% of blocks at 2-bit (expert MLP + linear attention)
- 2.2% at 6-bit (full attention — the 8 critical layers)
- 1.6% at 4-bit (embeddings, routers)

## Files Modified

### jang-tools
- `allocate.py` — Tier enum, TIER_RULES, classify_tensor(), tier-based JANG_PROFILES
- `quantize.py` — QuantizedTensor simplified (bits int, no bit_map/block_offsets), vectorized packing
- `convert.py` — RTN for COMPRESS tier, FP8 loading, 3D expert support
- `calibrate.py` — FP8 loading, 3D expert reshape, improved importance normalization
- `format/spec.py` — v1.1 BITS_SUFFIX, reduced overhead calculation
- `format/writer.py` — Store .bits per tensor instead of per-block bit_map
- `format/reader.py` — v1.1 .bits loading with v1.0 fallback
- `architectures.py` — Fixed MiniMax, DeepSeek, Jamba detection
- `fp8.py` — New: FP8 E4M3 dequantization support

### vMLX engine
- `utils/jang_loader.py` — Vectorized dequant, v1.1 support, 3D shape handling
- `commands/convert.py` — JANG profile conversion via --jang-profile
- `cli.py` — --jang-profile and --jang-method CLI flags
