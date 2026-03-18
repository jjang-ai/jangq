# Experiment 050: Full JANG Development Session Log

**Date**: 2026-03-15
**Author**: Jinho Jang (eric@jangq.ai)
**Status**: IN PROGRESS

## Session Overview

Complete development session building JANG from quantization tools through inference testing.
Everything done in one session: format design, architecture support, conversions, testing.

## Timeline

### Phase 1: Transfer & Initial Setup
- Transferred Qwen3.5-35B-A3B (67 GB) and 122B-A10B (233 GB) FP16 from Mac Studio via TB5
- TB5 HTTP streaming: ~8 GB/s, 300 GB in under a minute
- Created venv, installed jang-tools

### Phase 2: Tier-Based Allocation System
- Replaced brittle per-tensor-name profiles with 3-tier architecture-agnostic system
- CRITICAL (attention, output head) / IMPORTANT (embeddings, routers, linear attn) / COMPRESS (MLP, experts)
- Profiles become simple 3-tuples: JANG_2L = (8, 6, 2)
- Works for ANY architecture — tested on Qwen3.5 (MoE+linear attn+VL), Qwen2.5 (dense), MiniMax (MoE)

### Phase 3: 3D Expert Tensor Support
- Qwen3.5 stores MoE experts as 3D tensors [256, out, in]
- Added reshape-to-2D for quantization, store original shape for restore
- Added `.shape` companion tensor in format
- conv1d [C,1,4] and patch_embed [1152,3,2,16,16] → passthrough as fp16

### Phase 4: Format v1.1 — Eliminated Per-Block Overhead
- Removed `bit_map` (per-block uint8) → `.bits` (single uint8 per tensor)
- Removed `block_offsets` (per-block uint32) → computable at runtime
- 35B: 17 GB → 14 GB (18% reduction)
- Overhead: 4 bytes/block (scales+zeros) vs 9 bytes/block (old)

### Phase 5: Performance Optimization
- **RTN for COMPRESS tier**: MSE only on CRITICAL/IMPORTANT (attention/embeddings)
- **Vectorized packing**: Replaced per-block Python loop with vectorized numpy
  - 2-bit: 0.01s for 8M values (fast path, 4 values/byte)
  - 3-bit: 0.06s for 8M values (vectorized scatter, was ~40s with Python loop)
  - **600x speedup** on 3-bit packing
- **16x speedup** on per-tensor quantization overall

### Phase 6: Architecture Detection Fixes
- MiniMax-M2.5: Fixed attn_type_list misdetection as Qwen3.5 deltanet
- DeepSeek-V3/R1: Added n_routed_experts config key
- Jamba: Added MoE detection in hybrid SSM branch
- FP8 E4M3: Added dequant support for MiniMax/DeepSeek source models
- weight_scale_inv: Skip FP8 companion tensors in calibrate/convert

### Phase 7: vMLX Integration
- jang_loader.py: Vectorized dequantization, v1.1 format, 3D expert shape handling
- convert.py: --jang-profile flag for JANG conversion from UI/CLI
- cli.py: --jang-profile, --jang-method flags
- models.ts: Shows "JANG_2L (2.28b)" in model list
- ModelConverter.tsx: All 11 JANG profiles available in converter UI
- Full OpenAI API tested: /v1/chat/completions works with JANG models
- Anthropic /v1/messages works
- Streaming, tool_calls, reasoning_content all working
- Bench command fixed to use JANG loader
- Missing jjqf_config.json added to detection

### Phase 8: Quality Testing

#### Qwen2.5-3B (Dense Transformer)
| Profile | Bits | Coherent? |
|---------|------|-----------|
| JANG_2S | 2.64 | No — 2b MLP too aggressive for 3B dense |
| JANG_2L | 3.06 | No — still too low |
| JANG_3M | 3.43 | No — repetition loops |
| **JANG_4M** | **4.22** | **Yes — correct answers** |

#### Qwen3.5-35B-A3B (MoE + GatedDeltaNet + VL)
| Profile | Bits | Size | Coherent? | Score |
|---------|------|------|-----------|-------|
| JANG_2S v1 | 2.12 | 8.88 GB | Partial — correct starts, then loops | 0/6 |
| JANG_2L v1 | 2.20 | 9.20 GB | Partial — photosynthesis worked | 1/6 |
| **JANG_2L v2** | **2.28** | **9.55 GB** | **5/6 correct, detailed answers** | **5/6** |
| JANG_3M | 3.10 | 12.99 GB | Testing... | TBD |

Key finding: GatedDeltaNet linear attention needs IMPORTANT tier (6-bit), not COMPRESS (2-bit).

#### MiniMax-M2.5 (230B MoE, FP8 source)
- JANG_2S converting on Mac Studio (~86% done)
- 47,928 tensors, 256 experts × 62 layers
- Expected: ~2.05 avg bits, ~30 GB

#### Qwen3.5-122B-A10B (MoE + GatedDeltaNet + VL)
- JANG_2L converting locally
- Expected: ~2.1 avg bits, ~30 GB

### Phase 9: vMLX Integration Audit
Comprehensive audit found 5 bugs, 2 HIGH fixed:
1. FIXED: jjqf_config.json missing from JANG_CONFIG_FILENAMES
2. FIXED: bench command bypassed JANG loader (crashed on JANG models)
3. PARTIAL: SessionCard badge uses path regex not jang_config.json
4. LOW: No smoke test after JANG conversion
5. LOW: Memory table doesn't highlight JANG bit width

## Models Produced

| Model | Profile | Bits | Size | Status |
|-------|---------|------|------|--------|
| Qwen2.5-0.5B-JANG_2S | JANG_2S | 2.91 | 0.17 GB | Done, tested (too small for quality) |
| Qwen2.5-3B-JANG_2S | JANG_2S | 2.64 | 0.95 GB | Done, broken at this size |
| Qwen2.5-3B-JANG_2L | JANG_2L | 3.06 | 1.10 GB | Done, broken at this size |
| Qwen2.5-3B-JANG_3M | JANG_3M | 3.43 | 1.23 GB | Done, broken at this size |
| Qwen2.5-3B-JANG_4M | JANG_4M | 4.22 | 1.52 GB | Done, WORKING |
| Qwen3.5-35B-JANG_2S | JANG_2S | 2.12 | 8.88 GB | Done, partial coherence |
| Qwen3.5-35B-JANG_2L | JANG_2L | 2.20 | 9.20 GB | Done, partial coherence |
| **Qwen3.5-35B-JANG_2L_v2** | **JANG_2L** | **2.28** | **9.55 GB** | **Done, 5/6 CORRECT** |
| Qwen3.5-35B-JANG_3M | JANG_3M | 3.10 | 12.99 GB | Done, testing |
| Qwen3.5-122B-JANG_2L | JANG_2L | ~2.1 | ~30 GB | Converting |
| MiniMax-M2.5-JANG_2S | JANG_2S | ~2.05 | ~30 GB | Converting on Mac Studio |

## Key Learnings

1. **Dense 3B models need JANG_4M minimum** — MLP doesn't have enough redundancy for 2-3 bit
2. **MoE models shine at 2-bit** — experts are highly redundant (256 experts, 8 active)
3. **GatedDeltaNet linear attention is NOT as robust as assumed** — needs 6-bit (IMPORTANT), not 2-bit (COMPRESS)
4. **Full softmax attention is the most critical** — 8-bit keeps attention patterns stable
5. **Packing is the bottleneck, not quantization** — vectorized packing gave 600x speedup
6. **Format overhead matters** — eliminating per-block metadata saved 18% on disk
7. **TB5 is insane** — 233 GB in ~30 seconds

## Files Modified (Complete List)

### jang-tools/jang_tools/
- allocate.py — Tier system, TIER_RULES, tier-based profiles, classify_tensor()
- quantize.py — Simplified QuantizedTensor, vectorized packing, 3D reshape
- convert.py — RTN for COMPRESS, FP8 support, profile in jang_config, 3D support
- calibrate.py — FP8 loading, 3D reshape, importance normalization fix
- pack.py — Vectorized general-case packing (3/5/6-bit)
- fp8.py — NEW: FP8 E4M3 dequantization
- architectures.py — MiniMax, DeepSeek, Jamba detection fixes
- format/spec.py — v1.1 BITS_SUFFIX, reduced overhead
- format/writer.py — .bits per tensor, .shape storage
- format/reader.py — v1.1 loading with v1.0 fallback

### vmlx_engine/
- utils/jang_loader.py — Vectorized dequant, v1.1 format, 3D handling, jjqf config
- utils/tokenizer.py — JANG routing (unchanged, already correct)
- commands/convert.py — --jang-profile JANG conversion
- cli.py — JANG flags, bench command fix

### panel/ (Electron UI)
- src/main/ipc/models.ts — Profile name display
- (SessionCard.tsx badge is PARTIAL — noted for future fix)

### Research
- experiments/046 — vMLX integration test
- experiments/047 — Tier system and format v1.1
- experiments/048 — 35B MoE inference results
- experiments/049 — MiniMax conversion
- experiments/050 — This full session log
