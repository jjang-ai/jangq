# JANG — Remaining Work (2026-03-17)

## CRITICAL: VL Files Missing on ALL JANG Models

Every Qwen3.5 model is a VLM but our JANG models are MISSING these files:
- `preprocessor_config.json` — required for image processing
- `video_preprocessor_config.json` — required for video processing

These must be copied from the FP16 source model for each model family.

### Status:
| Model | VL Files | chat_template | Tested VL |
|-------|----------|---------------|-----------|
| 4B JANG_4S | MISSING | in tokenizer_config.json | YES (works) |
| 4B JANG_4K | MISSING | in tokenizer_config.json | YES (works) |
| 9B JANG_4S | MISSING | in tokenizer_config.json | YES (works) |
| 9B JANG_4K | MISSING | in tokenizer_config.json | YES (works) |
| 35B JANG_4K | MISSING | in tokenizer_config.json | NOT TESTED |
| 35B JANG_2S | MISSING | in tokenizer_config.json | NOT TESTED |
| 122B JANG_4K | MISSING | in tokenizer_config.json | NOT TESTED |
| 122B JANG_2S | MISSING | in tokenizer_config.json | NOT TESTED |
| MiniMax JANG_2L | N/A (not VLM) | chat_template.jinja (COPIED) | N/A |

### Fix:
Copy from source for each model family:
```bash
# Qwen3.5 (all sizes share same preprocessor)
cp source/preprocessor_config.json model/
cp source/video_preprocessor_config.json model/
```

## CRITICAL: HF Repos Need Full Re-Upload (v2 + VL)

ALL repos on JANGQ-AI are still v1 format:
| Repo | Current | Needs |
|------|---------|-------|
| Qwen3.5-35B-A3B-JANG_4K | v1 (5 .jang shards) | v2 + VL files + updated README |
| Qwen3.5-35B-A3B-JANG_2S | v1 | v2 + VL files |
| Qwen3.5-122B-A10B-JANG_4K | v1 (15 .jang shards) | v2 + VL files |
| Qwen3.5-122B-A10B-JANG_2S | v1 | v2 + VL files |
| Qwen3.5-122B-A10B-JANG_1L | v1 | v2 + VL files |
| MiniMax-M2.5-JANG_2L | v1 (16 .jang shards) | v2 + chat_template.jinja |

### New repos to create:
- Qwen3.5-4B-JANG_4S
- Qwen3.5-9B-JANG_4S
- Qwen3.5-35B-A3B-JANG_4S (after conversion)

## MiniMax Rules (MUST document)

### group_size:
- Expert MLP: gs=128 (150+ experts, gather_qmm cache pressure)
- Router/gate: gs=64 at Q8 ALWAYS (precision-critical, tiny tensor)
- Per-tensor group_size now implemented in converter

### Tokenizer:
- `chat_template.jinja` is a SEPARATE FILE (not in tokenizer_config.json)
- `tokenizer.json` must have NFC normalizer (mlx_lm.convert corrupts it)
- Copy from `mlx-community/MiniMax-M2.5-4bit` or source after any conversion

### Inference:
- temp=1.0 REQUIRED (greedy causes loops)
- top_p=0.95, top_k=40, do_sample=true
- Thinking is ALWAYS ON (template injects `<think>`)

## Benchmarks Still Needed

| Benchmark | Status |
|-----------|--------|
| 35B MLX 5-bit MMLU | NEED (download on Mac Studio) |
| 35B JANG_4S MMLU | NEED (convert from 35B source) |
| 122B JANG_4S MMLU | NEED (convert from 122B source) |
| 397B any MMLU | NEED (model ready on Mac Studio) |
| MiniMax MMLU (re-verify) | DONE (74% from previous) |
| ALL tok/s | Add to all future benchmarks |

## Code Fixes Applied Today

1. VLM loader: fixed class_predicate (all layers quantized)
2. Stripped .importance from all models (~8 GB saved)
3. Fixed converter to not save importance inline
4. Fixed calibrator to save imatrix to OUTPUT dir
5. Per-tensor group_size (router gs=64, experts gs=128)
6. _fix_quantized_bits now infers both bits AND group_size
7. Mamba A_log/D tier classification case-sensitivity fix
8. upgrade_v1_to_v2 metadata={"format":"mlx"} fix
9. Broader exception handling for VLM processor
10. SwitchLinear bias detection fix
