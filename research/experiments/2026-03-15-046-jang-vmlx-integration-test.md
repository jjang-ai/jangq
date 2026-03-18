# Experiment 046: JANG + vMLX Integration — End-to-End Inference

**Date**: 2026-03-15
**Author**: Jinho Jang (eric@jangq.ai)
**Status**: WORKING

## Summary

First successful end-to-end inference through JANG format → vMLX engine → OpenAI API.

## Pipeline Tested

```
JANG .jang.safetensors model
  → vmlx_engine.utils.jang_loader.load_jang_model()
  → Vectorized dequantization to float16 MLX arrays
  → mlx_lm model architecture (auto-detected from config.json)
  → vMLX SimpleEngine
  → OpenAI-compatible /v1/chat/completions endpoint
  → Tool parser auto-configured (qwen)
```

## Test Model: Qwen2.5-0.5B-JANG_2S

- Profile: JANG_2S (CRITICAL=6b, IMPORTANT=4b, COMPRESS=2b)
- Actual bits: 2.91 avg
- Weight size: 0.17 GB qweight
- GPU memory: 0.92 GB active
- Load time: 8.2s (dequant) + 2s (server startup) = ~10s total

## Results

### OpenAI API Response (working)

```json
POST /v1/chat/completions
{
    "model": "jang",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 50,
    "temperature": 0.0
}

Response: 200 OK
  - 46 tokens generated
  - 144.5 tok/s
  - finish_reason: "stop"
  - tool_calls: null (correct)
  - reasoning_content: null (correct)
```

### Text Quality

Output is incoherent — **expected** for 0.5B at 2.9 bits. Small models (< 1B) break at
low bits regardless of allocation strategy. This matches prior findings (experiment 037):
JANG shows wins starting at 3B+ models.

### What's Working

1. JANG model detection (`jang_config.json` → is_jang_model() → True)
2. Vectorized dequantization (no per-block Python loop)
3. v1.1 format support (`.bits` per tensor, no `bit_map`/`block_offsets`)
4. Shape restoration for 3D+ tensors via `.shape` companion
5. Model architecture auto-detection from `config.json`
6. OpenAI `/v1/chat/completions` endpoint
7. `/v1/models` endpoint
8. Tool parser auto-configuration
9. Reasoning parser support (available via `--reasoning-parser`)
10. Memory reporting (GPU and system)

### Performance

- Load: 8.2s for 0.5B model (dequant-at-load)
- Generation: 144.5 tok/s (0.5B is tiny, expected fast)
- Memory: 0.92 GB GPU for float16 dequantized model

## Architecture Support

JANG → vMLX now supports all architectures that mlx_lm supports:
- Dense transformers (Llama, Qwen, Gemma, Phi, Mistral)
- MoE (Mixtral, Qwen3.5 MoE, DeepSeek, MiniMax)
- Hybrid SSM (Jamba)
- Vision-Language (Qwen-VL)

## v1.1 Format Changes

Eliminated per-block metadata overhead:
- Removed: `bit_map` (per-block uint8) — replaced by `.bits` (single uint8 per tensor)
- Removed: `block_offsets` (per-block uint32) — computable from bits × block_index
- Result: ~2.6 GB saved on 35B model (17 GB → 14 GB)
- Overhead: scales + zeros only (4 bytes per block vs 9 bytes before)

## Files Modified

- `/Users/eric/mlx/vllm-mlx/vmlx_engine/utils/jang_loader.py` — vectorized dequant, v1.1 support
- `/Users/eric/mlx/vllm-mlx/vmlx_engine/commands/convert.py` — JANG profile conversion via `--jang-profile`
- `/Users/eric/mlx/vllm-mlx/vmlx_engine/cli.py` — `--jang-profile` and `--jang-method` flags
- `/Users/eric/jang/jang-tools/jang_tools/quantize.py` — uniform-per-tensor QuantizedTensor
- `/Users/eric/jang/jang-tools/jang_tools/format/writer.py` — v1.1 `.bits` storage
- `/Users/eric/jang/jang-tools/jang_tools/format/reader.py` — v1.1 reading with v1.0 fallback
- `/Users/eric/jang/jang-tools/jang_tools/format/spec.py` — reduced overhead calculation

## Next Steps

1. Test with 3B+ models for quality validation (JANG_3M, JANG_4M)
2. Test 35B JANG_2S through vMLX (need to verify dequant time at scale)
3. Benchmark against MLX uniform quantization (same model, same bits, compare coherence)
4. Test full vMLX features: streaming, tool calling, reasoning parsers, continuous batching
5. Optimize: consider keeping weights quantized in GPU memory with on-the-fly Metal dequant
