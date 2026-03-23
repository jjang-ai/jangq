# Mistral Small 4 Conversion Log

## Status: JANG_4M CONVERTING (attempt 5) — JANG_2L works but 2-bit too aggressive

## Pre-Flight Audit Results (ALL CLEAR)

- **401 tensors to quantize** (MLA attention, MoE experts, shared experts, projector, embeddings, lm_head)
- **1084 tensors skipped** (360 activation_scale, 360 scale_inv, 218 vision, 144 norms, 2 misc)
- **0 problems found**
- All last dims >= 64 (divisible by group_size)
- Vision patch_conv correctly skipped
- FP8 dequant: `fp8_decode(uint8) * weight_scale_inv` (multiply, not divide)

## Tensor Structure Per Layer

| Tensor | Shape | FP8? | JANG Tier |
|--------|-------|------|-----------|
| self_attn.q_a_proj.weight | (1024, 4096) | uint8 | CRITICAL |
| self_attn.q_b_proj.weight | (4096, 1024) | uint8 | CRITICAL |
| self_attn.kv_a_proj_with_mqa.weight | (320, 4096) | uint8 | CRITICAL |
| self_attn.kv_b_proj.weight | (6144, 256) | uint8 | CRITICAL |
| self_attn.o_proj.weight | (4096, 4096) | uint8 | CRITICAL |
| self_attn.q_a_layernorm.weight | (1024,) | bf16 | SKIP |
| self_attn.kv_a_layernorm.weight | (256,) | bf16 | SKIP |
| mlp.experts.gate_up_proj | (128, 4096, 4096) | uint8 | COMPRESS |
| mlp.experts.down_proj | (128, 4096, 2048) | uint8 | COMPRESS |
| mlp.gate.weight | (128, 4096) | bf16 | CRITICAL |
| mlp.shared_experts.gate_proj.weight | (2048, 4096) | uint8 | CRITICAL |
| mlp.shared_experts.up_proj.weight | (2048, 4096) | uint8 | CRITICAL |
| mlp.shared_experts.down_proj.weight | (4096, 2048) | uint8 | CRITICAL |
| input_layernorm.weight | (4096,) | bf16 | SKIP |
| post_attention_layernorm.weight | (4096,) | bf16 | SKIP |

Vision (218 tensors): ALL skipped → passthrough as float16
Projector (3 tensors): linear_1, linear_2, patch_merger → quantized

## Previous Failures

### Attempt 1 (MacBook): Crashed at tokenizer copy
- special_tokens_map.json was corrupt (HTML error page from HTTP download)
- Quantization was 83% complete before crash
- Fix: created valid special_tokens_map.json

### Attempt 2 (Mac Studio external): Crashed at 93% quantization
- vision_tower.patch_conv.weight (1024, 3, 14, 14) tried to quantize
- Root cause: consolidated.safetensors used `vision_encoder` naming, not `vision_tower`
- Converter only checked `.visual.`, `vision_tower`, `vision_model` — missed `vision_encoder`
- Fix 1: Added `vision_encoder` to skip pattern
- Fix 2: Deleted consolidated shards, only model-* remain
- Fix 3: Added `patch_conv` to passthrough pattern

### Attempt 3 (Mac Studio internal): Completed but NaN at inference
- Conversion completed successfully (~37 GB JANG_2L output)
- Inference produced NaN at layer 0 attention
- **ROOT CAUSE: bfloat16 scale_inv silently failed to load**
  - `weight_scale_inv` tensors stored as BF16 in safetensors
  - `safe_open(framework='numpy')` can't handle bfloat16 → throws exception
  - Converter catches exception but `scale_inv=None` passed to `load_fp8_tensor`
  - FP8 decode returns raw values (max=448) WITHOUT scale multiplication
  - Correct: q_b_proj max=0.3008 (448 × 0.000671)
  - Broken: q_b_proj max=448.0 (raw FP8 decode, no scale)
  - Result: attention Q values ~27632 → dot product overflows float16 → inf → NaN
- Fix 1: `_load_bf16_from_header()` reads raw bfloat16 bytes from safetensors
  - Converts bf16 → f32 via uint16 left-shift (bf16 = upper 16 bits of f32)
  - Handles both scalar (attention) and 3D (128,1,1) expert scale_inv
- Fix 2: `rope_interleave=True` support in DeepSeek V2 attention
  - Mistral 4 uses interleaved RoPE, DeepSeek V2 hardcoded `traditional=True`
  - Added `rope_interleave` to ModelArgs, passed to YaRN rope as `traditional=not rope_interleave`

### Attempt 4 (Mac Studio internal): JANG_2L works at basic factual, fails complex
- FP8 dequant fix + rope_interleave fix applied
- Conversion completed: 29.69 GB, 2.14 bpw
- Simple factual: "The capital of France is Paris." ✓
- Math/reasoning/coding: Degenerates into repetition loops ✗
- **ROOT CAUSE: Architecture mapping was incomplete (3 missing pieces)**
  1. `norm_topk_prob=True`: Top-k expert scores must normalize to sum=1.
     DeepSeek V2 mlx-lm code didn't implement this. Without it, scores sum ~0.77
     and expert weighting is wrong. **This was the key fix that made factual work.**
  2. `llama_4_scaling_beta=0.1`: Mistral 4 inherits DeepSeek V3 position-dependent
     query scaling: `q *= 1 + 0.1*log(1 + floor(pos/8192))`. Only matters for
     long sequences (at pos=0, scale=1.0).
  3. `rope_interleave=True`: Already fixed in attempt 4. Mistral 4 uses interleaved
     RoPE, DeepSeek V2 hardcoded traditional=True.
- **Additional finding**: Mistral 4 inherits from DeepSeek V3, not V2.
  HF transformers has `Mistral4Attention(DeepseekV3Attention)`.
- **2-bit quality**: Too aggressive for 128 experts. Expert 53 output (max=226) is
  5x larger than shared expert (max=8.85). At 2-bit, quantization noise compounds
  through the SiLU*up → down_proj computation and overwhelms the signal.
- Starting JANG_4M (4-bit experts) for proper quality.

## Architecture Summary

- model_type: mistral3 (top) / mistral4 (text_config)
- 119B total, 6B active (8B with embeddings)
- 128 experts, top-4 active
- MLA attention: kv_lora_rank=256, q_lora_rank=1024
- Pixtral vision: 24 layers, 1540px, 218 tensors
- FP8 E4M3 source with per-tensor scalar scales
- Chat template: reasoning_effort="none"/"high", [THINK]...[/THINK] tags
- Apache 2.0 license

## Expected Output

- JANG_2L: ~32 GB, 2.09 bpw
- 98.3% at 2-bit, 0.2% at 6-bit, 1.4% at 8-bit
