# Mistral Small 4 Conversion Log

## Status: IN PROGRESS

### Architecture Detection (FIXED)
- model_type: mistral3 (top-level) / mistral4 (text_config)
- Detected: MoE + MLA + Vision
- 128 experts, top-4 active
- kv_lora_rank: 256, q_lora_rank: 1024
- Pixtral vision encoder (24 layers)
- FP8 E4M3 source (uint8 + per-tensor scalar scale)

### Fixes Applied
1. `architectures.py`: Fixed VLM early-return to check `n_routed_experts` (was only checking `num_local_experts`)
2. `architectures.py`: Fixed MLA detection to check `text_config.kv_lora_rank` and `text_config.model_type == "mistral4"`
3. `architectures.py`: Fixed `attention_type` override to not clobber MLA from `_classify_architecture`
4. `architectures.py`: Added vision layer configs to MoE return (for VLM+MoE models)
5. `fp8.py`: Added per-tensor scalar scale support (Mistral format: `weight * scale_inv`)
6. `fp8.py`: Added per-expert 3D scale support `(128, 1, 1)`

### FP8 Dequantization
- Format: uint8 E4M3FN + bfloat16 scalar `weight_scale_inv`
- Dequant: `fp8_decode(uint8) * weight_scale_inv` (multiply, NOT divide — name is misleading)
- Result: values in range [-0.14, 0.15] — proper neural net weights
- `activation_scale` tensors: dropped (not needed for weight-only quantization)

### Conversion Test
- Running on MacBook (113 GB source)
- Profile: JANG_2L, block_size=64
- Expected output: ~30-35 GB

### Still TODO
- Verify conversion completes without errors
- Handle fused `gate_up_proj` splitting for pre-stacked experts
- Handle `activation_scale` and `_scale_inv` key filtering
- Test inference with mlx-lm (needs mistral4 model support)
- MMLU benchmark
- Speed test
- VLM test
- HuggingFace upload
