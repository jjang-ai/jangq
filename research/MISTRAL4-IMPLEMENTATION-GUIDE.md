# Mistral Small 4 (119B) — JANG Implementation Guide

## For MLX Studio / Third-Party Integration

### Model Variants

| Profile | Size | bpw | RAM Usage | Gen Speed | Best For |
|---------|------|-----|-----------|-----------|----------|
| JANG_4M | 57 GB | 4.08 | ~69 GB | 25 tok/s (f16), 75 tok/s (bf16) | 128 GB Macs |
| JANG_2L | 30 GB | 2.14 | ~40 GB | 25 tok/s | 64 GB Macs |

### Required Patches (6 fixes for DeepSeek V2 backend)

The model uses `model_type: "mistral3"` (top-level VLM wrapper) with `text_config.model_type: "mistral4"` (MLA + MoE text model). Since mlx-lm has no native mistral4 support, we route through DeepSeek V2 with these patches:

1. **mistral3.py**: Route `mistral4` text_config to `deepseek_v2.Model` with config translation (rope_parameters → rope_scaling, llama_4_scaling_beta)

2. **deepseek_v2.py patches**:
   - `rope_interleave: bool = False` in ModelArgs → uses `traditional=not rope_interleave` in YaRN RoPE
   - `norm_topk_prob: bool = False` in ModelArgs → normalizes top-k scores to sum=1 in MoEGate
   - `_rope_attn_scaling`: computed as `mscale_num / mscale_den` (= 1.0 for Mistral 4), applied to q_pe/k_pe ONLY
   - Attention scale: plain `q_head_dim ** -0.5` = 0.0884, NO mscale*mscale
   - Llama 4 scaling: `1 + beta * log(1 + floor(pos / max_pos))` on queries

3. **JANG loader**: Gate weight dequant from uint32 → bfloat16 (MUST stay bfloat16, not float16)

### Chat Template

```
<s>[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS][INST]{user_message}[/INST]
```

For reasoning mode (not working at 4-bit):
```
<s>[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS][INST]{user_message}[/INST]
```

### EOS Handling
- `eos_token_id: 2` (`</s>`)
- `bos_token_id: 1` (`<s>`)

### Dtype Behavior
- Gate weight: bfloat16 (from dequant)
- Gate matmul (f16 @ bf16): promotes to float32
- This makes the entire model compute in float32 after layer 0's MoE
- This is CORRECT and REQUIRED — converting gate to float16 breaks routing

### Known Limitations at 4-bit

1. **Short answers (< 30 tokens)**: Work well — 70%+ accuracy on factual Q&A
2. **Long generation (> 30 tokens)**: Degrades into repetition
   - Root cause: MoE routing sensitivity amplifies tiny computation differences
   - Periodic cache refresh (every 15 tokens) helps for knowledge tasks
   - Code generation and creative writing remain degraded
3. **Reasoning mode**: [THINK] tags not generated (quantization lost this alignment)
4. **MMLU format**: Model can't follow "just answer with letter A/B/C/D" instruction
5. **Math**: Partial — knows answers for worded questions, fails with symbolic notation

### Periodic Cache Refresh (Recommended)

For responses > 30 tokens, recompute full context from scratch every 15 tokens:

```python
if (token_count + 1) % 15 == 0:
    cache = make_prompt_cache(model)
    logits = model(mx.array([all_tokens]), cache=cache)
```

Speed impact: ~18 tok/s (vs 25 tok/s without refresh). Significantly improves coherency.

### VLM (Vision)

- Vision tower: 218 tensors, Pixtral architecture, preserved as float16
- Processor: PixtralImageProcessorFast, patch_size=14, max 1540px
- Needs mlx-vlm integration for image processing
- NOT YET TESTED with JANG quantized weights

### Speed Reference

| Mode | Prefill | Generation | Memory |
|------|---------|------------|--------|
| float16 | 130 tok/s | 25 tok/s | 69 GB |
| bfloat16 | 215 tok/s | 75 tok/s | 68 GB |
| With cache refresh (f16) | 130 tok/s | 18 tok/s | 69 GB |

### Architecture Reference

```
Model: mistral3 (VLM wrapper)
├── vision_tower: Pixtral (24 layers, 218 tensors, float16)
├── multi_modal_projector: linear_1, linear_2, patch_merger
└── language_model: deepseek_v2
    ├── embed_tokens: (131072, 4096) — 4-bit
    ├── layers × 36:
    │   ├── self_attn: MLA
    │   │   ├── q_a_proj: (1024, 4096) — 8-bit
    │   │   ├── q_a_layernorm: (1024,) — float
    │   │   ├── q_b_proj: (4096, 1024) — 8-bit
    │   │   ├── kv_a_proj_with_mqa: (320, 4096) — 8-bit
    │   │   ├── kv_a_layernorm: (256,) — float
    │   │   ├── kv_b_proj: (6144, 256) — 8-bit
    │   │   └── o_proj: (4096, 4096) — 8-bit
    │   └── mlp: MoE
    │       ├── gate: (128, 4096) — bfloat16 (dequanted from 8-bit)
    │       ├── switch_mlp: SwitchGLU
    │       │   ├── gate_proj: (128, 2048, 4096) — 4-bit
    │       │   ├── up_proj: (128, 2048, 4096) — 4-bit
    │       │   └── down_proj: (128, 4096, 2048) — 4-bit
    │       └── shared_experts: MLP
    │           ├── gate_proj: (2048, 4096) — 4-bit
    │           ├── up_proj: (2048, 4096) — 4-bit
    │           └── down_proj: (4096, 2048) — 4-bit
    ├── norm: RMSNorm — float
    └── lm_head: (131072, 4096) — 8-bit
```
