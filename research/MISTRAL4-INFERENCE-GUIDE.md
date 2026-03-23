# Mistral Small 4 (119B) — Inference Guide

## For MLX Studio Integration
Created by Jinho Jang (eric@jangq.ai) | Date: 2026-03-22

---

## Text-Only Inference (JANG loader)

```python
from jang_tools.loader import load_jang_model
from mlx_lm import generate

model, tokenizer = load_jang_model("/path/to/Mistral-Small-4-119B-JANG_2L")
messages = [{"role": "user", "content": "What is 15 * 23?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
output = generate(model, tokenizer, prompt=prompt, max_tokens=200, verbose=True)
```

The JANG loader automatically:
- Detects MLA model (kv_lora_rank > 0) and sets bfloat16 compute
- Handles gate weight (float16 passthrough, no quantization)
- Fixes per-tensor quantization bits via _fix_quantized_bits

## VLM Inference (mlx-vlm 0.4.1+)

See /Users/eric/jang/research/examples/mistral4_vlm_inference.py

## Reasoning Mode

```python
messages = [{"role": "user", "content": "Solve: 17 * 24"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
prompt = prompt.replace('reasoning_effort": "none', 'reasoning_effort": "high')
output = generate(model, tokenizer, prompt=prompt, max_tokens=500)
# Output contains [THINK]...[/THINK] tags
```

## Chat Template Format

Default: `<s>[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS][INST]{msg}[/INST]`
Reasoning: `<s>[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS][INST]{msg}[/INST]`

Special tokens: BOS=1, EOS=2, [INST]=3, [/INST]=4, [THINK]=34, [/THINK]=35

## Speed Reference

| Model | Gen tok/s | Prefill tok/s | RAM | Disk |
|-------|-----------|---------------|-----|------|
| JANG_2L | 77-84 | 195-261 | 40 GB | 30 GB |
| JANG_4M | 76-82 | 206-241 | 68 GB | 57 GB |

All bfloat16 compute on M3 Ultra 256 GB.

## Required Patches on Inference Machine

7 patches in deepseek_v2.py + mistral3.py. See MISTRAL4-COMPLETE-JOURNEY.md for details.
Key: rope_interleave=True -> traditional=True, plain attention scale 0.0884, norm_topk_prob, auto-bfloat16.

## Files Required in Model Directory

config.json, jang_config.json, generation_config.json, tokenizer.json,
tokenizer_config.json, special_tokens_map.json, chat_template.jinja,
processor_config.json, preprocessor_config.json, model.safetensors.index.json,
model-*.safetensors, images/ (for VLM testing)
