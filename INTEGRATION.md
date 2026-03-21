# JANG Integration Guide for MLX Apps

> How to add JANG model support to your MLX inference application.
>
> Created by Jinho Jang (eric@jangq.ai) — [jangq.ai](https://jangq.ai)

## Overview

JANG models are MLX-native safetensors with mixed-precision quantization. They load instantly via `mx.load()` mmap — no custom format parsing needed. The key difference from standard MLX quantized models is that **different tensors have different bit widths** (e.g., attention at 8-bit, MLP at 2-bit).

**If your app already loads MLX quantized models, JANG support requires minimal changes.**

## Quick Integration (Simplest Path)

```python
# 1. Detect JANG model
def is_jang_model(model_path):
    """Check if directory contains a JANG model."""
    from pathlib import Path
    for name in ["jang_config.json", "jjqf_config.json", "jang_cfg.json"]:
        if (Path(model_path) / name).exists():
            return True
    return False

# 2. Load with jang-tools (pip install jang[mlx])
from jang_tools.loader import load_jang_model, load_jang_vlm_model

model, tokenizer = load_jang_model("/path/to/jang/model")
# model is a standard mlx_lm model — use it exactly like any MLX model
# tokenizer is a standard HuggingFace tokenizer
```

That's it. The returned `model` and `tokenizer` are fully compatible with `mlx_lm.generate()`, `generate_step()`, and any standard MLX inference pipeline.

## File Structure

A JANG v2 model directory contains:

```
model-00001-of-00043.safetensors   # MLX-native quantized weights (uint32 packed)
model-00002-of-00043.safetensors   # Standard safetensors format — mx.load() works directly
...
model.safetensors.index.json       # Standard HF shard index

config.json                        # HuggingFace model config + quantization key
jang_config.json                   # JANG-specific metadata (profile, bits, source model)
tokenizer_config.json              # Tokenizer with inline chat template
tokenizer.json                     # Tokenizer vocabulary
chat_template.jinja                # Chat template (also inline in tokenizer_config)

# Optional:
preprocessor_config.json           # VLM image preprocessor
video_preprocessor_config.json     # VLM video preprocessor
modeling_*.py                      # Custom model code (trust_remote_code models)
configuration_*.py                 # Custom config (trust_remote_code models)
```

## Weight Format

JANG v2 weights are stored in **standard MLX quantized format** — the same `uint32` packed format that `mx.quantize()` produces:

```
{name}.weight    # uint32, shape (out_features, packed_in_features)
{name}.scales    # float16, shape (out_features, n_groups)
{name}.biases    # float16, shape (out_features, n_groups)
```

**No custom dequantization needed.** MLX's `QuantizedLinear` and `QuantizedEmbedding` handle these natively. The only difference from uniform MLX quantization is that different tensors may have different bit widths and group sizes.

## Key Concept: Mixed Bit Widths

In a standard MLX 4-bit model, every tensor uses 4-bit quantization. In JANG, different tensors use different bit widths:

| Tensor Type | Typical Bits | Why |
|-------------|:----:|-----|
| Attention Q/K/V/O | 6-8 | Controls coherence |
| MoE routers/gates | 8 | Controls expert routing |
| Embeddings | 4-6 | First layer, errors propagate |
| Expert MLP (gate/up/down) | 2-4 | 95%+ of params, redundancy absorbs errors |
| Norms | float16 | Tiny, kept full precision |
| Vision encoder | float16 | Kept full precision |

### Inferring Bit Width Per Tensor

The bit width and group size for each tensor can be inferred from the weight and scales shapes:

```python
def infer_bits_and_group_size(weight_shape, scales_shape):
    """
    Infer quantization bits and group_size from tensor shapes.

    weight: (out_features, packed_per_row) — uint32 packed
    scales: (out_features, n_groups) — float16

    Equation: packed_per_row * 32 / bits == n_groups * group_size
    Also: in_features = n_groups * group_size
    """
    w_cols = weight_shape[-1]  # packed columns
    s_cols = scales_shape[-1]  # number of groups

    for bits in [2, 3, 4, 5, 6, 8]:
        elem_per_u32 = 32 // bits
        in_features = w_cols * elem_per_u32
        group_size = in_features // s_cols if s_cols > 0 else 0
        if group_size > 0 and group_size * s_cols == in_features:
            return bits, group_size

    return None, None
```

### Setting Up QuantizedLinear Per Tensor

When loading JANG weights, each `QuantizedLinear` layer needs its `bits` and `group_size` set to match the actual quantization of that specific tensor:

```python
import mlx.nn as nn

# After loading weights into the model skeleton:
for name, module in model.named_modules():
    if isinstance(module, nn.QuantizedLinear) and hasattr(module, 'scales'):
        bits, gs = infer_bits_and_group_size(
            module.weight.shape, module.scales.shape
        )
        if bits is not None:
            module.bits = bits
            module.group_size = gs
```

The JANG loader does this automatically via `_fix_quantized_bits()`.

## Loading Step by Step

If you want to implement JANG loading without depending on the `jang-tools` package:

### Step 1: Read Configs

```python
import json
from pathlib import Path

model_path = Path("/path/to/jang/model")

# Read JANG config
jang_cfg = json.loads((model_path / "jang_config.json").read_text())
profile = jang_cfg["quantization"]["profile"]       # e.g., "JANG_2L"
actual_bits = jang_cfg["quantization"]["actual_bits"] # e.g., 2.31
block_size = jang_cfg["quantization"]["block_size"]   # e.g., 64

# Read model config
config = json.loads((model_path / "config.json").read_text())
# config has a "quantization" key: {"group_size": 64, "bits": 2}
# The "bits" is the MINIMUM bit width (COMPRESS tier) — used as default
```

### Step 2: Create Model Skeleton

```python
from mlx_lm.utils import load_model, load_config

config = load_config(model_path)
# Ensure quantization config exists
if "quantization" not in config:
    config["quantization"] = {"group_size": block_size, "bits": min_bits}

model, config = load_model(model_path, lazy=True, strict=False, model_config=config)
```

### Step 3: Upgrade SwitchLinear for MoE

For MoE models, the model skeleton creates `SwitchLinear` layers. These need to be upgraded to `QuantizedSwitchLinear` to handle quantized expert weights:

```python
from mlx_lm.models.switch_layers import SwitchLinear, QuantizedSwitchLinear

for name, module in model.named_modules():
    if isinstance(module, SwitchLinear):
        ql = QuantizedSwitchLinear(
            module.input_dims, module.output_dims, module.num_experts,
            bias=module.bias is not None,
            group_size=config["quantization"]["group_size"],
            bits=config["quantization"]["bits"],
        )
        # Replace in model tree
        parts = name.rsplit('.', 1)
        parent = model
        for p in parts[0].split('.'):
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        setattr(parent, parts[1], ql)
```

### Step 4: Load Weights

```python
import mlx.core as mx
import gc

# Get weight files
index = json.loads((model_path / "model.safetensors.index.json").read_text())
shards = sorted(set(index["weight_map"].values()))

for shard in shards:
    weights = mx.load(str(model_path / shard))

    # Filter out calibration data and MTP weights
    weights = {k: v for k, v in weights.items()
               if not k.endswith(".importance") and not k.startswith("mtp.")}

    # Apply model's sanitize (renames HF keys to MLX keys)
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    model.load_weights(list(weights.items()), strict=False)
    del weights
    gc.collect()
```

### Step 5: Fix Per-Tensor Bit Widths

```python
# After all weights loaded, infer actual bits/gs per tensor
for name, module in model.named_modules():
    if not isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
        continue
    if not hasattr(module, 'scales') or not hasattr(module, 'weight'):
        continue

    bits, gs = infer_bits_and_group_size(
        module.weight.shape, module.scales.shape
    )
    if bits is not None:
        module.bits = bits
        module.group_size = gs
```

### Step 6: bfloat16 for Large Expert Models

Models with 512+ experts and hidden_size >= 4096 overflow float16 at the shared expert down_proj. Auto-detect and switch to bfloat16:

```python
text_cfg = config.get("text_config", config)
n_experts = text_cfg.get("num_experts",
    text_cfg.get("num_local_experts",
    text_cfg.get("n_routed_experts", 0)))
hidden = text_cfg.get("hidden_size", 0)

if n_experts >= 512 and hidden >= 4096:
    model.set_dtype(mx.bfloat16)
```

**Affected models:** Qwen3.5-397B (512 experts, hidden=4096), Nemotron-3-Super-120B (512 experts, hidden=4096).

**Not affected:** All models with 256 or fewer experts or hidden_size < 4096.

### Step 7: Materialize and Return

```python
mx.eval(model.parameters())

from mlx_lm.utils import load_tokenizer
tokenizer = load_tokenizer(model_path)

# model and tokenizer are now ready for inference
```

## Special Handling: Nemotron-H Models

Nemotron-H architecture (`model_type: "nemotron_h"`) requires additional handling:

### 1. Weight Name Mapping

JANG v2 stores expert weights as `switch_mlp.up_proj` / `switch_mlp.down_proj`, but mlx-lm's Nemotron model expects `switch_mlp.fc1` / `switch_mlp.fc2`:

```python
if model_type == "nemotron_h":
    renames = {
        "switch_mlp.up_proj": "switch_mlp.fc1",
        "switch_mlp.down_proj": "switch_mlp.fc2",
    }
    renamed = {}
    for k, v in weights.items():
        new_k = k
        for old, new in renames.items():
            if old in k:
                new_k = k.replace(old, new)
                break
        renamed[new_k] = v
    weights = renamed
```

### 2. Gate Weight Dequantization

The MoE gate in Nemotron-H is `nn.Linear` (not `QuantizedLinear`), but JANG stores gate weights as quantized uint32. The gate must be dequantized during loading:

```python
# Collect gate scales/biases during weight iteration
if ".gate." in key and key.endswith(".scales"):
    gate_parts[prefix]["scales"] = value
    continue  # Don't load directly

# After collecting, dequantize:
# Try bits in order [8, 6, 4, 3, 2] (gate is typically 8-bit CRITICAL tier)
for bits in [8, 6, 4, 3, 2]:
    elem_per_u32 = 32 // bits
    real_cols = gate_weight.shape[-1] * elem_per_u32
    gs = real_cols // scales.shape[-1]
    if gs > 0 and gs * scales.shape[-1] == real_cols:
        dequantized = mx.dequantize(gate_weight, scales, biases, gs, bits)
        # Store as bfloat16 for 512-expert models, float16 otherwise
        weights[gate_key] = dequantized.astype(mx.bfloat16)
        break
```

### 3. Drop MTP Keys

Nemotron models include multi-token prediction weights (`mtp.*`) that are not used at inference:

```python
weights = {k: v for k, v in weights.items() if not k.startswith("mtp.")}
```

## Chat Template Handling

All JANG models include chat templates with `enable_thinking` toggle for reasoning models:

```python
# Check if model supports thinking mode
tokenizer_config = json.loads((model_path / "tokenizer_config.json").read_text())
has_thinking = "enable_thinking" in str(tokenizer_config.get("chat_template", ""))

# Apply chat template
messages = [{"role": "user", "content": "Hello"}]

if has_thinking:
    # With reasoning (model thinks step by step)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True
    )

    # Without reasoning (direct answer)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False
    )
else:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
```

**Important:** When `enable_thinking` is not passed, models default to thinking ON. If your app wants thinking OFF by default, always pass `enable_thinking=False`.

When thinking is ON, the model output contains `<think>...</think>` tags:
```
<think>
The user asks about the capital of France. The answer is Paris.
</think>

Paris
```

Your app should:
1. Parse and optionally display the `<think>...</think>` reasoning
2. Show the content after `</think>` as the main response
3. Provide a UI toggle for users to enable/disable thinking

## VLM (Vision-Language) Support

JANG VLM models include vision encoder weights preserved in float16. Load via:

```python
from jang_tools.loader import load_jang_vlm_model

model, processor = load_jang_vlm_model("/path/to/vlm/model")

# Use processor to format image prompts
messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "Describe this image."},
]}]
prompt = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# Generate with mlx-vlm
from mlx_vlm import generate as vlm_generate
result = vlm_generate(model, processor, prompt=prompt,
                       image=["photo.jpg"], max_tokens=200)
```

VLM models are identified by:
- `preprocessor_config.json` exists in the model directory
- `jang_config.json` has `architecture.has_vision: true`
- `config.json` has `vision_config` key present

## Config Reference

### jang_config.json

```json
{
    "format": "jang",
    "format_version": "2.0",
    "quantization": {
        "method": "jang-importance",
        "profile": "JANG_2L",
        "target_bits": 2.0,
        "actual_bits": 2.31,
        "block_size": 64,
        "bit_widths_used": [2, 6, 8],
        "quantization_backend": "mx.quantize"
    },
    "source_model": {
        "name": "Qwen/Qwen3.5-122B-A10B",
        "dtype": "bfloat16",
        "parameters": "122B"
    },
    "architecture": {
        "type": "hybrid_moe_ssm",
        "attention": "none",
        "has_vision": true,
        "has_ssm": true,
        "has_moe": true
    }
}
```

### config.json quantization key

```json
{
    "quantization": {
        "group_size": 64,
        "bits": 2
    }
}
```

The `bits` value is the **minimum** (COMPRESS tier). Use `jang_config.json` `actual_bits` for the true average.

## OpenAI-Compatible API

JANG models work with any OpenAI-compatible API server. The model loads like any MLX model, so existing API implementations work without modification once the model is loaded.

## Edge Cases and Gotchas

### 1. strict=False Required
Always use `strict=False` when calling `model.load_weights()`. JANG models may have tensors that don't exactly match the skeleton.

### 2. Non-Quantized Tensors
Some tensors are stored as float16 (not quantized):
- All norm weights (LayerNorm, RMSNorm)
- Bias terms
- Vision encoder Conv2D weights (patch_embed)
- Small 1D tensors

### 3. Group Size Varies Per Tensor
Most tensors use `group_size=64`, but:
- MoE router/gate tensors always use `group_size=64`
- Expert MLP tensors on 150+ expert models use `group_size=128` (speed optimization)

### 4. Consolidated vs Standard Safetensors
Check for both `model.safetensors.index.json` and `consolidated.safetensors.index.json`.

### 5. trust_remote_code Models
Some models (Nemotron-H, MiniMax) include custom `.py` files needed for architecture loading.

### 6. KV Cache Compatibility
JANG models use the same KV cache format as standard MLX models. No changes needed.

## Supported Models

| Architecture | Models | Special Handling |
|---|---|---|
| Qwen3.5 MoE+SSM | 35B, 122B, 397B | VLM, bfloat16 for 397B |
| Nemotron-H | Super-120B, Cascade-2-30B | fc1/fc2 rename, gate dequant |
| MiniMax-M2.5 | 172B MoE | group_size=128 |
| Qwen3.5 Dense | 4B, 9B, 27B | VLM |
| Mistral Small 4 | 119B MoE+MLA+VL | Coming soon |

## Testing Your Integration

```python
# Minimal test: load and generate one token
from pathlib import Path
import mlx.core as mx

model_path = Path("path/to/jang/model")

# Your loading code here...
model, tokenizer = your_load_function(model_path)

# Verify
tokens = mx.array([[tokenizer.encode("Hello")[0]]])
logits = model(tokens)

assert not mx.any(mx.isnan(logits)).item(), "NaN in output!"
assert logits.shape[-1] == tokenizer.vocab_size, "Wrong vocab size!"

print(f"Model loaded: {mx.get_active_memory()/1024**3:.1f} GB")
print("Integration test passed!")
```

## Questions?

- GitHub: [github.com/jjang-ai/jangq](https://github.com/jjang-ai/jangq)
- HuggingFace: [huggingface.co/JANGQ-AI](https://huggingface.co/JANGQ-AI)
- Email: eric@jangq.ai
- MLX Studio: [mlx.studio](https://mlx.studio)
