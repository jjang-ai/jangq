# JANG Format Specification v2.0

> Mixed-Precision Quantization for Apple Silicon
> Created by Jinho Jang (eric@jangq.ai)

## Overview

JANG is an open file format for storing mixed-precision quantized LLM weights
optimized for Apple Silicon inference. Each tensor is quantized at a single
bit width (2, 3, 4, 5, 6, or 8 bits), assigned based on its sensitivity tier.

Models stay quantized in GPU memory and dequantize on-the-fly using MLX native
Metal kernels (`quantized_matmul`, `gather_qmm`).

## v2 vs v1

| | v2 (current) | v1 (legacy) |
|---|---|---|
| **Storage** | MLX-native safetensors (uint32 weights) | JANG-packed uint8 qweight |
| **Load time** | **Instant** (mx.load mmap) | 5-10 min (repack needed) |
| **File naming** | `model-NNNNN.safetensors` | `model-NNNNN.jang.safetensors` |
| **Index** | `model.safetensors.index.json` | `model.jang.index.json` |
| **File size** | Same as v1 | Same as v2 |
| **config.json** | Has `quantization` key | No quantization key |

v2 stores weights in the exact format MLX expects — no conversion at load time.
Like GGUF: the file IS the runtime format.

Existing v1 models can be upgraded: `jang upgrade <model_path>`

## Directory Layout

### v2 (MLX-native)

```
model-name-JANG_2L/
  config.json                   # HuggingFace config + quantization key
  tokenizer.json                # HuggingFace tokenizer (unmodified)
  tokenizer_config.json         # Tokenizer settings (unmodified)
  special_tokens_map.json       # Special token mappings (unmodified)
  jang_config.json              # JANG quantization metadata
  model-00001-of-NNNNN.safetensors     # MLX-native quantized weights
  model.safetensors.index.json         # Standard safetensors index
```

### v1 (legacy)

```
model-name-JANG_2L/
  config.json
  tokenizer.json
  jang_config.json
  model-00001-of-NNNNN.jang.safetensors  # JANG-packed uint8 weights
  model.jang.index.json                   # JANG shard index
```

## v2 Weight Storage

Each quantized weight matrix is stored as three companion tensors in standard
safetensors format, using MLX-native naming and shapes:

### weight — packed quantized data

**Name**: `layers.N.self_attn.q_proj.weight`
**Type**: uint32
**Shape**: `(out_features, packed_in_features)` or `(num_experts, out_features, packed_in_features)`

Contains the quantized weight values packed into uint32 words, exactly as MLX's
`quantized_matmul` kernel expects. Each row is independently packed — no
cross-row byte spanning.

`packed_in_features = ceil(in_features * bits / 32)`

### scales — per-group scale factors

**Name**: `layers.N.self_attn.q_proj.scales`
**Type**: float16
**Shape**: `(out_features, n_groups)` or `(num_experts, out_features, n_groups)`

Scale factor for dequantization: `dequantized = quantized_int * scale + bias`

`n_groups = ceil(in_features / group_size)`

### biases — per-group bias values

**Name**: `layers.N.self_attn.q_proj.biases`
**Type**: float16
**Shape**: `(out_features, n_groups)` or `(num_experts, out_features, n_groups)`

Bias for asymmetric dequantization. Stored directly (no zero-point round-trip).

## jang_config.json

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
    "quantization_scheme": "asymmetric",
    "quantization_backend": "mx.quantize"
  },
  "source_model": {
    "name": "Qwen/Qwen3.5-122B-A10B",
    "dtype": "bfloat16",
    "parameters": "122B"
  },
  "runtime": {
    "total_weight_bytes": 49000000000,
    "total_weight_gb": 45.6
  }
}
```

### Required fields

- `format`: must be `"jang"`
- `format_version`: `"2.0"` for v2, `"1.1"` for v1
- `quantization.profile`: JANG profile name (e.g., `"JANG_2L"`)
- `quantization.block_size`: integer (default 64)
- `quantization.bit_widths_used`: array of integers, bit widths present
- `source_model.name`: HuggingFace model ID or name

## config.json (v2)

v2 models add a `quantization` key to the standard HuggingFace config:

```json
{
  "model_type": "qwen3_5_moe",
  "hidden_size": 3072,
  "quantization": {
    "group_size": 64,
    "bits": 2
  }
}
```

The `bits` value is the COMPRESS tier (most common bit width). The loader uses
this to create the model skeleton with `QuantizedLinear` layers, then corrects
per-layer bit widths by inspecting tensor shapes.

## Tier System

JANG classifies every tensor into a sensitivity tier:

| Tier | Examples | Sensitivity |
|------|----------|-------------|
| CRITICAL | Full softmax attention (q/k/v/o_proj), lm_head, MoE routers, MLA projections, SSM state | Highest — controls coherence |
| IMPORTANT | Embeddings, linear attention (GatedDeltaNet), shared experts | Moderate — degrades but doesn't break |
| COMPRESS | MLP gate/up/down, MoE experts, vision FFN, SSM projections | Most robust — bulk of parameters |

Profiles assign bits per tier: `(CRITICAL_bits, IMPORTANT_bits, COMPRESS_bits)`.

## Bit Width Inference

In v2, bit width is NOT stored as a companion tensor. The loader infers it
from the weight tensor shape:

```python
in_dim = scales.shape[-1] * group_size
actual_bits = (weight.shape[-1] * 32) // in_dim
```

This works because `packed_in_features = ceil(in_features * bits / 32)`.

## MoE Expert Storage

For MoE models, expert weights are pre-stacked into 3D tensors and renamed
to MLX's `switch_mlp` convention during conversion:

| Source (HuggingFace) | v2 Output (MLX-native) |
|---|---|
| `experts.gate_up_proj.weight` [E, 2I, H] | `switch_mlp.gate_proj.weight` [E, I, packed] + `switch_mlp.up_proj.weight` [E, I, packed] |
| `experts.down_proj.weight` [E, H, I] | `switch_mlp.down_proj.weight` [E, H, packed] |
| `experts.N.w1.weight` (per-expert) | `switch_mlp.gate_proj.weight` [E, I, packed] (stacked) |

This eliminates per-expert tensor iteration at load time.

## Non-quantized Tensors

Some tensors are stored in full precision (float16) without quantization:

- `model.norm.weight` — RMSNorm weights
- `layers.N.input_layernorm.weight` — per-layer norms
- Vision encoder weights (VL models)
- All `.bias` tensors

## Inference Loading

### v2 (instant)

```python
# 1. mx.load() via mmap — instant, no data copying
weights = mx.load("model-00001-of-00004.safetensors")

# 2. model.sanitize() handles any remaining renames
weights = model.sanitize(weights)

# 3. Load into QuantizedLinear / QuantizedSwitchLinear
model.load_weights(list(weights.items()), strict=False)

# 4. Fix per-layer bit widths (inferred from tensor shapes)
_fix_quantized_bits(model)
```

### v1 (legacy, slow)

```
1. Read .qweight (uint8), .scales, .biases, .bits from .jang.safetensors
2. Repack uint8 → uint32 (reinterpret bytes + reshape)
3. Reshape scales/biases from flat to (out_dim, n_groups)
4. Split gate_up_proj, stack per-expert weights
5. Load into model
```

## Upgrading v1 → v2

```bash
jang upgrade /path/to/model

# Or in Python:
from jang_tools.loader import upgrade_v1_to_v2
upgrade_v1_to_v2("/path/to/model")
```

This repacks the weights once and replaces the .jang.safetensors files with
standard .safetensors files. Model size stays the same.

## Compatibility

- v2 files are standard safetensors — any safetensors reader can parse them.
- `jang_config.json` is what makes a directory a JANG model.
- The JANG loader auto-detects v1 vs v2 and loads both.
- v1 models work forever — no forced migration.

## Versioning

- **1.0**: Original format with per-block `bit_map` and `block_offsets`
- **1.1**: Single `.bits` per tensor, `.shape` for original dimensions
- **2.0**: MLX-native storage — uint32 weights, float16 scales/biases in standard safetensors. Instant loading.
