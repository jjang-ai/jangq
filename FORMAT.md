# MLXQ Format Specification v1.0

> Mixed-Precision Importance Quantization for Apple Silicon
> Created by Eric Jang (eric@vmlx.net)

## Overview

MXQ is an open file format for storing mixed-precision quantized LLM weights
optimized for Apple Silicon inference. Each weight block can use a different
bit width (2, 3, 4, 5, 6, or 8 bits), allocated based on importance scoring
during calibration.

## Directory Layout

An MXQ model is a directory containing:

```
model-name-MXQ-Xbit/
  config.json                   # HuggingFace model config (unmodified)
  tokenizer.json                # HuggingFace tokenizer (unmodified)
  tokenizer_config.json         # Tokenizer settings (unmodified)
  special_tokens_map.json       # Special token mappings (unmodified)
  mxq_config.json               # MXQ quantization metadata
  mxq_imatrix.safetensors       # Importance matrix used for quantization
  model-00001-of-NNNNN.mxq.safetensors  # Quantized weight shards
  model.mxq.index.json          # Shard index mapping tensor names to files
```

## mxq_config.json

```json
{
  "format": "mxq",
  "format_version": "1.0",
  "quantization": {
    "method": "mxq-importance",
    "target_bits": 2.5,
    "actual_bits": 2.51,
    "block_size": 64,
    "calibration_dataset": "mxq-calib-v1",
    "calibration_samples": 512,
    "scoring_method": "awq+hessian",
    "bit_widths_used": [2, 3, 4, 5, 6, 8],
    "quantization_scheme": "asymmetric"
  },
  "source_model": {
    "name": "Qwen/Qwen3.5-72B",
    "dtype": "bfloat16",
    "parameters": "72B",
    "sha256": "..."
  },
  "quality_metrics": {
    "perplexity_bf16": 5.21,
    "perplexity_mxq": 5.38,
    "perplexity_uniform_4bit": 5.42
  },
  "runtime": {
    "recommended_memory_gb": 24,
    "total_weight_bytes": 23068672000
  }
}
```

### Required fields

- `format`: must be `"mxq"`
- `format_version`: semantic version string (currently `"1.0"`)
- `quantization.method`: must be `"mxq-importance"`
- `quantization.target_bits`: float, the target average bits per weight
- `quantization.actual_bits`: float, the actual average bits achieved
- `quantization.block_size`: integer, weights per quantization block (default 64)
- `quantization.bit_widths_used`: array of integers, bit widths present in this model
- `source_model.name`: string, HuggingFace model ID or name of source model

## Weight Tensor Storage

Each quantized weight matrix is stored as a set of companion tensors in
safetensors format. For a weight named `layers.N.self_attn.q_proj.weight`,
the following tensors are stored:

### qweight — packed quantized data

**Name**: `layers.N.self_attn.q_proj.qweight`
**Type**: uint8
**Shape**: (total_packed_bytes,)

Contains the quantized weight values packed at their assigned bit widths.
Blocks are stored contiguously — block 0 first, then block 1, etc.

Within each block, values are packed LSB-first (least significant bit first)
into bytes. For non-byte-aligned bit widths (3, 5, 6), values span byte
boundaries.

**Packing example (3-bit, block of 8 weights for illustration):**

```
Values: [5, 3, 7, 1, 0, 6, 2, 4]  (each 0-7, fits in 3 bits)

Bit stream (LSB first per value):
  101 011 111 001 000 110 010 100

Packed into bytes:
  Byte 0: 101_011_11  = 0xEF  (bits 0-7)
  Byte 1: 1_001_000_1  = 0x91  (bits 8-15)
  Byte 2: 10_010_100  = 0x94  (bits 16-23)

Total: 8 values × 3 bits = 24 bits = 3 bytes
```

### scales — per-block scale factors

**Name**: `layers.N.self_attn.q_proj.scales`
**Type**: float16
**Shape**: (n_blocks,)

Scale factor for dequantization: `dequantized = (raw_int - zero) * scale`

### zeros — per-block zero points

**Name**: `layers.N.self_attn.q_proj.zeros`
**Type**: float16
**Shape**: (n_blocks,)

Zero point for asymmetric quantization.

### bit_map — per-block bit widths

**Name**: `layers.N.self_attn.q_proj.bit_map`
**Type**: uint8
**Shape**: (n_blocks,)

The bit width assigned to each block. Values must be in {2, 3, 4, 5, 6, 8}.

### block_offsets — byte offsets into qweight

**Name**: `layers.N.self_attn.q_proj.block_offsets`
**Type**: uint32
**Shape**: (n_blocks,)

Byte offset of each block's data within the qweight array. Required because
variable bit widths mean block sizes in bytes are not uniform.

`block_offsets[i] = sum(ceil(block_size * bit_map[j] / 8) for j in 0..i-1)`

## Dequantization

To recover the float16 value for weight `w` in block `b`:

```
bits = bit_map[b]
byte_offset = block_offsets[b]
in_block_idx = w - b * block_size
bit_offset = in_block_idx * bits
byte_idx = byte_offset + bit_offset // 8
bit_shift = bit_offset % 8
mask = (1 << bits) - 1
raw = (read_uint16(qweight[byte_idx:byte_idx+2]) >> bit_shift) & mask
dequantized = float16(raw - zeros[b]) * scales[b]
```

Where `read_uint16` reads two bytes in little-endian order. Reading uint16
handles the case where a value spans a byte boundary.

## Block Layout

Weights in a matrix of shape (out_features, in_features) are divided into
blocks along the in_features (input channel) dimension:

```
For weight matrix W[out, in]:
  n_blocks_per_row = ceil(in_features / block_size)
  total_blocks = out_features * n_blocks_per_row

  Block index for W[i, j]:
    block = i * n_blocks_per_row + j // block_size
```

This layout ensures that blocks correspond to contiguous weight groups along
the input channel — matching the access pattern during matrix multiplication
(input activations are dot-producted with weight rows).

## Non-quantized Tensors

Some tensors are stored in full precision (float16 or bfloat16) without
quantization:

- `model.embed_tokens.weight` — stored at bit width specified in mxq_config
  (typically 4-bit or higher, but uses the same MXQ block format)
- `model.norm.weight` — RMSNorm weights, stored as float16
- `lm_head.weight` — may be quantized at higher bit width or stored as float16

Non-quantized tensors use standard safetensors storage (no companion tensors).

## Shard Index

`model.mxq.index.json` maps tensor names to shard files:

```json
{
  "metadata": {
    "format": "mxq",
    "total_size": 23068672000
  },
  "weight_map": {
    "layers.0.self_attn.q_proj.qweight": "model-00001-of-00004.mxq.safetensors",
    "layers.0.self_attn.q_proj.scales": "model-00001-of-00004.mxq.safetensors",
    "layers.0.self_attn.q_proj.zeros": "model-00001-of-00004.mxq.safetensors",
    "layers.0.self_attn.q_proj.bit_map": "model-00001-of-00004.mxq.safetensors",
    "layers.0.self_attn.q_proj.block_offsets": "model-00001-of-00004.mxq.safetensors"
  }
}
```

## Importance Matrix

`mxq_imatrix.safetensors` stores the importance scores used during
quantization, enabling reproducibility and re-quantization at different
bit targets:

```
layers.N.self_attn.q_proj.importance   # (n_blocks,) float32
layers.N.self_attn.q_proj.act_norms    # (in_features,) float32
```

## Compatibility

- MXQ files use standard safetensors format — any safetensors reader can
  parse the container, even if it doesn't understand MXQ semantics.
- Model config and tokenizer files are unmodified HuggingFace format.
- The `mxq_config.json` file is the indicator that a directory contains
  an MXQ model — loaders should check for its presence.

## Versioning

The format version follows semantic versioning:
- **1.x**: backward-compatible additions (new optional fields)
- **2.0**: breaking changes to tensor layout or packing

Loaders should check `format_version` and reject versions they don't support.
