# Experiment 053: Quantized-In-Memory Inference — Debug Log

**Date**: 2026-03-15
**Author**: Jinho Jang (eric@jangq.ai)
**Status**: IN PROGRESS

## Problem

JANG models must stay quantized in GPU memory (like GGUF), not expand to float16.
The dequant-at-load approach was fundamentally wrong — a 230B model at 2-bit should use
~60 GB RAM, not 460 GB.

## Approach: Repack JANG → MLX Native Quantized Format

MLX has `quantized_matmul` and `gather_qmm` that dequantize on-the-fly during matmul.
We repack JANG uint8 packed weights → MLX uint32 format at load time.

### Verified Mapping
- JANG qweight (uint8 LSB) → view as uint32 (same bytes, zero copy)
- JANG scale → MLX scale (same)
- JANG zero → MLX bias = -scale * zero
- JANG block_size=64 → MLX group_size=64
- All bit widths (2,3,4,5,6,8) supported

### Verified: No Speed Regression with Mixed Bits
- Mixed 2+8 bit: 5490 fwd/s
- Uniform 2-bit: 5731 fwd/s
- NO regression — safe to use per-tensor bit widths

## Attempt Log

### Attempt 1: Basic repack
- Created `_repack_jang_to_mlx()` replacing `_dequantize_jang_weights()`
- Set up `quantization` config for QuantizedLinear creation
- **Result**: Loaded at 0.23 GB (correct!) but QuantizedEmbedding had wrong bits
- **Error**: `ValueError: Shape of scales and biases does not match` for embed_tokens

### Attempt 2: Fix QuantizedEmbedding bits
- Added `_fix_quantized_bits()` to correct bits from weight shape
- Include `QuantizedEmbedding` in the fix
- **Result**: Model loaded, generated text — 0.5B model works!
- **Success on 0.5B**: 0.23 GB memory, generates text

### Attempt 3: Test on 35B MoE
- **Error**: `ValueError: gather_mm shape mismatch` — MoE expert tensors wrong shape
- **Cause**: 3D expert tensors flattened to 2D, but MoE uses `gather_mm`/`gather_qmm`

### Attempt 4: Keep 3D expert tensors
- Modified repacker to keep `[num_experts, out, packed]` shape for 3D
- **Error**: Same gather_mm error — SwitchLinear expects float, gets uint32

### Attempt 5: Pre-split gate_up_proj
- Pre-split `experts.gate_up_proj [256,1024,2048]` → `switch_mlp.gate_proj [256,512,2048]` + `switch_mlp.up_proj`
- Also pre-split `experts.down_proj` → `switch_mlp.down_proj`
- **Error**: `_fix_quantized_bits` accesses `.scales` on SwitchLinear (doesn't have it)

### Attempt 6: Upgrade SwitchLinear → QuantizedSwitchLinear BEFORE loading
- Added `_upgrade_switch_to_quantized()` that replaces all SwitchLinear with
  QuantizedSwitchLinear in the model skeleton before `load_weights()`
- **Error**: `AttributeError: 'list' object has no attribute '39'` — module path
  traversal didn't handle list indices for `model.layers.39`

### Attempt 7: Fix list index traversal
- Handle `p.isdigit()` → `parent[int(p)]` in module path traversal
- **Testing now...**

## Architecture Challenge: Qwen3.5 MoE Weight Flow

The Qwen3.5 MoE model has a complex weight flow:

1. HF safetensors stores: `experts.gate_up_proj [256, 1024, 2048]` (fused gate+up for all experts)
2. `sanitize()` splits to: `switch_mlp.gate_proj [256, 512, 2048]` + `switch_mlp.up_proj [256, 512, 2048]`
3. These go into `SwitchLinear` layers (non-quantized, uses `gather_mm`)
4. When quantized, `SwitchLinear.to_quantized()` → `QuantizedSwitchLinear` (uses `gather_qmm`)

For JANG:
- JANG stores `experts.gate_up_proj` as quantized uint8
- Repacker must: unpack → repack as uint32 → pre-split → name as switch_mlp.*
- Model skeleton must have QuantizedSwitchLinear (not SwitchLinear) for the weights to load correctly
- `gather_qmm` handles the per-expert quantized matmul in Metal
