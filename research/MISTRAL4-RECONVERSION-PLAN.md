# Mistral Small 4 — Reconversion Plan

## Changes to convert.py

### Change 1: Vision Conv Weight Transpose (line ~384)
**BEFORE:** passthrough stores conv weight as-is (OIHW format from PyTorch/HF)
**AFTER:** Transpose 4D conv weights from OIHW → OHWI for MLX's conv2d

```python
# BEFORE:
passthrough[tensor_name] = w.astype(np.float16) if w.dtype != np.float16 else w

# AFTER:
w_out = w.astype(np.float16) if w.dtype != np.float16 else w
if len(shape) == 4:  # Conv: (O, I, H, W) → (O, H, W, I)
    w_out = np.transpose(w_out, (0, 2, 3, 1))
passthrough[tensor_name] = w_out
```

**Verification:** MLX conv2d expects weight (out_channels, kH, kW, in_channels).
Source: (1024, 3, 14, 14) → target: (1024, 14, 14, 3)

### Change 2: MoE Gate as Float16 Passthrough (new, before line ~440)
**BEFORE:** Gate goes through normal quantization path (FP8→f32→f16→8bit)
**AFTER:** Gate stored as float16 passthrough (no quantization)

```python
# After loading weights, before quantization:
if _is_moe_gate(tensor_name):
    passthrough[tensor_name] = weights.astype(np.float16)
    offset += n_blocks
    continue
```

Where `_is_moe_gate` checks: `.mlp.gate.weight` but NOT `gate_proj` or `switch_mlp.gate_proj`

**Why:** Gate is (128, 4096) per layer × 36 layers = 18 MB total.
Negligible size impact. Eliminates ALL gate quantization issues:
- No bf16/f16 precision problems
- No float32 promotion (gate stays f16)
- Routing precision = same as source model
- Speed: full float16 throughput (~80 tok/s)

### Change 3: Verify Skip Patterns
Currently skipped: activation_scale, scale_inv, vision_encoder, patch_conv
Need to verify: all Mistral 4 metadata tensors are properly skipped.

### Change 4: Verify All Tensor Names Map Correctly
JANG stores: language_model.model.layers.N.mlp.switch_mlp.{gate_proj,up_proj,down_proj}
JANG stores: language_model.model.layers.N.mlp.gate.weight (float16 passthrough)
JANG stores: language_model.model.layers.N.self_attn.{q_a_proj,q_b_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj}
JANG stores: language_model.model.layers.N.{input_layernorm,post_attention_layernorm}.weight (float)
JANG stores: language_model.model.layers.N.self_attn.{q_a_layernorm,kv_a_layernorm}.weight (float)
JANG stores: vision_tower.* (float16 passthrough, with conv transposed)

## Profile: JANG_2L
- CRITICAL (8-bit): attention (q_a, q_b, kv_a, kv_b, o_proj), embed, lm_head
- IMPORTANT (6-bit): shared experts
- COMPRESS (2-bit): routed experts (gate_proj, up_proj, down_proj)
- Expected size: ~30 GB
- Gate: float16 passthrough (18 MB)

## Pre-flight Checks
- [ ] Source model exists: /Users/eric/models/Mistral-Small-4-119B
- [ ] Internal drive has space: need ~30 GB free
- [ ] All skip patterns verified
- [ ] Conv transpose tested
- [ ] Gate passthrough tested
- [ ] config.json will have dtype: "bfloat16" or auto-bf16 in loader handles it

---

## Conversion Attempt Log

### Attempt 1 (v2): CRASHED — double offset increment
**Error:** `IndexError: index 0 is out of bounds for axis 0 with size 0` at line 449
**Cause:** Gate passthrough code at line 445 did `offset += n_blocks` but offset was ALREADY
incremented at line 412 (before the gate check). Double increment caused the next tensor's
`bit_alloc` slice to be empty.
**Fix:** Removed `offset += n_blocks` from gate passthrough block (line 445). Added comment.
**Time wasted:** ~35 min (allocation completed, crashed at 97% quantization)

### Attempt 2 (v3): RUNNING
Started at ~3:30 PM. Expected completion ~4:35 PM.
- Allocation: ~30 min
- Quantization: ~35 min (401 tensors, 36 gates as f16 passthrough)
- Write: ~5 min

### Changes Applied in This Conversion
1. **Conv transpose (line 385-387):** `np.transpose(w_out, (0, 2, 3, 1))` for 4D weights
2. **Gate f16 passthrough (line 443-447):** MoE gate stored as float16, no quantization
3. **FP8 bf16 scale_inv (line 399-401):** `_load_bf16_from_header()` for bfloat16 scalars
4. **Double offset fix (line 445):** Removed duplicate `offset += n_blocks`

### Expected Output
- Profile: JANG_2L (CRITICAL=8, IMPORTANT=6, COMPRESS=2)
- Size: ~30 GB
- 36 gate weights as float16 passthrough (~18 MB)
- Vision conv weights transposed to OHWI
- All other weights quantized per JANG_2L allocation

### Post-Conversion Test Plan
1. Load with JANG loader (auto-bf16 for MLA models)
2. Verify speed: target 80+ tok/s gen
3. Verify quality: photosynthesis, code, math, reasoning, [THINK] tags
4. Verify VLM: load vision tower, test image description
5. Verify gate dtype: should be float16 (not bf16, not quantized)
6. Verify conv weight shape: (1024, 14, 14, 3) not (1024, 3, 14, 14)
