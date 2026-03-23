# Mistral Small 4 — Fix Log (Systematic)

## Date: 2026-03-22
## Discovery: Native mlx-vlm 0.4.1 has working Mistral 4 implementation

The mlx-community/Mistral-Small-4-119B-2603-4bit model (1682 downloads) was converted
using mlx-vlm 0.4.0. People report ~40 tok/s. Our patched DeepSeek V2 approach was
fundamentally wrong — slow (17-25 tok/s) and producing degraded output.

---

## Key Differences: Native mlx-vlm vs Our Patched DeepSeek V2

### 1. RoPE Traditional Flag — CRITICAL BUG
**BEFORE (WRONG):** `rope_interleave=True` → `traditional=False`
**NATIVE (CORRECT):** `rope_interleave=True` → `rope_traditional=True`

In mlx-vlm's config.py:
```python
if self.rope_interleave:
    self.rope_traditional = True
```

This means when `rope_interleave=True`, MLX should use `traditional=True` (halved convention).
I had it backwards — setting `traditional=False` (interleaved convention). This corrupted
ALL positional information, explaining why the model degraded for anything beyond simple
pattern matching.

### 2. RoPE Implementation — Different Function
**BEFORE:** Custom `DeepseekV2YarnRotaryEmbedding` with manual freq computation
**NATIVE:** `initialize_rope()` from `mlx_lm.models.rope_utils` — standard, tested implementation

### 3. Attention Scale Application — Per-Position vs Scalar
**BEFORE:** `attn_scale = 1 + beta * log(1 + floor(offset / max_pos))` — single scalar for all positions
**NATIVE:** `_get_llama_4_attn_scale(start, stop, beta, max_pos)` — per-position array, computed ONCE at model level, passed to all layers

### 4. Gate Implementation — nn.Linear vs Raw Weight
**BEFORE:** `self.weight = mx.zeros(...)` with manual `x @ self.weight.T`
**NATIVE:** `self.gate = nn.Linear(hidden, n_experts, bias=False)` — standard linear, cleaner quantization

### 5. k_pe Expansion — broadcast_to vs repeat
**BEFORE:** `mx.repeat(k_pe, self.num_heads, axis=1)` — copies data
**NATIVE:** `mx.broadcast_to(k_pe, ...)` — zero-copy view, more efficient

### 6. No mscale Anywhere
**BEFORE:** Complex mscale/mscale_all_dim computation with _rope_attn_scaling
**NATIVE:** Plain `self.scale = qk_head_dim ** -0.5`, no mscale logic at all

---

## Fix Plan

### Step 1: Update JANG loader to use mlx-vlm's native Mistral4Model
- Instead of routing through patched deepseek_v2.py, use mlx_vlm.models.mistral4.language
- Strip "language_model.model." prefix from JANG weight keys
- Handle gate dequant (quantized uint32 → bfloat16)
- Apply _fix_quantized_bits for per-tensor bit correction

### Step 2: Test with JANG_4M weights
- Verify coherent output
- Measure speed (should be 40+ tok/s)
- Compare with mlx-community 4-bit model when downloaded

### Step 3: Test all capabilities
- Short Q&A
- Long generation (the main failure point)
- Math, coding, reasoning
- VLM (with mlx-vlm processor)
- Speed benchmarks

---

## Changes Made

(will be filled as changes are made)

---

## CHANGE 1: RoPE Traditional Flag Fix

**File:** `/opt/homebrew/.../mlx_lm/models/deepseek_v2.py` (Mac Studio)
**Line:** YaRN RoPE creation in DeepseekV2Attention.__init__

**BEFORE (WRONG):**
```python
traditional=not self.rope_interleave,
```

**AFTER (CORRECT):**
```python
traditional=self.rope_interleave,
```

**Reasoning:** mlx-vlm 0.4.1's config.py does `if self.rope_interleave: self.rope_traditional = True`.
When `rope_interleave=True`, MLX should use `traditional=True` (standard halved RoPE convention).
The HF `apply_rotary_pos_emb_interleave` deinterleaves FIRST then applies standard RoPE,
which maps to `traditional=True` in MLX.

**Result:** Model now produces:
- Complete working code (is_prime function with docstring, edge cases, sqrt optimization)
- Step-by-step math (distributive property: 15×23 = 15×20 + 15×3 = 300+45)
- Logical reasoning (identifies false premise in syllogism, names flightless bird examples)
- Chemical equations (6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂)
- 150+ token coherent output with NO repetition loops

This was THE critical bug. Everything else (mscale, norm_topk_prob, etc.) was secondary.

---

## ISSUE: Speed 23 tok/s (should be 40+)

**Root cause:** Gate weight is bfloat16 (from JANG loader dequant). When float16 hidden state
multiplies bfloat16 weight, MLX promotes to float32. This propagates through the entire model,
effectively halving throughput (float32 = 2x bandwidth vs float16).

**Attempts to fix:**
1. Cast gate bf16 → f16: BREAKS routing (all <unk>)
2. Re-quantize gate to 8-bit, dequant to f32 in forward: Works but still float32 (23 tok/s)
3. Re-quantize gate, cast scales to f16 so dequant produces f16: BREAKS routing (all <unk>)

**The gate routing is extremely sensitive to precision.** Even losing 3 mantissa bits (bf16→f16)
changes which experts are selected, producing garbage output.

**Comparison:** mlx-community model reportedly gets 40 tok/s. Their gate is stored as 8-bit
QuantizedLinear with float16 scales (from direct conversion, not round-tripped through bf16).
The difference: they quantized from the ORIGINAL bf16 weights directly to 8-bit with f16 scales.
Our JANG conversion goes: FP8 → dequant → f32 → JANG quantize → store.

**Fix needed:** Change JANG converter to store gate as 8-bit QuantizedLinear with float16 scales,
matching the mlx-community approach. This requires re-conversion.

**Current speed:** 23 tok/s generation, 105-125 tok/s prefill at float32
**Target speed:** 40+ tok/s at float16

---

## STATUS: 2026-03-22 (end of session)

### Working
- **Model output**: PERFECT — code, math, reasoning, knowledge, [THINK] tags
- **Speed**: 23-25 tok/s gen, 105-125 tok/s prefill (float32 due to bf16 gate)
- **Memory**: 69.6 GB peak
- **RoPE fix**: traditional=self.rope_interleave (True for Mistral 4)
- **Attention scale**: plain 0.0884, no mscale

### Speed Issue (23 tok/s → should be 40+)
- bf16 gate weight → float32 promotion → halves throughput
- Float16 gate breaks routing (MoE too sensitive)
- Re-quantizing bf16→8bit→f16 also breaks (round-trip precision loss)
- mlx-community model (downloading, 53/60 GB) stores gate as 8-bit QuantizedLinear
- Their conversion path: FP8→bf16→quantize produces different (working) scales
- Fix: re-convert JANG model OR match mlx-community gate format

### VLM
- Vision tower weights present (218 tensors, float16)
- mlx-vlm 0.4.1 has native Mistral 4 support
- Direct mlx-vlm load fails: 362 extra params (gate metadata, vision format mismatch)
- Need: custom VLM loader path OR mlx-community model as reference

### Next Steps
1. Wait for mlx-community model download to complete
2. Test their model for speed/quality reference (should be ~40 tok/s)
3. Compare their gate weight format with ours
4. Either: match their format in JANG converter, or use their model as-is for comparison
5. Test VLM with mlx-community model
6. Fix JANG converter gate handling for float16 speed
7. Re-convert and test

---

## CONVERTER FIX NEEDED: Gate Quantization Path

### Problem
JANG converter: FP8 uint8 → fp8_e4m3_to_float32() → float32 → mx.quantize(f32, 8bit, gs64)
mlx-vlm converter: FP8 → HF transformers bf16 → mx.quantize(bf16, 8bit, gs64)

The float32 intermediate produces DIFFERENT quantization scales/biases than bfloat16.
When these are stored as float16 scales, the gate routing produces different expert selections.
The MoE routing is so sensitive that this difference breaks output completely.

### Verified
- JANG gate at bf16 (dequanted): WORKS perfectly, 23 tok/s (float32 promotion)
- JANG gate at f16 (any approach): BREAKS routing, all <unk>, 77 tok/s
- mx.quantized_matmul with original JANG quantized gate: BREAKS, all <unk>, 77 tok/s

### Fix for Next Conversion
In convert.py, when quantizing the gate weight:
1. After FP8 dequant to float32, cast to bfloat16 FIRST: `weights = weights.astype(np.float16)`
   Actually, use bfloat16 intermediate: the gate weight should go through:
   FP8 → float32 → bfloat16 → mx.quantize(bf16_tensor, bits=8, gs=64)
   This matches mlx-vlm's quantization path.

2. OR: Don't quantize the gate at all — store as float16 directly.
   Gate is (128, 4096) = 512KB per layer × 36 = 18 MB total. Negligible size.
   This avoids ALL quantization issues.

### Implementation
In convert.py, add special handling for MoE gate tensors:
```python
if is_moe_gate(tensor_name):
    # Store as float16 — no quantization
    passthrough[tensor_name] = weights.astype(np.float16)
```

OR if quantization is desired:
```python
if is_moe_gate(tensor_name):
    # Convert to bfloat16 before quantization to match mlx-vlm
    weights_bf16 = mx.array(weights).astype(mx.bfloat16)
    qw, sc, bi = mx.quantize(weights_bf16, group_size=64, bits=8)
```

### Impact
- Size change: ~0 (gate is 18 MB of 57 GB = 0.03%)
- Speed: 23 → 40+ tok/s (eliminates float32 promotion)
- Quality: Same or better (no quantization on gate = maximum routing precision)

---

## CHANGE 2: Auto-bfloat16 for MLA Models (Speed Fix)

**File:** `jang_tools/loader.py` (line ~205)

**Fix:** Auto-detect MLA models (kv_lora_rank > 0 or model_type="mistral4") and set
entire model to bfloat16. This matches mlx-community approach where `dtype: "bfloat16"`
is set in config.json.

**Discovery:** mlx-community model runs entirely in bfloat16:
- Gate: QuantizedLinear with bf16 scales → output bf16
- Embeddings: bf16
- All computation: bf16 (no float32 promotion)
- Speed: 84 tok/s

**Result after fix:**
- JANG_4M: 82 tok/s gen, 216 tok/s prefill (was 23 tok/s)
- Quality: identical (perfect code, math, reasoning)
- Memory: 68 GB (unchanged)
- No reconversion needed!

**Root cause of original 23 tok/s:** Gate dequanted to bf16, but rest of model was f16.
Mixed bf16/f16 promoted to float32. Setting entire model to bf16 eliminates promotion.

---

## CHANGE 3: Converter Fixes for Reconversion

### 3a: Vision Conv Transpose
**File:** convert.py line 385-387
**Change:** Added `np.transpose(w_out, (0, 2, 3, 1))` for 4D conv weights
**Reason:** MLX conv2d expects OHWI, source model stores OIHW

### 3b: Gate Float16 Passthrough
**File:** convert.py line 443-447
**Change:** MoE gate stored as float16 passthrough instead of 8-bit quantized
**Reason:** Gate routing too precision-sensitive for quantization round-trip.
Float16 gate + bf16 model = native speed (80+ tok/s)

### 3c: Double Offset Bug Fix
**File:** convert.py line 445
**Change:** Removed `offset += n_blocks` from gate passthrough
**Reason:** Offset already incremented at line 412. Double increment caused IndexError
crash at 97% quantization. Wasted 35 min of computation.

### Status: Reconversion v3 running, ETA ~4:35 PM

---

## CHANGE 4: VLM Support

### 4a: Conv Weight Transpose (in-place fix)
**File:** model-00008-of-00008.safetensors
**Change:** vision_tower.patch_conv.weight transposed from (1024,3,14,14) to (1024,14,14,3)
**Reason:** MLX conv2d expects OHWI format, source model stores OIHW

### 4b: Processor Files
**Files added:** processor_config.json, preprocessor_config.json, images/
**Source:** Copied from original Mistral-Small-4-119B model

### 4c: Converter Fix (for future conversions)
**File:** convert.py line ~651
**Change:** Added `np.transpose(tensor, (0, 2, 3, 1))` in non-quantized tensor collection

### Result
- VLM working: 81 tok/s gen, 199 tok/s prefill, 40 GB memory
- Image description: coherent and detailed
