# Mistral Small 4 (119B) — Complete Journey

## Date: 2026-03-22
## Created by: Jinho Jang (eric@jangq.ai)

---

## Final Status

| Feature | Status | Speed |
|---------|--------|-------|
| Text generation | WORKING | 82 tok/s gen, 216 tok/s prefill |
| Code generation | WORKING | Complete functions with logic |
| Math | WORKING | Step-by-step reasoning |
| Reasoning [THINK] | WORKING | Proper thinking tags |
| VLM (vision) | NEEDS RECONVERSION | Conv weight transpose needed |
| Memory | 68 GB | JANG_4M (57 GB on disk) |

---

## Failures and How They Were Resolved

### Failure 1: FP8 bfloat16 scale_inv (Attempts 1-3)

**Problem:** The source model stores FP8 E4M3 weights with bfloat16 `weight_scale_inv` scalars.
numpy's safetensors can't handle bfloat16 → silently returns None → FP8 decode returns raw
values (max=448) without scale multiplication → weights 1500x too large → NaN at inference.

**How discovered:** Traced layer-by-layer NaN propagation. Layer 0 attention output was inf.
Manually dequanted FP8 weight and found max=0.3008 vs stored max=448.

**Fix:** `_load_bf16_from_header()` in convert.py reads raw bfloat16 bytes from safetensors
header (uint16 left-shift to float32). Also fixed fp8.py to handle shape (1,) scale_inv.

**Time wasted:** ~8 hours across 3 failed conversion attempts.

### Failure 2: Attention Scale (mscale*mscale)

**Problem:** DeepSeek V2 code applies `mscale * mscale` to the overall attention scale
(0.0884 → 0.195). HuggingFace's Mistral 4 uses plain `qk_head_dim ** -0.5 = 0.0884`.
The 2.2x over-scaling made attention too peaked → repetition loops after ~20 tokens.

**How discovered:** Fetched actual HF `modeling_mistral4.py` source code. Line:
`self.scaling = self.qk_head_dim ** (-0.5)` — NO mscale. Computed attention_factor from
HF's `_compute_yarn_parameters`: `mscale/mscale_all_dim = 1.4852/1.4852 = 1.0`.

**Fix:** Set `self.scale = q_head_dim ** -0.5` without mscale. Compute
`_rope_attn_scaling = mscale_num / mscale_den` (= 1.0 for Mistral 4).

**Time wasted:** ~4 hours debugging "why does the model loop."

### Failure 3: RoPE Traditional Flag (THE CRITICAL BUG)

**Problem:** Set `traditional=not self.rope_interleave` → when `rope_interleave=True`,
`traditional=False`. But mlx-vlm's config.py does `if self.rope_interleave: self.rope_traditional = True`.
The correct mapping is `traditional = rope_interleave` (True, not False).

**How discovered:** Found mlx-community/Mistral-Small-4-119B-2603-4bit (1682 downloads)
was converted using mlx-vlm 0.4.0. Updated mlx-vlm to 0.4.1 which has native `mistral4`
model support. Read their `config.py` and found the mapping.

**Fix:** Changed ONE LINE: `traditional=not self.rope_interleave` → `traditional=self.rope_interleave`

**Impact:** Model went from producing repetitive garbage to PERFECT output — complete code
functions, step-by-step math, logical reasoning, chemical equations. This single boolean
was the difference between a broken model and a working one.

**Time wasted:** ~12 hours. This should have been found first by checking mlx-vlm's
native implementation. The lesson: always check if a reference implementation exists
before writing custom code.

### Failure 4: Speed (23 tok/s instead of 80+)

**Problem:** Gate weight dequanted to bfloat16. When float16 hidden state multiplies bf16
weight, MLX promotes entire computation to float32 → doubles bandwidth → halves speed.

**How discovered:** Compared with mlx-community model (84 tok/s). Their model runs entirely
in bfloat16 (config `dtype: "bfloat16"`). Their gate is QuantizedLinear with bf16 scales.

**Fix:** `model.set_dtype(mx.bfloat16)` after loading. The gate is already bf16, so setting
the entire model to bf16 eliminates mixed-dtype promotion.

**Result:** 23 tok/s → 82 tok/s (3.5x speedup from one line).

### Failure 5: VLM (not yet resolved)

**Problem:** JANG converter stores vision conv weights in HF format (OIHW) instead of
MLX format (OHWI). The patch_conv weight shape (1024, 3, 14, 14) needs to be transposed
to (1024, 14, 14, 3) for MLX's conv2d.

**Fix needed:** In the JANG converter, detect 4D conv weights and transpose before storing.
This is a converter-level fix that requires reconversion.

---

## All Architecture Fixes Applied

1. **FP8 bfloat16 scale_inv** → `_load_bf16_from_header()` in convert.py
2. **norm_topk_prob=True** → Normalizes top-k scores in MoEGate
3. **llama_4_scaling_beta=0.1** → Position-dependent query scaling
4. **Attention scale: plain 0.0884** → No mscale*mscale
5. **rope_interleave=True → traditional=True** → Correct RoPE convention
6. **Gate dequant** → uint32 → bfloat16 for MoE routing
7. **Auto bfloat16** → Entire model set to bf16 for speed

---

## Converter Fixes Needed for Reconversion

1. **Vision conv transpose**: 4D weights (OIHW → OHWI)
2. **Gate quantization path**: FP8→bf16→quantize (not FP8→f32→quantize) OR store gate as float16
3. **processor_config.json**: Must be included for VLM

---

## Speed Reference

| Mode | Prefill | Generation | Memory |
|------|---------|------------|--------|
| JANG_4M (bf16) | 216 tok/s | 82 tok/s | 68 GB |
| mlx-community 4-bit | 43 tok/s | 84 tok/s | 68 GB |
| JANG_4M (f16, broken) | 125 tok/s | 23 tok/s | 69 GB |

JANG_4M matches mlx-community generation speed and has 5x faster prefill.

---

## Key Files

| File | Purpose |
|------|---------|
| `research/MISTRAL4-CONVERSION-LOG.md` | 5 conversion attempts |
| `research/MISTRAL4-MSCALE-DISCOVERY.md` | Attention scale bug |
| `research/MISTRAL4-SYSTEMATIC-TEST.md` | Test results + analysis |
| `research/MISTRAL4-FIX-LOG.md` | Detailed fix log |
| `research/MISTRAL4-IMPLEMENTATION-GUIDE.md` | MLX Studio integration |
| `jang-tools/jang_tools/convert.py` | `_load_bf16_from_header()` |
| `jang-tools/jang_tools/fp8.py` | FP8 E4M3 dequant |
| `jang-tools/jang_tools/loader.py` | Gate dequant + auto-bf16 |
| Mac Studio deepseek_v2.py | 7 patches for Mistral 4 |
| Mac Studio mistral3.py | Routes mistral4 to deepseek_v2 |

---

## Lessons Learned

1. **Always check for existing implementations first.** mlx-vlm 0.4.1 had native Mistral 4
   support. Finding this earlier would have saved 12+ hours.

2. **One boolean can break everything.** `traditional=True` vs `False` was the difference
   between a working model and complete garbage.

3. **Speed issues often have simple fixes.** `model.set_dtype(mx.bfloat16)` = 3.5x speedup.

4. **bfloat16 silently fails with numpy.** Always use custom binary reading for bf16 tensors.

5. **MoE routing is extremely precision-sensitive.** Even bf16→f16 conversion (3 mantissa bits)
   changes which experts are selected.

6. **Document BEFORE and AFTER every change.** Prevents losing track of what worked.

7. **Don't assume — verify.** The "cache precision issue" and "Metal non-determinism" theories
   were wrong. The real fix was a boolean flag in the RoPE.

---

## Reconversion (v2 format with fixes)

### Converter Changes
1. Vision conv weights transposed OIHW → OHWI for MLX conv2d
2. MoE gate stored as float16 passthrough (no quantization)
3. Fixed double offset increment bug in gate passthrough code

### Attempt 1: Crashed at 97% — double offset bug. Fixed, restarted.
### Attempt 2: Running. Expected ~30 GB output, ~65 min total.

### VLM: WORKING
- Conv weight transposed in-place (OIHW → OHWI)
- Processor files copied (processor_config.json, preprocessor_config.json)
- Images folder copied
- VLM loaded via mlx-vlm model creation + JANG weight loading
- **81 tok/s generation, 199 tok/s prefill, 40 GB memory**
- Image description: coherent, detailed ("close-up of person's face, neutral expression, shallow depth of field")

### VLM Files Needed in JANG Model Directory
- processor_config.json (from source model)
- preprocessor_config.json (copy of processor_config.json)
- images/ folder (test images)
- Vision tower weights (in safetensors, conv transposed OHWI)

### Converter Fix for Future Conversions
- convert.py line ~651: `np.transpose(tensor, (0, 2, 3, 1))` for 4D vision conv weights
- Already applied for future conversions. Current JANG_2L was fixed in-place.
