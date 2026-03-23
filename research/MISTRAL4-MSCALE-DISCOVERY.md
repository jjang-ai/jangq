# Mistral Small 4: The mscale Attention Scale Bug

## Discovery Date: 2026-03-22

## The Bug

DeepSeek V2's MLX implementation applies `mscale * mscale` to the **overall attention scale**, but HuggingFace's Mistral 4 implementation applies it only to **RoPE cos/sin** (and for Mistral 4 specifically, it cancels to 1.0).

This is mathematically non-equivalent for MLA (Multi-head Latent Attention) because the key vector has two distinct parts:
- `k_nope`: content-based (no positional encoding)
- `k_rope`: position-based (with RoPE)

### DeepSeek V2 code (WRONG for Mistral 4):
```python
self.scale = q_head_dim ** -0.5  # 0.0884
if mscale_all_dim:
    mscale = yarn_get_mscale(factor, mscale_all_dim)
    self.scale = self.scale * mscale * mscale  # 0.195 — applied to EVERYTHING
```

### HuggingFace Mistral 4 code (CORRECT):
```python
self.scaling = self.qk_head_dim ** (-0.5)  # 0.0884 — plain, no mscale
# mscale goes into RoPE attention_scaling → applied to cos/sin
# For Mistral 4: attention_factor = mscale/mscale_all_dim = 1.4852/1.4852 = 1.0
```

### Why it matters for MLA:

Attention score = `scale * (q_nope @ k_nope^T + q_rope @ k_rope^T)`

**DeepSeek V2 approach** (scale overall by mscale²):
`= 0.195 * nope_dot + 0.195 * rope_dot`

**HF approach** (scale only rope via cos/sin):
`= 0.0884 * nope_dot + 0.0884 * F² * rope_dot`

Where F = attention_factor applied to cos/sin.

When F=1.0 (Mistral 4): HF gives `0.0884 * (nope + rope)`, DeepSeek V2 gives `0.195 * (nope + rope)`.

The 2.2x over-scaling of attention scores makes softmax too peaked → model over-attends to dominant tokens → repetition loops.

### Why DeepSeek V2 models tolerate this:

DeepSeek V2 has `qk_nope_head_dim=128, qk_rope_head_dim=64`. The nope part is 2/3 of the key, so over-scaling nope matters less. Also, the original DeepSeek V2 was TRAINED with this scale, so its weights compensate.

Mistral 4 has `qk_nope_head_dim=64, qk_rope_head_dim=64`. Equal 50/50 split means the error has MAXIMUM impact. And Mistral 4 was trained with the HF attention scale (0.0884), not the V2 scale (0.195).

## How It Was Found

1. **Symptom**: Model produces coherent short factual answers ("Paris") but degenerates into repetition loops for math, reasoning, coding
2. **Eliminated causes**: Not quantization (same at 2-bit and 4-bit), not KV cache (same without cache), not tokenizer (identical to HF)
3. **Fetched actual HF transformers `modeling_mistral4.py`** — found `self.scaling = qk_head_dim ** -0.5` with NO mscale
4. **Computed attention_factor** from HF's `_compute_yarn_parameters`: `mscale/mscale_all_dim = 1.4852/1.4852 = 1.0`
5. **Verified**: Setting scale to 0.0884 immediately improved output coherency from gibberish to correct multi-sentence responses

## The Fix

```python
# Instead of:
self.scale = self.scale * mscale * mscale

# Use HF-compatible attention_factor:
mscale_num = yarn_get_mscale(factor, config.rope_scaling.get("mscale", 1))
mscale_den = yarn_get_mscale(factor, mscale_all_dim)
attn_factor = mscale_num / mscale_den  # 1.0 for Mistral 4
# Apply attn_factor to rope output (q_pe, k_pe), not overall scale
```

## Impact

- **Mistral Small 4 119B**: Goes from completely broken to functional
- **DeepSeek V2 models**: No change (mscale values are different, and models trained with V2 scale)
- **Any future MLA model using HF's yarn**: Needs the HF-compatible attention_factor

## Timeline of Debugging

- **Attempt 1-2**: FP8 dequant crashes (corrupt files, missing skip patterns)
- **Attempt 3**: FP8 bfloat16 scale_inv silently failed → weights 1500x too large → NaN
- **Attempt 4**: FP8 fix applied, NaN gone but gibberish output
- **Architecture fixes**: norm_topk_prob, rope_interleave, llama4_scaling_beta added
- **norm_topk_prob=True**: Enabled simple factual recall ("Paris")
- **mscale discovery**: Reading actual HF Mistral4 source code line by line revealed the scale discrepancy
- **Scale fix**: Model now produces coherent multi-sentence output

## All Fixes Applied for Mistral 4

1. FP8 bfloat16 scale_inv loading (convert.py + fp8.py)
2. rope_interleave=True → traditional=False in MLX rope
3. norm_topk_prob=True in MoE gate
4. llama_4_scaling_beta=0.1 for position-dependent query scaling
5. **Attention scale: plain 1/sqrt(d), no mscale*mscale**
6. attention_factor = mscale/mscale_all_dim applied to rope output only
