# 397B Fix: bfloat16 Compute Dtype

Created by Jinho Jang (eric@jangq.ai) — 2026-03-19

## The Problem

Qwen3.5-397B-A17B JANG models produce NaN regardless of bit allocation (2-bit, 3-bit,
even 8-bit shared expert). The overflow happens at layer 22's shared expert down_proj.

## Root Cause (Confirmed by Diagnostic)

MLX defaults to **float16** activations (max 65,504). The overflow chain:

```
Layer 22 shared expert:
  Input to MLP:          max = 12.37  (normal)
  gate_proj output:      max = 119.25 (normal for this model)
  up_proj output:        max = 69.50  (normal)
  SiLU(gate) * up:       max = 8,288  (fits float16)
  down_proj(8288 input): dot product across 1024 dims → 281,675 → EXCEEDS 65,504 → inf
```

The down_proj matmul accumulates in float32 internally (Metal kernel), but the
OUTPUT is written as float16. 281,675 > 65,504 → float16 inf → NaN propagates.

**This is NOT a quantization error.** The gate output of 119 is valid (model actually
produces this at full precision). The shared expert was at 8-bit. The problem is purely
the float16 output dtype of the matmul.

## The Fix: bfloat16

Cast embedding output to bfloat16 before the forward pass. All subsequent operations
propagate in bfloat16/float32.

| Property | float16 | bfloat16 |
|----------|---------|----------|
| Max value | 65,504 | 3.4×10^38 |
| Mantissa | 10-bit | 7-bit |
| Precision | ~0.001 at 1.0 | ~0.008 at 1.0 |

bfloat16 trades precision for range. At 2-4 bit quantization, the quantization noise
is orders of magnitude larger than the bfloat16 precision loss. Quality impact: zero.

## Verification (2026-03-19)

```
float16:  Layer 22 → inf → NaN (every time, every bit allocation)
bfloat16: All 60 layers clean, logits valid, top token produced

After norm: max=36.41 nan=False
Logits: nan=False inf=False
Top token: 11 = ","
BFLOAT16 FULL PASS: SUCCESS
```

## Why Previous Fixes Failed

| Attempt | What we tried | Why it failed |
|---------|--------------|---------------|
| shared_expert → CRITICAL (8-bit) | More bits on shared expert | Still overflows float16 (8-bit output = 119, valid) |
| MLP asymmetry (gate=4, up=2, down=3) | Protect gate_proj | Still overflows float16 (gate at ANY bits can produce 119) |
| JANG_3M (all 3-bit experts) | Higher bits everywhere | Still overflows float16 |
| JANG_2L + MLP fix | Mixed bits with floors | Still overflows float16 |
| Input scaling (÷128) | Reduce input magnitude | Rescaling overflows float16 on the way back |

**All failed because the problem was never the bit allocation.** It was the compute dtype.

## Which Models Need bfloat16

| Condition | Why | Models |
|-----------|-----|--------|
| 512+ experts | Less specialized experts → larger gate outputs | Qwen3.5-397B, Nemotron-120B |
| hidden_size ≥ 4096 | Longer dot products → larger sums | Same |
| Both together | Guaranteed to overflow float16 | Same |

256-expert models (35B, 122B, MiniMax) with hidden ≤ 3072 do NOT overflow float16.

## Implementation

In the JANG loader, after loading weights, cast embedding output to bfloat16:

```python
# In model forward pass or loader wrapper
h = embed_tokens(input_ids).astype(mx.bfloat16)
```

All subsequent operations propagate in bfloat16. No reconversion needed. Existing
JANG models work as-is.

## What the MLP Asymmetry Fix Still Does

The gate_proj=4-bit / down_proj=3-bit floors are still valuable insurance:
- Reduces the magnitude of quantization errors (119 instead of potentially worse)
- Makes the model more robust even in bfloat16
- Prevents quality degradation from extreme gate errors
- But they don't prevent float16 overflow — only bfloat16 does that
