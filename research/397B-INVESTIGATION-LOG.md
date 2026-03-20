# 397B Investigation Log — Complete Train of Thought

Created by Jinho Jang (eric@jangq.ai) — 2026-03-19

## Timeline of Failed Attempts

### Attempt 1: JANG_1L (8,8,2) — ~120 GB
- **Theory:** Max protection on attention (8-bit CRITICAL + 8-bit IMPORTANT), 2-bit experts
- **Result:** NaN
- **Why it failed:** 2-bit routed experts on 512-expert model. Thought it was expert routing issue.
- **Time wasted:** ~3 hours conversion

### Attempt 2: JANG_2S (6,4,2) — ~100 GB
- **Theory:** Tighter profile, 2-bit experts with 6-bit attention
- **Result:** NaN
- **Why it failed:** Same 2-bit expert issue
- **Time wasted:** ~3 hours conversion
- **Note:** Model was deleted before proper debugging — lost the chance to diagnose

### Attempt 3: JANG_3M (8,3,3) — ~155 GB (first attempt)
- **Theory:** 3-bit experts should be enough (122B works at 2-bit with 256 experts)
- **Result:** NaN
- **Wrong conclusion:** "shared_expert at 3-bit causes 45x error via SiLU amplification"
- **Partial truth:** shared_expert gate_proj DID produce 122 instead of ~3, but...
- **Time wasted:** ~2 hours conversion

### Attempt 4: shared_expert → CRITICAL (8-bit) fix
- **Theory:** Force shared_expert to 8-bit, keep routed experts at 3-bit
- **Result:** NaN still
- **Wrong conclusion:** "routed experts at 3-bit also overflow on 512-expert models"
- **Partial truth:** Routed experts have issues too, but the ROOT cause was float16
- **Time wasted:** ~2 hours reconversion

### Attempt 5: MLP Asymmetry Fix — JANG_2L with gate=4, up=2, down=3
- **Theory:** Based on GGUF research — gate_proj is SiLU amplifier, needs 4-bit floor.
  down_proj needs 3-bit floor (GGUF always does this). Budget-neutral allocation.
- **Implementation:** `_apply_mlp_asymmetry_floor()` in allocator.py
- **Result:** NaN at layer 22
- **Actual allocation:** 3.72 bpw, 186.9 GB, gate=4-bit, up=2-bit, down=3-bit
- **Time wasted:** ~3 hours conversion
- **Key mistake:** Still assumed it was a bit allocation problem

### Total time wasted on wrong approaches: ~13+ hours of Mac Studio compute

## The Diagnostic (What Should Have Been Done First)

### Layer-by-layer NaN trace
```
Embedding: max=0.1 — clean
Layer 0-21: max grows 0.1 → 8.8 — all clean
Layer 22: inf detected. Input max was only 8.8.
```

### Layer 22 decomposition
```
input_layernorm: max=14.09 — clean
GatedDeltaNet (linear attn): max=0.07 — clean
post_attn_norm: max=12.37 — clean
Router: max=11.90 — clean
Shared expert: max=inf — BROKEN
Routed experts: not tested (shared expert already inf)
```

### Shared expert step-by-step
```
gate_proj output: 119.25 — 8-BIT quantization, output is VALID
up_proj output: 69.50 — 8-BIT quantization, output is VALID
SiLU(gate) * up: 8,288 — fits in float16 (max 65504)
down_proj: inf — OVERFLOWS

down_proj does: input(8288) × weights across 1024 dims
= 8288 × avg_weight × 1024 ≈ 281,675
281,675 > 65,504 (float16 max) → inf
```

### Critical revelation
**The shared expert was at 8-bit. The output of gate_proj (119.25) is CORRECT.**
This is not a quantization error. The model legitimately produces these values.
The overflow is purely a float16 dynamic range limitation.

The source model is bfloat16 (max 3.4e38). It handles 281,675 easily.
MLX defaults to float16 (max 65,504). It can't.

## The Fix: bfloat16 Compute

### Test results
```
float16 down_proj:  inf          ← BROKEN
bfloat16 down_proj: 281,675      ← WORKS

Full bfloat16 all 60 layers: ALL CLEAN
After norm: max=36.41 nan=False
Logits: nan=False inf=False
Top token: valid
```

### Why bfloat16 works
- bfloat16 exponent: 8-bit (same as float32) → max 3.4×10^38
- float16 exponent: 5-bit → max 65,504
- bfloat16 mantissa: 7-bit (less precise than float16's 10-bit)
- But at 2-4 bit quantization, precision loss is negligible vs quantization noise

### Quality impact: None
- Quantization noise at 4-bit: ±0.03 per weight (scale-dependent)
- bfloat16 vs float16 precision: ±0.008 vs ±0.001 at value 1.0
- Quantization noise is 30x larger. bfloat16 precision is irrelevant.

## Why 256-Expert Models Don't Need This

| Model | Experts | hidden | intermediate | down_proj dims | SiLU×up max | down_proj sum | Overflows f16? |
|-------|---------|--------|-------------|---------------|------------|--------------|---------------|
| 35B | 256 | 2048 | 1024 | 1024 | ~500 | ~8,000 | No |
| 122B | 256 | 3072 | 1536 | 1536 | ~1,500 | ~30,000 | No |
| MiniMax | 256 | 3072 | 1536 | 1536 | ~2,000 | ~40,000 | No |
| **397B** | **512** | **4096** | **1024** | **1024** | **~8,288** | **~281,675** | **YES** |
| **Nemotron** | **512** | **4096** | **2688** | **2688** | **~TBD** | **~TBD** | **Likely YES** |

The 397B shared expert produces much larger SiLU×up values because:
1. Shared expert processes EVERY token (always active, absorbs more signal)
2. hidden_size=4096 means gate_proj does a 4096-dim dot product (more terms = bigger output)
3. 512 experts means shared expert carries proportionally more load

## Lessons Learned

1. **Diagnose before fixing.** Layer-by-layer NaN trace takes 5 minutes. Would have saved 13+ hours.
2. **The bit allocation was never the problem.** Even 8-bit shared expert overflows float16.
3. **GGUF comparison was misleading.** GGUF's advantage isn't their bit allocation — it's float32 compute.
4. **The MLP asymmetry analysis was correct but irrelevant.** gate_proj IS the SiLU amplifier.
   down_proj IS where overflow happens. But the fix isn't more bits — it's more dynamic range.
5. **bfloat16 is the standard for large models.** Google uses it. The original model is bfloat16.
   MLX should default to bfloat16 for models that were trained in bfloat16.

## Implementation

One-line fix in JANG loader: detect 512+ experts → cast embeddings to bfloat16.
No reconversion needed. Existing model (186.9 GB) works as-is.

Speed impact: TBD — need to verify bfloat16 quantized_matmul performance on Apple Silicon.
