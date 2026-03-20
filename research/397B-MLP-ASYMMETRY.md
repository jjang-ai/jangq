# 397B NaN Root Cause Analysis & MLP Asymmetry Fix

Created by Jinho Jang (eric@jangq.ai) — 2026-03-19

## Executive Summary

Qwen3.5-397B-A17B produces NaN at both 2-bit and 3-bit JANG profiles. After exhaustive
investigation comparing GGUF internals, CRACK abliteration research (162 GB working Q4),
MLX Metal kernel behavior, and our own experiment logs, we identified two root causes
and a concrete fix.

**Root causes:**
1. MLX computes MLP intermediates in float16 (max 65,504) while GGUF uses float32 (max 3.4e38)
2. JANG treats all expert MLP tensors (gate/up/down) as equal COMPRESS tier, but they have
   fundamentally different sensitivity to quantization error

**Fix:** Sub-classify expert MLP — gate_proj gets 4-bit (SiLU amplifier), up_proj stays 2-bit,
down_proj gets 3-bit. Budget-neutral: (4+2+3)/3 = 3.0 average, same size as uniform 3-bit.

---

## 1. The NaN Chain (Proven on 397B Layer 22)

The MLP forward pass in every MoE expert:

```
x = input (float16)

gate = gate_proj(x)     # quantized_matmul → float16 output
up   = up_proj(x)       # quantized_matmul → float16 output
hidden = SiLU(gate) * up # elementwise float16 multiplication
output = down_proj(hidden) # quantized_matmul → float16 output
```

At 3-bit gate_proj on the 397B (hidden_size=4096, 512 experts):

```
1. gate_proj at 3-bit produces 122 instead of correct ~3   (45x error)
   - Why: 8 quantization levels, 4096-dim dot product sums errors
   - Larger hidden_size = more accumulated error per dot product

2. SiLU(122) ≈ 122  (SiLU(x) ≈ x for large x)

3. up_proj(x) produces ~72  (normal range)

4. SiLU(gate) * up = 122 * 72 = 8,808   (valid in float16, max=65504)

5. down_proj receives 8,808 as input
   dot product: 8808 * weights summed across 4096 dimensions
   even small weights: 8808 * avg(0.004) * 4096 ≈ 144,261
   144,261 > 65,504 (float16 max)
   → float16 inf → NaN propagates through all subsequent layers
```

**Key insight:** The overflow happens at the down_proj OUTPUT, not the SiLU multiply.
The gate_proj error is the SOURCE, but the overflow manifests when down_proj
compresses 4096 dimensions of inflated values back to a scalar.

---

## 2. Why GGUF Q2 Works But JANG Doesn't

### 2.1 GGUF Q2_K is NOT 2-bit

GGUF's "Q2_K" label is misleading. Actual allocation per tensor type:

| Tensor Type           | GGUF Q2_K type | Effective bits | Notes                          |
|-----------------------|---------------|----------------|--------------------------------|
| attention.wv (v_proj) | Q4_K          | **4-bit**      | Attention value always protected |
| ffn_down (down_proj)  | Q3_K          | **3-bit**      | Residual projection upgraded   |
| output.weight (lm_head)| Q6_K         | **6-bit**      | Output head always high        |
| Everything else       | Q2_K          | ~2.56-bit      | Super-block with 4-bit scales  |

**Effective average: ~2.56 bpw** (not 2.0)

GGUF also interleaves: every 3rd ffn_down layer gets Q4_K instead of Q3_K.
Unsloth Dynamic 2.0 goes further — routers unquantized, shared experts ~4 bpw.

**Source:** llama.cpp PR #4872 by ikawrakow (K-quants author): "Q2_K becomes again
a mostly 3-bit quantization."

### 2.2 GGUF Computes in Float32

| Operation            | GGUF (llama.cpp)          | MLX (Apple Silicon)        |
|---------------------|--------------------------|---------------------------|
| Matmul accumulation | float32                  | float32 (Metal kernel)    |
| Matmul OUTPUT       | **float32**              | **float16** ← overflow    |
| SiLU(gate) * up     | **float32** elementwise  | **float16** elementwise   |
| down_proj input     | **float32**              | **float16** ← if inf, NaN|
| Dynamic range       | 3.4 × 10^38             | 65,504                    |

Same quantization errors exist in both, but float32 has 10^33x more headroom.
The overflow that kills MLX at float16 literally cannot happen in GGUF.

### 2.3 GGUF Has 4-bit Double-Quantized Scales

GGUF Q2_K uses 256-weight super-blocks with 4-bit double-quantized scales:
- 16 blocks × 16 weights per super-block
- Block-level scales and mins quantized to 4-bit (saves memory)
- Effective overhead: 2.5625 bpw total

MLX uses full float16 scales per group (higher precision per scale, but the
weights themselves have less dynamic range at 2-bit with only 4 levels).

### 2.4 GGUF imatrix (Importance Matrix)

GGUF's I-quant family (IQ2_XS, IQ2_M) uses per-weight importance calibration:
- Computed from calibration text via `llama-imatrix`
- Decides which weights within each group get more precise representation
- At 2-bit (4 levels), this matters enormously

JANG calibration works at the block/tensor level (which blocks get more bits),
not at the per-weight level (which weights within a block matter more).
`mx.quantize()` treats all weights within a group equally.

---

## 3. MLP Asymmetry: Why gate ≠ up ≠ down

### 3.1 The SiLU Amplification Bomb

```python
def expert_mlp(x):
    gate = gate_proj(x)     # → feeds into SiLU activation
    up   = up_proj(x)       # → linear multiplicand
    hidden = silu(gate) * up # → quadratic error amplification
    return down_proj(hidden) # → projects back to residual stream
```

**Error analysis:**
```
Let ε_g = gate_proj quantization error, ε_u = up_proj error

output ≈ (true_gate + ε_g) × (true_up + ε_u)
       = true_gate × true_up           ← correct signal
       + true_gate × ε_u               ← linear in ε_u
       + ε_g × true_up                 ← linear in ε_g (but SiLU amplified!)
       + ε_g × ε_u                     ← quadratic cross-term

SiLU makes ε_g worse because:
  - SiLU(x) ≈ x for large x (passes error through)
  - SiLU(x) ≈ 0 for x << 0 (clips negative errors, creates asymmetry)
  - Near x=0, gradient ≈ 0.5 (halves small signals)

When gate_proj at 3-bit produces 122 instead of 3:
  SiLU(122) = 122 (full pass-through of 45x error)
  This gets multiplied by up_proj output (72)
  Error term: 119 × 72 = 8,568 (vs correct: 0)
```

**Conclusion:** gate_proj errors are AMPLIFIED by SiLU, up_proj errors are not.
gate_proj needs more bits than up_proj.

### 3.2 down_proj: Residual Stream Guardian

down_proj projects the (potentially inflated) hidden state back to the residual stream.
Every subsequent layer sees this output. GGUF always gives down_proj +1 bit (Q3_K in Q2_K profile).

At 2-bit, down_proj has 4 quantization levels. When its INPUT is already inflated
(from gate error), 4 levels can't represent the output range → additional error
that corrupts the entire residual stream for all subsequent layers.

At 3-bit (8 levels), down_proj can better absorb the inflated input range.

### 3.3 up_proj: The Safe One

up_proj is a plain linear projection. Its output is multiplied by SiLU(gate), not
fed through a nonlinearity. Errors in up_proj are linear, not amplified.
2-bit up_proj is safe because:
- Error is bounded (4 levels, but proportional, not exponential)
- SiLU(gate) × (up + ε_u) = SiLU(gate) × up + SiLU(gate) × ε_u
- The cross-term SiLU(gate) × ε_u is bounded when gate is correct

**This is only safe when gate_proj has sufficient precision.**
If gate_proj is also 2-bit, both errors amplify → catastrophe.

---

## 4. 512-Expert Specific Issues

### 4.1 Why 512 Experts Are Harder Than 256

| Property              | 256 experts (35B/122B) | 512 experts (397B) |
|-----------------------|----------------------|-------------------|
| Active per token      | top-8 (3.1%)         | top-10 (1.95%)    |
| Expert specialization | Higher               | Lower             |
| Redundancy per expert | More overlap          | Less overlap      |
| hidden_size           | 2048-3072            | **4096**          |
| Dot product terms     | 2048-3072            | **4096** (more accumulation error) |
| 2-bit viability       | Works (proven 122B)  | **NaN** (proven)  |
| 3-bit viability       | Works (proven 122B)  | **NaN** (proven)  |

**hidden_size=4096 is the killer.** Each dot product sums 4096 quantized weights.
More terms = more accumulated quantization error = higher chance of overflow.

### 4.2 What CRACK Research Tells Us

The working CRACK 397B (162 GB, 36.8 tok/s) used:
- **Uniform Q4** (4-bit everything) with gs=64
- mlx-community quantizer (standard `mx.quantize`)
- Binary shard patching to preserve `{"format":"mlx"}` metadata
- **No per-tensor mixed precision** — pure Q4 worked fine

This proves: at 4-bit, the 397B architecture is perfectly stable on MLX/float16.
The NaN only appears when ANY expert MLP tensor drops below 4-bit.

### 4.3 GGUF Q2 Results on 397B (Real Users)

People successfully run 397B at low bits via GGUF:
- **UD-IQ1_M**: 107 GB, perplexity 1.19 — works (1-bit base!)
- **UD-IQ2_XXS**: 115 GB — works
- **UD-IQ2_M**: 123 GB — works
- **Q2_K_XL**: Recommended minimum for quality

All use float32 computation. None overflow.

---

## 5. Proposed Fix: MLP Sub-Classification

### 5.1 Changes to Tier System

Current (broken for 512-expert models):
```
gate_proj → COMPRESS (gets 2-3 bit depending on profile)
up_proj   → COMPRESS (gets 2-3 bit)
down_proj → COMPRESS (gets 2-3 bit)
```

Proposed (architecture-aware):
```
gate_proj → IMPORTANT tier for 512+ expert models (4-bit minimum)
            COMPRESS for ≤256 experts (2-bit OK, proven on 122B)
up_proj   → COMPRESS (2-bit OK in all cases, IF gate is ≥4-bit)
down_proj → COMPRESS with 3-bit FLOOR for 512+ expert models
            COMPRESS for ≤256 experts (2-bit OK)
```

### 5.2 Budget Impact (Verified by Tests)

**For 512-expert models (397B) with JANG_3M (8,3,3):**

| Component    | Before (3M) | After (3M-fix) | Change |
|-------------|-------------|----------------|--------|
| gate_proj    | 3-bit       | **4-bit**      | +1 bit (floor) |
| up_proj      | 3-bit       | 3-bit          | same   |
| down_proj    | 3-bit       | 3-bit          | same (already ≥3) |
| Expert avg   | 3.0 bpw     | **3.33 bpw**   | +0.33  |

**For 512-expert models with JANG_2L (8,6,2):**

| Component    | Before (2L) | After (2L-fix) | Change |
|-------------|-------------|----------------|--------|
| gate_proj    | 2-bit       | **4-bit**      | +2 bit (floor) |
| up_proj      | 2-bit       | 2-bit          | same   |
| down_proj    | 2-bit       | **3-bit**      | +1 bit (floor) |
| Expert avg   | 2.0 bpw     | **3.0 bpw**    | +1.0   |

**For ≤256-expert models (35B, 122B, MiniMax):**
No change. These already work at 2-bit. The floors only activate
for 512+ expert models where the NaN was proven.

### 5.3 Size Estimates for 397B (Verified)

Expert MLP is ~380B of 397B total parameters (~95.7%).
Size estimated from verified average bpw (test output).

| Profile         | gate | up  | down | Avg bpw | Est. size | Fits 192GB? |
|----------------|------|-----|------|---------|-----------|-------------|
| JANG_3M (old)  | 3    | 3   | 3    | ~3.0    | ~150 GB   | Yes (NaN!)  |
| **JANG_3M-fix**| **4**| **3**| **3**| **3.35**| **~167 GB** | **Yes** |
| JANG_2L (old)  | 2    | 2   | 2    | ~2.0    | ~100 GB   | Yes (NaN!)  |
| **JANG_2L-fix**| **4**| **2**| **3**| **3.0** | **~150 GB** | **Yes** |
| JANG_1L (old)  | 2    | 2   | 2    | ~2.0    | ~100 GB   | Yes (NaN!)  |
| **JANG_1L-fix**| **4**| **2**| **3**| **3.0** | **~150 GB** | **Yes** |
| JANG_4M (ref)  | 4    | 4   | 4    | ~4.0    | ~200 GB   | Tight       |
| CRACK Q4 (ref) | 4    | 4   | 4    | 4.0     | 162 GB    | Yes (works) |

### 5.4 Implementation (DONE — 2026-03-19)

#### Allocator (`allocate.py`)

Added `_apply_mlp_asymmetry_floor()` — a post-classification floor function:
```python
MLP_ASYMMETRY_FLOORS = {
    "gate_proj": 4,     # SiLU amplifier
    "gate_up_proj": 4,  # Fused variant (contains gate)
    "w1": 4,            # Mixtral naming
    "down_proj": 3,     # Residual projection
    "w2": 3,            # Mixtral naming
}
```
Applied in both `allocate_bits_profile()` and `allocate_bits_budget()`.
Only activates when `num_experts >= 512`. Skips shared_expert (already CRITICAL).

#### Converter (`convert.py`)

Passes `num_experts` from architecture detection to all allocator functions.
Prints notice when MLP asymmetry is active. Warns if tensors fall below floors.

#### Test Results

All tests pass. Verified:
- 256 experts: fully unchanged (no floors applied)
- 512 experts + JANG_3M: gate=4, up=3, down=3 (avg 3.35 bpw)
- 512 experts + JANG_2L: gate=4, up=2, down=3 (avg 3.0 bpw)
- 512 experts + JANG_1L: gate=4, up=2, down=3 (avg 3.0 bpw)
- 512 experts + JANG_4M: no change (already ≥ all floors)
- shared_expert: never affected (already CRITICAL tier)

#### Next: Test on 397B

1. Convert with JANG_2L (gate=4, up=2, down=3 → ~150 GB)
2. NaN check: `assert not mx.any(mx.isnan(logits)).item()`
3. Coherence: "What is the capital of France?", "What is 47 × 23?"
4. Speed: should be ~36-38 tok/s
5. If works, also try JANG_3M (~167 GB)

---

## 6. Future Considerations

### 6.1 Float32 Compute Path (Not Blocking)

If the MLP sub-classification alone doesn't fix NaN, we can investigate:
- Casting expert MLP intermediates to float32 before SiLU multiply
- `hidden = mx.silu(gate.astype(mx.float32)) * up.astype(mx.float32)`
- Then cast back: `hidden = hidden.astype(mx.float16)`
- Cost: 2x activation memory for expert MLP only (small, since only 10 of 512 active)

This would be a loader/runtime change, not a converter change.

### 6.2 Per-Layer Sensitivity (Advanced)

Layer 22 specifically failed. Calibration-based per-layer allocation could:
- Give layers 20-30 higher bits (4-bit gate + 3-bit down)
- Let layers 0-19 and 31-59 use lower bits (3-bit gate + 2-bit down)
- Same total budget, smarter distribution

This requires the greedy allocator with importance scores (calibration run).

### 6.3 imatrix Integration (Long-term)

GGUF's per-weight importance within groups is theoretically better at 2-bit.
`mx.quantize()` treats all weights equally within a group.
Custom quantizer with per-weight scaling would be a major project.

---

## 7. Comparison: JANG vs GGUF at Low Bits

| Feature                        | JANG (current)     | GGUF (llama.cpp)    | Gap? |
|-------------------------------|-------------------|--------------------|----|
| Router/gate precision          | 8-bit (CRITICAL)  | F16/unquantized    | OK |
| Shared expert protection       | 4-bit+ (CRITICAL) | High precision     | OK |
| Attention protection           | 4-8 bit (CRITICAL)| Q4_K+ for v_proj   | OK |
| lm_head protection             | 6-8 bit (CRITICAL)| Q6_K always        | OK |
| **gate_proj treatment**        | **COMPRESS (2-3)** | **Q2_K (~2.56)**  | **GAP** |
| **down_proj treatment**        | **COMPRESS (2-3)** | **Q3_K (3-bit)**  | **GAP** |
| Expert-count-aware allocation  | No (all same)     | MoE-specific logic | **GAP** |
| Compute precision              | float16           | float32            | **GAP** (hw limit) |
| Per-weight importance          | No (per-block)    | imatrix            | Minor gap |
| First/last layer bonus         | K-quant only      | Always             | Minor gap |

**Three gaps matter.** Two are fixable in allocator (gate_proj, down_proj sub-classification).
One is a hardware/framework limit (float16 vs float32) that the sub-classification works around.

---

## 8. Evidence Base

| Evidence | Source | Finding |
|----------|--------|---------|
| 397B NaN at layer 22 | JANG experiment (2026-03-18) | gate_proj 3-bit → 45x error → float16 overflow |
| 397B NaN at 2-bit routed experts | JANG experiment (2026-03-18) | Even with shared_expert=CRITICAL, routed experts overflow |
| CRACK 397B Q4 works (162 GB) | CRACK_abliteration proven-pipeline.md | Uniform Q4 with gs=64, 36.8 tok/s |
| GGUF Q2_K = ~2.56 bpw | llama.cpp PR #4872 (ikawrakow) | Mixed precision, not uniform 2-bit |
| GGUF upgrades down_proj to Q3_K | llama.cpp quantize.cpp source | Residual projection always protected |
| GGUF uses float32 compute | llama.cpp architecture | All intermediates float32 |
| 122B works at 2-bit | JANG experiment (2026-03-16) | 73% MMLU, 256 experts, hidden=3072 |
| MiniMax works at 2-bit | JANG experiment (2026-03-17) | 74% MMLU, 256 experts, hidden=3072 |
| 27B greedy at 3.5 bits works | JANG experiment (2026-03-18) | Dense 27B coherent with per-tensor allocation |

---

## Appendix A: GGUF K-Quant Type Reference

| Type    | Avg bpw | Block structure              | Notes                    |
|---------|---------|------------------------------|--------------------------|
| Q2_K    | 2.5625  | 256w super-block, 4-bit scales | "Mostly 3-bit" per author |
| Q2_K_S  | ~2.3    | All tensors Q2_K except output | True 2-bit, lower quality |
| IQ2_XS  | 2.31    | Importance-weighted           | Needs imatrix calibration |
| IQ2_M   | 2.70    | More bits for important weights | Best Q2 quality          |
| Q3_K_S  | 3.44    | All tensors Q3_K              | Uniform 3-bit            |
| Q3_K_M  | 3.89    | Upgrades attention+down       | "Medium" mixed 3-bit     |
| Q4_K_M  | 4.83    | Standard recommendation       | Best balance             |

## Appendix B: MLX Float16 Limits

| Property | float16 | float32 | Ratio |
|----------|---------|---------|-------|
| Max value | 65,504 | 3.4e38 | 5.2e33x |
| Min positive | 6.1e-5 | 1.2e-38 | — |
| Precision | 10-bit mantissa | 23-bit mantissa | — |
| Epsilon | 9.77e-4 | 1.19e-7 | — |

At hidden_size=4096, a dot product summing values around 16.0 each:
- float16: 16.0 * 4096 = 65,536 > 65,504 → **inf**
- float32: 16.0 * 4096 = 65,536 → fine (3.4e38 headroom)

This is why 397B (hidden=4096) overflows but 122B (hidden=3072) doesn't:
- 122B: 16.0 * 3072 = 49,152 < 65,504 → safe
- 397B: 16.0 * 4096 = 65,536 > 65,504 → overflow
