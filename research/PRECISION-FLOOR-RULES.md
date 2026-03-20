# JANG Precision Floor Rules

Created by Jinho Jang (eric@jangq.ai) — 2026-03-18
Updated 2026-03-19: Added MLP asymmetry rules for 512+ expert models

## The Problem

Quantization errors in always-active components (shared expert, attention) compound
through every layer. The SiLU gate multiplication in MLP (`SiLU(gate) * up`) amplifies
errors quadratically — small per-weight errors become massive output errors.

**Proven on Qwen3.5-397B-A17B:** 3-bit shared expert → gate_proj output 122 instead of
~3 (45x error) → SiLU*up = 8808 → down_proj overflow → float16 inf → NaN propagates.

## Root Cause Analysis

### Why SiLU multiplication is an amplification bomb:

```
gate_proj(x) → SiLU activation → multiply with up_proj(x)

If gate has ε_g error and up has ε_u error:
  output ≈ (true_gate + ε_g) × (true_up + ε_u)
         = true_gate × true_up + true_gate × ε_u + ε_g × true_up + ε_g × ε_u

The cross terms amplify: ε_g × true_up can be LARGE if true_up is large
And ε_g × ε_u compounds quadratically
```

### Why shared expert is more sensitive than routed experts:

- **Shared expert**: processes EVERY token → errors compound across ALL layers
- **Routed experts**: process only tokens routed to them → errors diluted by top-K averaging
- Same MLP architecture, vastly different error propagation

### Why larger hidden_size makes it worse:

- Dot product sums more terms: hidden_size=4096 sums 4096 quantized weights
- More terms = more accumulated quantization error
- At 3-bit, each error is larger than at 4-bit
- 397B (4096 hidden) overflows, 122B (3072 hidden) doesn't

### Why 2-bit routed experts fail on 512 experts:

- With 512 experts and top-10 routing, each expert is less specialized
- 2-bit quantization loses the fine-grained weight differences between experts
- The model can't distinguish expert specializations → routing becomes meaningless

## MLP Asymmetry (512+ Expert Models)

**Discovery (2026-03-19):** Expert MLP tensors are NOT equal in sensitivity.
GGUF figured this out — JANG now matches their approach.

### gate_proj = SiLU Amplifier (4-bit minimum)

gate_proj feeds into SiLU activation, which passes errors through for large values.
The error gets multiplied by up_proj output → quadratic amplification.

At 3-bit on hidden_size=4096: gate_proj produces 122 instead of ~3 (45x error).
SiLU(122) × up(72) = 8,808. down_proj dot product across 4096 dims:
8808 × weights × 4096 → exceeds float16 max (65,504) → inf → NaN.

At 4-bit: error is ~3x instead of 45x. SiLU(3×3) × 72 ≈ 648. Safe.

### up_proj = Linear Multiplicand (no floor)

up_proj is a plain linear projection. Its errors are bounded and linear.
2-bit is safe **only when gate_proj has sufficient precision** (4+ bit).

### down_proj = Residual Guardian (3-bit minimum)

down_proj projects back to the residual stream. Every subsequent layer sees
this output. GGUF always gives down_proj +1 bit (Q3_K even in Q2_K profile).
At 2-bit, the 4 quantization levels can't represent the output range when
input is inflated from gate errors.

### Budget Impact

For 2-bit profiles (JANG_2L, JANG_1L):
- Before: gate=2, up=2, down=2 → avg 2.0 (NaN on 512+)
- After:  gate=4, up=2, down=3 → avg 3.0 (safe)

For 3-bit profiles (JANG_3M):
- Before: gate=3, up=3, down=3 → avg 3.0 (NaN on 512+)
- After:  gate=4, up=3, down=3 → avg 3.33 (safe)

### GGUF Comparison

| Feature | JANG (after fix) | GGUF Q2_K |
|---------|-----------------|-----------|
| gate_proj floor | 4-bit | Q2_K (~2.56) |
| down_proj floor | 3-bit | Q3_K (3-bit) |
| up_proj | 2-bit OK | Q2_K (~2.56) |
| Router/gate | 8-bit | Unquantized |
| Compute dtype | float16 (MLX) | float32 |

GGUF can get away with lower gate_proj because float32 computation
(max 3.4e38) never overflows. MLX float16 (max 65,504) needs the 4-bit floor.

## Precision Floor Rules

| Component | Min Bits | Tier | Reasoning |
|-----------|---------|------|-----------|
| MoE router/gate | 8 | CRITICAL | Controls expert routing |
| Shared expert MLP | 4 | CRITICAL | Always active, SiLU amplifies errors |
| Shared expert gate | 8 | CRITICAL | Controls shared expert contribution |
| Attention Q/K/V/O | 4 | CRITICAL | Controls coherence |
| lm_head | 4 | CRITICAL | Output head, affects every token |
| Mamba A_log, D | 4 | CRITICAL | SSM state matrices |
| Embeddings | 4 | IMPORTANT | First layer, errors propagate |
| Linear attention (GDN) | 3 | IMPORTANT | Always active but resilient |
| **Expert gate_proj (512+)** | **4** | **FLOOR** | **SiLU amplifier, float16 overflow** |
| **Expert down_proj (512+)** | **3** | **FLOOR** | **Residual stream projection** |
| Expert up_proj | 2 | COMPRESS | Linear multiplicand, safe |
| Routed experts (≤256) | 2 | COMPRESS | Expert redundancy absorbs errors |
| Dense MLP | 3 | COMPRESS | No redundancy, 2-bit = quality cliff |

## Profile Safety Matrix

| Profile | Dense | MoE ≤256 | MoE 512+ (with MLP fix) |
|---------|-------|----------|------------------------|
| JANG_1L (8,8,2) | Bad (2-bit MLP) | ✓ | ✓ (gate=4, up=2, down=3) |
| JANG_2L (8,6,2) | Bad (2-bit MLP) | ✓ | ✓ (gate=4, up=2, down=3) |
| JANG_2S (6,4,2) | Bad (2-bit MLP) | ✓ | ✓ (gate=4, up=2, down=3) |
| JANG_3M (8,3,3) | Bad (3-bit MLP) | ✓ | ✓ (gate=4, up=3, down=3) |
| JANG_3L (8,4,3) | Bad (3-bit MLP) | ✓ | ✓ (gate=4, up=3, down=3) |
| **JANG_4S (6,4,4)** | **✓ (84.5% on 27B)** | ✓ | ✓ (no floor needed) |
| JANG_4K (budget) | OK (small models lose) | ✓ | ✓ (no floor needed) |
| JANG_4M (8,4,4) | ✓ | ✓ | ✓ (no floor needed) |

Dense model note: JANG_4S is the best dense profile. It matches MLX 4-bit quality
(84.5% on 27B) with only attention bumped to 6-bit. JANG_4K (budget-neutral) can
downgrade some tensors below 4-bit, which hurts small dense models (4B/9B lose ~2-5%).

## Implementation

The allocator (`allocate.py`) enforces MLP asymmetry floors via `_apply_mlp_asymmetry_floor()`.
This is a post-classification floor applied in both `allocate_bits_profile()` and
`allocate_bits_budget()`. It only activates when `num_experts >= 512`.

The converter (`convert.py`) passes `num_experts` from architecture detection to all
allocator functions. It prints a notice when MLP asymmetry is active and warns if
any tensor falls below its floor.

## Affected Models

| Model | Experts | hidden_size | MLP fix needed? |
|-------|---------|-------------|-----------------|
| Qwen3.5-4B | 0 | 2560 | No (dense) |
| Qwen3.5-9B | 0 | 4096 | No (dense) |
| Qwen3.5-35B-A3B | 256 | 2048 | No (256 experts) |
| Qwen3.5-122B-A10B | 256 | 3072 | No (256 experts) |
| MiniMax-M2.5 | 256 | 3072 | No (256 experts) |
| **Qwen3.5-397B-A17B** | **512** | **4096** | **YES** |
| **Nemotron-3-Super-120B** | **512** | **4096** | **YES** |

## Verification

After EVERY conversion, run NaN check before testing coherence:
```python
tokens = mx.array([[tokenizer.encode("Hello")[0]]])
logits = model(tokens)
assert not mx.any(mx.isnan(logits)).item(), "NaN detected — precision floor violated"
```
