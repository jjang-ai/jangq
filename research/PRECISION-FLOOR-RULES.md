# JANG Precision Floor Rules

Created by Jinho Jang (eric@jangq.ai) — 2026-03-18

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
| Routed experts (≤256) | 2 | COMPRESS | Expert redundancy absorbs errors |
| Routed experts (512+) | 3 | COMPRESS | Less redundancy per expert |
| Dense MLP | 3 | COMPRESS | No redundancy, 2-bit = quality cliff |

## Profile Safety Matrix

| Profile | Dense | MoE ≤256 | MoE 512+ |
|---------|-------|----------|----------|
| JANG_1L (8,8,2) | — | ✓ | ✗ (2-bit experts NaN) |
| JANG_2L (8,6,2) | — | ✓ | ✗ |
| JANG_2S (6,4,2) | — | ✓ | ✗ |
| JANG_3M (8,3,3) | — | ✓* | ✓ (with shared_expert fix) |
| JANG_3L (8,4,3) | — | ✓ | ✓ |
| JANG_4S (6,4,4) | ✓ | ✓ | ✓ |
| JANG_4K (budget) | ✓ | ✓ | ✓ |

*3M needs shared_expert=CRITICAL (fixed 2026-03-18)

## Implementation

The allocator (`allocate.py`) classifies shared_expert as CRITICAL tier.
The converter (`convert.py`) prints precision warnings when dangerous combinations
are detected (shared_expert <4-bit with hidden_size≥4096).

## Verification

After EVERY conversion, run NaN check before testing coherence:
```python
tokens = mx.array([[tokenizer.encode("Hello")[0]]])
logits = model(tokens)
assert not mx.any(mx.isnan(logits)).item(), "NaN detected — precision floor violated"
```
