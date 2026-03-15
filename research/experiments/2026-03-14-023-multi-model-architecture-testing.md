# Experiment 023: Multi-Model Architecture Testing

**Date**: 2026-03-14
**Author**: Eric Jang (eric@vmlx.net)
**Status**: IN PROGRESS

## Purpose

Test MXQ across diverse model architectures to validate:
1. Architecture detection correctness
2. Quantization pipeline handles different tensor layouts
3. Inference engine produces correct output for each architecture
4. Quantization quality at different bit widths per model size

## Models Under Test

| Model | Arch | Attention | RoPE | Biases | Head dim | Params | Status |
|-------|------|-----------|------|--------|----------|--------|--------|
| Qwen2.5-0.5B | qwen2 | GQA 14Q/2KV (7:1) | Non-trad θ=1M | Q/K/V (72) | 64 | 494M | TESTED — 8-bit correct |
| Qwen2.5-3B | qwen2 | GQA 16Q/2KV (8:1) | Non-trad θ=1M | Q/K/V (66) | 128 | 3B | TESTING |
| SmolLM2-1.7B | llama | MHA 32/32 | Traditional θ=130K | None (0) | 64 | 1.7B | DOWNLOADED |
| TinyLlama-1.1B | llama | GQA 32Q/4KV (8:1) | Traditional θ=10K | None (0) | 64 | 1.1B | DOWNLOADED |
| Phi-2 | phi | MHA 32/32 | Partial RoPE θ=10K | ALL (211) | 80 | 2.7B | DOWNLOADED — needs MLP adaptation |
| StableLM2-1.6B | stablelm | MHA 32/32 | Partial RoPE θ=10K | ALL (121) | 64 | 1.6B | DOWNLOADED — needs partial RoPE |

### Architecture Compatibility Matrix

| Feature | Qwen2 | Llama/SmolLM | TinyLlama | Phi-2 | StableLM2 |
|---------|-------|-------------|-----------|-------|-----------|
| Quantizable by MXQ | ✓ | ✓ | ✓ | ✗ needs work | ✗ needs work |
| RoPE mode | Non-trad | Traditional | Traditional | Partial | Partial |
| Attention biases | Q/K/V | None | None | All | All + norm |
| MLP structure | SwiGLU | SwiGLU | SwiGLU | Dense (fc1/fc2) | SwiGLU |
| Norm type | RMSNorm | RMSNorm | RMSNorm | LayerNorm | LayerNorm(+bias) |
| Tied embeddings | Yes | Yes | No | No | No |

### MXQ Adaptation Needed Per Architecture

**Ready now (standard transformer with SwiGLU):**
- Qwen2 family — tested, works at 8-bit
- Llama family (SmolLM2, TinyLlama) — should work with traditional RoPE

**Needs adaptation:**
- Phi-2: uses `fc1/fc2` instead of `gate_proj/up_proj/down_proj`, GELU activation
- StableLM2: uses partial rotary (only rotates first N dims of each head), LayerNorm with bias
- Both use LayerNorm instead of RMSNorm

## Results So Far

### Qwen2.5-0.5B (494M params)

| Bit Width | Top Token | Correct? | Top Logit | Ref Logit |
|-----------|-----------|----------|-----------|-----------|
| 2.76-bit | 'ém' | NO | 12.68 | 14.31 |
| 4-bit | 'è' | NO | 12.35 | 14.31 |
| 6-bit | '仫' | NO | 14.55 | 14.31 |
| 8-bit | '2' ✓ | YES | 14.39 | 14.31 |

**Conclusion**: 0.5B model too small for quantization below 8-bit with RTN.
The model lacks redundancy — every weight matters at this scale.

### Qwen2.5-3B (3B params)

Reference output: "The answer is 4. Is there anything else I can help you with?"

| Bit Width | Top Token | Output | Correct? |
|-----------|-----------|--------|----------|
| 4-bit | 'How' | "How can I-2-2+2+3?" | NO |
| 8-bit | Testing... | | |

**Key metrics for 3B at 4-bit:**
- Load time: 0.37s (mmap zero-copy)
- GPU memory: 1843.8 MB (1.44 GB weights + overhead)
- L0 norm: 14.63
- Quantization time: 564s (RTN, unoptimized)

## Mathematical Analysis: Why RTN Fails at Low Bits

### Quantization Error Per Block

For a block of 64 weights with variance σ² and bit width b:

```
MSE_block ≈ σ² × Δ² / 12
where Δ = range / (2^b - 1)
```

At 4-bit (16 levels): Δ = range/15 → MSE ∝ range²/2700
At 8-bit (256 levels): Δ = range/255 → MSE ∝ range²/780000

Ratio: 4-bit MSE is ~289x worse than 8-bit.

### Error Propagation Through N Layers

For additive errors that don't interact:
```
Total MSE ≈ N × per_layer_MSE
```

But transformer layers are NOT independent — each layer's output
feeds into the next layer's input. The error compounds multiplicatively:

```
effective_error ≈ (1 + ε)^N × original_error
where ε = per_layer_relative_error
```

For N=24 layers (0.5B) with 4-bit ε ≈ 0.05:
```
(1.05)^24 ≈ 3.2 — errors amplified 3.2x
```

For N=36 layers (3B) with 4-bit ε ≈ 0.02 (larger model = more redundancy):
```
(1.02)^36 ≈ 2.0 — errors amplified 2.0x
```

### Why GPTQ Would Help

RTN rounds each weight independently — no error compensation.
GPTQ compensates: when weight[i] is rounded up, subsequent weights
are adjusted down to minimize the total output error.

The Hessian-weighted error compensation:
```
δ_remaining = -(w[i] - Q(w[i])) × H⁻¹[i,:] / H⁻¹[i,i]
```

This reduces effective per-layer error by 2-5x at 4-bit,
which would bring 4-bit quality close to 6-8 bit RTN.

### Why Activation-Aware Calibration Would Help

Our current weight-only calibration uses:
```
importance = variance × range × magnitude
```

AWQ-style calibration uses:
```
importance = ||activations||₂ × |weight|
```

The activation term captures which weights are actually USED during
inference — some weights with high magnitude are rarely activated.
AWQ calibration typically improves bit allocation quality by 10-20%
at the same average bit width.

## Action Items

- [ ] Test 3B at 8-bit (should be correct)
- [ ] Test 3B at 6-bit (borderline)
- [ ] Test SmolLM2-1.7B (different architecture)
- [ ] Test TinyLlama (traditional RoPE + GQA)
- [ ] Implement GPTQ-style quantization
- [ ] Implement activation-aware calibration
- [ ] Test with GPTQ + AWQ at 4-bit on 3B
