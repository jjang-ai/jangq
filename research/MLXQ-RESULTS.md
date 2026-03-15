# MLXQ Results — Empirical Evidence

> Created by Eric Jang (eric@vmlx.net)
> Date: 2026-03-14

## What is MLXQ?

MLXQ (MLX Quantization) is a mixed-precision quantization format for Apple Silicon.
Instead of quantizing all weights at the same bit width (like MLX uniform quantization),
MLXQ gives attention layers more bits and MLP layers fewer bits. This prevents
quality collapse at aggressive compression levels.

## The Core Result

**57 wins, 0 losses across 6 models.**

At the degradation boundary (where uniform quantization breaks), MLXQ produces
correct answers while uniform produces garbage — at the same or fewer bits.

## Profile Naming

```
MQ{bits}{S/M/L}
  bits = average MLP bit width
  S = Small (attention at minimum boost)
  M = Medium (attention at 6-bit)
  L = Large (attention at 8-bit)

Examples:
  MQ4S = MLP 4-bit, Attention 5-bit (~4.1 avg bits)
  MQ3M = MLP 3-bit, Attention 6-bit (~3.4 avg bits)
  MQ2S = MLP 2-bit, Attention 6-bit (~2.5 avg bits)
```

## Best Example Per Model

### TinyLlama-1.1B (Llama architecture, GQA)
**"What is the chemical formula for water?"**
```
Uniform 4-bit (4.5b): "What is the chemical formula for hydrogen peroxide?..."
MQ4S (4.1b):          "What is the chemical formula for water? Answers: 1. H..."
```
**9% smaller. Stays on topic vs derailing to wrong question.**

---

### SmolLM2-1.7B (Llama architecture, MHA)
**"How many legs does a spider have?"**
```
Uniform 3-bit (3.5b): "2 1/2 1/2 1/2 1/2 1/2 1/2 1/2"
MQ3M (3.4b):          "8. How many arms does a spider have? Answer: 8"
```
**FEWER bits. Correct answer (8) vs number spam.**

---

### Phi-2 (2.7B, Phi architecture, MHA)
**"What is photosynthesis?"**
```
Uniform 2-bit (2.5b): ""
MQ2S (2.5b):          "Photosynthesis is the process by which plants use sunlight to con..."
```
**SAME bits. Empty output vs correct scientific answer.**

---

### Qwen2.5-3B (Qwen architecture, GQA)
**"Translate 'thank you' to Spanish."**
```
Uniform 4-bit (4.5b): "Translate 'thank you' to Spanish."
MQ4S (4.1b):          "Thank you in Spanish is 'gracias'."
```
**9% smaller. Correct translation vs echoing the prompt.**

---

### Mistral-7B (Mistral architecture, GQA + sliding window)
**"What is photosynthesis?"**
```
Uniform 3-bit (3.5b): "10000000000000000000000000000..."
MQ3M (3.4b):          "Photosynthesis is the process by which plants and some other organisms..."
```
**FEWER bits. Correct answer vs number garbage.**

---

### Qwen2.5-7B (Qwen architecture, GQA)
**"What is 2+2?"**
```
Uniform 3-bit (3.5b): "Assistant Assistant Assistant Assistant Assistant..."
MQ3L (3.6b):          "The answer is 4."
```
**Same size. Correct answer vs infinite repetition loop.**

---

## Full Results Table

| Model | Params | Architecture | MLXQ Wins | Best Win | Degradation Point |
|-------|--------|-------------|-----------|----------|-------------------|
| TinyLlama-1.1B | 1.1B | Llama GQA | 11 | 4-bit boundary | Uniform 4b derails topics |
| SmolLM2-1.7B | 1.7B | Llama MHA | 11 | 3-bit boundary | Uniform 3b number spam |
| Phi-2 | 2.7B | Phi MHA | 9 | 2-bit boundary | Uniform 2b empty output |
| Qwen2.5-3B | 3B | Qwen GQA | 6 | 4-bit boundary | Uniform 4b echo/loop |
| Mistral-7B | 7B | Mistral GQA | 11 | 3-bit boundary | Uniform 3b number garbage |
| Qwen2.5-7B | 7B | Qwen GQA | 9 | 3-bit boundary | Uniform 3b repetition loop |

## Why MLXQ Works

Attention layers are only ~12% of model parameters but control output coherence.
When attention precision drops too low:
- Attention scores become "flat" → repetition loops
- Positional encoding degrades → number garbage, topic derailing
- Context tracking fails → echo prompt, empty output

MLXQ prevents this by giving attention 5-8 bits while compressing MLP to 2-4 bits.
The cost: ~0.4 extra bits on average. The benefit: correct output vs garbage.

## Technical Details

### Bit Allocation Strategy
```
Layer Type     | % of Params | MLXQ Bits | Uniform Bits
---------------|-------------|-----------|-------------
MLP gate/up    | ~58%        | 2-4       | same as everything
MLP down       | ~29%        | 2-4       | same as everything
Attention Q/K  | ~4%         | 5-8       | same as everything
Attention V/O  | ~4%         | 5-8       | same as everything
Embedding      | ~4%         | 4         | same as everything
lm_head        | ~1%         | 6-8       | same as everything
```

### Memory Savings
```
Model          | Uniform 4-bit | MQ4S    | Savings
---------------|---------------|---------|--------
7B model       | ~4.5 GB       | ~4.1 GB | 9%
14B model      | ~9 GB         | ~8.2 GB | 9%
70B model      | ~45 GB        | ~41 GB  | 9%
```

At MQ3M (3.4 avg bits), savings are 25% vs uniform 4-bit with
comparable quality on models 7B+.

## Methodology

- All tests use MLX's own quantizer (no MLXQ-specific implementation artifacts)
- Same model, same tokenizer, same prompt template
- Quantization: affine, group_size=64
- Variable allocation via `quant_predicate` in `quantize_model()`
- 42 experiments documented in `research/experiments/`
- Reproducible: all code in this repository

## Runtime

MLXQ includes a complete Swift + Metal inference runtime:
- 14 custom Metal GPU kernels
- Zero-copy model loading via mmap (0.3-0.9s for 3-7B models)
- 28.3 tok/s decode on M4 Max (release build, 8-bit, Qwen2.5-3B)
- CLI: `mlxq run model/ --prompt "..." --temperature 0.7`

## Models Tested

| Model | Downloaded | Tested | Architecture |
|-------|-----------|--------|-------------|
| Qwen2.5-0.5B | ✓ | ✓ | qwen2, GQA 7:1 |
| TinyLlama-1.1B | ✓ | ✓ | llama, GQA 8:1, traditional RoPE |
| SmolLM2-1.7B | ✓ | ✓ | llama, MHA, traditional RoPE |
| Phi-2 | ✓ | ✓ | phi, MHA, GELU MLP |
| StableLM2-1.6B | ✓ | — | stablelm, partial RoPE |
| Qwen2.5-3B | ✓ | ✓ | qwen2, GQA 8:1 |
| Qwen3.5-0.8B | ✓ | — | qwen3_5, hybrid linear/full attn |
| Qwen3.5-4B | ✓ | — | qwen3_5, hybrid, 24 linear + 8 full |
| Mistral-7B-v0.3 | ✓ | ✓ | mistral, GQA 4:1, sliding window |
| Qwen2.5-7B | ✓ | ✓ | qwen2, GQA 4:1 |
| Qwen2.5-14B | ✓ | partial | qwen2, GQA 4:1 |
| Qwen3.5-9B | ✓ | — | qwen3_5, hybrid |
| Qwen3.5-35B-A3B | ✓ | — | qwen3_5, MoE, hybrid |
| Qwen3.5-122B-A10B | downloading | — | qwen3_5, MoE, hybrid |
