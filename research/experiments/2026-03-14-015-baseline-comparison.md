# Experiment 015: Baseline Model Comparison

**Date**: 2026-03-14
**Author**: Eric Jang (eric@vmlx.net)
**Status**: IMPORTANT — confirms MXQ runtime has a bug, model itself works

## Setup

- **Model**: Qwen2.5-0.5B (base, bfloat16)
- **Reference**: mlx-lm (Apple's MLX framework)
- **Same prompt**: "What is 2+2?"

## Results

### MLX (full precision, reference)
```
Prompt: "The answer to 1+1 is"
Output: "2. The answer to 1+2"

Prompt: "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
Output: "2+2=4\nassistant\n2+2=4"
```

### MXQ Runtime (our quantized model)
```
Prompt: Same chat template, identical token IDs (verified)
Output: "pecting, 证<作之告-C...hsfp-8-d12+^ c."
```

## Analysis

The model produces correct output through MLX. The same model,
quantized through our pipeline (with verified-correct dequantization),
produces garbage through our Swift+Metal runtime.

**This definitively confirms**: the issue is in the MXQ inference engine,
not in the quantization quality or the model itself.

**What's verified correct**:
- Token IDs: identical to HuggingFace reference
- Embedding dequant: GPU matches CPU (bit-identical)
- RMSNorm: correct output
- GEMV: meaningful projections
- Single-layer trace: all values reasonable
- Q/K/V biases: loaded and applied

**What remains suspect**:
- Multi-layer forward pass: 24 layers of possibly compounding error
- Attention kernel: GQA head mapping, softmax reduction
- The forward pass buffer management over multiple tokens
- lm_head GEMV over 151,936 outputs

## Next Steps

The most productive approach is to implement a CPU-side forward pass
for one token (using numpy) and compare the output logits with the GPU.
This will definitively identify which kernel is wrong.
