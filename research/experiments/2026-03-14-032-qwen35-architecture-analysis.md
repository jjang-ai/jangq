# Experiment 032: Qwen 3.5 Architecture Analysis

**Date**: 2026-03-14
**Author**: Eric Jang (eric@vmlx.net)
**Status**: ANALYSIS — major new architecture features identified

## Qwen3.5-0.8B Architecture

This is a fundamentally different architecture from Qwen2.5:

### Hybrid Attention (Linear + Full)
```python
layer_types: ['linear_attention', 'linear_attention', 'linear_attention',
              'full_attention', 'linear_attention', 'linear_attention',
              'linear_attention', 'full_attention', ...]
```
- Every 4th layer is **full attention** (standard Q×K^T softmax)
- Other 3/4 of layers are **linear attention** (SSM-like, no softmax)
- `full_attention_interval: 4`

### Linear Attention (SSM-like)
- `linear_conv_kernel_dim: 4` — causal 1D convolution
- `linear_key_head_dim: 128` — separate K head dimension for linear attn
- `linear_value_head_dim: 128` — separate V head dimension
- `linear_num_key_heads: 16` — 16 KV heads for linear attention
- `mamba_ssm_dtype: float32` — SSM states in float32

### RoPE
- `rope_theta: 10000000` (10M, not 1M like Qwen2.5)
- `partial_rotary_factor: 0.25` — only rotates 25% of head dimensions!
- `mrope_interleaved: True` — multi-resolution RoPE for vision
- `mrope_section: [11, 11, 10]` — 3 sections for temporal/height/width

### Other
- `head_dim: 256` — large head dimension
- `attn_output_gate: True` — gated attention output
- `mtp_num_hidden_layers: 1` — multi-token prediction head
- Vision-language model with image_token_id, video_token_id

### Quantization Implications for MLXQ

1. **Linear attention layers** have different weight structure than full attention:
   - May have different sensitivity profile
   - SSM states in float32 → keep high precision
   - conv1d weights need special handling

2. **Partial RoPE** means only 25% of head dimensions are rotated:
   - Our RoPE kernel needs to handle partial rotation
   - The unrotated dimensions pass through unchanged

3. **Large head_dim (256)** changes GEMV/attention kernel tuning

4. **Mixed layer types** require per-layer-type quantization strategy:
   - Full attention layers: use standard attention quantization
   - Linear attention layers: use SSM quantization (protect conv/state params)

5. **Vision encoder** needs separate quantization handling (typically less compressible)

## Comparison with Qwen2.5

| Feature | Qwen2.5-3B | Qwen3.5-0.8B |
|---------|-----------|-------------|
| Attention | GQA only | Hybrid linear/full |
| Layer types | All same | Mixed per layer |
| Head dim | 128 | 256 |
| RoPE | Non-traditional, θ=1M | Partial (25%), θ=10M, MRoPE |
| Conv | None | 1D causal (k=4) |
| SSM | None | Yes (in linear layers) |
| Vision | No | Yes (VL model) |
| Gating | No | attn_output_gate |

## MLXQ Adaptation Needed

1. Add `linear_attention` layer type to architecture detection
2. Add partial RoPE support to Metal kernel
3. Add conv1d quantization support
4. Per-layer-type bit allocation: linear attn vs full attn
5. Vision encoder handling (if quantizing VL models)

This is the cutting-edge architecture that MLXQ should target to
differentiate from GGUF and other formats that don't handle hybrids well.
