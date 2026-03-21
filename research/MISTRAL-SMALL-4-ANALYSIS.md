# Mistral Small 4 (119B) — Architecture Analysis for JANG + MLX Studio

Created by Jinho Jang (eric@jangq.ai) — 2026-03-21

## Model Summary

- **Name**: Mistral-Small-4-119B-2603
- **Parameters**: 119B total, 6B active (8B with embeddings)
- **Architecture**: `Mistral3ForConditionalGeneration` with `mistral4` text model
- **License**: Apache 2.0
- **Source format**: FP8 (not BF16!)
- **VLM**: Yes (Pixtral vision encoder)
- **Reasoning**: Yes (configurable `reasoning_effort`)
- **Context**: 1M tokens

## Architecture Deep Dive

### MoE Configuration
- **128 routed experts** per layer, **top-4 active**
- **1 shared expert** per layer
- **36 layers** total
- All layers have MoE (no dense layers, `first_k_dense_replace: 0`)
- Expert naming: `layers.N.experts.E.w1/w2/w3` (Mixtral-style, per-expert)
- Shared expert: `layers.N.shared_experts.w1/w2/w3`
- Gate/router: `layers.N.gate.weight`
- Expert intermediate: 2048 (small per-expert)
- Shared expert intermediate: implied from w1/w2/w3 shapes

### Attention — MLA (Multi-head Latent Attention)
**This is the biggest change from Mistral 3.** Mistral 4 uses DeepSeek-V2 style latent attention:

- `kv_lora_rank: 256` — KV latent dimension
- `q_lora_rank: 1024` — Q latent dimension
- `qk_head_dim: 128`
- `qk_nope_head_dim: 64` — non-positional encoding head dim
- `qk_rope_head_dim: 64` — RoPE head dim
- `v_head_dim: 128`
- `num_attention_heads: 32`
- `num_key_value_heads: 32` (MHA, not GQA)

**Weight names for MLA:**
```
layers.0.attention.wq_a.weight          — Q latent projection (hidden → q_lora_rank=1024)
layers.0.attention.q_a_norm.weight      — Q latent norm
layers.0.attention.wq_b.weight          — Q decompression (q_lora_rank → heads)
layers.0.attention.wkv_a_with_mqa.weight — KV latent projection (hidden → kv_lora_rank=256)
layers.0.attention.kv_a_norm.weight     — KV latent norm
layers.0.attention.wkv_b.weight         — KV decompression (kv_lora_rank → heads)
layers.0.attention.wo.weight            — Output projection
```

**MLA is CRITICAL for JANG tier classification.** These latent projections are compression bottlenecks — must be high-bit (same as DeepSeek-V2/V3).

### Vision Encoder (Pixtral)
- **220 tensors** total
- Architecture: ViT with patch_conv (not patch_embed)
- `vision_encoder.patch_conv.weight` — Conv2D, keep as float
- `vision_encoder.transformer.layers.N.attention.wq/wk/wv/wo.weight`
- `vision_encoder.transformer.layers.N.feed_forward.w1/w2/w3.weight`
- `vision_encoder.ln_pre.weight`
- 24 ViT layers, 16 heads, 1024 hidden
- Image size: 1540, patch size: 14
- **NOT quantized in source** (modules_to_not_convert includes vision_tower)

### Multimodal Projector
- Only 1 tensor: `pre_mm_projector_norm.weight`
- Plus `multimodal_projector_bias: False`
- Projector activation: GELU
- `spatial_merge_size: 2`
- Simple design — norm before projection

### FP8 Source Format
All text model weights are in FP8 format:
```
layers.0.experts.0.w1.weight          — FP8 weights
layers.0.experts.0.w1.qscale_weight   — weight quantization scale
layers.0.experts.0.w1.qscale_act      — activation quantization scale
```

Each weight tensor has 2 companion tensors: `qscale_weight` and `qscale_act`.
- `qscale_weight`: per-tensor or per-block scale for dequantizing weights
- `qscale_act`: activation scale (for static quantization inference, not needed for JANG)

**To dequantize**: `float_weight = fp8_weight * qscale_weight`

### RoPE Configuration
- `rope_type: "yarn"` with factor 128
- `original_max_position_embeddings: 8192`
- `rope_interleave: true` (interleaved RoPE, not sequential)
- `llama_4_scaling_beta: 0.1`
- Support for 1M context via YaRN scaling

### Weight Count Breakdown
- **42,741 total weights**
- Per layer: 1190 tensors (128 experts × 3 weights × 3 (weight+2 scales) + attention + norms)
- 36 layers × ~1100 + vision 220 + embedding/lm_head

## What JANG Needs

### 1. FP8 Dequantization in Converter
The converter must dequantize FP8 → float32 before quantizing to JANG:
```python
# For each weight tensor:
fp8_weight = load("layers.0.experts.0.w1.weight")
scale = load("layers.0.experts.0.w1.qscale_weight")
float_weight = fp8_weight.astype(float32) * scale
# Then quantize with mx.quantize()
```
- Skip `qscale_act` tensors (not needed for weight-only quantization)
- Vision encoder weights are already float (not FP8)

### 2. Tier Classification Updates (allocate.py)
New MLA tensor patterns for Mistral 4:
```
wq_a, wq_b         → CRITICAL (Q latent compression/decompression)
wkv_a_with_mqa      → CRITICAL (KV latent compression)
wkv_b               → CRITICAL (KV decompression)
q_a_norm, kv_a_norm → CRITICAL (latent norms)
wo                   → CRITICAL (output projection)
gate                 → CRITICAL (MoE router)
shared_experts.w1/w2/w3 → CRITICAL (always active)
experts.N.w1/w3     → COMPRESS (gate_proj/up_proj equivalent)
experts.N.w2        → COMPRESS (down_proj equivalent)
```

Note: w1=gate_proj, w2=down_proj, w3=up_proj (Mixtral naming convention)

### 3. Weight Name Mapping
Mistral 4 uses non-standard naming:
```
Mistral 4 name              → Standard / MLX name
layers.N.attention.wq_a     → layers.N.self_attn.q_a_proj
layers.N.attention.wkv_a_with_mqa → layers.N.self_attn.kv_a_proj_with_mqa
layers.N.attention.wkv_b    → layers.N.self_attn.kv_b_proj
layers.N.attention.wq_b     → layers.N.self_attn.q_b_proj
layers.N.attention.wo       → layers.N.self_attn.o_proj
layers.N.experts.E.w1       → gate_proj (SiLU gate)
layers.N.experts.E.w2       → down_proj
layers.N.experts.E.w3       → up_proj
layers.N.gate.weight        → block_sparse_moe.gate.weight
layers.N.shared_experts.*   → block_sparse_moe.shared_expert.*
vision_encoder.*            → model.vision_tower.*
```

### 4. Consolidated → Standard Safetensors
Source uses `consolidated.safetensors.index.json` with Mistral-native naming.
JANG converter needs to either:
- Read consolidated format directly
- Or convert names during loading

## What MLX Studio Needs

### 1. New Model File: `mistral4.py`
mlx-lm currently has `mistral3.py` but NOT `mistral4.py`. The key difference is **MLA attention** (like DeepSeek-V2). Need:

```python
class Mistral4Attention(nn.Module):
    """Multi-head Latent Attention (MLA) for Mistral 4"""
    def __init__(self, args):
        # Q path: x → wq_a (compress to q_lora_rank=1024) → norm → wq_b (expand to heads)
        self.wq_a = nn.Linear(hidden, q_lora_rank, bias=False)
        self.q_a_norm = nn.RMSNorm(q_lora_rank)
        self.wq_b = nn.Linear(q_lora_rank, num_heads * qk_head_dim, bias=False)

        # KV path: x → wkv_a (compress to kv_lora_rank=256) → norm → wkv_b (expand to heads)
        self.wkv_a = nn.Linear(hidden, kv_lora_rank + qk_rope_head_dim, bias=False)
        self.kv_a_norm = nn.RMSNorm(kv_lora_rank)
        self.wkv_b = nn.Linear(kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=False)

        # Output
        self.wo = nn.Linear(num_heads * v_head_dim, hidden, bias=False)

    def __call__(self, x, mask=None, cache=None):
        # Q: compress → norm → expand → split nope/rope → apply RoPE to rope part
        q_latent = self.wq_a(x)
        q_latent = self.q_a_norm(q_latent)
        q = self.wq_b(q_latent)  # (B, L, num_heads * qk_head_dim)

        # KV: compress → split latent/rope → norm latent → expand
        kv_a = self.wkv_a(x)  # (B, L, kv_lora_rank + qk_rope_head_dim)
        kv_latent, k_rope = split(kv_a, [kv_lora_rank, qk_rope_head_dim])
        kv_latent = self.kv_a_norm(kv_latent)
        kv = self.wkv_b(kv_latent)  # (B, L, num_heads * (qk_nope + v_head))
        # Split into k_nope and v

        # Apply RoPE to rope parts only (interleaved)
        # Attention with nope + rope concatenated keys
        # Cache stores the LATENT (kv_lora_rank) not the full KV — major memory saving
```

### 2. MoE Layer: `Mistral4MoE`
Similar to existing MoE but with:
- 128 experts, top-4 routing
- Shared expert (always active)
- w1/w2/w3 naming (not gate_proj/up_proj/down_proj)
- FP8 dequantization on load

### 3. Vision Encoder: Pixtral
Already exists in mlx-lm as `pixtral.py`. May need minor updates for Mistral 4's specific config.

### 4. FP8 Weight Loading
New weight format — each tensor has:
- `.weight` — FP8 quantized data
- `.qscale_weight` — dequantization scale
- `.qscale_act` — activation scale (drop)

Need FP8 dequant in the model loader:
```python
for name, tensor in weights.items():
    if name.endswith('.qscale_act'):
        continue  # Skip activation scales
    if name.endswith('.qscale_weight'):
        continue  # Will be consumed during dequant
    if f"{name}.qscale_weight" in weights:
        # FP8 tensor — dequantize
        scale = weights[f"{name}.qscale_weight"]
        tensor = tensor.astype(mx.float32) * scale
        tensor = tensor.astype(mx.bfloat16)
```

### 5. Consolidated Safetensors Format
Source model uses `consolidated.safetensors.index.json` instead of `model.safetensors.index.json`. Need to handle both.

### 6. YaRN RoPE
Complex RoPE configuration with interleaved layout, YaRN scaling, and multiple parameters. Need to implement or port from transformers.

### 7. Sanitize/Weight Mapping
Map Mistral-native names to mlx-lm conventions:
- `layers.N.attention.*` → `model.layers.N.self_attn.*`
- `layers.N.experts.*` → `model.layers.N.block_sparse_moe.experts.*`
- `layers.N.gate.*` → `model.layers.N.block_sparse_moe.gate.*`
- `layers.N.shared_experts.*` → `model.layers.N.block_sparse_moe.shared_expert.*`
- Vision encoder mapping

## Complexity Assessment

| Component | Difficulty | Notes |
|-----------|-----------|-------|
| FP8 dequantization | Medium | Known format, just scale × weight |
| MLA attention | **Hard** | New architecture, latent caching, split nope/rope |
| MoE with shared expert | Easy | Same as existing (128 experts, top-4) |
| Vision (Pixtral) | Easy | Already in mlx-lm |
| Weight name mapping | Medium | Consolidated format + non-standard names |
| YaRN RoPE | Medium | Complex but documented |
| Chat template + reasoning | Easy | Standard handling |
| JANG tier classification | Easy | Add MLA patterns |

**Estimated effort**:
- JANG converter support: 2-3 hours (FP8 dequant + name mapping + MLA tiers)
- MLX Studio model file: 4-6 hours (MLA attention from scratch + testing)
- Full pipeline (convert + load + inference + benchmark): 1-2 days

## Key Differences from Models We Support

| Feature | Qwen3.5 | MiniMax | Nemotron-H | **Mistral 4** |
|---------|---------|---------|------------|--------------|
| Attention | GQA + GatedDeltaNet | MHA | GQA (sparse) | **MLA (latent)** |
| MoE experts | 256-512 | 256 | 128-512 | **128** |
| Source format | BF16 | FP8 | FP8 | **FP8** |
| Vision | Qwen3_5 VL | None | None | **Pixtral** |
| Weight naming | HF standard | HF standard | HF standard | **Consolidated (Mistral-native)** |
| KV cache | Full KV | Full KV | Full + Mamba state | **Latent (compressed)** |
| Context | 256K | 256K | 1M | **1M** |

The biggest new challenge is **MLA attention** — we haven't implemented this for any model yet. It's the same architecture as DeepSeek-V2/V3, just with Mistral-specific naming.

## Updated Findings (2026-03-21)

### Two Weight Formats Available

The source model has BOTH formats:
1. `consolidated-*.safetensors` — Mistral-native naming (w1/w2/w3, wq_a/wkv_b)
2. `model-*.safetensors` — HuggingFace standard naming (gate_proj, q_a_proj, etc.)

**Use the `model-*` format.** It has:
- Standard HF naming our converter already handles
- Pre-stacked 3D expert tensors (128, out, in) — already JANG format
- Fused `gate_up_proj` — split same as Qwen3.5
- MLA naming matches DeepSeek-V2 (`kv_a_proj_with_mqa`, etc.)

### FP8 Format Details

Weights are `uint8` (E4M3 FP8) with per-tensor scales:
```
*.weight           — uint8, the FP8 weights
*.weight_scale_inv — bfloat16 scalar, inverse scale
*.activation_scale — bfloat16 scalar (drop, not needed for weight-only)
```

Dequantize: `float_weight = uint8_weight.astype(float32) * (1.0 / weight_scale_inv)`

For pre-stacked experts: `experts.gate_up_proj_scale_inv` has shape `(128, 1, 1)` — per-expert scale.

### What Already Works in JANG

1. **MLA tier classification** — `q_a_proj`, `kv_a_proj_with_mqa`, etc. already in TIER_RULES as CRITICAL
2. **Fused gate_up_proj splitting** — same as Qwen3.5 MoE
3. **Pre-stacked experts** — already the format JANG v2 stores
4. **Gate is already float** — (128, 4096) bfloat16, no dequant needed
5. **128 experts** — no bfloat16 or MLP asymmetry needed (not 512)

### What Needs Adding

1. **FP8 `uint8 + scale_inv` dequant** — new format (MiniMax used block-wise FP8, this is per-tensor)
2. **`model_type: "mistral4"` in architectures.py** — new model type detection
3. **mlx-lm model file** — needs `mistral4.py` with MLA attention (or check if existing `mistral3.py` can handle it)
4. **Drop `activation_scale` tensors** during conversion
5. **Handle `language_model.model.layers.*` prefix** — VLM naming with `language_model.` prefix

### Tensor Summary (model-* format)

| Tensor | Shape | dtype | Tier |
|--------|-------|-------|------|
| `embed_tokens.weight` | (131072, 4096) | bf16 | IMPORTANT |
| `lm_head.weight` | (131072, 4096) | bf16 | CRITICAL |
| `self_attn.q_a_proj.weight` | (1024, 4096) | uint8 | CRITICAL |
| `self_attn.q_b_proj.weight` | (4096, 1024) | uint8 | CRITICAL |
| `self_attn.kv_a_proj_with_mqa.weight` | (320, 4096) | uint8 | CRITICAL |
| `self_attn.kv_b_proj.weight` | (6144, 256) | uint8 | CRITICAL |
| `self_attn.o_proj.weight` | (4096, 4096) | uint8 | CRITICAL |
| `mlp.experts.gate_up_proj` | (128, 4096, 4096) | uint8 | COMPRESS |
| `mlp.experts.down_proj` | (128, 4096, 2048) | uint8 | COMPRESS |
| `mlp.gate.weight` | (128, 4096) | bf16 | CRITICAL |
| `shared_experts.gate_proj.weight` | (2048, 4096) | uint8 | CRITICAL |
| `shared_experts.down_proj.weight` | (4096, 2048) | uint8 | CRITICAL |
| `shared_experts.up_proj.weight` | (2048, 4096) | uint8 | CRITICAL |
| `vision_encoder.*` | various | bf16 | IMPORTANT |
