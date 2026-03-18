# JANG Model Reference — Architecture & Quantization Notes

> Sourced from CRACK abliteration research. Critical info for JANG quantization quality.

## Critical Rules

- **MiniMax group_size MUST be 128** (not 64) — cache pressure causes 15-25% speed loss at 64
- **MoE gate/router MUST be 8-bit** — lower bits break expert routing (especially 512-expert models)
- **dtype bug**: `mx.quantize()` returns float32 scales/biases → cast to bfloat16 or speed drops 60%
- **VL models**: use `mlx_vlm.convert` not `mlx_lm.convert` — wrong converter strips vision tower
- **Per-tensor quant config breaks speed**: 397B drops from 32 tok/s to 9.4 tok/s — use uniform per-tier only
- **MiniMax temp=1.0 mandatory** — greedy decoding causes thinking loops

## Qwen 3.5 Family

### Common
- EOS tokens: `[248046, 248044]` (NOT Qwen 2.5's `[151645, 151643]`)
- Chat template: thinking mode with `enable_thinking` parameter
- Layer pattern: [SSM, SSM, SSM, FA] repeating (`full_attention_interval=4`)
- SSM type: GatedDeltaNet linear attention
- All models have vision encoder (VL capable)

### 0.8B (Dense)
- 24 layers (6 FA + 18 SSM), hidden=1024, inter=3584
- Q/KV heads: 8/2, head_dim=256
- tie_word_embeddings=true, single shard
- Speed: ~300 tok/s Q4

### 2B (Dense)
- 24 layers (6 FA + 18 SSM), hidden=2048, inter=6144
- Q/KV heads: 16/2, head_dim=256
- tie_word_embeddings=true
- Speed: ~250 tok/s Q4

### 4B (Dense)
- 32 layers (8 FA + 24 SSM), hidden=2560, inter=9216
- Q/KV heads: 16/8, head_dim=256
- tie_word_embeddings=true
- Speed: ~150 tok/s Q4

### 9B (Dense)
- 32 layers (8 FA + 24 SSM), hidden=4096, inter=12288
- Q/KV heads: 32/8, head_dim=256
- tie_word_embeddings=false
- Speed: not documented

### 27B (Dense)
- 64 layers (16 FA + 48 SSM), hidden=5120
- Q/KV heads: unknown/unknown, head_dim=256
- tie_word_embeddings=false
- Speed: Q4=37 tok/s, Q6=27 tok/s, Q8=22 tok/s

### 35B-A3B (MoE)
- 40 layers (10 FA + 30 SSM), hidden=2048
- 256 experts, 8 active, head_dim=256
- tie_word_embeddings=true
- Speed: Q4=88 tok/s, Q8=80 tok/s

### 122B-A10B (MoE)
- 48 layers (12 FA + 36 SSM), hidden=3072
- 256 experts, 8 active, head_dim=256
- Q/KV heads: 32/2 (GQA 16:1)
- Expert intermediate: 1024, shared expert: 1024
- tie_word_embeddings=false
- **mrope_section fix**: set to `[11, 11, 10]` if config has `[]`
- Speed: Q4=56-58 tok/s

### 397B-A17B (MoE, text-only)
- 60 layers (15 FA + 45 SSM), hidden=4096
- **512 experts, 10 active** (highest expert count)
- Shared expert: 1 (always active)
- MTP head: strip during conversion
- **Gate MUST be Q8** — lower bits break routing for 512 experts
- **GPU Metal timeout on large expert tensors** — need chunked quantization
- **Per-tensor quant config = 9.4 tok/s** — DO NOT USE, uniform only
- Speed: Q4=32 tok/s (uniform)

## MiniMax-M2.5

- 62 layers, ALL standard attention (NO SSM)
- hidden=3072, inter=1536 per expert
- 256 experts (full), 8 active
- Routing: sigmoid + bias (NOT softmax)
- **group_size=128 MANDATORY** — 64 causes 15-25% speed loss
- **FP8 block-wise source**: 128×128 blocks, formula: `w_bf16 = w_fp8 * scale_inv[i//128, j//128]`
- **Tokenizer corruption**: `mlx_lm.convert` strips NFC normalizer → thinking loops. Copy tokenizer from `mlx-community/MiniMax-M2.5-4bit`
- **temp=1.0 mandatory** — greedy = guaranteed loop
- EOS: 200020, think: 200050, /think: 200051
- Speed: Q4=50-53 tok/s, Q8=38-40 tok/s (at group_size=128)

## Step 3.5 Flash (121B/149B)

- 45 layers, full + sliding attention (NO SSM)
- hidden=4096
- Sliding window: 512 tokens (narrow)
- Full attn: Q=64 heads, KV=8, head_dim=128
- Sliding attn: Q=96 heads, KV=8, head_dim=128
- Novel `g_proj` per head
- Dense L0-L2: inter=11264, MoE L3-L44: expert_inter=1280
- 173-216 experts, 8 active
- Router: sigmoid + bias + scaling=3.0
- Speed: Q4=~48 tok/s

## GPT-OSS 120B

- 36 layers, alternating sliding/full attention (NO SSM)
- hidden=2880, head_dim=64 (smallest)
- Q=64 heads, KV=8
- FFN inter=2880 (1:1 ratio — unusual)
- 128 experts, **4 active** (lowest k)
- **mxfp4 native source** (NOT post-training quantized)
- **FP16 overflow at L30** — MUST use BF16
- Speed: ~80 tok/s Q4

## Nemotron-H Super 120B

- 88 layers: 40 Mamba-2 SSM + 40 MoE FFN + 8 Dense Attention
- hidden=4096, head_dim=128
- 512 routed + 1 shared expert, **22 active** (highest k)
- Expert inter=2688, shared inter=5376
- LatentMoE: tokens compressed 4096→1024 before routing
- **trust_remote_code=True required**
- **Stock mlx_lm BROKEN for LatentMoE inference**
- Non-standard naming: `backbone.layers.N.mixer.*`
- MTP: 1040 tensors to strip
- Vocab: 131072

## JANG Quantization Implications

### What JANG must handle per model:
| Model | group_size | Router bits | Special |
|-------|-----------|-------------|---------|
| Qwen 3.5 (all) | 64 | 8 (CRITICAL) | GatedDeltaNet SSM layers |
| MiniMax M2.5 | **128** | 8 (CRITICAL) | FP8 source, sigmoid routing |
| Step Flash | 64 | 8 (CRITICAL) | g_proj, narrow sliding window |
| GPT-OSS | 64 | 8 (CRITICAL) | mxfp4 source, BF16 only |
| Nemotron-H | 64 | 8 (CRITICAL) | LatentMoE, custom naming |
| DeepSeek | 64 | 8 (CRITICAL) | MLA latent projections |

### Speed rules:
- Cast mx.quantize() scales/biases to bfloat16 ALWAYS
- MiniMax: group_size=128 or lose 15-25% speed
- 397B: NO per-tensor quant config (9.4 vs 32 tok/s)
- MiniMax: temp=1.0 required (greedy loops)
