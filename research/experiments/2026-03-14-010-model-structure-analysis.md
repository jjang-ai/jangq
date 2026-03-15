# Experiment 010: Model Structure Analysis for Forward Pass

**Date**: 2026-03-14
**Author**: Eric Jang (eric@vmlx.net)
**Status**: ANALYSIS — informs inference engine design

## Purpose

Before wiring the forward pass, verify the exact tensor layout and
architecture details of the Qwen2.5-0.5B MXQ model to ensure correctness.

## Findings

### Model Architecture
- **Type**: qwen2 (standard transformer with GQA)
- **Layers**: 24
- **Hidden**: 896
- **Intermediate**: 4864
- **Q Heads**: 14
- **KV Heads**: 2 (GQA ratio = 7:1)
- **Head dim**: 64 (= 896 / 14)
- **Vocab**: 151,936 (151,643 base + 293 special tokens)
- **Max seq**: 131,072 (128K context)
- **Tied embeddings**: YES — lm_head shares weights with embed_tokens

### Quantized Tensors (169 total)
Per layer (24 layers × 7 weight matrices = 168, plus embed_tokens = 169):
- `model.embed_tokens` — 2,127,104 blocks, ALL at 4-bit (floor enforced)
- `model.layers.N.self_attn.{q,k,v,o}_proj` — mixed 2/3/4-bit
- `model.layers.N.mlp.{gate,up,down}_proj` — mixed 2/3/4-bit

### Embedding Tensor Analysis
```
model.embed_tokens.qweight:       (68,067,328,) uint8   — 68 MB packed
model.embed_tokens.bit_map:       (2,127,104,)  uint8   — all 4-bit
model.embed_tokens.scales:        (2,127,104,)  float16
model.embed_tokens.zeros:         (2,127,104,)  float16
model.embed_tokens.block_offsets: (2,127,104,)  uint32
```

Math verification:
- embed_tokens shape: (151936, 896) = 136,134,656 weights
- blocks: 136,134,656 / 64 = 2,127,104 blocks ✓
- At 4-bit: 2,127,104 × 32 bytes = 68,067,328 bytes ✓

### Non-quantized Tensors (290 total)
- 48 RMSNorm weights: `model.layers.N.{input_layernorm,post_attention_layernorm}.weight` — (896,) float16
- 1 final norm: `model.norm.weight` — (896,) float16
- 241 bias terms from attention/MLP layers

### Tokenizer (Qwen2Tokenizer — BPE)
- Type: BPE with byte fallback
- Base vocab: 151,643
- Merges: 151,387
- Special tokens: 22 (im_start, im_end, vision_start, etc.)
- BOS: None (Qwen doesn't use BOS)
- EOS: `<|endoftext|>` (id 151643)
- Chat markers: `<|im_start|>` (151644), `<|im_end|>` (151645)
- Max position: 131,072

## Design Decisions for Forward Pass

### 1. Embedding Lookup (Critical)
The embedding is quantized at 4-bit. For a single token lookup:
- Need to extract row `token_id` from the (151936, 896) quantized matrix
- Each row = 896 weights = 14 blocks of 64 weights each
- 14 blocks × 4-bit × 64 weights = 14 × 32 bytes = 448 bytes to read per token
- Then dequant 14 blocks → 896 float16 values
- This is efficient — only 448 bytes from GPU memory per token

**Implementation**: Specialized kernel `mxq_embedding_dequant` that reads
a single row from the quantized embedding table.

### 2. Tied Embeddings / LM Head (Critical)
Since `tie_word_embeddings: true`, the lm_head uses embed_tokens weights
transposed. For token generation:
- Need: logits[v] = sum_h(hidden[h] × embed[v][h]) for all v in vocab
- This is a dequant GEMV with the embedding matrix: output (151936,) = embed (151936, 896) × hidden (896,)
- Our existing `mxq_dequant_gemv` kernel can handle this

### 3. GQA Attention Layout
- 14 Q heads, 2 KV heads → each KV head serves 7 Q heads
- Q projection: (896 → 896) = 14 heads × 64 dim
- K projection: (896 → 128) = 2 heads × 64 dim
- V projection: (896 → 128) = 2 heads × 64 dim
- O projection: (896 → 896) = 14 heads × 64 dim

KV cache layout: (max_seq, 2, 64) = 128 float16 values per position
- KV cache for 2K context: 24 layers × 2K × 128 × 2 bytes × 2 (K+V) = 24.6 MB
- KV cache for 32K context: 24 layers × 32K × 128 × 2 × 2 = 393 MB

### 4. RoPE Parameters
- theta_base: 1,000,000 (Qwen uses 1M base, not 10K)
- Need to verify: `rope_theta` in config.json

## Key Risks
1. RoPE theta base — if wrong, attention patterns will be completely wrong
2. GQA head mapping — the `head * n_kv_heads / n_heads` mapping must be integer
3. Tied embedding dequant — transposed access pattern is different from normal GEMV
4. Chat template — Qwen uses `<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`
