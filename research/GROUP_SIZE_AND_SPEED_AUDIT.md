# Group Size, Speed & Size Audit

Created by Jinho Jang (eric@jangq.ai) — 2026-03-17

## Problem Statement

JANG uses a single global `block_size` (group_size) for ALL tensors. But different tensor types
need different group sizes for optimal speed, quality, and correctness:

- MoE routers: MUST be gs=64 at 8-bit (MiniMax breaks otherwise)
- Expert MLP with 150+ experts: gs=128 for speed (cache pressure fix)
- Everything else: gs=64 (standard)

The current code applies ONE group_size to everything. This causes:
1. Router at gs=128 when it should be gs=64 → potential quality issues
2. No way to mix group sizes within a model

## What Needs to Change

### 1. Per-tensor group_size in converter — DONE
- [x] `_get_tensor_group_size()` function: router/gate → gs=64, everything else → model default
- [x] Modified `convert.py` quantization loop to use per-tensor `tensor_gs`
- [x] RTN fallback path also uses `tensor_gs`
- [x] `num_experts` moved to module scope (was inside if-block)
- [x] config.json stores the DOMINANT group_size (mlx_lm reads it for model init)

### 2. Loader compatibility — DONE
- [x] `_fix_quantized_bits` now infers BOTH bits AND group_size from tensor shapes
- [x] Router-aware: tries gs=64 first for `.gate` tensors
- [x] Prefers module's initialized gs (from config.json) for non-router tensors
- [x] Handles ambiguity: (4-bit,gs=128) vs (8-bit,gs=64) resolved by name-based heuristic
- [x] Tested on real Qwen3.5-4B model — all layers correct

### 3. Size verification
- [ ] Confirm disk size ≈ RAM size for v2 models
- [ ] Check that mixed gs doesn't inflate model size

### 4. Speed verification — PENDING (needs MiniMax v2 to finish converting)
- [ ] Benchmark tok/s on MiniMax JANG_2L with gs=128 + router gs=64
- [ ] Compare with CRACK reference speeds (Q4: ~50 tok/s)

## Math: Why group_size Matters for Speed

### gather_qmm kernel (MoE expert dispatch)
- For each token, 8 experts are selected out of 256
- Each expert's weight is a quantized matrix
- gather_qmm needs to fetch the right rows from each selected expert
- With gs=64: each group = 64 weights packed into 64*bits/32 uint32 values
  - At 2-bit: 64*2/32 = 4 uint32 per group (very small → cache line waste)
  - At 4-bit: 64*4/32 = 8 uint32 per group
- With gs=128: each group = 128 weights
  - At 2-bit: 128*2/32 = 8 uint32 per group (better cache utilization)
  - At 4-bit: 128*4/32 = 16 uint32 per group
- 256 experts × 8 active = many small random accesses → cache pressure
- Larger groups = fewer groups per row = fewer random accesses = faster

### quantized_matmul kernel (standard layers)
- Sequential access pattern, not random
- gs=64 vs gs=128 makes less difference
- gs=64 gives slightly better quantization quality (more fine-grained scales)

### Router/gate tensor
- Very small tensor: (num_experts, hidden_size) = (256, 3072)
- Total: 786K params = tiny
- gs=64 vs gs=128 doesn't matter for speed (too small)
- gs=64 gives better precision for this critical tensor
- MUST be 8-bit regardless

## Rules (Final)

| Tensor Type | Bit Width | group_size | Why |
|-------------|-----------|-----------|-----|
| MoE router/gate | 8 | **64** | Precision critical, tiny tensor |
| Shared expert gate | 8 | **64** | Same as router |
| Expert MLP (150+ experts) | 2-4 | **128** | Speed (gather_qmm cache) |
| Expert MLP (<150 experts) | 2-4 | 64 | Not enough experts for cache issue |
| Attention (q/k/v/o_proj) | 4-8 | 64 | quantized_matmul, precision matters |
| Linear attention (DeltaNet) | 4-8 | 64 | Same as attention |
| Embedding | 4-8 | 64 | Standard |
| Vision encoder | 4-8 | 64 | Standard |
| Mamba (A_log, D, dt_proj) | 8 | 64 | Precision critical |

## Implementation — COMPLETED 2026-03-17

1. [x] `_get_tensor_group_size()` in convert.py — router/gate → gs=64, rest → model default
2. [x] Quantization loop uses per-tensor `tensor_gs` (both mx.quantize and RTN paths)
3. [x] config.json stores DOMINANT group_size (mlx_lm reads it for model init)
4. [x] `_fix_quantized_bits` infers both bits AND group_size from tensor shapes
5. [x] Router-aware: prefers gs=64 for `.gate` tensors to resolve shape ambiguity

## MiniMax-Specific Rules (from CRACK research)

### Architecture
- 62 layers, ALL standard attention (no SSM/hybrid)
- 256 experts per layer (REAP 172B: 192 experts), 8 active per token
- Hidden: 3072, Intermediate: 1536 per expert
- Routing: Sigmoid + bias correction (not softmax)
- Source format: FP8 (float8_e4m3fn) with block-wise 128×128 scaling
- model_type: `minimax_m2` → mlx_lm `models/minimax.py`

### group_size by quant level (CRACK-verified)
| Quant Level | Expert MLP gs | Router gs | Reference |
|-------------|---------------|-----------|-----------|
| 2-bit | 128 | 64 (Q8) | JANG JANG_2L |
| 4-bit | 128 (172B) or 64 (general) | 64 (Q8) | mlx-community |
| 6-bit | 64 | 64 (Q8) | dealignai |
| 8-bit | 64 | 64 (Q8) | dealignai |

### Tokenizer (CRITICAL)
- `chat_template.jinja` is a SEPARATE FILE (not in tokenizer_config.json)
- `tokenizer.json` must have `"normalizer": {"type": "NFC"}`
- `pre_tokenizer` must be `Sequence[Split(GPT-2 regex), ByteLevel]`
- mlx_lm.convert CORRUPTS both → must copy from source or mlx-community after conversion

### Inference Settings (REQUIRED)
- temperature: 1.0 (REQUIRED — greedy/temp=0 causes infinite thinking loops)
- top_p: 0.95, top_k: 40, do_sample: true
- Thinking is ALWAYS ON (template unconditionally injects `<think>`)
- Optional: repetition_penalty: 1.1

### Speed Reference (Mac Studio M4 Ultra 256GB)
| Model | Profile | tok/s | GPU | Load Time |
|-------|---------|-------|-----|-----------|
| MiniMax JANG_2L (v2) | 2-bit | **48-50** | 63 GB | 67s (I/O) |
| MiniMax MLX Q4 | 4-bit | ~50 | ~91 GB | similar |
| MiniMax MLX Q6 | 6-bit | ~42 | ~131 GB | similar |
| MiniMax MLX Q8 | 8-bit | ~38 | ~171 GB | similar |

Key: 2-bit JANG matches 4-bit MLX speed at 30% less RAM.

## Qwen3.5 Layer Types (for reference)

### Full Attention (every 4th layer)
- q_proj, k_proj, v_proj, o_proj — standard GQA
- q_norm, k_norm — QK normalization
- JANG tier: CRITICAL (6-8 bit)

### Linear Attention / GatedDeltaNet (75% of layers)
- in_proj_qkv (fused Q+K+V)
- in_proj_z (Z gate)
- in_proj_a, in_proj_b (SSM projections)
- conv1d, A_log, dt_bias, norm
- out_proj
- JANG tier: IMPORTANT (4-8 bit) except out_proj → COMPRESS

### MoE (every layer on MoE models)
- experts.gate_up_proj (3D batched, fused) → split to gate_proj + up_proj
- experts.down_proj (3D batched)
- gate.weight (router — MUST be 8-bit)
- shared_expert.{gate,up,down}_proj
- shared_expert_gate.weight
- JANG tier: experts → COMPRESS, router → CRITICAL, shared → IMPORTANT

### Vision Encoder (Qwen3.5 VLM)
- 24-27 ViT blocks
- 297-333 vision tensors
- JANG preserves all vision tensors (quantized at COMPRESS tier)
- Requires preprocessor_config.json + video_preprocessor_config.json
