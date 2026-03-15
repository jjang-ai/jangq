# Modern Model Architectures and Attention Mechanisms for Quantization

This document catalogs the attention types, model architectures, weight naming conventions, and quantization considerations observed across 15+ models during the CRACK abliteration research project (Feb-Mar 2026). The focus is on practical details that affect weight surgery and quantization pipelines on Apple Silicon (MLX/Metal).

Source: `/Users/eric/CRACK_abliteration/` -- docs, research, CLAUDE.md, HANDOFF_NOTES.md, and per-model findings.

---

## Table of Contents

1. [Attention Type Taxonomy](#1-attention-type-taxonomy)
2. [Architecture Taxonomy](#2-architecture-taxonomy)
3. [Model Catalog](#3-model-catalog)
4. [Per-Model Architecture Details](#4-per-model-architecture-details)
5. [Weight Naming Conventions](#5-weight-naming-conventions)
6. [Quantization Considerations by Architecture](#6-quantization-considerations-by-architecture)
7. [Cross-Architecture Comparison Tables](#7-cross-architecture-comparison-tables)
8. [Lessons for Quantization Engine Design](#8-lessons-for-quantization-engine-design)

---

## 1. Attention Type Taxonomy

Six distinct attention mechanisms appear across the models studied.

### 1.1 Multi-Head Attention (MHA)

Standard transformer attention with independent Q, K, V projections per head. Each head has its own query, key, and value parameters. Not observed in the large MoE models studied (all use GQA or variants), but the foundational mechanism.

### 1.2 Grouped Query Attention (GQA)

Multiple query heads share a single key-value head group. Reduces KV cache size proportionally to the grouping ratio.

| Model | Q Heads | KV Heads | Ratio | Head Dim |
|-------|---------|----------|-------|----------|
| Qwen 3.5 4B | 20 | 4 | 5:1 | 128 |
| Qwen 3.5 9B | 28 | 4 | 7:1 | 128 |
| Qwen 3.5 27B | 24 | 4 | 6:1 | 256 |
| Qwen 3.5 35B (MoE) | 16 | 4 | 4:1 | 128 |
| Qwen 3.5 122B (MoE) | 32 | 2 | 16:1 | 256 |
| Qwen 3.5 397B (MoE) | 32 | 2 | 16:1 | 256 |
| MiniMax M2.5 172B | 48 | 8 | 6:1 | 128 |
| GPT OSS 120B | 64 | 8 | 8:1 | 64 |
| GLM-4.7 | 96 | 8 | 12:1 | 128 |
| Step 3.5 Flash (full) | 64 | 8 | 8:1 | 128 |
| Step 3.5 Flash (sliding) | 96 | 8 | 12:1 | 128 |
| Nemotron Super 120B | 32 | 2 | 16:1 | 128 |

GQA ratio affects quantization because `k_proj` and `v_proj` tensors are much smaller than `q_proj`. With 16:1 ratios (Qwen 122B/397B, Nemotron), the KV projection tensors are tiny relative to the query projection.

### 1.3 GatedDeltaNet (Linear Attention / SSM)

Qwen 3.5's hybrid architecture uses GatedDeltaNet as a linear attention / SSM mechanism in 75% of layers (every non-FA layer). This is NOT standard transformer attention -- it is a recurrent mechanism that processes sequences in linear time.

Key properties:
- Carries information via a **recurrent state channel** invisible to residual-stream interventions
- Has `conv1d` (kernel_size=4), `A_log`, `dt_bias` parameters (state-space model components)
- Output projection (`linear_attn.out_proj.weight`) projects back into the residual stream
- Config fields: `linear_num_value_heads: 64`, `linear_num_key_heads: 16`, `linear_conv_kernel_dim: 4`

**Tensor naming (Qwen 3.5 SSM layers):**
```
linear_attn.out_proj.weight      [hidden, value_heads * head_dim]
linear_attn.in_proj_qkv.weight   [fused_qkv_dim, hidden]
linear_attn.in_proj_z.weight     [z_dim, hidden]
linear_attn.in_proj_a.weight     [num_heads, hidden]
linear_attn.in_proj_b.weight     [num_heads, hidden]
linear_attn.A_log                [num_heads]  (F32)
linear_attn.conv1d.weight        [fused_dim, 1, 4]
linear_attn.dt_bias              [num_heads]
linear_attn.norm.weight          [norm_dim]
```

**Quantization impact:** The `A_log`, `dt_bias`, and `conv1d` parameters are small and should stay at high precision (bf16/f32). The main SSM projections (`in_proj_*`, `out_proj`) can be quantized like attention projections, but the SSM recurrent state dynamics make these weights more sensitive to quantization noise than standard attention weights.

### 1.4 Mamba-2 (Selective State Space Model)

Used in Nemotron Super 120B. A different SSM variant from GatedDeltaNet, with selective scanning where state transition matrices are input-dependent.

Key properties:
- 128 heads, 64 head_dim per head
- `conv_kernel=4`, `ssm_state=128`
- NO attention mechanism whatsoever -- pure SSM
- Output projection (`mixer.out_proj.weight`) projects into residual stream

**Tensor naming (Nemotron Mamba layers):**
```
backbone.layers.N.mixer.A_log                  [state transition -- keep high precision]
backbone.layers.N.mixer.D                      [skip connection]
backbone.layers.N.mixer.conv1d.weight          [kernel_size=4]
backbone.layers.N.mixer.conv1d.bias
backbone.layers.N.mixer.dt_bias                [time step]
backbone.layers.N.mixer.in_proj.weight         [input projection]
backbone.layers.N.mixer.norm.weight
backbone.layers.N.mixer.out_proj.weight        [output projection]
```

**Quantization impact:** Mamba parameters (`A_log`, `D`, `dt_bias`, `conv1d`) are critical for state dynamics. The `A_log` tensor must stay at original precision (f32). Aggressive quantization of Mamba parameters destroys sequence modeling -- Nemotron Q2 was abandoned because the hybrid architecture has a precision floor above Q2.

### 1.5 Sliding Window Attention

Used in GPT OSS 120B and Step 3.5 Flash. Every other layer (or a subset) uses a limited attention window instead of full context.

| Model | Window Size | Pattern |
|-------|-------------|---------|
| GPT OSS 120B | 128 tokens | Alternating (even=sliding, odd=full) |
| Step 3.5 Flash | 512 tokens | Non-full layers |

Sliding window layers have the same weight structure as full attention (same q/k/v/o_proj shapes), but different attention masks at inference time. From a quantization perspective, there is no difference in tensor structure. However, Step 3.5 Flash has different numbers of query heads for sliding vs full attention layers (96 vs 64), resulting in different `q_proj` and `o_proj` shapes.

### 1.6 Attention Sinks (Learned)

GPT OSS 120B introduces per-head learned sink tokens:
```
self_attn.sinks: [64]  (one per head, FP16/BF16)
```

These provide anchor tokens that attention can always attend to regardless of position. Small tensors that must stay at high precision.

---

## 2. Architecture Taxonomy

### 2.1 Pure Transformer MoE

Standard transformer blocks with MoE replacing the MLP. Every layer has both attention and MoE FFN.

**Models:** MiniMax M2.5, GPT OSS 120B, GLM-4.7, INTELLECT 3.1

Properties:
- All layers have attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
- All layers have MoE expert routing
- No SSM bypass channel
- Refusal signal (for abliteration) concentrates in residual stream -- no alternative pathway

### 2.2 Hybrid SSM + Full Attention MoE

Qwen 3.5 family architecture. Alternates between GatedDeltaNet (SSM) layers and full attention layers, with MoE FFN in all layers.

**Models:** Qwen 3.5 4B, 9B, 27B (dense), 35B, 122B, 212B, 262B, 307B/397B

Properties:
- Every 4th layer is `full_attention`; other 3 are `linear_attention` (GatedDeltaNet SSM)
- MoE FFN present in ALL layers (both FA and SSM) for MoE variants
- Dense variants (4B, 9B, 27B) have standard MLP instead of MoE
- Dual information pathway: residual stream (shared) + SSM recurrent state (SSM-only)
- Config: `full_attention_interval: 4`, `layer_types: [...]` array

**Quantization impact:** The SSM parameters (`A_log`, `dt_bias`, `conv1d`) require higher precision than attention weights. The `linear_attn.out_proj` is as important as `self_attn.o_proj` for residual stream integrity.

### 2.3 Three-Component Hybrid (Mamba + MoE + Dense Attention)

Nemotron Super 120B architecture. Three completely separate layer types that do NOT share function within a layer.

**Model:** Nemotron 3 Super 120B

Properties:
- 88 layers total: 40 Mamba-2 (M), 40 MoE FFN (E), 8 Dense Attention (*)
- Pattern: `MEMEMEM*EMEMEMEM*E...`
- Mamba layers: pure SSM, no attention, no MoE
- MoE layers: pure FFN, no attention, no SSM
- Dense layers: pure GQA attention, no MoE, no SSM
- Only 8/88 layers (9%) have attention -- fewest of any model studied
- LatentMoE: tokens compressed 4096 to 1024 before expert routing

This is fundamentally different from Qwen 3.5 where attention and MoE coexist in each layer.

### 2.4 Dual Attention MoE (Full + Sliding)

Step 3.5 Flash architecture. Attention in every layer (no SSM), but alternates between full and sliding window attention.

**Model:** Step 3.5 Flash 121B/149B

Properties:
- 45 layers, all have attention
- Full attention at: L0, L4, L8, L12, ... L44 (every 4th, 12 total)
- Sliding attention at: all others (33 total), window=512
- Different head counts: full=64Q, sliding=96Q (both 8 KV)
- Novel `g_proj` (per-head attention gate): `[64, 4096]` or `[96, 4096]`
- MoE in layers L3-L44, dense MLP in L0-L2
- Sigmoid routing with bias and scaling factor 3.0

### 2.5 Dense Transformer (Hybrid SSM+FA)

Qwen 3.5 dense variants. Same hybrid SSM+FA attention pattern but with standard MLP instead of MoE.

**Models:** Qwen 3.5 4B, 9B, 27B

Properties:
- Same SSM+FA hybrid attention as MoE variants
- Standard `mlp.down_proj`, `mlp.gate_proj`, `mlp.up_proj` instead of expert routing
- All sizes are VL (vision-language) with early fusion

---

## 3. Model Catalog

Complete list of models with architectural parameters.

| Model | Type | Layers | Hidden | Experts (active) | FFN per Expert | Attn Type | VL | Head Dim |
|-------|------|--------|--------|-------------------|----------------|-----------|-----|----------|
| Qwen 3.5 4B | Dense | 32 | 2560 | N/A | 6912 (4x) | Hybrid SSM+FA | Yes | 128 |
| Qwen 3.5 9B | Dense | 40 | 3584 | N/A | 10240 (4x) | Hybrid SSM+FA | Yes | 128 |
| Qwen 3.5 27B | Dense | 64 | 5120 | N/A | 17408 (4x) | Hybrid SSM+FA | Yes | 256 |
| Qwen 3.5 35B | MoE | 40 | 2048 | 256 (4) | 5632 (4x) | Hybrid SSM+FA | Yes | 128 |
| Qwen 3.5 122B | MoE | 48 | 3072 | 256 (8) | 1024 | Hybrid SSM+FA | Yes | 256 |
| Qwen 3.5 212B REAP | MoE | 60 | 4096 | 267 (10) | 1024 | Hybrid SSM+FA | Yes | 256 |
| Qwen 3.5 262B REAP | MoE | 60 | 4096 | 333 (10) | 1024 | Hybrid SSM+FA | Yes | 256 |
| Qwen 3.5 397B | MoE | 60 | 4096 | 512 (10) | 1024 | Hybrid SSM+FA | Yes | 256 |
| MiniMax M2.5 172B | MoE | 62 | 3072 | 256->192 REAP (8) | 1536 (0.5x) | Pure transformer | No | 128 |
| MiniMax M2.5 139B | MoE | 62 | 3072 | 256->154 REAP (8) | 1536 (0.5x) | Pure transformer | No | 128 |
| GPT OSS 120B | MoE | 36 | 2880 | 128 (4) | 2880 (1x) | Sliding+Full | No | 64 |
| INTELLECT 3.1 | MoE | 46 | 5120 | 128 (8) | ~20480 (4x) | Pure transformer | No | 128 |
| Step 3.5 Flash 121B | MoE | 45 | 4096 | 173 REAP (8) | 1280 (0.31x) | Full+Sliding | No | 128 |
| Step 3.5 Flash 149B | MoE | 45 | 4096 | 216 REAP (8) | 1280 (0.31x) | Full+Sliding | No | 128 |
| GLM-4.7 218B | MoE | 92 | 5120 | 96 REAP (8) | 1536 (0.3x) | Pure transformer | No | 128 |
| GLM-4.7 268B | MoE | 92 | 5120 | 120 REAP (8) | 1536 (0.3x) | Pure transformer | No | 128 |
| Nemotron Super 120B | Hybrid MoE | 88 | 4096 | 512 (22) | 2688 (expert), 5376 (shared) | Mamba+Dense attn | No | 128 |

---

## 4. Per-Model Architecture Details

### 4.1 Qwen 3.5 Family (Hybrid SSM+FA)

**Architecture identifier:** `Qwen3_5MoeForConditionalGeneration` (MoE) / `Qwen3_5ForConditionalGeneration` (dense)

**Layer pattern:** Every 4th layer is `full_attention`; the other 3 are `linear_attention` (GatedDeltaNet SSM).

```
L0-2:  linear_attention (SSM)    L3:  full_attention
L4-6:  linear_attention (SSM)    L7:  full_attention
...continuing every 4 layers...
```

**FA layers by model size:**
- 4B (32L): [3,7,11,15,19,23,27,31] -- 8 FA layers
- 9B (40L): every 4th -- 10 FA layers
- 27B (64L): [3,7,11,...,63] -- 16 FA layers
- 35B (40L): every 4th -- 10 FA layers
- 122B (48L): [3,7,11,15,19,23,27,31,35,39,43,47] -- 12 FA layers
- 397B (60L): [3,7,11,...,59] -- 15 FA layers

**MoE expert format (MoE variants):** Batched 3D tensors after MLX conversion.
```
mlp.experts.down_proj:       [num_experts, hidden_size, expert_intermediate]
mlp.experts.gate_up_proj:    [num_experts, fused_dim, hidden_size]
mlp.gate.weight:             [num_experts, hidden_size]
mlp.shared_expert.down_proj: [hidden_size, expert_intermediate]
mlp.shared_expert_gate:      [1, hidden_size]
```

**All Qwen 3.5 models are VL.** Unified early-fusion vision architecture. All sizes have vision built in -- there are no separate `Qwen3.5-VL-*` repos. Pipeline tag: `image-text-to-text`.

Vision encoder: Qwen3VL-style ViT, 27 layers, 1152 hidden, 333 tensors. Stored in final safetensor shards (e.g., shards 38-39 for 122B).

**VL naming convention difference:**
- VL: `language_model.model.layers.N.self_attn.o_proj.{weight,scales,biases}`
- Non-VL/MLX: `model.layers.N.self_attn.o_proj.{weight,scales,biases}`

The tensor DATA (raw bytes) is compatible between VL and non-VL because both use identical quantization parameters.

**QK normalization:** Present in 122B (`self_attn.q_norm`, `self_attn.k_norm`). May affect activation ranges.

### 4.2 MiniMax M2.5

**Architecture identifier:** `MiniMaxM2ForCausalLM` (custom, requires `trust_remote_code`)

Key architectural distinctions:
- **Sigmoid routing with bias correction** (not softmax like Qwen/GPT OSS)
- **0.5x FFN ratio** (intermediate=1536, hidden=3072) -- smaller experts than typical
- **All layers have standard attention** -- no SSM, no hybrid, no sliding window
- **Switch_mlp naming** for batched experts in MLX format
- **Safety distributed across ALL 4 attention projections** (q/k/v/o_proj), not just o_proj
- Text-only (no VL)

**Tensor naming (MLX Q4):**
```
model.layers.N.self_attn.o_proj.weight           [3072, 6144]
model.layers.N.self_attn.q_proj.weight           [6144, 3072]
model.layers.N.self_attn.k_proj.weight           [1024, 3072]
model.layers.N.self_attn.v_proj.weight           [1024, 3072]
model.layers.N.block_sparse_moe.switch_mlp.down_proj.weight  [192, 3072, 1536]  (3D batched)
model.layers.N.block_sparse_moe.gate.weight      [192, 3072]
```

**Critical tokenizer bug:** `mlx_lm.convert` corrupts the MiniMax tokenizer by stripping the NFC normalizer and GPT-2 regex pre-tokenizer. The corrected `tokenizer.json` and `tokenizer_config.json` must be copied from a known good reference after every conversion. Failure to do so causes infinite thinking loops.

**Inference requirement:** `temperature=1.0` is mandatory. Greedy/temp=0 causes infinite repetition loops. Published `generation_config.json` specifies `temp=1.0, top_p=0.95, top_k=40`.

### 4.3 GPT OSS 120B

**Architecture identifier:** `GptOssForCausalLM`

Key architectural distinctions:
- **Native mxfp4 quantization** -- expert weights trained in mxfp4, not post-training quantized
- **1:1 FFN ratio** -- `intermediate_size = hidden_size = 2880`. Zero redundancy.
- **SwiGLU activation clamping** to [-7, 7]
- **Attention sinks** -- per-head learned sink tokens `self_attn.sinks: [64]`
- **YaRN RoPE** with theta=150K, factor=32
- **Attention is FP16/BF16** (not quantized), experts are mxfp4
- **Harmony channel system** for structured safety reasoning (analysis, commentary, final channels)
- Text-only (no VL)

**Native mxfp4 packing detail:**
```
group_size=32, bits=4
8 elements per U32 word
Expert shape unpacked: [128, 2880, 2880]
Scales: uint8, 2880/32 = 90 groups per row
No biases tensor in mxfp4 mode (only weight + scales)
```

**Size breakdown:** Attention (FP16) = 1.9 GB (2.9%), MLP/Experts (mxfp4) = 61.0 GB (93.5%), Other = 2.3 GB (3.6%). Total: 65.2 GB.

**FP16 overflow problem:** The model was trained in BF16 (range up to 3.4e38). FP16 conversion (max 65,504) causes overflow in deep layers -- residual stream hidden states grow exponentially through 36 layers. Solution: cast all 579 FP16 tensors to BF16 at load time or on disk. The mxfp4 expert weights (uint32 packed) stay unchanged.

**Tensor naming:**
```
self_attn.q_proj.weight:       [4096, 2880]  BF16
self_attn.k_proj.weight:       [512, 2880]   BF16
self_attn.v_proj.weight:       [512, 2880]   BF16
self_attn.o_proj.weight:       [2880, 4096]  BF16
self_attn.{q,k,v,o}_proj.bias             BF16
self_attn.sinks:               [64]          BF16
mlp.experts.down_proj.weight:  [128, 2880, 360]  U32 (packed mxfp4)
mlp.experts.down_proj.scales:  [128, 2880, 90]   U8
mlp.experts.down_proj.bias:    [128, 2880]        FP16 (linear bias, NOT quant bias)
mlp.router.weight:             [128, 2880]        BF16
mlp.router.bias:               [128]              BF16
```

### 4.4 GLM-4.7

**Architecture identifier:** `GLM4MoeForCausalLM`

Key architectural distinctions:
- **Pure transformer MoE** (no SSM, no hybrid)
- **Sigmoid routing with bias correction** (same as MiniMax)
- **Small expert FFN** (1536 intermediate) -- same as MiniMax
- **switch_mlp naming** in MLX (same as MiniMax after sanitize stacking)
- **Partial RoPE** -- only 50% of head_dim gets rotary embeddings
- **QK normalization** (RMSNorm on Q and K after projection)
- **Attention bias** on q/k/v projections (but NOT on o_proj)
- **Shared expert** (1 per layer, always active)
- **Dense layers 0-2** (standard MLP, intermediate=12288)
- **FP8 source format** (e4m3fn, per-channel symmetric)
- Text-only (no VL), MIT license

**Tensor naming (FP8 source):**
```
model.layers.N.self_attn.o_proj.weight:    [5120, 12288] F8_E4M3
model.layers.N.self_attn.q_proj.weight:    [12288, 5120] F8_E4M3 + bias [12288]
model.layers.N.self_attn.k_proj.weight:    [1024, 5120]  F8_E4M3 + bias [1024]
model.layers.N.self_attn.v_proj.weight:    [1024, 5120]  F8_E4M3 + bias [1024]
```

**After MLX conversion (sanitize stacks experts into 3D):**
```
model.layers.N.mlp.switch_mlp.down_proj.weight:   [96, 5120, 1536]
model.layers.N.mlp.switch_mlp.gate_proj.weight:   [96, 1536, 5120]
model.layers.N.mlp.switch_mlp.up_proj.weight:     [96, 1536, 5120]
```

**Status:** Build was planned but abandoned. Architecture analysis and pre-flight completed.

### 4.5 Step 3.5 Flash

**Architecture identifier:** `Step3p5ForCausalLM` (custom, requires `trust_remote_code`)

Key architectural distinctions:
- **Smallest FFN ratio** of any model studied: 0.31x (expert intermediate=1280, hidden=4096)
- **Dual attention types** with different head counts: full=64Q, sliding=96Q
- **Novel `g_proj`** -- per-head attention gate: `[64, 4096]` (full) or `[96, 4096]` (sliding)
- **Sigmoid routing** with bias + scaling factor 3.0
- **Dense layers L0-L2** (standard MLP, intermediate=11264)
- **Partial rotary:** 0.5 for full attention, 1.0 for sliding
- **Different RoPE theta:** 5M for full attention, 10K for sliding
- **SwiGLU clamping** only in L43-L44 ([-7, 7])
- **Shared expert** always active (1280 intermediate)
- BF16 source format, text-only (no VL)

**Tensor naming (BF16 source / PyTorch):**
```
model.layers.N.self_attn.o_proj.weight     [4096, 8192] or [4096, 12288]
model.layers.N.self_attn.g_proj.weight     [64, 4096] or [96, 4096]  (NOVEL)
model.layers.N.moe.down_proj.weight        [173, 4096, 1280]  (3D batched)
model.layers.N.moe.gate.weight             [173, 4096]
model.layers.N.moe.router_bias             [173] (F32)
model.layers.N.share_expert.down_proj.weight  [4096, 1280]
```

**MLX key remapping:**
```
moe.down_proj     -> mlp.switch_mlp.down_proj   (3D batched)
moe.gate          -> mlp.gate.gate               (router, Q8)
moe.router_bias   -> mlp.gate.router_bias        (F32)
share_expert      -> mlp.share_expert
```

**Quantization finding:** Despite the 0.31x FFN ratio, surgery at s=4.0 did NOT cause the expected coherence cliff. Fresh `mlx_lm.convert` Q4/Q6/Q8 from BF16 surgery output correctly preserved the surgery signal -- contradicting the Qwen 397B VL finding where fresh quantization destroyed surgery signal.

### 4.6 Nemotron 3 Super 120B

**Architecture identifier:** `NemotronHForCausalLM` (custom, requires `trust_remote_code`)

Key architectural distinctions:
- **Three completely separate layer types** (not combined like Qwen): Mamba-2 SSM (M), MoE FFN (E), Dense Attention (*)
- **LatentMoE routing** -- tokens compressed from 4096 to 1024 before expert routing
- **relu2 activation** (squared ReLU: `max(0,x)^2`) in all MLP layers
- **22 active experts** -- highest of any model studied
- **Only 8 attention layers** out of 88 total (9% -- lowest of any model)
- **Per-expert weights work in 1024-dim latent space**, not full 4096 hidden
- **Multi-token prediction (MTP)** head -- 1,040 extra tensors to strip
- BF16 source format, text-only (no VL)

**Layer pattern:** `MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*...`
Dense (*) layers at positions: L7, L16, L25, L36, L47, L58, L69, L78

**Tensor naming (post-MLX-conversion, retains `backbone` prefix):**

Mamba (M) layers -- 9 tensors each:
```
backbone.layers.N.mixer.A_log
backbone.layers.N.mixer.D
backbone.layers.N.mixer.conv1d.{weight,bias}
backbone.layers.N.mixer.dt_bias
backbone.layers.N.mixer.in_proj.weight
backbone.layers.N.mixer.norm.weight
backbone.layers.N.mixer.out_proj.weight
```

Dense Attention (*) layers -- 5 tensors each:
```
backbone.layers.N.mixer.q_proj.weight
backbone.layers.N.mixer.k_proj.weight
backbone.layers.N.mixer.v_proj.weight
backbone.layers.N.mixer.o_proj.weight
backbone.layers.N.norm.weight
```

MoE Expert (E) layers -- ~1031 tensors each:
```
backbone.layers.N.mixer.switch_mlp.fc1.weight    [512, 2688, 4096]  (up_proj stacked)
backbone.layers.N.mixer.switch_mlp.fc2.weight    [512, 4096, 2688]  (down_proj stacked)
backbone.layers.N.mixer.shared_experts.down_proj.weight    [4096, 5376]
backbone.layers.N.mixer.gate.weight              [512, 4096]
backbone.layers.N.mixer.gate.e_score_correction_bias
backbone.layers.N.mixer.fc1_latent_proj.weight   [1024, 4096]  (compression)
backbone.layers.N.mixer.fc2_latent_proj.weight   [4096, 1024]  (decompression)
```

**Critical LatentMoE finding:** Expert `down_proj` tensors are [1024, 2688] (latent space), NOT [4096, 2688]. A 4096-dim refusal vector cannot be applied to 1024-dim expert weights. Valid surgery targets must have 4096 as an output dimension: Dense o_proj, shared expert down_proj, fc2_latent_proj (decompression), and Mamba out_proj.

**Stock mlx_lm does NOT support LatentMoE.** The `NemotronHMoE` class creates `SwitchMLP(config.hidden_size=4096, ...)` but experts work in 1024-dim latent space. A local patch was required. Without the patch, the 1120 latent projection tensors are silently dropped, producing garbage output. All published models were made private after discovering this.

### 4.7 INTELLECT 3.1

Standard MoE transformer. 46 layers, 5120 hidden, 128 experts (8 active). No SSM, no sliding window, no thinking/CoT mechanism. Used as a baseline reference -- cleanest surgery results because no thinking mechanism to create loops.

---

## 5. Weight Naming Conventions

Six MoE weight naming patterns are supported by the CRACK surgeon:

| Architecture | Down Projection Pattern |
|---|---|
| Dense MLP | `model.layers.N.mlp.down_proj.weight` |
| DeepSeek/Qwen per-expert | `model.layers.N.mlp.experts.M.down_proj.weight` |
| Shared expert | `model.layers.N.mlp.shared_expert.down_proj.weight` |
| GPT OSS (batched 3D) | `model.layers.N.mlp.experts.down_proj.weight` |
| Mixtral | `model.layers.N.block_sparse_moe.experts.M.w2.weight` |
| MiniMax (switch_mlp) | `model.layers.N.block_sparse_moe.switch_mlp.down_proj.weight` |

**Nemotron uses a non-standard prefix:** `backbone.layers.N.mixer.*` instead of `model.layers.N.*`. This requires either key remapping during conversion or custom surgeon regex patterns.

**VL models add a prefix:** `language_model.model.layers.N.*` instead of `model.layers.N.*` for the language model weights. Vision tensors are under `model.visual.*`.

**3D batched expert tensors** (`[num_experts, out_dim, in_dim]`) are created by MLX `sanitize()` methods that stack per-expert 2D weights into a single 3D tensor per layer. The refusal vector applies identically to all experts via vectorized operations across the expert dimension.

---

## 6. Quantization Considerations by Architecture

### 6.1 Critical Precision Tensors (Must Stay High Precision)

Across all architectures, certain tensors must NOT be aggressively quantized:

| Tensor Type | Why | Models |
|---|---|---|
| Router/gate weights | Routing precision destroyed at Q2 | All MoE |
| `shared_expert_gate` | Controls shared expert mixing | Qwen |
| `embed_tokens`, `lm_head` | Embedding quality critical | All |
| `A_log`, `dt_bias` | SSM state dynamics | Qwen SSM, Nemotron Mamba |
| `conv1d` | SSM convolution kernel | Qwen SSM, Nemotron Mamba |
| `e_score_correction_bias` | Routing bias correction | GLM, Nemotron |
| `self_attn.sinks` | Attention anchors | GPT OSS |
| Attention biases | Input bias terms | GPT OSS, GLM |
| `fc1/fc2_latent_proj` | MoE latent compression | Nemotron |

For Q2 quantization: the CRACK streaming quantizer keeps these 122 critical tensors at bf16 via an `is_critical_precision()` check. Q3 (8 vals/group) survives with quantized gates; Q2 (4 vals/group) does NOT.

### 6.2 mxfp4 vs Affine Quantization

**mxfp4 (GPT OSS 120B native):**
- No `.biases` tensors -- only `.weight` and `.scales`
- `group_size=32, bits=4`
- Packed as uint32 (8 elements per word), scales as uint8
- Native to the model (weights trained in mxfp4)
- Dequant -> modify -> requant round-trip is lossy because the new quantization grid doesn't match the training grid

**Affine quantization (most MLX models):**
- Three components per tensor: `.weight` (packed), `.scales`, `.biases`
- Standard group sizes: 64 (most models), 128 (MiniMax Q6/Q8 for speed)
- After surgery, all three (weight + scales + biases) must be updated atomically

### 6.3 Group Size Impact on Speed

MiniMax MoE models with many experts (154-192) showed 15-25% speed regression with `group_size=64` at Q6/Q8 due to `gather_qmm` kernel cache pressure. At `group_size=128`, Q4/Q6/Q8 achieve identical compute speeds and max hardware bandwidth (~40-53 tok/s on M3 Ultra).

Rule: **Use `group_size=128` for MoE models with >128 experts at Q6/Q8.**

### 6.4 Per-Tensor Quantization Overrides

Adding per-tensor quantization overrides to `config.json` (e.g., keeping surgery tensors at Q4 while rest is Q5) causes mlx_lm to use a slow `class_predicate` loading path, dropping speed from 32+ tok/s to ~9.4 tok/s. Solution: uniform `Q_BITS` for all tensors, apply surgery before quantization.

### 6.5 Q6/Q8 Need Higher Surgery Strength

Cross-model pattern: finer quantization (Q6/Q8) preserves the original safety infrastructure better, requiring approximately 1.5x the Q4 surgery strength.

| Model | Q4 Strength | Q6/Q8 Strength | Multiplier |
|---|---|---|---|
| Step 3.5 Flash 121B | s=4.0 | s=6.0 | 1.5x |
| Step 3.5 Flash 149B | s=4.0 | s=6.20 | 1.55x |
| MiniMax 172B | s=2.50 | s=3.00 | 1.2x |
| Qwen 122B CRACK-X Q4 | s=7.50 | Q6: s=8.60, Q8: s=7.90 | 1.05-1.15x |

Exception: Nemotron Q6 needed LOWER strength than Q4 (s=1.00 vs s=1.10) because the Q6 model was built from Q4 (re-quantization), not from FP16 source. The surgery signal was already baked into Q4 weights, and Q6's finer grid preserved it more precisely.

### 6.6 FP16 Overflow in BF16-Trained Models

GPT OSS 120B was trained in BF16 (range up to 3.4e38). Converting to FP16 (max 65,504) causes overflow in deep layers where residual stream activations grow exponentially. Layer-by-layer trace showed activation overflow at L30 (value=Inf) producing NaN from L31 onward. Solution: serve all non-mxfp4 tensors as BF16, not FP16.

### 6.7 Surgery-Before-Quantization vs Post-Quantization Surgery

Two approaches with different tradeoffs:

**Surgery-before-quantization** (`streaming_quantizer.py` approach):
- Apply CRACK formula to FP16/BF16 weights, then quantize at target bit width
- No mixed quantization needed, no per-tensor overrides
- Surgery signal permanently baked in regardless of quantization level
- Used for Qwen 397B Q2/Q3/Q4/Q5 builds

**Post-quantization surgery** (binary patch approach):
- Dequantize target tensors, modify, requantize with fresh scales/biases
- Preserves original quantization grid for unmodified tensors
- Risk: requantization grid doesn't match original, can destroy surgery signal
- Used for Qwen 397B Q4 CRACK REAP (binary patch of proven Q4 bytes)

**Cross-quantization** (MiniMax Q6/Q8 solution):
- Dequantize Q4 CRACK weights (proven surgery) to FP32
- Requantize at Q6/Q8 precision
- Binary patch into Q6/Q8 base shards
- Only working approach for MiniMax Q6/Q8 where direct toolkit surgery at Q6/Q8 produced gibberish

### 6.8 GPU Metal Timeout on Large 3D Expert Tensors

Quantizing 3D batched expert tensors (e.g., `[512, 4096, 1024]`) on GPU causes Metal timeout. Solution: hybrid device switching -- `mx.set_default_device(mx.cpu)` for load and save, switch to GPU only for individual tensor quantization operations. Proven on Qwen 397B and Nemotron 120B.

### 6.9 Precision Floor for Hybrid Architectures

Nemotron's hybrid Mamba+MoE+Attention architecture has a precision floor above Q2. Even with critical tensors (gates, embeddings, A_log) kept at bf16, Mamba parameters (conv1d, dt_bias, D) and MoE latent projections cannot survive Q2 quantization. Q2 was abandoned for Nemotron.

Qwen 3.5 Q2 also showed degraded thinking (surgery + quantization compounding). The chat template for Q2 defaults thinking OFF.

### 6.10 `mx.save_safetensors()` Speed Corruption

Using `mx.save_safetensors()` to save modified safetensor files strips metadata and reorders tensors, causing catastrophic speed regression (36.8 tok/s -> 2.3 tok/s on Qwen 397B). The correct approach is binary patching: read original shard bytes, parse safetensors header, replace ONLY modified tensor data regions, write back with original header unchanged.

### 6.11 RepE (PCA) vs Mean-Diff for Non-Standard Architectures

Standard CRACK uses mean-diff vectors (average of harmful - harmless activations). On Nemotron's LatentMoE architecture, the mean-diff was orthogonal to the actual dominant safety direction (PC1) in critical middle layers. RepE PCA extraction (SVD on difference matrix) found the correct direction, achieving 12/12 compliance where standard CRACK maxed out at 1-2/8.

This matters for quantization because the surgery direction vector affects how much each weight tensor changes, which in turn affects quantization grid selection.

---

## 7. Cross-Architecture Comparison Tables

### 7.1 Routing Mechanisms

| Model | Routing | Top-k | Notes |
|---|---|---|---|
| Qwen 3.5 (all MoE) | Softmax | 4-10 | Standard normalized |
| MiniMax M2.5 | Sigmoid + bias | 8 | Non-normalized, bias correction |
| GPT OSS 120B | Softmax | 4 | Standard |
| INTELLECT 3.1 | Softmax | 8 | Standard |
| Step 3.5 Flash | Sigmoid + bias + scale 3.0 | 8 | Non-normalized |
| GLM-4.7 | Sigmoid + bias | 8 | Same mechanism as MiniMax |
| Nemotron Super 120B | Sigmoid (latent) | 22 | Routes in 1024-dim, highest k |

### 7.2 SSM Bypass Channels

| Model | SSM Type | SSM Layers | % of Total | Bypass Risk |
|---|---|---|---|---|
| Qwen 3.5 (all) | GatedDeltaNet | 75% | High | Recurrent state invisible to weight surgery |
| Nemotron Super 120B | Mamba-2 | 40/88 (45%) | High | Selective scanning, separate from MoE |
| MiniMax M2.5 | None | 0% | None | Pure attention |
| GPT OSS 120B | None | 0% | None | Pure attention |
| Step 3.5 Flash | None | 0% | None | Pure attention |
| GLM-4.7 | None | 0% | None | Pure attention |
| INTELLECT 3.1 | None | 0% | None | Pure attention |

### 7.3 FFN Expansion Ratios

| Model | FFN Ratio | Expert Intermediate | Hidden | Impact |
|---|---|---|---|---|
| Step 3.5 Flash | 0.31x | 1280 | 4096 | Smallest ratio, surprisingly robust |
| GLM-4.7 | 0.3x (MoE) / 2.4x (dense) | 1536 / 12288 | 5120 | Split dense/MoE |
| MiniMax M2.5 | 0.5x | 1536 | 3072 | Small experts |
| Nemotron Super | 0.66x (expert) / 1.3x (shared) | 2688 / 5376 | 4096 | Latent bottleneck |
| GPT OSS 120B | 1.0x | 2880 | 2880 | ZERO redundancy, razor-thin coherence window |
| Qwen 3.5 122B/397B | ~0.25x (per-expert) | 1024 | 3072/4096 | Many small experts |
| Qwen 3.5 4B/9B/27B | ~3-4x (dense) | 6912-17408 | 2560-5120 | Standard dense |
| INTELLECT 3.1 | ~4x | ~20480 | 5120 | Standard |

Lower FFN ratio = less redundancy = more sensitive to weight modification = narrower safe strength range for surgery/quantization.

### 7.4 Special Mechanisms by Model

| Feature | Models That Have It |
|---|---|
| Attention sinks (learned) | GPT OSS 120B |
| Attention biases (q/k/v) | GPT OSS 120B, GLM-4.7 |
| QK normalization | Qwen 3.5 122B+, GLM-4.7 |
| Partial RoPE (50%) | GLM-4.7 |
| SwiGLU clamping [-7,7] | GPT OSS 120B, Step 3.5 Flash L43-44 |
| Per-head attention gate (g_proj) | Step 3.5 Flash |
| LatentMoE (compressed routing) | Nemotron Super 120B |
| Multi-token prediction (MTP) | Qwen 3.5, Nemotron Super 120B |
| Dual RoPE theta (per attn type) | Step 3.5 Flash |
| Early-fusion VL | All Qwen 3.5 |
| Harmony channels (structured CoT) | GPT OSS 120B |

---

## 8. Lessons for Quantization Engine Design

### 8.1 Tensor Detection Must Be Architecture-Aware

A quantization engine must handle at minimum six different weight naming conventions for MoE expert down projections, plus non-standard prefixes (`backbone.layers.N.mixer.*` for Nemotron, `language_model.model.layers.N.*` for VL models).

### 8.2 3D Batched Expert Tensors Need Special Handling

After MLX `sanitize()` stacking, expert weights are 3D: `[num_experts, out_dim, in_dim]`. Operations (abliteration, quantization analysis) must be vectorized across the expert dimension (axis 0). GPU Metal may timeout on very large 3D tensors (512 experts) -- CPU fallback needed.

### 8.3 mxfp4 Is Structurally Different from Affine

mxfp4 has no `.biases` tensor. A quantization engine that assumes weight + scales + biases will crash on mxfp4 models. Detection must check the quantization `mode` in config.json.

### 8.4 Critical Precision Tensors Must Be Detected Per-Architecture

Router gates, embeddings, SSM parameters, attention sinks, latent projections, and routing biases all need to stay at high precision. The set of critical tensors varies by architecture. A general-purpose function like `is_critical_precision(tensor_name)` must handle all conventions.

### 8.5 BF16 vs FP16 Matters

Models trained in BF16 (GPT OSS, Nemotron) may overflow FP16 range in deep layers. The quantization engine should preserve the original dtype for non-quantized tensors (scales, biases, norms) and never silently convert BF16 to FP16.

### 8.6 Group Size Interacts with Expert Count

Standard `group_size=64` causes speed regression on MoE models with many experts (150+) at Q6/Q8 due to `gather_qmm` kernel cache pressure. Use `group_size=128` for high-expert-count models at Q6+.

### 8.7 Per-Tensor Config Causes Speed Regression

Adding per-tensor quantization overrides to config.json forces mlx_lm into a slow `class_predicate` loading path. Uniform quantization config is mandatory for full speed.

### 8.8 Vision Tensors Must Be Stripped or Preserved Deliberately

Qwen 3.5 VL models have 333 vision tensors in the final shards. `mlx_lm.convert` strips them; `mlx_vlm.convert` preserves them. Using the wrong converter produces a broken model. MTP tensors (785+ tensors) must always be stripped for inference.

### 8.9 Tokenizer Corruption During Conversion

`mlx_lm.convert` regenerates `tokenizer.json`, which can strip NFC normalization and custom pre-tokenizer patterns (confirmed on MiniMax). Always verify tokenizer integrity after conversion, or copy known-good tokenizer files.

### 8.10 Quantization-Strength Interaction Is Non-Monotonic

At certain quantization levels, the Q4 grid aligns differently with modified weights. A strength of s=0.95 may produce worse results than s=0.90 or s=1.10 because the quantized values snap to different grid points. Fine-grained sweeps at 0.05 increments are necessary to find the sweet spot.

---

## Sources

All information is from files under `/Users/eric/CRACK_abliteration/`:

- `README.md` -- CRACK tool architecture and math
- `CLAUDE.md` -- Agent onboarding, MoE support, hybrid SSM/FA details, model matrix
- `HANDOFF_NOTES.md` -- MiniMax pipeline, cross-quantization, tokenizer bugs
- `docs/MEMORY_INDEX.md` -- Architecture quick reference, surgery strengths, speed reference
- `docs/glm47/ARCHITECTURE_ANALYSIS.md` -- GLM-4.7 architecture and cross-model comparison
- `docs/glm47/CRACK_PREFLIGHT.md` -- GLM-4.7 tensor structure and strategy analysis
- `docs/gpt_oss_120b/GPT_OSS_120B_ANALYSIS.md` -- GPT OSS architecture, mxfp4, FP16 overflow
- `docs/step35_flash/FINDINGS.md` -- Step 3.5 Flash architecture, dual attention, probe/surgery
- `docs/minimax_m25/FINDINGS.md` -- MiniMax tokenizer corruption, inference requirements
- `docs/minimax_m25/MINIMAX_CRACK_NATIVE_EXPERIMENTS.md` -- MiniMax native surgery, dimension bugs
- `docs/nemotron_super_120b/FINDINGS.md` -- Nemotron architecture, LatentMoE, RepE breakthrough
- `docs/nemotron_super_120b/CRACK_PLAN.md` -- Nemotron build plan, weight naming
- `docs/nemotron_super_120b/BUILD_LOG.md` -- Nemotron conversion, probe, LatentMoE bug, post-mortem
- `docs/qwen35_122b/122b-architecture.md` -- Qwen 122B architecture vs 397B
- `docs/qwen35_27b/PIPELINE.md` -- Qwen 27B dense SSM+FA architecture
- `docs/research/MOE_SOTA_ANALYSIS.md` -- MoE safety bypass state of the art
- `research/122b-qwen3.5/docs/ARCHITECTURE.md` -- Qwen 122B tensor shapes
