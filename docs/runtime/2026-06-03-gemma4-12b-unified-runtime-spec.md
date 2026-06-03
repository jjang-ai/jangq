# Gemma 4 12B Unified — Runtime Integration Specification

**Date:** 2026-06-03
**Model:** `google/gemma-4-12B-it`
**Audience:** runtime implementers for `/Users/eric/vmlx` (Python) and `/Users/eric/vmlx-swift` (Swift).
**Status:** authoritative contract. Every non-obvious claim is cited `file:field` or `file:line`. Facts that could not be verified against a real artifact are explicitly marked **UNVERIFIED**.

## Source artifacts used for verification

| Artifact | Path |
| --- | --- |
| Config | `/Users/eric/models/google/gemma-4-12B-it/config.json` |
| Generation config | `/Users/eric/models/google/gemma-4-12B-it/generation_config.json` |
| Processor config | `/Users/eric/models/google/gemma-4-12B-it/processor_config.json` |
| Tokenizer config | `/Users/eric/models/google/gemma-4-12B-it/tokenizer_config.json` |
| Chat template | `/Users/eric/models/google/gemma-4-12B-it/chat_template.jinja` |
| Tokenizer | `/Users/eric/models/google/gemma-4-12B-it/tokenizer.json` |
| Model card | `/Users/eric/models/google/gemma-4-12B-it/README.md` |
| Weights (header inspected) | `/Users/eric/models/google/gemma-4-12B-it/model.safetensors` (677 tensors, single shard) |
| MLX VLM ref | `jang-tools/.venv/.../mlx_vlm/models/gemma4/{gemma4,language,vision,audio,config,rope_utils,processing_gemma4}.py` |
| MLX LM ref | `jang-tools/.venv/.../mlx_lm/models/{gemma4_text,gemma4}.py`, `tool_parsers/gemma4.py` |
| JANG converter | `jang-tools/jang_tools/convert_gemma4_mxfp.py` |
| Capability map | `jang-tools/jang_tools/capabilities.py:74-77` |

All venv paths are under `/Users/eric/jang/jang-tools/.venv/lib/python3.11/site-packages/`.

---

## 1. Identity & high-level summary

| Property | Value | Source |
| --- | --- | --- |
| Architecture | `Gemma4UnifiedForConditionalGeneration` | `config.json:architectures` |
| Top model_type | `gemma4_unified` | `config.json:model_type` |
| Sub-configs | `gemma4_unified_text` / `gemma4_unified_vision` / `gemma4_unified_audio` | `config.json:text_config.model_type`, `vision_config.model_type`, `audio_config.model_type` |
| transformers version | `5.10.0.dev0` | `config.json:transformers_version` |
| Total params | **11,959,730,224 (~11.96B)** counting tied embedding once | computed from safetensors header |
| Dtype | bf16 | `config.json:dtype`; all 677 tensors `BF16` in header |
| Density | **DENSE** (not MoE) | `text_config.enable_moe_block=false`, `num_experts=null`, `moe_intermediate_size=null` |
| Modalities | text + image + audio + video (**omni-modal, encoder-free / early-fusion**) | README:25,68; `config.json` has vision+audio sub-configs |
| Context length | **131072 (128K)** for runtime purposes | `text_config.max_position_embeddings=131072` |
| License | Apache 2.0 (Gemma 4 license link) | README frontmatter `license: apache-2.0` |
| Gating | base model `google/gemma-4-12B`; standard Gemma license click-through | README frontmatter `base_model` |
| `pipeline_tag` | `any-to-any` | README frontmatter |

> **CONTEXT-LENGTH CONFLICT (flag for integrators).** The model card text and the dense-model table both claim **256K** context for the 12B Unified (README:27, README:60, README:78, README:120). But `config.json:text_config.max_position_embeddings` is **131072 (128K)**, and the README's own "Increased Context Window" bullet says *"The small models feature a 128K context window, while the medium models support 256K"* (README:41) — implying 12B may be a "small" model. **Use the config value (128K) as the runtime maximum** unless Google ships a corrected config; treat README 256K as **UNVERIFIED** against the shipped config. RoPE theta for global layers is 1e6, consistent with long context either way.

---

## 2. Text backbone spec

All values from `config.json:text_config` unless noted. Module layout verified against the safetensors header and `mlx_vlm/models/gemma4/language.py`.

| Field | Value |
| --- | --- |
| `hidden_size` | 3840 |
| `num_hidden_layers` | 48 |
| `intermediate_size` | 15360 (GeGLU MLP) |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 8 (sliding layers, GQA) |
| `num_global_key_value_heads` | 1 (full layers, MQA) |
| `head_dim` | 256 (sliding layers) |
| `global_head_dim` | 512 (full layers) |
| `vocab_size` | 262144 |
| `tie_word_embeddings` | **true** (no `lm_head` tensor) |
| `final_logit_softcapping` | 30.0 |
| `rms_norm_eps` | 1e-6 |
| `hidden_activation` | `gelu_pytorch_tanh` (GeGLU; `nn.gelu_approx` in MLX) |
| `attention_bias` | false (all linears bias-free) |
| `sliding_window` | 1024 |
| `attention_k_eq_v` | **true** (full layers derive V from K) |
| `use_bidirectional_attention` | `"vision"` |
| `enable_moe_block` | false |
| `num_kv_shared_layers` | 0 (KV-sharing path inactive on 12B) |
| `hidden_size_per_layer_input` | 0 (Per-Layer-Embedding path inactive on 12B) |
| `use_double_wide_mlp` | false |

> **12B does NOT use the small-model machinery.** `mlx_vlm`/`mlx_lm` gemma4 code carries Per-Layer-Embedding (PLE), KV-sharing, double-wide MLP, and MoE branches for the E2B/E4B/26B variants. On 12B these are **all disabled** because `hidden_size_per_layer_input=0`, `num_kv_shared_layers=0`, `use_double_wide_mlp=false`, `enable_moe_block=false`. Runtimes targeting only 12B may skip those branches entirely, but must still parse the config flags to confirm they are zero/false.

### 2.1 Per-decoder-block module inventory (verified from safetensors header)

Tensor key pattern (HF, pre-sanitize): `model.language_model.layers.{i}.<sub>`.

| Submodule | Sliding layer shape | Full layer shape (i ∈ {5,11,17,23,29,35,41,47}) |
| --- | --- | --- |
| `self_attn.q_proj.weight` | `[4096, 3840]` (16×256) | `[8192, 3840]` (16×512) |
| `self_attn.k_proj.weight` | `[2048, 3840]` (8×256) | `[512, 3840]` (1×512) |
| `self_attn.v_proj.weight` | `[2048, 3840]` (8×256) | **ABSENT** |
| `self_attn.o_proj.weight` | `[3840, 4096]` | `[3840, 8192]` |
| `self_attn.q_norm.weight` | `[256]` | `[512]` |
| `self_attn.k_norm.weight` | `[256]` | `[512]` |
| `self_attn.v_norm` | weightless (no tensor) | weightless (no tensor) |
| `input_layernorm.weight` | `[3840]` | `[3840]` |
| `post_attention_layernorm.weight` | `[3840]` | `[3840]` |
| `pre_feedforward_layernorm.weight` | `[3840]` | `[3840]` |
| `post_feedforward_layernorm.weight` | `[3840]` | `[3840]` |
| `mlp.gate_proj.weight` | `[15360, 3840]` | same |
| `mlp.up_proj.weight` | `[15360, 3840]` | same |
| `mlp.down_proj.weight` | `[3840, 15360]` | same |
| `layer_scalar` | `[1]` | `[1]` |

Verified: exactly **40 of 48** layers carry `v_proj`; the 8 missing are `[5,11,17,23,29,35,41,47]` — precisely the `full_attention` layers (computed from header).

Non-layer text tensors:
- `model.language_model.embed_tokens.weight` `[262144, 3840]` (tied to output)
- `model.language_model.norm.weight` `[3840]` (final RMSNorm)

### 2.2 Four LayerNorms per block + q/k norm + layer_scalar

Per-block normalization order (verified `language.py:316-361`, identical in `gemma4_text.py:342-387`):

```
residual = x
h = input_layernorm(x)                  # pre-attention norm
h = self_attn(h)                         # see §3
h = post_attention_layernorm(h)          # post-attention norm
h = residual + h
residual = h
h = pre_feedforward_layernorm(h)         # pre-MLP norm
h = mlp(h)                               # GeGLU
h = post_feedforward_layernorm(h)        # post-MLP norm
h = residual + h
h = h * layer_scalar                     # per-layer learned scalar (shape [1])
```

- This is the Gemma "sandwich norm" (both pre- and post- on attention and FFN), now **4 norms per block**.
- `q_norm` and `k_norm` are RMSNorm applied per-head over `head_dim` (256 sliding / 512 full); `v_norm` is **weightless** RMSNorm (no learnable scale).
- `layer_scalar` is a single learned scalar `[1]` applied multiplicatively to the block output (`h = h * layer_scalar`, `language.py:360-361`). One per layer; must be loaded and applied — it is **not** an optional decoration.

---

## 3. Attention in depth

### 3.1 Layer schedule (verified `config.json:text_config.layer_types`, 48 entries)

Pattern: 5× `sliding_attention` then 1× `full_attention`, repeated 8 times.

| Layer idx | Type |
| --- | --- |
| 0,1,2,3,4 | sliding |
| **5** | **full** |
| 6,7,8,9,10 | sliding |
| **11** | **full** |
| 12–16 | sliding |
| **17** | **full** |
| 18–22 | sliding |
| **23** | **full** |
| 24–28 | sliding |
| **29** | **full** |
| 30–34 | sliding |
| **35** | **full** |
| 36–40 | sliding |
| **41** | **full** |
| 42–46 | sliding |
| **47** | **full** |

Full-attention layer indices: **[5, 11, 17, 23, 29, 35, 41, 47]** (8 layers; final layer is always global, README:51).

### 3.2 Per-layer-type attention parameters

| Property | Sliding layers (40) | Full layers (8) |
| --- | --- | --- |
| Attention type | local sliding window | global full attention |
| Window | 1024 (`sliding_window`) | unbounded |
| Q heads | 16 | 16 |
| KV heads | 8 (**GQA**) | 1 (**MQA**) |
| head_dim | 256 | 512 (`global_head_dim`) |
| K=V (k_eq_v) | no | **yes** — V derived from K, no `v_proj` |
| RoPE | default, θ=10000 | proportional (p-RoPE), θ=1e6, `partial_rotary_factor=0.25` |
| q/k/v norm | q_norm,k_norm (scale), v_norm (no scale) | same; q_norm/k_norm dim 512 |
| attention scale | **`scale = 1.0`** (see §3.5) | **`scale = 1.0`** |

### 3.3 `attention_k_eq_v` mechanics (critical for loaders)

On full-attention layers (`config.json:text_config.attention_k_eq_v=true`):
- There is **no `v_proj` weight**. A loader that assumes every attention block has q/k/v/o **will fail** on layers [5,11,17,23,29,35,41,47].
- At runtime, V is set to the **raw K projection output, BEFORE `k_norm`** (`language.py:215-221`): `keys = k_proj(x); values = keys` (the assignment happens before `k_norm` is applied to `keys`).
- `keys` then get `k_norm` + RoPE; `values` get the weightless `v_norm` and **no RoPE** (`language.py:225-230`).
- `num_global_key_value_heads=1` → MQA: one shared KV head broadcast across the 16 query heads.

Loader rule: build the attention module's V path conditionally on layer type. Sliding layers: load `v_proj`. Full layers: skip `v_proj`, alias `values := k_proj(x)` pre-norm.

### 3.4 Exact attention forward order (from `mlx_vlm/models/gemma4/language.py:199-243`)

```
B, L, _ = x.shape
queries = q_proj(x).reshape(B, L, n_heads, head_dim)
queries = q_norm(queries)                      # q_norm BEFORE transpose/rope
# --- KV path (skipped when shared_kv provided; not used on 12B) ---
keys = k_proj(x).reshape(B, L, n_kv_heads, head_dim)
values = keys                  if use_k_eq_v   # full layers: V from raw K
       = v_proj(x)...          otherwise       # sliding layers
offset = cache.offset if cache else 0
keys = k_norm(keys); keys = keys.transpose(0,2,1,3); keys = rope(keys, offset)
values = v_norm(values); values = values.transpose(0,2,1,3)   # v: NO rope
if cache: keys, values = cache.update_and_fetch(keys, values)
# --- query rope (after KV path so offset is set) ---
queries = queries.transpose(0,2,1,3); queries = rope(queries, offset)
out = sdpa(queries, keys, values, scale=1.0, mask=mask)
out = out.transpose(0,2,1,3).reshape(B, L, -1)
return o_proj(out)
```

Note ordering nuance: `q_norm` is applied to the `[B,L,H,D]` layout **before** the transpose to `[B,H,L,D]`; RoPE is applied **after** transpose. K follows the same: `k_norm` on `[B,L,H,D]`, then transpose, then RoPE. V gets `v_norm` then transpose, **never** RoPE. `mlx_lm/models/gemma4_text.py:231-276` is identical except it calls `cache.update_and_fetch` after the query-rope block (functionally equivalent for non-shared layers).

### 3.5 Attention scale — `scale = 1.0` (verify against your SDPA)

Both reference implementations set `self.scale = 1.0` and pass it straight into SDPA (`language.py:164,238-240`; `gemma4_text.py:203,271-272`). They do **not** use the conventional `1/sqrt(head_dim)`. The QK-norm (RMSNorm on q and k) plus learned scales is what controls logit magnitude, so the explicit softmax scale is unity.

**Integration rule:** pass `scale=1.0` to the fused SDPA for Gemma 4. Do **not** auto-apply `head_dim**-0.5`. (UNVERIFIED whether HF transformers folds an equivalent factor into the q/k norm weights — but matching the MLX reference, which is byte-loaded from these weights and produces correct output, means using `scale=1.0`.) This is identical for sliding (head_dim 256) and full (head_dim 512) layers.

### 3.6 Masks

`_make_masks` builds **one mask per layer type** and reuses it (`language.py:448-461`):
- full layers → causal mask (`create_attention_mask(h, c)`)
- sliding layers → causal + window mask (`create_attention_mask(h, c, window_size=1024)`)

### 3.7 Bidirectional vision attention

`use_bidirectional_attention="vision"` (`config.json:text_config`). Within a span of image (and video) soft-token positions, attention is **bidirectional** (not causal) — image tokens attend to each other both ways, while text remains causal. This is an early-fusion property: the decoder itself encodes the image patches, so the patch span needs full intra-span visibility.

> **GAP / UNVERIFIED in the MLX reference.** The `mlx_vlm` gemma4 text path (`language.py:_make_masks`) builds only causal/sliding masks and does **not** construct a bidirectional sub-mask over image/audio token spans. The bidirectional behavior is declared in config but the reference text decoder does not appear to implement it for the early-fusion 12B path. Runtime authors who need exact parity must build a custom mask: causal everywhere, except set the lower-triangle-blocking to 0 within each contiguous run of `image_token_id`/`video_token_id` (and per the config flag, NOT audio — flag is literally `"vision"`). Treat the precise span boundaries (inclusive of `boi`/`eoi`?) as **UNVERIFIED**; confirm against HF `Gemma4Unified` modeling code before shipping vision.

### 3.8 Final logit softcapping

After the tied-embedding output projection, logits are softcapped: `logits = tanh(logits / 30.0) * 30.0` (`language.py:600-601`, `final_logit_softcapping=30.0`). There is **no** per-layer attention-logit softcap in Gemma 4 (unlike Gemma 2). Only the final logits are capped.

---

## 4. RoPE — dual scheme

`config.json:text_config.rope_parameters` is a **dict keyed by layer type**, not a flat config:

```json
"rope_parameters": {
  "full_attention":    {"rope_type":"proportional", "partial_rotary_factor":0.25, "rope_theta":1000000.0},
  "sliding_attention": {"rope_type":"default",                                   "rope_theta":10000.0}
}
```

### 4.1 Sliding layers — default RoPE, θ=10000

Standard `nn.RoPE(head_dim=256, base=10000, traditional=False)` over the full 256-dim head (`rope_utils.py:109-110`).

### 4.2 Full layers — proportional p-RoPE, θ=1e6, partial_rotary_factor=0.25

Implemented by `ProportionalRoPE` (`rope_utils.py:9-83`). Semantics verified from the code:

- `rotated_dims = 2 * int(partial_rotary_factor * dims // 2)`. For `dims=512, factor=0.25` → `int(0.25*512//2)=64` angles → **`rotated_dims = 128`**. So only **128 of the 512** head dims are rotated (25%); the remaining 384 dims pass through unrotated.
- **"proportional" = frequencies computed relative to the FULL head dim, not the rotated subset** (`rope_utils.py:36`: `exponents = arange(0, rotated_dims, 2)/dims` — denominator is `dims=512`, not `rotated_dims`). This is the load-bearing difference from a plain partial-rotary RoPE, which would divide by `rotated_dims`.
- Rotation layout follows HF `rotate_half`: the head is split into left/right halves; the first `rotated_dims//2 = 64` elements of each half are gathered, rotated together via `mx.fast.rope`, and scattered back (`rope_utils.py:45-83`). The unrotated tail of each half is left untouched.
- `factor=1.0` (no length scaling beyond the proportional frequency base). `base = rope_theta = 1e6`.

> **Implementation trap.** A naive "partial rotary" that computes inverse frequencies over `arange(0, rotated_dims)/rotated_dims` will produce wrong angles. Gemma 4 p-RoPE divides by the **full** `dims`. Verify your Swift/Python RoPE divides the exponent by 512 (the full global head dim), applies rotation only to the first 64 dims of each 256-dim half, and leaves the rest identity.

---

## 5. Normalization conventions — MIGRATION TRAP

> **🚨 RMSNorm changed from Gemma 1/2/3.** Gemma 1–3 used `x * (1 + w)` (`scale_shift = 1`, the "+1 offset"). **Gemma 4 uses stock RMSNorm `x * w` (`scale_shift = 0`, NO +1).**

Verified:
- `mlx_vlm/models/gemma4/language.py:6` imports `from mlx.nn import RMSNorm` (stock, no offset) and uses it directly for `input_layernorm`, `post_attention_layernorm`, `pre/post_feedforward_layernorm`, `q_norm`, `k_norm`, final `norm`.
- `RMSNormZeroShift` docstring (`language.py:34-35`): *"Gemma4 RMSNorm with scale_shift=0.0 (weight used directly, no +1 offset)."*
- `v_norm` = `RMSNormNoScale` (`language.py:23-31`): `mx.fast.rms_norm(x, None, eps)` — weightless.
- The JANG converter passes norm weights through unchanged and comments the trap (`convert_gemma4_mxfp.py:17-20,186-188`): *"Gemma 1/2/3 used x*(1+w) ... Gemma 4 uses stock nn.RMSNorm => x*w. So norm weights are passed through as-is; we do NOT add 1.0."*

**Integration rule:** any code path that previously added 1.0 to Gemma norm weights (or used a `gemma_rms_norm` kernel with `scale_shift=1`) **must not** do so for Gemma 4. Use a plain RMSNorm. Getting this wrong yields a model that loads but produces garbage.

### 5.1 Embedding scale

Token embeddings are scaled by `embed_scale = hidden_size**0.5 = sqrt(3840) ≈ 61.97` immediately after lookup (`language.py:376,477`; `gemma4_text.py:402,528`). This is applied to BOTH the text-token path and the multimodal `inputs_embeds` path (`gemma4.py:85-86`). Multimodal soft-token features are scattered **after** this scaling (they carry their own projection scale internally).

---

## 6. Multimodal — omni-modal, encoder-free, early-fusion

### 6.1 The core architectural fact

12B Unified is **encoder-free**: there is **no standalone ViT and no audio conformer** in the checkpoint. Image patches and audio frames are linearly projected directly into the shared decoder embedding space; the decoder itself does the cross-modal encoding (README:25,68). This is what "Unified" means.

### 6.2 Multimodal tensors actually present (verified from safetensors header)

These are the **only** non-text, non-decoder tensors in the checkpoint:

| Tensor | Shape | Role |
| --- | --- | --- |
| `model.vision_embedder.patch_dense.weight` | `[3840, 6912]` | linear patch projection (6912 = 3 × 48²; effective patch 48 = patch_size 16 × pooling 3) |
| `model.vision_embedder.patch_dense.bias` | `[3840]` | |
| `model.vision_embedder.patch_ln1.weight` / `.bias` | `[6912]` | pre-projection LayerNorm (over flattened patch) |
| `model.vision_embedder.patch_ln2.weight` / `.bias` | `[3840]` | post-projection LayerNorm |
| `model.vision_embedder.pos_embedding` | `[1120, 2, 3840]` | learned positional embedding; 1120 = max soft-token budget, dim 2 = (x,y) axes |
| `model.vision_embedder.pos_norm.weight` / `.bias` | `[3840]` | positional LayerNorm |
| `model.embed_vision.embedding_projection.weight` | `[3840, 3840]` | vision→decoder projection |
| `model.embed_audio.embedding_projection.weight` | `[3840, 640]` | audio→decoder projection (audio embed dim 640) |

Note these `vision_embedder.*` tensors use **LayerNorm with bias** (patch_ln1/ln2/pos_norm all have `.bias`), unlike the decoder's bias-free RMSNorm.

### 6.3 GAP: mlx_vlm 0.5.0 expects a full ViT/Conformer that 12B does NOT ship

This is the most important multimodal finding. The released 12B checkpoint and the `mlx_vlm` gemma4 model **disagree on the multimodal weight layout**:

- `mlx_vlm/models/gemma4/vision.py:392-411` (`VisionModel`) expects weights under `vision_tower.patch_embedder.*` and `vision_tower.encoder.layers.{0..15}.*` — a **full 16-layer SigLIP-style ViT** with `q/k/v/o_proj`, MLPs, 4 norms per block, a `patch_embedder.input_proj` (`[hidden, 3*patch²]`) and `position_embedding_table`. None of these exist in the 12B checkpoint (verified: 0 `vision_tower` tensors, 0 `audio_tower` tensors).
- `mlx_vlm/models/gemma4/audio.py:460` (`AudioEncoder`) expects a full **Conformer** stack (conv subsampling `SSCPConvBlock`, `ConformerBlock` × N, relative-position attention). Also absent.
- `mlx_vlm/models/gemma4/gemma4.py:48-65` constructs `self.vision_tower = VisionModel(...)` and `self.audio_tower = AudioEncoder(...)` unconditionally — i.e. the loaded class graph has slots for a ViT/Conformer that the 12B weights cannot fill, while the **actual** `vision_embedder.*` tensors have **no destination module** in that class.

**Conclusion for integrators:** the stock `mlx_vlm` gemma4 model targets the **E2B/E4B/26B/31B** variants (which DO ship `vision_tower`/`audio_tower` encoders). It is **NOT** a correct runtime for the encoder-free 12B Unified multimodal path. To support image/audio/video on 12B, a runtime must implement the thin `vision_embedder` / `embed_vision` / `embed_audio` early-fusion path itself:

1. Image: processor produces pixel patches (channel-first, rescaled 1/255, **not** standardized). The official `Gemma4Unified` model then **rescales patches `[0,1] → [-1,1]` via `2 * (x - 0.5)`** *before* the patch embedder (verified in HF `modeling_gemma4.py`; this step is easy to miss and a verified red image is read as black/white without it). Flatten each 48×48 patch → 6912-vector. `patch_ln1` → `patch_dense` (→3840) → add `pos_embedding` indexed by (x,y) patch coords → `pos_norm`/`patch_ln2`. Then a **`Gemma4VisionPooler` + `Gemma4VisionRotaryEmbedding`** stage (both **weightless** — no tensors in the checkpoint; pooling uses `pooling_kernel_size=3`, rotary is computed) and finally `embed_vision.embedding_projection` (3840→3840). Scatter results into the decoder embedding stream at `image_token_id` positions. (The exact ln1/ln2/pos/pooler ordering is **UNVERIFIED** against HF modeling code — the `mlx_vlm` `VisionPatchEmbedder` does not match these tensor names, so derive ordering from HF `Gemma4Unified` source, classes `Gemma4MultimodalEmbedder` / `Gemma4VisionPooler` / `Gemma4VisionRotaryEmbedding`.)
2. Audio: 640-dim audio frames → `embed_audio.embedding_projection` (640→3840) → scatter at `audio_token_id`.
3. There is **no transformer encoder with weights** to run for either modality on 12B — just the embedder linears + a weightless vision pooler/rotary + the shared decoder. (Contrast the E2B/E4B/26B/31B variants, which DO ship `vision_tower`/`audio_tower` encoder weights.)
4. **Soft-token count is dynamic, not fixed 280.** The number of image soft tokens is computed from the resized image dims and pooling, e.g. a 768-px image → 768/16/3 = 16 per side → **256** tokens (`image_seq_length=280` is the *max/default* budget, not a constant). The runtime MUST emit exactly as many `<\|image\|>` placeholders as the embedder produces feature rows, or the media span mask misaligns. Supported budgets: 70/140/280/560/1120.

The **text-only** path of `mlx_vlm`/`mlx_lm` gemma4 IS correct for 12B (verified weight-for-weight in §2/§3). The gap is strictly the multimodal embedding stage.

### 6.4 Multimodal token IDs (verified `config.json` + `tokenizer.json`)

| Token | ID | Config field | tokenizer string |
| --- | --- | --- | --- |
| image (soft) | 258880 | `image_token_id` | `<\|image\|>` |
| audio (soft) | 258881 | `audio_token_id` | `<\|audio\|>` |
| end-of-image | 258882 | `eoi_token_id` | `<image\|>` |
| end-of-audio | 258883 | `eoa_token_index` *(note: `_index`, not `_id`)* | `<audio\|>` |
| video (soft) | 258884 | `video_token_id` | `<\|video\|>` |
| begin-of-image | 255999 | `boi_token_id` | `<\|image>` |
| begin-of-audio | 256000 | `boa_token_id` | `<\|audio>` |

> Note: the config key for end-of-audio is `eoa_token_index` (different suffix than the others). Do not look for `eoa_token_id`.

### 6.5 Processor / feature-extractor config (verified `processor_config.json`)

| Field | Value |
| --- | --- |
| Processor class | `Gemma4UnifiedProcessor` |
| Image processor | `Gemma4UnifiedImageProcessor` |
| `image_seq_length` (soft tokens/image, default) | 280 |
| Image supported soft-token budgets | 70, 140, 280, 560, 1120 (README:401; `processing_gemma4.py:31`) |
| `patch_size` | 16 |
| `pooling_kernel_size` | 3 (effective patch 48) |
| `do_normalize` | **false** (image_mean=[0,0,0], image_std=[1,1,1]) |
| `do_rescale` | true, `rescale_factor` = 1/255 |
| `do_convert_rgb` | true; `do_resize` true |
| Video processor | `Gemma4UnifiedVideoProcessor`, `max_soft_tokens=70`/frame, `num_frames=32`, `do_normalize=true` |
| Audio feature extractor | `Gemma4UnifiedAudioFeatureExtractor`, `feature_size=640`, `audio_samples_per_token=640` |
| `audio_seq_length` (soft tokens/clip) | 750 |
| `audio_ms_per_token` | 40 |
| `sampling_rate` | 16000 Hz |
| Audio max length | 30 s (README:428) |
| Video max length | 60 s @ 1 fps (README:428) |

> **Image preprocessing trap:** images are **rescaled only** (÷255), **NOT** mean/std normalized (`do_normalize=false`). Video frames ARE normalized (`do_normalize=true`) — different from images. Modality ordering for best results: **image BEFORE text, audio AFTER text** (README:388-390).

### 6.6 Bundle-side verification & live runtime parity (2026-06-03)

The JANG MXFP4 / MXFP8 / JANG_4M bundles were verified **bit-exact-faithful** for the multimodal path, so any vision/audio bug is in the *runtime math*, not the weights:

- All **11** multimodal tensors present, correct shapes, **no inf/nan, none all-zero**; `relerr ≈ 0` vs the bf16 source.
- fp16 passthrough is **range-safe**: max |value| among multimodal tensors is ≤157 and ≤1280 for norms — far under fp16's 65504. (Small values → fp16 actually holds *more* mantissa than the bf16 source, hence lossless.) So a runtime seeing ones/zeros for `vision_embedder.*` has a **sanitize/key-mapping** bug, not a bundle bug. Expected post-sanitize keys: `vision_embedder.*`, `embed_vision.*`, `embed_audio.*` (leading `model.` stripped).
- `tokenizer.json` / `processor_config.json` / `generation_config.json` are **byte-identical** to source; all media token IDs + `suppress_tokens=[258883,258882]` intact.

**Live parity gaps being worked in the Swift runtime (vMLX-Swift `codex/gemma4-12b-unified-runtime`), for reference:** (a) the `2*(x-0.5)` patch rescale (§6.3.1); (b) dynamic 256-token count vs fixed 280 (§6.3.4); (c) exact `patch_ln1`/`pos`/`patch_ln2`/pooler ordering; (d) the bidirectional vision-span attention mask. A solid-red image still reading as black/white/audio with the bytes verified clean points at (a)/(c)/(d), not the checkpoint. Validate any bundle's *text* decoder independently with `scripts/gemma4/g4_coherence.py`.

---

## 7. MTP (Multi-Token Prediction) — NONE in 12B

Verified: **zero** `mtp.*` / `multi_token` / `nextn` tensors in the checkpoint (header scan returned empty). The `mlx_vlm` gemma4 `language.py` carries speculative-decoding / drafter scaffolding (`rollback_speculative_cache`, `capture_layer_ids`, `hidden_sink`), but **no MTP/drafter weights ship with 12B**. The JANG converter records `"mtp": "none"` (`convert_gemma4_mxfp.py:27,279,385`).

**Integration rule:** do not allocate or expect an MTP head. Speculative decoding, if desired, must use an external draft model, not a built-in MTP layer.

---

## 8. KV cache topology

Verified `make_cache` (`mlx_vlm/models/gemma4/language.py:686-700`; `mlx_lm/models/gemma4_text.py:662-675`):

| Layer type | Cache class | Params |
| --- | --- | --- |
| `full_attention` (8 layers) | unbounded `KVCache` | grows with sequence |
| `sliding_attention` (40 layers) | `RotatingKVCache` | `max_size = sliding_window = 1024`, `keep = 0` |

- One cache per layer, ordered by `layer_types`. Because `num_kv_shared_layers=0` on 12B, the cache list spans all 48 layers (no shared-KV truncation).
- Full layers store MQA KV: 1 head × head_dim 512. Sliding layers store GQA KV: 8 heads × head_dim 256. **Cache element widths differ by layer type** — allocate per-layer, not a uniform `[heads, head_dim]`.
- RotatingKVCache with `keep=0` means no "always-keep" prefix tokens; the window is a pure rolling 1024.

**Memory implication:** only the 8 global layers grow with context; the 40 local layers are capped at 1024 each. This is what makes 128K context tractable.

---

## 9. generation_config & sampling defaults

Verified `generation_config.json`:

| Field | Value | Meaning |
| --- | --- | --- |
| `do_sample` | true | sampling on by default |
| `temperature` | 1.0 | |
| `top_k` | 64 | |
| `top_p` | 0.95 | |
| `eos_token_id` | `[1, 106, 50]` | stop on any: `<eos>`(1), `<turn\|>`(106, EOT), `<\|tool_response>`(50) |
| `suppress_tokens` | `[258883, 258882]` | never emit `<audio\|>`(eoa) / `<image\|>`(eoi) end-markers |
| `pad_token_id` | 0 | `<pad>` |
| `bos_token_id` | 2 | `<bos>` |

> **eos set is three tokens.** Note `config.json:eos_token_id` is `[1, 106]` but `generation_config.json:eos_token_id` is `[1, 106, 50]`. Use the **generation_config** value (includes 50 = `<|tool_response>`): when the model emits the tool-response opener it has finished its turn and is waiting for a tool result, so generation must halt there. `<turn|>`(106) is the normal end-of-turn stop.
>
> **suppress_tokens** prevents the model from hallucinating image/audio *end* markers (258882/258883) into text output. A runtime's logit processor must mask these two IDs to `-inf` before sampling to match HF behavior.

---

## 10. Chat template, reasoning (thinking) & tool-calling

### 10.1 Turn structure (verified `chat_template.jinja`)

- Prompt opens with `bos_token` (`<bos>`, id 2) (line 177).
- A turn is `<|turn>{role}\n ... <turn|>\n` where role ∈ {`system`, `user`, `model`} (assistant→`model`, lines 219,234,351). `<|turn>` = id 105, `<turn|>` = id 106 (EOT).
- System/developer content, tool definitions, and the thinking-open token all live in the **first `<|turn>system` block** (lines 179-205).
- `add_generation_prompt` appends `<|turn>model\n` and, **when thinking is disabled**, an empty thought channel `<|channel>thought\n<channel|>` (lines 356-362).

### 10.2 Reasoning / "thinking" channel

- Enabled via `enable_thinking` kwarg. When true, `<|think|>\n` (id 98) is injected at the very top of the first system turn (lines 182-185).
- Model reasoning is emitted in a **thought channel**: `<|channel>thought\n{reasoning}\n<channel|>` (line 240; `<|channel>`=100, `<channel|>`=101).
- **Disabled-thinking behavior:** the model still emits the channel tags but with an empty body — `<|channel>thought\n<channel|>` then the final answer (README:380-381). The template pre-seeds this empty block on the generation prompt (line 360).
- **History rule:** in multi-turn, prior model turns must contain only the final answer — **strip thinking from history** (README:386; the template's `strip_thinking` macro, lines 148-158, removes `<|channel>...<channel|>` spans from rendered assistant content).
- Capability map sets `think_in_template = False` for gemma4 (`capabilities.py:74`): the assistant turn does NOT auto-open inside a think block on no-think prompts, so the reasoning parser should not route all output to reasoning_content by default.

### 10.3 Tool-calling wire format

Verified from `chat_template.jinja` and `mlx_lm/tool_parsers/gemma4.py`.

**Tool definitions** (in system turn): `<|tool>declaration:{name}{description:<|"|>...<|"|>,parameters:{...},type:<|"|>OBJECT<|"|>}<tool|>` (lines 86-117,196-203). `<|tool>`=46, `<tool|>`=47.

**Model tool call** (emitted by model): `<|tool_call>call:{name}{args}<tool_call|>` (lines 243-258; `tool_parsers/gemma4.py:64-65`). `<|tool_call>`=48, `<tool_call|>`=49.

**Argument syntax (Gemma-4-specific, NOT JSON):**
- `call:name{...}` with **balanced braces**.
- **Unquoted keys**: `{key:value,key2:value2}`.
- **Strings delimited by `<|"|>`** (id 52), not double quotes: `name:<|"|>San Francisco<|"|>`.
- Booleans `true`/`false`; arrays `[...]`; nested objects `{...}`.

The parser (`tool_parsers/gemma4.py`):
- regex `call:([\w-]+)(\{...balanced...\})` with recursive brace matching (`:17-20`).
- `_gemma4_args_to_json` (`:23-43`): extracts `<|"|>...<|"|>` strings → placeholders, quotes bare keys (`(?<=[{,])(\w+):` → `"\1":`), restores strings as JSON. Then `json.loads`.
- `x-parser: gemma4-tool-call`, `x-regex: call\:(?P<name>\w+)(?P<arguments>\{.*\})`, iterator `<\|tool_call>(.*?)<tool_call\|>` (`tokenizer_config.json:response_schema`).

**Tool response** (fed back to model): `<|tool_response>response:{name}{key:value,...}<tool_response|>` (lines 160-173). `<|tool_response>`=50, `<tool_response|>`=51. Generation halts on emitting `<|tool_response>`(50, in eos set) — runtime sends the tool result wrapped in this block on the next turn.

**JANG capability routing:** `gemma4` → reasoning_parser `gemma4`, tool_parser `gemma4` (`capabilities.py:74-77`). The runtime must register a `gemma4` tool parser implementing the unquoted-key + `<|"|>`-string decoding above; a generic JSON parser will fail.

### 10.4 response_schema (verified `tokenizer_config.json:response_schema`)

Top-level extraction regex separates the assistant turn into `thinking` / `tool_calls` / `content`:
```
(<\|channel>thought\n(?P<thinking>.*?)<channel\|>)?(?P<tool_calls><\|tool_call>.*<tool_call\|>)?(?P<content>...)?(?:<turn\|>|<\|tool_response>)?
```
Roles: `assistant` with optional `content`, `thinking`, `tool_calls[]` fields.

---

## 11. Tokenizer details

Verified `tokenizer_config.json` + `tokenizer.json`:

| Property | Value |
| --- | --- |
| Class | `GemmaTokenizer` (BPE, `tokenizer.json:model.type = BPE`) |
| Backend | `tokenizers` |
| Vocab size | 262144 |
| `model_max_length` | 1000000000000000019884624838656 (i.e. "unbounded" sentinel; rely on `max_position_embeddings`=131072) |
| `padding_side` | left |
| bos / eos / pad / unk / mask | `<bos>`=2 / `<eos>`=1 / `<pad>`=0 / `<unk>`=3 / `<mask>`=4 |

### 11.1 Full special-token table (verified from `tokenizer.json` added_tokens)

| String | ID | Config role |
| --- | --- | --- |
| `<pad>` | 0 | pad |
| `<eos>` | 1 | eos |
| `<bos>` | 2 | bos |
| `<unk>` | 3 | unk |
| `<mask>` | 4 | mask |
| `<\|tool>` | 46 | tool-def open (std_token) |
| `<tool\|>` | 47 | tool-def close (etd_token) |
| `<\|tool_call>` | 48 | tool-call open (stc_token) |
| `<tool_call\|>` | 49 | tool-call close (etc_token) |
| `<\|tool_response>` | 50 | tool-response open (str_token) — also in eos set |
| `<tool_response\|>` | 51 | tool-response close (etr_token) |
| `<\|"\|>` | 52 | tool-arg string delimiter (escape_token) |
| `<\|think\|>` | 98 | thinking enable (think_token) |
| `<\|channel>` | 100 | channel open (soc_token) |
| `<channel\|>` | 101 | channel close (eoc_token) |
| `<\|turn>` | 105 | turn open (sot_token) |
| `<turn\|>` | 106 | turn close / EOT (eot_token) — in eos set |
| `<\|image>` | 255999 | begin-of-image (boi) |
| `<\|audio>` | 256000 | begin-of-audio (boa) |
| `<\|image\|>` | 258880 | image soft token |
| `<\|audio\|>` | 258881 | audio soft token |
| `<image\|>` | 258882 | end-of-image (eoi) — suppressed |
| `<audio\|>` | 258883 | end-of-audio (eoa) — suppressed |
| `<\|video\|>` | 258884 | video soft token |

24 added tokens total. String↔ID mapping cross-verified against `tokenizer_config.json` token aliases (`sot_token`/`eot_token`/`soc_token`/`eoc_token`/`stc_token`/`etc_token`/`std_token`/`etd_token`/`str_token`/`etr_token`/`think_token`/`escape_token`).

---

## 12. Quantized-bundle on-disk layout (JANG MXFP)

Produced by `jang-tools/jang_tools/convert_gemma4_mxfp.py`. This defines exactly what the Python/Swift loaders see on disk for a JANG bundle.

### 12.1 Key remap (sanitize), verified `convert_gemma4_mxfp.py:92-99`

```
1. strip leading "model."          : model.X                         -> X
2. language_model.* (not .model.)  : language_model.foo              -> language_model.model.foo
```
Examples:
- `model.language_model.layers.0.self_attn.q_proj.weight` → `language_model.model.layers.0.self_attn.q_proj.weight`
- `model.language_model.embed_tokens.weight` → `language_model.model.embed_tokens.weight`
- `model.language_model.norm.weight` → `language_model.model.norm.weight`
- `model.vision_embedder.patch_dense.weight` → `vision_embedder.patch_dense.weight` (model. stripped only)
- `model.embed_vision.embedding_projection.weight` → `embed_vision.embedding_projection.weight`
- `model.embed_audio.embedding_projection.weight` → `embed_audio.embedding_projection.weight`

This matches `mlx_vlm/models/gemma4/gemma4.py:228-237` sanitize, so a JANG bundle loads under the `mlx_vlm` gemma4 *text* layout.

### 12.2 What gets quantized vs passthrough (verified `quant_policy`, `:102-128`)

| Tensor class | Treatment |
| --- | --- |
| decoder linears: q/k/v/o_proj, gate/up/down_proj | **MX affine quantized** (`mxfp4` or `mxfp8`), emits `.weight` + `.scales` (+ `.biases`) |
| tied `embed_tokens.weight` (2D) | **quantized** (used as both embedding + output via `as_linear`) |
| all `*norm*` weights (input/post_attn/pre+post_ffw/q_norm/k_norm/final + vision LNs) | **fp16 passthrough, NO +1** |
| `layer_scalar` (per-layer scalar) | fp16 passthrough |
| `vision_embedder.*`, `embed_vision.*`, `embed_audio.*`, any `vision_tower`/`audio_tower` | **fp16 passthrough** (early-fusion embedders preserved) |
| `pos_embedding`, biases, single-token names | fp16 passthrough |
| `*_scale_inv` | skipped |

Default group_size = 32 (`convert_gemma4_mxfp.py:247`).

### 12.3 Bundle config additions (verified `:376-426`)

`config.json` gets:
```json
"weight_format": "mxfp4"|"mxfp8",
"quantization": {
  "bits": 4|8, "group_size": 32, "mode": "mxfp4|mxfp8",
  "quantization_backend": "mx.quantize",
  "norm_convention": "gemma4_scale_shift_zero",
  "multimodal": "fp16_passthrough_embedders_early_fusion",
  "mtp": "none"
}
```
Plus a `jang_config.json` with `has_vision:true`, `has_audio:true`, `runtime.attention:"hybrid_swa_full_5to1"`, `runtime.sliding_window:1024`, `runtime.attention_k_eq_v_on_full_layers:true`, `runtime.full_attention_layers:[5,11,17,23,29,35,41,47]`, and `mtp_policy:"none"`.

### 12.4 Sidecars copied (verified `:61-73,223-236`)

`tokenizer.json`, `tokenizer_config.json`, `generation_config.json`, `chat_template.jinja`, `processor_config.json`, README, LICENSE (when present). The `.jinja` template is also folded into `tokenizer_config.chat_template` for runtimes that only read that field.

> The JANG bundle does **not** quantize or restructure the multimodal embedders — they stay fp16. So a JANG bundle still requires the runtime to implement the §6.3 early-fusion embedding path for vision/audio. Text-only inference needs none of the multimodal tensors.

---

## 13. Integration checklist — vmlx (Python) & vmlx-swift (Swift)

Both runtimes must implement/verify the following. Gotchas flagged with ⚠️.

### Text backbone (required for any inference)
- [ ] Parse `gemma4_unified` top config; read `text_config` for all dims (§2). Detect dense (`enable_moe_block=false`) and skip MoE/PLE/KV-share/double-wide paths (all zero/false on 12B).
- [ ] ⚠️ **RMSNorm with NO +1** (`scale_shift=0`). Use plain RMSNorm for every norm including q/k norm and final norm. Do not reuse a Gemma-1/2/3 `x*(1+w)` kernel. (§5)
- [ ] Apply `embed_scale = sqrt(3840)` after token lookup AND on the multimodal `inputs_embeds` path. (§5.1)
- [ ] Four norms per block + sandwich residual order + `h *= layer_scalar` at block end (load the `[1]` scalar per layer). (§2.2)
- [ ] GeGLU MLP: `down(gelu_approx(gate(x)) * up(x))`, intermediate 15360. (§2.1)
- [ ] Tied embeddings: output logits via `embed_tokens.as_linear` (no `lm_head`). (§2)
- [ ] Final logit softcap `tanh(x/30)*30`. (§3.8)

### Attention (required)
- [ ] ⚠️ Per-layer-type attention shapes: sliding = GQA 16q/8kv × head_dim 256; full = MQA 16q/1kv × **global_head_dim 512**. Build q/k/o projection sizes from layer type. (§3.2)
- [ ] ⚠️ **k_eq_v on the 8 full layers [5,11,17,23,29,35,41,47]: no `v_proj` tensor.** Loader must not require v_proj on those layers; set `V := k_proj(x)` BEFORE k_norm. (§3.3)
- [ ] Norm order: q_norm/k_norm on `[B,L,H,D]` before transpose; RoPE after transpose; v_norm (weightless) then transpose, V gets NO RoPE. (§3.4)
- [ ] ⚠️ Attention `scale = 1.0` (not `1/sqrt(head_dim)`) for both layer types. (§3.5)
- [ ] Per-type masks: full = causal; sliding = causal + window 1024. (§3.6)
- [ ] ⚠️ Dual RoPE: sliding = default θ=10000 over full 256 dims; full = **proportional p-RoPE** θ=1e6, rotate only first 128 of 512 dims, frequency exponent divided by **512 (full dim)**, HF rotate_half layout. (§4)
- [ ] ⚠️ Bidirectional vision masking declared (`use_bidirectional_attention="vision"`) but NOT implemented in the MLX text reference — implement custom span mask if shipping vision, verify against HF source. (§3.7)

### Cache (required)
- [ ] ⚠️ Heterogeneous cache: `KVCache` (unbounded) for full layers; `RotatingKVCache(max_size=1024, keep=0)` for sliding layers. Per-layer KV widths differ (full: 1×512, sliding: 8×256). (§8)

### Sampling / chat / tools
- [ ] Sampling defaults: do_sample, T=1.0, top_k=64, top_p=0.95. (§9)
- [ ] ⚠️ eos set = `{1, 106, 50}` (use generation_config, not config). Stop on `<turn|>` and `<|tool_response>`. (§9)
- [ ] ⚠️ Logit suppression of 258882, 258883 (image/audio end markers). (§9)
- [ ] Register the `gemma4` chat template (turns, thought channel, empty-thought on no-think). (§10.1-10.2)
- [ ] ⚠️ Register a `gemma4` tool parser: `call:name{...}` with unquoted keys + `<|"|>` string delimiters — NOT JSON. Decode per `tool_parsers/gemma4.py`. (§10.3)
- [ ] ⚠️ Strip thinking from conversation history; `think_in_template=False`. (§10.2)

### Multimodal (only if supporting image/audio/video)
- [ ] ⚠️ **Encoder-free / early-fusion.** Do NOT load a ViT/Conformer — none ship in 12B. The stock `mlx_vlm` gemma4 `vision_tower`/`audio_tower` graph is for E2B/E4B/26B/31B, not 12B. Implement the thin `vision_embedder`/`embed_vision`/`embed_audio` path. (§6.1-6.3)
- [ ] Multimodal token IDs + soft-token counts (image 280 default / budgets 70-1120, video 70/frame × ≤32 frames, audio 750). (§6.4-6.5)
- [ ] ⚠️ Image preprocessing: rescale ÷255, **do_normalize=false** (no mean/std); video DOES normalize. Modality order: image before text, audio after text. (§6.5)
- [ ] `eoa` config key is `eoa_token_index` (atypical suffix). (§6.4)

### MTP
- [ ] ⚠️ **No native MTP.** Do not expect/allocate an MTP head. Spec-decode via external draft model only. (§7)

### JANG bundle loading
- [ ] Expect sanitized keys: `model.` stripped, `language_model.` → `language_model.model.`. (§12.1)
- [ ] Quantized: decoder linears + tied embed (mxfp4/8, `.weight`/`.scales`/`.biases`, group_size 32). fp16 passthrough: all norms, layer_scalar, multimodal embedders, pos_embedding. (§12.2)
- [ ] Read `config.quantization.norm_convention == "gemma4_scale_shift_zero"` as the explicit NO-+1 signal. (§12.3)

---

## 14. Open / UNVERIFIED items (do not ship as fact)

1. **Context length 128K vs 256K.** Config says 131072; README says 256K for 12B. Use config; confirm with Google.
2. **Bidirectional-vision mask exact span.** Declared in config; not implemented in the MLX text reference. Confirm inclusive/exclusive of `boi`/`eoi` and whether it covers video against HF `Gemma4Unified` modeling code.
3. **vision_embedder forward ordering** (patch_ln1 → dense → pos → ln2/pos_norm sequencing). The shipped `vision_embedder.*` tensor names do not map to `mlx_vlm`'s `VisionPatchEmbedder`; derive exact graph from HF source.
4. **Whether HF folds a softmax scale into q/k norm weights** (we match the MLX reference's `scale=1.0`, which loads these weights directly and is treated as correct).
5. End-to-end multimodal numerical parity has not been run here — only the text path and weight layout are verified against the shipped checkpoint.
