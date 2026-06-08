# Nemotron 3 Ultra 550B-A55B NVFP4 Source Contract

Date: 2026-06-04

This note records the source-side architecture and precision contract for
`nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` before any JANG or JANGTQ
conversion work. It is intentionally source-first: do not infer coherent JANG
profiles from model name or total parameter count alone.

## Intake

- Source repo: `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`
- Local staging path:
  `/Volumes/EricsLLMDrive/sources/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`
- Hub revision observed during intake:
  `0f2eb8bac913b7ea768d8dec8e2e889fcecf6480`
- Source format: mixed precision ModelOpt checkpoint with safetensors shards.
- Source shard count: 113 `.safetensors` weight shards plus JSON sidecars.
- Expected source weight size from HEAD checks: about `328 GiB`.
- BF16 sibling exists, but is not the first conversion target:
  `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16`, about `1044 GiB`.

## Top-Level Model Contract

The source `config.json` identifies this as Nemotron-H, not a Llama-style MoE:

- `model_type`: `nemotron_h`
- `architectures`: `["NemotronHForCausalLM"]`
- `dtype`: `bfloat16`
- `vocab_size`: `131072`
- `hidden_size`: `8192`
- `max_position_embeddings`: `262144`
- `num_nextn_predict_layers`: `1`
- `mtp_layers_block_type`: `["attention", "moe"]`
- `bos_token_id`: `1`
- `eos_token_id`: source config `2`; generation config `[2, 11]`
- `pad_token_id`: `0`

Generation config uses `eos_token_id=[2, 11]`. Preserve this distinction when
building runtime metadata and chat-template notes.

## Chat Template / Parser Contract

The source ships `chat_template.jinja` and `tokenizer_config.json` with explicit
reasoning and tool-call tokens. These files are part of the runtime contract
and must be copied into every JANG/JANGTQ bundle.

Observed template behavior:

- `enable_thinking` defaults to `true`.
- `enable_thinking=false` starts assistant generation with `<think></think>`.
- `medium_effort=true` appends `{reasoning effort: efficient}` to the last user
  turn.
- Historical assistant reasoning is carried as `<think>...</think>`, with
  history truncation controlled by `truncate_history_thinking`.
- Tool definitions are rendered inside `<tools>...</tools>`.
- Tool calls are rendered as
  `<tool_call><function=name><parameter=arg>...</parameter></function></tool_call>`.
- Tool results are rendered as `<tool_response>...</tool_response>` inside a
  user turn.

README runtime guidance names the parsers as `nemotron_v3` / `nemotron_3` for
reasoning and `qwen3_coder` for tool calls. Tool calling with reasoning also
requires chat-template kwargs equivalent to `enable_thinking=true` and
`force_nonempty_content=true`.

JANG runtimes should therefore preserve the source template and expose a
Nemotron-v3 reasoning parser plus Qwen3-coder-style XML tool parser. Do not
silently substitute a Llama, ChatML-only, or JSON-function-call parser.

## Modality Contract

The inspected source weight index has no vision/audio/projector namespaces:

- `vision`: `0`
- `visual`: `0`
- `image`: `0`
- `audio`: `0`
- `speech`: `0`
- `encoder`: `0`
- `projector` / `mm_projector`: `0`

This Ultra checkpoint is a text-generation hybrid SSM/MoE/attention model, not
a VL or audio bundle. The only auxiliary generative namespace found in the
weight index is `mtp.*`.

## Backbone Layer Layout

`layers_block_type` has 108 backbone layers:

- Mamba layers: 48
- MoE layers: 48
- Attention layers: 12

Layer indices:

- Mamba:
  `0, 2, 4, 6, 9, 11, 13, 16, 18, 20, 22, 25, 27, 29, 31, 34, 36, 38, 41, 43, 45, 47, 50, 52, 54, 56, 59, 61, 63, 66, 68, 70, 72, 75, 77, 79, 81, 84, 86, 88, 91, 93, 95, 97, 100, 102, 104, 106`
- MoE:
  `1, 3, 5, 8, 10, 12, 15, 17, 19, 21, 24, 26, 28, 30, 33, 35, 37, 40, 42, 44, 46, 49, 51, 53, 55, 58, 60, 62, 65, 67, 69, 71, 74, 76, 78, 80, 83, 85, 87, 90, 92, 94, 96, 99, 101, 103, 105, 107`
- Attention:
  `7, 14, 23, 32, 39, 48, 57, 64, 73, 82, 89, 98`

The sparse attention placement matters for coherence. Attention projections are
not the main compression target here because the source already excludes them
from ModelOpt quantization.

## Attention Contract

- Attention heads: `64`
- KV heads: `2`
- Head dim: `128`
- Attention bias: `false`
- Attention dropout: `0.0`
- RoPE:
  - `partial_rotary_factor`: `1.0`
  - `rope_theta`: `10000`
- Sliding window: `null`

The 12 attention layers carry `q_proj`, `k_proj`, `v_proj`, and `o_proj`.
`hf_quant_config.json` excludes all 48 attention projection modules from source
quantization. Treat them as precision-critical until safetensor header audits
prove their exact stored dtype.

## Mamba / SSM Contract

- Mamba layers: 48
- `mamba_num_heads`: `256`
- `mamba_head_dim`: `64`
- `ssm_state_size`: `128`
- `conv_kernel`: `4`
- `chunk_size`: `128`
- `mamba_ssm_cache_dtype`: `float32`
- `use_mamba_kernels`: `true`
- `mamba_hidden_act`: `silu`
- `mamba_proj_bias`: `false`
- `use_conv_bias`: `true`

Source precision:

- `mixer.in_proj`: FP8, 48 modules
- `mixer.out_proj`: FP8, 48 modules
- `mixer.conv1d`: excluded from quantization, 48 modules
- SSM scalar/state parameters such as `A_log`, `D`, and `dt_bias` must be kept
  out of low-bit expert policies. Confirm their stored dtype from safetensor
  headers during audit.

For JANG/JANGTQ coherence, the SSM cache and conv state are runtime semantics,
not metadata decoration. Any profile that changes Mamba projection precision
must be tested with prefill and decode, not only single-token tensor loading.

## MoE / LatentMoE Contract

- MoE layers: 48
- Routed experts per MoE layer: `512`
- Shared experts per MoE layer: `1`
- Experts active per token: `22`
- `n_groups`: `8`
- `topk_group`: `1`
- `norm_topk_prob`: `true`
- `routed_scaling_factor`: `5.0`
- `moe_intermediate_size`: `5120`
- `moe_shared_expert_intermediate_size`: `10240`
- `moe_latent_size`: `2048`
- MLP activation: `relu2`
- MLP bias: `false`

LatentMoE is mandatory for this model. Each MoE layer has:

- `fc1_latent_proj`
- `fc2_latent_proj`
- router gate
- 512 routed experts
- one shared expert path

Do not route Ultra through old Nemotron-H code that only assumes the Nano/Omni
shape. Existing loaders must support the latent expert projection wrapper before
instantiating the model skeleton.

## Source Precision Map

Observed from `hf_quant_config.json`:

- Producer: `modelopt`, version `1.0.0`
- Overall quantization: `MIXED_PRECISION`
- KV cache quantization: `FP8`
- Quantized layer entries: `49344`
- Excluded modules: `243`

Source quantized modules:

| Role | Count | Source precision |
| --- | ---: | --- |
| Routed expert `up_proj` / `down_proj` | 49152 | NVFP4, group size 16 |
| Shared expert `up_proj` / `down_proj` | 96 | FP8 |
| Mamba `in_proj` / `out_proj` | 96 | FP8 |

Source excluded modules:

| Role | Count | Source precision handling |
| --- | ---: | --- |
| Embeddings | 1 | excluded from quant config |
| Attention `q/k/v/o_proj` | 48 | excluded from quant config |
| MoE router gates | 48 | excluded from quant config |
| LatentMoE `fc1/fc2_latent_proj` | 96 | excluded from quant config |
| Mamba `conv1d` | 48 | excluded from quant config |
| `lm_head` | 1 | excluded from quant config |
| `mtp*` | 1 pattern | excluded from quant config |

Interpretation: excluded does not mean optional. These are likely the
precision-critical control plane for routing, attention, embeddings, MTP, and
SSM state update. The conversion audit must read safetensor headers and record
the exact stored dtype/shape for each excluded class before deciding any JANG
or JANGTQ bit policy.

## Early Safetensor Header Observations

These observations came from the first downloaded shards during intake. They
are useful for profile design, but the final audit still must scan all 113
shards.

Layer 0 Mamba / embedding shard:

| Tensor | Stored dtype | Shape | Notes |
| --- | --- | --- | --- |
| `backbone.embeddings.weight` | `BF16` | `[131072, 8192]` | excluded from source quant config |
| `backbone.layers.0.mixer.A_log` | `BF16` | `[256]` | SSM state/control |
| `backbone.layers.0.mixer.D` | `BF16` | `[256]` | SSM state/control |
| `backbone.layers.0.mixer.dt_bias` | `BF16` | `[256]` | SSM state/control |
| `backbone.layers.0.mixer.conv1d.weight` | `BF16` | `[18432, 1, 4]` | excluded from source quant config |
| `backbone.layers.0.mixer.conv1d.bias` | `BF16` | `[18432]` | excluded from source quant config |
| `backbone.layers.0.mixer.norm.weight` | `BF16` | `[16384]` | normalization |
| `backbone.layers.0.norm.weight` | `BF16` | `[8192]` | normalization |
| `backbone.layers.0.mixer.in_proj.weight` | `F8_E4M3` | `[35072, 8192]` | source FP8 |
| `backbone.layers.0.mixer.out_proj.weight` | `F8_E4M3` | `[8192, 16384]` | source FP8 |

Layer 1 MoE shard:

| Tensor | Stored dtype | Shape | Notes |
| --- | --- | --- | --- |
| `backbone.layers.1.mixer.gate.weight` | `F32` | `[512, 8192]` | router/control plane |
| `backbone.layers.1.mixer.fc1_latent_proj.weight` | `BF16` | `[2048, 8192]` | LatentMoE control plane |
| `backbone.layers.1.mixer.fc2_latent_proj.weight` | `BF16` | `[8192, 2048]` | LatentMoE control plane |
| `backbone.layers.1.mixer.shared_experts.up_proj.weight` | `F8_E4M3` | `[10240, 8192]` | source FP8 |
| `backbone.layers.1.mixer.shared_experts.down_proj.weight` | `F8_E4M3` | `[8192, 10240]` | source FP8 |
| `backbone.layers.1.mixer.experts.0.up_proj.weight` | `U8` | `[5120, 1024]` | packed NVFP4 representation |
| `backbone.layers.1.mixer.experts.0.down_proj.weight` | `U8` | `[2048, 2560]` | packed NVFP4 representation |

Layer 7 attention shard:

| Tensor | Stored dtype | Shape | Notes |
| --- | --- | --- | --- |
| `backbone.layers.7.mixer.q_proj.weight` | `BF16` | `[8192, 8192]` | excluded from source quant config |
| `backbone.layers.7.mixer.k_proj.weight` | `BF16` | `[256, 8192]` | 2 KV heads x 128 head dim |
| `backbone.layers.7.mixer.v_proj.weight` | `BF16` | `[256, 8192]` | 2 KV heads x 128 head dim |
| `backbone.layers.7.mixer.o_proj.weight` | `BF16` | `[8192, 8192]` | excluded from source quant config |

Partial-shard dtype summary after 14 downloaded safetensor shards:

| Tensor class | Observed stored dtype(s) |
| --- | --- |
| Attention weights | `BF16` |
| Conv1d weights | `BF16` |
| Embedding weights | `BF16` |
| Router gate weights | `F32` |
| LatentMoE projection weights | `BF16` |
| Mamba projection weights | `F8_E4M3` |
| Routed expert packed weights | `U8` |
| Shared expert weights | `F8_E4M3` |
| Quant scale tensors | `F32`, `F8_E4M3` |

This confirms the first coherent policy: do not make a uniform "1-bit" or
"TQ everything" model. The source itself is not uniform. It uses BF16/F32 for
control-plane tensors, FP8 for always-on/shared projection paths, and packed
NVFP4 for routed experts.

## Candidate JANG Profiles

### JANG_1L

Goal: smallest linear/affine JANG lane that remains coherent enough to load and
run. This is not a quality claim until live prompts pass.

Initial policy hypothesis:

- Preserve embeddings and `lm_head` at source precision or conservative affine.
- Preserve attention projections at source precision first; only lower after
  attention-layer ablation proves stability.
- Preserve router gates, LatentMoE projections, Mamba conv/state parameters,
  and any enabled MTP path at source or conservative precision.
- Compress routed experts most aggressively, because source already places the
  550B parameter mass there with NVFP4 group size 16.
- Treat Mamba `in_proj/out_proj` and shared experts separately from routed
  experts because source uses FP8, not NVFP4.
- Do not reduce router gates below their observed `F32` source precision in the
  first build.
- Do not reduce attention, LatentMoE projections, SSM state parameters, or
  embeddings below observed `BF16` in the first build.

### Smallest Coherent JANGTQ

Goal: smallest TurboQuant routed-expert lane that still preserves routing,
attention, SSM, parser/template behavior, and embeddings coherently.

Initial policy hypothesis:

- Route TurboQuant only over routed expert `up_proj/down_proj` first.
- Keep `F32` gates and `BF16` latent projections, attention, Mamba conv/state,
  embeddings, and `lm_head` out of low-bit TQ on the first coherent build.
- For the hard-under-128 GiB build, drop `mtp.*` explicitly because current JANG
  inference is normal autoregressive decode and the MTP block only helps if a
  tested speculative accept/reject loop consumes it. The output bundle must set
  `num_nextn_predict_layers=0`, `mtp_layers_block_type=[]`, and
  `keeps_mtp=false`, then verify no `mtp.*` keys remain in the output index.
- Decide whether shared experts and Mamba projections stay FP8/affine/source
  after header audit. They are always-active or shared paths and should not be
  collapsed into the routed-expert bit policy.
- Preserve `mxtq_bits` dual-form decoding if per-role policies are introduced:
  flat integer for legacy and dictionary for role-specific control.

Potential naming:

- `NVIDIA-Nemotron-3-Ultra-550B-A55B-JANG_1L`
- `NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L` or explicit
  `JANGTQ_ROUTED1L` if routed-only TQ needs to be distinguished.

Do not publish or document final names externally until the bundle loads,
headers validate, and live generation is coherent.

## Required Audit Before Conversion

1. Verify local snapshot completeness against the 113-shard NVFP4 source repo.
2. Parse safetensor headers without loading full tensors.
3. Produce a tensor-class table:
   - source key pattern
   - destination key pattern
   - shape
   - stored dtype
   - source precision class
   - proposed JANG/JANGTQ handling
4. Confirm MTP tensor naming and whether existing loader filters or loads MTP.
5. Confirm current `nemotron_h` runtime supports:
   - 108-layer layout
   - 512 routed experts
   - `moe_latent_size=2048`
   - Mamba cache/state layout
   - attention layers at the sparse positions above
6. Run a no-conversion loader/header dry run before building any output bundle.
7. Build a tiny layer/header audit artifact under `docs/runtime/logs/`.
8. Only then choose the first `JANG_1L` or smallest coherent `JANGTQ` policy.

## Non-Negotiable Coherence Checks

- Header completeness: no missing source tensor classes.
- Config fidelity: exact layer ordering preserved; MTP source contract recorded,
  and output metadata explicitly marks MTP dropped/disabled when using the
  hard-under-128 build.
- Router fidelity: gates and top-k semantics preserved.
- LatentMoE fidelity: latent projections present and wired before expert matmul.
- SSM fidelity: Mamba conv/state/cache path works in prefill and decode.
- Attention fidelity: 12 sparse attention layers use correct head/KV geometry.
- EOS fidelity: config and generation config token IDs preserved.
- Runtime proof: real prompt, not just conversion success.
