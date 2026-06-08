# Nemotron 3 Ultra JANGTQ_1L vMLX Engine Handoff

Date: 2026-06-04

Scope: notes for future vMLX Python and vMLX Swift engine agents implementing
runtime support for:

`/Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L`

This is an engine handoff, not a production-ready declaration. The bundle
loads and executes, but parser, multi-turn, cache-stack, and quality gates are
still partial.

Related gates:

- `docs/runtime/2026-06-04-nemotron-ultra-cache-block-contract.md`
- `docs/runtime/2026-06-04-nemotron-ultra-runtime-nuances-and-gates.md`
- `docs/runtime/2026-06-04-nemotron-ultra-long-coherence-and-vl-proof.md`
- `jang-tools/examples/nemotron_ultra/README.md`

## Artifact Facts

- Source repo: `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`
- Source path:
  `/Volumes/EricsLLMDrive/sources/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`
- Output path:
  `/Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L`
- Output size: `98G`
- Output shards: `51`
- Weight keys: `1503`
- `model_type`: `nemotron_h`
- Layers: `108`
- Layer split: 48 Mamba, 48 MoE, 12 attention
- MTP: dropped from output (`mtp.*` keys: `0`)
- Modalities: text-only. Source index has no `vision`, `visual`, `image`,
  `audio`, `speech`, `encoder`, `projector`, or `mm_projector` tensor keys.
  Source folder also has no `processor_config.json` or
  `preprocessor_config.json`.
- `jang_config.capabilities`: stamped for vMLX as
  `family=nemotron_h`, `modality=text`, `cache_type=hybrid`,
  `reasoning_parser=deepseek_r1`, `tool_parser=nemotron`,
  `think_in_template=true`.

## Quant Layout

The bundle is intentionally mixed precision:

| Tensor family | Runtime representation |
| --- | --- |
| Routed MoE experts `up_proj/down_proj` | 1-bit JANGTQ/TurboQuant |
| Mamba `in_proj/out_proj` | 8-bit MLX affine |
| Shared expert `up_proj/down_proj` | 8-bit MLX affine |
| Attention q/k/v/o | BF16 passthrough |
| Router gate and correction bias | F32/BF16 source precision |
| LatentMoE `fc1/fc2_latent_proj` | BF16 passthrough |
| Mamba conv/state/norm | BF16 passthrough |
| Embeddings and `lm_head` | BF16 passthrough |
| MTP | dropped/disabled |

Do not implement this as "1-bit everything". The control plane is the reason
the model remains runnable.

## Decode Loop Shape

The local MLX reference is `mlx_lm.models.nemotron_h`.

Layer dispatch is driven by `layers_block_type` through the normalized pattern:

- `mamba` -> `M`: `NemotronHMamba2Mixer`
- `attention` -> `*`: `NemotronHAttention`
- `moe` -> `E`: `NemotronHMoE`

The forward loop keeps one cache entry only for Mamba and attention layers.
MoE layers have no cache entry.

Pseudocode:

```text
hidden = embeddings(input_ids)
attn_mask = create_attention_mask(hidden, cache[first_attention_cache_index])
ssm_mask = create_ssm_mask(hidden, cache[first_mamba_cache_index])
cache_counter = 0

for layer in 108 layers:
    if layer is Mamba or Attention:
        layer_cache = cache[cache_counter]
        cache_counter += 1
    else:
        layer_cache = nil

    mask = attn_mask if layer is Attention else ssm_mask
    hidden = hidden + layer.mixer(layer.norm(hidden), mask, layer_cache)

logits = lm_head(final_norm(hidden))
```

Cache list length is `60`: 48 `ArraysCache(size=2)` entries for Mamba layers
and 12 `KVCache` entries for attention layers, in layer order excluding MoE.

## Mamba / SSM Cache Contract

Each Mamba cache entry is an `ArraysCache(size=2)`:

- slot `0`: convolution state, shape `[batch, conv_kernel - 1, conv_dim]`
- slot `1`: SSM recurrent state, shape determined by MLX `ssm_update`
- `lengths` / offset must advance when SSM emits tokens

For Ultra:

- `conv_kernel=4`
- `conv_dim = intermediate_size + 2 * n_groups * ssm_state_size`
- here `intermediate_size` means the Mamba mixer's local value from
  `mamba_num_heads * mamba_head_dim = 256 * 64 = 16384`, not the config
  field `intermediate_size=5120` used by the MoE expert MLP
- Mamba `n_groups=8`
- `ssm_state_size=128`
- `conv_dim=18432`
- `time_step_limit` must be numeric. The output config normalizes source
  `[0.0, {"__float__": "Infinity"}]` to `[0.0, 1e20]`.

Do not accept a prefix/paged cache hit for this model unless all corresponding
Mamba companion states are present and complete, or unless the engine rederives
the exact missing SSM states before decode.

## Attention KV Cache Contract

The attention layers are sparse and use standard causal GQA:

- attention layer count: `12`
- q heads: `64`
- kv heads: `2`
- head dim: `128`
- q shape after projection: `[batch, 64, tokens, 128]`
- k/v shape after projection: `[batch, 2, tokens, 128]`
- no sliding window
- no MLA/compressor/indexer pool
- no rotating KV

Only the 12 attention entries are eligible for TurboQuant KV cache. The 48
Mamba states are not KV and must be encoded as companion state, not squeezed
through a KV-only codec.

## MoE Runtime Contract

Each MoE layer uses LatentMoE:

1. router gate computes logits from hidden size 8192
2. sigmoid router scores are computed in stable precision
3. group selection uses output/source config `n_group=1`, `topk_group=1`
4. select `num_experts_per_tok=22`
5. normalize selected original sigmoid scores when `norm_topk_prob=true`
6. multiply scores by `routed_scaling_factor=5.0`
7. project hidden through `fc1_latent_proj` to latent size 2048
8. run routed `switch_mlp.fc1/fc2` experts in latent space
9. project routed result through `fc2_latent_proj` back to hidden size 8192
10. add always-active shared expert path

The JANGTQ experts are pre-stacked:

- `backbone.layers.N.mixer.switch_mlp.fc1.tq_packed`
- `backbone.layers.N.mixer.switch_mlp.fc1.tq_norms`
- `backbone.layers.N.mixer.switch_mlp.fc1.tq_bits`
- `backbone.layers.N.mixer.switch_mlp.fc2.tq_packed`
- `backbone.layers.N.mixer.switch_mlp.fc2.tq_norms`
- `backbone.layers.N.mixer.switch_mlp.fc2.tq_bits`

There are 48 `fc1` groups and 48 `fc2` groups.

## TurboQuant Weight Runtime

The model-weight TQ path is not KV compression:

- `tq_packed`: codebook indices packed into `uint32`
- `tq_norms`: per-output-row norms
- `tq_bits`: scalar bit width, `1` for this bundle
- `jangtq_runtime.safetensors`: runtime signs and codebooks
- sidecar keys:
  `codebook.2048.1`, `codebook.5120.1`, `signs.2048.42`, `signs.5120.42`

Runtime decode order for a TQ expert matmul:

1. Hadamard-rotate input using the sidecar signs for logical `in_features`.
2. Unpack `tq_packed` indices according to `tq_bits`.
3. Look up codebook values for `(in_features, bits)`.
4. Multiply by `tq_norms`.
5. Accumulate selected expert rows.

Python loader requirement already implemented in `jang_tools.load_jangtq`:

- recognize `backbone.layers.N.mixer.switch_mlp.{fc1,fc2}.tq_*`
- bypass runtime restacking
- create `TurboQuantSwitchLinear` with logical input width from the existing
  MLX module, not from packed storage width alone

Swift agents should mirror this pre-stacked path and not build a permanent
overlay or per-expert in-memory stack.

## TurboQuant KV Cache Boundary

TurboQuant KV is a cache codec for the 12 attention layer K/V tensors only.
It is orthogonal to JANGTQ model weights.

See the dedicated cache handoff:
`docs/runtime/2026-06-04-nemotron-ultra-cache-block-contract.md`.

Engine requirements:

- cache policy salt must include KV mode, key/value bits, group size, max KV,
  prompt-boundary raw-KV mode, model revision/path, quant profile, parser mode,
  and MTP mode
- raw prompt-boundary K/V should be persisted before generated-token TQ KV
  quantization when the existing cache stack expects that boundary
- TQ KV hit is valid only if the matching 48 Mamba companion states are also
  valid for the same token prefix
- if companion state is missing, either reject the hit or rederive the SSM
  states before decode; do not continue with KV only

Parser and modality state are also cache-correctness inputs. A prefix captured
inside an open reasoning/tool span is not a clean hit unless the parser state
object is captured and restored. This artifact has modality policy `text`; do
not create fake VL/audio cache entries for it.

## Cache Blocks And Async Rederive

Recommended block policy for vMLX agents:

1. Store paged KV blocks for attention layers with the normal parent-hash block
   chain.
2. Store SSM companion state keyed by the exact accepted token prefix and model
   cache-policy salt.
3. Treat a prefix hit as a pair: `(attention KV blocks, Mamba companion state)`.
4. If KV blocks hit but SSM state misses:
   - do not release reusable KV blocks globally
   - mark the request as needing SSM rederive
   - either re-prefill from the longest prefix that has complete companion
     state, or run async/idle rederive to fill the missing companion
   - never decode from stale or absent Mamba state
5. Async rederive must materialize clean `ArraysCache` states after a verified
   prompt boundary, not after rejected draft/spec tokens.
6. Accepting cache blocks means accepting both cache components at the same
   token boundary. A partial hit must downgrade to a shorter complete prefix.

For Swift specifically, keep the known SSM warm-pass issues in mind:

- L1/L2 companion hash formulas must match.
- Store/fetch boundaries must use exact stripped token count, not floor-64
  alignment.
- Inline capture should be wired into the prefill path when possible.
- BatchEngine and DFlash paths need the same companion capture/rederive hooks,
  not only single-stream generation.

## MTP / Spec Decode

The source has one MTP block (`num_nextn_predict_layers=1`,
`mtp_layers_block_type=["attention","moe"]`), but this JANGTQ_1L bundle drops
all `mtp.*` tensors and sets:

- `num_nextn_predict_layers=0`
- `mtp_layers_block_type=[]`
- `runtime.keeps_mtp=false`
- `quantization.drops_mtp=true`

Do not implement speculative decode for this artifact. If a future MTP-preserved
bundle is built, it needs private draft cache state, accept/reject verification,
and rejection-safe discard of draft KV plus draft SSM state. None of that is
part of this 98G bundle.

## Parser / Template Contract

The source template is not generic ChatML only.

- `enable_thinking` defaults to true.
- `enable_thinking=false` still emits `<think></think>` in the generation
  prompt.
- `medium_effort=true` appends `{reasoning effort: efficient}` to the last user
  turn.
- Tool definitions are rendered under `<tools>`.
- Tool calls use Qwen3-coder-style XML:
  `<tool_call><function=name><parameter=arg>...</parameter></function></tool_call>`
- Tool responses use `<tool_response>...</tool_response>`.
- README guidance names reasoning parser `nemotron_v3` / `nemotron_3` and tool
  parser `qwen3_coder`.
- Tool calling with reasoning requires template kwargs equivalent to
  `enable_thinking=true` and `force_nonempty_content=true`.

Current proof shows tag leakage in short no-thinking rows, so do not claim the
parser is done. Fix parser/template routing, not logits, sampler, or forced
tag closers.

The 96-token greedy long probe confirms the same parser issue at longer
budget: simple answers remain semantically correct and warm decode is about
`8.1 tok/s`, but no-thinking rows leak `</think>` and repeat answer text.
Treat this as partial coherence, not production chat readiness.

### Parser Name Nuances

Current JANG and vMLX registries do not all use the same name for the same
wire format:

- JANG `capabilities.py` currently maps `nemotron_h` to
  `reasoning_parser=deepseek_r1`, `tool_parser=nemotron`,
  `think_in_template=true`.
- Both `deepseek_r1` and `qwen3` reasoning parsers handle
  `<think>...</think>`, implicit `</think>` closure, and truncated reasoning.
  Ultra source docs name `nemotron_v3` / `nemotron_3`; vMLX agents should map
  those aliases to the same `<think>` parser behavior and then prove
  no-thinking, thinking, and streaming reset rows.
- Swift `ToolCallFormat.xml_function` is the low-level exact XML function
  parser for `<tool_call><function=name><parameter=...>...`, and its comment
  references `qwen3_coder`.
- Swift `vMLXEngine` also has `NemotronToolCallParser`, registered as
  `nemotron` / `nemotron_h`; it parses the same
  `<tool_call><function=name><parameter=...>` envelope plus JSON body fallback.
- Do not route Ultra to the generic Qwen JSON parser. In Swift registry terms,
  `qwen3_coder` currently aliases to `QwenToolCallParser`, which parses
  `<tool_call>{json}</tool_call>` and is not the Ultra XML-function body.
  Route Ultra tool calls through `nemotron`/`nemotron_h` if that parser remains
  equivalent, or through an explicit `xml_function` / `qwen3_coder_xml`
  alias.

Parser acceptance rows must separate three cases:

- no-thinking visible answer with stray `</think>` stripped from visible
  content and logged as model tag leakage
- thinking-on trace with reasoning routed to `reasoning_content`; if
  `</think>` is not reached before `max_tokens`, visible content is empty and
  the result is marked truncated/unfinished
- tool-call output where natural-language preamble, reasoning, tool call, and
  post-call text are split into the correct OpenAI-compatible fields

## Current Proof

Build and structural proof:

- output size: `98G`
- `mtp.*` output keys: `0`
- TQ pre-stacked groups: `96`
- structural load: `51 shards`, `Expert groups to stack: 0`,
  `pre-stacked: 96`, `Replaced 96 modules`, `LAYERS 108`
- one-token forward: `34.3s`, logits shape `(1, 1, 131072)`

Default-sampler coherence/speed probe:

- log:
  `docs/runtime/logs/2026-06-04-nemotron-ultra-jangtq1l-coherence-speed-probe.json`
- sampler: `temperature=1.0`, `top_p=0.95` from `generation_config.json`
- `nt_math_default`: `4<|im_end|>`, `2` generated tokens, cold TTFT `34.593s`
- `nt_capital_default`: `Tokyo is the capital of Japan...`, `24` generated
  tokens, warm TTFT `1.342s`, decode `3.303 tok/s`
- `think_math_default`: reasoning trace begins correctly but does not reach a
  final answer inside `32` tokens, warm TTFT `1.503s`, decode `3.256 tok/s`

Interpretation: load, forward, and small default-sampler generation work. Speed
was about `3.25-3.30 tok/s` before the Nemotron-specific `SwitchMLP` fusion,
about `3.30-3.44 tok/s` with only the corrected fused path, and about
`8.26-8.34 tok/s` after BF16 activation retention. Coherence is partial: short
factual answers are right, but no-thinking rows can leak `</think>` or repeat,
and thinking rows need parser extraction before they are useful.

## Current Speed Diagnosis

The measured speed is real but not an optimized target.

Live checks after the default-sampler probe:

- macOS memory pressure reported `89%` free.
- The largest non-model resident process was Parallels at about `8.5 GB RSS`.
- The bundle structurally loaded with all 96 routed expert projections as
  `TurboQuantSwitchLinear`.
- Sample patched modules:
  - layer 1 `fc1`: `in=2048`, `out=5120`, `bits=1`,
    packed shape `(512, 5120, 64)`
  - layer 1 `fc2`: `in=5120`, `out=2048`, `bits=1`,
    packed shape `(512, 2048, 160)`
- Original loader patch report before the Nemotron-specific fusion:
  - `Replaced 96 modules`
  - `Patched SwitchGLU class for fused gate+up (0 TQ instances)`
  - `SwitchMLP fused fc1+relu2+fc2 available but disabled`
  - `P15 mx.compile(router-only) applied to 0 MoE class(es)`
  - `P18 QKV fusion: 0 class(es), 0 instances`
- Current structural load report:
  - `Patched SwitchMLP class for fused fc1+relu2+fc2 (48 TQ instances)`
  - `Patched Nemotron-H activation dtype widening (BF16 residual/lm_head inputs)`

Interpretation:

- The model-weight TQ/Hadamard matmul path is active and real:
  `TurboQuantSwitchLinear.__call__` uses `gather_tq_matmul` with packed weights,
  norms, signs, and codebooks.
- Nemotron-H `SwitchMLP(fc1 -> relu2 -> fc2)` is now covered by default using a
  broadcast gather helper for `fc1`, then ReLU-squared, then per-row `fc2`.
- The larger speed bug was activation dtype widening. The layer stack widened
  hidden states to `float32`; BF16 weights then received float32 activations in
  later layers and in `lm_head`.
- Therefore `8.26-8.34 tok/s` is the current Python loader speed row, not
  evidence that the optimized vMLX engine path is complete.
- Closing Slack/Chrome will not turn this into a 40 tok/s row. Reducing memory
  pressure may help TTFT and paging noise, but the main speed work is a
  broader decode-runtime path: lower fixed per-layer overhead, cache-stack
  support, and Swift/Python engine integration.

### Activation Dtype Investigation

Probe logs:

- No-load status summary:
  `jang-tools/examples/nemotron_ultra/runtime_status_report.py`
- No-load speed experiment ranking:
  `jang-tools/examples/nemotron_ultra/speed_experiment_plan.py`
- No-load runtime speed gate:
  `jang-tools/examples/nemotron_ultra/runtime_speed_gate.py`
- No-load proof bundle refresh:
  `jang-tools/examples/nemotron_ultra/refresh_runtime_proof_bundle.py`
- No-load before/after speed compare:
  `jang-tools/examples/nemotron_ultra/compare_runtime_speed_logs.py`
- No-load log-bundle validation:
  `jang-tools/examples/nemotron_ultra/validate_runtime_log_bundle.py`
- Candidate runtime suite:
  `jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py`
  Runs live speed, layer decode, long coherence, Mamba/MoE component, and
  projection tradeoff probes into a candidate log directory, validates the log
  bundle, then refreshes proof and compares against baseline.
- Cold-skewed first layer probe:
  `docs/runtime/logs/2026-06-04-nemotron-ultra-layer-decode-probe.json`
- Warm layer probe before BF16 retention:
  `docs/runtime/logs/2026-06-04-nemotron-ultra-layer-decode-warm-probe.json`
- Warm layer probe after BF16 retention:
  `docs/runtime/logs/2026-06-04-nemotron-ultra-layer-decode-bf16-activation-probe.json`
- Live generator probe after BF16 retention:
  `docs/runtime/logs/2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json`
- Current layer split:
  `docs/runtime/logs/2026-06-04-nemotron-ultra-layer-decode-current-probe.json`
- Greedy sampler check:
  `docs/runtime/logs/2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-greedy-live-probe.json`
- Projection tradeoff check:
  `docs/runtime/logs/2026-06-04-nemotron-ultra-projection-tradeoff-probe.json`
- Mamba component check:
  `docs/runtime/logs/2026-06-04-nemotron-ultra-mamba-component-probe.json`

Findings:

- The first layer probe falsely made `lm_head` look cold-cost dominated, so a
  warmed decode probe was added before drawing conclusions.
- A real post-layer hidden state was `float32`. Feeding it to `lm_head` cost
  about `22-26 ms`; casting the same hidden to BF16 cost about `4-6 ms`.
- Warm synchronized decode improved from `320.10 ms/token` to `144.71 ms/token`.
- Final `norm_f + lm_head` improved from `115.20 ms` to `4.24 ms`.
- Live generator speed improved to about `8.26-8.34 tok/s`.
- Current warmed synchronized split is about `67.53 ms` MoE,
  `62.42 ms` Mamba, `9.32 ms` attention, and `4.84 ms` final norm/lm_head.
- Weighted MoE decode is a small cleanup, not a major fix. It avoids
  materializing `(..., K, latent)` in batch=1 decode and returns the weighted
  latent sum directly. The measured short live row was about `8.23-8.26 tok/s`;
  the layer split was about `65.77 ms` MoE, `64.16 ms` Mamba, `8.99 ms`
  attention, and `4.32 ms` final norm/lm_head.
- Greedy sampling is not faster (`7.32-7.80 tok/s` in the short probe), so the
  remaining bottleneck is model forward, not top-p sampling.
- 8-bit affine projections should stay quantized for speed and size:
  `mamba_in_proj`, `mamba_out_proj`, `shared_up`, and `shared_down` all ran
  faster than temporary BF16 dequantized copies.
- Block RMSNorm already returns BF16 under the activation-retention patch, so
  do not add another mixer-input cast unless a future probe proves a regression.
- BF16 qkv fusion for Nemotron attention was effectively neutral in a one-layer
  A/B. Since attention is now the smallest bucket, do not prioritize it ahead
  of MoE/Mamba.
- Whole-`NemotronHMoE` `mx.compile` is not a safe default yet. A one-off live
  probe spent a long post-load window compiling without producing a first token.
- Mamba decode-only manual depthwise conv was exact but slower than MLX grouped
  `Conv1d`; leave it alone unless replacing it with a proper Metal fused
  conv/update kernel.
- Mamba component timing points first at projection/dispatch overhead:
  first-layer `full_mamba_mixer` about `1.20 ms`, `in_proj` about `0.835 ms`,
  `out_proj` about `0.470 ms`, grouped `conv` about `0.216 ms`, and
  `ssm_update` about `0.190 ms`.
- `skip_params_eval=True` speed probes show cold compile TTFT. With default
  loader warmup enabled, total load including warmup was about `77.44s`, the
  warmup phase was about `33.0s`, first real request TTFT was about `1.01s`,
  and decode was about `7.98 tok/s`.

Runtime controls:

- default: BF16 activation retention enabled
- disable for A/B: `JANGTQ_DISABLE_NEMOTRON_ACTIVATION_BF16=1`
- default: weighted MoE decode enabled
- disable for A/B: `JANGTQ_DISABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH=1`

### SwitchMLP Fast-Path Investigation

A compiled `SwitchMLP(fc1 -> relu2 -> fc2)` path was added for Nemotron-H.

Probe logs:

- Bad first attempt:
  `docs/runtime/logs/2026-06-04-nemotron-ultra-jangtq1l-switchmlp-fastpath-probe.json`
- Broadcast-fixed attempt:
  `docs/runtime/logs/2026-06-04-nemotron-ultra-jangtq1l-switchmlp-fastpath-broadcast-fix-probe.json`
- Broadcast-helper microbench:
  `docs/runtime/logs/2026-06-04-nemotron-ultra-switchmlp-fc1-broadcast-microbench.json`
- Corrected live probe:
  `docs/runtime/logs/2026-06-04-nemotron-ultra-jangtq1l-switchmlp-broadcast-helper-full-live-probe.json`

Findings:

- The first attempt was incorrect: `fc1` is a broadcast gather over K selected
  experts, but the compiled path fed only one input row into a per-row gather
  kernel. Output became token salad.
- The broadcast fix (`mx.repeat(x_rot, k, axis=0)`) restored short-prompt
  coherence, but was slower than the default path at about `2.71-2.87 tok/s`.
- The final helper `make_gather_tq_decode_broadcast(...)` removes that repeat
  cost. Isolated `fc1` gather improved from median `0.573917 ms` to
  `0.436500 ms`.
- The corrected path is enabled by default. Disable only for A/B with
  `JANGTQ_DISABLE_NEMOTRON_SWITCHMLP_FASTPATH=1` or
  `JANGTQ_ENABLE_NEMOTRON_SWITCHMLP_FASTPATH=0`.
- End-to-end decode gain is real but small: old default about
  `3.25-3.30 tok/s`, corrected path about `3.30-3.44 tok/s`.
- Lowering top-k to 8 did not materially improve decode, so do not treat router
  top-k alone as the speed fix.

## Python Engine Checklist

1. Detect this as `model_type=nemotron_h`, not Omni/VL.
2. Normalize config fields: `num_hidden_layers=108`, numeric
   `time_step_limit`.
3. Use `jang_tools.load_jangtq.load_jangtq_model` or mirror its Nemotron
   pre-stacked `fc1/fc2` TQ loader.
4. Cache topology: 12 attention KV entries plus 48 Mamba `ArraysCache` entries.
5. Enable TurboQuant KV only for the attention K/V entries.
6. Reject or rederive when SSM companion is absent; no KV-only hits.
7. Reject or reroute image/audio/video requests for this artifact.
8. Parser: Nemotron-v3 reasoning plus Qwen3-coder XML tools.
9. MTP mode: disabled/dropped for this artifact.
10. Required proof before pass: multi-turn cache hit with companion state,
    parser no-leak row, tool-call row, long decode no-leak/no-repeat row,
    VL/media negative row, token/s row, and memory footprint row.

## Swift Engine Checklist

1. Add/verify `nemotron_h` dispatch with 108-layer hybrid pattern.
2. Add `NemotronUltraConfiguration` normalization for missing
   `num_hidden_layers` and JSON infinity in `time_step_limit`.
3. Implement/verify `NemotronHModel.newCache`: 48 `MambaCache`/Arrays cache
   entries plus 12 KV entries in layer order.
4. Add pre-stacked `switch_mlp.fc1/fc2` JANGTQ expert loading.
5. Add TurboQuant codebook/sign sidecar lookup for `2048` and `5120` logical
   input widths at `1` bit.
6. Ensure MLXPress bundle facts mark this text-only: no media cache gates.
7. Reject or reroute image/audio/video requests for this artifact. Future
   multimodal bundles must salt cache reuse by processor identity, media token
   expansion, encoder state, projector state, and pre-encoded media state.
8. Mark hybrid companion state required for prefix/paged/disk hits.
9. Do not use generic paged KV hit as proof unless SSM companion state is also
   accepted.
10. Parser registration: Nemotron-v3 reasoning and Qwen3-coder XML tools.
11. Required proof before pass: no-thinking row, thinking row, tool-call row,
    three-turn cache row, SSM companion hit/rederive logs, TurboQuant KV logs,
    long decode no-leak/no-repeat row, VL/media negative row, token/s, and
    Activity Monitor `phys_footprint`.
