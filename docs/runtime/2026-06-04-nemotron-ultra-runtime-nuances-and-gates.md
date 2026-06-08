# Nemotron 3 Ultra Runtime Nuances And Gates

Date: 2026-06-04

Artifact:
`/Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L`

This note is the short "do not forget the sharp edges" checklist for future
JANG, vMLX Python, and vMLX Swift work. The build is real enough to load,
forward, and answer short prompts, but it is not yet a full production chat
runtime.

Related handoffs:

- `docs/runtime/2026-06-04-nemotron-ultra-vmlx-engine-handoff.md`
- `docs/runtime/2026-06-04-nemotron-ultra-cache-block-contract.md`
- `docs/runtime/2026-06-04-nemotron-ultra-long-coherence-and-vl-proof.md`
- `jang-tools/examples/nemotron_ultra/README.md`

No-load status command:

```sh
PYTHONPATH=jang-tools \
  jang-tools/.venv/bin/python \
  jang-tools/examples/nemotron_ultra/runtime_status_report.py \
  --log-dir docs/runtime/logs
```

Use this first before rerunning the 98G bundle. It summarizes current speed,
coherence, layer split, projection tradeoff, cache, and VL gates from saved
probe logs.

No-load speed experiment plan:

```sh
PYTHONPATH=jang-tools \
  jang-tools/.venv/bin/python \
  jang-tools/examples/nemotron_ultra/speed_experiment_plan.py \
  --log-dir docs/runtime/logs \
  --out docs/runtime/logs/2026-06-04-nemotron-ultra-speed-experiment-plan.md
```

Use this before touching runtime code. It ranks the next MoE/Mamba speed
experiments from measured logs and lists the negative controls that current
evidence says not to chase.

No-load speed gate:

```sh
PYTHONPATH=jang-tools \
  jang-tools/.venv/bin/python \
  jang-tools/examples/nemotron_ultra/runtime_speed_gate.py \
  --log-dir docs/runtime/logs \
  --out docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-gate.md
```

Use `--strict` in CI-style checks when partial bottlenecks should fail the
command. Without `--strict`, the gate writes the current FIXED/PARTIAL/BLOCKED
state and exits successfully if no required proof log is missing.

No-load proof bundle refresh:

```sh
PYTHONPATH=jang-tools \
  jang-tools/.venv/bin/python \
  jang-tools/examples/nemotron_ultra/refresh_runtime_proof_bundle.py \
  --log-dir docs/runtime/logs \
  --summary-out docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-proof-refresh.md
```

This regenerates the status report, speed experiment plan, and speed gate with
consistent paths. Add `--strict-gate` when a PARTIAL speed gate should make the
wrapper exit nonzero.

No-load before/after compare:

```sh
PYTHONPATH=jang-tools \
  jang-tools/.venv/bin/python \
  jang-tools/examples/nemotron_ultra/compare_runtime_speed_logs.py \
  --baseline-log-dir docs/runtime/logs \
  --candidate-log-dir docs/runtime/logs \
  --out docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-compare.md
```

Use this after a runtime change by pointing `--candidate-log-dir` at the new
probe logs. It compares token/s, manual decode total, MoE/Mamba/attention
buckets, final `lm_head`, and long-coherence leak/repeat/EOS counts.

No-load log-bundle validation:

```sh
PYTHONPATH=jang-tools \
  jang-tools/.venv/bin/python \
  jang-tools/examples/nemotron_ultra/validate_runtime_log_bundle.py \
  --log-dir docs/runtime/logs
```

Run this before comparing a candidate directory. It verifies that the live
speed, layer decode, long coherence, Mamba component, MoE component, and
projection tradeoff logs are present and contain the required metrics.

Candidate runtime suite:

```sh
PYTHONPATH=jang-tools \
  jang-tools/.venv/bin/python \
  jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py \
  --candidate-log-dir docs/runtime/logs/candidate-YYYYMMDD-HHMM \
  --baseline-log-dir docs/runtime/logs
```

This is not no-load: it runs the real live speed, layer decode, long coherence,
Mamba component, MoE component, and projection tradeoff probes before validating
the candidate log bundle, regenerating the candidate proof bundle, and running
the baseline comparison. Use
`--skip-model-probes` only as a wrapper smoke test.

## Current Status

- FIXED: source download, conversion, output bundle under 128 GiB, MTP dropped.
- FIXED: Python structural load for 108 layers and 96 pre-stacked routed
  JANGTQ expert projections.
- FIXED: current bundle `jang_config.json` is stamped with vMLX-facing
  `capabilities` (`family=nemotron_h`, `modality=text`, `cache_type=hybrid`,
  `reasoning_parser=deepseek_r1`, `tool_parser=nemotron`).
- FIXED: real one-token forward with logits shape `(1, 1, 131072)`.
- FIXED: short coherence smoke rows: arithmetic and factual answer are
  semantically correct in the default path.
- PARTIAL: speed. Corrected Nemotron `SwitchMLP` fusion plus BF16 activation
  retention move warm short-row decode to about `8.26-8.34 tok/s`, but this is
  still below the optimized target.
- PARTIAL: parser/template. No-thinking rows can emit stray `</think>`; parser
  routing must remove markers from visible content while logging the leak.
- PARTIAL: thinking. Thinking traces start correctly, but short budgets may not
  reach final visible content.
- PARTIAL: long coherence. The greedy 96-token probe preserves simple factual
  and arithmetic answers at about `8.1 tok/s`, but visible `</think>` leakage
  and answer repetition remain.
- PARTIAL: tools. Source template uses XML function calls; parser contract is
  documented and tested, but no live tool-dispatch row has passed yet.
- BLOCKED for production: multi-turn cache reuse with complete Mamba companion
  state has not been proven.

## Bit Layout

Do not describe this bundle as "1-bit everything".

- Routed expert `up_proj/down_proj`: 1-bit TurboQuant.
- Mamba `in_proj/out_proj`: 8-bit MLX affine from source FP8.
- Shared expert `up_proj/down_proj`: 8-bit MLX affine from source FP8.
- Attention q/k/v/o: BF16 passthrough.
- Router gates and correction bias: source precision.
- LatentMoE `fc1_latent_proj/fc2_latent_proj`: BF16 passthrough.
- Embeddings, norms, Mamba conv/state tensors, and `lm_head`: passthrough.
- MTP tensors: dropped.

The output is about `98G`; source NVFP4 is about `328G`.

## Layer And Cache Topology

Layer count is 108 with this repeating hybrid pattern:

- 48 Mamba layers
- 48 MoE layers
- 12 attention layers

Cache list length is 60, not 108:

- 48 Mamba `ArraysCache(size=2)` entries
- 12 attention KV entries
- MoE layers have no cache entry

TurboQuant KV cache applies only to the 12 attention K/V tensors. Mamba states
are companion state and must be keyed, accepted, rejected, and rederived as
their own object. A prefix hit is valid only when attention KV and all required
Mamba companion states match the same token boundary and cache-policy salt.

## SSM Naming Trap

The config has both `n_group` and `n_groups`.

- `n_group=1`, `topk_group=1`: MoE router group selection.
- `n_groups=8`: Mamba SSM grouping.

Do not use MoE `n_group` for SSM state shape.

Also, config `intermediate_size=5120` is the MoE expert width. The local
Mamba mixer computes its own effective intermediate width:

`mamba_num_heads * mamba_head_dim = 256 * 64 = 16384`

Then:

`conv_dim = 16384 + 2 * 8 * 128 = 18432`

## MoE/TurboQuant Runtime

Each MoE layer is LatentMoE:

1. hidden 8192 to latent 2048 through `fc1_latent_proj`
2. route top-22 of 512 experts
3. expert `fc1`: latent 2048 to expert width 5120
4. ReLU squared
5. expert `fc2`: 5120 to latent 2048
6. `fc2_latent_proj` returns to hidden 8192
7. shared expert path is added

The JANGTQ keys are pre-stacked:

- `backbone.layers.N.mixer.switch_mlp.fc1.tq_*`
- `backbone.layers.N.mixer.switch_mlp.fc2.tq_*`

Sidecar widths:

- `signs.2048.42`, `codebook.2048.1` for `fc1`
- `signs.5120.42`, `codebook.5120.1` for `fc2`

The Hadamard/TurboQuant matmul is correct in the default path. Nemotron
`SwitchMLP(fc1 -> relu2 -> fc2)` is now fused by default with a broadcast gather
helper for `fc1`. The first coherent experiment used
`mx.repeat(x_rot, k, axis=0)` and was slower; the current helper preserves the
same broadcast semantics without materializing repeated rows.

Runtime controls:

- default: corrected `SwitchMLP` fused path enabled
- disable for A/B: `JANGTQ_DISABLE_NEMOTRON_SWITCHMLP_FASTPATH=1`
- legacy disable form: `JANGTQ_ENABLE_NEMOTRON_SWITCHMLP_FASTPATH=0`

Measured effect:

- isolated `fc1` broadcast gather: `0.573917 ms` repeat path to `0.436500 ms`
- short live rows: old default about `3.25-3.30 tok/s`; corrected fused path
  about `3.30-3.44 tok/s`
- `JANGTQ_TOPK_OVERRIDE=8` did not materially improve decode, so the remaining
  bottleneck is fixed per-layer TQ/Hadamard/latent/shared/Mamba overhead, not
  just top-k expert count

## Activation Dtype Runtime

The larger speed bug was activation dtype widening, not RAM pressure. The layer
stack widened decode hidden states to `float32`; BF16 weights then received
float32 activations in later layers and in `lm_head`.

Measured evidence:

- warmed synchronized decode before BF16 retention:
  `320.10 ms/token`
- final `norm_f + lm_head` before BF16 retention:
  `115.20 ms`
- direct `lm_head` isolation on a real post-layer hidden state:
  `22-26 ms` as `float32`, `4-6 ms` as BF16
- warmed synchronized decode after BF16 retention:
  `144.71 ms/token`
- final `norm_f + lm_head` after BF16 retention:
  `4.24 ms`
- live generator probe after BF16 retention:
  `8.26-8.34 tok/s`
- fresh short live probe with current defaults:
  `8.16 tok/s`
- weighted MoE short live probe:
  `8.23-8.26 tok/s`
- greedy sampler probe:
  `7.32-7.80 tok/s`, so top-p sampling is not the main remaining bottleneck

Loader behavior:

- default: Nemotron-H block residual outputs cast back to incoming BF16 dtype
- default: final normalized hidden state casts to embedding dtype before
  `lm_head`
- disable for A/B: `JANGTQ_DISABLE_NEMOTRON_ACTIVATION_BF16=1`

This does not change the bundle weights. It keeps the runtime activation dtype
consistent with the BF16/pass-through tensor policy and avoids accidental
float32 matmul paths.

## Current Remaining Speed Split

After BF16 activation retention, warmed synchronized decode is about
`144 ms/token`. With weighted MoE decode enabled, the current layer split is:

- MoE layers: `65.77 ms` total across 48 layers
- Mamba layers: `64.16 ms` total across 48 layers
- attention layers: `8.99 ms` total across 12 layers
- final norm/lm_head: `4.32 ms`

The no-load report script prints this split from the saved weighted-MoE layer
probe log.

The no-load speed experiment plan ranks the next credible runtime work:

1. MoE routed/shared scheduling or fused decode kernel.
2. Mamba projection/dispatch fusion or lower-overhead state update.
3. Joint MoE+Mamba scheduling path.
4. Ahead-of-time warmup plan for TTFT, not steady tok/s.

Do not chase attention first. It is now the smallest layer bucket.

Do not dequantize 8-bit affine Mamba/shared projections as a speed fix. The
projection tradeoff probe
`docs/runtime/logs/2026-06-04-nemotron-ultra-projection-tradeoff-probe.json`
shows quantized affine is faster than temporary BF16 copies:

- `mamba_in_proj`: quantized `0.95 ms`, BF16 `1.37 ms`
- `mamba_out_proj`: quantized `0.51 ms`, BF16 `0.67 ms`
- `shared_up`: quantized `0.35 ms`, BF16 `0.47 ms`
- `shared_down`: quantized `0.36 ms`, BF16 `0.54 ms`

Next plausible speed work is lower-overhead MoE/Mamba scheduling or custom
fused decode kernels, not RAM-heavy BF16 expansion and not sampler changes.

Weighted MoE decode:

- default: enabled
- disable for A/B: `JANGTQ_DISABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH=1`
- behavior: for batch=1/small-top-k decode, `SwitchMLP` returns the
  score-weighted latent sum directly instead of materializing the full
  `(..., K, latent)` tensor and running a separate weighted-sum kernel
- effect: small/noisy positive row, not a major speed fix; layer probe MoE
  total moved from about `67.53 ms` to `65.77 ms`, live short row moved to
  about `8.23-8.26 tok/s`

Additional negative checks:

- Block RMSNorm outputs are already BF16 under the activation-retention patch;
  there is no remaining float32 mixer-input bug to fix in the current loader.
- BF16 attention qkv fusion was effectively neutral on one attention layer
  (`~0.72 ms` original vs `~0.71 ms` fused without cache), and attention is only
  about `9.32 ms` total across the 12 attention layers.
- One-off whole-`NemotronHMoE` `mx.compile` was not promoted: on-demand compile
  did not produce a first token after a long post-load compile window, so it is
  unsuitable as a default runtime path without a separate ahead-of-time warmup
  design.
- Mamba decode-only manual depthwise conv was exact but slower than MLX
  grouped `Conv1d`. The generic conv path is already good for this kernel
  shape (`conv_dim=18432`, kernel `4`, groups `18432`), so do not replace it
  with Python-level multiply/sum unless a Metal fused kernel is written.
- The Mamba component probe
  `docs/runtime/logs/2026-06-04-nemotron-ultra-mamba-component-probe.json`
  puts first-layer `full_mamba_mixer` at about `1.20 ms`, with `in_proj`
  about `0.835 ms`, `out_proj` about `0.470 ms`, grouped `conv` about
  `0.216 ms`, and `ssm_update` about `0.190 ms`. That points first at
  projection/dispatch fusion, not a Python-level conv rewrite.
- Manual cached argmax generation did not beat `mlx_lm.generate_step`.
  After warmup on the same prompt, `generate_step` measured about
  `8.35-8.39 tok/s`, while a direct cached `model(token, cache)` argmax loop
  measured about `7.77-7.81 tok/s`. The generation wrapper is not the current
  speed bottleneck.
- The no-load runtime speed gate currently reports PARTIAL: best live speed
  clears `8.0 tok/s`, attention and final `lm_head` buckets stay under their
  ceilings, but MoE and Mamba both remain over `40 ms` and long-coherence
  parser/repetition checks still fail.

## Startup Warmup

Do not confuse `skip_params_eval=True` probe TTFT with production warmed TTFT.
The live speed probes skip loader warmup to keep repeated experiments bounded,
so their first row often shows about `34s` cold TTFT from JIT/compile startup.

With default loader warmup enabled (`skip_params_eval=False`):

- total load including warmup: about `77.44s`
- built-in warmup phase: about `33.0s`
- first real request TTFT after warmup: about `1.01s`
- short decode row: about `7.98 tok/s`

The warmup is doing useful work: it moves the cold compile cost out of the
first user request. The remaining startup problem is load/warmup duration, not
steady request TTFT.

## Parser And Template

The source template is not plain ChatML:

- `enable_thinking` defaults true.
- `enable_thinking=false` still emits a closed `<think></think>` assistant
  prefill.
- `medium_effort=true` appends `{reasoning effort: efficient}`.
- Tool definitions are inside `<tools>`.
- Tool calls are XML function calls:
  `<tool_call><function=name><parameter=arg>...</parameter></function></tool_call>`.
- Tool responses use `<tool_response>...</tool_response>`.

Parser naming is inconsistent across stacks:

- Source docs name `nemotron_v3` / `nemotron_3` reasoning and `qwen3_coder`
  tools.
- JANG currently stamps `nemotron_h` as `deepseek_r1` reasoning and
  `nemotron` tools.
- Swift `ToolCallFormat.xml_function` is the exact low-level XML function
  parser.
- Swift `NemotronToolCallParser` also parses the same XML function envelope.
- Swift `QwenToolCallParser` is not enough for Ultra XML-function bodies
  because it expects `<tool_call>{json}</tool_call>`.

Acceptance requires separate no-thinking, thinking, streaming-reset, and tool
rows. Do not hide parser bugs with forced sampler settings or prompt suffixes.

## Long Decode Coherence

The current bounded long probe is documented in
`docs/runtime/2026-06-04-nemotron-ultra-long-coherence-and-vl-proof.md`.

The result is PARTIAL:

- no-thinking factual and arithmetic rows contain the expected answers
- EOS is reached on those rows
- warm decode is about `8.1 tok/s`
- visible `</think>` markers leak into raw decoded text
- answer text repeats before EOS
- the thinking row reaches the correct arithmetic answer in reasoning but does
  not reach EOS within 96 generated tokens

Do not treat this as a parser fix. The raw model output still needs reasoning
and tool parser routing before it is acceptable for chat display.

## VL Processing Boundary

This artifact is text-only. The source snapshot has no processor/preprocessor
config files and no source tensor keys matching vision, audio, image, video,
encoder, or projector namespaces.

vMLX must not open a VL/audio cache lane for this artifact. Media inputs should
be rejected or routed to a different multimodal bundle. Future Omni/VL/audio
bundles need a larger accepted-prefix tuple that includes processor identity,
media token expansion, encoder state, projector state, and any pre-encoded
media cache state.

## MTP

The source has MTP tensors, but this bundle intentionally drops them:

- `num_nextn_predict_layers=0`
- `mtp_layers_block_type=[]`
- output `mtp.*` keys: `0`

Do not add speculative decode for this artifact. A future MTP-preserved bundle
would need private draft KV and private draft SSM state with rejection-safe
discard.

## Real-Working Gate

Before calling this fully working, prove:

- load and one-token forward after a cold start
- no-thinking row with no visible tag leakage
- thinking row with reasoning/content split and truncated reasoning marked
- XML tool-call row parsed into structured tool calls
- long decode row with no visible marker leakage and no runaway repetition
- three-turn chat row
- prefix cache hit where attention KV and Mamba companion state both hit
- prefix cache miss/rederive row when Mamba state is absent
- TurboQuant KV row limited to the 12 attention layers
- VL/media negative row proving this text-only artifact rejects or reroutes
  media requests instead of creating fake media cache state
- memory footprint row under the target machine constraint
- speed row after warm compile with the default path and any proposed fast path
