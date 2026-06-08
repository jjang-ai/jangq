# MiMo-V2.5 JANG_2L Quantization Contract

Date: 2026-05-27

## 2026-05-28 Current Proven State

This section supersedes older candidate notes below.

Working local bundle:

```sh
/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L
```

Status:

- structural verifier passes: `109180` tensors, `150` shards, `113.25 GB` payload, `106G` on disk
- promoted on 2026-05-28 from the `JANG_2L_322_D3E16_CPUAFFINE` candidate
- old broken canonical directory preserved as `MiMo-V2.5-JANG_2L-broken-20260528-pre-qkv-cpuaffine`
- MTP dropped: `runtime.mtp_mode=absent`, `bundle_has_mtp=false`
- vision/audio preserved: `visual.*`, `audio_encoder.*`, `speech_embeddings.*`, and `audio_tokenizer/` are present
- routed experts use deterministic CPU-packed min/max affine, not `mx.quantize`
- routed bit plan: layers `1..16` use `gate/up/down = 3/2/3`; layers `17..47` use `3/2/2`
- qkv source loader now deinterleaves the TP=4 on-disk layout before conversion

Root causes fixed:

- Source fused `qkv_proj.weight` is TP=4 rank-interleaved on disk:
  `rank0(Q,K,V), rank1(Q,K,V), rank2(Q,K,V), rank3(Q,K,V)`.
  The MLX runtime expects logical `[all_Q, all_K, all_V]`. Treating the source rows as flat `[Q,K,V]` corrupted attention before quantization.
- The source profile probe used min/max affine, but the converter previously used `mx.quantize` for low-bit routed experts. On a real 2-bit MiMo expert tensor, that produced about `0.75` relative mismatch versus the intended min/max affine dequant. The converter now CPU-packs 2/3/4/5/6/8-bit routed experts into MLX-compatible `uint32 + scales + biases`.

Runtime proof:

```text
verify_bundle /Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L
[verify] config OK: profile=JANG_2L_322_D3E16 routed_bits={'gate_proj': 3, 'up_proj': 2, 'down_proj': 2}
[verify] 109180 tensors across 150 shards, 113.25 GB
[verify] audio_tokenizer/ present
[verify] ✓ bundle passes structural checks

layer_diff_probe --layers 6:
embed rel=0.000000 last_rel=0.000000
layer 00 rel=0.002011 last_rel=0.003583
layer 01 rel=0.052082 last_rel=0.067350
layer 02 rel=0.097741 last_rel=0.114769
layer 03 rel=0.178181 last_rel=0.112273
layer 04 rel=0.378475 last_rel=0.123007
layer 05 rel=0.451865 last_rel=0.281141
```

Generation proof:

```text
Canonical path, Name the capital city of France.  cached --max-tokens 16
prompt_tps=2.835
generation_tps=1.974
peak_memory_gb=114.536
The capital city of France is **Paris**

Canonical path, What is 2 + 2? Answer in one short sentence.  cached --max-tokens 16
prompt_tps=3.444
generation_tps=2.640
peak_memory_gb=114.537
4.

Candidate path before promotion, Name the capital city of France.  --no-cache-greedy --max-tokens 8
The capital city of France is **Paris

Candidate path before promotion, Name the capital city of France.  cached --max-tokens 8
The capital city of France is **Paris

Candidate path before promotion, What is photosynthesis? Answer in one sentence.  cached --max-tokens 24
Photosynthesis is the process by which green plants, algae, and some bacteria convert sunlight, water, and carbon dioxide into
```

Runtime module proof:

```text
layer0.qkv   mlx.nn.layers.quantized.QuantizedLinear bits=8 group=64
layer1.qkv   mlx.nn.layers.quantized.QuantizedLinear bits=8 group=64
layer1.gate  mlx_lm.models.switch_layers.QuantizedSwitchLinear bits=3 group=128
layer1.up    mlx_lm.models.switch_layers.QuantizedSwitchLinear bits=2 group=128
layer1.down  mlx_lm.models.switch_layers.QuantizedSwitchLinear bits=3 group=128
layer17.down mlx_lm.models.switch_layers.QuantizedSwitchLinear bits=2 group=128
lm_head      mlx.nn.layers.linear.Linear BF16 passthrough
```

This runtime-module proof is historical for the deleted affine JANG_2L bundle.
The current rebuild contract requires q8 affine `embed_tokens` and `lm_head`
sidecars.

Speed/headroom proof on the local M5 Max:

- `mx.device_info()["max_recommended_working_set_size"] = 110100 MiB`
- corrected candidate model bytes during `mlx_lm.generate`: `106102 MiB`
- canonical cached France smoke: prompt `2.835 tok/s`, generation `1.974 tok/s`, wall `0.954 tok/s`
- canonical cached math smoke: prompt `3.444 tok/s`, generation `2.640 tok/s`, wall `1.249 tok/s`
- candidate cached photosynthesis smoke after split measurement: prompt `3.080 tok/s`, generation `1.773 tok/s`, wall `0.949 tok/s`
- no-cache greedy France smoke: `0.120 tok/s`; this recomputes the full sequence every token and is only a cache-isolation diagnostic

Wired-limit boundary:

- `mlx_lm.generate` already wraps generation in `mx.set_wired_limit(max_recommended_working_set_size)`.
- On this machine, setting a wired limit above the MLX-reported maximum fails; it is not a valid fix for the near-limit bundle.
- Verified locally on 2026-05-28: `1.05x`, `1.10x`, and `1.20x` of the recommended wired limit all raise `ValueError: Setting a wired limit larger than the maximum working set size is not allowed`.
- The remaining speed problem is practical memory/compute headroom, not a missing wired-limit call. The current affine MLX path is coherent but not a `30 tok/s` path on this local M5 Max.

Rejected faster fit candidate:

```sh
/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L_322_CPUAFFINE-candidate
```

- structural verifier passes: `109180` tensors, `150` shards, `108.95 GB` payload, `102G` on disk
- payload by role: routed experts `97.778 GB`, qkv `3.052 GB`, o_proj `3.221 GB`, visual `1.457 GB`, audio `0.522 GB`
- runtime loaded model bytes: `102006 MiB`
- cached France smoke: coherent, `generation_tps=5.626`, output `The capital city of France is **Paris**.`
- cached photosynthesis smoke: mostly coherent but degraded grammar near the end
- cached arithmetic smoke failed: output `45.` for `What is 2 + 2?`
- conclusion: useful speed/fit evidence, not final

## Source Truth

Local source: `/Volumes/EricsLLMDrive/jangq-ai/sources/MiMo-V2.5`

Verified by `jang-tools/tests/mimo_v2_contract_test.py`:

- `model_type=mimo_v2`
- 48 decoder layers, 256 routed experts, top-8
- Full attention layers: 9
- SWA layers: 39
- Full attention KV heads: 4
- SWA KV heads: 8
- Full qkv shape: `(13568, 4096)` = q `12288`, k `768`, v `512`
- SWA qkv shape: `(14848, 4096)` = q `12288`, k `1536`, v `1024`
- `attention_value_scale=0.707`
- partial RoPE factor `0.334`, applied to the first 64 dims of q/k
- full-attention `rope_theta=10000000`
- SWA `swa_rope_theta=10000`
- SWA attention sink bias is present; full-attention sink is not
- `visual.*`, `audio_encoder.*`, and `model.mtp.*` tensors are present
- text `o_proj` weights are ignored by source FP8 quantization and must remain bf16/passthrough

The upstream README's KV-head table is not the source of truth here. Config plus tensor shapes win.

## Historical Broken Bundle Status

Old bundle, now renamed:
`/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L-broken-20260528-pre-qkv-cpuaffine`

Verified on 2026-05-27:

- structural verifier passes: `109184` tensor keys across `106` shards
- safetensor payload: `104.63 GB`
- Finder/du footprint: `98G`
- `model.mtp.*` tensors removed: `mtp_keys=0`
- vision preserved: `visual_keys=364`
- audio preserved: `audio_encoder_keys=75`, `speech_embeddings=20`, `audio_tokenizer/` present
- lazy `mlx_lm` load passes with the in-tree `mimo_v2` registration
- runtime metadata says `bundle_has_mtp=false`, `mtp_mode=absent`
- generation quality is not proven yet; previous report of gibberish means this is not publish-ready

Decode smoke on 2026-05-27:

- normal `What is 2 + 2?`: emitted arithmetic garbage (`... 11} - -0`)
- `JANG_MIMO_DISABLE_SINK=1`: emitted arithmetic garbage (`-0101 - - -`)
- raw `2 + 2 =`: emitted arithmetic garbage (`- - - -0000 -00 -`)

Layer-local evidence:

- layer 0 MLX vs Torch source reference is close: relative RMSE about `0.032`
- layer 1 MLX vs Torch source reference is much worse: relative RMSE about `0.175`
- layer 1 is the first routed MoE layer, so the remaining issue is routed expert quantization/error accumulation, not sink bias or full-attention RoPE

Failed candidate:

- rebuilt `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L-f16sidecar-candidate` with affine `.scales`/`.biases` stored as `F16` instead of `BF16`
- structural verifier passed and sidecar dtypes were correct
- text smoke still emitted arithmetic garbage (`02 - -1 -01`)
- candidate was deleted to reclaim `98G`

The first runtime ablation switch is:

```sh
JANG_MIMO_DISABLE_SINK=1
```

That switch keeps the 39 SWA sink-bias tensors loadable but bypasses the sink path in attention forward. Use it for a normal-vs-sink-off prompt comparison before deeper layer diffs.

## Source-Allocation Candidate

Candidate built on 2026-05-27:

```sh
/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L-sourcealloc-candidate
```

This candidate is obsolete after the 2026-06-07 vMLX speed/root-cause audit.
Do not rebuild this exact source-allocation profile: leaving `embed_tokens`
and `lm_head` as BF16 made the bundle contradict its affine metadata and forced
vMLX to repair the bookends with post-load runtime quantization.

Corrected rebuild contract:

- `model.embed_tokens`: 8-bit affine group-64 triplet
- `lm_head`: 8-bit affine group-64 triplet
- `jang_config.json`: required
- routed experts: 2-bit affine group-64
- attention qkv and layer-0 dense MLP: 8-bit affine group-64
- text `o_proj`, norms, sinks, visual, audio: BF16 passthrough
- router gate and `e_score_correction_bias`: FP32 passthrough
- MTP: absent

Build result:

- `105` shards
- `105.80 GB` safetensor payload
- affine tensors: `36147`
- BF16 passthrough tensors: `645`
- FP32 passthrough tensors: `94`
- affine bit distribution: `2b=36096`, `8b=51`

No-cache smoke still fails:

```text
<|im_start|>assistant
1 - - - -} -2
```

With `enable_thinking=False`, no-cache smoke still fails:

```text
<think></think>0 - -0 -10 -00 - -
```

Conclusion: preserving source BF16 embed/head was disproven by the later vMLX
audit. Correct rebuilt bundles must emit affine sidecars for both bookends; this
still is not the whole coherence or speed fix.

## JANG_2L Quantization Policy

Use an affine JANG_2L bundle first. Do not start with JANGTQ until the MLX model path is coherent.

Required policy:

- routed experts: affine 2-bit for bulk expert weights
- expert floors: do not let router-critical or residual-sensitive pieces collapse below the established floor; gate/up/down floors must be explicit in metadata
- attention qkv: 8-bit affine
- attention o_proj: bf16 passthrough
- embeddings and lm_head: 8-bit affine group-64 bookends with `.weight`, `.scales`, and `.biases`
- router gate and `e_score_correction_bias`: FP32 passthrough; source tensors are FP32 and routing must not be quantized
- norms and biases: passthrough
- visual tower: passthrough for first working bundle
- audio encoder and audio tokenizer assets: preserve/copy for first working bundle; do not silently drop
- MTP tensors: dropped for the current audio+vision bundle. Do not auto-enable speculative decoding; runtime mode is `absent`.

## JANG_2K / K Variant

Deleted bundle: `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2K`

It was built and verified on 2026-05-27, then deleted because the `122G` on-disk footprint was too large for the fit target.

- structural verifier passes: `109184` tensor keys across `129` shards
- safetensor payload: `129.86 GB`
- Finder/du footprint: `122G`
- `model.mtp.*` tensors removed: `mtp_keys=0`
- vision preserved: `visual_keys=364`
- audio preserved: `audio_encoder_keys=75`, `speech_embeddings=20`, `audio_tokenizer/` present
- lazy `mlx_lm` load passes with the in-tree `mimo_v2` registration
- `routed_expert_bits={"gate_proj": 2, "up_proj": 2, "down_proj": 4}`
- converter bit distribution: `2b=24064`, `4b=12032`, `8b=53`

Rebuild command:

```sh
python -m jang_tools.mimo_v2.convert_jang \
  --src /Volumes/EricsLLMDrive/jangq-ai/sources/MiMo-V2.5 \
  --dst ~/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2K \
  --profile 2k \
  --drop-mtp
```

Policy:

- routed `gate_proj`: 2-bit affine
- routed `up_proj`: 2-bit affine
- routed `down_proj`: 4-bit affine
- attention/embed/lm_head/layer-0 dense: 8-bit affine
- attention `o_proj`, norms, routers, vision, and audio: passthrough as in `JANG_2L`

This is the `2/2/4` K profile for MiMo, but it is not currently present locally
and was not generation-quality-cleared before deletion.

## Smaller Than JANG_2L

Do not call a smaller affine profile `1L`: MLX affine quantization rejects 1-bit and supports only `2, 3, 4, 5, 6, 8` bits. In this repo, the safe smaller affine path is named `JANG_2S`.

Current promoted `JANG_2L` payload by tensor class:

| Class | Payload |
| --- | ---: |
| routed experts | `102.073 GB` |
| attention qkv | `3.052 GB` |
| text attention o_proj | `3.221 GB` |
| embeddings | `1.250 GB` |
| lm_head | `1.250 GB` |
| layer-0 dense MLP | `0.214 GB` |
| visual tower | `1.457 GB` |
| audio encoder + speech embeddings | `0.522 GB` |

The routed experts dominate and are already near the fit floor. A bookend-only
trim is modest; a material speed/size change has to change the routed expert
allocation or move to a JANGTQ/runtime-offload path.

- attention qkv: reduce 8-bit -> 6-bit
- layer-0 dense MLP: reduce 8-bit -> 6-bit
- text attention o_proj: reduce bf16 passthrough -> 8-bit affine
- embeddings and lm_head: q8 affine group-64 sidecars are required; do not ship
  BF16 bookends under affine bundle metadata
- keep routers, norms, sink bias passthrough
- keep visual/audio passthrough for the audio+vision target
- keep MTP absent

Expected size impact: modest, not dramatic. To get a much smaller artifact, use
a JANGTQ/codebook path after affine decode is coherent.

`JANG_2S` build command:

```sh
python -m jang_tools.mimo_v2.convert_jang \
  --src /Volumes/EricsLLMDrive/jangq-ai/sources/MiMo-V2.5 \
  --dst /Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2S \
  --profile 2s \
  --drop-mtp
```

Boundary: `JANG_2S` is still JANG affine, not JANGTQ. It should be treated as a fit experiment and must pass the same normal-vs-sink-off decode smoke before any speed or quality claim.

## JANGTQ Status

Affine `JANG_2L` is now generation-coherent at the promoted path above. Do not
start JANGTQ until the affine baseline is kept as the control and the JANGTQ
candidate is checked against the same structural verifier, layer-diff probe, and
text smokes.

## Runtime Metadata

Both `config.json` and `jang_config.json` must stamp enough data for vMLX to avoid guessing:

- `capabilities.family = "mimo_v2"`
- `capabilities.modalities = ["text", "vision", "audio"]`
- `capabilities.cache_type = "kv"`
- `capabilities.reasoning_parser = "think_xml"`
- `capabilities.tool_parser = "xml_function"` and `supports_tools=true`; the source template emits `<tool_call><function=...><parameter=...>` blocks, which map to vMLX `XMLFunctionParser`
- `capabilities.supports_thinking = true`
- `capabilities.supports_tools = true` only if the tokenizer/template contract is verified
- `runtime.bundle_has_mtp = false`
- `runtime.mtp_mode = "absent"`
- base decode is autoregressive; no native accept/reject speculative decode path is available in this bundle
- include `mxtq_bits` and `routed_expert_bits` style fields even for affine/JANG if downstream code relies on them for bit accounting
- include attention subtype facts: full/SWA layer counts, full/SWA KV heads, qkv split sizes, value scale, SWA window, and sink-bias support
- include cache topology facts: hybrid full/SWA KV, prefix cache supported, L2 disk cache supported, TurboQuant KV only for ordinary full-attention `KVCacheSimple` layers, and native rotating KV for SWA layers

## MLX Model Port Requirements

The MLX port must mirror `modeling_mimo_v2.py`, not MiniMax assumptions:

- fused `qkv_proj` split depends on layer type
- q/k head dim is 192; v head dim is 128
- only q/k receive RoPE
- RoPE applies to 64 dims, then concatenates no-RoPE dims back
- value states are multiplied by `attention_value_scale` before cache update
- SWA mask uses window 128
- SWA sink bias adds an extra softmax column and then drops that probability before multiplying V
- no MTP path in base decode
- vision/audio paths can be load-preserved before full multimodal inference is exposed, but they must not be mislabeled as absent

## vMLX Python Engine Acceptance

Before claiming "ready":

- register/resolve `mimo_v2` in the Python runtime path
- model-config registry returns family `mimo_v2`, cache `kv`, and the intended parser/tool policy
- loader does not auto-enable MTP for this bundle
- cache key includes model config/runtime fingerprint
- prefix cache hit works on a repeated prompt
- paged cache hit works with the asymmetric full/SWA KV dimensions
- L2 disk cache writes and restores
- TQ-native disk path is either proven compatible or explicitly skipped
- live TurboQuant KV auto mode does not replace nonstandard cache slots incorrectly
- a short autoregressive text smoke uses bundle `generation_config.json` defaults, with no hidden sampler clamps or forced thinking text

## vMLX Swift Cache Proof

Current source-side proof on `/Users/eric/vmlx-swift`:

- `Tests/MLXLMTests/MiMoV2FlashCacheTopologyTests.swift` covers `model_type=mimo_v2` dispatch, full/SWA per-layer KV heads, fused qkv splitting, `attention_value_scale`, full-layer-only TurboQuant KV promotion, native L2 disk round trip, TurboQuant-full-layer + rotating-SWA L2 round trip, topology snapshot tags, and `CacheCoordinator` L2 prefix hit/restore for the hybrid full/SWA cache.
- `scripts/vmlx-architecture-cache-proof-check.sh` now requires the MiMo topology suite, full/SWA per-layer proof, full-layer-only TurboQuant KV proof, L2 native/TurboQuant round trips, `CacheCoordinator` L2 prefix hit proof, and Osaurus-facing `mimo_v2 -> xml_function + think_xml` autodetect.
- `ToolCallFormat.infer(from: "mimo_v2") == .xmlFunction`, while older/upstream aliases such as `mimo_v2_flash` stay plain unless bundle metadata explicitly opts in.
- `reasoningStampFromModelType("mimo_v2") == "think_xml"`, while older/upstream aliases stay plain unless bundle metadata explicitly opts in.

Verified locally on 2026-05-27:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swift test --filter MiMoV2FlashCacheTopologyTests --jobs 1 --no-parallel

DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swift test --filter ToolTests/testToolCallFormatInference --jobs 1 --no-parallel

DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swift test --filter ReasoningStampFromModelTypeTests/testMiMoV2GetsThinkXml --jobs 1 --no-parallel

DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swift test --filter VMLXServerRuntimeSettingsTests/automaticRuntimeCachePolicyCoversDownloadedArchitectureFamilies --jobs 1 --no-parallel
```

Boundary: this is cache/parser/runtime-source proof for vMLX Swift. The promoted
Python/MLX affine bundle has coherent text smokes, but the full vMLX runtime path
still needs a model-load generation proof before any Swift serving speed claim.

## Speed Status

Current promoted `JANG_2L` Python/MLX decode measurements on the local M5 Max:

- cached France, 33-token prompt: prompt `2.835 tok/s`, generation `1.974 tok/s`, output `The capital city of France is **Paris**`
- cached math, 40-token prompt: prompt `3.444 tok/s`, generation `2.640 tok/s`, output `4.`
- no-cache greedy France smoke: about `0.120 tok/s`; this intentionally recomputes the full sequence each token and is not a serving-speed number

The loaded modules are on MLX quantized paths:

- attention/bookend affine modules load as `QuantizedLinear`
- routed experts load as `QuantizedSwitchLinear`
- switched expert execution uses `mx.gather_qmm`

So the current speed issue is not an obvious Python expert loop or dense-dequant fallback. Do not claim `30 tok/s`; that is not proven for this coherent affine bundle on the local M5 Max.

Do not publish speed claims until:

- normal, `JANG_MIMO_DISABLE_SINK=1`, and `--no-cache-greedy` prompt smokes identify whether sink/cache are involved
- a coherent decode path runs with the bundle defaults
- token/s is measured separately for prefill and decode
- vMLX prefix cache, paged/L2 cache, and TurboQuant-KV compatibility are tested against the hybrid full/SWA cache topology

## 2026-05-27 Coherence Isolation

Confirmed not primary causes:

- chat template: source template renders the expected MiMo system/user/assistant turns and carries XML tool-call plus `<think>` policy
- parser metadata: bundle capabilities declare `think_xml` and `xml_function`
- MRoPE/2D/3D visual RoPE: text-only failure reproduces with no image/video/audio tokens, so multimodal RoPE is not on the active path
- text RoPE: layer-0 source vs MLX diff was small, and `mx.fast.rope(... traditional=False)` matched the source half-split rotary check
- attention sink: disabling sink still produced gibberish
- cache: no-cache greedy still produced gibberish
- routed gate: MLX and torch selected the same expert set for layer-1 routed MoE on every checked token
- switched quantized matmul: `mx.gather_qmm` matched explicit MLX dequantized expert math within dtype-level error

Current proven allocation/runtime findings:

- source `model.embed_tokens.weight` and `lm_head.weight` are BF16; current built bundle quantized both to 8-bit affine
- hot-swapping only source BF16 `lm_head` did not change bad tokens; hot-swapping source BF16 embed+head changed bad tokens but did not restore coherence, so this is a real bundle bug but not the whole root cause
- source `quantization_config` is FP8 E4M3 `[128,128]` with text `self_attn.o_proj` ignored; qkv extra scale rows are unused padding, not the current bug
- layer-1 attention is clean: current MLX quantized attention vs source attention on the same input is about `0.36%` relative RMSE
- routed gate is clean: MLX and torch selected the same expert set for layer-1 routed MoE on every checked token
- switched quantized matmul is clean: sorted `mx.gather_qmm` matched explicit MLX dequantized expert math on the real 38-token prompt shape
- layer-4 activation spike is real in source math too: source routed MLP max on the checked token was about `1446`, quantized was about `1152`
- selected layer-1 source-vs-2-bit expert outputs show about `16-19%` relative RMSE
- group-size 32 improves only modestly; `2:2:4`/`2K` improves but still leaves roughly `9-13%` relative RMSE on sampled experts

Open proof target: continue layer-by-layer source diff beyond layer 4 using the source-allocation candidate. Do not assert 2-bit inadequacy until a source-faithful layer diff identifies the first output-significant divergence.

## 2026-05-28 Follow-Up Proof

Corrected affine candidates built locally:

| Bundle | Routed bits | Expert group | Payload | Finder/du | Result |
| --- | --- | ---: | ---: | ---: | --- |
| `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L-fixed-candidate` | gate/up/down = `4/2/3` | `128` | `134.19 GB` | `126G` | structural pass; no-cache first token matches source top token, but layer drift remains high |
| `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_3E-candidate` | gate/up/down = `3/3/3` | `128` | `134.19 GB` | about `126G` | structural pass; better layer diff; cached generation OOMs on 128 GB |

The original MiMo `JANG_2L` converter was not implementing current JANG 256-expert rules:

- old: routed experts uniform `2/2/2`, group size `64`
- fixed 2L candidate: routed experts `4/2/3`, group size `128`, tier metadata `8/6/2`
- 3E candidate: routed experts uniform `3/3/3`, group size `128`

Layer-1 source MoE error on the real `What is 2 + 2?` prompt:

| Expert split | Avg expert bits | Layer-1 MoE rel RMSE | Layer-1 last-token rel RMSE |
| --- | ---: | ---: | ---: |
| `2/2/2` | `2.00` | `0.193923` | `0.191402` |
| `4/2/3` | `3.00` | `0.134370` | `0.129612` |
| `2/3/2` | `2.33` | `0.114733` | `0.111938` |
| `2/3/3` | `2.67` | `0.076821` | `0.073441` |
| `3/3/3` | `3.00` | `0.047167` | `0.045627` |
| `4/4/4` | `4.00` | `0.015100` | `0.014628` |

Layer diffs against the source checkpoint:

```text
JANG_2L fixed 4/2/3:
layer 00 rel=0.002660 last_rel=0.002875
layer 01 rel=0.130051 last_rel=0.122613
layer 04 rel=0.880028 last_rel=0.229189

JANG_3E 3/3/3:
layer 00 rel=0.002660 last_rel=0.002875
layer 01 rel=0.059664 last_rel=0.043672
layer 04 rel=0.765754 last_rel=0.150442
layer 11 rel=0.707878 last_rel=0.173356
```

Important template/logit finding:

- The source checkpoint itself predicts the first token `' -'` for the exact rendered `enable_thinking=False` arithmetic prompt ending in `<think></think>`.
- Therefore a one-token dash after that prompt is not by itself proof of incoherence.
- A better coherence gate must use multi-token generation and at least one non-arithmetic prompt.

Runtime fit finding on the local 128 GB Mac:

- `JANG_3E-candidate` loads and verifies, but cached generation fails with:
  `kIOGPUCommandBufferCallbackErrorOutOfMemory`.
- `mlx-lm` warns the model requires `126070 MB`, above the recommended `110100 MB` limit.
- No-cache generation avoids the cache OOM but is too slow for a practical runtime gate on this machine.

Current conclusion:

- Uniform `2/2/2` and corrected `4/2/3` affine 2L are not sufficient by layer-diff proof.
- Uniform `3/3/3` is much closer to source but does not fit cached MLX runtime on 128 GB with audio/vision preserved.
- A smaller mixed profile such as `2/3/2` fits the size direction but drifts badly by layer 4 in source simulation.
- The remaining viable path is not another blind global profile; it needs either a profile that keeps source-close early routed layers while reducing later layers enough to fit, or a JANGTQ/runtime-offload path that makes the `3/3/3`-quality profile fit.

Runtime probe commands:

```sh
PYTHONPATH=/Users/eric/jang/jang-tools \
  /Users/eric/jang/jang-tools/.venv/bin/python \
  jang-tools/examples/mimo_v2/text_smoke.py \
  /Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L \
  --max-tokens 8 --no-cache-greedy

PYTHONPATH=/Users/eric/jang/jang-tools \
  /Users/eric/jang/jang-tools/.venv/bin/python \
  jang-tools/examples/mimo_v2/expert_quant_probe.py \
  --src /Volumes/EricsLLMDrive/jangq-ai/sources/MiMo-V2.5 \
  --bundle /Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L
```

## Current Guard

Run:

```sh
uv run --project jang-tools pytest -q jang-tools/tests/mimo_v2_contract_test.py
```

This verifies the current source contract and the FP8 E4M3 `weight_scale_inv` codec against a real expert tensor.

Current verification commands:

```sh
uv run --project jang-tools python -m jang_tools.mimo_v2.verify_bundle \
  /Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L

uv run --project jang-tools pytest -q \
  jang-tools/jang_tools/mimo_v2/tests/test_fp8_codec.py \
  jang-tools/tests/mimo_v2_contract_test.py

uv run --project jang-tools python -m py_compile jang-tools/jang_tools/mimo_v2/*.py
```

Runtime smoke examples live under:

```sh
jang-tools/examples/mimo_v2/
```
