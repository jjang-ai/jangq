# Nemotron 3 Ultra JANGTQ_1L Build Proof

Date: 2026-06-04

## Artifact

- Source:
  `/Volumes/EricsLLMDrive/sources/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`
- Output:
  `/Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L`
- Output size: `98G`
- Output shards: `51`

## Quantization Policy

- Routed backbone experts: ModelOpt NVFP4 to 1-bit TurboQuant.
- Mamba `in_proj/out_proj`: source FP8 dequantized to 8-bit MLX affine.
- Shared expert `up_proj/down_proj`: source FP8 dequantized to 8-bit MLX
  affine.
- Attention, router gates, LatentMoE projections, embeddings, norms, Mamba
  conv/state tensors, and `lm_head`: passthrough.
- MTP: dropped for the hard-under-128 GiB build.

## Structural Audit

- Output weight keys: `1503`
- `mtp.*` output keys: `0`
- `config.num_nextn_predict_layers`: `0`
- `config.mtp_layers_block_type`: `[]`
- `jang_config.runtime.keeps_mtp`: `false`
- `jang_config.quantization.drops_mtp`: `true`
- `jang_config.capabilities`: stamped as
  `family=nemotron_h`, `modality=text`, `cache_type=hybrid`,
  `reasoning_parser=deepseek_r1`, `tool_parser=nemotron`
- Pre-stacked TQ routed groups: `96`
- `switch_mlp.fc1.tq_packed`: `48`
- `switch_mlp.fc2.tq_packed`: `48`
- `jangtq_runtime.safetensors`: present
- Sidecar keys:
  `codebook.2048.1`, `codebook.5120.1`, `signs.2048.42`, `signs.5120.42`
- `chat_template.jinja`, `tokenizer_config.json`, and
  `generation_config.json`: present in output.

## Loader Fixes

- Added derived `num_hidden_layers=108` to the output config because local
  `mlx_lm.models.nemotron_h.ModelArgs` requires it even though source config
  encodes the layer count through `layers_block_type`.
- Normalized `time_step_limit=[0.0, {"__float__": "Infinity"}]` to
  `[0.0, 1e20]` because MLX `ssm.compute_dt` passes the value to `mx.clip`.
- Added Nemotron-H pre-stacked TQ recognition in `load_jangtq.py` for
  `backbone.layers.N.mixer.switch_mlp.{fc1,fc2}.tq_*`.
- `skip_params_eval=True` now skips automatic JANGTQ warmup for structural
  smoke loads.

## Proof Commands

Targeted tests:

```sh
jang-tools/.venv/bin/pytest \
  jang-tools/tests/nemotron_ultra_converter_contract_test.py \
  jang-tools/tests/nemotron_ultra_loader_contract_test.py \
  jang-tools/tests/nemotron_ultra_parser_contract_test.py \
  jang-tools/tests/nemotron_ultra_artifact_contract_test.py -q
```

Result after parser/artifact-contract coverage was added: `14 passed`.

Structural load:

```sh
PYTHONPATH=jang-tools JANGTQ_WIRED_LIMIT_GB=82 \
jang-tools/.venv/bin/python - <<'PY'
from pathlib import Path
from jang_tools.load_jangtq import load_jangtq_model
p = Path("/Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L")
model, tokenizer = load_jangtq_model(p, skip_params_eval=True)
print("LAYERS", len(model.backbone.layers))
PY
```

Result: load passed with `51 shards`, `TQ groups: 96`,
`Expert groups to stack: 0`, `pre-stacked: 96`, `Replaced 96 modules`,
`LAYERS 108`, and EOS merged to `[2, 11]`.

One-token forward:

```sh
PYTHONPATH=jang-tools JANGTQ_WIRED_LIMIT_GB=105 \
jang-tools/.venv/bin/python <one-token-forward-script>
```

Result: `FORWARD_DONE 34.3`, logits shape `(1, 1, 131072)`.

Tiny generation smoke row:

- Prompt: `What is 2+2? Answer briefly.`
- Template kwargs: `enable_thinking=false`
- Max new tokens: `12`
- Result: `4. P.S. The answer is 4.</think>`
- Runtime: `38.43s`

Interpretation: the bundle loads, executes the real forward path, and produced a
semantically correct answer in the tiny generation row. The emitted closing
`</think>` after a thinking-off prompt means parser/template behavior still
needs cleanup before this is a polished chat runtime.

Default-sampler probe:

- Log:
  `docs/runtime/logs/2026-06-04-nemotron-ultra-jangtq1l-coherence-speed-probe.json`
- Sampler: `temperature=1.0`, `top_p=0.95` from `generation_config.json`
- `nt_math_default`: `4<|im_end|>`, `2` generated tokens, cold TTFT `34.593s`
- `nt_capital_default`: `Tokyo is the capital of Japan...`, `24` generated
  tokens, warm TTFT `1.342s`, decode `3.303 tok/s`
- `think_math_default`: reasoning trace starts correctly, but no final answer
  within `32` tokens, warm TTFT `1.503s`, decode `3.256 tok/s`
- Speed diagnosis: all 96 routed expert projections are
  `TurboQuantSwitchLinear`. Nemotron-H uses
  `SwitchMLP(fc1 -> relu2 -> fc2)`, not `SwitchGLU`, so it needs a
  Nemotron-specific fused path.

Saved-output parser probe:

- Script: `jang-tools/examples/nemotron_ultra/parser_probe.py`
- Log:
  `docs/runtime/logs/2026-06-04-nemotron-ultra-jangtq1l-parser-probe.json`
- `nt_math_default`: visible content `4<|im_end|>`, no reasoning, no tool calls.
- `nt_capital_default`: visible content preserved, stray `</think>` recorded in
  `visible_think_marker_leaks`.
- `think_math_default`: routed to `reasoning_content`, visible content `null`,
  `truncated_reasoning=true`.

SwitchMLP fast-path investigation and fix:

- First attempt log:
  `docs/runtime/logs/2026-06-04-nemotron-ultra-jangtq1l-switchmlp-fastpath-probe.json`
  failed coherence with token-salad output because `fc1` broadcast was handled
  as per-row gather.
- Broadcast-fixed log:
  `docs/runtime/logs/2026-06-04-nemotron-ultra-jangtq1l-switchmlp-fastpath-broadcast-fix-probe.json`
  restored short-prompt coherence but slowed warm decode to about
  `2.71-2.87 tok/s`.
- Final fix: `make_gather_tq_decode_broadcast(...)` avoids
  `mx.repeat(x_rot, k, axis=0)` while preserving the correct broadcast gather
  semantics for `fc1`.
- Microbench:
  `docs/runtime/logs/2026-06-04-nemotron-ultra-switchmlp-fc1-broadcast-microbench.json`
  measured old repeat+per-row median `0.573917 ms` vs new broadcast median
  `0.436500 ms`, a `1.31x` isolated `fc1` gather improvement.
- Live default-enabled corrected path:
  `docs/runtime/logs/2026-06-04-nemotron-ultra-jangtq1l-switchmlp-broadcast-helper-full-live-probe.json`
  measured warm decode about `3.30-3.44 tok/s` on the short rows.
- Decision: corrected `SwitchMLP` fusion is enabled by default. Disable only
  for A/B with `JANGTQ_DISABLE_NEMOTRON_SWITCHMLP_FASTPATH=1` or
  `JANGTQ_ENABLE_NEMOTRON_SWITCHMLP_FASTPATH=0`. This is a modest fix, not a
  complete decode-speed solution.

Activation dtype widening fix:

- Warm layer probe before the fix:
  `docs/runtime/logs/2026-06-04-nemotron-ultra-layer-decode-warm-probe.json`
  measured manual synchronized decode at `320.10 ms`; final
  `norm_f + lm_head` alone was `115.20 ms` because the layer stack widened
  hidden states to `float32`.
- Direct isolation confirmed the same hidden state fed to `lm_head` costs about
  `22-26 ms` as `float32`, but about `4-6 ms` when cast back to BF16.
- Loader fix: Nemotron-H block residual outputs now cast back to the incoming
  BF16 activation dtype, and the final normalized hidden state casts to the
  embedding dtype before `lm_head`.
- Warm layer probe after the fix:
  `docs/runtime/logs/2026-06-04-nemotron-ultra-layer-decode-bf16-activation-probe.json`
  measured manual synchronized decode at `144.71 ms`; final
  `norm_f + lm_head` was `4.24 ms`.
- Live default sampler after the fix:
  `docs/runtime/logs/2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json`
  measured warm decode about `8.26-8.34 tok/s`.
- Disable for A/B with `JANGTQ_DISABLE_NEMOTRON_ACTIVATION_BF16=1`.

## Remaining Gates

- Longer multi-token generation.
- Live reasoning parser extraction for Nemotron-v3 `<think>` traces.
- Live tool-call parser extraction/dispatch for XML function tool calls.
- Cache reuse / multi-turn behavior.
- Quality probe beyond the arithmetic smoke row.
