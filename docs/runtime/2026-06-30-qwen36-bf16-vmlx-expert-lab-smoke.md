# Qwen3.6 BF16/vMLX Expert Lab Smoke - 2026-06-30

## Scope

Guarded one-prompt runtime proof for the original BF16/F16 source model:

`/Users/hermes/Documents/Codex/2026-06-22/rea/work/models/Qwen3.6-35B-A3B`

This verifies the Expert Lab Python runner can load the BF16 source through `mlx_lm`/vMLX, trace all Qwen MoE routers, and apply a mask to a selected expert without falling back to the JANGTQ review bundle.

## Environment

- `jang_tools`: 2.5.31
- `mlx`: 0.31.2
- `mlx-lm`: 0.31.3
- Device: `Device(gpu, 0)`
- Source model shape from `config.json`: 40 layers, 256 experts, top-k 8
- Source model size: 67 GB, 26 safetensor shards

## Current Runner Check - 2026-07-01

The current checked-in runner was re-exercised against the same original BF16 source. A deliberately low trace cap (`--max-trace-tokens 128`) now fails honestly:

```text
BF16/vMLX token_trace covers 128 of 720 routed layer-token records; increase --max-trace-tokens for prompt 'full-smoke-hello'
```

With `--max-trace-tokens 1024`, the one-prompt source smoke succeeds:

- artifact root: `/tmp/jang-vmlx-qwen36-smoke-20260701T024711Z-trace1024`
- `runtime_mode: bf16_vmlx`
- `backend: vmlx`
- `runtime_metal_enabled: true`
- `source_model_path`: `/Users/hermes/Documents/Codex/2026-06-22/rea/work/models/Qwen3.6-35B-A3B`
- `hooked_moe_layers: 40`
- `expected_moe_layers: 40`
- `hook_coverage_complete: true`
- `layer_stats` rows: 40
- `token_trace` rows: 720
- output text: `Here`

The same prompt was rerun with a BF16/vMLX mask:

- artifact root: `/tmp/jang-vmlx-qwen36-smoke-20260701T024744Z-masked-l0e0`
- mask: `{"disabled_by_layer": {"0": [0]}}`
- `mask_applied: true`
- `disabled_expert_count: 1`
- `hook_coverage_complete: true`
- `layer_stats` rows: 40
- `token_trace` rows: 720
- disabled-selected leak count: 0

The app's current Swift path passes `--max-trace-tokens` through from `maxTraceTokens`, whose default is `32768`, so the UI default is above the observed 720-record smoke requirement.

## Current Semantic Sidecars - 2026-07-01

The older semantic generations under `/tmp/jang-expert-lab-vmlx-semantic-fulltrace-20260630T132754` are stale under the current evidence contract: rebuilding sidecars from them now fails because they are missing per-prompt decode settings evidence.

A fresh current-format compact semantic run was created instead:

- artifact root: `/tmp/jang-expert-lab-vmlx-semantic-current-20260701T024938Z`
- suite: `/tmp/jang-expert-lab-vmlx-semantic-fulltrace-20260630T132754/suite.jsonl`
- mask: `/tmp/jang-expert-lab-vmlx-semantic-fulltrace-20260630T132754/mask.json`
- sidecars: `/tmp/jang-expert-lab-vmlx-semantic-current-20260701T024938Z/evidence`
- prompt count: 9
- baseline route records: 8,440
- masked route records: 8,440
- eval trace rows: 16,880
- `generation_settings_checked: true`
- `runtime_mode: bf16_vmlx`
- `runtime_backend: vmlx`
- `hooked_moe_layers: 40`
- `expected_moe_layers: 40`
- `hook_coverage_complete: true`
- `mask_applied: true`
- `disabled_expert_count: 1`
- `missing_semantic_coverage: []`
- semantic coverage includes Chinese, non-English, multilingual, translation, English-dominant, and unknown-language-role probes
- `risky_prompt_ids: ["math_arithmetic"]`
- `regression_severity: high`
- `safe_drop_candidates: []`

This is the desired truth-first behavior: current BF16/vMLX sidecar construction succeeds from real source evidence, while the resulting high-risk/no-safe-drop outcome still blocks prune authorization.

## Current Reviewed 50-Prompt Suite - 2026-07-01

A current-format 50-prompt reviewed suite was run against the original BF16 source through vMLX:

- artifact root: `/tmp/jang-vmlx-qwen36-reviewed50-current`
- suite: `/tmp/jang-vmlx-qwen36-reviewed50-current/suite.jsonl`
- prompt count: 50
- semantic coverage: complete, including Chinese, non-English, multilingual, translation, English-dominant, unknown-language-role, and safety/medical/legal probes
- generation depth: 16 tokens baseline and masked mean
- baseline route records: 91,480
- masked route records: 91,480
- eval trace rows: 182,960
- `generation_settings_checked: true`
- `runtime_mode: bf16_vmlx`
- `runtime_backend: vmlx`
- `runtime_metal_enabled: true`
- source model path: `/Users/hermes/Documents/Codex/2026-06-22/rea/work/models/Qwen3.6-35B-A3B`
- `hooked_moe_layers: 40`
- `expected_moe_layers: 40`
- `hook_coverage_complete: true`

The first reviewed one-drop-per-layer mask used the lowest zero-hit expert in each layer:

- mask: `/tmp/jang-vmlx-qwen36-reviewed50-current/mask-one-per-layer-lowhit.json`
- sidecars: `/tmp/jang-vmlx-qwen36-reviewed50-current/evidence-one-per-layer`
- disabled experts: 40
- `risky_prompt_ids: []`
- `high_risk_domains: []`
- `safeDropCandidates`: 40

It authorized a structurally valid 255-expert-per-layer BF16 hard prune:

`/Users/hermes/Documents/Codex/2026-06-22/rea/work/models/Qwen3.6-35B-A3B-BF16-pruned255-reviewed50-20260701`

Structural verification passed (`router_rows_match: true`, `expert_rows_match: true`, 26 shards, 120 shape updates), and the pruned source loaded through vMLX with 40/40 hooks. Same-suite pruned behavior did not pass the reviewed-output gate: max normalized delta was `0.9828` against the reviewed baseline/masked envelope.

A second reviewed one-drop-per-layer mask used the highest zero-hit expert in each layer to minimize expert-ID shifts:

- mask: `/tmp/jang-vmlx-qwen36-reviewed50-current/mask-one-per-layer-highzero.json`
- sidecars: `/tmp/jang-vmlx-qwen36-reviewed50-current/evidence-one-per-layer-highzero`
- disabled experts: 40
- `risky_prompt_ids: []`
- `high_risk_domains: []`
- `safeDropCandidates`: 40

It authorized a second structurally valid BF16 hard prune:

`/Users/hermes/Documents/Codex/2026-06-22/rea/work/models/Qwen3.6-35B-A3B-BF16-pruned255-highzero-reviewed50-20260701`

Structural verification again passed (`router_rows_match: true`, `expert_rows_match: true`, 26 shards, 120 shape updates), and the pruned source loaded through vMLX with 40/40 hooks. Same-suite pruned behavior still did not pass the reviewed-output gate: max normalized delta was `0.9825`, with `english_02` switching from the reviewed `Thinking Process... Deconstruct the Request` wording to `Here's a thinking process... Analyze User Input`.

The product gate was updated to compare pruned BF16/F16 output against the reviewed baseline/masked output envelope instead of the masked text only. This prevents a false block when pruned output matches the original reviewed baseline rather than the masked run. Both hard-pruned outputs above still remain blocked after that relaxation because at least one prompt falls outside both reviewed outputs.

Result: the BF16/vMLX review, mask, sidecar, keep-map, and immutable hard-prune paths are exercised on the real source model, but final JANG/JANGTQ quantization remains correctly locked because pruned-source same-suite behavior has not passed.

## Commands

Baseline:

```bash
RUN_ROOT=/tmp/jang-expert-lab-vmlx-smoke-20260630T132247
mkdir -p "$RUN_ROOT/out"
printf '%s\n' '{"id":"smoke_math","domain":"math","text":"Return only the number: 2 + 2.","tags":["math"],"max_new_tokens":1}' > "$RUN_ROOT/suite.jsonl"
PYTHONPATH=jang-tools python3 -m jang_tools expert-lab-vmlx \
  /Users/hermes/Documents/Codex/2026-06-22/rea/work/models/Qwen3.6-35B-A3B \
  --suite "$RUN_ROOT/suite.jsonl" \
  --output "$RUN_ROOT/out" \
  --max-tokens 1 \
  --emit-token-trace \
  --max-trace-tokens 80 \
  --temperature 0 \
  --top-p 1 \
  --top-k 0
```

Masked rerun using expert 193, observed in baseline layer 0 routing:

```bash
printf '%s\n' '{"disabled_by_layer":{"0":[193]}}' > "$RUN_ROOT/mask.json"
mkdir -p "$RUN_ROOT/masked_out"
PYTHONPATH=jang-tools python3 -m jang_tools expert-lab-vmlx \
  /Users/hermes/Documents/Codex/2026-06-22/rea/work/models/Qwen3.6-35B-A3B \
  --suite "$RUN_ROOT/suite.jsonl" \
  --output "$RUN_ROOT/masked_out" \
  --mask "$RUN_ROOT/mask.json" \
  --max-tokens 1 \
  --emit-token-trace \
  --max-trace-tokens 80 \
  --temperature 0 \
  --top-p 1 \
  --top-k 0
```

## Observed Proof

Baseline summary:

- `ok: true`
- `hooked_moe_layers: 40`
- `expected_moe_layers: 40`
- `hook_coverage_complete: true`
- `runtime_mode: bf16_vmlx`
- `backend: vmlx`
- `runtime_metal_enabled: true`
- `source_model_path` equals the BF16 source path above
- `layer_stats` rows: 40
- `token_trace` rows captured: 80

Mask summary:

- `ok: true`
- `mask_applied: true`
- `disabled_expert_count: 1`
- `disabled_by_layer: {"0": [193]}`
- `hooked_moe_layers: 40`
- `expected_moe_layers: 40`
- `hook_coverage_complete: true`

Baseline layer 0 selected expert 193:

```json
{"token_index":0,"layer":0,"selected_experts":[193,244,253,21,213,129,38,132],"disabled_experts":[],"effective_top_k":8}
```

Masked layer 0 replaced expert 193 and recorded the disable:

```json
{"token_index":0,"layer":0,"selected_experts":[16,244,253,21,213,129,38,132],"disabled_experts":[193],"effective_top_k":8}
```

Post-run artifact inspection found:

- baseline layer 0 expert 193 hits: 2
- masked layer 0 expert 193 hits: 0
- masked trace leak count for expert 193: 0

## Limits

This is a one-prompt, one-token smoke. It proves the real BF16/vMLX load, complete router hook coverage, and mask application path for the Qwen3.6 source. It does not prove the full 50+ prompt reviewed suite, same-suite prune verification, or post-quant verification.

## Historical Semantic Same-Suite Probe - 2026-06-30

After the one-prompt smoke, a compact semantic suite was run through the same BF16/vMLX path to exercise the same-suite baseline/masked shape with required semantic probes represented. These artifacts are historical: the current sidecar builder rejects the old generation rows because they do not contain per-prompt decode settings evidence. Use the 2026-07-01 current semantic sidecars above for the active evidence contract.

Artifact root:

`/tmp/jang-expert-lab-vmlx-semantic-fulltrace-20260630T132754`

The suite contains nine prompts:

- `math_arithmetic`
- `code_swift`
- `format_json`
- `instruction_following`
- `reasoning_logic`
- `safety_medical`
- `chinese_translation`
- `english_dominant`
- `unknown_language_role`

Covered semantic labels:

- `math`
- `code`
- `formatting`
- `instruction_following`
- `reasoning`
- `safety_medical_legal_sensitive`
- `multilingual`
- `non_english`
- `chinese`
- `translation`
- `english_dominant`
- `unknown_language_role`

Commands used the same runner with `--max-tokens 1`, `--emit-token-trace`, and `--max-trace-tokens 4096` for both baseline and masked runs. Expert `238` in layer `0` was selected from observed baseline routing and written to:

`/tmp/jang-expert-lab-vmlx-semantic-fulltrace-20260630T132754/mask.json`

Generated sidecar-style evidence:

- `evidence/comparison_summary.json`
- `evidence/eval.jsonl`
- `evidence/eval_trace.jsonl`
- `evidence/eval_index.json`

The historical evidence was produced with this command shape:

```bash
PYTHONPATH=jang-tools python3 -m jang_tools expert-lab-vmlx-build-eval \
  --suite /tmp/jang-expert-lab-vmlx-semantic-fulltrace-20260630T132754/suite.jsonl \
  --baseline-generations /tmp/jang-expert-lab-vmlx-semantic-fulltrace-20260630T132754/baseline/generations.jsonl \
  --masked-generations /tmp/jang-expert-lab-vmlx-semantic-fulltrace-20260630T132754/masked/generations.jsonl \
  --mask /tmp/jang-expert-lab-vmlx-semantic-fulltrace-20260630T132754/mask.json \
  --output /tmp/jang-expert-lab-vmlx-semantic-fulltrace-20260630T132754/evidence-cli
```

Historical builder output:

- `evidence-cli/comparison_summary.json`
- `evidence-cli/eval.jsonl`
- `evidence-cli/eval_trace.jsonl`
- `evidence-cli/eval_index.json`

Validation results:

- prompt count: 9
- baseline route records: 8,440
- masked route records: 8,440
- total eval trace rows: 16,880
- baseline layer-stat prompt count: 9
- masked layer-stat prompt count: 9
- baseline/masked layer-stat rows per prompt: 40
- `runtime_mode: bf16_vmlx`
- `runtime_backend: vmlx`
- `runtime_metal_enabled: true`
- `hooked_moe_layers: 40`
- `expected_moe_layers: 40`
- `hook_coverage_complete: true`
- `mask_applied: true`
- `disabled_expert_count: 1`
- semantic coverage missing: none
- masked layer 0 expert `238` hits: 0 for every prompt
- reusable builder risk result: `math_arithmetic` is marked high-risk because the one-token masked output differed materially from baseline
- reusable builder safe-drop candidates: none

This historical compact suite showed the intended same-suite baseline/masked artifact shape from the real BF16 source, including Chinese and non-English probes, full routed-layer stats, untruncated token traces, runtime metadata, and no disabled-expert leakage. It is not current prune-authorization evidence because it lacks decode settings in the generation rows. The current 2026-07-01 rerun above preserves the useful proof while satisfying the active decode-settings contract, and still correctly refuses safe-drop authorization because the compact one-token suite has a high-risk prompt and no safe-drop candidates.

## Validator-Gated Prune Workflow Refinement - 2026-07-01

The Expert Lab prune workflow now treats prompt validators as the behavioral authority instead of exact text matching. Exact output deltas remain visible as watch evidence, but a prompt can only contribute safe-drop evidence after the unmasked BF16/vMLX baseline passes that prompt's validator. Baseline-invalid prompts are retained in artifacts and UI classifications, but excluded from prune authorization.

Implemented workflow contract:

- Prompt metadata carries validator kind, expected behavior, source, availability, and reason fields through the vMLX eval sidecars.
- Same-suite baseline, masked, hard-pruned, and post-quant comparisons report validator outcomes and classification counts for `baseline_invalid`, `preserved`, `degraded`, and `inconclusive`.
- Safe-drop gates only count baseline-qualified prompts. A baseline-qualified masked or pruned failure is reported as degraded evidence and blocks the drop until investigated or kept.
- Expert Lab and prequant artifacts expose baseline-qualified counts, pass rates, missing baseline-qualified coverage, degraded prompt IDs, and semantic coverage.
- Preflight and post-convert gates unlock final quantization only after baseline-qualified BF16/vMLX masked verification and hard-pruned source verification pass with no degraded prompts and no missing coverage.

Code paths updated:

- `jang-tools/jang_tools/expert_lab_vmlx.py`
- `jang-runtime/Sources/JANGExpertLab/JANGExpertLab.swift`
- `JANGStudio/JANGStudio/Wizard/ExpertLabSheet.swift`
- `JANGStudio/JANGStudio/Wizard/PrequantPruneSheet.swift`
- `JANGStudio/JANGStudio/Verify/PreflightRunner.swift`
- `JANGStudio/JANGStudio/Verify/PostConvertVerifier.swift`

Verification:

```bash
python3 -m py_compile jang-tools/jang_tools/expert_lab_vmlx.py
swift test --package-path jang-runtime
xcodebuild test -project JANGStudio/JANGStudio.xcodeproj -scheme JANGStudio -destination 'platform=macOS' -only-testing:JANGStudioTests/PreflightRunnerTests -only-testing:JANGStudioTests/PostConvertVerifierTests -only-testing:JANGStudioTests/ExpertPrunePlanBuilderTests -only-testing:JANGStudioTests/ExpertLabWorkflowFlowTests -quiet
xcodebuild build -project JANGStudio/JANGStudio.xcodeproj -scheme JANGStudio -destination 'platform=macOS' -quiet
```

Results:

- Python sidecar builder compile: passed.
- JANG runtime package: 110 tests passed, 16 skipped, 0 failures.
- Focused JANG Studio prune/preflight/post-convert/workflow suite: passed.
- JANG Studio macOS build: passed.

The immutable BF16 source path remained outside the worktree changes:

`/Users/hermes/Documents/Codex/2026-06-22/rea/work/models/Qwen3.6-35B-A3B`

## Fresh Real-Source Validator-Qualified End-to-End Run - 2026-07-01

Active evidence root:

`/Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128`

Immutable source used for all baseline and masked BF16/vMLX evidence:

`/Users/hermes/Documents/Codex/2026-06-22/rea/work/models/Qwen3.6-35B-A3B`

This run supersedes the earlier short marker pilot and the broad thinking-preface pilot. The active suite is a 51-prompt marker-validator suite with `max_new_tokens=128`, so prompt-specific validator markers are visible in BF16/vMLX raw thinking traces and in post-quant no-thinking outputs. The suite includes one intentional baseline-invalid control.

Suite:

- path: `/Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/suite-validator51-marker128.jsonl`
- sha256: `42498fa6c51b55304ba57029e45609a9b59e99b5b2852cd1ec89d9138dd248aa`
- prompt count: 51
- baseline-qualified prompts: 50
- intentional baseline-invalid prompt: `baseline_invalid_math_01`

### Same-Suite BF16/vMLX Baseline

Command log:

`/Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/baseline-command.log`

Command shape:

```bash
PYTHONPATH=jang-tools python3 -m jang_tools --quiet-text expert-lab-vmlx \
  /Users/hermes/Documents/Codex/2026-06-22/rea/work/models/Qwen3.6-35B-A3B \
  --suite /Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/suite-validator51-marker128.jsonl \
  --output /Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/baseline \
  --max-tokens 128 \
  --emit-token-trace --max-trace-tokens 32768 \
  --temperature 0 --top-p 1 --top-k 0
```

Results:

- prompt count: 51
- elapsed: 220.169 seconds
- runtime mode/backend: `bf16_vmlx` / `vmlx`
- source path: `/Users/hermes/Documents/Codex/2026-06-22/rea/work/models/Qwen3.6-35B-A3B`
- routed hook coverage: 40 / 40, complete
- mask applied: false
- generated tokens: 128 per prompt
- validator counts: 50 passed, 1 failed, 0 inconclusive
- baseline-invalid prompt: `baseline_invalid_math_01`

### Same-Suite Masked BF16/vMLX Verification

Fresh mask derived only from the marker-128 baseline route evidence:

`/Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/mask-one-per-layer-highzero-marker128.json`

Mask summary:

- disabled experts: 40
- disabled layers: 40
- zero-hit baseline evidence per selected layer: min 7, max 78
- first layers: layer 0 expert 220, layer 1 expert 229, layer 2 expert 214, layer 3 expert 254, layer 4 expert 237, layer 5 expert 247, layer 6 expert 253, layer 7 expert 252

Command log:

`/Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/masked-command.log`

Command shape:

```bash
PYTHONPATH=jang-tools python3 -m jang_tools --quiet-text expert-lab-vmlx \
  /Users/hermes/Documents/Codex/2026-06-22/rea/work/models/Qwen3.6-35B-A3B \
  --suite /Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/suite-validator51-marker128.jsonl \
  --output /Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/masked \
  --mask /Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/mask-one-per-layer-highzero-marker128.json \
  --max-tokens 128 \
  --emit-token-trace --max-trace-tokens 32768 \
  --temperature 0 --top-p 1 --top-k 0
```

Results:

- prompt count: 51
- elapsed: 226.324 seconds
- runtime mode/backend: `bf16_vmlx` / `vmlx`
- source path: `/Users/hermes/Documents/Codex/2026-06-22/rea/work/models/Qwen3.6-35B-A3B`
- routed hook coverage: 40 / 40, complete
- mask applied: true
- disabled experts: 40
- generated tokens: 128 per prompt

Sidecar build log:

`/Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/build-eval-command.log`

Sidecar outputs:

- comparison: `/Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/evidence/comparison_summary.json`
- eval rows: `/Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/evidence/eval.jsonl`
- eval trace: `/Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/evidence/eval_trace.jsonl`
- eval index: `/Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/evidence/eval_index.json`
- copied suite: `/Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/evidence/suite.jsonl`
- copied mask: `/Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/evidence/mask.json`

Validator-qualified classification results:

- preserved: 50
- baseline_invalid: 1
- degraded: 0
- inconclusive: 0
- baseline-qualified masked pass rate: 1.0
- safe-drop candidates: 40
- risky prompts: none
- high-risk domains: none
- missing semantic coverage: none
- baseline route records: 333,440
- masked route records: 333,440
- eval trace rows: 666,880

The baseline-invalid row `baseline_invalid_math_01` remained visible, had `baselinePassed=false`, `maskedPassed=false`, and `safeDropEvidenceEligible=false`. It did not authorize any safe drops.

### Degraded Negative Control

A negative control mutated the masked output for baseline-qualified prompt `math_01` so its validator marker was absent.

Artifacts:

- mutated masked generations: `/Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/degraded-control/masked-generations-degraded.jsonl`
- degraded evidence: `/Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/evidence-degraded-control`
- reviewed keep-map gate log: `/Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/degraded-control/reviewed-keep-map-gate.log`

Negative-control classification results:

- preserved: 49
- baseline_invalid: 1
- degraded: 1 (`math_01`)
- inconclusive: 0
- baseline-qualified masked pass rate: 0.98
- safe-drop candidates: 0
- regression severity: critical

The production reviewed keep-map gate rejected the degraded control:

`reviewed keep-map failed same-suite comparison gate: masked pass rate 96% is below baseline 98%`

The clean reviewed keep map passed the same gate:

- keep map: `/Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/reviewed-keep-map-marker128.json`
- sha256: `79d7dc9e6839ba26f06763ea59a8374313d7c5296e1ab3fc711b299b49fef2ec`
- gate log: `/Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/reviewed-keep-map-clean-gate.log`
- result: 40 layers, 255 kept experts per layer, method `validator_qualified_bf16_vmlx_safe_drop_v1`

### Hard-Pruned BF16 Source

Command log:

`/Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/prequant-prune-command.log`

Command shape:

```bash
PYTHONPATH=jang-tools python3 -m jang_tools prequant-prune-qwen-moe \
  /Users/hermes/Documents/Codex/2026-06-22/rea/work/models/Qwen3.6-35B-A3B \
  /Volumes/Portable2TB/HermesVault/Models/Qwen3.6-35B-A3B-BF16-pruned255-marker128-20260701T134115Z \
  --keep-map /Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/reviewed-keep-map-marker128.json \
  --require-reviewed-comparison \
  --json
```

Output:

`/Volumes/Portable2TB/HermesVault/Models/Qwen3.6-35B-A3B-BF16-pruned255-marker128-20260701T134115Z`

Results:

- source experts: 256
- pruned experts: 255
- layers: 40
- shape updates: 120
- shard count: 26
- size: 67G
- structural verification: ok
- router rows match: true
- expert rows match: true
- embedded review sidecars: present
- initial pruned-suite verification ready: false, as expected before pruned generation

### Hard-Pruned BF16/vMLX Same-Suite Verification

Command log:

`/Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/pruned-command.log`

Command shape:

```bash
PYTHONPATH=jang-tools python3 -m jang_tools --quiet-text expert-lab-vmlx \
  /Volumes/Portable2TB/HermesVault/Models/Qwen3.6-35B-A3B-BF16-pruned255-marker128-20260701T134115Z \
  --suite /Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/suite-validator51-marker128.jsonl \
  --output /Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/pruned-run \
  --max-tokens 128 \
  --emit-token-trace --max-trace-tokens 32768 \
  --temperature 0 --top-p 1 --top-k 0
```

Bundle-local pruned verification sidecars:

- generations: `/Volumes/Portable2TB/HermesVault/Models/Qwen3.6-35B-A3B-BF16-pruned255-marker128-20260701T134115Z/expert_lab_pruned_generations.jsonl`
- summary: `/Volumes/Portable2TB/HermesVault/Models/Qwen3.6-35B-A3B-BF16-pruned255-marker128-20260701T134115Z/expert_lab_pruned_generation_summary.json`
- review summary: `/Volumes/Portable2TB/HermesVault/Models/Qwen3.6-35B-A3B-BF16-pruned255-marker128-20260701T134115Z/expert_lab_review_summary.json`

Results:

- prompt count: 51
- elapsed: 218.703 seconds
- runtime mode/backend: `bf16_vmlx` / `vmlx`
- source path: `/Volumes/Portable2TB/HermesVault/Models/Qwen3.6-35B-A3B-BF16-pruned255-marker128-20260701T134115Z`
- routed hook coverage: 40 / 40, complete
- generated tokens: 128 per prompt
- pruned suite verification ready: true
- baseline-qualified pruned pass rate: 1.0
- preserved: 50
- baseline_invalid: 1
- degraded: 0
- inconclusive: 0

This hard-pruned source was produced from the immutable BF16 source and did not use the existing JANGTQ review bundle as prune authority.

### Post-Quant Validator Verification

Quantization was unlocked after the clean masked and hard-pruned BF16/vMLX validator gates passed.

Conversion command log:

`/Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/convert-jang4k-command.log`

Command shape:

```bash
PYTHONPATH=jang-tools python3 -m jang_tools convert \
  /Volumes/Portable2TB/HermesVault/Models/Qwen3.6-35B-A3B-BF16-pruned255-marker128-20260701T134115Z \
  -p JANG_4K \
  -o /Volumes/Portable2TB/HermesVault/Models/Qwen3.6-35B-A3B-pruned255-marker128-JANG_4K-20260701T134115Z
```

Converted output:

`/Volumes/Portable2TB/HermesVault/Models/Qwen3.6-35B-A3B-pruned255-marker128-JANG_4K-20260701T134115Z`

Conversion results:

- format: JANG v2
- profile: `JANG_4K`
- actual bits: 3.98
- block size: 128
- loaded shard count in post-quant runtime: 30
- size: 18G
- capabilities: `qwen3_5_moe`, hybrid MoE/SSM, vision

Post-quant verification command log:

`/Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/postquant-marker128-command.log`

Post-quant verification used the same reviewed suite and the same prompt validators, with `enable_thinking=false` to match the app smoke path. The Homebrew Python 3.14 path is still missing `mlx_vlm` and `PIL`, and `pip install --user mlx-vlm==0.6.3` is blocked by PEP 668. The completed post-quant run therefore used the working local Python 3.11 environment:

`/Users/hermes/Documents/Codex/2026-06-28/we/work/orinth-vmlx-py311/bin/python`

That environment has `mlx_lm 0.31.3`, `mlx_vlm 0.6.3`, and `PIL 12.2.0`.

Post-quant sidecars:

- smoke JSONL: `/Volumes/Portable2TB/HermesVault/Models/Qwen3.6-35B-A3B-pruned255-marker128-JANG_4K-20260701T134115Z/expert_lab_smoke.jsonl`
- smoke summary: `/Volumes/Portable2TB/HermesVault/Models/Qwen3.6-35B-A3B-pruned255-marker128-JANG_4K-20260701T134115Z/expert_lab_smoke_summary.json`
- final comparison: `/Volumes/Portable2TB/HermesVault/Models/Qwen3.6-35B-A3B-pruned255-marker128-JANG_4K-20260701T134115Z/expert_lab_final_comparison.json`

Results:

- model load time: 6.514 seconds
- peak RSS: about 20.5 GB
- smoke prompts run: 51
- smoke prompts passed: 51
- smoke failures: none
- post-quant same-suite ready: true
- baseline-qualified prompt count: 50
- post-quant baseline-qualified pass rate: 1.0
- preserved: 50
- baseline_invalid: 1
- degraded: 0
- inconclusive: 0
- runtime mode/backend: `post_quant_jang` / `jang_tools inference`
- format/version: `jang` / `2.0`
- quantization bits: 3, 4, 5, 8
- quantization block size: 128

The baseline-invalid control remained visible in post-quant output:

- prompt: `baseline_invalid_math_01`
- classification: `baseline_invalid`
- baseline qualified: false
- post-quant validator pass: null
- safe-drop evidence eligible: false
- output: `BASELINE_INVALID_VISIBLE remains visible but cannot authorize expert drops.`

### Required Automated Tests

Python compile:

- command log: `/Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/test-py-compile.log`
- command: `python3 -m py_compile jang-tools/jang_tools/expert_lab_vmlx.py`
- result: passed

JANG runtime Swift package:

- command log: `/Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/test-swift-package.log`
- command: `swift test --package-path jang-runtime`
- result: 110 tests executed, 16 skipped, 0 failures

Focused JANGStudio XCTest suites:

- successful command log: `/Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/test-xcode-focused-tmp-arm64-nocov.log`
- successful result bundle: `/Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/test-xcode-focused-tmp-arm64-nocov.xcresult`
- staged source copy: `/tmp/jangq-xcode-test-marker128-20260701T1438Z`
- command shape:

```bash
xcodebuild test \
  -project /tmp/jangq-xcode-test-marker128-20260701T1438Z/JANGStudio/JANGStudio.xcodeproj \
  -scheme JANGStudio \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath /tmp/jangq-xcode-test-marker128-20260701T1438Z/DerivedData \
  -parallel-testing-enabled NO \
  -enableCodeCoverage NO \
  -only-testing:JANGStudioTests/PreflightRunnerTests \
  -only-testing:JANGStudioTests/PostConvertVerifierTests \
  -only-testing:JANGStudioTests/ExpertPrunePlanBuilderTests \
  -only-testing:JANGStudioTests/ExpertLabWorkflowFlowTests
```

Focused XCTest results:

- `ExpertLabWorkflowFlowTests`: 35 tests, 0 failures
- `ExpertPrunePlanBuilderTests`: 19 tests, 0 failures
- `PostConvertVerifierTests`: 85 tests, 0 failures
- `PreflightRunnerTests`: 88 tests, 0 failures
- selected total: 227 tests, 0 failures
- final status: `** TEST SUCCEEDED **`

The focused tests were first attempted from the live checkout with the requested project path and then with explicit `arch=arm64`, serial testing, and code coverage disabled. Both live-checkout attempts were interrupted after the app-hosted XCTest process blocked on a Foundation file read under `~/Documents`; no test failure was reported. The same source tree rsynced to `/tmp` completed the required focused suites successfully. The interrupted logs are preserved at:

- `/Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/test-xcode-focused.log`
- `/Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/test-xcode-focused-arm64-nocov.log`

JANGStudio build:

- command log: `/Volumes/Portable2TB/HermesVault/Artifacts/jang-expert-lab/validator-prune-20260701T134115Z/marker128/test-xcode-build.log`
- command: `xcodebuild build -project JANGStudio/JANGStudio.xcodeproj -scheme JANGStudio -destination 'platform=macOS' -quiet`
- result: passed
- notes: existing compiler warnings remain in `JANGMetalDevice.swift`, `JANGTQMatmul.swift`, and `SafetensorsReader.swift`; no build failure

### Final Status

Fresh real-source validator-qualified end-to-end evidence is complete:

- immutable BF16/vMLX unmasked baseline: passed
- prompt validators applied per prompt: passed
- baseline-invalid prompt visible and excluded from safe-drop evidence: passed
- same-suite masked BF16/vMLX verification: passed
- degraded baseline-qualified negative control blocks prune/quant approval: passed
- hard-pruned BF16 source from immutable-source-derived output: passed
- hard-pruned same-suite BF16/vMLX verification: passed
- post-quant same-suite validator verification: passed under the working Python 3.11 `mlx_vlm` runtime
- required automated tests/build: passed, with focused XCTest rerun from `/tmp` to avoid the live-checkout app-hosted `~/Documents` file-read hang

No degraded baseline-qualified prompts were accepted by the prune or quant gates. No baseline-invalid prompt contributed safe-drop authority.
