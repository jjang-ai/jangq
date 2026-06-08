# Nemotron Ultra Runtime Patch Spec

log_dir: `docs/runtime/logs`
current_status: `PARTIAL`

## Current Speed State
- manual_decode_total_ms: `143.237`
- manual_implied_tps: `6.981`
- moe_ms: `65.773`
- mamba_ms: `64.157`
- attention_ms: `8.990`
- norm_lm_head_ms: `4.317`
- moe_mamba_pct: `90.710`

## Target Cuts
- `10.000` tok/s needs `43.237` ms cut; single measured row/path enough: `MoE:full_moe`, `MoE:switch_mlp`, `Mamba:full_mamba_mixer`
- `12.000` tok/s needs `59.904` ms cut; single measured row/path enough: `MoE:full_moe`
- `15.000` tok/s needs `76.571` ms cut; single measured row/path enough: `MoE:full_moe`

## Implementation Lanes
### 1. MoE routed/shared scheduling and switch_mlp path
- id: `moe-routed-shared-scheduling`
- implementation_surface: JANG loader/TurboQuant MoE scheduling only; no bundle expansion.
- why:
  - MoE bucket is 65.773 ms across 48 layers.
  - `switch_mlp` projects to 54.264 ms; 50% cut implies 8.613 tok/s synchronized.
  - `full_moe` is an inclusive path row at 101.148 ms, so path-level scheduling is the highest-leverage MoE target.
- do:
  - preserve router top-k and weighted expert semantics
  - preserve routed expert 1-bit layout and shared expert 8-bit layout
  - reduce per-layer dispatch/synchronization around routed/shared expert execution
  - measure `switch_mlp`, `shared_experts`, and layer E bucket after every candidate
- do_not:
  - do not lower router top-k as the primary speed fix
  - do not expand quantized experts to full precision
  - do not promote a change unless long-coherence counts do not regress
- proof:
  - candidate_command: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
  - post_check_command: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`
  - expected_compare_statuses: `IMPROVED`
  - required_outputs: `2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json`, `2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json`, `2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json`, `2026-06-04-nemotron-ultra-mamba-component-probe.json`, `2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json`, `2026-06-04-nemotron-ultra-projection-tradeoff-probe.json`, `2026-06-04-nemotron-ultra-runtime-speed-compare.json`, `2026-06-04-nemotron-ultra-runtime-speed-gate.json`, `2026-06-04-nemotron-ultra-token-speed-budget.json`, `2026-06-04-nemotron-ultra-agent-handoff.json`, `2026-06-04-nemotron-ultra-runtime-cache-parser-contract.json`

### 2. Mamba projection/dispatch path
- id: `mamba-projection-dispatch`
- implementation_surface: JANG loader/runtime Mamba path only; keep 8-bit affine projections unless new proof reverses it.
- why:
  - Mamba bucket is 64.157 ms across 48 layers.
  - `full_mamba_mixer` projects to 57.480 ms; 50% cut implies 8.734 tok/s synchronized.
  - `in_proj` projects to 40.062 ms; it is larger than conv/SSM update in the saved component probe.
- do:
  - attack projection/dispatch overhead before grouped conv rewrites
  - preserve 8-bit affine projection path unless a new projection tradeoff probe reverses the result
  - preserve Mamba companion cache/state order for hybrid prefix cache compatibility
  - measure M bucket plus `in_proj`, `out_proj`, `conv`, and `ssm_update` after every candidate
- do_not:
  - do not dequantize Mamba projections to BF16 as a default speed fix
  - do not treat attention KV cache work as a substitute for Mamba state proof
  - do not change cache topology without rerunning cache/block handoff checks
- proof:
  - candidate_command: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-mamba-dispatch --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id mamba-projection-dispatch --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
  - post_check_command: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id mamba-projection-dispatch --candidate-log-dir docs/runtime/logs/candidate-mamba-dispatch --out docs/runtime/logs/candidate-mamba-dispatch/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-mamba-dispatch/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`
  - expected_compare_statuses: `IMPROVED`
  - required_outputs: `2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json`, `2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json`, `2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json`, `2026-06-04-nemotron-ultra-mamba-component-probe.json`, `2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json`, `2026-06-04-nemotron-ultra-projection-tradeoff-probe.json`, `2026-06-04-nemotron-ultra-runtime-speed-compare.json`, `2026-06-04-nemotron-ultra-runtime-speed-gate.json`, `2026-06-04-nemotron-ultra-token-speed-budget.json`, `2026-06-04-nemotron-ultra-agent-handoff.json`, `2026-06-04-nemotron-ultra-runtime-cache-parser-contract.json`

## Global Non-Goals
- do not chase attention first while attention is 8.990 ms and below the gate ceiling
- do not hide parser/coherence failures with prompts, forced tags, or sampler tweaks
- do not enable MTP/speculative decode for this MTP-dropped bundle
- do not call a speed lane fixed without compare, gate, and long-coherence proof

## Runtime Controls
- disable_weighted_moe_fastpath: `JANGTQ_DISABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH=1`
- disable_activation_bf16: `JANGTQ_DISABLE_NEMOTRON_ACTIVATION_BF16=1`
- disable_switchmlp_fastpath: `JANGTQ_DISABLE_NEMOTRON_SWITCHMLP_FASTPATH=1`
