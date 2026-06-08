# Nemotron Ultra MoE Candidate Contract

log_dir: `docs/runtime/logs`
lane_id: `moe-routed-shared-scheduling`
status: `READY`

## Current Speed
- moe_ms: `65.773`
- mamba_ms: `64.157`
- manual_decode_total_ms: `143.237`
- best_live_tps: `8.335`

## First Target
- target_tps: `10.000`
- required_total_cut_ms: `43.237`
- moe_cut_ms_proportional: `21.888`
- moe_cut_pct_of_current_moe: `33.277`
- moe_per_layer_cut_ms: `0.456`

## MoE Invariants
- hidden_shape: `[1, 1, 8192]`
- indices_shape: `[1, 1, 22]`
- scores_shape: `[1, 1, 22]`
- routed_shape: `[1, 1, 22, 2048]`
- latent_shape: `[1, 1, 2048]`
- routed_expert_bits: `{'down_proj': 1, 'up_proj': 1}`
- shared_expert_bits: `8`
- keeps_latent_moe_bf16: `True`
- keeps_router_gates_source_precision: `True`
- drops_mtp: `True`

## Preconditions
- none

## Do
- preserve router top-k and weighted expert semantics
- preserve routed expert 1-bit layout and shared expert 8-bit layout
- reduce per-layer dispatch/synchronization around routed/shared expert execution
- measure `switch_mlp`, `shared_experts`, and layer E bucket after every candidate

## Do Not
- do not lower router top-k as the primary speed fix
- do not expand quantized experts to full precision
- do not promote a change unless long-coherence counts do not regress

## Acceptance Checks
- runtime-speed compare status is IMPROVED.
- MoE bucket drops enough to move target token/s budget without Mamba or coherence regression.
- long coherence leak/repeat/EOS counts do not regress.

## Commands
- candidate: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
- post_check: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`

## Required Outputs
- `2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json`
- `2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json`
- `2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json`
- `2026-06-04-nemotron-ultra-mamba-component-probe.json`
- `2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json`
- `2026-06-04-nemotron-ultra-projection-tradeoff-probe.json`
- `2026-06-04-nemotron-ultra-runtime-speed-compare.json`
- `2026-06-04-nemotron-ultra-runtime-speed-gate.json`
- `2026-06-04-nemotron-ultra-token-speed-budget.json`
- `2026-06-04-nemotron-ultra-agent-handoff.json`
- `2026-06-04-nemotron-ultra-runtime-cache-parser-contract.json`

## Source Files
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-patch-spec.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-shape-contract.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-token-speed-budget.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-issue-ledger.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.json`
