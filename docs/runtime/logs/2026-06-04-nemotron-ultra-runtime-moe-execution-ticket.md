# Nemotron Ultra MoE Execution Ticket

log_dir: `docs/runtime/logs`
lane_id: `moe-routed-shared-scheduling`
status: `READY`
candidate_status: `MISSING`
guard_status: `READY`
cleanup_status: `READY`
contract_status: `READY`
manifest_status: `PARTIAL`

## Failures
- none

## Warnings
- none

## Target
- moe_cut_ms_proportional: `21.888`
- moe_cut_pct_of_current_moe: `33.277`
- moe_per_layer_cut_ms: `0.456`
- required_total_cut_ms: `43.237`
- target_tps: `10.000`

## Invariants
- drops_mtp: `True`
- hidden_shape: `[1, 1, 8192]`
- indices_shape: `[1, 1, 22]`
- keeps_latent_moe_bf16: `True`
- keeps_router_gates_source_precision: `True`
- latent_shape: `[1, 1, 2048]`
- routed_expert_bits: `{'down_proj': 1, 'up_proj': 1}`
- routed_shape: `[1, 1, 22, 2048]`
- scores_shape: `[1, 1, 22]`
- shared_expert_bits: `8`

## Execution Order
1. confirm this ticket status is READY
2. run candidate command exactly once for the selected MoE lane
3. run post_check command to write ACCEPTED/REJECTED/BLOCKED verdict
4. rerun candidate index to surface lane status
5. rerun proof refresh to update manifest, ledger, and next runbook

## Commands
- candidate: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
- post_check: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`
- refresh_after_cleanup: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/refresh_runtime_proof_bundle.py --log-dir docs/runtime/logs --summary-out docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-proof-refresh.md`
- post_candidate_index: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/runtime_candidate_index.py --log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --out docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-index.md --json-out docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-index.json`
- post_candidate_refresh: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/refresh_runtime_proof_bundle.py --log-dir docs/runtime/logs --summary-out docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-proof-refresh.md`

## Acceptance Checks
- runtime-speed compare status is IMPROVED.
- MoE bucket drops enough to move target token/s budget without Mamba or coherence regression.
- long coherence leak/repeat/EOS counts do not regress.

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

## Do Not
- do not run the Mamba lane until this MoE lane has accepted evidence
- do not treat an improved short live row as accepted without long coherence and experiment_result_check
- do not change MTP, VL/audio, parser, or hybrid cache assumptions for this speed lane

## Source Files
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-cleanup-ready-check.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-candidate-contract.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-index.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-proof-manifest.json`
