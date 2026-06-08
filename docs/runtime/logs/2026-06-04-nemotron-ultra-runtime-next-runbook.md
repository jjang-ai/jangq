# Nemotron Ultra Runtime Next Runbook

log_dir: `docs/runtime/logs`
runbook_status: `READY`
current_runtime_status: `PARTIAL`
host_status: `READY`
shape_status: `READY`

## Next Lane
- id: `moe-routed-shared-scheduling`
- kind: `speed_candidate`
- status: `READY`
- title: MoE routed/shared scheduling

## Why This Lane
- MoE bucket is 65.773 ms across 48 layers.
- `switch_mlp` projects to 54.264 ms; 50% cut implies 8.613 tok/s synchronized.
- `full_moe` is an inclusive path row at 101.148 ms, so path-level scheduling is the highest-leverage MoE target.

## Host Cleanup
- Close or stop the high-RSS vMLX server before loading the 98G Nemotron bundle.
- Rerun host_runtime_readiness.py and runtime_lane_readiness_matrix.py after cleanup.
- Proceed with the candidate command only when the selected lane is READY, or consciously accept WATCH noise.

## Do
- preserve router top-k and weighted expert semantics
- preserve routed expert 1-bit layout and shared expert 8-bit layout
- reduce per-layer dispatch/synchronization around routed/shared expert execution
- measure `switch_mlp`, `shared_experts`, and layer E bucket after every candidate

## Do Not
- do not lower router top-k as the primary speed fix
- do not expand quantized experts to full precision
- do not promote a change unless long-coherence counts do not regress

## Commands
- candidate: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
- post_check: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`

## Proof Sequence
- Refresh no-load proof bundle.
- Run runtime_lane_readiness_matrix.py.
- Run exactly one speed_candidate lane.
- Run that lane's post_check_command.
- Accept only IMPROVED compare status with no long-coherence/cache/modality regressions.
