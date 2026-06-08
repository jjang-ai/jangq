# Nemotron Ultra Runtime Candidate Launch Guard

log_dir: `docs/runtime/logs`
status: `READY`
allow_watch: `False`
runbook_status: `READY`
host_status: `READY`

## Selected Lane
- id: `moe-routed-shared-scheduling`
- kind: `speed_candidate`
- status: `READY`
- candidate_index_status: `MISSING`
- title: MoE routed/shared scheduling

## Warnings
- none

## Failures
- none

## Commands
- candidate: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
- post_check: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`
- refresh_after_cleanup: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/refresh_runtime_proof_bundle.py --log-dir docs/runtime/logs --summary-out docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-proof-refresh.md`

## Source Files
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-next-runbook.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-index.json`
