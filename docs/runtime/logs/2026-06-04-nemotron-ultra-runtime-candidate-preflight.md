# Nemotron Ultra Runtime Candidate Preflight

lane_id: `moe-routed-shared-scheduling`
log_dir: `docs/runtime/logs`
status: `READY`

## Fixed
- found manifest: docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-proof-manifest.json
- found host readiness: docs/runtime/logs/2026-06-04-nemotron-ultra-host-runtime-readiness.json
- found shape contract: docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-shape-contract.json
- found patch spec: docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-patch-spec.json
- found experiment queue: docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json
- lane moe-routed-shared-scheduling is present in required lane registries
- manifest status is PARTIAL
- shape contract is READY
- host readiness is READY

## Warnings
- none

## Failures
- none

## Commands
- candidate: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
- dry_run: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96 --dry-run`
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

## Expected Compare Statuses
- `IMPROVED`
