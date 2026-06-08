# Nemotron Ultra Runtime Cleanup Ready Check

log_dir: `docs/runtime/logs`
status: `READY`

## Fixed
- found host readiness: docs/runtime/logs/2026-06-04-nemotron-ultra-host-runtime-readiness.json
- found host cleanup runbook: docs/runtime/logs/2026-06-04-nemotron-ultra-host-cleanup-runbook.json
- found candidate launch guard: docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.json
- host readiness is READY
- host cleanup runbook is READY
- candidate launch guard is READY

## Blockers
- none

## Failures
- none

## Manual Actions
- Stop or close the owning app for unrelated high-RSS model servers before loading the 98G Nemotron bundle.
- Confirm the PID still matches the expected process before stopping anything.
- After cleanup, rerun the refresh command and require this check plus the launch guard to be READY.

## Verify Commands
- refresh: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/refresh_runtime_proof_bundle.py --log-dir docs/runtime/logs --summary-out docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-proof-refresh.md`
- strict_guard: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/runtime_candidate_launch_guard.py --log-dir docs/runtime/logs --strict`
- strict_lane_matrix: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/runtime_lane_readiness_matrix.py --log-dir docs/runtime/logs --strict`

## Source Files
- `docs/runtime/logs/2026-06-04-nemotron-ultra-host-runtime-readiness.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-host-cleanup-runbook.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.json`
