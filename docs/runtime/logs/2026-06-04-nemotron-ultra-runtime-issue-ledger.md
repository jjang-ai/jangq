# Nemotron Ultra Runtime Issue Ledger

log_dir: `docs/runtime/logs`
status: `OPEN`
current_runtime_status: `PARTIAL`

## Status Counts
- OPEN: `3`
- BLOCKED: `0`
- FIXED: `2`

## Target Summary
- 10.000 tok/s needs 43.237 ms total cut (MoE 21.888 ms, Mamba 21.350 ms proportional).
- 12.000 tok/s needs 59.904 ms total cut (MoE 30.325 ms, Mamba 29.579 ms proportional).
- 15.000 tok/s needs 76.571 ms total cut (MoE 38.762 ms, Mamba 37.809 ms proportional).

## Issues

### NU-SPEED-001: MoE routed/shared decode path dominates token latency
- status: `OPEN`
- severity: `critical`
- evidence:
  - MoE bucket is 65.773 ms.
  - Current next speed lane is `moe-routed-shared-scheduling`: MoE routed/shared scheduling.
  - 10.000 tok/s needs 43.237 ms total cut (MoE 21.888 ms, Mamba 21.350 ms proportional).
- next_actions:
  - Run exactly one MoE scheduling candidate after host readiness is READY or WATCH is accepted.
  - Preserve router top-k, weighted expert semantics, routed 1-bit experts, and shared 8-bit experts.
  - Compare candidate logs with compare_runtime_speed_logs.py and experiment_result_check.py.
- acceptance_checks:
  - runtime-speed compare status is IMPROVED.
  - MoE bucket drops enough to move target token/s budget without Mamba or coherence regression.
  - long coherence leak/repeat/EOS counts do not regress.
- source_files:
  - `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-gate.json`
  - `docs/runtime/logs/2026-06-04-nemotron-ultra-token-speed-budget.json`
  - `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-next-runbook.json`

### NU-SPEED-002: Mamba projection/dispatch path dominates token latency
- status: `OPEN`
- severity: `critical`
- evidence:
  - Mamba bucket is 64.157 ms.
  - Speed gate records projection/dispatch as the current Mamba target before conv rewrite.
  - 12.000 tok/s needs 59.904 ms total cut (MoE 30.325 ms, Mamba 29.579 ms proportional).
- next_actions:
  - Keep this as the second speed-candidate lane after MoE scheduling evidence is gathered.
  - Preserve projected shape [1, 1, 35072], gate shape [1, 1, 16384], SSM state size 128, groups 8.
  - Recheck mamba_component_probe.py and layer_decode_probe.py after any candidate.
- acceptance_checks:
  - Mamba bucket drops without changing cache cardinality or Mamba shape contract.
  - Attention and norm/lm_head remain below current ceilings.
  - long coherence leak/repeat/EOS counts do not regress.
- source_files:
  - `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-gate.json`
  - `docs/runtime/logs/2026-06-04-nemotron-ultra-token-speed-budget.json`
  - `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-shape-contract.json`

### NU-COHERENCE-001: Long decode still leaks/repeats or misses EOS
- status: `OPEN`
- severity: `high`
- evidence:
  - MoE remains a bottleneck at 65.773 ms Mamba remains a bottleneck at 64.157 ms coherence gate remains partial (leaks=['factual_japan', 'arithmetic_brief', 'reasoning_apples'], repeats=['factual_japan', 'arithmetic_brief'], no_eos=['reasoning_apples'])
- next_actions:
  - Treat coherence as a regression gate for speed candidates, not a sampler/prompt masking target.
  - Use long_decode_coherence_probe.py after any accepted speed candidate.
- acceptance_checks:
  - No visible thinking marker leaks.
  - Repeat fraction stays below gate threshold.
  - Expected rows reach EOS within the probe limit.
- source_files:
  - `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-gate.json`
  - `docs/runtime/logs/2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json`

### NU-HOST-001: Host RAM/process state can add noise to expensive probes
- status: `FIXED`
- severity: `medium`
- evidence:
  - host_runtime_readiness status is `READY`.
- next_actions:
  - Use host_cleanup_runbook.py before loading the 98G bundle.
  - Rerun host_runtime_readiness.py, runtime_lane_readiness_matrix.py, and runtime_next_runbook.py after cleanup.
- acceptance_checks:
  - Host readiness is READY, or WATCH is explicitly accepted before candidate timing.
  - No unrelated high-RSS model server is competing with the Nemotron candidate run.
- source_files:
  - `docs/runtime/logs/2026-06-04-nemotron-ultra-host-runtime-readiness.json`
  - `docs/runtime/logs/2026-06-04-nemotron-ultra-host-cleanup-runbook.json`

### NU-FIXED-001: Already-fixed runtime buckets must stay fixed
- status: `FIXED`
- severity: `info`
- evidence:
  - attention bucket is 8.990 ms.
  - norm/lm_head bucket is 4.317 ms.
  - best live speed 8.335 tok/s clears floor 8.000
  - attention bucket 8.990 ms is below ceiling 10.000
  - norm/lm_head 4.317 ms is below ceiling 5.000
  - Mamba component evidence points to projection/dispatch before conv rewrite
- next_actions:
  - Do not prioritize attention or lm_head while MoE/Mamba remain above bottleneck threshold.
  - Keep these buckets in every compare report as regression checks.
- acceptance_checks:
  - Attention remains below 10 ms.
  - norm/lm_head remains below 5 ms.
- source_files:
  - `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-gate.json`

## Commands
- refresh_manifest: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/runtime_proof_manifest.py --log-dir docs/runtime/logs`
- rerun_ledger: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/runtime_issue_ledger.py --log-dir docs/runtime/logs`
