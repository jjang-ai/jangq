# Nemotron Ultra Runtime Proof Manifest

log_dir: `docs/runtime/logs`
status: `PARTIAL`

## Current Metrics
- best_live_tps: `8.335`
- manual_decode_total_ms: `143.237`
- manual_implied_tps: `6.981`
- moe_ms: `65.773`
- mamba_ms: `64.157`
- attention_ms: `8.990`
- norm_lm_head_ms: `4.317`
- moe_plus_mamba_pct_of_total: `90.710`
- best_live_source: `2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json::think_math_default`

## Fixed Evidence
- best live speed 8.335 tok/s clears floor 8.000
- attention bucket 8.990 ms is below ceiling 10.000
- norm/lm_head 4.317 ms is below ceiling 5.000
- Mamba component evidence points to projection/dispatch before conv rewrite

## Partial Evidence
- MoE remains a bottleneck at 65.773 ms
- Mamba remains a bottleneck at 64.157 ms
- coherence gate remains partial (leaks=['factual_japan', 'arithmetic_brief', 'reasoning_apples'], repeats=['factual_japan', 'arithmetic_brief'], no_eos=['reasoning_apples'])

## Lanes
- `moe-routed-shared-scheduling` (speed_candidate): MoE routed/shared scheduling; expected=['IMPROVED']
- `mamba-projection-dispatch` (speed_candidate): Mamba projection/dispatch fusion; expected=['IMPROVED']
- `weighted-moe-ablation` (negative_control): Weighted MoE fast-path A/B; expected=['FAIL', 'UNCHANGED']
- `activation-bf16-ablation` (negative_control): BF16 activation retention guard; expected=['FAIL']

## Commands
- refresh: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/refresh_runtime_proof_bundle.py --log-dir docs/runtime/logs --summary-out docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-proof-refresh.md`
- strict_gate: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/runtime_speed_gate.py --log-dir docs/runtime/logs --strict`

## Artifacts
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-proof-refresh.md`: present, optional, combined no-load refresh output
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-log-bundle-validation.md`: present, optional, required log presence validation
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-status-report.md`: present, optional, human-readable current status
- `docs/runtime/logs/2026-06-04-nemotron-ultra-host-runtime-readiness.json`: present, optional, host memory/disk readiness snapshot
- `docs/runtime/logs/2026-06-04-nemotron-ultra-host-runtime-readiness.md`: present, optional, human-readable host readiness snapshot
- `docs/runtime/logs/2026-06-04-nemotron-ultra-host-cleanup-runbook.json`: present, optional, host cleanup runbook
- `docs/runtime/logs/2026-06-04-nemotron-ultra-host-cleanup-runbook.md`: present, optional, human-readable host cleanup runbook
- `docs/runtime/logs/2026-06-04-nemotron-ultra-speed-experiment-plan.md`: present, optional, ranked speed experiment plan
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-issue-ledger.json`: present, optional, runtime issue ledger
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-issue-ledger.md`: present, optional, human-readable runtime issue ledger
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-index.json`: present, optional, runtime candidate index
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-index.md`: present, optional, human-readable runtime candidate index
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.json`: present, optional, runtime candidate launch guard
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.md`: present, optional, human-readable runtime candidate launch guard
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-cleanup-ready-check.json`: present, optional, runtime cleanup ready check
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-cleanup-ready-check.md`: present, optional, human-readable runtime cleanup ready check
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-candidate-contract.json`: present, optional, MoE candidate contract
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-candidate-contract.md`: present, optional, human-readable MoE candidate contract
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-execution-ticket.json`: present, optional, MoE execution ticket
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-execution-ticket.md`: present, optional, human-readable MoE execution ticket
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-surface-map.json`: present, optional, MoE runtime source surface map
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-surface-map.md`: present, optional, human-readable MoE runtime source surface map
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-patch-plan.json`: present, optional, MoE runtime patch plan
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-patch-plan.md`: present, optional, human-readable MoE runtime patch plan
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-delta-contract.json`: present, optional, MoE runtime delta acceptance contract
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-delta-contract.md`: present, optional, human-readable MoE runtime delta acceptance contract
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-mamba-candidate-contract.json`: present, optional, Mamba candidate contract
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-mamba-candidate-contract.md`: present, optional, human-readable Mamba candidate contract
- `docs/runtime/logs/2026-06-04-nemotron-ultra-token-speed-budget.json`: present, required, target token/s millisecond budgets
- `docs/runtime/logs/2026-06-04-nemotron-ultra-token-speed-budget.md`: present, optional, human-readable target budgets
- `docs/runtime/logs/2026-06-04-nemotron-ultra-component-budget-matrix.json`: present, optional, component-level token/s sensitivity matrix
- `docs/runtime/logs/2026-06-04-nemotron-ultra-component-budget-matrix.md`: present, optional, human-readable component budget matrix
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-patch-spec.json`: present, optional, implementation-facing runtime patch spec
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-patch-spec.md`: present, optional, human-readable runtime patch spec
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-shape-contract.json`: present, optional, runtime shape and bit contract
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-shape-contract.md`: present, optional, human-readable runtime shape contract
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-preflight.json`: present, optional, runtime candidate preflight
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-preflight.md`: present, optional, human-readable candidate preflight
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-lane-readiness-matrix.json`: present, optional, all-lane runtime readiness matrix
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-lane-readiness-matrix.md`: present, optional, human-readable lane readiness matrix
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-next-runbook.json`: present, optional, next runtime runbook
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-next-runbook.md`: present, optional, human-readable next runtime runbook
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json`: present, required, machine-readable experiment queue
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.md`: present, optional, human-readable experiment queue
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-gate.json`: present, required, machine-readable speed gate
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-gate.md`: present, optional, human-readable speed gate
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-fix-acceptance.json`: present, optional, runtime speed fix acceptance audit
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-fix-acceptance.md`: present, optional, human-readable speed fix acceptance audit
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-compare.json`: present, optional, baseline-vs-baseline compare for current logs
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-compare.md`: present, optional, human-readable compare report
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-cache-parser-contract.json`: present, optional, cache/parser/runtime nuance contract
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-cache-parser-contract.md`: present, optional, human-readable cache/parser/runtime nuance contract
- `docs/runtime/logs/2026-06-04-nemotron-ultra-agent-handoff.json`: present, required, machine-readable downstream agent handoff
- `docs/runtime/logs/2026-06-04-nemotron-ultra-agent-handoff.md`: present, optional, human-readable downstream agent handoff
- `docs/runtime/logs/2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json`: present, required, best live decode source
- `docs/runtime/logs/2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json`: present, required, layer decode bucket source
- `docs/runtime/logs/2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json`: present, required, long coherence source
- `docs/runtime/logs/2026-06-04-nemotron-ultra-mamba-component-probe.json`: present, required, Mamba component source
- `docs/runtime/logs/2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json`: present, required, MoE component source
- `docs/runtime/logs/2026-06-04-nemotron-ultra-projection-tradeoff-probe.json`: present, required, projection tradeoff source
