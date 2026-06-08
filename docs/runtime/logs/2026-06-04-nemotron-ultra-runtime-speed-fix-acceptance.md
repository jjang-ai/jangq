# Nemotron Ultra Runtime Speed Fix Acceptance

log_dir: `docs/runtime/logs`
status: `PARTIAL`
target_tps: `10.000`
max_moe_ms: `40.000`
max_mamba_ms: `40.000`
require_speed_gate_fixed: `True`

## Current
- manifest_status: `PARTIAL`
- ledger_status: `OPEN`
- speed_gate_status: `PARTIAL`
- candidate_index_status: `OPEN`
- best_live_tps: `8.335`
- manual_decode_total_ms: `143.237`
- moe_ms: `65.773`
- mamba_ms: `64.157`
- attention_ms: `8.990`
- norm_lm_head_ms: `4.317`

## Fixed
- none

## Partial
- speed gate is PARTIAL, not FIXED
- no speed_candidate lane has ACCEPTED evidence
- best live token/s 8.335 is below target 10.000
- MoE bucket 65.773 ms exceeds acceptance ceiling 40.000
- Mamba bucket 64.157 ms exceeds acceptance ceiling 40.000
- coherence gate remains partial (leaks=['factual_japan', 'arithmetic_brief', 'reasoning_apples'], repeats=['factual_japan', 'arithmetic_brief'], no_eos=['reasoning_apples'])

## Blockers
- none

## Accepted Speed Lanes
- none

## Source Files
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-gate.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-index.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-token-speed-budget.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-proof-manifest.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-issue-ledger.json`
