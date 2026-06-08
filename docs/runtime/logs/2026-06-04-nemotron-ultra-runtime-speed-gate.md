# Nemotron Ultra Runtime Speed Gate

log_dir: `docs/runtime/logs`
status: `PARTIAL`

## Fixed Evidence
- best live speed 8.335 tok/s clears floor 8.000
- attention bucket 8.990 ms is below ceiling 10.000
- norm/lm_head 4.317 ms is below ceiling 5.000
- Mamba component evidence points to projection/dispatch before conv rewrite

## Partial Evidence
- MoE remains a bottleneck at 65.773 ms
- Mamba remains a bottleneck at 64.157 ms
- coherence gate remains partial (leaks=['factual_japan', 'arithmetic_brief', 'reasoning_apples'], repeats=['factual_japan', 'arithmetic_brief'], no_eos=['reasoning_apples'])

## Failures

## Current Buckets
- best_live: `8.335 tok/s` from `2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json::think_math_default`
- manual_decode_total_ms: `143.237`
- moe_ms: `65.773`
- mamba_ms: `64.157`
- attention_ms: `8.990`
- norm_lm_head_ms: `4.317`
