# Nemotron Ultra MoE Delta Contract

log_dir: `docs/runtime/logs`
lane_id: `moe-routed-shared-scheduling`
status: `READY`

## Baseline
- best_live_tps: `8.335`
- manual_decode_total_ms: `143.237`
- manual_implied_tps: `6.981`
- moe_ms: `65.773`
- mamba_ms: `64.157`
- attention_ms: `8.990`
- norm_lm_head_ms: `4.317`

## Target
- target_tps: `10.000`
- target_ms_per_token: `100.000`
- required_total_cut_ms: `43.237`
- moe_cut_ms_proportional: `21.888`
- moe_cut_pct_of_current_moe: `33.277`
- target_moe_ms_for_proportional_10tps: `43.886`
- acceptance_max_moe_ms: `40.000`
- acceptance_max_mamba_ms: `40.000`

## Acceptance Thresholds
- experiment_result_status: `ACCEPTED`
- compare_status: `IMPROVED`
- gate_status: `FIXED for final speed acceptance; PARTIAL only means candidate moved one bucket`
- best_live_tps: `>= 10.000 for final speed acceptance`
- moe_ms: `must improve versus baseline and should fall below 43.886 ms for a 10 tok/s trajectory`
- final_moe_ceiling_ms: `40.000`
- final_mamba_ceiling_ms: `40.000`
- coherence: `no regression in leak, repeat, or EOS counts`
- regression_guards:
  - Mamba ms must not materially regress
  - attention ms must stay under fixed gate ceiling
  - norm/lm_head ms must stay under fixed gate ceiling
  - parser/tool/reasoning behavior must not be hidden by prompt or sampler guards

## Ordered MoE Steps
- `moe-01-path-scheduling` component=`full_moe` projected_total_ms=`101.148` 25pct_tps=`8.478` 50pct_tps=`10.792`
- `moe-02-switchmlp-routed-kernels` component=`switch_mlp` projected_total_ms=`54.264` 25pct_tps=`7.712` 50pct_tps=`8.613`
- `moe-03-shared-experts-overlap` component=`shared_experts` projected_total_ms=`27.720` 25pct_tps=`7.336` 50pct_tps=`7.729`

## Negative Controls
- weighted-moe-ablation is diagnostic only and must not be promoted as a speed fix
- activation-bf16-ablation is diagnostic only and must not be promoted as a speed fix

## Commands
- candidate: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
- post_check: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`
- post_candidate_index: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/runtime_candidate_index.py --log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --out docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-index.md --json-out docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-index.json`
- post_candidate_refresh: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/refresh_runtime_proof_bundle.py --log-dir docs/runtime/logs --summary-out docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-proof-refresh.md`
- acceptance_strict: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/runtime_speed_fix_acceptance.py --log-dir docs/runtime/logs --strict`

## Failures
- none

## Source Files
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-candidate-contract.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-token-speed-budget.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-fix-acceptance.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-patch-plan.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-execution-ticket.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-index.json`
