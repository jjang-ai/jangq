# Nemotron Ultra MoE Patch Plan

log_dir: `docs/runtime/logs`
lane_id: `moe-routed-shared-scheduling`
status: `READY`
speed_acceptance_status: `PARTIAL`

## Current Speed
- best_live_tps: `8.335`
- mamba_ms: `64.157`
- manual_decode_total_ms: `143.237`
- moe_ms: `65.773`

## Target
- moe_cut_ms_proportional: `21.888`
- moe_cut_pct_of_current_moe: `33.277`
- moe_per_layer_cut_ms: `0.456`
- required_total_cut_ms: `43.237`
- target_tps: `10.000`

## Ordered Steps
### moe-01-path-scheduling: Reduce full MoE path scheduling overhead first
- component: `full_moe` (inclusive_path)
- projected_total_ms: `101.148`
- 25pct_cut_tps: `8.478`
- 50pct_cut_tps: `10.792`
- goal: Attack the inclusive NemotronHMoE path before isolated micro-optimizations.
- surfaces:
  - `nemotron-weighted-moe-patch`: `jang-tools/jang_tools/load_jangtq.py` (JANGTQ_DISABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH@1616, Nemotron-H MoE weighted SwitchMLP decode@1656, _switchmlp_weighted_decode@1463)
  - `switchmlp-fastpath-toggle`: `jang-tools/jang_tools/load_jangtq.py` (JANGTQ_DISABLE_NEMOTRON_SWITCHMLP_FASTPATH@1583, JANGTQ_ENABLE_NEMOTRON_SWITCHMLP_FASTPATH@1582)
  - `moe-component-proof`: `jang-tools/examples/nemotron_ultra/moe_component_probe.py` (full_moe@98, moe.switch_mlp@62, shared_experts@69, weighted_decode@64)
  - `candidate-verdict-proof`: `jang-tools/examples/nemotron_ultra/experiment_result_check.py` (ACCEPTED@171, MoE lane did not improve moe_ms@145, moe-routed-shared-scheduling@144)
- validation:
  - rerun moe_component_probe.py and require full_moe plus switch_mlp timing to move
  - rerun layer_decode_probe.py and require E bucket improvement
  - rerun long_decode_coherence_probe.py and reject marker/repeat/EOS regression
- non_goals:
  - do not lower top-k
  - do not bypass weighted expert scores
  - do not count a short live row as acceptance without experiment_result_check
### moe-02-switchmlp-routed-kernels: Optimize SwitchMLP routed gate/up/down execution
- component: `switch_mlp` (substep)
- projected_total_ms: `54.264`
- 25pct_cut_tps: `7.712`
- 50pct_cut_tps: `8.613`
- goal: Reduce routed 1-bit expert dispatch around fused gate/up and gather down kernels.
- surfaces:
  - `fused-gate-up-kernel`: `jang-tools/jang_tools/turboquant/fused_gate_up_kernel.py` (JANGTQ_MPP_NAX@18, def fused_gate_up_swiglu_matmul@399, make_fused_gate_up_swiglu_decode@271)
  - `routed-gather-kernel`: `jang-tools/jang_tools/turboquant/gather_tq_kernel.py` (JANGTQ_GATHER_OPT@39, def gather_tq_matmul@695, make_gather_tq_decode_broadcast@658, make_gather_tq_decode_per_row@593)
  - `grouped-nax-proof-surface`: `jang-tools/jang_tools/turboquant/mpp_nax_kernel.py` (build_sorted_group_tiles@590, fused_gate_up_swiglu_mpp_nax_grouped_from_rot@835, gather_tq_matmul_mpp_nax_grouped_from_rot@641)
  - `moe-component-proof`: `jang-tools/examples/nemotron_ultra/moe_component_probe.py` (full_moe@98, moe.switch_mlp@62, shared_experts@69, weighted_decode@64)
  - `candidate-verdict-proof`: `jang-tools/examples/nemotron_ultra/experiment_result_check.py` (ACCEPTED@171, MoE lane did not improve moe_ms@145, moe-routed-shared-scheduling@144)
- validation:
  - compare fused_gate_up and gather path timings through moe_component_probe.py
  - run candidate suite with default env, then compare against weighted-MoE and activation-BF16 negative controls
  - preserve routed_expert_bits up/down=1 and indices shape [1,1,22]
- non_goals:
  - do not expand routed experts to BF16
  - do not make grouped NAX the default without candidate proof
  - do not edit vMLX or MLX Studio for this JANG lane
### moe-03-shared-experts-overlap: Measure shared expert overlap only after routed path moves
- component: `shared_experts` (substep)
- projected_total_ms: `27.720`
- 25pct_cut_tps: `7.336`
- 50pct_cut_tps: `7.729`
- goal: Treat shared experts as secondary unless routed scheduling leaves them dominant.
- surfaces:
  - `nemotron-weighted-moe-patch`: `jang-tools/jang_tools/load_jangtq.py` (JANGTQ_DISABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH@1616, Nemotron-H MoE weighted SwitchMLP decode@1656, _switchmlp_weighted_decode@1463)
  - `moe-component-proof`: `jang-tools/examples/nemotron_ultra/moe_component_probe.py` (full_moe@98, moe.switch_mlp@62, shared_experts@69, weighted_decode@64)
  - `candidate-verdict-proof`: `jang-tools/examples/nemotron_ultra/experiment_result_check.py` (ACCEPTED@171, MoE lane did not improve moe_ms@145, moe-routed-shared-scheduling@144)
- validation:
  - require shared_experts timing to improve without increasing switch_mlp
  - preserve shared_expert_bits=8
  - keep speed acceptance PARTIAL unless token/s target, bucket ceilings, and candidate acceptance all pass
- non_goals:
  - do not dequantize shared experts by default
  - do not optimize shared path before routed path evidence

## Commands
- candidate: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
- post_check: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`
- post_candidate_refresh: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/refresh_runtime_proof_bundle.py --log-dir docs/runtime/logs --summary-out docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-proof-refresh.md`
- acceptance: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/runtime_speed_fix_acceptance.py --log-dir docs/runtime/logs --strict`

## Failures
- none

## Source Files
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-surface-map.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-component-budget-matrix.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-candidate-contract.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-execution-ticket.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-fix-acceptance.json`
