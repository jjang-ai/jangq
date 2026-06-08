# Nemotron Ultra MoE Runtime Surface Map

log_dir: `docs/runtime/logs`
source_root: `jang-tools`
lane_id: `moe-routed-shared-scheduling`
status: `READY`
ticket_status: `READY`
contract_status: `READY`

## Target
- moe_cut_ms_proportional: `21.888`
- moe_cut_pct_of_current_moe: `33.277`
- moe_per_layer_cut_ms: `0.456`
- required_total_cut_ms: `43.237`
- target_tps: `10.000`

## Current Speed
- best_live_tps: `8.335`
- mamba_ms: `64.157`
- manual_decode_total_ms: `143.237`
- moe_ms: `65.773`

## Component Timings
- full_moe: `2.107 ms`
- switch_mlp: `1.130 ms`
- shared_experts: `0.577 ms`
- fc1_latent_proj: `0.231 ms`
- gate: `0.221 ms`
- fc2_latent_proj: `0.212 ms`
- norm: `0.203 ms`
- score_weighted_sum: `0.176 ms`

## Surfaces
| id | status | file | role | anchors |
| --- | --- | --- | --- | --- |
| `loader-hydration` | `READY` | `jang-tools/jang_tools/load_jangtq.py` | hydrates switch_mlp tensors into TurboQuantSwitchLinear modules | def _hydrate_jangtq_model@831, TurboQuantSwitchLinear@48, switch_mlp@499 |
| `nemotron-weighted-moe-patch` | `READY` | `jang-tools/jang_tools/load_jangtq.py` | patches NemotronHMoE decode to call weighted SwitchMLP and shared experts | JANGTQ_DISABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH@1616, Nemotron-H MoE weighted SwitchMLP decode@1656, _switchmlp_weighted_decode@1463 |
| `switchmlp-fastpath-toggle` | `READY` | `jang-tools/jang_tools/load_jangtq.py` | controls the SwitchMLP fast path and its negative-control env toggle | JANGTQ_DISABLE_NEMOTRON_SWITCHMLP_FASTPATH@1583, JANGTQ_ENABLE_NEMOTRON_SWITCHMLP_FASTPATH@1582 |
| `activation-bf16-toggle` | `READY` | `jang-tools/jang_tools/load_jangtq.py` | preserves or disables BF16 activation retention for negative-control proof | JANGTQ_DISABLE_NEMOTRON_ACTIVATION_BF16@1676 |
| `routed-gather-kernel` | `READY` | `jang-tools/jang_tools/turboquant/gather_tq_kernel.py` | routed down/fc2 gather matmul kernel for selected experts | def gather_tq_matmul@695, make_gather_tq_decode_broadcast@658, make_gather_tq_decode_per_row@593, JANGTQ_GATHER_OPT@39 |
| `fused-gate-up-kernel` | `READY` | `jang-tools/jang_tools/turboquant/fused_gate_up_kernel.py` | fused gate/up/SwiGLU routed expert path used by SwitchMLP | def fused_gate_up_swiglu_matmul@399, make_fused_gate_up_swiglu_decode@271, JANGTQ_MPP_NAX@18 |
| `grouped-nax-proof-surface` | `READY` | `jang-tools/jang_tools/turboquant/mpp_nax_kernel.py` | same-expert grouped tile helpers for possible routed scheduling work | build_sorted_group_tiles@590, gather_tq_matmul_mpp_nax_grouped_from_rot@641, fused_gate_up_swiglu_mpp_nax_grouped_from_rot@835 |
| `moe-component-proof` | `READY` | `jang-tools/examples/nemotron_ultra/moe_component_probe.py` | measures gate, switch_mlp, weighted_decode, shared_experts, weighted sum, and full_moe | moe.switch_mlp@62, weighted_decode@64, shared_experts@69, full_moe@98 |
| `candidate-verdict-proof` | `READY` | `jang-tools/examples/nemotron_ultra/experiment_result_check.py` | accepts/rejects the MoE candidate using compare, speed gate, and handoff invariants | moe-routed-shared-scheduling@144, MoE lane did not improve moe_ms@145, ACCEPTED@171 |

## Runtime Controls
- disable_weighted_moe_fastpath: `JANGTQ_DISABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH=1`
- disable_switchmlp_fastpath: `JANGTQ_DISABLE_NEMOTRON_SWITCHMLP_FASTPATH=1`
- legacy_disable_switchmlp_fastpath: `JANGTQ_ENABLE_NEMOTRON_SWITCHMLP_FASTPATH=0`
- disable_activation_bf16: `JANGTQ_DISABLE_NEMOTRON_ACTIVATION_BF16=1`
- gather_opt: `JANGTQ_GATHER_OPT`
- mpp_nax: `JANGTQ_MPP_NAX`
- mpp_nax_strict: `JANGTQ_MPP_NAX_STRICT=1`

## Commands
- candidate: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
- post_check: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`

## Non Goals
- do not edit vMLX or MLX Studio for this JANG handoff
- do not expand routed 1-bit or shared 8-bit tensors to full precision
- do not lower top-k or hide parser/coherence issues to make a speed row look better
- do not run the Mamba lane until the MoE lane has accepted evidence

## Missing
- none

## Source Files
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-candidate-contract.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-execution-ticket.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-token-speed-budget.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json`
