# Nemotron Ultra Mamba Candidate Contract

log_dir: `docs/runtime/logs`
lane_id: `mamba-projection-dispatch`
status: `BLOCKED`

## Current Speed
- mamba_ms: `64.157`
- moe_ms: `65.773`
- manual_decode_total_ms: `143.237`
- best_live_tps: `8.335`

## Target
- target_tps: `12.000`
- required_total_cut_ms: `59.904`
- mamba_cut_ms_proportional: `29.579`
- mamba_cut_pct_of_current_mamba: `46.105`
- mamba_per_layer_cut_ms: `0.616`

## Mamba Invariants
- hidden_shape: `[1, 1, 8192]`
- normed_shape: `[1, 1, 8192]`
- projected_shape: `[1, 1, 35072]`
- gate_shape: `[1, 1, 16384]`
- conv_dim: `18432`
- conv_input_shape: `[1, 1, 18432]`
- conv_output_shape: `[1, 1, 18432]`
- ssm_out_shape: `[1, 1, 16384]`
- intermediate_size: `16384`
- num_heads: `256`
- n_groups: `8`
- ssm_state_size: `128`
- mamba_projection_bits: `8`
- fp8_projection_affine_bits: `8`
- fp8_projection_group_size: `128`
- drops_mtp: `True`

## Preconditions
- MoE lane evidence is MISSING; run/accept MoE lane before Mamba lane

## Do
- attack projection/dispatch overhead before grouped conv rewrites
- preserve 8-bit affine projection path unless a new projection tradeoff probe reverses the result
- preserve Mamba companion cache/state order for hybrid prefix cache compatibility
- measure M bucket plus `in_proj`, `out_proj`, `conv`, and `ssm_update` after every candidate

## Do Not
- do not dequantize Mamba projections to BF16 as a default speed fix
- do not treat attention KV cache work as a substitute for Mamba state proof
- do not change cache topology without rerunning cache/block handoff checks

## Acceptance Checks
- Mamba bucket drops without changing cache cardinality or Mamba shape contract.
- Attention and norm/lm_head remain below current ceilings.
- long coherence leak/repeat/EOS counts do not regress.

## Commands
- candidate: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-mamba-dispatch --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id mamba-projection-dispatch --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
- post_check: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id mamba-projection-dispatch --candidate-log-dir docs/runtime/logs/candidate-mamba-dispatch --out docs/runtime/logs/candidate-mamba-dispatch/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-mamba-dispatch/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`

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

## Source Files
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-patch-spec.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-shape-contract.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-token-speed-budget.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-issue-ledger.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-index.json`
