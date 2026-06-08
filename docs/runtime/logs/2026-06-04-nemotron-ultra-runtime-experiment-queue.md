# Nemotron Ultra Runtime Experiment Queue

baseline_log_dir: `docs/runtime/logs`
bundle: `/Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L`
candidate_root: `docs/runtime/logs`

## Current Baseline
- best_live_tps: `8.335`
- manual_decode_total_ms: `143.237`
- moe_ms: `65.773`
- mamba_ms: `64.157`
- moe_plus_mamba_pct_of_total: `90.71%`

## Lanes
### MoE routed/shared scheduling
- id: `moe-routed-shared-scheduling`
- kind: `speed_candidate`
- goal: Reduce MoE bucket without changing routing semantics or routed expert bit layout.
- evidence: MoE is 65.773 ms; first target needs about 21.888 ms MoE reduction for 10.0 tok/s.
- patch_surface: JANG loader/TurboQuant MoE scheduling only; no bundle expansion.
- env: none
- expected_compare_statuses: `IMPROVED`
- command: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
- post_check_command: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`
- required_outputs: `2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json`, `2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json`, `2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json`, `2026-06-04-nemotron-ultra-mamba-component-probe.json`, `2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json`, `2026-06-04-nemotron-ultra-projection-tradeoff-probe.json`, `2026-06-04-nemotron-ultra-runtime-speed-compare.json`, `2026-06-04-nemotron-ultra-runtime-speed-gate.json`, `2026-06-04-nemotron-ultra-token-speed-budget.json`, `2026-06-04-nemotron-ultra-agent-handoff.json`, `2026-06-04-nemotron-ultra-runtime-cache-parser-contract.json`
- acceptance: candidate compare status is IMPROVED with no failures; MoE bucket improves and Mamba/attention/lm_head do not materially regress; long coherence leak/repeat/no_eos counts do not regress; candidate handoff remains text-only, MTP-disabled, and hybrid-cache aware

### Mamba projection/dispatch fusion
- id: `mamba-projection-dispatch`
- kind: `speed_candidate`
- goal: Reduce Mamba bucket by attacking projection and dispatch overhead before conv rewrites.
- evidence: Mamba is 64.157 ms; first target needs about 21.350 ms Mamba reduction for 10.0 tok/s.
- patch_surface: JANG loader/runtime Mamba path only; keep 8-bit affine projections unless new proof reverses it.
- env: none
- expected_compare_statuses: `IMPROVED`
- command: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-mamba-dispatch --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id mamba-projection-dispatch --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
- post_check_command: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id mamba-projection-dispatch --candidate-log-dir docs/runtime/logs/candidate-mamba-dispatch --out docs/runtime/logs/candidate-mamba-dispatch/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-mamba-dispatch/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`
- required_outputs: `2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json`, `2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json`, `2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json`, `2026-06-04-nemotron-ultra-mamba-component-probe.json`, `2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json`, `2026-06-04-nemotron-ultra-projection-tradeoff-probe.json`, `2026-06-04-nemotron-ultra-runtime-speed-compare.json`, `2026-06-04-nemotron-ultra-runtime-speed-gate.json`, `2026-06-04-nemotron-ultra-token-speed-budget.json`, `2026-06-04-nemotron-ultra-agent-handoff.json`, `2026-06-04-nemotron-ultra-runtime-cache-parser-contract.json`
- acceptance: candidate compare status is IMPROVED with no failures; Mamba bucket improves and MoE/attention/lm_head do not materially regress; long coherence leak/repeat/no_eos counts do not regress; candidate handoff remains text-only, MTP-disabled, and hybrid-cache aware

### Weighted MoE fast-path A/B
- id: `weighted-moe-ablation`
- kind: `negative_control`
- goal: Confirm the current weighted-MoE default remains beneficial after any MoE refactor.
- evidence: Weighted MoE is a small positive/noisy improvement and must not silently regress.
- patch_surface: A/B proof lane only; no code change implied.
- env: `JANGTQ_DISABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH=1`
- expected_compare_statuses: `FAIL`, `UNCHANGED`
- command: `JANGTQ_DISABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH=1 PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-disable-weighted-moe --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id weighted-moe-ablation --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
- post_check_command: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id weighted-moe-ablation --candidate-log-dir docs/runtime/logs/candidate-disable-weighted-moe --out docs/runtime/logs/candidate-disable-weighted-moe/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-disable-weighted-moe/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`
- required_outputs: `2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json`, `2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json`, `2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json`, `2026-06-04-nemotron-ultra-mamba-component-probe.json`, `2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json`, `2026-06-04-nemotron-ultra-projection-tradeoff-probe.json`, `2026-06-04-nemotron-ultra-runtime-speed-compare.json`, `2026-06-04-nemotron-ultra-runtime-speed-gate.json`, `2026-06-04-nemotron-ultra-token-speed-budget.json`, `2026-06-04-nemotron-ultra-agent-handoff.json`, `2026-06-04-nemotron-ultra-runtime-cache-parser-contract.json`
- acceptance: compare must not be treated as a speed fix; if the lane is faster, preserve evidence before changing the default; long coherence leak/repeat/no_eos counts do not regress; candidate handoff remains text-only, MTP-disabled, and hybrid-cache aware

### BF16 activation retention guard
- id: `activation-bf16-ablation`
- kind: `negative_control`
- goal: Guard the large lm_head/activation dtype speed fix from accidental rollback.
- evidence: BF16 retention moved synchronized decode from about 320 ms/token to about 144 ms/token.
- patch_surface: A/B proof lane only; should be slower and marked as a negative-control regression.
- env: `JANGTQ_DISABLE_NEMOTRON_ACTIVATION_BF16=1`
- expected_compare_statuses: `FAIL`
- command: `JANGTQ_DISABLE_NEMOTRON_ACTIVATION_BF16=1 PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-disable-activation-bf16 --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id activation-bf16-ablation --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
- post_check_command: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id activation-bf16-ablation --candidate-log-dir docs/runtime/logs/candidate-disable-activation-bf16 --out docs/runtime/logs/candidate-disable-activation-bf16/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-disable-activation-bf16/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`
- required_outputs: `2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json`, `2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json`, `2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json`, `2026-06-04-nemotron-ultra-mamba-component-probe.json`, `2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json`, `2026-06-04-nemotron-ultra-projection-tradeoff-probe.json`, `2026-06-04-nemotron-ultra-runtime-speed-compare.json`, `2026-06-04-nemotron-ultra-runtime-speed-gate.json`, `2026-06-04-nemotron-ultra-token-speed-budget.json`, `2026-06-04-nemotron-ultra-agent-handoff.json`, `2026-06-04-nemotron-ultra-runtime-cache-parser-contract.json`
- acceptance: compare should fail or clearly regress speed versus baseline; lm_head/norm and manual decode regressions confirm BF16 retention is still required; do not promote this lane as a candidate fix; candidate handoff remains text-only, MTP-disabled, and hybrid-cache aware

## Notes
- Run one candidate lane at a time; the model is about 98G and probes are expensive.
- Lane-specific env vars are already embedded in the generated command.
- Do not call a speed lane fixed until compare, gate, and long-coherence rows agree.
- Negative-control lanes are guards; expected regressions should not be promoted as fixes.
