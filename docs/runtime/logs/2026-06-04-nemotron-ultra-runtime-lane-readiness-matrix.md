# Nemotron Ultra Runtime Lane Readiness Matrix

log_dir: `docs/runtime/logs`
status: `READY`
queue: `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json`

## Lanes
| lane | kind | status | warnings | failures | expected compare |
| --- | --- | --- | --- | --- | --- |
| `moe-routed-shared-scheduling` | `speed_candidate` | `READY` | `0` | `0` | `IMPROVED` |
| `mamba-projection-dispatch` | `speed_candidate` | `READY` | `0` | `0` | `IMPROVED` |
| `weighted-moe-ablation` | `negative_control` | `READY` | `0` | `0` | `FAIL`, `UNCHANGED` |
| `activation-bf16-ablation` | `negative_control` | `READY` | `0` | `0` | `FAIL` |

## Commands
- `moe-routed-shared-scheduling` candidate: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
- `moe-routed-shared-scheduling` post_check: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`
- `mamba-projection-dispatch` candidate: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-mamba-dispatch --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id mamba-projection-dispatch --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
- `mamba-projection-dispatch` post_check: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id mamba-projection-dispatch --candidate-log-dir docs/runtime/logs/candidate-mamba-dispatch --out docs/runtime/logs/candidate-mamba-dispatch/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-mamba-dispatch/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`
- `weighted-moe-ablation` candidate: `JANGTQ_DISABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH=1 PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-disable-weighted-moe --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id weighted-moe-ablation --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
- `weighted-moe-ablation` post_check: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id weighted-moe-ablation --candidate-log-dir docs/runtime/logs/candidate-disable-weighted-moe --out docs/runtime/logs/candidate-disable-weighted-moe/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-disable-weighted-moe/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`
- `activation-bf16-ablation` candidate: `JANGTQ_DISABLE_NEMOTRON_ACTIVATION_BF16=1 PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py --candidate-log-dir docs/runtime/logs/candidate-disable-activation-bf16 --baseline-log-dir docs/runtime/logs --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id activation-bf16-ablation --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --live-max-tokens 32 --long-max-tokens 96`
- `activation-bf16-ablation` post_check: `PYTHONPATH=jang-tools jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id activation-bf16-ablation --candidate-log-dir docs/runtime/logs/candidate-disable-activation-bf16 --out docs/runtime/logs/candidate-disable-activation-bf16/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-disable-activation-bf16/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`

## Warnings And Failures
### moe-routed-shared-scheduling
- warning: none
- failure: none
### mamba-projection-dispatch
- warning: none
- failure: none
### weighted-moe-ablation
- warning: none
- failure: none
### activation-bf16-ablation
- warning: none
- failure: none

## Interpretation
- Run speed_candidate lanes for proposed runtime fixes; run negative_control lanes as guards after related changes.
- WATCH means proof wiring is usable but host/readiness warnings should be handled before loading the 98G bundle.
- BLOCKED means missing or stale proof files must be refreshed before a candidate suite.
