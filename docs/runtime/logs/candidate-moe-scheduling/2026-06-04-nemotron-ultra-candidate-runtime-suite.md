# Nemotron Ultra Candidate Runtime Suite

## Dry Run

````text
dry run only; no commands executed
````

## Planned Model Probes
- Live speed probe: `/Users/eric/jang/jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/live_speed_probe.py --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --rows full --max-tokens 32 --wired-limit-gb 105 --out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json`
- Layer decode probe: `/Users/eric/jang/jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/layer_decode_probe.py --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --wired-limit-gb 105 --out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json`
- Long coherence probe: `/Users/eric/jang/jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/long_decode_coherence_probe.py --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --rows full --max-tokens 96 --sampler greedy --wired-limit-gb 105 --out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json`
- Mamba component probe: `/Users/eric/jang/jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/mamba_component_probe.py --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --layers 1 --repeats 5 --warmup 2 --wired-limit-gb 105 --out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-mamba-component-probe.json`
- MoE component probe: `/Users/eric/jang/jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/moe_component_probe.py --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --layers 1 --repeats 5 --warmup 2 --wired-limit-gb 105 --out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json`
- Projection tradeoff probe: `/Users/eric/jang/jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/projection_tradeoff_probe.py --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --repeats 10 --warmup 3 --wired-limit-gb 105 --out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-projection-tradeoff-probe.json`

## Planned Report Commands
- Candidate log bundle validation: `/Users/eric/jang/jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/validate_runtime_log_bundle.py --log-dir docs/runtime/logs/candidate-moe-scheduling --out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-runtime-log-bundle-validation.md`
- Candidate proof refresh: `/Users/eric/jang/jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/refresh_runtime_proof_bundle.py --log-dir docs/runtime/logs/candidate-moe-scheduling --bundle /Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L --summary-out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-runtime-proof-refresh.md`
- Baseline vs candidate compare: `/Users/eric/jang/jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/compare_runtime_speed_logs.py --baseline-log-dir docs/runtime/logs --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-runtime-speed-compare.md --json-out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-runtime-speed-compare.json`
- Experiment result check: `/Users/eric/jang/jang-tools/.venv/bin/python jang-tools/examples/nemotron_ultra/experiment_result_check.py --queue-json docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json --lane-id moe-routed-shared-scheduling --candidate-log-dir docs/runtime/logs/candidate-moe-scheduling --out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.md --json-out docs/runtime/logs/candidate-moe-scheduling/2026-06-04-nemotron-ultra-experiment-result-check.json --strict`
