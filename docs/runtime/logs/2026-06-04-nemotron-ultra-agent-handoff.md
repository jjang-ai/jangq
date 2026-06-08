# Nemotron Ultra Agent Runtime Handoff

bundle: `/Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L`
log_dir: `docs/runtime/logs`
handoff_status: `PARTIAL`
speed_gate: `PARTIAL`

## Artifact
- profile: `JANGTQ_1L`
- format: `jangtq` `2.0`
- estimated_output_gib: `98.35`
- shard_count: `51`
- drops_mtp: `True`
- capabilities: `{"cache_type": "hybrid", "family": "nemotron_h", "modality": "text", "reasoning_parser": "deepseek_r1", "supports_thinking": true, "supports_tools": true, "think_in_template": true, "tool_parser": "nemotron"}`
- mxtq_bits: `{"mamba_projection": 8, "routed_expert": {"down_proj": 1, "up_proj": 1}, "shared_expert": 8}`

## Topology
- layers_total: `108`
- mamba/moe/attention: `48` / `48` / `12`
- cache_entries: `60` = `48` Mamba companion states + `12` attention KV entries

## Current Speed Buckets
- best_live_tps: `8.335` from `2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json::think_math_default`
- manual_decode_total_ms: `143.237`
- moe_ms: `65.773`
- mamba_ms: `64.157`
- attention_ms: `8.990`
- norm_lm_head_ms: `4.317`

## Fixed Evidence
- best live speed 8.335 tok/s clears floor 8.000
- attention bucket 8.990 ms is below ceiling 10.000
- norm/lm_head 4.317 ms is below ceiling 5.000
- Mamba component evidence points to projection/dispatch before conv rewrite

## Partial Evidence
- MoE remains a bottleneck at 65.773 ms
- Mamba remains a bottleneck at 64.157 ms
- coherence gate remains partial (leaks=['factual_japan', 'arithmetic_brief', 'reasoning_apples'], repeats=['factual_japan', 'arithmetic_brief'], no_eos=['reasoning_apples'])

## Parser And Coherence
- parser_status: `PARTIAL` marker_leak_rows=['nt_capital_default'] truncated_reasoning_rows=['think_math_default'] tool_rows=0
- long_coherence_status: `PARTIAL` leaks=['factual_japan', 'arithmetic_brief', 'reasoning_apples'] repeats=['factual_japan', 'arithmetic_brief'] no_eos=['reasoning_apples']

## Token Speed Budgets
- target `10.000` tok/s: cut `43.237` ms total; proportional MoE `21.888` ms, Mamba `21.350` ms
- target `12.000` tok/s: cut `59.904` ms total; proportional MoE `30.325` ms, Mamba `29.579` ms
- target `15.000` tok/s: cut `76.571` ms total; proportional MoE `38.762` ms, Mamba `37.809` ms

## Runtime Controls
- disable_switchmlp_fastpath: `JANGTQ_DISABLE_NEMOTRON_SWITCHMLP_FASTPATH=1`
- legacy_disable_switchmlp_fastpath: `JANGTQ_ENABLE_NEMOTRON_SWITCHMLP_FASTPATH=0`
- disable_activation_bf16: `JANGTQ_DISABLE_NEMOTRON_ACTIVATION_BF16=1`
- disable_weighted_moe_fastpath: `JANGTQ_DISABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH=1`

## Cache And Modality Gates
- cache_type: `hybrid`
- text_only: `True`
- kv_cache_boundary: `TurboQuant KV applies only to attention KV entries.`
- mamba_state_boundary: `Prefix hits require matching Mamba companion states.`
- vl_policy: `Reject or reroute media requests; this artifact has no VL/audio tensors or processor configs.`
- mtp_policy: `Disabled for this bundle; draft KV/SSM state is out of scope.`

## Next Experiments
- MoE routed/shared scheduling or fused decode kernel
- Mamba projection/dispatch fusion or fused decode state update
- Joint MoE+Mamba dispatch-boundary reduction
- rerun layer decode and live speed after any runtime change
- rerun long coherence; speed wins must not regress parser-visible output

## Negative Controls
- Do not chase attention first while it remains under the current ceiling.
- Do not dequantize 8-bit affine projections as a speed fix without new proof.
- Do not lower router top-k as the main fix; saved top-k probe did not materially improve decode.
- Do not hide parser/coherence failures with prompt suffixes or sampler tricks.
- Do not enable speculative/MTP decode for this MTP-dropped bundle.
