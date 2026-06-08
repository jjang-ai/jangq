# Nemotron Ultra Cache Parser Contract

log_dir: `docs/runtime/logs`
status: `PARTIAL`
speed_acceptance_status: `PARTIAL`
candidate_status: `OPEN`

## Cache Contract
- cache_type: `hybrid`
- cache_entries: `60`
- mamba_companion_state_entries: `48`
- attention_kv_cache_entries: `12`
- mamba_layers: `48`
- attention_layers: `12`
- kv_cache_boundary: `TurboQuant KV applies only to attention KV entries.`
- mamba_state_boundary: `Prefix hits require matching Mamba companion states.`
- prefix_cache_acceptance: `["attention KV hit is insufficient without the matching 48 Mamba companion states", "cache restore must preserve layer order and cache ordinal mapping", "parser streaming state must be salted/restored with the cache key"]`

## Parser Contract
- reasoning_parser: `deepseek_r1`
- tool_parser: `nemotron`
- supports_thinking: `True`
- supports_tools: `True`
- think_in_template: `True`
- parser_probe: `{"marker_leak_rows": ["nt_capital_default"], "parser": "deepseek_r1 compatible <think> parser + Ultra XML function calls", "rows": 3, "status": "PARTIAL", "tool_rows": 0, "truncated_reasoning_rows": ["think_math_default"]}`
- long_coherence: `{"leak_rows": ["factual_japan", "arithmetic_brief", "reasoning_apples"], "no_eos_rows": ["reasoning_apples"], "repeat_rows": ["factual_japan", "arithmetic_brief"], "rows": 3, "status": "PARTIAL"}`
- acceptance: `["no new visible <think>, </think>, <tool_call>, or <tool_response> leakage", "no truncated reasoning rows versus baseline", "tool-call parser remains Nemotron XML compatible", "do not hide parser failures with prompt suffixes, forced tags, or sampler penalties"]`

## Modality And MTP
- modality: `text`
- text_only: `True`
- vl_policy: `Reject or reroute media requests; this artifact has no VL/audio tensors or processor configs.`
- audio_policy: `No audio tensors or processor configs are present; reject or reroute audio requests.`
- mtp_policy: `Disabled for this bundle; draft KV/SSM state is out of scope.`
- drops_mtp: `True`

## Partial
- parser probe is PARTIAL
- long coherence is PARTIAL
- speed fix acceptance is PARTIAL
- candidate index is OPEN

## Failures
- none

## Source Files
- `docs/runtime/logs/2026-06-04-nemotron-ultra-agent-handoff.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-shape-contract.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-jangtq1l-parser-probe.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-fix-acceptance.json`
- `docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-index.json`
