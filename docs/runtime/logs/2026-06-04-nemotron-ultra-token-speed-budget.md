# Nemotron Ultra Token Speed Budget

log_dir: `docs/runtime/logs`
layer_log: `2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json`

## Current Baseline
- manual_decode_total_ms: `143.237`
- manual_implied_tps: `6.981`
- best_live_tps: `8.335` from `2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json::think_math_default`
- moe_ms: `65.773`
- mamba_ms: `64.157`
- attention_ms: `8.990`
- norm_lm_head_ms: `4.317`
- other_ms: `0.000`
- moe_plus_mamba: `129.930` ms (`90.71%` of manual decode)

## Target Budgets
| target tok/s | target ms/token | total cut needed | total cut % | MoE cut | Mamba cut | per MoE layer | per Mamba layer | MoE/Mamba enough |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `10.000` | `100.000` | `43.237` | `30.19%` | `21.888` (`33.28%`) | `21.350` (`33.28%`) | `0.4560` | `0.4448` | `True` |
| `12.000` | `83.333` | `59.904` | `41.82%` | `30.325` (`46.10%`) | `29.579` (`46.10%`) | `0.6318` | `0.6162` | `True` |
| `15.000` | `66.667` | `76.571` | `53.46%` | `38.762` (`58.93%`) | `37.809` (`58.93%`) | `0.8075` | `0.7877` | `True` |

## Interpretation
- Use manual synchronized decode for millisecond budgets; use live speed for user-visible baseline.
- If a target is not reachable by MoE/Mamba only, attention/lm_head/loop work also must move.
- Per-layer cuts are proportional planning numbers, not proof of an implementation strategy.
