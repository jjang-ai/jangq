# Nemotron Ultra Component Budget Matrix

log_dir: `docs/runtime/logs`

## Sources
- layer: `2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json`
- moe: `2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json`
- mamba: `2026-06-04-nemotron-ultra-mamba-component-probe.json`
- projection: `2026-06-04-nemotron-ultra-projection-tradeoff-probe.json`
- token_budget: `2026-06-04-nemotron-ultra-token-speed-budget.json`

## Current Baseline
- manual_decode_total_ms: `143.237`
- manual_implied_tps: `6.981`
- moe_ms: `65.773`
- mamba_ms: `64.157`
- attention_ms: `8.990`
- norm_lm_head_ms: `4.317`
- moe_mamba_pct: `90.710`

## Component Cut Scenarios
| family | role | component | per-layer median | projected total | family coverage | 25% cut tps | 50% cut tps | 100% cut tps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MoE | `inclusive_path` | `full_moe` | `2.107` | `101.148` | `153.8%` | `8.478` | `10.792` | `23.759` |
| MoE | `substep` | `switch_mlp` | `1.130` | `54.264` | `82.5%` | `7.712` | `8.613` | `11.239` |
| MoE | `substep` | `shared_experts` | `0.577` | `27.720` | `42.1%` | `7.336` | `7.729` | `8.657` |
| MoE | `substep` | `fc1_latent_proj` | `0.231` | `11.068` | `16.8%` | `7.119` | `7.262` | `7.566` |
| MoE | `substep` | `gate` | `0.221` | `10.602` | `16.1%` | `7.113` | `7.250` | `7.539` |
| MoE | `substep` | `fc2_latent_proj` | `0.212` | `10.172` | `15.5%` | `7.108` | `7.238` | `7.515` |
| MoE | `substep` | `norm` | `0.203` | `9.732` | `14.8%` | `7.102` | `7.227` | `7.490` |
| MoE | `substep` | `score_weighted_sum` | `0.176` | `8.468` | `12.9%` | `7.086` | `7.194` | `7.420` |
| Mamba | `inclusive_path` | `full_mamba_mixer` | `1.197` | `57.480` | `89.6%` | `7.760` | `8.734` | `11.661` |
| Mamba | `substep` | `in_proj` | `0.835` | `40.062` | `62.4%` | `7.506` | `8.116` | `9.692` |
| Mamba | `substep` | `out_proj` | `0.470` | `22.564` | `35.2%` | `7.268` | `7.578` | `8.287` |
| Mamba | `substep` | `conv` | `0.216` | `10.360` | `16.1%` | `7.110` | `7.243` | `7.526` |
| Mamba | `substep` | `ssm_update` | `0.190` | `9.124` | `14.2%` | `7.094` | `7.211` | `7.456` |
| Mamba | `substep` | `mamba_norm_gated` | `0.178` | `8.544` | `13.3%` | `7.087` | `7.196` | `7.424` |
| Mamba | `substep` | `outer_norm` | `0.171` | `8.216` | `12.8%` | `7.083` | `7.188` | `7.406` |

## Target Coverage
- `10.000` tok/s needs `43.237` ms cut; single measured row/path enough: `MoE:full_moe`, `MoE:switch_mlp`, `Mamba:full_mamba_mixer`
- `12.000` tok/s needs `59.904` ms cut; single measured row/path enough: `MoE:full_moe`
- `15.000` tok/s needs `76.571` ms cut; single measured row/path enough: `MoE:full_moe`

## Projection Tradeoff
- `mamba_in_proj`: quantized `0.951` ms, BF16 `1.365` ms, speedup `1.43x`
- `mamba_out_proj`: quantized `0.512` ms, BF16 `0.672` ms, speedup `1.31x`
- `shared_up`: quantized `0.351` ms, BF16 `0.466` ms, speedup `1.33x`
- `shared_down`: quantized `0.356` ms, BF16 `0.545` ms, speedup `1.53x`

## Interpretation
- Component/path totals are projected from first measured layer medians; use them for ranking, not final proof.
- `full_*` rows are inclusive path measurements, not additive leaf substeps.
- Rows with large projected totals are plausible speed targets; small rows cannot move token/s enough alone.
- The current projection tradeoff says quantized 8-bit affine projections are faster than temporary BF16 copies.
