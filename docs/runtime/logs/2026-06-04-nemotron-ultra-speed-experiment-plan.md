# Nemotron Ultra Speed Experiment Plan

log_dir: `docs/runtime/logs`
layer_log: `2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json`
moe_log: `2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json`
mamba_log: `2026-06-04-nemotron-ultra-mamba-component-probe.json`

## Current Bottleneck
- manual synchronized decode: `143.237 ms/token`
- implied synchronized throughput: `6.981 tok/s`
- best live generator row: `8.335 tok/s`
- MoE total: `65.773 ms` across 48 layers
- Mamba total: `64.157 ms` across 48 layers
- attention total: `8.990 ms` across 12 layers
- final norm/lm_head: `4.317 ms`

## Ranked Experiments
| Rank | Experiment | Evidence | Target | Sensitivity |
| --- | --- | --- | --- | --- |
| 1 | MoE routed/shared scheduling or fused decode kernel | `E=65.773 ms`; single-layer `switch_mlp=1.130 ms`, `shared_experts=0.577 ms` | reduce fixed per-layer MoE overhead | 10% MoE cut implies `7.317 tok/s` synchronized |
| 2 | Mamba fused decode kernel / lower-overhead state update | `M=64.157 ms`; first-layer `in_proj=0.835 ms`, `out_proj=0.470 ms`, `conv=0.216 ms`, `ssm=0.190 ms` | reduce projection/dispatch overhead first | 10% Mamba cut implies `7.309 tok/s` synchronized |
| 3 | Joint MoE+Mamba scheduling path | `E+M=129.930 ms` dominates decode | attack Python/MLX dispatch and sync boundaries | 10% combined cut implies `7.678 tok/s` synchronized |
| 4 | Ahead-of-time warmup plan | cold JIT is about 33s, warmed TTFT about 1s | make startup predictable, not steady decode faster | improves TTFT, not tok/s |

## Negative Controls
- Do not chase attention first: it is the smallest bucket after BF16 retention.
- Do not dequantize 8-bit Mamba/shared projections for speed; current probe says quantized is faster.
- Do not lower router top-k as the main fix; top-k 8 did not materially improve live decode.
- Do not replace `mlx_lm.generate_step` with a Python argmax loop; manual loop was slower.
- Do not hide parser/coherence problems with prompt suffixes, forced closing tags, or sampler tricks.

## Projection Evidence
- `mamba_in_proj`: quantized `0.951 ms`, BF16 dequantized `1.365 ms`, quantized speedup `1.43x`
- `mamba_out_proj`: quantized `0.512 ms`, BF16 dequantized `0.672 ms`, quantized speedup `1.31x`
- `shared_up`: quantized `0.351 ms`, BF16 dequantized `0.466 ms`, quantized speedup `1.33x`
- `shared_down`: quantized `0.356 ms`, BF16 dequantized `0.545 ms`, quantized speedup `1.53x`

## Mamba Component Evidence
- `outer_norm`: `0.171 ms`
- `in_proj`: `0.835 ms`
- `conv`: `0.216 ms`
- `ssm_update`: `0.190 ms`
- `mamba_norm_gated`: `0.178 ms`
- `out_proj`: `0.470 ms`
- `full_mamba_mixer`: `1.197 ms`
Interpretation: the generic grouped conv and SSM update are not the largest isolated Mamba substeps in this probe; projection/dispatch fusion is the more credible first Mamba speed target.

## Next Proof Rows
- rerun layer decode after any MoE or Mamba runtime change
- rerun live speed probe after warm compile
- rerun long coherence probe; speed wins cannot regress parser/coherence
- keep cache/VL proof separate from speed proof
