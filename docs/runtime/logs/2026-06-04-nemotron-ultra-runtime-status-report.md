# Nemotron Ultra Runtime Status

log_dir: docs/runtime/logs

## Speed
FIXED/PARTIAL: best observed warm decode row is 8.335 tok/s
source: 2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json :: think_math_default
remaining: MoE/Mamba forward overhead; not sampler or generic generation loop

## Coherence
PARTIAL: visible parser marker leakage, high repeated n-gram fraction, at least one row did not reach EOS
- factual_japan: eos=True speed=8.178 leaks=['</think>'] repeat_fraction=0.5151515151515151
- arithmetic_brief: eos=True speed=8.136 leaks=['</think>'] repeat_fraction=0.43902439024390244
- reasoning_apples: eos=False speed=8.133 leaks=['</think>'] repeat_fraction=0.06451612903225806

## Layer Split
FOUND
manual_decode_total_ms: 143.23724881978706
norm_lm_head_ms: 4.3167500407435
- MoE: total_ms=65.773 count=48 median_ms=1.269
- Mamba: total_ms=64.157 count=48 median_ms=1.241
- Attention: total_ms=8.99 count=12 median_ms=0.747

## Projection Tradeoff
FOUND
Use quantized 8-bit affine projections unless a new probe proves otherwise.
- mamba_in_proj: quantized_median_ms=0.951 bf16_median_ms=1.365 speedup=1.43x
- mamba_out_proj: quantized_median_ms=0.512 bf16_median_ms=0.672 speedup=1.31x
- shared_up: quantized_median_ms=0.351 bf16_median_ms=0.466 speedup=1.33x
- shared_down: quantized_median_ms=0.356 bf16_median_ms=0.545 speedup=1.53x

## Mamba Component
FOUND
- outer_norm: median_ms=0.171
- in_proj: median_ms=0.835
- conv: median_ms=0.216
- ssm_update: median_ms=0.190
- mamba_norm_gated: median_ms=0.178
- out_proj: median_ms=0.470
- full_mamba_mixer: median_ms=1.197
Interpretation: projection/dispatch fusion is a better first target than a Python-level conv rewrite.

## Cache / VL Gates
PARTIAL: cache and VL gates are documented, not live-proven in vMLX.
- TurboQuant KV only covers 12 attention layers.
- Full prefix hit also requires 48 Mamba companion states.
- Parser streaming state must be salted/restored.
- This artifact is text-only; media requests must reject or reroute.
