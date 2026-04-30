# Laguna (poolside) — JANG quant + runtime

`model_type=laguna` — 33B/3B agentic-coding MoE, 40 layers, hybrid SWA+full
with PER-LAYER head count (48 full / 64 SWA), dual RoPE (full=YaRN/swa=default),
256 routed experts top-8 + 1 shared, sigmoid routing with per-head gating
(`g_proj`), q_norm/k_norm in attention. Text-only; no VL/audio/video.

## Convert

```
# JANGTQ2 (~7 GB)
python -m jang_tools.convert_laguna_jangtq \
    ~/.mlxstudio/models/_sources/Laguna-XS.2 \
    ~/.mlxstudio/models/JANGQ-AI/Laguna-XS.2-JANGTQ2  JANGTQ2

# MXFP4 (~17 GB)
python -m jang_tools.convert_laguna_mxfp4 \
    ~/.mlxstudio/models/_sources/Laguna-XS.2 \
    ~/.mlxstudio/models/OsaurusAI/Laguna-XS.2-mxfp4
```

Or one-shot: `bash scripts/quant_laguna.sh`

## Runtime (Python)

```
python -m jang_tools.laguna.runtime \
    --src ~/.mlxstudio/models/JANGQ-AI/Laguna-XS.2-JANGTQ2 \
    --prompt 'def fibonacci(n):' --max-new 32
```

Auto-detects `weight_format` (bf16 / mxtq / mxfp4) and dispatches. The
sanity rule (per `feedback_runtime_before_quant`): always run a
`--no-cache` greedy loop on the bf16 source first to verify the MLX port
matches the reference, THEN test quantized bundles.

## Runtime (Swift)

`swift/Sources/JANGRuntime/BundleLoader.swift` opens the bundle and
returns a `BundleHandle` with `QuantMeta` (format + bits + group_size).
Weight realization currently delegates to vmlx-swift's `vMLXLMCommon`
TurboQuant kernels (planned binding). The `JANGCxx` C++ shim
(`jang_tq_decode_bf16`) gives a stable C ABI for that bind.

For now the Swift surface validates bundle metadata + drives Python
runtime via subprocess; once vmlx-swift exposes the kernels publicly the
shim body becomes the production path.
