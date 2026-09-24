# `jang_tools.dsv4`

DeepSeek-V4 (Flash 284B / Pro 1.6T) JANG quantization + MLX runtime.

## Convert a model

```bash
# JANGTQ2 V3 — current DSV4 runtime-candidate lane. Drops MTP, keeps
# non-routed modules 8-bit affine, and applies an explicit V3 routed bit plan.
DSV4_V3_PLAN_PATH=/path/to/dsv4_v3_bit_plan.json \
python -m jang_tools.dsv4.convert_dsv4_jangtq \
  --src <path/to/DSV4-Flash-source> \
  --dst <path/to/output-bundle> \
  --profile 2 --format jangtq --variant V3

# uniform `std` JANGTQ2 — baseline/repro only. It keeps MTP and is
# not a production-cleared DSV4 runtime candidate.
python -m jang_tools.dsv4.convert_dsv4_jangtq \
  --src <path/to/DSV4-Flash-source> \
  --dst <path/to/std-baseline-output-bundle> \
  --profile 2 --format jangtq --variant std

# JANG_2L — 2-bit affine everywhere (107 GB).
python -m jang_tools.dsv4.convert_dsv4_jangtq \
  --src <path/to/source> --dst <path/to/out> \
  --profile 2 --format jang

# JANGTQ4 — 4-bit affine routed + 8-bit attn (173 GB). Highest fidelity at 4-bit.
python -m jang_tools.dsv4.convert_dsv4_jangtq \
  --src <path/to/source> --dst <path/to/out> \
  --profile 4 --format jangtq
```

Source can be the original FP4+FP8 release OR a BF16 dequant — the convert
script auto-detects and falls through to `mx.quantize` when source lacks
`.scale` siblings.

## Run

```python
from jang_tools.load_jangtq import load_jangtq_model
from jang_tools.dsv4.encoding_adapter import load_encoding  # set DSV4_ENCODING_DIR env
from mlx_lm import generate

model, tok = load_jangtq_model("<path/to/bundle>")
encode_messages = load_encoding().encode_messages
prompt = encode_messages([{"role": "user", "content": "..."}], thinking_mode="chat")
out = generate(model, tok, prompt=prompt, max_tokens=200)
```

## Bundle formats

| Format | Routed | Non-routed | Size (V4-Flash) | Notes |
|---|---|---|---|---|
| `jang` profile=2 | 2-bit affine | 2-bit affine | 107 GB | Most compact |
| `jang` profile=4 | 4-bit affine | 4-bit affine | 173 GB | Highest fidelity |
| `jangtq` profile=2 `--variant V3` | 2/4-bit MXTQ with explicit V3 plan | 8-bit affine | ~80 GB | Current runtime-candidate lane; production builds need `DSV4_V3_PLAN_PATH` |
| `jangtq` profile=2 `--variant std` | uniform 2-bit MXTQ | 8-bit affine | ~74 GB | Baseline/repro only; keeps MTP, not production-cleared |
| `jangtq` profile=4 | 4-bit affine | 8-bit affine | 173 GB | High fidelity, larger |

**Avoid** mxfp4-direct-copy + bf16-passthrough (HP) format — unstable inference
(residual stream explodes over 43 layers).

## Architecture notes

DSV4 differs significantly from V3:
- mHC (Manifold Hyper-Connections) wraps every block: residual stream has
  `hc_mult=4` parallel copies, collapsed and re-mixed via Sinkhorn-normalized
  doubly-stochastic comb matrix per block.
- MLA with `head_dim=512` (vs typical 128). Single KV head broadcast via GQA.
- Grouped low-rank output projection (`o_groups=8`, `o_lora_rank=1024`).
- Sqrtsoftplus expert routing with `noaux_tc` bias for top-k.
- Hash routing for first 3 layers (`tid2eid` token→expert lookup).
- Per-layer RoPE: `compress_ratio > 0` uses `compress_rope_theta=160000` + YaRN;
  `compress_ratio == 0` uses `rope_theta=10000` no YaRN.
- Attention sink: learned per-head bias logit, prepended to softmax, dropped after.
- Inverse RoPE on attention output before grouped O projection.
- SwiGLU clamp: `silu(min(gate, 10)) * clip(up, ±10)`.
- Compressor + Indexer for compressed long-context attention (compress_ratios `{4, 128}`
  alternating across middle 41 layers).
- Ratio-4 compressor rows are overlapping: decode state must retain the
  previous complete window plus the current partial/complete window, then build
  rows from previous-window first-half features and current-window second-half
  features. `DeepseekV4Cache.accumulate_overlap_windows()` owns this state for
  MLX cache/paged/L2 compatibility; a plain remainder buffer is wrong.

## Verification

`diff_one_block.py` / `diff_per_op.py` / `diff_real_layer0.py` compare the
MLX runtime against the torch reference at varying granularities.

Focused overlap-cache contracts live in
`tests/test_dsv4_overlap_cache.py`. They pin source-equivalent prefill
retention, decode-boundary row construction, and `Compressor` pool growth for
`compress_ratio=4`.
