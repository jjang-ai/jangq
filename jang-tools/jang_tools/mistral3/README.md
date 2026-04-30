# Mistral-Medium-3.5-128B — JANG quant + runtime

`model_type=mistral3` wrapper / `Mistral3ForConditionalGeneration` with:
- text inner: `ministral3` — dense GQA 96/8, head_dim 128, 88 layers,
  hidden 12288, 256K YaRN (theta=1M, factor=64, orig=4096)
- vision: `pixtral` — 48 layers, hidden 1664, head_dim 104, image_size 1540,
  patch 14, spatial_merge_size 2

Source FP8 e4m3 with **per-tensor** scales (`weight_block_size: null`).
`modules_to_not_convert` keeps `model.vision_tower`, `model.multi_modal_projector`,
and `lm_head` in bf16 in the source — the converters below honor that and
carry those modules through unchanged.

## Convert

```
# JANGTQ2 (~36 GB) — text decoder MXTQ-2, vision/projector/lm_head bf16
python -m jang_tools.convert_mistral3_jangtq \
    ~/.mlxstudio/models/_sources/Mistral-Medium-3.5-128B \
    ~/.mlxstudio/models/JANGQ-AI/Mistral-Medium-3.5-128B-JANGTQ2  JANGTQ2

# MXFP4 (~70 GB) — text decoder mx.quantize bits=4 group=32
python -m jang_tools.convert_mistral3_mxfp4 \
    ~/.mlxstudio/models/_sources/Mistral-Medium-3.5-128B \
    ~/.mlxstudio/models/OsaurusAI/Mistral-Medium-3.5-128B-mxfp4
```

Or one-shot: `bash scripts/quant_mistral3.sh`

## Runtime (Python, text + image)

```
python -m jang_tools.mistral3.runtime \
    --src ~/.mlxstudio/models/JANGQ-AI/Mistral-Medium-3.5-128B-JANGTQ2 \
    --prompt 'Describe this image.' \
    --image /path/to/photo.jpg
```

Image preprocessing path: `jang_tools.vl.pixtral.PixtralImageProcessor`
matches Mistral 3.5's pixtral spec (image_size 1540, patch 14, spatial
merge 2). The runtime emits `[image_token_id] * num_patches` placeholders
that the LM forward replaces with vision-tower embeddings.

## Runtime (Swift, native pixtral preprocessing)

`swift/Sources/JANGImage/PixtralImageProcessor.swift` is the native Swift
port — uses CoreGraphics for resize + Accelerate for normalization.
Output matches the Python reference modulo resize-filter rounding (test
with `swift/Tests/JANGTests/PixtralTests.swift`).

The text decoder + vision tower live in vmlx-swift `vMLXLMCommon` once
the JANGTQ kernel binding is published; this Swift package handles the
preprocessing + bundle metadata layer.

## Eval

- HumanEval+ pass@1 (text only)
- MMLU
- Multilingual coverage (24 langs per upstream): FR / ES / DE / IT / PT / NL /
  ZH / JA / KO / AR sanity samples
- 256k context recall
- VL: short-caption + VQA on a small held-out set (vision tower is bf16
  in JANGTQ2, so VL quality should match upstream FP8 baseline within
  decode-noise)
