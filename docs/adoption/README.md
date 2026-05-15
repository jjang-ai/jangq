# Adopting JANG in your project

**JANG** is a mixed-precision quantization format for Apple Silicon that extends the MLX safetensors
convention. A JANG model is an mmap-able safetensors directory where each tensor is quantized to the
bit count that suits its role — attention at 6-8 bits, expert MLPs at 2-4 bits — so the model stays
coherent at drastically smaller sizes.

This directory is for **framework authors**: people who want to load or serve JANG models in their
own code.

If you are an end user who just wants to convert a model, open the main
[JANG Studio](https://jangq.ai) app instead.

## Pick your path

| You are... | Start here |
|---|---|
| A Python developer | [EXAMPLES/python.py](EXAMPLES/python.py) |
| A Swift developer | [EXAMPLES/swift.swift](EXAMPLES/swift.swift) |
| Running an agent / server | [EXAMPLES/server.md](EXAMPLES/server.md) |
| Adding JANG support to a new framework | [PORTING.md](PORTING.md) |
| Publishing a converted model | [MODEL_CARD_TEMPLATE.md](MODEL_CARD_TEMPLATE.md) |

## Format reference

- Canonical spec: [`FORMAT.md`](../../FORMAT.md) (top-level of this repo)
- JANGTQ (TurboQuant) extension: see `FORMAT.md` and [PORTING.md](PORTING.md) section "JANGTQ extensions"
- Model card schema: see [MODEL_CARD_TEMPLATE.md](MODEL_CARD_TEMPLATE.md)

## JANG vs JANGTQ

Two families of output:

- **JANG** — every architecture supported. Mixed-precision affine quantization using MLX-native
  `uint32` weight packing. Loads via any MLX-compatible runtime with one extra dequant step.
- **JANGTQ** (TurboQuant) — model-family gated support for the converters listed by
  `python -m jang_tools profiles --json` under `jangtq_families` (Qwen/MiniMax,
  Hy3, DeepSeek-V4, Kimi K2.6, ZAYA/ZAYA1-VL, Ling/Bailing, Nemotron-H,
  Laguna, and Mistral3/4 today). Uses a codebook-based format for expert MLP
  weights that cuts size by another 30-50% at the same quality. Requires a
  specialized loader (`jang_tools.load_jangtq.load_jangtq_model` in Python,
  or `JANGTQGenerator` from the `JANG` SwiftPM product in Swift — see `jang-runtime/Sources/JANG/JANGTQGenerator.swift`).

## Frameworks currently supporting JANG

- **MLX** via the `jang` Python package — reference implementation (Python)
- **JANGKit** — Swift high-level facade (`JANGKit.Model.load` + `generate`) built on top of `JANG`
- **JANGCore** — Swift/Metal on-disk format reader (manifest, expert index, tensor loading) shipped in JANG Studio
- **[MLX Studio](https://mlx.studio)** — GUI inference with JANG support
- **Osaurus** — OpenAI-compatible HTTP server (JANG + JANGTQ)

If you are porting to a new framework, read [PORTING.md](PORTING.md) first — it documents the
on-disk layout, per-tensor dequant math, and JANGTQ codebook structure.

## Questions?

File an issue at https://github.com/jjang-ai/jangq/issues with the `adoption` label.

Created by Jinho Jang (`eric@jangq.ai`).
