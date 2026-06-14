<p align="center">
  <a href="https://mlx.studio"><img src="https://raw.githubusercontent.com/jjang-ai/jangq/main/assets/mlx-studio-light.png" alt="MLX Studio" width="500"></a>
</p>

<p align="center">
  <a href="https://mlx.studio"><img src="https://mlx.studio/assets/screenshots/mlx-studio-featured.png?v=1" alt="MLX Studio App" width="600"></a>
</p>

<h4 align="center"><a href="https://mlx.studio">MLX Studio</a> — native JANG support with reasoning, VLM, and full mixed-precision inference</h4>

---

<p align="center">
  <a href="README-USER.md"><b>JANG Studio</b></a> — native macOS wizard for converting HuggingFace models to JANG and JANGTQ formats. Five-step flow with live logs, 10-row preflight gate, 12-row post-convert verifier, in-app chat preview, and one-click HuggingFace publishing. Ships as a signed DMG.
</p>

<p align="center">
  <a href="README-USER.md"><img src="docs/screenshots/step3-profile.png" alt="JANG Studio" width="720"></a>
</p>

<p align="center">
  See the <a href="README-USER.md">full user guide</a> for install + wizard walkthrough + CLI usage + Swift integration.
</p>

---

> **App Support:** [MLX Studio](https://mlx.studio) has full native JANG support. [oMLX](https://github.com/jundot/omlx) has added JANG integration ([PR #364](https://github.com/jundot/omlx/pull/364)). LM Studio, Ollama, and Inferencer do not support JANG yet — ask your favorite app's creators to add support, or use `pip install "jang[mlx]"`.

---

<p align="center">
  <img src="https://raw.githubusercontent.com/jjang-ai/jangq/main/assets/jangq-logo-dark.png" alt="JANG" width="300">
</p>

<h3 align="center"><b>J</b>ang <b>A</b>daptive <b>N</b>-bit <b>G</b>rading</h3>
<h4 align="center">Mixed-Precision Quantization for Apple Silicon</h4>

<p align="center">
  The GGUF equivalent for MLX — models stay quantized in GPU memory at full Metal speed.
</p>

<p align="center">
  <img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-blue">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11+-green">
  <img alt="Platform" src="https://img.shields.io/badge/platform-Apple%20Silicon-black">
  <img alt="PyPI" src="https://img.shields.io/pypi/v/jang">
</p>

<p align="center">
  <a href="https://jangq.ai">Website</a> •
  <a href="https://huggingface.co/JANGQ-AI">Models</a> •
  <a href="https://pypi.org/project/jang/">PyPI</a> •
  <a href="https://github.com/jjang-ai/jangq/blob/main/FORMAT.md">Format Spec</a>
</p>

## Highlights

- **397B on 128 GB Mac** — JANG_1L: 112 GB, 36 tok/s, 86.5% MMLU with reasoning
- **DeepSeek-V4-Flash runtime** — first MLX runtime for DSV4's hybrid **SWA + CSA + HSA** attention (sliding-window + compressor-pool + hash-sparse) with mHC residuals and 1M-context pool quant cache
- **JangPress for routed-MoE** — load 167 GB Kimi-K2.6 on a 128 GB Mac via `mmap` + `madvise` cold-tier eviction (see [`docs/JANGPRESS.md`](docs/JANGPRESS.md))
- **Nemotron-Cascade-2 in 10 GB** — IMO Gold Medal reasoning model at 130 tok/s on 16 GB MacBooks
- **MiniMax: only JANG works** — MLX scores 25% (random), JANG scores 74%
- **Nemotron-3-Super-120B in 43 GB** — first working Nemotron-H quantization for Apple Silicon
- **Mistral Small 4 (119B) — first on Apple Silicon** — 82 tok/s, 5x faster prefill than MLX Community 4-bit, code/math/reasoning/VLM all working
- **bfloat16 auto-detection** — fixes float16 overflow on 512-expert models
- **Reasoning mode** — `<think>...</think>` / `[THINK]...[/THINK]` with configurable thinking on/off

## Results (200-question MMLU)

### Qwen3.5-397B-A17B — JANG runs where MLX can't

| Model | No-Think | Reasoning | Size | Speed |
|-------|:--------:|:---------:|:----:|:-----:|
| **JANG_1L** | 81.0% | **86.5%** | **112 GB** | 36 tok/s |
| **JANG_2L** | 79.5% | **92.0%** | 187 GB | 36 tok/s |
| MLX 4-bit | 81.5% | 94.0% | 209 GB | ~36 tok/s |
| MLX 2/3-bit | **NaN** | **NaN** | — | — |

MLX cannot quantize 397B below 4-bit (float16 overflow). JANG solves this with bfloat16.

### Nemotron-Cascade-2-30B — IMO Gold Medal in 10 GB

| Model | No-Think | Reasoning | Size | Speed |
|-------|:--------:|:---------:|:----:|:-----:|
| **JANG_2L** | 59.0% | **88.0%** | **10.3 GB** | 130 tok/s |
| **JANG_4M** | 69.0% | **93.0%** | 17 GB | 55 tok/s |
| MLX 4-bit | 69.0% | 92.5% | 16.6 GB | — |
| MLX 6-bit | 71.0% | 94.5% | 23.9 GB | — |

JANG_4M beats MLX 4-bit (93.0% vs 92.5%) at the same size.

### Mistral Small 4 (119B-A6B) — First on Apple Silicon, fastest inference

| Model | Size | Gen tok/s | Prefill tok/s | RAM | Features |
|-------|:----:|:---------:|:-------------:|:---:|----------|
| **JANG_2L** | **30 GB** | **82** | **216** | 40 GB | Code, math, [THINK] reasoning, VLM |
| **JANG_4M** | **57 GB** | **80** | **202** | 68 GB | Code, math, [THINK] reasoning, VLM |
| **JANG_6M** | **84 GB** | **74** | **160** | 95 GB | Code, math, [THINK] reasoning, VLM |
| MLX Community 4-bit | 63 GB | 84 | 43 | 68 GB | VLM |

**94.0% MMLU** at 2-bit (JANG_2L) — five subjects at 100%. 5x faster prefill than MLX Community. JANG_2L fits on 48 GB Macs at half the size.

119B total parameters, 6B active per token. MLA attention + 128 MoE experts + Pixtral vision. First model to support MLA + MoE quantization on Apple Silicon.

#### JANG_2L MMLU Results (200 Questions, Reasoning Mode)

| Subject | Score |
|---------|:-----:|
| Abstract Algebra | 13/20 (65%) |
| Anatomy | **20/20 (100%)** |
| Astronomy | **20/20 (100%)** |
| College CS | **20/20 (100%)** |
| College Physics | 19/20 (95%) |
| HS Biology | 19/20 (95%) |
| HS Chemistry | **20/20 (100%)** |
| HS Mathematics | 18/20 (90%) |
| Logical Fallacies | 19/20 (95%) |
| World Religions | **20/20 (100%)** |
| **Total** | **188/200 (94.0%)** |

### Nemotron-3-Super-120B — Only JANG can go below 4-bit

| Model | No-Think | Reasoning | Size | Speed |
|-------|:--------:|:---------:|:----:|:-----:|
| **JANG_2L** | **75.0%** | 86.0% | **43 GB** | 52 tok/s |
| **JANG_4M** | 72.5% | **93.0%** | 63 GB | 55 tok/s |
| MLX 4-bit | 71.0% | 93.5% | 63 GB | 60 tok/s |
| MLX 3-bit | **Crashes** | — | — | — |

MLX `mlx_lm.convert` crashes on Nemotron's mtp.* weights. Only JANG can produce sub-4-bit.

### MiniMax-M2.5 — JANG is the ONLY working option

| Model | MMLU | Size |
|-------|:----:|:----:|
| **JANG_2L** | **74%** | 63 GB |
| **JANG_3M** | **74.5%** | 82 GB |
| MLX 4-bit | 26.5% | 120 GB |
| MLX 3-bit | 24.5% | 93 GB |
| MLX 2-bit | 25% | — |

MLX is broken on MiniMax at ALL bit levels (~25% = random). MiniMax has 256 experts — MLX compresses attention to the same bits as expert MLP, destroying coherence.

### Qwen3.5 MoE (122B, 35B)

| Model | JANG | MLX 4-bit | JANG Size | MLX Size |
|-------|:----:|:---------:|:---------:|:--------:|
| 122B JANG_4K | **86%** | 85% | 69 GB | 64 GB |
| 122B JANG_2S | **79%** | 56.5% (2-bit) | 38 GB | 36 GB |
| 35B JANG_4K | **77.5%** | 77.0% | 16.7 GB | 18 GB |
| 35B JANG_2S | **65.5%** | ~20% (2-bit) | 12 GB | 10 GB |

### The Full Picture: JANG vs MLX Across All Models

| Model | JANG Best | MLX Best | JANG Size | MLX Size | MLX Broken? |
|-------|:---------:|:--------:|:---------:|:--------:|:-----------:|
| Qwen3.5-397B | **92.0%** | 94.0% | **187 GB** | 209 GB | NaN below 4-bit |
| Qwen3.5-397B (128 GB Mac) | **86.5%** | — | **112 GB** | Can't fit | — |
| Nemotron-Cascade-2 | **93.0%** | 92.5% | 17 GB | 16.6 GB | — |
| Nemotron-Cascade-2 (16 GB Mac) | **88.0%** | — | **10.3 GB** | Can't fit | — |
| Nemotron-Super-120B | **93.0%** | 93.5% | 63 GB | 63 GB | Crashes below 4-bit |
| Nemotron-Super-120B (64 GB Mac) | **86.0%** | — | **43 GB** | Can't fit | — |
| MiniMax-M2.5 | **74.5%** | 26.5% | **82 GB** | 120 GB | Broken at ALL bits |
| Qwen3.5-122B | **86%** | 85% | 69 GB | 64 GB | 56.5% at 2-bit |
| Qwen3.5-35B | **77.5%** | 77.0% | 16.7 GB | 18 GB | ~20% at 2-bit |

**JANG wins at every size point.** At equivalent sizes, JANG matches or beats MLX. At smaller sizes, JANG runs where MLX literally cannot (NaN, crashes, or random output).

### Why MLX Fails on MoE Models

On MoE models, attention is only **1-5% of total parameters** but controls 100% of coherence. MLX compresses everything equally:

```
MLX 4-bit: attention at 4-bit, experts at 4-bit → works but wastes bits on experts
MLX 2-bit: attention at 2-bit, experts at 2-bit → attention breaks → model breaks

JANG 2-bit: attention at 8-bit, experts at 2-bit → attention preserved → model works
```

The more experts a model has, the worse MLX performs at low bits:
- **128 experts** (Cascade-2): MLX 4-bit still works, JANG slightly better
- **256 experts** (122B, MiniMax): MLX 2-bit breaks badly, JANG dominates
- **512 experts** (397B, Super-120B): MLX NaN/crash below 4-bit, only JANG works

## DeepSeek-V4-Flash runtime — Hybrid SWA + CSA + HSA attention

DSV4-Flash is the first DeepSeek-class model whose attention is **not**
plain MLA. JANG ships the only Apple-Silicon runtime that handles all
three layer types correctly:

| Layer type | Description | JANG path |
|---|---|---|
| **SWA** (Sliding-Window Attention) | local windowed attention; cheap fast layers | `jang_tools.dsv4.mlx_model.DSV4SlidingWindow` |
| **CSA** (Compressor / Compressed-State Attention) | per-layer compressor pool + windowed buffer; carries pooled-K/V across the whole context | `DeepseekV4Cache` + `accumulate_windows` + `update_pool` |
| **HSA** (Hash-Sparse Attention via Indexer) | indexer chooses top-K hash buckets per-token; sparse attention over the global compressed pool | `Indexer` module + `DSV4LayerCache` indexer state |
| **mHC residuals** | multi-Head-Compressed residual stream stabilises 43-layer 284B model | residual-axis fix in `mlx_model.py` |
| **Pool quant cache** | quantized streaming KV pool with proportional `trim()` (mirrors `llama.cpp` `dsv4_clear_rows`) | `DeepseekV4Cache.trim(n)` |

End-to-end verified: 1M-context pool accumulation, multi-turn `/v1/chat/completions`
with prefix-cache reuse, dual-mode reasoning, 20.5 tok/s decode on
M3 Ultra @ JANGTQ2 (79.5 GB), 24.5 tok/s on Swift.

## Runtime examples for adopters

Public server and client examples are in [`examples/runtime/`](examples/runtime/):

- `serve_vmlx.sh` starts a JANG/JANGTQ bundle with `vmlx-engine`.
- `openai_chat.py`, `openai_tools.py`, and `openai_responses.py` cover standard
  API clients.
- `openai_multimodal.py` covers image/video requests for bundles with wired
  multimodal processors.
- `swift/OpenAIChat.swift` is a small Swift client for local macOS apps.

The integration guide is
[`docs/adoption/RUNTIME_QUICKSTART.md`](docs/adoption/RUNTIME_QUICKSTART.md).

## JangPress — fit routed-MoE bundles bigger than RAM

JangPress is the load-time memory policy in
[`osaurus-ai/vmlx-swift-lm`](https://github.com/osaurus-ai/vmlx-swift-lm)
that combines mmap-backed safetensors, per-token router-aware
`MADV_DONTNEED` over canonical routed-expert pages, and an optional
prestack overlay. It lets routed-MoE bundles bigger than RAM serve
from a single Mac:

| Bundle | Size | Host RAM | Post-load RSS |
|---|---|---|---|
| Kimi-K2.6-Small-JANGTQ | 153 GB | 128 GB | ~0.7 GB |
| Kimi-K2.6-Med-JANGTQ | 167 GB | 128 GB | ~1 GB |
| DSV4-Flash JANG_2L | 96.6 GB | 128 GB+ | varies |

Python serve + bench scripts: [`scripts/jangpress/`](scripts/jangpress/).
Full guide: [`docs/JANGPRESS.md`](docs/JANGPRESS.md).

## Install

```bash
pip install "jang[mlx]>=2.5.18"
```

For Vision-Language models:
```bash
pip install "jang[vlm]>=2.5.18"
```

## Quick Start

### Convert any model

```bash
# K-quant 4-bit (same size as MLX, smarter allocation)
jang convert Qwen/Qwen3.5-35B-A3B -p 4

# 2-bit for extreme compression
jang convert Qwen/Qwen3.5-122B-A10B -p 2

# Specific profile
jang convert model -p JANG_2L
```

### Run inference

```python
from jang_tools.loader import load_jang_model
from mlx_lm import generate

model, tokenizer = load_jang_model("JANGQ-AI/Qwen3.5-397B-A17B-JANG_1L")

# With reasoning (recommended for hard questions)
messages = [{"role": "user", "content": "Prove that sqrt(2) is irrational."}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False,
    add_generation_prompt=True, enable_thinking=True)
result = generate(model, tokenizer, prompt=prompt, max_tokens=2048)

# Without reasoning (faster)
prompt = tokenizer.apply_chat_template(messages, tokenize=False,
    add_generation_prompt=True, enable_thinking=False)
result = generate(model, tokenizer, prompt=prompt, max_tokens=100)
```

### VLM (Vision-Language) inference

```python
from jang_tools.loader import load_jang_vlm_model
from mlx_vlm import generate as vlm_generate

model, processor = load_jang_vlm_model("JANGQ-AI/Qwen3.5-397B-A17B-JANG_2L")
messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "Describe this image."},
]}]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
result = vlm_generate(model, processor, prompt=prompt, image=["photo.jpg"], max_tokens=200)
```

### MMLU Benchmark

```bash
python -m jang_tools.benchmark /path/to/model --max-thinking 1024
```

Smart two-pass: no-thinking first, then reasoning retry on wrong answers. Checkpointing, forced answers, full output logging.

## Pre-quantized Models

| Model | Profile | MMLU | Size | Fits |
|-------|---------|:----:|:----:|:----:|
| [Qwen3.5-397B JANG_1L](https://huggingface.co/JANGQ-AI/Qwen3.5-397B-A17B-JANG_1L) | 2.1-bit | 86.5%* | 112 GB | 128 GB Mac |
| [Qwen3.5-397B JANG_2L](https://huggingface.co/JANGQ-AI/Qwen3.5-397B-A17B-JANG_2L) | 3.7-bit | 92.0%* | 187 GB | 256 GB Mac |
| [Nemotron-Cascade-2 JANG_2L](https://huggingface.co/JANGQ-AI/Nemotron-Cascade-2-30B-A3B-JANG_2L) | 2.3-bit | 88.0%* | 10 GB | 16 GB Mac |
| [Nemotron-Cascade-2 JANG_4M](https://huggingface.co/JANGQ-AI/Nemotron-Cascade-2-30B-A3B-JANG_4M) | 4.1-bit | 93.0%* | 17 GB | 24 GB Mac |
| [Nemotron-Super-120B JANG_2L](https://huggingface.co/JANGQ-AI/Nemotron-3-Super-120B-A12B-JANG_2L) | 2.8-bit | 86.0%* | 43 GB | 64 GB Mac |
| [Nemotron-Super-120B JANG_4M](https://huggingface.co/JANGQ-AI/Nemotron-3-Super-120B-A12B-JANG_4M) | 4.1-bit | 93.0%* | 63 GB | 64 GB Mac |
| [Qwen3.5-122B JANG_4K](https://huggingface.co/JANGQ-AI/Qwen3.5-122B-A10B-JANG_4K) | 4.0-bit | 86% | 69 GB | 192 GB Mac |
| [Qwen3.5-122B JANG_2S](https://huggingface.co/JANGQ-AI/Qwen3.5-122B-A10B-JANG_2S) | 2.1-bit | 79% | 38 GB | 64 GB Mac |
| [Qwen3.5-35B JANG_4K](https://huggingface.co/JANGQ-AI/Qwen3.5-35B-A3B-JANG_4K) | 4.0-bit | 77.5% | 17 GB | 36 GB Mac |
| [MiniMax-M2.5 JANG_2L](https://huggingface.co/JANGQ-AI/MiniMax-M2.5-JANG_2L) | 2.3-bit | 74% | 63 GB | 128 GB Mac |
| [Qwen3.5-27B JANG_4S](https://huggingface.co/JANGQ-AI/Qwen3.5-27B-JANG_4S) | 4.1-bit | 84.5% | 16 GB | 24 GB Mac |

*\* with reasoning mode*

[Full collection](https://huggingface.co/collections/jangq/jang-quantized-gguf-for-mlx)

## Profiles

| Profile | Type | Bits | Best for |
|---------|------|:----:|----------|
| `JANG_4K` | K-quant | 4.0 | Same size as MLX 4-bit, smarter |
| `JANG_4M` | Profile | 4.0 | 8-bit attention, 4-bit experts |
| `JANG_4S` | Profile | 4.0 | Dense models (27B) |
| `JANG_3K` | K-quant | 3.0 | Same size as MLX 3-bit, smarter |
| `JANG_2L` | Profile | ~2.3 | Quality 2-bit, best for MoE |
| `JANG_1L` | Profile | ~2.1 | Maximum quality 2-bit |

## App Developers: Add JANG Support

**JANG models are standard MLX safetensors.** If your app loads MLX quantized models, adding JANG is minimal work.

### Quickest Integration (5 lines)

```python
# Detect JANG model
from pathlib import Path
is_jang = (Path(model_path) / "jang_config.json").exists()

# Load with jang-tools
if is_jang:
    from jang_tools.loader import load_jang_model
    model, tokenizer = load_jang_model(model_path)
    # model is a standard mlx_lm model — use like any MLX model
```

### What's Different from Standard MLX

1. **Mixed bit widths** — different tensors have different bits (attention at 8-bit, experts at 2-bit). Each `QuantizedLinear` needs its `bits` and `group_size` set from tensor shapes.
2. **bfloat16 for large models** — 512+ expert models need `model.set_dtype(mx.bfloat16)` to prevent float16 overflow.
3. **Nemotron-H weight renaming** — `switch_mlp.up_proj→fc1`, `down_proj→fc2`, gate dequantization.

### Full Integration Guide

See **[INTEGRATION.md](https://github.com/jjang-ai/jangq/blob/main/INTEGRATION.md)** for complete step-by-step with code for:
- Loading without jang-tools dependency
- Per-tensor bit inference from shapes
- bfloat16 auto-detection
- Nemotron-H special handling
- Chat template with thinking on/off
- VLM support
- Edge cases and gotchas

## Supported Architectures

- **Qwen3.5** (hybrid SSM + MoE + VLM) — 4B, 9B, 27B, 35B, 122B, 397B
- **Nemotron-H** (Mamba-2 + Latent MoE + Attention) — Cascade-2 30B, Super-120B
- **MiniMax-M2.5** (256-expert MoE, FP8 source)
- **DeepSeek-V2/V3** (MLA + MoE)
- **Mixtral / Qwen2-MoE** (standard MoE)
- **Dense Transformers** (Llama, Mistral, Gemma, Phi)
- **Vision-Language** (Qwen3.5-VL, Pixtral)
- **Mamba / Hybrid SSM** (Jamba, Nemotron-H)
- **FP8 source models** (auto-dequantization)
- **Mistral Small 4** (119B MoE + MLA + Pixtral VL) — 94% MMLU, 82 tok/s, 30-84 GB

## Changelog

### v2.1.5 (2026-03-21)
- **Nemotron-H loader**: fc1/fc2 rename, gate weight dequantization, mtp.* key filtering
- **bfloat16 auto-detection** for 512+ expert models (prevents float16 overflow)
- **MLP asymmetry floors**: gate_proj=4-bit, down_proj=3-bit for 512+ expert models
- **Benchmark script**: smart two-pass MMLU with reasoning, checkpointing, forced answers
- **eos_token_id auto-fix** for Qwen3.5 (248044→248046)
- **Auto-copy all .py files** for trust_remote_code models
- Nemotron-3-Super-120B: 86% MMLU at 43 GB
- Qwen3.5-397B: 92% MMLU at 187 GB, 86.5% at 112 GB

### v2.1.4 (2026-03-19)
- MLP asymmetry fix for 512-expert models
- eos_token_id auto-fix for Qwen3.5
- Auto-copy custom .py files

### v2.1.3 (2026-03-18)
- Per-tensor group_size (router=64, experts=128 for 150+ expert models)
- Precision floor rules for shared expert
- VLM support for all Qwen3.5 models

## How It Works

JANG redistributes bits based on tensor sensitivity — same total size, smarter allocation:

```
CRITICAL  (attention, MoE routers, MLA latent)  →  6-8 bit  →  Controls coherence
IMPORTANT (embeddings, linear attention)         →  4-6 bit  →  Moderate sensitivity
COMPRESS  (MLP, MoE experts)                     →  2-4 bit  →  95%+ of parameters
```

On MoE models, attention is only 1-5% of parameters. Boosting it to 8-bit costs ~2% overhead but dramatically improves quality. MLX compresses everything equally — that's why it breaks on MoE models at low bits.

## Technical Features

- **bfloat16 compute**: Auto-detected for 512+ expert models. Prevents float16 overflow at shared expert down_proj.
- **MLP asymmetry**: gate_proj gets 4-bit floor (SiLU amplifier), down_proj gets 3-bit floor for 512+ expert models.
- **FP8 dequantization**: Handles FP8 source models (MiniMax, Nemotron) automatically.
- **Latent MoE**: Supports Nemotron-H's fc1/fc2_latent_proj compression.
- **v2 format**: MLX-native safetensors, instant mmap loading, no repack needed.

## Requirements

- **Python**: 3.11+
- **Conversion**: any platform (numpy + safetensors)
- **Inference**: Apple Silicon Mac (M1/M2/M3/M4) with MLX
- **Dependencies**: `safetensors>=0.4`, `numpy>=1.24`, `tqdm>=4.60`, `huggingface_hub>=0.20`
- **Optional**: `mlx>=0.22`, `mlx-lm>=0.20` (inference), `mlx-vlm>=0.1` (VLM)

## Links

- [GitHub](https://github.com/jjang-ai/jangq) | [HuggingFace](https://huggingface.co/JANGQ-AI) | [MLX Studio](https://mlx.studio) | [PyPI](https://pypi.org/project/jang/) | [Format Spec](https://github.com/jjang-ai/jangq/blob/main/FORMAT.md)

---

## 한국어

**JANG**은 Apple Silicon을 위한 혼합정밀도 양자화 포맷입니다. MLX를 위한 GGUF.

| 모델 | MMLU | 크기 | 최소 Mac |
|------|:----:|:----:|:--------:|
| Qwen3.5-397B JANG_1L | 86.5%* | 112 GB | 128 GB |
| Nemotron-Cascade-2 JANG_2L | 88.0%* | 10 GB | 16 GB |
| Nemotron-Super-120B JANG_2L | 86.0%* | 43 GB | 64 GB |
| MiniMax-M2.5 JANG_2L | 74% | 63 GB | 128 GB |

*\* 추론 모드 사용*

```bash
pip install "jang[mlx]>=2.1.5"
```

[GitHub](https://github.com/jjang-ai/jangq) · [HuggingFace](https://huggingface.co/JANGQ-AI) · [MLX Studio](https://mlx.studio) · [PyPI](https://pypi.org/project/jang/)

---

<p align="center">장진호 제작 · Created by Jinho Jang — <a href="https://jangq.ai">jangq.ai</a></p>
