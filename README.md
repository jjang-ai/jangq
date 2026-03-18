<p align="center">
  <a href="https://mlx.studio"><img src="https://raw.githubusercontent.com/jjang-ai/jangq/main/assets/mlx-studio-light.png" alt="MLX Studio" width="500"></a>
</p>

<p align="center">
  <a href="https://mlx.studio"><img src="https://mlx.studio/assets/screenshots/mlx-studio-featured.png?v=1" alt="MLX Studio App" width="600"></a>
</p>

<h4 align="center"><a href="https://mlx.studio">MLX Studio</a> — the only app that natively supports JANG models</h4>

---

> **Early Adoption:** LM Studio, Ollama, oMLX, Inferencer do **not** support JANG yet. Use **[MLX Studio](https://mlx.studio)** or `pip install "jang[mlx]"`. Ask your favorite app's creators to add JANG support!

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

## Results (200-question MMLU)

### MoE at 4-bit: JANG_4K beats MLX

| Model | JANG_4K | MLX 4-bit | JANG Size | MLX Size |
|-------|---------|-----------|-----------|----------|
| Qwen3.5-122B | **86%** | 85% | 69 GB | 64 GB |
| Qwen3.5-35B | **77.5%** | 75.5% | **16.7 GB** | 18 GB |

### MoE at 2-bit: JANG dominates

| Model | JANG_2S | MLX 2-bit | JANG Size | MLX Size |
|-------|---------|-----------|-----------|----------|
| Qwen3.5-122B | **79%** | 56.5% | 38 GB | 36 GB |
| Qwen3.5-35B | **65.5%** | ~20% | 12 GB | 10 GB |

### MiniMax: JANG is the ONLY working option

| Model | JANG_2L | MLX 4-bit | MLX 3-bit | MLX 2-bit |
|-------|---------|-----------|-----------|-----------|
| MiniMax-M2.5 | **74%** | 26.5% | 24.5% | 25% |

MLX is broken on MiniMax at ALL bit levels (~25% = random). JANG scores 74%.

### Dense/Hybrid at 2-bit: JANG saves what MLX destroys

| Model | JANG_2S | MLX 2-bit | JANG Size | MLX Size |
|-------|---------|-----------|-----------|----------|
| Qwen3.5-4B | **28.5%** | 12.5% | 1.5 GB | 1.2 GB |
| Qwen3.5-9B | **25.5%** | 22.0% | 3.4 GB | 2.7 GB |

At 3-bit and 4-bit, MLX uniform is better on dense models — JANG's value is at 2-bit (where uniform fails) and on MoE (where attention is < 5% of params).

## Install

```bash
pip install jang
```

For inference on Apple Silicon:
```bash
pip install "jang[mlx]"
```

For Vision-Language models:
```bash
pip install "jang[vlm]"
```

## Quick Start

### Convert any model

```bash
# K-quant 4-bit (same size as MLX, smarter allocation)
jang convert Qwen/Qwen3.5-35B-A3B -p 4

# 2-bit for extreme compression
jang convert Qwen/Qwen3.5-122B-A10B -p 2

# Specific profile
jang convert model -p JANG_2S
```

### Run inference

```python
from jang_tools.loader import load_jang_model
from mlx_lm.sample_utils import make_sampler
from mlx_lm.generate import generate_step
import mlx.core as mx

model, tokenizer = load_jang_model("JANGQ-AI/Qwen3.5-122B-A10B-JANG_2S")
sampler = make_sampler(temp=0.7)

tokens = tokenizer.encode("What is photosynthesis?")
for tok, _ in generate_step(prompt=mx.array(tokens), model=model, max_tokens=200, sampler=sampler):
    t = tok.item() if hasattr(tok, 'item') else int(tok)
    print(tokenizer.decode([t]), end="", flush=True)
    if t == tokenizer.eos_token_id:
        break
```

### Upgrade v1 models to v2 (instant loading)

```bash
jang upgrade /path/to/model
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `jang convert <model> -p <profile>` | Convert HuggingFace model to JANG |
| `jang upgrade <model>` | Upgrade v1 model to v2 (instant load) |
| `jang inspect <model>` | Show bit allocation and model info |
| `jang validate <model>` | Validate a JANG model directory |
| `jang estimate <params>` | Estimate sizes (e.g., `jang estimate 122B`) |

## v2 Format — Instant Loading

JANG v2 stores weights in MLX-native format. Like GGUF — the file IS the runtime format. No conversion at load time.

| | v2 (current) | v1 (legacy) |
|---|---|---|
| Load time | **Seconds** (mmap) | 5-10 minutes (repack) |
| File size | Same | Same |

New conversions automatically use v2. Existing v1 models can be upgraded with `jang upgrade`.

## Profiles

| Profile | Type | Bits | Best for |
|---------|------|------|----------|
| `JANG_4K` | K-quant | 4.0 | Same size as MLX 4-bit, smarter |
| `JANG_3K` | K-quant | 3.0 | Same size as MLX 3-bit, smarter |
| `JANG_2S` | Profile | ~2.1 | Tightest 2-bit, near MLX 2-bit size |
| `JANG_2L` | Profile | ~2.3 | Quality 2-bit |
| `JANG_1L` | Profile | ~2.2 | Maximum quality 2-bit |

## Pre-quantized Models

| Model | Profile | MMLU (200q) | Size | Best for |
|-------|---------|------------|------|----------|
| [Qwen3.5-122B-A10B](https://huggingface.co/JANGQ-AI/Qwen3.5-122B-A10B-JANG_4K) | JANG_4K | **86%** | 69 GB | 192+ GB Mac |
| [Qwen3.5-122B-A10B](https://huggingface.co/JANGQ-AI/Qwen3.5-122B-A10B-JANG_2S) | JANG_2S | **79%** | 38 GB | 64+ GB Mac |
| [Qwen3.5-35B-A3B](https://huggingface.co/JANGQ-AI/Qwen3.5-35B-A3B-JANG_4K) | JANG_4K | **77.5%** | 16.7 GB | 36+ GB Mac |
| [Qwen3.5-35B-A3B](https://huggingface.co/JANGQ-AI/Qwen3.5-35B-A3B-JANG_2S) | JANG_2S | **65.5%** | 12 GB | 24+ GB Mac |
| [MiniMax-M2.5](https://huggingface.co/JANGQ-AI/MiniMax-M2.5-JANG_2L) | JANG_2L | **74%** | 89 GB | 192+ GB Mac |
| Qwen3.5-9B | JANG_2S | **25.5%** | 3.4 GB | 8 GB MacBook |
| Qwen3.5-4B | JANG_2S | **28.5%** | 1.5 GB | 8 GB MacBook |

## Supported Architectures

Dense Transformer, Mixture of Experts, Hybrid SSM, Linear Attention (GatedDeltaNet), MLA (DeepSeek), Vision-Language, Mamba, FP8 source models (MiniMax, DeepSeek).

## How It Works

JANG redistributes bits based on tensor sensitivity — same total size, smarter allocation:

```
CRITICAL  (attention, MoE routers)   →  6-8 bit  →  Controls coherence
IMPORTANT (embeddings, linear attn)  →  4-6 bit  →  Moderate sensitivity
COMPRESS  (MLP, MoE experts)         →  2-4 bit  →  98% of parameters
```

K-quant profiles (JANG_4K, JANG_3K) redistribute within the same bit budget — boost attention, compensate with least-important MLP. Same size as MLX, smarter allocation. Like GGUF K-quants.

## Requirements

- **Python**: 3.11+
- **Conversion**: any platform (numpy + safetensors)
- **Inference**: Apple Silicon Mac (M1/M2/M3/M4) with MLX
- **Dependencies**: `safetensors>=0.4`, `numpy>=1.24`, `tqdm>=4.60`, `huggingface_hub>=0.20`
- **Optional**: `mlx>=0.22`, `mlx-lm>=0.20` (for inference), `mlx-vlm>=0.1` (for VLM)

## Links

- [GitHub](https://github.com/jjang-ai/jangq) | [HuggingFace](https://huggingface.co/JANGQ-AI) | [MLX Studio](https://mlx.studio) | [PyPI](https://pypi.org/project/jang/) | [Format Spec](https://github.com/jjang-ai/jangq/blob/main/FORMAT.md)

---

## 한국어

### JANG이란?

**JANG**은 Apple Silicon을 위한 오픈소스 혼합정밀도 양자화 포맷입니다. MLX를 위한 GGUF와 같은 역할을 합니다.

### 결과 (200문항 MMLU)

#### 4-bit: JANG_4K가 MLX 4-bit보다 우수 (MoE 모델)

| 모델 | JANG_4K | MLX 4-bit | 크기 |
|------|---------|-----------|------|
| Qwen3.5-122B | **86%** | 85% | 69 vs 64 GB |
| Qwen3.5-35B | **77.5%** | 75.5% | **16.7** vs 18 GB |

#### 2-bit: JANG이 MLX를 압도

| 모델 | JANG_2S | MLX 2-bit | 크기 |
|------|---------|-----------|------|
| Qwen3.5-122B | **79%** | 56.5% | 38 vs 36 GB |
| Qwen3.5-35B | **65.5%** | ~20% | 12 vs 10 GB |

#### MiniMax: JANG만 작동

| 모델 | JANG_2L | MLX 4-bit | MLX 3-bit | MLX 2-bit |
|------|---------|-----------|-----------|-----------
| MiniMax-M2.5 | **74%** | 26.5% | 24.5% | 25% |

### 설치

```bash
pip install "jang[mlx]"
```

### 호환성

현재 **[MLX Studio](https://mlx.studio)**만 JANG 포맷을 기본 지원합니다. LM Studio, Ollama, oMLX, Inferencer 등은 아직 지원하지 않습니다. 좋아하는 앱의 개발자에게 JANG 지원을 요청해 주세요!

[GitHub](https://github.com/jjang-ai/jangq) · [HuggingFace](https://huggingface.co/JANGQ-AI) · [MLX Studio](https://mlx.studio) · [PyPI](https://pypi.org/project/jang/)

---

<p align="center">장진호 제작 · Created by Jinho Jang — <a href="https://jangq.ai">jangq.ai</a></p>

<p align="center">
  <a href="https://ko-fi.com/jangml"><img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support on Ko-fi"></a>
</p>
