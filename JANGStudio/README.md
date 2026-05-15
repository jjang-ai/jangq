# JANG Studio

Native macOS wizard that converts HuggingFace models (BF16 / FP16 / FP8) to JANG and JANGTQ formats. Built on top of the `jang-tools` Python pipeline — same quantization, zero setup.

## Install

Download the latest `JANGStudio.dmg` from [Releases](https://github.com/jjang-ai/jangq/releases?q=jang-studio), drag `JANG Studio.app` to `/Applications`. **macOS 15+, Apple Silicon.**

## What it does

5-step wizard:

1. **Pick your model folder** (BF16, FP16, or FP8 HuggingFace directory)
2. **Confirm the detected architecture** (dense vs MoE, MLA vs full attention, image-VL vs video-VL)
3. **Choose a profile** — JANG (all architectures) or JANGTQ (shown only for families with a real converter)
4. **Run** — live logs, phase progress, cancel if you need to
5. **Verify** — 12-row post-convert checklist proves `jang_config.json`, tokenizer, chat template (inline / `.jinja` / `.json`), shards match the index, `generation_config.json`, and `num_hidden_layers > 0` all landed before you can finish

See [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) for step-by-step screenshots.

## System requirements

- macOS 15 (Sequoia) or later
- Apple Silicon (M1 or later)
- RAM >= 1.5x source model size (conversion peaks are high)
- Free disk >= output model size * 1.1

## Profiles cheat sheet

| Bit tier | JANG profiles | JANGTQ profiles |
|---:|:---|:---|
| 1-bit | JANG_1L | JANGTQ1 (family-gated experimental) |
| 2-bit | JANG_2S, JANG_2M, JANG_2L | JANGTQ2 |
| 3-bit | JANG_3K, JANG_3S, JANG_3M, JANG_3L | JANGTQ3 |
| 4-bit | JANG_4K (default), JANG_4S, JANG_4M, JANG_4L | JANGTQ4 |
| mixed | — | JANGTQ_K (4/2/2 routed experts) |
| 5/6-bit | JANG_5K, JANG_6K, JANG_6M | — |

JANG works on every architecture. JANGTQ is enabled per `jangtq_families` from
`python -m jang_tools profiles --json`; GLM stays disabled until its runtime
path is proven.

## Docs

- [User Guide](docs/USER_GUIDE.md) — wizard walkthrough
- [Troubleshooting](docs/TROUBLESHOOTING.md) — common errors
- [Contributing](docs/CONTRIBUTING.md) — dev setup
- [Progress Protocol](docs/PROGRESS_PROTOCOL.md) — JSONL schema (for replacing the GUI with other frontends)

## Creator

Created by Jinho Jang (`eric@jangq.ai`) · [jangq.ai](https://jangq.ai)
