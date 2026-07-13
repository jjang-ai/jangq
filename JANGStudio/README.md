# JANG Studio

Native macOS wizard that converts HuggingFace models (BF16 / FP16 / FP8) to JANG and JANGTQ formats. Built on top of the `jang-tools` Python pipeline — same quantization, zero setup.

## Install

Download the latest `JANGStudio.dmg` from [Releases](https://github.com/jjang-ai/jangq/releases?q=jang-studio), drag `JANG Studio.app` to `/Applications`. **macOS 15+, Apple Silicon.**

## What it does

Two product modes (sidebar filters to match):

### Convert (default — 4 steps)

1. **Source Model** — pick a HuggingFace folder (BF16 / FP16 / FP8); architecture is detected inline
2. **Conversion Profile** — JANG (all arches) or JANGTQ (Qwen 3.6 & MiniMax in v1); Advanced overrides (force dtype / block size) live here
3. **Build / Convert** — live logs, phase progress, cancel if you need to
4. **Verify** — post-convert checklist proves `jang_config.json`, tokenizer, chat template, shards, etc.

### Expert Lab (MoE only — 6 steps)

Source → Expert Review → Prune Review → Profile → Build / Convert → Verify. On MoE sources, choose **Expert Lab** on Source to trace routing, mask/compare experts, hard-prune BF16/F16, then quantize the reviewed pruned source.

See [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) for step-by-step walkthrough.

## System requirements

- macOS 15 (Sequoia) or later
- Apple Silicon (M1 or later)
- RAM >= 1.5x source model size (conversion peaks are high)
- Free disk >= output model size * 1.1

## Profiles cheat sheet

| Bit tier | JANG profiles | JANGTQ profiles |
|---:|:---|:---|
| 1-bit | JANG_1L | — |
| 2-bit | JANG_2S, JANG_2M, JANG_2L | JANGTQ2 |
| 3-bit | JANG_3K, JANG_3S, JANG_3M, JANG_3L | JANGTQ3 |
| 4-bit | JANG_4K (default), JANG_4S, JANG_4M, JANG_4L | JANGTQ4 |
| 5/6-bit | JANG_5K, JANG_6K, JANG_6M | — |

JANG works on every architecture. JANGTQ v1 supports `qwen3_5_moe` (Qwen 3.6) and `minimax_m2` only; GLM is coming in v1.1.

## Docs

- [User Guide](docs/USER_GUIDE.md) — wizard walkthrough
- [Design Direction](docs/DESIGN_DIRECTION.md) — canonical UI style reference for the compact pro-noir Expert Lab direction
- [Troubleshooting](docs/TROUBLESHOOTING.md) — common errors
- [Contributing](docs/CONTRIBUTING.md) — dev setup
- [Progress Protocol](docs/PROGRESS_PROTOCOL.md) — JSONL schema (for replacing the GUI with other frontends)

## Creator

Created by Jinho Jang (`eric@jangq.ai`) · [jangq.ai](https://jangq.ai)
