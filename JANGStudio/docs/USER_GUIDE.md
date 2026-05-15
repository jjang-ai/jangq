# User Guide

*(Screenshots to be captured from the first signed build.)*

## Step 1 — Source model

Click **Choose Folder...** and pick a HuggingFace model directory (one containing `config.json` and `.safetensors` shards). JANG Studio auto-detects:
- `model_type` (e.g., `qwen3_5_moe`, `minimax_m2`, `deepseek_v4`, `kimi_k25`, `zaya1_vl`, `llama`)
- Dense vs MoE (expert count when MoE)
- Source dtype (BF16 / FP16 / FP8)
- Image-VL (preprocessor_config.json) and video-VL (video_preprocessor_config.json)
- Total disk size + shard count

## Step 2 — Architecture

Confirm the summary. Use **Advanced overrides** only when auto-detection gets something wrong (rare — usually only when a brand-new architecture ships before `jang-tools` knows about it).

- **Force dtype** — override bf16/fp16 selection (use when the detected dtype is wrong).
- **Force block size** — override the automatic 32/64/128 pick for quantization groups.

## Step 3 — Profile

Two tabs:
- **JANG** — every architecture supported. Pick by bit tier (1/2/3/4/5/6-bit) and sensitivity letter (S/M/L/K).
- **JANGTQ** — enabled only when the detected `model_type` has a family entry in `jangtq_families`. The profile menu is filtered per family, so unsupported combinations such as ZAYA `JANGTQ3` are not selectable.

Pre-flight panel runs live checks as you change options:
source readable · config.json parses · output dir valid · disk space · RAM adequate · JANGTQ arch supported · JANGTQ profile supported · JANGTQ dtype (BF16/FP8) · bf16 forced for 512+ expert models · hadamard-vs-2bit sanity · bundled Python healthy.

You can't click **Start Conversion** until all required checks are green.

## Step 4 — Run

- Top: macro progress bar (`[N/5] phase`).
- Middle: fine progress bar with the current tensor name.
- Bottom: live log stream (monospace, auto-scroll, copyable).
- Right rail: elapsed time + peak RAM.
- **Cancel** button sends SIGTERM, waits 3s, then SIGKILL if still alive. Partial output stays on disk — the verifier will flag it as incomplete.
- On failure: **Retry** or **Copy Diagnostics** (saves a zip to `~/Desktop/` with plan.json + run.log + events.jsonl + system.json + verify.json for filing a bug).

## Step 5 — Verify

12-row checklist auto-runs after a successful conversion:
1. `jang_config.json` exists + JSON-valid
2. `format == "jang"` + `format_version >= 2.0`
3. `jang validate` passes (schema)
4. `capabilities` stamp present
5. Chat template present (inline / `.jinja` / `.json` — any one satisfies)
6. Tokenizer files — `tokenizer.json` or `tokenizer.model` + `tokenizer_config.json` + `special_tokens_map.json`
7. Shards match `model.safetensors.index.json`
8. `preprocessor_config.json` if VL
8b. `video_preprocessor_config.json` if video-VL
9. `modeling_*.py` + `configuration_*.py` if MiniMax-class
10. Tokenizer class concrete (not `TokenizersBackend` — warn only)
11. `generation_config.json` (warn only — HF falls back to defaults)
12. `num_hidden_layers > 0` in `config.json`

Any required-row red blocks **Finish**. Use **Copy Diagnostics** to file a bug with a full repro bundle. All green → **Reveal in Finder**, **Copy Path**, **Convert another**, or **Finish**.
