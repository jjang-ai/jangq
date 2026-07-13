# User Guide

*(Screenshots to be captured from the first signed build.)*

## Modes

JANG Studio has two workflows:

| Mode | When | Sidebar steps |
|---|---|---|
| **Convert** (default) | Any model; skip expert pruning | Source → Profile → Build / Convert → Verify |
| **Expert Lab** | MoE only; analyze & hard-prune experts first | Source → Expert Review → Prune Review → Profile → Build / Convert → Verify |

Dense models never show Expert Lab steps. After a reviewed BF16/F16 prune is adopted, the wizard returns to **Convert** for final quantization.

---

## Convert path

### 1 — Source model

Click **Choose Folder...** and pick a HuggingFace model directory (one containing `config.json` and `.safetensors` shards). JANG Studio auto-detects:

- `model_type` (e.g., `qwen3_5_moe`, `minimax_m2`, `llama`)
- Dense vs MoE (expert count when MoE)
- Source dtype (BF16 / FP16 / FP8)
- Image-VL / video-VL preprocessor presence
- Total disk size + shard count

On MoE sources, a **Convert | Expert Lab** segmented control appears. Leave **Convert** selected for direct quantize.

### 2 — Conversion profile

- **JANG** — every architecture. Pick by bit tier (1/2/3/4/5/6-bit).
- **JANGTQ** — enabled only when the architecture is on the whitelist (`qwen3_5_moe`, `qwen3_5_moe_text`, `minimax_m2` in v1).

**Advanced overrides** (JANG family): force dtype and force block size when auto-detection needs a nudge.

Pre-flight runs live as you change options. You cannot start conversion until required checks are green. JANGTQ also requires a Studio converter module mapping (whitelist alone is not enough).

### 3 — Build / Convert

- Macro + fine progress bars and live log stream
- **Cancel** sends SIGTERM, then SIGKILL after 3s
- On failure: **Retry** or **Copy Diagnostics**

### 4 — Verify

Post-convert checklist (schema, tokenizer, shards, capabilities, optional Expert Lab same-suite evidence for reviewed prune flows). All required rows must pass before **Finish**.

---

## Expert Lab path (MoE)

1. On Source, select **Expert Lab**, then **Analyze Experts Before Pruning**.
2. **Expert Review** — run prompt suites through BF16/vMLX, inspect the expert atlas, mask/compare, export a prune plan.
3. **Prune Review** — hard-prune a new BF16/F16 source (original source stays immutable).
4. After adopt, mode returns to **Convert** for Profile → Run → Verify with the pruned source.

**Router-Only Prune** remains available under Convert as a non-reviewed escape hatch; it cannot unlock final quantization gates that require same-suite Expert Lab evidence.

---

## Tips

- Most beginners: pick folder → accept recommendation → Start Conversion.
- RAM ≈ 1.5× source size at peak; free disk ≈ estimated output × 1.1.
- Diagnostics zip lands on Desktop (plan.json, logs, verify results).
