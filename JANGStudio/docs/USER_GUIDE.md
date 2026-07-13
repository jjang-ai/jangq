# User Guide

*(Screenshots to be captured from the first signed build.)*

## Modes

JANG Studio has these workflows:

| Mode | When | Sidebar steps |
|---|---|---|
| **Convert** (default) | Any model; skip expert pruning | Source → Profile → Build / Convert → Verify |
| **Intent Prune** (Qwen MoE) | Shape experts by capability + safety stance, then Convert | Sheet from Source → adopt → Profile → Build / Convert → Verify |
| **Expert Lab** (Advanced) | MoE power users; atlas / mask / compare | Source → Expert Review → Prune Review → Profile → Build / Convert → Verify |

Dense models never show Expert Lab steps. After Intent Prune or a reviewed BF16/F16 prune is adopted, the wizard returns to **Convert** for final quantization.

---

## Convert path

### 1 — Source model

Click **Choose Folder...** and pick a HuggingFace model directory (one containing `config.json` and `.safetensors` shards). JANG Studio auto-detects:

- `model_type` (e.g., `qwen3_5_moe`, `minimax_m2`, `llama`)
- Dense vs MoE (expert count when MoE)
- Source dtype (BF16 / FP16 / FP8)
- Image-VL / video-VL preprocessor presence
- Total disk size + shard count

On MoE sources, a **Convert | Expert Lab** segmented control appears. Leave **Convert** selected for direct quantize or Intent Prune.

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

## Intent Prune (Qwen MoE, preferred prune path)

For raw BF16/F16 `qwen3_5_moe` / `qwen3_5_moe_text` sources, Source shows **Shape model (Intent Prune)** as the primary prune CTA.

Workflow (design-b quality loop): **Evidence → Shape → Prune → Quality → Convert**.

1. **Evidence** — attach `expert_transitions.jsonl` from a **real-domain** BF16/vMLX trace (preferred over marker-only suites). Generate via Advanced Expert Lab if needed.
2. **Shape** — main selection surface:
   - **Keep** tiles (green): Coding, Math, Writing, Science/Bio, Multilingual, Tools/Agentic, Long context, Generalist, English-dominant, Reasoning
   - **Drop** tiles (red): Chinese, multilingual, translation, Spanish, creative, knowledge, tools, long context; Safety-heavy switches to CRACK stance
   - Quick presets (Coding+Math, Drop Chinese, EN agent, …)
   - **Safety stance**: Keep (default) / Balanced / **CRACK** (confirm required; `-CRACK` folder suffix)
   - **Size budget**: Light / Standard / Aggressive → keep-K
3. **Prune** — **Preview scores** (`intent-prune-score` with `--intent` + `--drop-intent`) then **Run Hard Prune**. Structural verify only.
4. **Quality** — explicit holdout checklist before Convert (math/code/language; CRACK anchors if CRACK). Acknowledge to unlock Convert.
5. **Convert pruned model** adopts the pruned folder and continues Profile → Run → Verify.

Unsupported MoE architectures show Direct Convert plus “Intent Prune coming for this architecture.”

---

## Expert Lab path (MoE, Advanced)

Expert Lab is under **Advanced** on Source (or the Expert Lab segment) for atlas / mask power users.

1. On Source, open **Advanced Expert Lab** (or select **Expert Lab** → **Analyze Experts Before Pruning**).
2. **Expert Review** — run prompt suites through BF16/vMLX, inspect the expert atlas, mask/compare, export a prune plan.
3. **Prune Review** — hard-prune a new BF16/F16 source (original source stays immutable).
4. After adopt, mode returns to **Convert** for Profile → Run → Verify with the pruned source.

**Router-Only Prune** remains available under Advanced as a non-reviewed escape hatch; it cannot unlock final quantization gates that require same-suite Expert Lab evidence.

---

## CRACK (abliteration)

**CRACK** is JANG’s product label for abliteration / reduced-refusal behavior (same brand as shipped `*-CRACK` models). In Intent Prune it is a safety stance that down-ranks experts specialized on safety/refusal paths while protecting your keep-intents. It is never the default; Keep is selected when Intent Prune opens.

---

## Tips

- Most beginners: pick folder → accept recommendation → Start Conversion.
- MoE shaping: Intent Prune with Coding + Standard budget, then Convert.
- RAM ≈ 1.5× source size at peak; free disk ≈ estimated output × 1.1.
- Diagnostics zip lands on Desktop (plan.json, logs, verify results).
