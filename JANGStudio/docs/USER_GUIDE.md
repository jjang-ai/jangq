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

1. Multi-select **capability chips** (Coding, Math, Writing, Science/Bio, Multilingual, Tools/Agentic, Long context, Generalist).
2. Choose **safety stance**:
   - **Keep** (default) — protect safety-path experts
   - **Balanced** — mild protection
   - **CRACK** — abliteration / reduced refusal-path specialization; requires an explicit confirmation checkbox; output folders use a `-CRACK` suffix
3. Pick a **size budget** (Light / Standard / Aggressive) → uniform keep-K experts per layer.
4. Attach `expert_transitions.jsonl` from a **Reviewed Prune 50** BF16/vMLX run (or generate it via Advanced Expert Lab).
5. **Preview scores** runs `intent-prune-score`; **Run Intent Prune** scores then hard-prunes a new BF16/F16 tree.
6. **Convert pruned model** adopts the pruned folder, sets workflow to Convert, and continues Profile → Run → Verify.

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
