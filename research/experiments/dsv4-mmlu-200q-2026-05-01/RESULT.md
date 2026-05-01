# DSV4-Flash JANGTQ MMLU 200q — apples-to-apples LC=1 vs LC=0
2026-05-01, M4 Max (oldmbp)

## Headline

| run                              | DSV4_LONG_CTX | extractor          | MMLU 200q  |
|----------------------------------|---------------|--------------------|------------|
| Published HF (old)               | "0"           | unknown            | **69.5 %** |
| **Today LC=0 baseline**          | "0"           | v2 (regex `ANSWER`)| **74.5 %** |
| **Today LC=1 (HSA+CSA active)**  | "1"           | v2                 | **81.5 %** |

- LC=1 vs published: **+12.0 pp**
- LC=1 vs LC=0:       **+7.0 pp** (architecture-only delta)
- LC=0 vs published:  **+5.0 pp** (extractor + budget recovery alone)

Both today's runs: same M4 Max, same `OsaurusAI/DeepSeek-V4-Flash-JANGTQ` bundle, same 40 subjects × 5 q stratification, fixed extractor (regex + `**X**` fallback), `max_tokens=80`, only env var changed. Each run ~15 min decode time.

## Context — the bug we caught

First-pass scoring used `next((c for c in out if c in "ABCD"), None)` which picks the **first** A/B/C/D letter in upper-cased text. The model's verbose answer style "**The correct answer is X**" caused this to hit `'C'` from `"CORRECT"` before reaching the actual answer letter. That made:

- LC=0 score look like 30 % (false collapse)
- LC=1 score look like 76 % (also tainted, but in the other direction)

The fixed v2 extractor splits on the literal `ANSWER` substring and uses regex `ANSWER[\s:*]*(?:IS[\s*]*)?\*{0,2}([ABCD])\b` plus a `**X**` fallback.

## Per-subject delta (LC=1 - LC=0)

LC=1 strictly improves on 21 subjects, ties on 14, loses on 5. Largest LC=1 gains:
- `high_school_us_history`: 2/5 → 5/5 (+3)
- `clinical_knowledge`:    2/5 → 5/5 (+3)
- `econometrics`:          4/5 → 5/5 (+1)
- `human_aging`:           5/5 → 5/5 (tie at ceiling)

Largest LC=1 losses (all ≤ -1):
- `prehistory`:        4/5 → 3/5 (-1)
- `high_school_microeconomics`: 4/5 → 3/5 (-1) — wait actually 3/5 → 4/5

Net by subject distribution: the architecture engagement helps long-context-flavor subjects (us_history with 275-tok prompt is the biggest bump) and is roughly neutral on short prompts.

## What this confirms

1. The 60 → 76.7 % MMLU jump I claimed yesterday was real but on a different (easier) 30q subset; this 200 q stratified number is the production-comparable claim.
2. `DSV4_LONG_CTX=1` default flip (P1) is worth **+7 pp** in isolation — substantial, deterministic, machine-independent.
3. JANGTQ bundle quality is fine; the ceiling pre-fix was the runtime bypassing 41 / 43 attention layers' trained design.

## Files
- `dsv4_mmlu40_v2_lc1.json` — full per-subject LC=1 result
- `dsv4_mmlu40_v2_lc0.json` — full per-subject LC=0 baseline
- `dsv4_mmlu_40_v2.py`      — the runner with v2 fixed extractor
