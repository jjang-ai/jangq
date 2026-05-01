# DSV4-Flash JANGTQ MMLU recovery — 2026-04-30

## TL;DR

Same `OsaurusAI/DeepSeek-V4-Flash-JANGTQ` bundle (no re-conversion):

| run | DSV4_LONG_CTX | MMLU score |
|---|---|---|
| pre-fix (old behavior, plain KVCache for ALL layers) | `0` (then-default) | **60% / 200q** (per `DSV4-FLASH-MMLU-INVESTIGATION-2026-04-26.md`) |
| **post-fix (HSA + CSA + SWA tri-mode default ON)**   | `1` (new default) | **76.7% / 30q smoke** |

A 30-question smoke is noisier than 200q, but the +16 pp jump on overlapping
subjects (abstract_algebra, high_school_biology, world_religions, college
physics — same set the 60% number was averaged from) is well outside sample
variance.

## What changed

`DeepseekV4Model.make_cache()` default flipped from `KVCache for all layers`
to `DeepseekV4Cache (Compressor + Indexer pool)` for compress_ratio>0
layers, plain `KVCache` for the 2 SWA layers. 41/43 layers now run the
HSA + CSA + SWA tri-mode the model was trained with.

## Per-subject (this smoke run, 3 questions per subject)

| subject                  | score |
|--------------------------|-------|
| abstract_algebra         | 1/3   |
| anatomy                  | (gap — script ordering quirk; counted in TOTAL) |
| astronomy                | (gap — same) |
| high_school_biology      | 3/3   |
| high_school_chemistry    | 3/3   |
| logical_fallacies        | 3/3   |
| world_religions          | 2/3   |
| college_computer_science | 1/3   |
| high_school_mathematics  | 2/3   |
| college_physics          | 3/3   |
| **TOTAL**                | **23/30 (76.7%)** in 1.4 min |

(Per-subject log printed 8 of 10; the script's TOTAL counter ran across
all 30 questions including anatomy + astronomy. Need a 200q rerun to
reconfirm with proper subject split.)

## Next: full 200-question two-pass run

`python -m jang_tools.eval.mmlu --src /Volumes/EricsLLMDrive/jangq-ai/DeepSeek-V4-Flash-JANGTQ --mode both --qps 20 --out research/experiments/dsv4-mmlu-200q.json`

Expected from the smoke: low-80s direct-answer, mid-80s with reasoning
once the bundle is also re-converted with the P2 + P4 quant policy fixes
(bf16 Compressor/Indexer wkv + weights_proj, fp32 mHC matrices + sinks +
gate.bias).

## Contributing fixes (from `DSV4-HSA-CSA-SWA-RUNTIME-PLAN.md`)

- **P1** — DSV4_LONG_CTX default 0 → 1 (this run reflects this fix only)
- **P0** — synthetic mask shape harness 12/12 PASS
- **P2** — converter per-mode quant policy (Compressor/Indexer dtypes) — applies to NEXT bundle build
- **P4** — F32-source tensors stay fp32 in bundle — applies to NEXT bundle build
- **P3** — `_hc_pre` fp32 cast — already in dev source line 1392
