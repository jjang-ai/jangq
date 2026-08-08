"""LiquidAI LFM2.5 dense hybrid (LIV short-conv + GQA) JANG tooling.

Created by Jinho Jang (eric@jangq.ai) — 2026-08-04.

Modules:
  calibrate — AWQ activation stats + QAT input-sample capture from the
              bf16 source via the live mlx_lm lfm2 forward.
  qat_gptq  — fixed-grid GPTQ codes-only learned rounding for affine and
              MXFP8 storage (byte-compatible with mx.quantize).
  convert   — source -> MXFP8 / JANG_6M bundle converter with AWQ folds,
              QAT, and the full metadata contract.
"""
