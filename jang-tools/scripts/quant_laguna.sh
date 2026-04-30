#!/usr/bin/env bash
# Convert poolside/Laguna-XS.2 (bf16 source) to JANGTQ2 + MXFP4.
# Prereqs: source bundle at $SRC, mlx + safetensors + tqdm in active venv.
set -euo pipefail

SRC="${SRC:-$HOME/.mlxstudio/models/_sources/Laguna-XS.2}"
OUT_BASE="${OUT_BASE:-$HOME/.mlxstudio/models}"
PY="${PY:-$HOME/jang/.venv/bin/python}"

mkdir -p "$OUT_BASE/JANGQ-AI" "$OUT_BASE/OsaurusAI"

echo "== Laguna-XS.2 → JANGTQ2 =="
"$PY" -m jang_tools.convert_laguna_jangtq \
    "$SRC" "$OUT_BASE/JANGQ-AI/Laguna-XS.2-JANGTQ2" JANGTQ2

echo "== Laguna-XS.2 → MXFP4 =="
"$PY" -m jang_tools.convert_laguna_mxfp4 \
    "$SRC" "$OUT_BASE/OsaurusAI/Laguna-XS.2-mxfp4"

echo "DONE. Bundles in $OUT_BASE"
