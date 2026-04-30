#!/usr/bin/env bash
# Convert mistralai/Mistral-Medium-3.5-128B (FP8 per-tensor source) to
# JANGTQ2 + MXFP4. Vision tower + multi_modal_projector + lm_head stay bf16.
set -euo pipefail

SRC="${SRC:-$HOME/.mlxstudio/models/_sources/Mistral-Medium-3.5-128B}"
OUT_BASE="${OUT_BASE:-$HOME/.mlxstudio/models}"
PY="${PY:-$HOME/jang/.venv/bin/python}"

mkdir -p "$OUT_BASE/JANGQ-AI" "$OUT_BASE/OsaurusAI"

echo "== Mistral-3.5-128B → JANGTQ2 =="
"$PY" -m jang_tools.convert_mistral3_jangtq \
    "$SRC" "$OUT_BASE/JANGQ-AI/Mistral-Medium-3.5-128B-JANGTQ2" JANGTQ2

echo "== Mistral-3.5-128B → MXFP4 =="
"$PY" -m jang_tools.convert_mistral3_mxfp4 \
    "$SRC" "$OUT_BASE/OsaurusAI/Mistral-Medium-3.5-128B-mxfp4"

echo "DONE. Bundles in $OUT_BASE"
