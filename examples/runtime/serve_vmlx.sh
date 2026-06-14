#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${JANG_MODEL:-}" ]]; then
  echo "ERROR: set JANG_MODEL=/path/to/model-JANG-or-JANGTQ" >&2
  exit 2
fi

HOST="${JANG_HOST:-127.0.0.1}"
PORT="${JANG_PORT:-8000}"
MODEL_ALIAS="${JANG_MODEL_ALIAS:-default}"
MAX_TOKENS="${JANG_MAX_TOKENS:-32768}"
STREAM_INTERVAL="${JANG_STREAM_INTERVAL:-1}"
KV_CACHE_QUANTIZATION="${JANG_KV_CACHE_QUANTIZATION:-auto}"
KV_CACHE_GROUP_SIZE="${JANG_KV_CACHE_GROUP_SIZE:-64}"
MAX_CACHE_BLOCKS="${JANG_MAX_CACHE_BLOCKS:-4096}"
BLOCK_DISK_CACHE_MAX_GB="${JANG_BLOCK_DISK_CACHE_MAX_GB:-0}"
TOOL_PARSER="${JANG_TOOL_PARSER:-auto}"
REASONING_PARSER="${JANG_REASONING_PARSER:-}"
API_KEY="${JANG_API_KEY:-}"

cmd=(
  vmlx-engine serve "$JANG_MODEL"
  --host "$HOST"
  --port "$PORT"
  --served-model-name "$MODEL_ALIAS"
  --max-tokens "$MAX_TOKENS"
  --stream-interval "$STREAM_INTERVAL"
  --continuous-batching
  --use-paged-cache
  --max-cache-blocks "$MAX_CACHE_BLOCKS"
  --kv-cache-group-size "$KV_CACHE_GROUP_SIZE"
  --enable-auto-tool-choice
  --tool-call-parser "$TOOL_PARSER"
)

if [[ "$KV_CACHE_QUANTIZATION" != "auto" ]]; then
  cmd+=(--kv-cache-quantization "$KV_CACHE_QUANTIZATION")
fi

if [[ "$BLOCK_DISK_CACHE_MAX_GB" != "0" ]]; then
  cmd+=(--enable-block-disk-cache --block-disk-cache-max-gb "$BLOCK_DISK_CACHE_MAX_GB")
fi

if [[ -n "$REASONING_PARSER" ]]; then
  cmd+=(--reasoning-parser "$REASONING_PARSER")
fi

if [[ -n "$API_KEY" ]]; then
  cmd+=(--api-key "$API_KEY")
fi

echo "Starting vmlx-engine at http://$HOST:$PORT/v1"
echo "Model: $JANG_MODEL"
exec "${cmd[@]}"
