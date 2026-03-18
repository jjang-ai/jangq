#!/bin/bash
# MiniMax M2.5 reconversion with v2 format + group_size=128
# Run on Mac Studio after 397B conversion finishes
# Created by Jinho Jang (eric@jangq.ai)

set -e

SOURCE="/Volumes/EricsLLMDrive/JANGQ-Library/sources/MiniMax-M2.5"
OUTDIR="/Volumes/EricsLLMDrive/JANGQ-Library/testing"

cd /Users/eric/jang

echo "============================================"
echo "  MiniMax M2.5 JANG Reconversion"
echo "  group_size=128 (MANDATORY for 256 experts)"
echo "  v2 format (MLX-native, instant load)"
echo "============================================"
echo ""

# Verify source exists
if [ ! -f "$SOURCE/config.json" ]; then
    echo "ERROR: MiniMax source not found at $SOURCE"
    exit 1
fi

# JANG_2L: (8, 6, 2) — proven 74% MMLU, best quality 2-bit
echo "=== [1/3] Converting JANG_2L (8,6,2) group_size=128 ==="
python3 -c "
from jang_tools.convert import convert_model
result = convert_model(
    '$SOURCE',
    '$OUTDIR/MiniMax-M2.5-JANG_2L-v2',
    profile='JANG_2L',
    block_size=128,
    quantization_method='mse',
)
import json
print(json.dumps(result, indent=2))
"

# JANG_1L: (8, 8, 2) — max quality 2-bit
echo ""
echo "=== [2/3] Converting JANG_1L (8,8,2) group_size=128 ==="
python3 -c "
from jang_tools.convert import convert_model
result = convert_model(
    '$SOURCE',
    '$OUTDIR/MiniMax-M2.5-JANG_1L',
    profile='JANG_1L',
    block_size=128,
    quantization_method='mse',
)
import json
print(json.dumps(result, indent=2))
"

# JANG_4K: K-quant 4-bit budget-neutral
echo ""
echo "=== [3/3] Converting JANG_4K (budget-neutral 4-bit) group_size=128 ==="
python3 -c "
from jang_tools.convert import convert_model
result = convert_model(
    '$SOURCE',
    '$OUTDIR/MiniMax-M2.5-JANG_4K-v2',
    profile='JANG_4K',
    block_size=128,
    quantization_method='mse',
)
import json
print(json.dumps(result, indent=2))
"

echo ""
echo "============================================"
echo "  All MiniMax conversions complete!"
echo "  Copy tokenizer from reference:"
echo "  cp /path/to/mlx-community/MiniMax-M2.5-4bit/tokenizer* to each output"
echo "============================================"
