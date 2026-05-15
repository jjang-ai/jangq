"""List JANG Studio-relevant capabilities: JANGTQ arch whitelist, 512+ expert types,
supported dtypes, block sizes, quantization methods, tokenizer class blocklist.

This is the authoritative source that Swift apps should query instead of
hardcoding these lists.
"""
from __future__ import annotations
import json
from typing import Any

from .jangtq_matrix import family_matrix, supported_model_types


# Known architectures with 512+ experts. JANG forces bfloat16 for these to
# avoid float16 overflow (per feedback_bfloat16_fix memory).
_KNOWN_512_EXPERT_TYPES = ["minimax_m2", "glm_moe_dsa"]

# Tokenizer classes that break Osaurus / swift-transformers serving.
# Model conversion remaps "TokenizersBackend" → "GPT2Tokenizer" automatically.
_TOKENIZER_CLASS_BLOCKLIST = ["TokenizersBackend"]

# Block sizes offered in the UI. Smaller = finer quantization granularity but
# larger metadata overhead. 64 is the default for most architectures.
_BLOCK_SIZES = [32, 64, 128]

# Supported source dtypes. mlx_lm / transformers read these directly.
_SUPPORTED_SOURCE_DTYPES = [
    {"name": "bfloat16", "alias": "bf16", "description": "HuggingFace standard for modern LLMs"},
    {"name": "float16", "alias": "fp16", "description": "Older standard; overflow risk on 512+ expert models"},
    {"name": "float8_e4m3fn", "alias": "fp8", "description": "8-bit float (MiniMax, DeepSeek native format)"},
    {"name": "float8_e5m2", "alias": "fp8-e5m2", "description": "Alternative FP8 encoding (less common)"},
]

# Quantization methods recognized by convert_model.
_METHODS = [
    {"name": "mse", "description": "Minimum-square-error weight search (default, recommended)"},
    {"name": "rtn", "description": "Round-to-nearest (fastest, slightly lower quality)"},
    {"name": "mse-all", "description": "MSE across all layer classes (slower, marginal gain)"},
]


def capabilities() -> dict[str, Any]:
    return {
        "jangtq_whitelist": supported_model_types(),
        "jangtq_families": family_matrix(),
        "known_512_expert_types": _KNOWN_512_EXPERT_TYPES,
        "supported_source_dtypes": _SUPPORTED_SOURCE_DTYPES,
        "block_sizes": _BLOCK_SIZES,
        "default_block_size": 64,
        "methods": _METHODS,
        "default_method": "mse",
        "tokenizer_class_blocklist": _TOKENIZER_CLASS_BLOCKLIST,
        "hadamard_default_for_bit_tier": {"2": False, "3": True, "4": True, "5": True, "6": True},
    }


def cmd_capabilities(args) -> None:
    data = capabilities()
    if args.json:
        print(json.dumps(data, indent=None))
    else:
        print("JANGTQ whitelist:")
        for a in data["jangtq_whitelist"]:
            print(f"  {a}")
        print("\nJANGTQ family profiles:")
        for model_type, info in sorted(data["jangtq_families"].items()):
            print(f"  {model_type:<18} {', '.join(info['profiles'])}")
        print(f"\n512+ expert types: {', '.join(data['known_512_expert_types'])}")
        print(f"\nSupported source dtypes:")
        for d in data["supported_source_dtypes"]:
            print(f"  {d['name']:<18} ({d['alias']})  {d['description']}")
        print(f"\nBlock sizes: {data['block_sizes']} (default: {data['default_block_size']})")
        print(f"\nMethods:")
        for m in data["methods"]:
            print(f"  {m['name']:<10}  {m['description']}")


def register(subparsers) -> None:
    p = subparsers.add_parser("capabilities", help="List capability flags (JANGTQ whitelist, dtypes, methods)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_capabilities)
