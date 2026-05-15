"""List all available JANG + JANGTQ profiles with their metadata.

JSON output shape:
  {
    "jang": [
      {"name": "JANG_4K", "critical_bits": 8, "important_bits": 4, "compress_bits": 4,
       "avg_bits": 4.23, "use": "default — K-quant 4-bit", "is_default": true},
      ...
    ],
    "jangtq": [
      {"name": "JANGTQ4", "bits": 4, "min_source_dtype": ["bfloat16", "float8_e4m3fn"]},
      ...
    ],
    "default_profile": "JANG_4K",
    "bit_to_profile": {"1": "JANG_1L", "2": "JANG_2S", "3": "JANG_3K", "4": "JANG_4K", ...}
  }
"""
from __future__ import annotations
import json
import sys
from typing import Any

from .allocate import JANG_PROFILES, JANG_K_TARGETS, BIT_TO_PROFILE
from .jangtq_matrix import family_matrix, profile_catalog


# Human-readable descriptions per profile. Kept inline because it pairs with
# allocate.py's authoritative list — if a profile is added there, add its
# description here in the same PR.
_PROFILE_DESCRIPTIONS = {
    "JANG_1L": "Maximum-protection 1-bit tier — 2-bit compress with 8-bit critical + important",
    "JANG_2S": "Tightest 2-bit — 6-bit critical, 4-bit important",
    "JANG_2M": "Balanced 2-bit — 8-bit critical, 4-bit important",
    "JANG_2L": "Best-quality 2-bit — 8-bit critical, 6-bit important (proven 73% MMLU on 122B)",
    "JANG_3S": "Small boost on attention only — 6-bit critical, 3-bit rest",
    "JANG_3M": "Full attention at 8-bit, everything else 3-bit",
    "JANG_3L": "Attention 8-bit, embeddings 4-bit",
    "JANG_3K": "K-quant 3-bit (budget-neutral, same size as MLX, smarter allocation)",
    "JANG_4S": "Small boost — 6-bit critical, 4-bit rest",
    "JANG_4M": "Full attention at 8-bit, rest at 4-bit (~2% overhead on MoE)",
    "JANG_4L": "Attention 8-bit, embeddings 6-bit (for dense models)",
    "JANG_4K": "K-quant 4-bit (budget-neutral) — THE DEFAULT",
    "JANG_5K": "K-quant 5-bit (budget-neutral)",
    "JANG_6K": "K-quant 6-bit (budget-neutral)",
    "JANG_6M": "Near-lossless — 8-bit critical + everything else 6-bit",
}

def list_profiles() -> dict[str, Any]:
    jang = []
    # JANG_PROFILES holds tier-based profiles (11 entries).
    for name, (crit, imp, comp) in JANG_PROFILES.items():
        avg = round((crit + imp + comp * 2) / 4, 2)  # rough avg; real val depends on arch
        jang.append({
            "name": name,
            "critical_bits": crit,
            "important_bits": imp,
            "compress_bits": comp,
            "avg_bits": avg,
            "description": _PROFILE_DESCRIPTIONS.get(name, ""),
            "is_default": False,
            "is_kquant": False,
        })
    # K-quant profiles (JANG_3K/4K/5K/6K) live only in JANG_K_TARGETS.
    # They don't have explicit tier tuples — avg_bits IS the target.
    for name, avg in JANG_K_TARGETS.items():
        jang.append({
            "name": name,
            "critical_bits": None,
            "important_bits": None,
            "compress_bits": None,
            "avg_bits": avg,
            "description": _PROFILE_DESCRIPTIONS.get(name, ""),
            "is_default": name == "JANG_4K",
            "is_kquant": True,
        })
    return {
        "jang": jang,
        "jangtq": profile_catalog(),
        "jangtq_families": family_matrix(),
        "default_profile": "JANG_4K",
        "bit_to_profile": {str(k): v for k, v in BIT_TO_PROFILE.items()},
    }


def cmd_profiles(args) -> None:
    data = list_profiles()
    if args.json:
        print(json.dumps(data, indent=None))
    else:
        print(f"JANG profiles ({len(data['jang'])}):")
        for p in data["jang"]:
            default_tag = " [DEFAULT]" if p["is_default"] else ""
            if p["is_kquant"]:
                tier_str = "k-quant"
            else:
                tier_str = f"{p['critical_bits']}/{p['important_bits']}/{p['compress_bits']}"
            print(f"  {p['name']:<10} {tier_str:<12}"
                  f"  avg={p['avg_bits']:.2f}{default_tag}  {p['description']}")
        print(f"\nJANGTQ profiles ({len(data['jangtq'])}):")
        for p in data["jangtq"]:
            print(f"  {p['name']:<10} {p['bits']}-bit  {p['description']}")
        print(f"\nJANGTQ families ({len(data['jangtq_families'])}):")
        for model_type, info in sorted(data["jangtq_families"].items()):
            profiles = ", ".join(info["profiles"])
            print(f"  {model_type:<18} {info['converter']}  [{profiles}]")


def register(subparsers) -> None:
    p = subparsers.add_parser("profiles", help="List available JANG + JANGTQ profiles with metadata")
    p.add_argument("--json", action="store_true", help="Emit JSON")
    p.set_defaults(func=cmd_profiles)
