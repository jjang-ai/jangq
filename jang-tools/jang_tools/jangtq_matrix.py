"""Authoritative JANGTQ model-family/profile compatibility matrix."""
from __future__ import annotations

from copy import deepcopy
from typing import Any


JANGTQ_PROFILE_CATALOG: list[dict[str, Any]] = [
    {
        "name": "JANGTQ1",
        "bits": 1,
        "avg_bits": 1.0,
        "min_source_dtype": ["bfloat16"],
        "description": "1-bit routed-expert TurboQuant; experimental and family-gated",
    },
    {
        "name": "JANGTQ2",
        "bits": 2,
        "avg_bits": 2.0,
        "min_source_dtype": ["bfloat16", "float8_e4m3fn"],
        "description": "2-bit TurboQuant routed experts",
    },
    {
        "name": "JANGTQ3",
        "bits": 3,
        "avg_bits": 3.0,
        "min_source_dtype": ["bfloat16", "float8_e4m3fn"],
        "description": "3-bit TurboQuant routed experts; only exposed where packing is proven",
    },
    {
        "name": "JANGTQ4",
        "bits": 4,
        "avg_bits": 4.0,
        "min_source_dtype": ["bfloat16", "float8_e4m3fn"],
        "description": "4-bit TurboQuant routed experts",
    },
    {
        "name": "JANGTQ_K",
        "bits": 3,
        "avg_bits": 2.67,
        "min_source_dtype": ["bfloat16", "float8_e4m3fn"],
        "description": "Mixed routed experts: gate/up 2-bit, down 4-bit",
    },
    {
        "name": "JANGTQ_1L",
        "bits": 2,
        "avg_bits": 2.6,
        "min_source_dtype": ["bfloat16"],
        "description": "Kimi legacy 1L policy: 2-bit routed experts plus 8-bit non-routed",
    },
    {
        "name": "JANGTQ_2L",
        "bits": 2,
        "avg_bits": 3.0,
        "min_source_dtype": ["bfloat16"],
        "description": "Kimi legacy 2L policy: 2-bit routed experts plus passthrough non-routed",
    },
    {
        "name": "JANGTQ_3L",
        "bits": 3,
        "avg_bits": 4.0,
        "min_source_dtype": ["bfloat16"],
        "description": "Kimi legacy 3L policy: 3-bit routed experts plus passthrough non-routed",
    },
]


def _family(
    converter: str,
    profiles: list[str],
    default_profile: str,
    *,
    invocation: str = "positional_progress",
    status: str = "supported",
    note: str = "",
) -> dict[str, Any]:
    return {
        "converter": converter,
        "profiles": profiles,
        "default_profile": default_profile,
        "invocation": invocation,
        "status": status,
        "note": note,
    }


JANGTQ_FAMILIES: dict[str, dict[str, Any]] = {
    "qwen3_5_moe": _family(
        "jang_tools.convert_qwen35_jangtq",
        ["JANGTQ2", "JANGTQ3", "JANGTQ4", "JANGTQ_K"],
        "JANGTQ2",
        note="Qwen3.5/3.6 MoE hybrid converter",
    ),
    "minimax_m2": _family(
        "jang_tools.convert_minimax_jangtq",
        ["JANGTQ2", "JANGTQ3", "JANGTQ4", "JANGTQ_K"],
        "JANGTQ2",
        note="MiniMax M2.x FP8/BF16 converter; K is the 4/2/2 routed profile",
    ),
    "minimax_m2_5": _family(
        "jang_tools.convert_minimax_jangtq",
        ["JANGTQ2", "JANGTQ3", "JANGTQ4", "JANGTQ_K"],
        "JANGTQ2",
        note="MiniMax M2.x alias",
    ),
    "minimax": _family(
        "jang_tools.convert_minimax_jangtq",
        ["JANGTQ2", "JANGTQ3", "JANGTQ4", "JANGTQ_K"],
        "JANGTQ2",
        note="MiniMax M2.x alias",
    ),
    "hy_v3": _family(
        "jang_tools.convert_hy3_jangtq",
        ["JANGTQ2", "JANGTQ_K", "JANGTQ4", "JANGTQ1"],
        "JANGTQ2",
        note="JANGTQ1 is mechanically supported but not a publishable quality profile",
    ),
    "deepseek_v4": _family(
        "jang_tools.dsv4.convert_dsv4_jangtq",
        ["JANGTQ2", "JANGTQ3", "JANGTQ4", "JANGTQ_K"],
        "JANGTQ2",
        invocation="dsv4_flags",
        note="DSV4 uses --src/--dst/--profile plus variant; JANGTQ2 maps to V3 by default",
    ),
    "kimi_k25": _family(
        "jang_tools.kimi_prune.convert_kimi_jangtq",
        ["JANGTQ_K", "JANGTQ_1L", "JANGTQ_2L", "JANGTQ_3L"],
        "JANGTQ_K",
        invocation="src_dst_profile_flags",
        note="Kimi K2.6/KimiK25 wrapper; K keeps VL files and uses 4/2/2 routed bits",
    ),
    "zaya": _family(
        "jang_tools.convert_zaya_jangtq",
        ["JANGTQ2", "JANGTQ4", "JANGTQ_K"],
        "JANGTQ_K",
        note="JANGTQ3 intentionally excluded for ZAYA packing/runtime constraints",
    ),
    "zaya1_vl": _family(
        "jang_tools.convert_zaya1_vl_jangtq",
        ["JANGTQ2", "JANGTQ4", "JANGTQ_K"],
        "JANGTQ_K",
        note="Vision sidecars preserved; JANGTQ3 intentionally excluded",
    ),
    "bailing_hybrid": _family(
        "jang_tools.convert_ling_jangtq",
        ["JANGTQ2", "JANGTQ4"],
        "JANGTQ2",
        note="Ling/Bailing JANGTQ3 is excluded until the packing path is independently proven",
    ),
    "bailing_moe_v2_5": _family(
        "jang_tools.convert_ling_jangtq",
        ["JANGTQ2", "JANGTQ4"],
        "JANGTQ2",
        note="Ling/Bailing alias",
    ),
    "nemotron_h": _family(
        "jang_tools.convert_nemotron_jangtq",
        ["JANGTQ2", "JANGTQ3", "JANGTQ4"],
        "JANGTQ2",
        note="Text-only Nemotron-H/Omni converter",
    ),
    "nemotron_h_v2": _family(
        "jang_tools.convert_nemotron_jangtq",
        ["JANGTQ2", "JANGTQ3", "JANGTQ4"],
        "JANGTQ2",
        note="Nemotron-H alias",
    ),
    "laguna": _family(
        "jang_tools.convert_laguna_jangtq",
        ["JANGTQ2", "JANGTQ3", "JANGTQ4"],
        "JANGTQ2",
        note="Laguna XS.2 routed experts",
    ),
    "mistral3": _family(
        "jang_tools.convert_mistral3_jangtq",
        ["JANGTQ2", "JANGTQ3", "JANGTQ4"],
        "JANGTQ2",
        note="Mistral3/Pixtral VL sidecars preserved",
    ),
    "mistral4": _family(
        "jang_tools.convert_mistral3_jangtq",
        ["JANGTQ2", "JANGTQ3", "JANGTQ4"],
        "JANGTQ2",
        note="Mistral4 alias using the Mistral3 converter",
    ),
}


def profile_catalog() -> list[dict[str, Any]]:
    return deepcopy(JANGTQ_PROFILE_CATALOG)


def family_matrix() -> dict[str, dict[str, Any]]:
    return deepcopy(JANGTQ_FAMILIES)


def supported_model_types() -> list[str]:
    return sorted(JANGTQ_FAMILIES)


def profiles_for_model(model_type: str) -> list[str]:
    return list(JANGTQ_FAMILIES.get(model_type, {}).get("profiles", []))
