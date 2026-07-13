from __future__ import annotations

import json

import pytest

from jang_tools.capabilities import build_capabilities, verify_directory


@pytest.mark.parametrize(
    ("arch", "config", "expected"),
    [
        (
            "zaya1_vl",
            {"model_type": "zaya1_vl"},
            {
                "family": "zaya1_vl",
                "modality": "vision",
                "reasoning_parser": None,
                "tool_parser": "zaya_xml",
                "supports_thinking": False,
                "cache_type": "hybrid",
            },
        ),
        (
            "laguna",
            {"model_type": "laguna"},
            {
                "family": "laguna",
                "reasoning_parser": "qwen3",
                "tool_parser": "qwen",
                "supports_thinking": True,
                "cache_type": "kv",
            },
        ),
        (
            "qwen3_moe",
            {"model_type": "qwen3_moe"},
            {
                "family": "qwen3_moe",
                "reasoning_parser": "qwen3",
                "tool_parser": "qwen",
                "supports_thinking": True,
                "cache_type": "kv",
            },
        ),
        (
            "qwen3_vl",
            {"model_type": "qwen3_vl"},
            {
                "family": "qwen3_vl",
                "modality": "vision",
                "reasoning_parser": "qwen3",
                "tool_parser": "qwen",
                "cache_type": "kv",
            },
        ),
        (
            "qwen2_vl",
            {"model_type": "qwen2_vl"},
            {
                "family": "qwen2_vl",
                "modality": "vision",
                "reasoning_parser": None,
                "tool_parser": "qwen",
                "supports_thinking": False,
                "cache_type": "kv",
            },
        ),
        (
            "qwen_mamba",
            {"model_type": "qwen_mamba"},
            {
                "family": "qwen_mamba",
                "reasoning_parser": None,
                "tool_parser": "qwen",
                "supports_thinking": False,
                "cache_type": "mamba",
            },
        ),
        (
            "mimo_v2",
            {"model_type": "mimo_v2"},
            {
                "family": "mimo_v2",
                "reasoning_parser": "think_xml",
                "tool_parser": "xml_function",
                "supports_thinking": False,
                "cache_type": "kv",
                "has_vision": False,
                "has_audio": False,
                "has_video": False,
            },
        ),
        (
            "minimax_m2",
            {"model_type": "minimax_m2"},
            {
                "family": "minimax_m2",
                "reasoning_parser": "minimax_m2",
                "tool_parser": "minimax",
                "supports_thinking": True,
                "cache_type": "kv",
            },
        ),
        (
            "minimax_m3_vl",
            {"model_type": "minimax_m3_vl"},
            {
                "family": "minimax_m3",
                "modality": "vision",
                "reasoning_parser": None,
                "tool_parser": "minimax_m3",
                "supports_thinking": True,
                "cache_type": "kv",
            },
        ),
        (
            "kimi_k25",
            {"model_type": "kimi_k25"},
            {
                "family": "kimi_k25",
                "reasoning_parser": "deepseek_r1",
                "tool_parser": "kimi",
                "supports_thinking": True,
                "cache_type": "kv",
            },
        ),
        (
            "step3p7",
            {"model_type": "step3p7"},
            {
                "family": "step3p7",
                "modality": "vision",
                "reasoning_parser": "qwen3",
                "tool_parser": "step3p5",
                "supports_thinking": True,
                "cache_type": "kv",
            },
        ),
        (
            "lfm2",
            {"model_type": "lfm2"},
            {
                "family": "lfm2",
                "reasoning_parser": "qwen3",
                "tool_parser": "lfm2",
                "supports_thinking": True,
                "cache_type": "hybrid",
            },
        ),
        (
            "gpt_oss",
            {"model_type": "gpt_oss"},
            {
                "family": "gpt_oss",
                "reasoning_parser": "openai_gptoss",
                "tool_parser": "glm47",
                "supports_thinking": True,
                "cache_type": "kv",
            },
        ),
        (
            "glm4_moe",
            {"model_type": "glm4_moe"},
            {
                "family": "glm4_moe",
                "reasoning_parser": "openai_gptoss",
                "tool_parser": "glm47",
                "supports_thinking": True,
                "cache_type": "kv",
            },
        ),
        (
            "gemma3",
            {"model_type": "gemma3"},
            {
                "family": "gemma3",
                "reasoning_parser": None,
                "tool_parser": "gemma3",
                "supports_thinking": False,
                "cache_type": "kv",
            },
        ),
        (
            "bailing_hybrid",
            {"model_type": "bailing_hybrid"},
            {
                "family": "bailing_hybrid",
                "reasoning_parser": None,
                "tool_parser": "deepseek",
                "supports_thinking": False,
                "cache_type": "hybrid",
            },
        ),
        (
            "deepseek_v4",
            {"model_type": "deepseek_v4"},
            {
                "family": "deepseek_v4",
                "reasoning_parser": "deepseek_r1",
                "tool_parser": "dsml",
                "supports_thinking": True,
                "cache_type": "kv",
            },
        ),
    ],
)
def test_capability_builder_covers_current_vmlx_runtime_families(
    arch: str,
    config: dict,
    expected: dict,
):
    caps = build_capabilities({"source_model": {"architecture": arch}}, config)

    assert caps is not None
    for key, value in expected.items():
        assert caps[key] == value


@pytest.mark.parametrize("arch", [
    "zaya1_vl",
    "mimo_v2",
    "minimax_m3_vl",
    "lfm2",
    "gemma3",
    "gpt_oss",
])
def test_verify_directory_accepts_vmlx_parser_names(tmp_path, arch: str):
    model_dir = tmp_path / arch
    model_dir.mkdir()
    config = {"model_type": arch}
    caps = build_capabilities({"source_model": {"architecture": arch}}, config)
    assert caps is not None

    (model_dir / "config.json").write_text(json.dumps(config))
    (model_dir / "jang_config.json").write_text(json.dumps({
        "source_model": {"architecture": arch},
        "capabilities": caps,
    }))

    ok, msg = verify_directory(model_dir)
    assert ok, msg
