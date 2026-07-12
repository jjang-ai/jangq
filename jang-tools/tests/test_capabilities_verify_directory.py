"""M150 (iter 72): capabilities.verify_directory / stamp_directory must
return (False, msg) for JSON parse errors instead of raising.

Pre-iter-72 a corrupt jang_config.json / config.json raised
JSONDecodeError mid-verify, breaking verify_capabilities's CLI harness
that expects (ok, msg) for EVERY failure mode. Iter-72 adds a local
_safe_load_json_dict helper that returns (None, msg) on failure and
wires it into both verify_directory and stamp_directory.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from jang_tools.capabilities import build_capabilities, stamp_directory, verify_directory
from jang_tools.convert import (
    _add_missing_minicpm_lm_head_alias,
    _append_alias_allocation_inputs,
    _normalize_minicpm_output_config,
    _quantized_output_tensor_name,
)
from jang_tools.architectures import detect_architecture


def _make_dir(tmp_path: Path, jang: dict | None, config: dict | None,
              raw_jang: str | None = None, raw_config: str | None = None) -> Path:
    d = tmp_path / "model"
    d.mkdir()
    if jang is not None:
        (d / "jang_config.json").write_text(json.dumps(jang))
    if config is not None:
        (d / "config.json").write_text(json.dumps(config))
    if raw_jang is not None:
        (d / "jang_config.json").write_text(raw_jang)
    if raw_config is not None:
        (d / "config.json").write_text(raw_config)
    return d


# ──────────── verify_directory ────────────


def test_verify_directory_malformed_jang_config_returns_false_with_path(tmp_path):
    d = _make_dir(tmp_path, jang=None, config=None, raw_jang="{ broken json")
    ok, msg = verify_directory(d)
    assert ok is False
    assert "not valid JSON" in msg
    assert "jang_config.json" in msg


def test_verify_directory_non_dict_jang_config_returns_false(tmp_path):
    d = _make_dir(tmp_path, jang=None, config=None, raw_jang="[1,2,3]")
    ok, msg = verify_directory(d)
    assert ok is False
    assert "expected a JSON object" in msg


def test_verify_directory_malformed_legacy_config_returns_false(tmp_path):
    # Legacy path: no jang_config.json, inline under config.json["jang"].
    # If config.json itself is malformed, must return (False, msg).
    d = tmp_path / "legacy"
    d.mkdir()
    (d / "config.json").write_text("not-json-at-all")
    ok, msg = verify_directory(d)
    assert ok is False
    assert "config.json" in msg
    assert "not valid JSON" in msg


def test_verify_directory_malformed_model_config_returns_false(tmp_path):
    # Valid jang_config.json but corrupt config.json.
    d = tmp_path / "model"
    d.mkdir()
    (d / "jang_config.json").write_text(json.dumps({
        "source_model": {"architecture": "qwen3"},
        "capabilities": {
            "reasoning_parser": "qwen3",
            "tool_parser": "qwen",
            "think_in_template": True,
            "supports_tools": True,
            "supports_thinking": True,
            "family": "qwen3",
            "modality": "text",
            "cache_type": "kv",
        },
    }))
    (d / "config.json").write_text("{bad")
    ok, msg = verify_directory(d)
    assert ok is False
    assert "config.json" in msg
    assert "not valid JSON" in msg


# ──────────── stamp_directory ────────────


def test_stamp_directory_malformed_jang_config_returns_false(tmp_path, capsys):
    d = _make_dir(tmp_path, jang=None, config=None, raw_jang="{ broken")
    result = stamp_directory(d, verbose=True)
    assert result is False
    captured = capsys.readouterr()
    # verbose=True means error should be printed.
    assert "SKIP" in captured.out
    assert "not valid JSON" in captured.out


def test_stamp_directory_malformed_config_json_returns_false(tmp_path, capsys):
    d = tmp_path / "model"
    d.mkdir()
    (d / "jang_config.json").write_text(json.dumps({
        "source_model": {"architecture": "qwen3"},
    }))
    (d / "config.json").write_text("not json")
    result = stamp_directory(d, verbose=True)
    assert result is False
    captured = capsys.readouterr()
    assert "SKIP" in captured.out
    assert "config.json" in captured.out


# ──────────── MiniCPM compatibility ────────────


def test_minicpm_capabilities_are_text_kv_without_optimistic_parsers():
    capabilities = build_capabilities({}, {"model_type": "minicpm"})

    assert capabilities == {
        "reasoning_parser": None,
        "tool_parser": None,
        "think_in_template": False,
        "supports_tools": False,
        "supports_thinking": False,
        "family": "minicpm",
        "modality": "text",
        "modalities": {
            "text": True,
            "vision": False,
            "audio": False,
            "video": False,
        },
        "has_vision": False,
        "has_audio": False,
        "has_video": False,
        "cache_type": "kv",
    }


def test_minicpm_capabilities_stamp_and_verify_round_trip(tmp_path):
    d = _make_dir(
        tmp_path,
        jang={"architecture": {"type": "minicpm"}},
        config={"model_type": "minicpm"},
    )

    assert stamp_directory(d, write=True, verbose=False) is True
    ok, message = verify_directory(d)

    assert ok is True
    assert message == "capabilities OK (family=minicpm)"
    stamped = json.loads((d / "jang_config.json").read_text())
    assert stamped["capabilities"]["tool_parser"] is None
    assert stamped["capabilities"]["supports_tools"] is False


def test_null_tool_parser_remains_invalid_for_other_families(tmp_path):
    capabilities = build_capabilities({}, {"model_type": "qwen3"})
    capabilities["tool_parser"] = None
    capabilities["supports_tools"] = False
    d = _make_dir(
        tmp_path,
        jang={
            "architecture": {"type": "qwen3"},
            "capabilities": capabilities,
        },
        config={"model_type": "qwen3"},
    )

    ok, message = verify_directory(d)

    assert ok is False
    assert message == "tool_parser=None is only valid for family='minicpm'"


def test_existing_tool_family_still_advertises_tool_support():
    capabilities = build_capabilities({}, {"model_type": "qwen3"})

    assert capabilities["tool_parser"] == "qwen"
    assert capabilities["supports_tools"] is True


def test_minicpm_family_detection_is_exact_and_excludes_vlm_wrappers():
    for model_type in (
        "minicpmv",
        "minicpm_v",
        "MiniCPMVForCausalLM",
        "notminicpmwrapper",
    ):
        assert build_capabilities({}, {"model_type": model_type}) is None


def test_minicpm_materializes_missing_lm_head_from_embedding():
    tensor_info = [
        ("model.embed_tokens.weight", (73448, 1024), 1_175_168, Path("model.safetensors"))
    ]
    source_names = {"model.embed_tokens.weight"}

    aliases = _add_missing_minicpm_lm_head_alias(
        "minicpm",
        tensor_info,
        source_names,
    )

    assert aliases == {"lm_head.weight": "model.embed_tokens.weight"}
    assert tensor_info[-1] == (
        "lm_head.weight",
        (73448, 1024),
        1_175_168,
        Path("model.safetensors"),
    )
    assert "lm_head.weight" in source_names
    assert _quantized_output_tensor_name("lm_head.weight", "minicpm") == "lm_head.weight"


def test_real_minicpm_head_uses_top_level_and_other_families_keep_sanitization():
    assert _quantized_output_tensor_name("lm_head.weight", "minicpm") == (
        "lm_head.weight"
    )
    assert _quantized_output_tensor_name("lm_head.weight", "llama") == (
        "language_model.lm_head.weight"
    )


def test_noncompact_allocator_accounts_for_virtual_lm_head():
    aliases = {"lm_head.weight": "model.embed_tokens.weight"}
    tensor_info = [
        ("model.embed_tokens.weight", (10, 4), 3, Path("model.safetensors")),
        ("lm_head.weight", (10, 4), 3, Path("model.safetensors")),
    ]
    importance = []
    names = []

    _append_alias_allocation_inputs(
        aliases,
        tensor_info,
        {},
        importance,
        names,
    )

    assert len(importance) == 1
    assert importance[0].tolist() == [0.5, 0.5, 0.5]
    assert names == ["lm_head.weight"] * 3


@pytest.mark.parametrize("source_tie", [None, False, True])
def test_minicpm_output_config_always_selects_independent_head(source_tie):
    config = {}
    if source_tie is not None:
        config["tie_word_embeddings"] = source_tie

    _normalize_minicpm_output_config("minicpm", config)

    assert config["model_type"] == "minicpm"
    assert config["rope_theta"] == 10000.0
    assert config["tie_word_embeddings"] is False


def test_unrelated_output_config_is_not_changed():
    config = {"tie_word_embeddings": True}

    _normalize_minicpm_output_config("llama", config)

    assert config["tie_word_embeddings"] is True


def test_lm_head_alias_is_narrow_and_does_not_replace_existing_head():
    existing = [
        ("model.embed_tokens.weight", (100, 64), 100, Path("model.safetensors")),
        ("lm_head.weight", (100, 64), 100, Path("model.safetensors")),
    ]
    existing_names = {item[0] for item in existing}

    assert _add_missing_minicpm_lm_head_alias(
        "minicpm", existing, existing_names
    ) == {}
    assert len(existing) == 2

    unrelated = [
        ("model.embed_tokens.weight", (100, 64), 100, Path("model.safetensors"))
    ]
    unrelated_names = {"model.embed_tokens.weight"}
    assert _add_missing_minicpm_lm_head_alias(
        "llama", unrelated, unrelated_names
    ) == {}
    assert len(unrelated) == 1


def test_explicit_untied_minicpm_without_head_is_rejected():
    tensor_info = [
        ("model.embed_tokens.weight", (100, 64), 100, Path("model.safetensors"))
    ]

    with pytest.raises(ValueError, match="tie_word_embeddings=false"):
        _add_missing_minicpm_lm_head_alias(
            "minicpm",
            tensor_info,
            {"model.embed_tokens.weight"},
            tie_word_embeddings=False,
        )


def test_minicpm_without_head_or_embedding_is_rejected():
    with pytest.raises(ValueError, match="neither lm_head.weight nor"):
        _add_missing_minicpm_lm_head_alias("minicpm", [], set())


def test_direct_architecture_detection_infers_exact_minicpm_source(tmp_path):
    config = {
        "architectures": ["MiniCPMForCausalLM"],
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "vocab_size": 73448,
        "intermediate_size": 4096,
        "scale_emb": 12,
        "scale_depth": 1.4,
        "dim_model_base": 256,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    assert detect_architecture(tmp_path).model_type == "minicpm"


@pytest.mark.parametrize(
    "config_extra,sidecar",
    [
        ({"vision_config": {"hidden_size": 1024}}, None),
        ({}, "preprocessor_config.json"),
        ({}, "video_preprocessor_config.json"),
    ],
)
def test_direct_architecture_detection_rejects_minicpm_modality_signals(
    tmp_path, config_extra, sidecar
):
    config = {
        "model_type": "minicpm",
        "architectures": ["MiniCPMForCausalLM"],
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "vocab_size": 73448,
        "intermediate_size": 4096,
        "scale_emb": 12,
        "scale_depth": 1.4,
        "dim_model_base": 256,
        **config_extra,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    if sidecar:
        (tmp_path / sidecar).write_text("{}")

    with pytest.raises(ValueError, match="text-only MiniCPM4-0.5B"):
        detect_architecture(tmp_path)


def test_minicpm_capabilities_reject_multimodal_signals(tmp_path):
    config = {"model_type": "minicpm", "vision_config": {}}
    assert build_capabilities({}, config) is None

    d = _make_dir(
        tmp_path,
        jang={
            "architecture": {"type": "minicpm"},
            "capabilities": {
                **build_capabilities({}, {"model_type": "minicpm"}),
                "modality": "vision",
                "modalities": {
                    "text": True,
                    "vision": True,
                    "audio": False,
                    "video": False,
                },
                "has_vision": True,
            },
        },
        config=config,
    )
    ok, message = verify_directory(d)
    assert ok is False
    assert message == "unsupported multimodal MiniCPM capability stamp"


@pytest.mark.parametrize(
    "sidecar",
    ["preprocessor_config.json", "video_preprocessor_config.json"],
)
def test_minicpm_capabilities_reject_modality_sidecars(tmp_path, sidecar):
    text_capabilities = build_capabilities({}, {"model_type": "minicpm"})
    d = _make_dir(
        tmp_path,
        jang={
            "architecture": {"type": "minicpm"},
            "capabilities": text_capabilities,
        },
        config={"model_type": "minicpm"},
    )
    (d / sidecar).write_text("{}")

    assert build_capabilities(
        {"architecture": {"type": "minicpm"}},
        {"model_type": "minicpm"},
        d,
    ) is None
    assert stamp_directory(d, write=True, verbose=False) is False
    ok, message = verify_directory(d)
    assert ok is False
    assert message == "unsupported multimodal MiniCPM capability stamp"
