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
import ast
from pathlib import Path

from jang_tools.capabilities import (
    stamp_directory,
    validate_capabilities_block,
    validate_jang_runtime_contract,
    verify_directory,
    write_validated_jang_config,
)


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


def test_authoritative_capabilities_reject_empty_family_even_when_unresolvable():
    caps = {
        "reasoning_parser": "qwen3",
        "tool_parser": "qwen",
        "think_in_template": True,
        "supports_tools": True,
        "supports_thinking": True,
        "family": "",
        "modality": "text",
        "cache_type": "kv",
    }

    ok, msg = validate_capabilities_block(caps)

    assert ok is False
    assert "family" in msg
    assert "non-empty" in msg


def test_authoritative_capabilities_reject_non_object_block():
    ok, msg = validate_capabilities_block(["qwen3"])

    assert ok is False
    assert "JSON object" in msg


def test_runtime_contract_preserves_legacy_but_rejects_familyless_chat_schema():
    assert validate_jang_runtime_contract({"format": "JANG"})[0] is True

    ok, msg = validate_jang_runtime_contract({"chat": {}})

    assert ok is False
    assert "model_family" in msg


def test_validated_writer_leaves_existing_bytes_untouched_on_rejection(tmp_path):
    d = _make_dir(
        tmp_path,
        jang={"format": "JANG", "preserved": True},
        config={"model_type": "qwen3"},
    )
    original = (d / "jang_config.json").read_bytes()

    try:
        write_validated_jang_config(
            d,
            {"capabilities": {"family": ""}},
            {"model_type": "qwen3"},
        )
    except ValueError as exc:
        assert "family" in str(exc)
    else:
        raise AssertionError("invalid authoritative stamp was written")

    assert (d / "jang_config.json").read_bytes() == original


def test_every_literal_authoritative_capabilities_block_names_a_family():
    """Catch the exact Raptor regression before a stamper can run.

    Dynamic blocks are checked by ``verify_directory``. Literal blocks are
    statically knowable and must never use ``capabilities`` as a friendly
    boolean summary without the runtime's required ``family`` key.
    """
    package = Path(__file__).resolve().parents[1] / "jang_tools"
    missing: list[str] = []
    for source_path in package.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), source_path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if not (
                    isinstance(key, ast.Constant)
                    and key.value == "capabilities"
                    and isinstance(value, ast.Dict)
                ):
                    continue
                keys = {
                    child.value
                    for child in value.keys
                    if isinstance(child, ast.Constant)
                    and isinstance(child.value, str)
                }
                if "family" not in keys:
                    missing.append(f"{source_path.relative_to(package)}:{node.lineno}")

    assert missing == [], f"literal capabilities blocks without family: {missing}"


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
