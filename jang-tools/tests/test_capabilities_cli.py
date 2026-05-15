"""Tests for jang_tools.capabilities_cli."""
import json
import subprocess
import sys

from jang_tools.capabilities_cli import capabilities


def test_capabilities_shape():
    c = capabilities()
    assert "jangtq_whitelist" in c
    assert "jangtq_families" in c
    assert "qwen3_5_moe" in c["jangtq_whitelist"]
    assert "minimax_m2" in c["jangtq_whitelist"]
    assert "hy_v3" in c["jangtq_whitelist"]
    assert "deepseek_v4" in c["jangtq_whitelist"]
    assert "zaya1_vl" in c["jangtq_whitelist"]
    assert "kimi_k25" in c["jangtq_whitelist"]
    assert c["default_method"] == "mse"
    assert c["default_block_size"] == 64
    assert len(c["methods"]) >= 3


def test_supported_dtypes_include_fp8():
    c = capabilities()
    aliases = [d["alias"] for d in c["supported_source_dtypes"]]
    assert "bf16" in aliases
    assert "fp16" in aliases
    assert "fp8" in aliases


def test_cli_json():
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "capabilities", "--json"],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(r.stdout)
    assert "jangtq_whitelist" in data
    assert 64 in data["block_sizes"]


def test_cli_human():
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "capabilities"],
        capture_output=True, text=True, check=True,
    )
    assert "qwen3_5_moe" in r.stdout
    assert "deepseek_v4" in r.stdout
    assert "bfloat16" in r.stdout
