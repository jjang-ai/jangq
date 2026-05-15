"""JANGTQ model-family compatibility matrix tests."""

from pathlib import Path

from jang_tools.inspect_source import _JANGTQ_V1_WHITELIST
from jang_tools.jangtq_matrix import JANGTQ_FAMILIES, supported_model_types
from jang_tools.kimi_prune.convert_kimi_jangtq import normalize_profile


def test_inspect_source_whitelist_comes_from_matrix():
    assert _JANGTQ_V1_WHITELIST == set(supported_model_types())
    assert "minimax_m2" in _JANGTQ_V1_WHITELIST
    assert "deepseek_v4" in _JANGTQ_V1_WHITELIST
    assert "kimi_k25" in _JANGTQ_V1_WHITELIST


def test_matrix_converter_modules_exist_for_exposed_families():
    root = Path(__file__).resolve().parents[1] / "jang_tools"
    for model_type, info in JANGTQ_FAMILIES.items():
        module = info["converter"]
        assert module.startswith("jang_tools.")
        rel = module.removeprefix("jang_tools.").replace(".", "/") + ".py"
        assert (root / rel).exists(), f"{model_type} exposes missing converter {module}"


def test_kimi_accepts_canonical_jangtq_spellings():
    assert normalize_profile("JANGTQ_1L") == "1L"
    assert normalize_profile("JANGTQ1") == "1L"
    assert normalize_profile("JANGTQ2") == "2L"
    assert normalize_profile("JANGTQ3") == "3L"
    assert normalize_profile("JANGTQ_K") == "K"
