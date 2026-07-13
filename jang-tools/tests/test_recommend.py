"""Tests for jang_tools.recommend — covers many model families."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from jang_tools.recommend import detect, recommend, _classify_family, _estimate_params_billion


def _make_model_dir(tmp_path: Path, cfg: dict, name: str = "m", *, extra_files: list[str] | None = None) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(cfg))
    for f in extra_files or []:
        (d / f).write_text("{}")
    return d


# Classify family — unit-level checks across many arch types

@pytest.mark.parametrize("model_type,experts,is_vl,is_video,expected_class", [
    ("llama",         0,   False, False, "dense_llm"),
    ("mistral",       0,   False, False, "dense_llm"),
    ("qwen3",         0,   False, False, "dense_llm"),
    ("gemma3",        0,   False, False, "dense_llm"),
    ("phi3",          0,   False, False, "dense_llm"),
    ("falcon",        0,   False, False, "dense_llm"),
    ("qwen2_moe",     8,   False, False, "moe_standard"),
    ("mixtral",       8,   False, False, "moe_standard"),
    ("qwen3_5_moe",   256, False, False, "moe_hybrid_ssm"),
    ("deepseek_v32",  256, False, False, "moe_mla"),
    ("mistral4",      128, False, False, "moe_mla"),
    ("minimax_m2",    512, False, False, "moe_large_expert"),
    ("glm_moe_dsa",   256, False, False, "moe_large_expert"),   # known 512-class
    ("nemotron_h",    128, False, False, "hybrid_ssm_mtp"),
    ("qwen2_vl",      0,   True,  False, "vl_image"),
    ("idefics3",      0,   True,  False, "vl_image"),
    ("qwen3_vl",      0,   True,  True,  "vl_video"),
    ("something_new", 0,   False, False, "dense_llm"),           # fallback
])
def test_classify_family(model_type, experts, is_vl, is_video, expected_class):
    result = _classify_family(model_type, experts, is_vl, is_video)
    # For qwen2_moe with 512+ experts we upgrade to moe_large_expert
    if model_type in ("qwen2_moe", "qwen3_5_moe") and experts >= 512:
        assert result == "moe_large_expert"
    else:
        assert result == expected_class


# Estimate params

def test_estimate_params_dense():
    cfg = {"hidden_size": 4096, "num_hidden_layers": 32, "vocab_size": 32000, "intermediate_size": 11008}
    b = _estimate_params_billion(cfg)
    assert 5 < b < 10  # Llama-7B ballpark


def test_estimate_params_moe():
    cfg = {"hidden_size": 2048, "num_hidden_layers": 28, "vocab_size": 151936,
           "intermediate_size": 5120, "num_experts": 128}
    b = _estimate_params_billion(cfg)
    # 128-expert MoE at this size is somewhere in the 30-200B range
    assert 10 < b < 500


def test_estimate_params_empty_config():
    assert _estimate_params_billion({}) == 0.0


# Full recommend — family selection

def test_recommend_qwen35_moe_recommends_jang_default(tmp_path):
    d = _make_model_dir(tmp_path, {
        "model_type": "qwen3_5_moe", "hidden_size": 2048, "num_hidden_layers": 28,
        "num_experts": 256, "vocab_size": 151936,
    })
    rec = recommend(d)
    assert rec["recommended"]["family"] == "jang"
    assert rec["recommended"]["profile"] == "JANG_4K"
    # Should offer JANGTQ as alternative (since qwen3_5_moe is whitelisted)
    assert any(a.get("family") == "jangtq" for a in rec["recommended"]["alternatives"])


def test_recommend_reads_nested_qwen36_text_config(tmp_path):
    d = _make_model_dir(tmp_path, {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "vocab_size": 248320,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
        },
    })
    rec = recommend(d)
    assert rec["detected"]["is_moe"] is True
    assert rec["detected"]["expert_count"] == 256
    assert rec["detected"]["family_class"] == "moe_hybrid_ssm"
    assert rec["recommended"]["profile"] == "JANG_4K"
    assert any(a.get("family") == "jangtq" for a in rec["recommended"]["alternatives"])


def test_recommend_minimax_forces_bfloat16(tmp_path):
    d = _make_model_dir(tmp_path, {"model_type": "minimax_m2", "num_experts": 512,
                                    "hidden_size": 6144, "num_hidden_layers": 40,
                                    "vocab_size": 200064, "intermediate_size": 12288})
    rec = recommend(d)
    assert rec["recommended"]["force_dtype"] == "bfloat16"
    assert any("bfloat16" in w for w in rec["warnings"])


# M131 (iter 53): _recommend_dtype peer-helper sweep.
#
# `_classify_family` promotes ANY MoE model to "moe_large_expert" when
# expert_count >= 512. But `_recommend_dtype` used a HARDCODED set
# `_BF16_REQUIRED = {"minimax_m2", "glm_moe_dsa"}` — so a 512-expert
# qwen3_5_moe (or any future 512-expert family not on that list) got
# force_dtype=None while `warnings` said "bfloat16 is required to avoid
# float16 overflow". Self-contradicting output; user sees conflicting
# advice and the model then OOMs or NaNs at float16 boundaries. Iter 53
# fixes by passing expert_count through to _recommend_dtype and checking
# >= 512 dynamically alongside the named-family set.

def test_recommend_dtype_forces_bfloat16_on_any_512_expert_model(tmp_path):
    """A hypothetical 512+ expert qwen3_5_moe triggers the warning but
    pre-iter-53 did NOT force bfloat16. The warning and recommendation
    contradicted each other."""
    d = _make_model_dir(tmp_path, {"model_type": "qwen3_5_moe", "num_experts": 512,
                                    "hidden_size": 3072, "num_hidden_layers": 48,
                                    "vocab_size": 151936, "intermediate_size": 3072})
    rec = recommend(d)
    assert any("bfloat16 is required" in w for w in rec["warnings"]), \
        f"512+ expert model should warn about bfloat16, got {rec['warnings']}"
    assert rec["recommended"]["force_dtype"] == "bfloat16", (
        "When the warning says 'bfloat16 is required to avoid float16 overflow', "
        "force_dtype must be bfloat16 — otherwise the wizard shows conflicting "
        "advice and the user runs with f16 → NaN at runtime."
    )


def test_recommend_dtype_uses_n_routed_experts_for_bf16_check(tmp_path):
    """Some HF configs expose expert count as `n_routed_experts` (DeepSeek
    family). The dynamic check must handle that key too."""
    d = _make_model_dir(tmp_path, {"model_type": "deepseek_v3", "n_routed_experts": 512,
                                    "hidden_size": 4096, "num_hidden_layers": 60,
                                    "vocab_size": 102400, "intermediate_size": 4096})
    rec = recommend(d)
    assert rec["recommended"]["force_dtype"] == "bfloat16"


def test_recommend_dtype_below_512_stays_auto(tmp_path):
    """Regression guard: a 256-expert qwen3_5_moe must NOT force bfloat16.
    The <512 auto path is important — users with FP16 models pay speed cost
    if we force bfloat16 unnecessarily."""
    d = _make_model_dir(tmp_path, {"model_type": "qwen3_5_moe", "num_experts": 256,
                                    "hidden_size": 3072, "num_hidden_layers": 48,
                                    "vocab_size": 151936, "intermediate_size": 3072})
    rec = recommend(d)
    assert rec["recommended"]["force_dtype"] is None, (
        f"256-expert model should stay on auto dtype, got {rec['recommended']['force_dtype']}"
    )


def test_recommend_large_expert_uses_jang_2l(tmp_path):
    d = _make_model_dir(tmp_path, {"model_type": "minimax_m2", "num_experts": 512,
                                    "hidden_size": 6144, "num_hidden_layers": 40,
                                    "vocab_size": 200064})
    rec = recommend(d)
    assert rec["recommended"]["profile"] == "JANG_2L"


def test_recommend_llama_dense_recommends_jang_4k(tmp_path):
    d = _make_model_dir(tmp_path, {"model_type": "llama", "hidden_size": 4096,
                                    "num_hidden_layers": 32, "vocab_size": 32000})
    rec = recommend(d)
    assert rec["recommended"]["family"] == "jang"
    assert rec["recommended"]["profile"] == "JANG_4K"
    # Dense llama is not JANGTQ-whitelisted
    assert not any(a.get("family") == "jangtq" for a in rec["recommended"]["alternatives"])


def test_recommend_vl_model_flags_vl_class(tmp_path):
    d = _make_model_dir(tmp_path, {"model_type": "qwen2_vl", "hidden_size": 1536,
                                    "num_hidden_layers": 28, "vocab_size": 151936},
                         extra_files=["preprocessor_config.json"])
    rec = recommend(d)
    assert rec["detected"]["is_vl"] is True
    assert rec["detected"]["family_class"] == "vl_image"


def test_recommend_video_vl_flagged(tmp_path):
    d = _make_model_dir(tmp_path, {"model_type": "qwen3_vl", "hidden_size": 2048,
                                    "num_hidden_layers": 28, "vocab_size": 151936},
                         extra_files=["preprocessor_config.json", "video_preprocessor_config.json"])
    rec = recommend(d)
    assert rec["detected"]["is_vl"] is True
    assert rec["detected"]["is_video_vl"] is True
    assert rec["detected"]["family_class"] == "vl_video"


def test_recommend_gated_model_warns(tmp_path):
    d = _make_model_dir(tmp_path, {"model_type": "llama", "hidden_size": 3072,
                                    "num_hidden_layers": 28, "vocab_size": 128256,
                                    "_name_or_path": "meta-llama/Llama-3.2-1B-Instruct"})
    rec = recommend(d)
    assert rec["detected"]["is_gated_model"] is True
    assert any("gated" in w.lower() for w in rec["warnings"])


def test_recommend_hadamard_off_at_2bit(tmp_path):
    # Force the model into moe_large_expert which defaults to JANG_2L
    d = _make_model_dir(tmp_path, {"model_type": "minimax_m2", "num_experts": 512,
                                    "hidden_size": 6144, "num_hidden_layers": 40,
                                    "vocab_size": 200064})
    rec = recommend(d)
    assert rec["recommended"]["profile"] == "JANG_2L"
    assert rec["recommended"]["hadamard"] is False


def test_recommend_hadamard_on_at_4bit(tmp_path):
    d = _make_model_dir(tmp_path, {"model_type": "qwen3_5_moe", "num_experts": 256,
                                    "hidden_size": 2048, "num_hidden_layers": 28,
                                    "vocab_size": 151936})
    rec = recommend(d)
    assert rec["recommended"]["profile"] == "JANG_4K"
    assert rec["recommended"]["hadamard"] is True


# M142 (iter 64): _recommend_hadamard uses JANG_PROFILES compress-tier
# lookup instead of a hardcoded profile-name list. Same fix shape as the
# Swift-side PreflightRunner.hadamardVsLowBits + symmetric to iter-54 M132's
# JANGTQ converter profile parsing.

def test_recommend_hadamard_uses_JANG_PROFILES_compress_tier():
    """Every JANG_2* profile has compress=2 → hadamard should be off for ALL
    of them, not just the hardcoded set in the old list."""
    from jang_tools.recommend import _recommend_hadamard
    # Pre-M142 hardcoded list was ("JANG_1L", "JANG_2S", "JANG_2M", "JANG_2L", "JANGTQ2").
    # Post-M142: all 2-bit compress profiles ping off, including any future
    # additions to JANG_PROFILES with compress=2.
    for p in ["JANG_1L", "JANG_2S", "JANG_2M", "JANG_2L"]:
        use_hadamard, _ = _recommend_hadamard(p)
        assert use_hadamard is False, f"{p} has compress=2 — hadamard should be off"
    # 3-bit+ profiles: hadamard on.
    for p in ["JANG_3S", "JANG_3M", "JANG_3L", "JANG_4S", "JANG_4M", "JANG_4L", "JANG_6M"]:
        use_hadamard, _ = _recommend_hadamard(p)
        assert use_hadamard is True, f"{p} has compress>=3 — hadamard should be on"


def test_recommend_hadamard_handles_JANGTQ_variants():
    """JANGTQ2 off; JANGTQ3 / JANGTQ4 on. Post-M142 this path is driven
    by parsing the suffix digit, not a membership check against a hardcoded
    list — so JANGTQ_2L / JANGTQ_4M legacy aliases work too."""
    from jang_tools.recommend import _recommend_hadamard
    assert _recommend_hadamard("JANGTQ2")[0] is False
    assert _recommend_hadamard("JANGTQ3")[0] is True
    assert _recommend_hadamard("JANGTQ4")[0] is True


def test_recommend_hadamard_k_quant_profiles():
    """K-quant profiles (JANG_3K, JANG_4K, JANG_5K, JANG_6K) use their
    target avg_bits as the compress-tier proxy. JANG_3K should flip ON;
    anything 2-bit would flip off (no current 2-bit K-quant but defensive)."""
    from jang_tools.recommend import _recommend_hadamard
    assert _recommend_hadamard("JANG_3K")[0] is True, "3-bit K-quant → hadamard on"
    assert _recommend_hadamard("JANG_4K")[0] is True
    assert _recommend_hadamard("JANG_5K")[0] is True
    assert _recommend_hadamard("JANG_6K")[0] is True


def test_recommend_hadamard_unknown_profile_defaults_to_on():
    """Unknown profile falls through to the default hadamard=on path
    (compress_bits=4 fallback). Matches the Swift-side pass behavior."""
    from jang_tools.recommend import _recommend_hadamard
    use_hadamard, reason = _recommend_hadamard("JANG_UNKNOWN_99Z")
    assert use_hadamard is True
    assert "3-bit" in reason or "higher" in reason


def test_recommend_surfaces_plain_english(tmp_path):
    d = _make_model_dir(tmp_path, {"model_type": "qwen3_5_moe", "num_experts": 256,
                                    "hidden_size": 2048, "num_hidden_layers": 28,
                                    "vocab_size": 151936})
    rec = recommend(d)
    assert rec["beginner_summary"]
    assert isinstance(rec["why_each_choice"], dict)
    for field in ("family", "profile", "method", "hadamard", "block_size", "force_dtype"):
        assert field in rec["why_each_choice"]
        assert len(rec["why_each_choice"][field]) > 20  # real prose, not placeholder


def test_recommend_unknown_arch_falls_back_safely(tmp_path):
    d = _make_model_dir(tmp_path, {"model_type": "some_unknown_future_arch",
                                    "hidden_size": 4096, "num_hidden_layers": 32,
                                    "vocab_size": 50000})
    rec = recommend(d)
    assert rec["recommended"]["family"] == "jang"
    assert rec["recommended"]["profile"] == "JANG_4K"
    assert any("unknown model_type" in w.lower() for w in rec["warnings"])


# CLI smoke tests

def test_cli_rejects_missing_model(tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "recommend",
         "--model", str(tmp_path / "nope"), "--json"],
        capture_output=True, text=True, check=False,
    )
    assert r.returncode == 2
    assert "not found" in r.stderr.lower()


def _assert_clean_recommend_error(r: subprocess.CompletedProcess) -> None:
    """M120 companion: same invariant as test_inspect_source._assert_clean_error.
    A corrupt config.json should surface a short stderr message, not a full
    Python traceback — JANG Studio's SourceStep treats nonzero exit as a soft
    failure and falls back to no-recommendation, but still logs stderr."""
    assert r.returncode != 0
    assert "Traceback" not in r.stderr, (
        "recommend leaked a Python traceback:\n" + r.stderr
    )
    assert "config.json" in r.stderr.lower()


def test_cli_rejects_malformed_config(tmp_path):
    """M120: corrupt config.json must not crash recommend with JSONDecodeError."""
    (tmp_path / "config.json").write_text("{bogus}")
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "recommend",
         "--model", str(tmp_path), "--json"],
        capture_output=True, text=True, check=False,
    )
    _assert_clean_recommend_error(r)


def test_cli_rejects_non_dict_config(tmp_path):
    """M120: list-root config.json used to AttributeError deep inside detect()."""
    (tmp_path / "config.json").write_text("[]")
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "recommend",
         "--model", str(tmp_path), "--json"],
        capture_output=True, text=True, check=False,
    )
    _assert_clean_recommend_error(r)


def test_cli_json_output(tmp_path):
    d = _make_model_dir(tmp_path, {"model_type": "qwen3_5_moe", "num_experts": 256,
                                    "hidden_size": 2048, "num_hidden_layers": 28,
                                    "vocab_size": 151936})
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "recommend",
         "--model", str(d), "--json"],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(r.stdout)
    assert data["detected"]["model_type"] == "qwen3_5_moe"
    assert data["recommended"]["profile"] == "JANG_4K"
    assert data["beginner_summary"]


def test_cli_human_output(tmp_path):
    d = _make_model_dir(tmp_path, {"model_type": "llama", "hidden_size": 4096,
                                    "num_hidden_layers": 32, "vocab_size": 128000})
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "recommend", "--model", str(d)],
        capture_output=True, text=True, check=True,
    )
    assert "Recommended:" in r.stdout
    assert "JANG_4K" in r.stdout
