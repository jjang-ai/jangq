"""Tests for jang_tools.examples — capability-aware snippet generation."""
import json
import subprocess
import sys
from pathlib import Path
import pytest

from jang_tools.examples import detect_capabilities, render_snippet, SUPPORTED_LANGS

FIXTURE_ROOT = Path(__file__).parent / "fixtures"


@pytest.fixture
def dense_model_dir(tmp_path):
    """A minimal valid JANG output for a dense chat model."""
    d = tmp_path / "dense"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "model_type": "qwen3",
        "num_hidden_layers": 28,
        "_name_or_path": "Qwen/Qwen3-0.6B-Base",
    }))
    (d / "jang_config.json").write_text(json.dumps({
        "format": "jang", "format_version": "2.0",
        "family": "jang", "profile": "JANG_4K",
        "capabilities": {"arch": "qwen3"},
        "quantization": {"actual_bits_per_weight": 4.23, "block_size": 64, "bit_widths_used": [3, 4, 6, 8]},
        "source_model": "Qwen/Qwen3-0.6B-Base",
    }))
    (d / "tokenizer_config.json").write_text(json.dumps({
        "chat_template": "{% for m in messages %}{{m.content}}{% endfor %}",
        "tokenizer_class": "Qwen2Tokenizer",
    }))
    return d


@pytest.fixture
def vl_model_dir(tmp_path):
    d = tmp_path / "vl"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "qwen2_vl", "num_hidden_layers": 28}))
    (d / "jang_config.json").write_text(json.dumps({
        "format": "jang", "family": "jang", "profile": "JANG_4K",
        "quantization": {"actual_bits_per_weight": 4.0, "block_size": 64},
    }))
    (d / "preprocessor_config.json").write_text("{}")
    (d / "tokenizer_config.json").write_text(json.dumps({"chat_template": "x"}))
    return d


def test_detect_dense_capabilities(dense_model_dir):
    caps = detect_capabilities(dense_model_dir)
    assert caps["model_type"] == "qwen3"
    assert caps["is_vl"] is False
    assert caps["is_video_vl"] is False
    assert caps["has_chat_template"] is True
    assert caps["profile"] == "JANG_4K"


def test_detect_vl_capabilities(vl_model_dir):
    caps = detect_capabilities(vl_model_dir)
    assert caps["is_vl"] is True
    assert caps["model_type"] == "qwen2_vl"


def test_render_python_dense(dense_model_dir):
    snippet = render_snippet(dense_model_dir, "python")
    # Should use dense loader
    # M45 (iter 20): symbol must be `load_jang_model`, not `load_model` —
    # the latter doesn't exist in jang_tools.loader and would ImportError
    # for every adopter copying this snippet. Tests previously pinned the
    # WRONG name, locking the bug in.
    assert "from jang_tools.loader import load_jang_model" in snippet
    assert "apply_chat_template" in snippet


def test_render_python_vl(vl_model_dir):
    snippet = render_snippet(vl_model_dir, "python")
    assert "load_jangtq_vlm" in snippet
    assert "image" in snippet.lower()


def test_render_swift_includes_path(dense_model_dir):
    snippet = render_snippet(dense_model_dir, "swift")
    assert "JANGCore" in snippet
    assert str(dense_model_dir.resolve()) in snippet


def test_render_server_has_curl(dense_model_dir):
    snippet = render_snippet(dense_model_dir, "server")
    assert "vmlx-engine serve" in snippet
    assert "curl" in snippet
    assert "127.0.0.1:8000" in snippet


def test_render_hf(dense_model_dir):
    snippet = render_snippet(dense_model_dir, "hf")
    assert "pip install" in snippet or "Python" in snippet


def test_render_rejects_unknown_lang(dense_model_dir):
    with pytest.raises(ValueError):
        render_snippet(dense_model_dir, "rust")


def test_cli_examples_json(dense_model_dir):
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "examples",
         "--model", str(dense_model_dir), "--lang", "python", "--json"],
        capture_output=True, text=True, check=False,
    )
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert data["lang"] == "python"
    # M45 pin: correct symbol is `load_jang_model`; `load_model` doesn't exist.
    assert "from jang_tools.loader import load_jang_model" in data["snippet"]
    assert "load_model(" not in data["snippet"], \
        "'load_model(' would ImportError for adopters — must be load_jang_model"


def test_cli_python_snippet_compiles(dense_model_dir):
    """Generated Python snippet must be syntactically valid."""
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "examples",
         "--model", str(dense_model_dir), "--lang", "python", "--json"],
        capture_output=True, text=True, check=True,
    )
    snippet = json.loads(r.stdout)["snippet"]
    # Don't EXEC — just compile. We can't actually import mlx in test env
    compile(snippet, "<eval>", "exec")  # raises SyntaxError if broken


# ────────────────────────────────────────────────────────────────────
# Iter 29: M93 — MiniMax is text-only regardless of stray preprocessor files
# ────────────────────────────────────────────────────────────────────

@pytest.fixture
def minimax_with_stray_preprocessor(tmp_path):
    """Simulates a bad state: a MiniMax output dir that somehow has a
    preprocessor_config.json (copy residue, user error, broken convert).
    The template must NOT emit VLM code per feedback_readme_standards.md
    rule 11: 'MiniMax is text-only — never include VLM code'.
    """
    d = tmp_path / "minimax"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "model_type": "minimax_m2",
        "num_hidden_layers": 80,
        "_name_or_path": "MiniMaxAI/MiniMax-M2",
    }))
    (d / "jang_config.json").write_text(json.dumps({
        "format": "jang", "family": "jang", "profile": "JANG_4K",
        "quantization": {"actual_bits_per_weight": 4.0, "block_size": 64},
    }))
    (d / "tokenizer_config.json").write_text(json.dumps({
        "chat_template": "x",
        "tokenizer_class": "Qwen2Tokenizer",
    }))
    # Plant the stray preprocessor file to simulate the failure mode.
    (d / "preprocessor_config.json").write_text("{}")
    (d / "video_preprocessor_config.json").write_text("{}")
    return d


def test_minimax_forced_text_only_even_with_stray_preprocessor(minimax_with_stray_preprocessor):
    """M93 regression: preprocessor files in a MiniMax dir must NOT flip is_vl."""
    caps = detect_capabilities(minimax_with_stray_preprocessor)
    assert caps["model_type"] == "minimax_m2"
    assert caps["is_vl"] is False, \
        "MiniMax must be text-only regardless of stray preprocessor_config.json"
    assert caps["is_video_vl"] is False, \
        "MiniMax must be text-only regardless of stray video_preprocessor_config.json"


def test_minimax_python_snippet_uses_text_loader_not_vlm(minimax_with_stray_preprocessor):
    """End-to-end: rendered Python snippet must use load_jang_model (text
    loader) NOT load_jangtq_vlm_model (VLM loader) for MiniMax even when
    preprocessor files are present in the output dir."""
    snippet = render_snippet(minimax_with_stray_preprocessor, "python")
    # Text-path markers present
    assert "load_jang_model" in snippet, \
        "MiniMax snippet must use text loader (M93)"
    # VLM-path markers absent — rule 11 compliance
    assert "load_jangtq_vlm_model" not in snippet, \
        "MiniMax must NOT emit VLM imports per feedback_readme_standards rule 11"
    assert "mlx_vlm" not in snippet, \
        "MiniMax must NOT import mlx_vlm"
    assert "Image.open" not in snippet, \
        "MiniMax must NOT reference image loading"


def test_minimax_aliases_also_forced_text_only(tmp_path):
    """The text-only set includes aliases `minimax` and `minimax_m2_5`.
    Pin these too so a future model_type string variant doesn't escape."""
    from jang_tools.examples import _TEXT_ONLY_MODEL_TYPES
    assert "minimax_m2" in _TEXT_ONLY_MODEL_TYPES
    assert "minimax_m2_5" in _TEXT_ONLY_MODEL_TYPES
    assert "minimax" in _TEXT_ONLY_MODEL_TYPES


def test_genuine_vl_model_still_vl(vl_model_dir):
    """Negative guard: a real VL model (qwen2_vl with preprocessor) must
    STILL be detected as VL. Ensures M93's enforcement didn't broadcast
    to all models."""
    caps = detect_capabilities(vl_model_dir)
    assert caps["model_type"] == "qwen2_vl"
    assert caps["is_vl"] is True, "real VL model must still be detected as VL"


# ────────────────────────────────────────────────────────────────────
# Iter 27: M90 — has_thinking capability flag + `thinking` YAML tag
# per feedback_readme_standards.md rule 10
# ────────────────────────────────────────────────────────────────────

@pytest.fixture
def qwen3_5_thinking_model_dir(tmp_path):
    """A minimal Qwen3.5 model dir with enable_thinking set. Triggers both
    has_reasoning AND has_thinking per the iter-27 capability split."""
    d = tmp_path / "qwen3_5_thinking"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5_moe",
        "num_hidden_layers": 28,
        "enable_thinking": True,
        "_name_or_path": "Qwen/Qwen3.5-MoE-Preview",
    }))
    (d / "jang_config.json").write_text(json.dumps({
        "format": "jang", "family": "jang", "profile": "JANG_4K",
        "quantization": {"actual_bits_per_weight": 4.0, "block_size": 64},
    }))
    (d / "tokenizer_config.json").write_text(json.dumps({
        "chat_template": "x",
        "tokenizer_class": "Qwen2Tokenizer",
    }))
    return d


def test_capabilities_reports_has_thinking_for_qwen3_5(qwen3_5_thinking_model_dir):
    caps = detect_capabilities(qwen3_5_thinking_model_dir)
    assert caps["has_thinking"] is True
    # has_reasoning is the broader capability — also true when enable_thinking is set
    assert caps["has_reasoning"] is True


def test_capabilities_has_thinking_false_when_no_enable_thinking(dense_model_dir):
    """Plain Qwen3 without enable_thinking must NOT trip has_thinking."""
    caps = detect_capabilities(dense_model_dir)
    assert caps["has_thinking"] is False


def test_modelcard_emits_both_reasoning_and_thinking_tags(qwen3_5_thinking_model_dir):
    """feedback_readme_standards.md rule 10: Qwen3.5 models must carry BOTH
    `reasoning` AND `thinking` YAML frontmatter tags. Prior to iter 27 only
    `reasoning` was emitted — `thinking` was missing despite the memory
    rule and despite being a DIFFERENT semantic flag (reasoning = capability,
    thinking = runtime toggle).
    """
    from jang_tools.modelcard import generate_card
    # M202 (iter 138): generate_card returns (card, license_unknown).
    card, _ = generate_card(qwen3_5_thinking_model_dir)
    # Both tags must appear in the YAML tags list.
    assert "- reasoning" in card, "missing `reasoning` tag"
    assert "- thinking" in card, "missing `thinking` tag (M90 regression)"


def test_modelcard_omits_thinking_tag_when_not_applicable(dense_model_dir):
    """Negative: a plain model without enable_thinking must NOT get the
    thinking tag. Prevents spurious tags on models that don't support it."""
    from jang_tools.modelcard import generate_card
    # M202 (iter 138): generate_card returns (card, license_unknown).
    card, _ = generate_card(dense_model_dir)
    assert "- thinking" not in card, "thinking tag wrongly added to non-thinking model"


def test_python_snippet_imports_resolve_to_real_symbols():
    """M45 (iter 20): compile-only validation doesn't catch import-name typos.
    The previous snippet said `from jang_tools.loader import load_model` —
    compiled fine, but adopters hit `ImportError: cannot import name 'load_model'`
    at runtime. Verify the symbols referenced in the Python snippet actually
    exist in the modules they're imported from.
    """
    import importlib
    # Non-VL template references:
    mod = importlib.import_module("jang_tools.loader")
    assert hasattr(mod, "load_jang_model"), \
        "jang_tools.loader.load_jang_model missing — python snippet will ImportError"
    # Negative: the buggy name must NOT exist OR at least not be what we use.
    # (Being extra-safe: don't assert absence in case someone later adds an alias.)

    # VL template references:
    vl_mod = importlib.import_module("jang_tools.load_jangtq_vlm")
    assert hasattr(vl_mod, "load_jangtq_vlm_model"), \
        "jang_tools.load_jangtq_vlm.load_jangtq_vlm_model missing — VL snippet will ImportError"


# ────────────────────────────────────────────────────────────────────
# M126 (iter 73): examples error messages name the failing file
# ────────────────────────────────────────────────────────────────────
#
# Pre-iter-73 detect_capabilities called json.loads on 3 different config
# files (config.json, jang_config.json, tokenizer_config.json). On any one
# being corrupt, cmd_examples's outer except-Exception emitted
# `ERROR: JSONDecodeError: Expecting value: line 1 column 1 (char 0)` —
# correct that it failed but didn't name WHICH config broke. User had to
# manually check 3 files. iter-73 routes each read through
# _read_json_object(path, purpose=…) so the error is specific.


def test_cli_examples_names_corrupted_config_json(tmp_path):
    """Corrupt config.json — error must name it."""
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text("{ not valid json")
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "examples",
         "--model", str(d), "--lang", "python"],
        capture_output=True, text=True, check=False,
    )
    assert r.returncode != 0
    assert "config.json" in r.stderr
    assert "not valid JSON" in r.stderr
    # Pre-M126, this test would only find "JSONDecodeError" with no path.


def test_cli_examples_names_corrupted_jang_config(tmp_path):
    """Valid config.json, corrupt jang_config.json — error must name
    jang_config specifically (not config.json)."""
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (d / "jang_config.json").write_text("broken")
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "examples",
         "--model", str(d), "--lang", "python"],
        capture_output=True, text=True, check=False,
    )
    assert r.returncode != 0
    assert "jang_config.json" in r.stderr, \
        f"M126: error must name jang_config.json, got: {r.stderr}"


def test_cli_examples_names_corrupted_tokenizer_config(tmp_path):
    """Valid config + jang_config, corrupt tokenizer_config — error must
    name tokenizer_config specifically."""
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (d / "jang_config.json").write_text(json.dumps({"profile": "JANG_4K"}))
    (d / "tokenizer_config.json").write_text("{broken}")
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "examples",
         "--model", str(d), "--lang", "python"],
        capture_output=True, text=True, check=False,
    )
    assert r.returncode != 0
    assert "tokenizer_config.json" in r.stderr, \
        f"M126: error must name tokenizer_config.json, got: {r.stderr}"
