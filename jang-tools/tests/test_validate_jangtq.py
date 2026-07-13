import json
import os
import subprocess
import sys
from pathlib import Path


def _env() -> dict[str, str]:
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[1])
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = root if not existing else os.pathsep.join([root, existing])
    return env


def _write_jangtq_bundle(path: Path, *, missing_shard: bool = False) -> None:
    path.mkdir()
    (path / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5_moe",
        "weight_format": "mxtq",
        "mxtq_bits": 4,
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 40,
            "num_experts": 256,
            "dtype": "bfloat16",
        },
    }))
    (path / "jang_config.json").write_text(json.dumps({
        "version": 2,
        "weight_format": "mxtq",
        "profile": "JANGTQ4",
        "source_model": {
            "name": "Qwen3.6-35B-A3B",
            "architecture": "qwen3_5_moe_text",
        },
        "quantization": {
            "method": "affine+mxtq",
            "group_size": 64,
            "bits_default": 4,
        },
        "capabilities": {
            "reasoning_parser": "qwen3",
            "tool_parser": "qwen",
            "think_in_template": True,
            "supports_tools": True,
            "supports_thinking": True,
            "family": "qwen3_5_moe",
            "modality": "text",
            "cache_type": "hybrid",
        },
    }))
    (path / "tokenizer.json").write_text("{}")
    (path / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "Qwen2Tokenizer",
        "chat_template": "{% for m in messages %}{{m.content}}{% endfor %}",
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
        "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
    }))
    (path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"format": "jangtq", "total_size": 0},
        "weight_map": {
            "language_model.model.embed_tokens.weight": "model-00001-of-00001.safetensors",
        },
    }))
    if not missing_shard:
        (path / "model-00001-of-00001.safetensors").write_bytes(b"")


def test_validate_accepts_jangtq_without_legacy_format_field(tmp_path):
    bundle = tmp_path / "qwen-jangtq"
    _write_jangtq_bundle(bundle)

    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "--quiet-text", "validate", str(bundle)],
        env=_env(),
        capture_output=True,
        text=True,
        check=True,
    )

    assert "VALID JANGTQ" in r.stdout
    assert "Qwen3.6-35B-A3B" in r.stdout
    assert "JANGTQ4" in r.stdout


def test_validate_jangtq_rejects_missing_index_shard(tmp_path):
    bundle = tmp_path / "bad-jangtq"
    _write_jangtq_bundle(bundle, missing_shard=True)

    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "--quiet-text", "validate", str(bundle)],
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert r.returncode != 0
    assert "missing shard" in r.stdout.lower()
