import json
from pathlib import Path

import pytest


BUNDLE = Path("/Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L")


pytestmark = pytest.mark.skipif(
    not (BUNDLE / "config.json").exists(),
    reason="Nemotron Ultra JANGTQ_1L bundle is not mounted",
)


def _json(name: str):
    return json.loads((BUNDLE / name).read_text())


def test_ultra_artifact_is_text_only_hybrid_with_expected_cache_topology():
    cfg = _json("config.json")
    layer_types = cfg["layers_block_type"]

    assert cfg["model_type"] == "nemotron_h"
    assert cfg["num_hidden_layers"] == 108
    assert len(layer_types) == 108
    assert layer_types.count("mamba") == 48
    assert layer_types.count("moe") == 48
    assert layer_types.count("attention") == 12
    assert layer_types.count("mamba") + layer_types.count("attention") == 60
    assert cfg["num_nextn_predict_layers"] == 0
    assert cfg["mtp_layers_block_type"] == []
    assert "vision_config" not in cfg
    assert "audio_config" not in cfg


def test_ultra_artifact_stamps_engine_capabilities():
    caps = _json("jang_config.json")["capabilities"]

    assert caps == {
        "cache_type": "hybrid",
        "family": "nemotron_h",
        "modality": "text",
        "reasoning_parser": "deepseek_r1",
        "supports_thinking": True,
        "supports_tools": True,
        "think_in_template": True,
        "tool_parser": "nemotron",
    }


def test_ultra_artifact_preserves_mamba_and_moe_group_names_distinctly():
    cfg = _json("config.json")

    assert cfg["n_group"] == 1
    assert cfg["topk_group"] == 1
    assert cfg["n_groups"] == 8
    assert cfg["mamba_num_heads"] * cfg["mamba_head_dim"] == 16384
    assert cfg["intermediate_size"] == 5120
    assert 16384 + 2 * cfg["n_groups"] * cfg["ssm_state_size"] == 18432
