"""MiMo-V2.5 source-contract tests for JANG_2L bring-up."""

from pathlib import Path
import json

import numpy as np
import pytest
from safetensors import safe_open


MIMO_SRC = Path("/Volumes/EricsLLMDrive/jangq-ai/sources/MiMo-V2.5")


pytestmark = pytest.mark.skipif(
    not MIMO_SRC.exists(),
    reason=f"MiMo source not mounted at {MIMO_SRC}",
)


def test_mimo_source_contract_matches_real_config_and_tensors():
    from jang_tools.mimo_v2.source_contract import inspect_mimo_source

    contract = inspect_mimo_source(MIMO_SRC)

    assert contract.model_type == "mimo_v2"
    assert contract.num_hidden_layers == 48
    assert contract.n_routed_experts == 256
    assert contract.num_experts_per_tok == 8
    assert contract.full_kv_heads == 4
    assert contract.swa_kv_heads == 8
    assert contract.full_qkv_shape == (13568, 4096)
    assert contract.swa_qkv_shape == (14848, 4096)
    assert contract.full_layer_count == 9
    assert contract.swa_layer_count == 39
    assert contract.has_visual_tensors is True
    assert contract.has_audio_tensors is True
    assert contract.has_mtp_tensors is True
    assert contract.ignored_text_o_proj_count == 48
    assert contract.capabilities["family"] == "mimo_v2"
    assert contract.capabilities["cache_type"] == "kv"
    assert contract.capabilities["reasoning_parser"] == "think_xml"
    assert contract.capabilities["tool_parser"] == "xml_function"
    assert contract.runtime["mtp_mode"] == "preserved_disabled"
    assert contract.runtime["cache_topology"]["turboquant_kv"] == "full_attention_layers_only"


def test_mimo_fp8_block_codec_matches_torch_reference_on_real_tensor():
    torch = pytest.importorskip("torch")
    from jang_tools.mimo_v2.fp8_block_codec import dequant_fp8_e4m3_scale_inv

    tensor_name = "model.layers.1.mlp.experts.0.down_proj.weight"
    scale_name = f"{tensor_name}_scale_inv"
    shard_path = MIMO_SRC / "model_pp0_ep0_shard0.safetensors"

    with safe_open(str(shard_path), framework="pt") as f:
        fp8_weight = f.get_tensor(tensor_name)
        scale_inv = f.get_tensor(scale_name)

    actual = dequant_fp8_e4m3_scale_inv(fp8_weight, scale_inv, out_dtype=torch.float32)

    scale_full = scale_inv.float().repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    expected = fp8_weight.float() * scale_full[: fp8_weight.shape[0], : fp8_weight.shape[1]]

    actual_np = actual.detach().cpu().numpy()
    expected_np = expected.detach().cpu().numpy()
    np.testing.assert_allclose(actual_np[:8, :8], expected_np[:8, :8], rtol=0, atol=0)
    assert actual.shape == fp8_weight.shape
    assert actual.dtype == torch.float32


def test_mimo_k_profile_metadata_targets_runtime_switch_mlp_modules(tmp_path):
    from jang_tools.mimo_v2.convert_jang import (
        QuantProfile,
        _write_config_json,
        classify,
        runtime_quant_base_for_weight,
    )

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "config.json").write_text(json.dumps({
        "model_type": "mimo_v2",
        "rope_theta": 10_000_000.0,
        "partial_rotary_factor": 0.334,
        "sliding_window": 128,
        "quantization_config": {"ignored": True},
    }))

    profile = QuantProfile.parse("2k")
    assert classify("model.layers.1.mlp.experts.0.gate_proj.weight", profile) == (2, "affine", 64)
    assert classify("model.layers.1.mlp.experts.0.up_proj.weight", profile) == (2, "affine", 64)
    assert classify("model.layers.1.mlp.experts.0.down_proj.weight", profile) == (4, "affine", 64)
    assert runtime_quant_base_for_weight(
        "model.layers.1.mlp.experts.42.down_proj.weight"
    ) == "model.layers.1.mlp.switch_mlp.down_proj"

    _write_config_json(src, dst, profile, 64, {
        "model.layers.1.mlp.switch_mlp.gate_proj": {"bits": 2, "group_size": 64, "mode": "affine"},
        "model.layers.1.mlp.switch_mlp.up_proj": {"bits": 2, "group_size": 64, "mode": "affine"},
        "model.layers.1.mlp.switch_mlp.down_proj": {"bits": 4, "group_size": 64, "mode": "affine"},
    })

    cfg = json.loads((dst / "config.json").read_text())
    assert cfg["jang_profile"] == "JANG_2K"
    assert cfg["quantization"]["bits"] == 8
    assert "overrides" not in cfg["quantization"]
    assert cfg["quantization"]["model.layers.1.mlp.switch_mlp.gate_proj"]["bits"] == 2
    assert cfg["quantization"]["model.layers.1.mlp.switch_mlp.down_proj"]["bits"] == 4
    assert cfg["mxtq_bits"] == {"gate_proj": 2, "up_proj": 2, "down_proj": 4}
    assert cfg["capabilities"]["cache_type"] == "kv"
    assert cfg["capabilities"]["reasoning"]["parser"] == "think_xml"
    assert cfg["capabilities"]["tools"]["parser"] == "xml_function"
    assert cfg["runtime"]["mtp_mode"] == "preserved_disabled"
    assert cfg["runtime"]["cache_topology"]["family"] == "hybrid_full_swa_kv"


def test_mimo_v2_shared_capability_resolver_preserves_parser_and_cache_policy():
    from jang_tools.capabilities import build_capabilities

    caps = build_capabilities(
        {"source_model": {"architecture": "mimo_v2"}},
        {"model_type": "mimo_v2"},
    )

    assert caps == {
        "reasoning_parser": "think_xml",
        "tool_parser": "xml_function",
        "think_in_template": False,
        "supports_tools": True,
        "supports_thinking": True,
        "family": "mimo_v2",
        "modality": "text",
        "cache_type": "kv",
    }

    flash_caps = build_capabilities({}, {"model_type": "mimo_v2_flash"})
    assert flash_caps == caps
