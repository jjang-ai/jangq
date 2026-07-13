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
    assert contract.capabilities["modality"] == "text"
    assert contract.capabilities["modalities"] == ["text"]
    assert contract.capabilities["preserved_modalities"] == ["vision", "audio"]
    assert contract.capabilities["unwired_modalities"] == ["vision", "audio"]
    assert contract.capabilities["multimodal_status"] == "weights_preserved_text_runtime"
    assert contract.capabilities["cache_type"] == "kv"
    assert contract.capabilities["reasoning_parser"] == "think_xml"
    assert contract.capabilities["tool_parser"] == "xml_function"
    assert contract.runtime["mtp_mode"] == "preserved_disabled"
    assert contract.runtime["cache_topology"]["family"] == "hybrid_full_swa_kv"
    assert contract.runtime["cache_topology"]["prefix_cache"] is True
    assert contract.runtime["cache_topology"]["l2_disk_cache"] is True
    assert contract.runtime["cache_topology"]["turboquant_kv"] == "full_attention_layers_only"
    assert contract.runtime["cache_topology"]["swa_layers"] == "rotating_kv_native"


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


def test_mimo_deinterleave_tp4_qkv_rows_synthetic():
    torch = pytest.importorskip("torch")
    from jang_tools.mimo_v2.weight_loader import deinterleave_tp_qkv_rows

    q_size = 8
    k_size = 4
    v_size = 4
    cols = 3
    raw_rows = []
    for rank in range(4):
        raw_rows.extend([100 + rank * 10 + i for i in range(q_size // 4)])
        raw_rows.extend([200 + rank * 10 + i for i in range(k_size // 4)])
        raw_rows.extend([300 + rank * 10 + i for i in range(v_size // 4)])
    raw = torch.tensor(raw_rows, dtype=torch.float32).reshape(-1, 1).repeat(1, cols)

    actual = deinterleave_tp_qkv_rows(raw, q_size=q_size, k_size=k_size, v_size=v_size, tp_size=4)
    expected_rows = [
        100, 101, 110, 111, 120, 121, 130, 131,
        200, 210, 220, 230,
        300, 310, 320, 330,
    ]
    expected = torch.tensor(expected_rows, dtype=torch.float32).reshape(-1, 1).repeat(1, cols)

    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=0, atol=0)


def test_mimo_source_full_qkv_scale_shape_exposes_tp4_rank_blocks():
    from safetensors import safe_open

    tensor_name = "model.layers.0.self_attn.qkv_proj.weight"
    scale_name = f"{tensor_name}_scale_inv"
    shard_path = MIMO_SRC / "model_pp0_ep0_shard1.safetensors"

    with safe_open(str(shard_path), framework="pt") as f:
        fp8_weight = f.get_tensor(tensor_name)
        scale_inv = f.get_tensor(scale_name)

    assert tuple(fp8_weight.shape) == (13568, 4096)
    # A logical [all_q, all_k, all_v] matrix would need ceil(13568 / 128)=106
    # scale rows. The source has 108 because the FP8 blocks are stored as four
    # TP-rank qkv row groups: 4 * ceil((3072 + 192 + 128) / 128).
    assert tuple(scale_inv.shape) == (108, 32)


def test_mimo_shard_index_deinterleaves_qkv_after_fp8_decode():
    torch = pytest.importorskip("torch")
    from jang_tools.mimo_v2.fp8_block_codec import dequant_fp8_e4m3_scale_inv
    from jang_tools.mimo_v2.weight_loader import MiMoShardIndex, deinterleave_tp_qkv_rows

    tensor_name = "model.layers.0.self_attn.qkv_proj.weight"
    scale_name = f"{tensor_name}_scale_inv"
    shard_path = MIMO_SRC / "model_pp0_ep0_shard1.safetensors"

    with safe_open(str(shard_path), framework="pt") as f:
        fp8_weight = f.get_tensor(tensor_name)
        scale_inv = f.get_tensor(scale_name)

    raw_decoded = dequant_fp8_e4m3_scale_inv(fp8_weight, scale_inv, out_dtype=torch.float32)
    expected = deinterleave_tp_qkv_rows(raw_decoded, q_size=12288, k_size=768, v_size=512, tp_size=4)
    actual = MiMoShardIndex(MIMO_SRC).read_tensor(tensor_name, out_dtype=torch.float32)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual[:16, :16].numpy(), expected[:16, :16].numpy(), rtol=0, atol=0)
    np.testing.assert_allclose(actual[12288:12304, :16].numpy(), expected[12288:12304, :16].numpy(), rtol=0, atol=0)


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
    assert cfg["capabilities"]["modalities"] == ["text"]
    assert cfg["capabilities"]["preserved_modalities"] == ["vision", "audio"]
    assert cfg["capabilities"]["unwired_modalities"] == ["vision", "audio"]
    assert cfg["capabilities"]["multimodal_status"] == "weights_preserved_text_runtime"
    assert cfg["capabilities"]["cache_type"] == "kv"
    assert cfg["capabilities"]["reasoning"]["parser"] == "think_xml"
    assert cfg["capabilities"]["tools"]["parser"] == "xml_function"
    assert cfg["runtime"]["mtp_mode"] == "preserved_disabled"
    assert cfg["runtime"]["multimodal_mode"] == "weights_preserved_text_runtime"
    assert cfg["runtime"]["cache_topology"]["family"] == "hybrid_full_swa_kv"
    assert cfg["runtime"]["cache_topology"]["prefix_cache"] is True
    assert cfg["runtime"]["cache_topology"]["l2_disk_cache"] is True
    assert cfg["runtime"]["cache_topology"]["turboquant_kv"] == "full_attention_layers_only"
    assert cfg["runtime"]["cache_topology"]["swa_layers"] == "rotating_kv_native"


def test_mimo_2s_profile_uses_sub105_affine_policy_metadata(tmp_path):
    from jang_tools.mimo_v2.convert_jang import QuantProfile, _write_config_json, classify

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "config.json").write_text(json.dumps({
        "model_type": "mimo_v2",
        "rope_theta": 10_000_000.0,
        "partial_rotary_factor": 0.334,
        "sliding_window": 128,
    }))

    profile = QuantProfile.parse("2s")
    assert classify("model.layers.1.mlp.experts.0.gate_proj.weight", profile) == (4, "affine", 64)
    assert classify("model.layers.1.mlp.experts.0.up_proj.weight", profile) == (2, "affine", 64)
    assert classify("model.layers.1.mlp.experts.0.down_proj.weight", profile) == (3, "affine", 64)
    assert classify("model.layers.0.self_attn.qkv_proj.weight", profile) == (6, "affine", 64)
    assert classify("model.layers.0.mlp.gate_proj.weight", profile) == (6, "affine", 64)
    assert classify("model.layers.0.self_attn.o_proj.weight", profile) == (8, "affine", 64)
    assert classify("model.embed_tokens.weight", profile) == (16, "passthrough_bf16", 0)
    assert classify("lm_head.weight", profile) == (16, "passthrough_bf16", 0)

    _write_config_json(src, dst, profile, 64, {
        "model.layers.1.mlp.switch_mlp.gate_proj": {"bits": 4, "group_size": 64, "mode": "affine"},
        "model.layers.1.mlp.switch_mlp.up_proj": {"bits": 2, "group_size": 64, "mode": "affine"},
        "model.layers.1.mlp.switch_mlp.down_proj": {"bits": 3, "group_size": 64, "mode": "affine"},
        "model.layers.0.self_attn.qkv_proj": {"bits": 6, "group_size": 64, "mode": "affine"},
        "model.layers.0.mlp.gate_proj": {"bits": 6, "group_size": 64, "mode": "affine"},
    })

    cfg = json.loads((dst / "config.json").read_text())
    assert cfg["jang_profile"] == "JANG_2S"
    assert cfg["mxtq_bits"] == {"gate_proj": 4, "up_proj": 2, "down_proj": 3}
    assert cfg["routed_expert_bits"] == {"gate_proj": 4, "up_proj": 2, "down_proj": 3}
    assert cfg["quantization"]["bits"] == 8
    assert cfg["quantization"]["model.layers.1.mlp.switch_mlp.gate_proj"]["bits"] == 4
    assert cfg["quantization"]["model.layers.1.mlp.switch_mlp.up_proj"]["bits"] == 2
    assert cfg["quantization"]["model.layers.1.mlp.switch_mlp.down_proj"]["bits"] == 3
    assert cfg["quantization"]["model.layers.0.self_attn.qkv_proj"]["bits"] == 6
    assert cfg["quantization"]["model.layers.0.mlp.gate_proj"]["bits"] == 6
    assert cfg["runtime"]["quantization_profile"] == "JANG_2S"


@pytest.mark.parametrize(
    ("raw", "name", "expert_bits", "qkv_bits", "dense_bits"),
    [
        ("2c", "JANG_2C", {"gate_proj": 4, "up_proj": 3, "down_proj": 3}, 6, 6),
        ("2x", "JANG_2X", {"gate_proj": 3, "up_proj": 2, "down_proj": 3}, 5, 6),
        ("2q", "JANG_2Q", {"gate_proj": 2, "up_proj": 2, "down_proj": 2}, 6, 6),
        ("2f", "JANG_2F", {"gate_proj": 2, "up_proj": 2, "down_proj": 2}, 4, 4),
    ],
)
def test_mimo_alternate_sub105_affine_profiles(raw, name, expert_bits, qkv_bits, dense_bits):
    from jang_tools.mimo_v2.convert_jang import QuantProfile, classify

    profile = QuantProfile.parse(raw)

    assert profile.name == name
    assert profile.expert_proj_bits == expert_bits
    assert classify("model.layers.1.mlp.experts.0.gate_proj.weight", profile) == (
        expert_bits["gate_proj"], "affine", 64
    )
    assert classify("model.layers.1.mlp.experts.0.up_proj.weight", profile) == (
        expert_bits["up_proj"], "affine", 64
    )
    assert classify("model.layers.1.mlp.experts.0.down_proj.weight", profile) == (
        expert_bits["down_proj"], "affine", 64
    )
    assert classify("model.layers.0.self_attn.qkv_proj.weight", profile) == (qkv_bits, "affine", 64)
    assert classify("model.layers.0.mlp.gate_proj.weight", profile) == (dense_bits, "affine", 64)
    assert classify("model.layers.0.self_attn.o_proj.weight", profile) == (
        profile.o_proj_bits, "affine", 64
    )
    expected_embed = (16, "passthrough_bf16", 0) if profile.token_io_bf16 else (profile.default_bits, "affine", 64)
    assert classify("model.embed_tokens.weight", profile) == expected_embed


def test_mimo_projection_specific_8bit_expert_profile_metadata():
    from jang_tools.mimo_v2.convert_jang import QuantProfile, classify

    profile = QuantProfile.parse("448g128b8q8")

    assert profile.name == "JANG_448G128B8Q8"
    assert profile.expert_proj_bits == {"gate_proj": 4, "up_proj": 4, "down_proj": 8}
    assert profile.expert_group_size == 128
    assert classify("model.layers.1.mlp.experts.0.gate_proj.weight", profile) == (4, "affine", 128)
    assert classify("model.layers.1.mlp.experts.0.up_proj.weight", profile) == (4, "affine", 128)
    assert classify("model.layers.1.mlp.experts.0.down_proj.weight", profile) == (8, "affine", 128)


def test_mimo_late_8bit_expert_override_profile_metadata():
    from jang_tools.mimo_v2.convert_jang import QuantProfile, classify

    profile = QuantProfile.parse("444g64l4x8")

    assert profile.name == "JANG_444G64L4X8B8Q8"
    assert profile.expert_group_size == 64
    assert classify("model.layers.43.mlp.experts.0.down_proj.weight", profile) == (4, "affine", 64)
    assert classify("model.layers.44.mlp.experts.0.gate_proj.weight", profile) == (8, "affine", 64)
    assert classify("model.layers.47.mlp.experts.0.down_proj.weight", profile) == (8, "affine", 64)


def test_mimo_early_8bit_expert_override_profile_metadata():
    from jang_tools.mimo_v2.convert_jang import QuantProfile, classify

    profile = QuantProfile.parse("444g64e4x8")

    assert profile.name == "JANG_444G64E4X8B8Q8"
    assert profile.expert_group_size == 64
    assert classify("model.layers.1.mlp.experts.0.gate_proj.weight", profile) == (8, "affine", 64)
    assert classify("model.layers.4.mlp.experts.0.down_proj.weight", profile) == (8, "affine", 64)
    assert classify("model.layers.5.mlp.experts.0.down_proj.weight", profile) == (4, "affine", 64)


def test_mimo_grouped_profile_can_keep_token_io_bf16():
    from jang_tools.mimo_v2.convert_jang import QuantProfile, classify

    profile = QuantProfile.parse("444g64t16")

    assert profile.name == "JANG_444G64B8Q8T16"
    assert profile.token_io_bf16 is True
    assert classify("model.embed_tokens.weight", profile) == (16, "passthrough_bf16", 0)
    assert classify("lm_head.weight", profile) == (16, "passthrough_bf16", 0)
    assert classify("model.layers.1.mlp.experts.0.down_proj.weight", profile) == (4, "affine", 64)


def test_mimo_grouped_profile_can_keep_non_expert_text_bf16():
    from jang_tools.mimo_v2.convert_jang import QuantProfile, classify

    profile = QuantProfile.parse("444g64n16")

    assert profile.name == "JANG_444G64B8Q8N16"
    assert profile.non_expert_text_bf16 is True
    assert classify("model.embed_tokens.weight", profile) == (16, "passthrough_bf16", 0)
    assert classify("lm_head.weight", profile) == (16, "passthrough_bf16", 0)
    assert classify("model.layers.0.self_attn.qkv_proj.weight", profile) == (16, "passthrough_bf16", 0)
    assert classify("model.layers.0.mlp.gate_proj.weight", profile) == (16, "passthrough_bf16", 0)
    assert classify("model.layers.1.mlp.experts.0.down_proj.weight", profile) == (4, "affine", 64)


def test_mimo_bf16_storage_dequantizes_fp8_source_weights():
    torch = pytest.importorskip("torch")
    from jang_tools.mimo_v2.convert_jang import read_bf16_storage_tensor

    class FakeIndex:
        def __init__(self, *, fp8: bool):
            self.fp8 = fp8
            self.calls = []

        def is_fp8_weight(self, name):
            return self.fp8

        def read_tensor(self, name, *, out_dtype):
            self.calls.append(("read_tensor", name, out_dtype))
            return torch.ones((2, 2), dtype=out_dtype)

        def read_passthrough(self, name):
            self.calls.append(("read_passthrough", name))
            return torch.ones((2, 2), dtype=torch.float32)

    fp8_idx = FakeIndex(fp8=True)
    fp8_tensor = read_bf16_storage_tensor(fp8_idx, "model.layers.0.self_attn.qkv_proj.weight")
    assert fp8_tensor.dtype == torch.bfloat16
    assert fp8_idx.calls == [
        ("read_tensor", "model.layers.0.self_attn.qkv_proj.weight", torch.bfloat16)
    ]

    plain_idx = FakeIndex(fp8=False)
    plain_tensor = read_bf16_storage_tensor(plain_idx, "model.embed_tokens.weight")
    assert plain_tensor.dtype == torch.bfloat16
    assert plain_idx.calls == [("read_passthrough", "model.embed_tokens.weight")]


def test_mimo_2s_verifier_expects_bf16_token_io_and_quantized_oproj():
    from jang_tools.mimo_v2.verify_bundle import _storage_policy_for_config

    policy = _storage_policy_for_config({
        "jang_profile": "JANG_2S",
        "runtime": {"bundle_has_mtp": False},
        "quantization": {},
    })

    assert policy.token_io_passthrough_bf16 is True
    assert policy.o_proj_quantized is True
    assert policy.prestacked_jangtq is False
    assert policy.bundle_has_mtp is False


@pytest.mark.parametrize(
    ("profile_name", "token_io_passthrough"),
    [
        ("JANG_2C", True),
        ("JANG_2X", True),
        ("JANG_2Q", False),
        ("JANG_2F", False),
    ],
)
def test_mimo_verifier_storage_policy_for_alternate_profiles(profile_name, token_io_passthrough):
    from jang_tools.mimo_v2.verify_bundle import _storage_policy_for_config

    policy = _storage_policy_for_config({
        "jang_profile": profile_name,
        "runtime": {"bundle_has_mtp": False},
        "quantization": {},
    })

    assert policy.token_io_passthrough_bf16 is token_io_passthrough
    assert policy.o_proj_quantized is True


def test_mimo_slim_322d3e_profile_carries_layer_specific_metadata(tmp_path):
    from jang_tools.mimo_v2.convert_jang import QuantProfile, _write_config_json, classify
    from jang_tools.mimo_v2.verify_bundle import _storage_policy_for_config

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "config.json").write_text(json.dumps({
        "model_type": "mimo_v2",
        "rope_theta": 10_000_000.0,
        "partial_rotary_factor": 0.334,
        "sliding_window": 128,
    }))

    profile = QuantProfile.parse("slim322d3e3")
    assert profile.name == "JANG_2R_322D3E3"
    assert profile.default_bits == 4
    assert profile.expert_group_size == 128
    assert classify("model.layers.1.mlp.experts.0.gate_proj.weight", profile) == (3, "affine", 128)
    assert classify("model.layers.1.mlp.experts.0.up_proj.weight", profile) == (2, "affine", 128)
    assert classify("model.layers.1.mlp.experts.0.down_proj.weight", profile) == (3, "affine", 128)
    assert classify("model.layers.4.mlp.experts.0.down_proj.weight", profile) == (2, "affine", 128)
    assert classify("model.layers.0.self_attn.qkv_proj.weight", profile) == (6, "affine", 64)
    assert classify("model.layers.0.mlp.gate_proj.weight", profile) == (6, "affine", 64)
    assert classify("model.layers.0.self_attn.o_proj.weight", profile) == (4, "affine", 64)
    assert classify("model.embed_tokens.weight", profile) == (4, "affine", 64)

    _write_config_json(src, dst, profile, 64, {
        "model.layers.1.mlp.switch_mlp.gate_proj": {"bits": 3, "group_size": 128, "mode": "affine"},
        "model.layers.1.mlp.switch_mlp.up_proj": {"bits": 2, "group_size": 128, "mode": "affine"},
        "model.layers.1.mlp.switch_mlp.down_proj": {"bits": 3, "group_size": 128, "mode": "affine"},
        "model.layers.4.mlp.switch_mlp.down_proj": {"bits": 2, "group_size": 128, "mode": "affine"},
        "model.layers.0.self_attn.qkv_proj": {"bits": 6, "group_size": 64, "mode": "affine"},
    })

    cfg = json.loads((dst / "config.json").read_text())
    assert cfg["jang_profile"] == "JANG_2R_322D3E3"
    assert cfg["routed_expert_bits"] == {"gate_proj": 3, "up_proj": 2, "down_proj": 2}
    assert cfg["runtime"]["routed_expert_bit_plan"]["group_size"] == 128
    assert cfg["runtime"]["routed_expert_bit_plan"]["layer_overrides"]["1"]["down_proj"] == 3
    assert cfg["runtime"]["routed_expert_bit_plan"]["layer_overrides"]["3"]["down_proj"] == 3
    assert "4" not in cfg["runtime"]["routed_expert_bit_plan"]["layer_overrides"]
    assert cfg["quantization"]["model.layers.1.mlp.switch_mlp.down_proj"]["bits"] == 3
    assert cfg["quantization"]["model.layers.1.mlp.switch_mlp.down_proj"]["group_size"] == 128

    policy = _storage_policy_for_config(cfg)
    assert policy.token_io_passthrough_bf16 is False
    assert policy.o_proj_quantized is True


def test_mimo_slim_322d3e2b6_profile_uses_six_bit_bookends_under_e2(tmp_path):
    from jang_tools.mimo_v2.convert_jang import QuantProfile, _write_config_json, classify
    from jang_tools.mimo_v2.verify_bundle import _storage_policy_for_config

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "config.json").write_text(json.dumps({
        "model_type": "mimo_v2",
        "rope_theta": 10_000_000.0,
        "partial_rotary_factor": 0.334,
        "sliding_window": 128,
    }))

    profile = QuantProfile.parse("slim322d3e2b6")
    assert profile.name == "JANG_2R_322D3E2B6"
    assert profile.default_bits == 6
    assert profile.expert_group_size == 128
    assert classify("model.embed_tokens.weight", profile) == (6, "affine", 64)
    assert classify("lm_head.weight", profile) == (6, "affine", 64)
    assert classify("model.layers.0.self_attn.qkv_proj.weight", profile) == (6, "affine", 64)
    assert classify("model.layers.0.self_attn.o_proj.weight", profile) == (4, "affine", 64)
    assert classify("model.layers.1.mlp.experts.0.down_proj.weight", profile) == (3, "affine", 128)
    assert classify("model.layers.2.mlp.experts.0.down_proj.weight", profile) == (3, "affine", 128)
    assert classify("model.layers.3.mlp.experts.0.down_proj.weight", profile) == (2, "affine", 128)

    _write_config_json(src, dst, profile, 64, {
        "model.embed_tokens": {"bits": 6, "group_size": 64, "mode": "affine"},
        "lm_head": {"bits": 6, "group_size": 64, "mode": "affine"},
        "model.layers.1.mlp.switch_mlp.down_proj": {"bits": 3, "group_size": 128, "mode": "affine"},
        "model.layers.2.mlp.switch_mlp.down_proj": {"bits": 3, "group_size": 128, "mode": "affine"},
        "model.layers.3.mlp.switch_mlp.down_proj": {"bits": 2, "group_size": 128, "mode": "affine"},
        "model.layers.0.self_attn.o_proj": {"bits": 4, "group_size": 64, "mode": "affine"},
    })

    cfg = json.loads((dst / "config.json").read_text())
    assert cfg["jang_profile"] == "JANG_2R_322D3E2B6"
    assert cfg["quantization"]["bits"] == 6
    assert cfg["runtime"]["routed_expert_bit_plan"]["layer_overrides"]["2"]["down_proj"] == 3
    assert "3" not in cfg["runtime"]["routed_expert_bit_plan"]["layer_overrides"]

    policy = _storage_policy_for_config(cfg)
    assert policy.default_bits == 6


def test_mimo_slim_333e1b6_profile_spends_headroom_on_first_up_and_down(tmp_path):
    from jang_tools.mimo_v2.convert_jang import QuantProfile, _write_config_json, classify
    from jang_tools.mimo_v2.verify_bundle import _storage_policy_for_config

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "config.json").write_text(json.dumps({
        "model_type": "mimo_v2",
        "rope_theta": 10_000_000.0,
        "partial_rotary_factor": 0.334,
        "sliding_window": 128,
    }))

    profile = QuantProfile.parse("slim333e1b6")
    assert profile.name == "JANG_2R_333E1B6"
    assert profile.default_bits == 6
    assert profile.expert_group_size == 128
    assert classify("model.layers.1.mlp.experts.0.gate_proj.weight", profile) == (3, "affine", 128)
    assert classify("model.layers.1.mlp.experts.0.up_proj.weight", profile) == (3, "affine", 128)
    assert classify("model.layers.1.mlp.experts.0.down_proj.weight", profile) == (3, "affine", 128)
    assert classify("model.layers.2.mlp.experts.0.up_proj.weight", profile) == (2, "affine", 128)
    assert classify("model.layers.2.mlp.experts.0.down_proj.weight", profile) == (2, "affine", 128)
    assert classify("model.embed_tokens.weight", profile) == (6, "affine", 64)

    _write_config_json(src, dst, profile, 64, {
        "model.layers.1.mlp.switch_mlp.gate_proj": {"bits": 3, "group_size": 128, "mode": "affine"},
        "model.layers.1.mlp.switch_mlp.up_proj": {"bits": 3, "group_size": 128, "mode": "affine"},
        "model.layers.1.mlp.switch_mlp.down_proj": {"bits": 3, "group_size": 128, "mode": "affine"},
        "model.layers.2.mlp.switch_mlp.up_proj": {"bits": 2, "group_size": 128, "mode": "affine"},
    })

    cfg = json.loads((dst / "config.json").read_text())
    assert cfg["jang_profile"] == "JANG_2R_333E1B6"
    assert cfg["runtime"]["routed_expert_bit_plan"]["layer_overrides"]["1"]["up_proj"] == 3
    assert cfg["runtime"]["routed_expert_bit_plan"]["layer_overrides"]["1"]["down_proj"] == 3
    assert "2" not in cfg["runtime"]["routed_expert_bit_plan"]["layer_overrides"]
    assert cfg["quantization"]["model.layers.1.mlp.switch_mlp.up_proj"]["bits"] == 3
    policy = _storage_policy_for_config(cfg)
    assert policy.default_bits == 6
    assert policy.token_io_passthrough_bf16 is False
    assert policy.o_proj_quantized is True


def test_mimo_slim_333e2b6q4_profile_trades_qkv_bits_for_two_early_333_layers(tmp_path):
    from jang_tools.mimo_v2.convert_jang import QuantProfile, _write_config_json, classify

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "config.json").write_text(json.dumps({
        "model_type": "mimo_v2",
        "rope_theta": 10_000_000.0,
        "partial_rotary_factor": 0.334,
        "sliding_window": 128,
    }))

    profile = QuantProfile.parse("slim333e2b6q4")
    assert profile.name == "JANG_2R_333E2B6Q4"
    assert profile.default_bits == 6
    assert profile.qkv_bits == 4
    assert classify("model.layers.1.mlp.experts.0.up_proj.weight", profile) == (3, "affine", 128)
    assert classify("model.layers.2.mlp.experts.0.up_proj.weight", profile) == (3, "affine", 128)
    assert classify("model.layers.3.mlp.experts.0.up_proj.weight", profile) == (2, "affine", 128)
    assert classify("model.layers.0.self_attn.qkv_proj.weight", profile) == (4, "affine", 64)
    assert classify("model.embed_tokens.weight", profile) == (6, "affine", 64)

    _write_config_json(src, dst, profile, 64, {
        "model.layers.1.mlp.switch_mlp.up_proj": {"bits": 3, "group_size": 128, "mode": "affine"},
        "model.layers.2.mlp.switch_mlp.up_proj": {"bits": 3, "group_size": 128, "mode": "affine"},
        "model.layers.0.self_attn.qkv_proj": {"bits": 4, "group_size": 64, "mode": "affine"},
    })

    cfg = json.loads((dst / "config.json").read_text())
    assert cfg["jang_profile"] == "JANG_2R_333E2B6Q4"
    assert cfg["quantization"]["bits"] == 6
    assert cfg["runtime"]["routed_expert_bit_plan"]["layer_overrides"]["2"]["up_proj"] == 3
    assert cfg["quantization"]["model.layers.0.self_attn.qkv_proj"]["bits"] == 4


def test_mimo_expert_keep_map_renumbers_experts_and_slices_router_rows(tmp_path):
    torch = pytest.importorskip("torch")
    from jang_tools.mimo_v2.convert_jang import (
        ExpertKeepMap,
        QuantProfile,
        _write_config_json,
        remap_expert_tensor_name,
        slice_router_tensor,
    )

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "config.json").write_text(json.dumps({
        "model_type": "mimo_v2",
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "text_config": {"n_routed_experts": 256},
        "rope_theta": 10_000_000.0,
        "partial_rotary_factor": 0.334,
        "sliding_window": 128,
    }))

    keep = ExpertKeepMap({1: [7, 2, 9], 2: [4, 5, 6]})

    assert remap_expert_tensor_name(
        "model.layers.1.mlp.experts.7.gate_proj.weight", keep
    ) == "model.layers.1.mlp.experts.0.gate_proj.weight"
    assert remap_expert_tensor_name(
        "model.layers.1.mlp.experts.2.gate_proj.scales", keep
    ) == "model.layers.1.mlp.experts.1.gate_proj.scales"
    assert remap_expert_tensor_name(
        "model.layers.1.mlp.experts.8.gate_proj.weight", keep
    ) is None

    router = torch.arange(256 * 4, dtype=torch.float32).reshape(256, 4)
    bias = torch.arange(256, dtype=torch.float32)
    assert slice_router_tensor("model.layers.1.mlp.gate.weight", router, keep).tolist() == router[[7, 2, 9]].tolist()
    assert slice_router_tensor(
        "model.layers.1.mlp.gate.e_score_correction_bias", bias, keep
    ).tolist() == [7.0, 2.0, 9.0]

    _write_config_json(src, dst, QuantProfile.parse("333"), 64, {}, expert_keep_map=keep)
    cfg = json.loads((dst / "config.json").read_text())
    assert cfg["n_routed_experts"] == 3
    assert cfg["text_config"]["n_routed_experts"] == 3
    assert cfg["runtime"]["expert_keep_map"]["keep_experts"] == 3
    assert cfg["runtime"]["expert_keep_map"]["layers"]["1"] == [7, 2, 9]


def test_mimo_k_profile_metadata_supports_no_mtp_target_bundle(tmp_path):
    from jang_tools.mimo_v2.convert_jang import QuantProfile, _write_config_json

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "config.json").write_text(json.dumps({
        "model_type": "mimo_v2",
        "rope_theta": 10_000_000.0,
        "partial_rotary_factor": 0.334,
        "sliding_window": 128,
    }))

    _write_config_json(src, dst, QuantProfile.parse("2k"), 64, {}, include_mtp=False)

    cfg = json.loads((dst / "config.json").read_text())
    assert cfg["runtime"]["bundle_has_mtp"] is False
    assert cfg["runtime"]["mtp_mode"] == "absent"
    assert cfg["runtime"]["cache_topology"]["prefix_cache"] is True
    assert cfg["runtime"]["cache_topology"]["l2_disk_cache"] is True


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
        "supports_thinking": False,
        "family": "mimo_v2",
        "modality": "text",
        "modalities": {"text": True, "vision": False, "audio": False, "video": False},
        "has_vision": False,
        "has_audio": False,
        "has_video": False,
        "cache_type": "kv",
    }

    flash_caps = build_capabilities({}, {"model_type": "mimo_v2_flash"})
    assert flash_caps == caps
