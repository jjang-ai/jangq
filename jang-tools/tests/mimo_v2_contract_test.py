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
    (src / "tokenizer_config.json").write_text(json.dumps({
        "chat_template": "<|im_start|>user\n{{ messages[0]['content'] }}<|im_end|>",
    }))

    profile = QuantProfile.parse("2k")
    assert classify("model.layers.1.mlp.experts.0.gate_proj.weight", profile) == (2, "affine", 128)
    assert classify("model.layers.1.mlp.experts.0.up_proj.weight", profile) == (2, "affine", 128)
    assert classify("model.layers.1.mlp.experts.0.down_proj.weight", profile) == (4, "affine", 128)
    assert runtime_quant_base_for_weight(
        "model.layers.1.mlp.experts.42.down_proj.weight"
    ) == "model.layers.1.mlp.switch_mlp.down_proj"

    _write_config_json(src, dst, profile, {
        "model.layers.1.mlp.switch_mlp.gate_proj": {"bits": 2, "group_size": 128, "mode": "affine"},
        "model.layers.1.mlp.switch_mlp.up_proj": {"bits": 2, "group_size": 128, "mode": "affine"},
        "model.layers.1.mlp.switch_mlp.down_proj": {"bits": 4, "group_size": 128, "mode": "affine"},
    })

    cfg = json.loads((dst / "config.json").read_text())
    assert cfg["jang_profile"] == "JANG_2K"
    assert cfg["chat_template"].startswith("<|im_start|>user")
    assert cfg["quantization"]["bits"] == 8
    assert "overrides" not in cfg["quantization"]
    assert cfg["quantization"]["model.layers.1.mlp.switch_mlp.gate_proj"]["bits"] == 2
    assert cfg["quantization"]["model.layers.1.mlp.switch_mlp.gate_proj"]["group_size"] == 128
    assert cfg["quantization"]["model.layers.1.mlp.switch_mlp.down_proj"]["bits"] == 4
    assert cfg["quantization"]["model.layers.1.mlp.switch_mlp.down_proj"]["group_size"] == 128
    assert cfg["mxtq_bits"] == {"gate_proj": 2, "up_proj": 2, "down_proj": 4}
    assert cfg["routed_expert_group_size"] == 128
    assert cfg["capabilities"]["cache_type"] == "kv"
    assert cfg["capabilities"]["reasoning"]["parser"] == "think_xml"
    assert cfg["capabilities"]["tools"]["parser"] == "xml_function"
    assert cfg["runtime"]["mtp_mode"] == "preserved_disabled"
    assert cfg["runtime"]["cache_topology"]["family"] == "hybrid_full_swa_kv"
    assert cfg["runtime"]["cache_topology"]["prefix_cache"] is True
    assert cfg["runtime"]["cache_topology"]["l2_disk_cache"] is True
    assert cfg["runtime"]["cache_topology"]["turboquant_kv"] == "full_attention_layers_only"
    assert cfg["runtime"]["cache_topology"]["swa_layers"] == "rotating_kv_native"


def test_mimo_jangtq_contract_emits_prestacked_switch_mlp_metadata(tmp_path):
    from jang_tools.mimo_v2.convert_jangtq import (
        JANGTQProfile,
        jangtq_runtime_base_for_expert_weight,
        tq_tensor_keys_for_expert_weight,
        write_jangtq_metadata,
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
    (src / "tokenizer_config.json").write_text(json.dumps({
        "chat_template": "<|im_start|>user\n{{ messages[0]['content'] }}<|im_end|>",
    }))

    profile = JANGTQProfile.parse("2")
    assert profile.name == "JANGTQ_2"
    assert profile.routed_expert_bits == {"gate_proj": 2, "up_proj": 2, "down_proj": 2}
    assert profile.bits_for_expert_name("model.layers.1.mlp.experts.42.gate_proj.weight") == 2

    base = jangtq_runtime_base_for_expert_weight(
        "model.layers.1.mlp.experts.42.down_proj.weight"
    )
    assert base == "model.layers.1.mlp.switch_mlp.down_proj"
    assert tq_tensor_keys_for_expert_weight(
        "model.layers.1.mlp.experts.42.down_proj.weight"
    ) == (
        "model.layers.1.mlp.switch_mlp.down_proj.tq_packed",
        "model.layers.1.mlp.switch_mlp.down_proj.tq_norms",
        "model.layers.1.mlp.switch_mlp.down_proj.tq_bits",
    )

    write_jangtq_metadata(src, dst, profile, include_mtp=False)

    cfg = json.loads((dst / "config.json").read_text())
    assert cfg["jang_profile"] == "JANGTQ_2"
    assert cfg["format"] == "jangtq"
    assert cfg["quantization"]["quant_method"] == "affine"
    assert cfg["quantization"]["mode"] == "affine"
    assert cfg["quantization"]["routed_experts"] == "tq_prestacked_switch_mlp"
    assert cfg["quantization"]["model.layers.0.self_attn.qkv_proj"] == {
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
    }
    assert cfg["quantization"]["model.layers.0.self_attn.o_proj"] == {
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
    }
    assert cfg["quantization"]["model.layers.47.self_attn.qkv_proj"] == {
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
    }
    assert cfg["quantization"]["model.layers.47.self_attn.o_proj"] == {
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
    }
    assert cfg["mxtq_bits"] == {"routed_expert": {"gate_proj": 2, "up_proj": 2, "down_proj": 2}}
    assert cfg["routed_expert_bits"] == {"gate_proj": 2, "up_proj": 2, "down_proj": 2}
    assert cfg["runtime"]["mtp_mode"] == "absent"
    assert cfg["runtime"]["bundle_has_mtp"] is False
    assert cfg["runtime"]["tq_layout"] == "prestacked_switch_mlp"

    jang_cfg = json.loads((dst / "jang_config.json").read_text())
    assert jang_cfg["format"] == "jangtq"
    assert jang_cfg["family"] == "mimo_v2"
    assert jang_cfg["profile"] == "JANGTQ_2"
    assert jang_cfg["mxtq_seed"] == 42
    assert jang_cfg["routed_expert_bits"] == {"gate_proj": 2, "up_proj": 2, "down_proj": 2}
    assert jang_cfg["tq_layout"] == "prestacked_switch_mlp"


def test_mimo_jangtq_sanitize_regularizes_tq_triplets_for_strict_load():
    from jang_tools.mimo_v2.mlx_model import regularize_jangtq_update_weights

    weights = {
        "model.layers.1.mlp.switch_mlp.gate_proj.tq_packed": "packed",
        "model.layers.1.mlp.switch_mlp.gate_proj.tq_norms": "norms",
        "model.layers.1.mlp.switch_mlp.gate_proj.tq_bits": "bits",
        "model.layers.1.mlp.gate.weight": "router",
    }

    out = regularize_jangtq_update_weights(weights)

    assert out["model.layers.1.mlp.switch_mlp.gate_proj.packed"] == "packed"
    assert out["model.layers.1.mlp.switch_mlp.gate_proj.norms"] == "norms"
    assert "model.layers.1.mlp.switch_mlp.gate_proj.tq_bits" not in out
    assert out["model.layers.1.mlp.gate.weight"] == "router"


def test_mimo_jangtq_runtime_setup_installs_fused_decode_and_inference_mode(monkeypatch):
    import os

    import jang_tools.jangrt.inference_mode as inference_mode
    import jang_tools.jangrt.jangtq_hydrate as jangtq_hydrate
    import jang_tools.jangrt.switchglu_decode as switchglu_decode
    from jang_tools.mimo_v2.mlx_model import prepare_mimo_jangtq_runtime

    calls = []
    monkeypatch.delenv("JANGTQ_GATHER_OPT", raising=False)

    def fake_hydrate(model, weights, *, mxtq_seed):
        calls.append(("gather_opt_at_hydrate", os.environ.get("JANGTQ_GATHER_OPT")))
        calls.append(("hydrate", model, dict(weights), mxtq_seed))
        return {"regular.weight": "regular"}

    def fake_install():
        calls.append(("install",))
        return True

    def fake_inference(model, *, label):
        calls.append(("inference", model, label))
        return {"training_modules_remaining": 0, "eval_called": True}

    monkeypatch.setattr(jangtq_hydrate, "hydrate_jangtq", fake_hydrate)
    monkeypatch.setattr(switchglu_decode, "install_switchglu_fused_decode", fake_install)
    monkeypatch.setattr(inference_mode, "ensure_inference_mode", fake_inference)

    model = object()
    weights = {
        "model.layers.1.mlp.switch_mlp.gate_proj.tq_packed": "packed",
        "model.layers.1.mlp.switch_mlp.gate_proj.tq_norms": "norms",
        "model.layers.1.mlp.switch_mlp.gate_proj.tq_bits": "bits",
    }

    out = prepare_mimo_jangtq_runtime(model, weights, mxtq_seed=7)

    assert calls[0] == ("gather_opt_at_hydrate", "16")
    assert calls[1][0] == "hydrate"
    assert calls[1][3] == 7
    assert calls[2] == ("install",)
    assert calls[3] == ("inference", model, "MiMo-JANGTQ")
    assert out["regular.weight"] == "regular"
    assert out["model.layers.1.mlp.switch_mlp.gate_proj.packed"] == "packed"
    assert out["model.layers.1.mlp.switch_mlp.gate_proj.norms"] == "norms"


def test_mimo_jangtq_runtime_respects_existing_gather_opt(monkeypatch):
    import os

    import jang_tools.jangrt.inference_mode as inference_mode
    import jang_tools.jangrt.jangtq_hydrate as jangtq_hydrate
    import jang_tools.jangrt.switchglu_decode as switchglu_decode
    from jang_tools.mimo_v2.mlx_model import prepare_mimo_jangtq_runtime

    calls = []
    monkeypatch.setenv("JANGTQ_GATHER_OPT", "24")
    monkeypatch.setattr(jangtq_hydrate, "hydrate_jangtq", lambda model, weights, *, mxtq_seed: calls.append(os.environ["JANGTQ_GATHER_OPT"]) or {})
    monkeypatch.setattr(switchglu_decode, "install_switchglu_fused_decode", lambda: True)
    monkeypatch.setattr(inference_mode, "ensure_inference_mode", lambda model, *, label: {})

    prepare_mimo_jangtq_runtime(object(), {}, mxtq_seed=7)

    assert calls == ["24"]


def test_mimo_moe_uses_weighted_jangtq_decode_when_available():
    from jang_tools.mimo_v2.mlx_model import maybe_weighted_jangtq_decode

    calls = []

    class Switch:
        training = False

        def _jangtq_weighted_decode(self, x, indices, scores):
            calls.append((x, indices, scores))
            return "weighted"

    out = maybe_weighted_jangtq_decode(Switch(), "x", "idx", "scores", batch=1)

    assert out == "weighted"
    assert calls == [("x", "idx", "scores")]


def test_mimo_moe_weighted_decode_falls_back_for_prefill_or_missing_method():
    from jang_tools.mimo_v2.mlx_model import maybe_weighted_jangtq_decode

    class Switch:
        training = False

        def _jangtq_weighted_decode(self, x, indices, scores):
            raise AssertionError("prefill must not use decode-only weighted path")

    assert maybe_weighted_jangtq_decode(Switch(), "x", "idx", "scores", batch=2) is None
    assert maybe_weighted_jangtq_decode(object(), "x", "idx", "scores", batch=1) is None


def test_mimo_gate_uses_compiled_decode_router(monkeypatch):
    import jang_tools.mimo_v2.mlx_model as mlx_model

    calls = []

    def fake_get(top_k, norm_topk_prob, routed_scaling):
        calls.append((top_k, norm_topk_prob, routed_scaling))

        def router(x_fp32, weight_fp32, bias_fp32):
            return "idx", "weights"

        return router

    monkeypatch.setattr(mlx_model, "get_compiled_mimo_decode_router", fake_get)

    assert mlx_model.run_compiled_mimo_decode_router(
        "x",
        "weight",
        "bias",
        top_k=8,
        norm_topk_prob=True,
        routed_scaling=1.0,
        batch=1,
    ) == ("idx", "weights")
    assert calls == [(8, True, 1.0)]


def test_mimo_gate_compiled_decode_router_falls_back_for_prefill():
    from jang_tools.mimo_v2.mlx_model import run_compiled_mimo_decode_router

    assert run_compiled_mimo_decode_router(
        "x",
        "weight",
        "bias",
        top_k=8,
        norm_topk_prob=True,
        routed_scaling=1.0,
        batch=2,
    ) is None


def test_jangtq_decode_profiler_routes_mimo_to_mimo_loader():
    from jang_tools.jangtq_decode_profiler import loader_kind_for_model_info

    assert loader_kind_for_model_info({"model_type": "mimo_v2"}) == "mimo-v2-runtime"


def test_mimo_runtime_quantizes_lm_head_when_requested():
    from jang_tools.mimo_v2.runtime import quantize_lm_head

    class FakeQuantizedLinear:
        @classmethod
        def from_linear(cls, linear, *, group_size, bits, mode):
            class Quantized:
                def __init__(self):
                    self.linear = linear
                    self.group_size = group_size
                    self.bits = bits
                    self.mode = mode

                def parameters(self):
                    return "params"

            return Quantized()

    class Model:
        lm_head = "linear"

    seen = []

    model = Model()
    info = quantize_lm_head(
        model,
        bits=8,
        group_size=64,
        quantized_linear_cls=FakeQuantizedLinear,
        eval_fn=seen.append,
    )

    assert info["enabled"] is True
    assert info["bits"] == 8
    assert info["group_size"] == 64
    assert seen == ["params"]
    assert model.lm_head.linear == "linear"
    assert model.lm_head.mode == "affine"


def test_mimo_runtime_loader_exposes_optional_lm_head_quantization():
    src = (Path(__file__).resolve().parents[1] / "jang_tools" / "mimo_v2" / "runtime.py").read_text()

    assert "def load(" in src
    assert "quantize_lm_head_bits" in src
    assert "quantize_lm_head_group_size" in src
    assert "lm_head_quantization" in src
    assert "mlx_lm.utils" in src


def test_mimo_attention_uses_builtin_sink_sdpa_with_manual_fallback():
    src = (Path(__file__).resolve().parents[1] / "jang_tools" / "mimo_v2" / "mlx_model.py").read_text()

    assert "sinks=self.attention_sink_bias" in src
    assert "JANG_MIMO_MANUAL_SINK_SDPA" in src
    assert "_sdpa_with_sink" in src


def test_mimo_slim_profile_reduces_bookends_without_lowering_routed_experts():
    from jang_tools.mimo_v2.convert_jang import QuantProfile, classify

    p2l = QuantProfile.parse("2")
    assert p2l.name == "JANG_2L"
    assert p2l.routed_expert_bits == {"gate_proj": 4, "up_proj": 2, "down_proj": 3}
    assert p2l.critical_bits == 8
    assert p2l.important_bits == 6
    assert p2l.compress_bits == 2
    assert classify("model.layers.1.mlp.experts.0.gate_proj.weight", p2l) == (4, "affine", 128)
    assert classify("model.layers.1.mlp.experts.0.up_proj.weight", p2l) == (2, "affine", 128)
    assert classify("model.layers.1.mlp.experts.0.down_proj.weight", p2l) == (3, "affine", 128)

    p2e4 = QuantProfile.parse("2e4")
    assert p2e4.name == "JANG_2L_E4"
    assert p2e4.routed_expert_bits == {"gate_proj": 2, "up_proj": 2, "down_proj": 2}
    assert classify("model.layers.1.mlp.experts.0.gate_proj.weight", p2e4) == (4, "affine", 128)
    assert classify("model.layers.4.mlp.experts.0.down_proj.weight", p2e4) == (4, "affine", 128)
    assert classify("model.layers.5.mlp.experts.0.gate_proj.weight", p2e4) == (2, "affine", 128)
    assert p2e4.expert_layer_bits is not None
    assert sorted(p2e4.expert_layer_bits) == [1, 2, 3, 4]

    p2e8 = QuantProfile.parse("2e8")
    assert p2e8.name == "JANG_2L_E8"
    assert classify("model.layers.8.mlp.experts.0.gate_proj.weight", p2e8) == (4, "affine", 128)
    assert classify("model.layers.9.mlp.experts.0.gate_proj.weight", p2e8) == (2, "affine", 128)

    p2c4l4 = QuantProfile.parse("2c4l4")
    assert p2c4l4.name == "JANG_2L_C4L4"
    assert classify("model.layers.1.mlp.experts.0.gate_proj.weight", p2c4l4) == (8, "affine", 128)
    assert classify("model.layers.4.mlp.experts.0.down_proj.weight", p2c4l4) == (8, "affine", 128)
    assert classify("model.layers.5.mlp.experts.0.gate_proj.weight", p2c4l4) == (2, "affine", 128)
    assert classify("model.layers.43.mlp.experts.0.gate_proj.weight", p2c4l4) == (2, "affine", 128)
    assert classify("model.layers.44.mlp.experts.0.up_proj.weight", p2c4l4) == (4, "affine", 128)
    assert classify("model.layers.47.mlp.experts.0.down_proj.weight", p2c4l4) == (4, "affine", 128)
    assert p2c4l4.expert_layer_bits is not None
    assert sorted(p2c4l4.expert_layer_bits) == [1, 2, 3, 4, 44, 45, 46, 47]

    profile = QuantProfile.parse("2s")

    assert profile.name == "JANG_2S"
    assert profile.default_bits == 6
    assert profile.routed_expert_bits == {"gate_proj": 4, "up_proj": 2, "down_proj": 3}
    assert classify("model.layers.1.mlp.experts.0.gate_proj.weight", profile) == (4, "affine", 128)
    assert classify("model.layers.1.mlp.experts.0.up_proj.weight", profile) == (2, "affine", 128)
    assert classify("model.layers.1.mlp.experts.0.down_proj.weight", profile) == (3, "affine", 128)
    assert classify("model.layers.1.self_attn.qkv_proj.weight", profile) == (6, "affine", 64)
    assert classify("model.embed_tokens.weight", profile) == (6, "affine", 64)
    assert classify("lm_head.weight", profile) == (6, "affine", 64)
    assert classify("model.layers.1.self_attn.o_proj.weight", profile) == (8, "affine", 64)
    assert classify("visual.blocks.0.attn.qkv.weight", profile) == (16, "passthrough_bf16", 0)
    assert classify(
        "audio_encoder.input_local_transformer.layers.0.self_attn.q_proj.weight",
        profile,
    ) == (16, "passthrough_bf16", 0)


def test_mimo_candidate_profiles_have_unambiguous_projection_bits():
    from jang_tools.mimo_v2.convert_jang import QuantProfile, classify

    p2g32 = QuantProfile.parse("2g32")
    assert p2g32.name == "JANG_2L_G32"
    assert classify("model.layers.1.mlp.experts.0.gate_proj.weight", p2g32) == (4, "affine", 32)
    assert classify("model.layers.1.mlp.experts.0.up_proj.weight", p2g32) == (2, "affine", 32)
    assert classify("model.layers.1.mlp.experts.0.down_proj.weight", p2g32) == (3, "affine", 32)
    assert classify("model.layers.1.self_attn.qkv_proj.weight", p2g32) == (8, "affine", 64)
    assert classify("model.embed_tokens.weight", p2g32) == (8, "affine", 64)
    assert classify("lm_head.weight", p2g32) == (8, "affine", 64)

    p422 = QuantProfile.parse("422")
    assert p422.routed_expert_bits == {"gate_proj": 4, "up_proj": 2, "down_proj": 2}
    assert classify("model.layers.1.mlp.experts.0.gate_proj.weight", p422) == (4, "affine", 128)
    assert classify("model.layers.1.mlp.experts.0.up_proj.weight", p422) == (2, "affine", 128)
    assert classify("model.layers.1.mlp.experts.0.down_proj.weight", p422) == (2, "affine", 128)

    p322 = QuantProfile.parse("322")
    assert p322.name == "JANG_2L_322"
    assert p322.routed_expert_bits == {"gate_proj": 3, "up_proj": 2, "down_proj": 2}
    assert classify("model.layers.1.mlp.experts.0.gate_proj.weight", p322) == (3, "affine", 128)
    assert classify("model.layers.1.mlp.experts.0.up_proj.weight", p322) == (2, "affine", 128)
    assert classify("model.layers.1.mlp.experts.0.down_proj.weight", p322) == (2, "affine", 128)

    p322g64 = QuantProfile.parse("322g64")
    assert p322g64.name == "JANG_2L_322_G64"
    assert p322g64.routed_expert_bits == {"gate_proj": 3, "up_proj": 2, "down_proj": 2}
    assert p322g64.expert_group_size == 64
    assert classify("model.layers.1.mlp.experts.0.gate_proj.weight", p322g64) == (3, "affine", 64)
    assert classify("model.layers.1.mlp.experts.0.up_proj.weight", p322g64) == (2, "affine", 64)
    assert classify("model.layers.1.mlp.experts.0.down_proj.weight", p322g64) == (2, "affine", 64)

    p322d3e16 = QuantProfile.parse("322d3e16")
    assert p322d3e16.name == "JANG_2L_322_D3E16"
    assert p322d3e16.routed_expert_bits == {"gate_proj": 3, "up_proj": 2, "down_proj": 2}
    assert p322d3e16.expert_layer_bits is not None
    assert classify("model.layers.1.mlp.experts.0.down_proj.weight", p322d3e16) == (3, "affine", 128)
    assert classify("model.layers.16.mlp.experts.0.down_proj.weight", p322d3e16) == (3, "affine", 128)
    assert classify("model.layers.17.mlp.experts.0.down_proj.weight", p322d3e16) == (2, "affine", 128)
    assert classify("model.layers.16.mlp.experts.0.gate_proj.weight", p322d3e16) == (3, "affine", 128)
    assert classify("model.layers.16.mlp.experts.0.up_proj.weight", p322d3e16) == (2, "affine", 128)

    p323 = QuantProfile.parse("323")
    assert p323.name == "JANG_2L_323"
    assert p323.routed_expert_bits == {"gate_proj": 3, "up_proj": 2, "down_proj": 3}
    assert classify("model.layers.1.mlp.experts.0.gate_proj.weight", p323) == (3, "affine", 128)
    assert classify("model.layers.1.mlp.experts.0.up_proj.weight", p323) == (2, "affine", 128)
    assert classify("model.layers.1.mlp.experts.0.down_proj.weight", p323) == (3, "affine", 128)

    p242 = QuantProfile.parse("242")
    assert p242.routed_expert_bits == {"gate_proj": 2, "up_proj": 4, "down_proj": 2}
    assert classify("model.layers.1.mlp.experts.0.gate_proj.weight", p242) == (2, "affine", 128)
    assert classify("model.layers.1.mlp.experts.0.up_proj.weight", p242) == (4, "affine", 128)
    assert classify("model.layers.1.mlp.experts.0.down_proj.weight", p242) == (2, "affine", 128)

    p333 = QuantProfile.parse("333")
    assert p333.name == "JANG_3E"
    assert p333.routed_expert_bits == {"gate_proj": 3, "up_proj": 3, "down_proj": 3}
    assert classify("model.layers.1.mlp.experts.0.gate_proj.weight", p333) == (3, "affine", 128)
    assert classify("model.layers.1.mlp.experts.0.up_proj.weight", p333) == (3, "affine", 128)
    assert classify("model.layers.1.mlp.experts.0.down_proj.weight", p333) == (3, "affine", 128)

    p233 = QuantProfile.parse("233")
    assert p233.name == "JANG_233"
    assert p233.routed_expert_bits == {"gate_proj": 2, "up_proj": 3, "down_proj": 3}
    assert classify("model.layers.1.mlp.experts.0.gate_proj.weight", p233) == (2, "affine", 128)
    assert classify("model.layers.1.mlp.experts.0.up_proj.weight", p233) == (3, "affine", 128)
    assert classify("model.layers.1.mlp.experts.0.down_proj.weight", p233) == (3, "affine", 128)


def test_mimo_layer_override_profile_stamps_runtime_quant_plan(tmp_path):
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
    (src / "tokenizer_config.json").write_text(json.dumps({"chat_template": "x"}))

    profile = QuantProfile.parse("2c4l4")
    _write_config_json(src, dst, profile, {
        "model.layers.1.mlp.switch_mlp.gate_proj": {"bits": 8, "group_size": 128, "mode": "affine"},
        "model.layers.5.mlp.switch_mlp.gate_proj": {"bits": 2, "group_size": 128, "mode": "affine"},
        "model.layers.47.mlp.switch_mlp.down_proj": {"bits": 4, "group_size": 128, "mode": "affine"},
    }, include_mtp=False)

    cfg = json.loads((dst / "config.json").read_text())
    assert cfg["jang_profile"] == "JANG_2L_C4L4"
    assert cfg["runtime"]["mtp_mode"] == "absent"
    assert cfg["routed_expert_bit_plan"]["default"] == {"gate_proj": 2, "up_proj": 2, "down_proj": 2}
    assert cfg["routed_expert_bit_plan"]["layer_overrides"]["1"] == {
        "gate_proj": 8,
        "up_proj": 8,
        "down_proj": 8,
    }
    assert cfg["routed_expert_bit_plan"]["layer_overrides"]["47"] == {
        "gate_proj": 4,
        "up_proj": 4,
        "down_proj": 4,
    }
    assert cfg["quantization"]["model.layers.1.mlp.switch_mlp.gate_proj"]["bits"] == 8
    assert cfg["quantization"]["model.layers.5.mlp.switch_mlp.gate_proj"]["bits"] == 2
    assert cfg["quantization"]["model.layers.47.mlp.switch_mlp.down_proj"]["bits"] == 4


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
