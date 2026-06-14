import torch

from jang_tools.capabilities import build_capabilities
from jang_tools.step37.convert_jang import _profile_bits, _tensor_group_size
from jang_tools.step37.nvfp4_codec import dequant_nvfp4_modelopt


def test_dequant_nvfp4_modelopt_rank2():
    weight = torch.tensor([[0x21, 0x43, 0x65, 0x87, 0xA9, 0xCB, 0xED, 0x0F]], dtype=torch.uint8)
    scale = torch.ones((1, 1), dtype=torch.float8_e4m3fn)
    scale2 = torch.ones((), dtype=torch.float32)

    out = dequant_nvfp4_modelopt(weight, scale, scale2, out_dtype=torch.float32)

    expected = torch.tensor(
        [[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0]],
        dtype=torch.float32,
    )
    assert torch.equal(out, expected)


def test_dequant_nvfp4_modelopt_rank3_scale2_per_expert():
    weight = torch.zeros((2, 1, 8), dtype=torch.uint8)
    weight[:, :, :] = 0x22
    scale = torch.ones((2, 1, 1), dtype=torch.float8_e4m3fn)
    scale2 = torch.tensor([1.0, 2.0], dtype=torch.float32)

    out = dequant_nvfp4_modelopt(weight, scale, scale2, out_dtype=torch.float32)

    assert out.shape == (2, 1, 16)
    assert torch.all(out[0] == 1.0)
    assert torch.all(out[1] == 2.0)


def test_step37_head_wise_attention_gate_is_protected():
    name = "model.language_model.layers.0.self_attn.g_proj.weight"

    assert _profile_bits(name, "JANG_2L", num_experts=288) == 8
    assert _tensor_group_size(name, (64, 4096), num_experts=288) == 128


def test_step37_capabilities_are_vlm_step3p5_runtime_route():
    caps = build_capabilities(
        {"source_model": {"architecture": "step3p7"}, "has_vision": True},
        {"model_type": "step3p7", "text_config": {"model_type": "step3p5"}, "vision_config": {}},
    )

    assert caps == {
        "reasoning_parser": "qwen3",
        "tool_parser": "step3p5",
        "think_in_template": True,
        "supports_tools": True,
        "supports_thinking": True,
        "family": "step3p7",
        "modality": "vision",
        # additive multimodal tri-state (audio/video support)
        "modalities": {"text": True, "vision": True, "audio": False, "video": False},
        "has_vision": True,
        "has_audio": False,
        "has_video": False,
        "cache_type": "kv",
    }
