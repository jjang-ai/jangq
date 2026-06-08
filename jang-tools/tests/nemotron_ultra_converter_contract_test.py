from pathlib import Path

from jang_tools.convert_nemotron_ultra_jangtq import classify_tensor, planned_output_keys


def test_ultra_smallest_coherent_policy_tq_only_routed_expert_weights():
    routed = "backbone.layers.1.mixer.experts.0.up_proj.weight"
    gate = "backbone.layers.1.mixer.gate.weight"
    latent = "backbone.layers.1.mixer.fc1_latent_proj.weight"
    attn = "backbone.layers.7.mixer.q_proj.weight"
    mamba = "backbone.layers.0.mixer.in_proj.weight"
    embed = "backbone.embeddings.weight"
    mtp = "mtp.0.layers.0.mixer.q_proj.weight"

    assert classify_tensor(routed, "U8") == ("routed_tq", 1)
    assert classify_tensor(gate, "F32") == ("passthrough", 32)
    assert classify_tensor(latent, "BF16") == ("passthrough", 16)
    assert classify_tensor(attn, "BF16") == ("passthrough", 16)
    assert classify_tensor(mamba, "F8_E4M3") == ("fp8_dequant_affine", 8)
    assert classify_tensor(embed, "BF16") == ("passthrough", 16)
    assert classify_tensor(mtp, "BF16") == ("drop_mtp", 0)


def test_ultra_routed_tq_output_keys_use_prestacked_nemotron_h_switch_mlp_path():
    keys = planned_output_keys("backbone.layers.1.mixer.experts.0.down_proj.weight", "routed_tq")

    assert keys == [
        "backbone.layers.1.mixer.switch_mlp.fc2.tq_packed",
        "backbone.layers.1.mixer.switch_mlp.fc2.tq_norms",
        "backbone.layers.1.mixer.switch_mlp.fc2.tq_bits",
    ]


def test_ultra_converter_stamps_vmlx_capability_contract():
    src = Path(__file__).resolve().parents[1] / "jang_tools" / "convert_nemotron_ultra_jangtq.py"
    text = src.read_text()

    assert '"family": "nemotron_h"' in text
    assert '"cache_type": "hybrid"' in text
    assert '"modality": "text"' in text
    assert '"reasoning_parser": "deepseek_r1"' in text
    assert '"tool_parser": "nemotron"' in text
    assert '"think_in_template": True' in text
