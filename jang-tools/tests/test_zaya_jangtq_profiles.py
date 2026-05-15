"""ZAYA/ZAYA1-VL JANGTQ profile metadata and source-scan tests."""

import numpy as np
from safetensors.numpy import save_file

from jang_tools.convert_zaya_common import (
    PROFILE_BITS,
    ZAYA_JANGTQ_K_BITS,
    scan_source,
)
from jang_tools.convert_zaya1_vl_common import zaya1_vl_capabilities


def test_zaya_jangtq_k_profile_is_mixed_4_2_2():
    assert PROFILE_BITS["JANGTQ_K"] == "mixed"
    assert ZAYA_JANGTQ_K_BITS == {
        "gate_proj": 2,
        "up_proj": 2,
        "down_proj": 4,
    }


def test_zaya_scan_preserves_vl_source_expert_key(tmp_path):
    shard = tmp_path / "model-00001-of-00001.safetensors"
    fc1 = "model.layers.3.mlp.zaya_block.experts.local_experts.2.linear_fc1.weight"
    fc2 = "model.layers.3.mlp.zaya_block.experts.local_experts.2.linear_fc2.weight"
    save_file(
        {
            fc1: np.zeros((4, 4), dtype=np.float32),
            fc2: np.zeros((2, 4), dtype=np.float32),
            "model.layers.3.input_layernorm.weight": np.zeros((4,), dtype=np.float32),
        },
        str(shard),
    )

    regular, experts = scan_source(tmp_path)

    assert regular[0][0] == "model.layers.3.input_layernorm.weight"
    assert experts[(3, 2)]["linear_fc1"][2] == fc1
    assert experts[(3, 2)]["linear_fc2"][2] == fc2


def test_zaya1_vl_capabilities_match_text_reasoning_policy():
    caps = zaya1_vl_capabilities()
    assert caps["family"] == "zaya1_vl"
    assert caps["modality"] == "vision"
    assert caps["supports_thinking"] is True
