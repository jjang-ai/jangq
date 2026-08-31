import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from jang_tools.format.aligned_safetensors import verify_safetensors_alignment
from jang_tools.format.writer import write_jang_v2_model


def test_jang_v2_index_total_size_includes_preflushed_shards(tmp_path):
    preflushed = tmp_path / "model-00001-of-NNNNN.safetensors"
    save_file(
        {"pre.weight": np.zeros((2, 2), dtype=np.float16)},
        str(preflushed),
        metadata={"format": "mlx"},
    )

    write_jang_v2_model(
        output_dir=tmp_path,
        tensors={"tail.weight": np.zeros((4,), dtype=np.uint32)},
        model_config={},
        jang_config={"quantization": {"bit_widths_used": [4], "block_size": 64}},
        preflushed_map={"pre.weight": preflushed.name},
    )

    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    shard_names = set(index["weight_map"].values())
    actual_size = sum((tmp_path / name).stat().st_size for name in shard_names)

    assert index["metadata"]["total_size"] == actual_size
    assert index["weight_map"]["pre.weight"] == "model-00001-of-00002.safetensors"
    assert index["weight_map"]["tail.weight"] == "model-00002-of-00002.safetensors"


def test_jang_v2_writer_emits_manifest_quantization_overrides(tmp_path):
    write_jang_v2_model(
        output_dir=tmp_path,
        tensors={
            "layers.0.self_attn.q_proj.weight": np.zeros((4,), dtype=np.uint32),
            "layers.0.self_attn.q_proj.scales": np.zeros((2,), dtype=np.float16),
            "layers.0.self_attn.q_proj.biases": np.zeros((2,), dtype=np.float16),
            "layers.0.mlp.switch_mlp.down_proj.weight": np.zeros((4,), dtype=np.uint32),
            "layers.0.mlp.switch_mlp.down_proj.scales": np.zeros((2,), dtype=np.float16),
            "layers.0.mlp.switch_mlp.down_proj.biases": np.zeros((2,), dtype=np.float16),
        },
        model_config={},
        jang_config={
            "quantization": {
                "bit_widths_used": [2, 8],
                "block_size": 64,
                "tensor_quantization_manifest": {
                    "layers.0.self_attn.q_proj": {
                        "bits": 8,
                        "group_size": 64,
                    },
                    "layers.0.mlp.switch_mlp.down_proj": {
                        "bits": 2,
                        "group_size": 64,
                        "storage_bits": 1,
                    },
                },
            }
        },
    )

    config = json.loads((tmp_path / "config.json").read_text())
    quant = config["quantization"]

    for shard in tmp_path.glob("model-*.safetensors"):
        tensor_count, unaligned_count = verify_safetensors_alignment(shard)
        assert tensor_count > 0
        assert unaligned_count == 0

    assert quant["bits"] == 2
    assert quant["group_size"] == 64
    assert quant["mode"] == "affine"
    assert quant["layers.0.self_attn.q_proj"] == {
        "bits": 8,
        "group_size": 64,
        "mode": "affine",
    }
    assert quant["layers.0.mlp.switch_mlp.down_proj"] == {
        "bits": 2,
        "group_size": 64,
        "mode": "affine",
        "storage_bits": 1,
    }


def test_jang_v2_writer_rejects_invalid_storage_bits(tmp_path):
    with pytest.raises(ValueError, match="Invalid storage bit width 7"):
        write_jang_v2_model(
            output_dir=tmp_path,
            tensors={"layers.0.mlp.up_proj.weight": np.zeros((4,), dtype=np.uint32)},
            model_config={},
            jang_config={
                "quantization": {
                    "bit_widths_used": [2],
                    "block_size": 64,
                    "tensor_quantization_manifest": {
                        "layers.0.mlp.up_proj": {
                            "bits": 2,
                            "group_size": 64,
                            "storage_bits": 7,
                        }
                    },
                }
            },
        )
