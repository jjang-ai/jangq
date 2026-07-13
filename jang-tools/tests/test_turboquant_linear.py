import json
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file

pytest.importorskip("mlx.core")

from jang_tools.turboquant.linear import tq_quantize_weight
from jang_tools.build_jangtq_sidecar import main as build_jangtq_sidecar


def test_tq_quantize_weight_rowwise_packs_2048_wide_three_bit_matrix():
    rng = np.random.default_rng(123)
    weight = rng.standard_normal((2, 2048), dtype=np.float32)

    result = tq_quantize_weight(weight, bits=3, seed=42)

    assert result["packed"].shape == (2, 205)
    assert result["packed"].dtype == np.uint32
    assert result["norms"].shape == (2,)


def test_sidecar_uses_tq_in_features_instead_of_padded_width(tmp_path: Path):
    shard = tmp_path / "model-00001-of-00001.safetensors"
    tensors = {
        "layer.switch_mlp.gate_proj.tq_packed": np.zeros((2, 205), dtype=np.uint32),
        "layer.switch_mlp.gate_proj.tq_norms": np.ones((2,), dtype=np.float16),
        "layer.switch_mlp.gate_proj.tq_bits": np.array([3], dtype=np.uint8),
        "layer.switch_mlp.gate_proj.tq_in_features": np.array([2048], dtype=np.int32),
    }
    save_file(tensors, str(shard))
    weight_map = {key: shard.name for key in tensors}
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"format": "jangtq"}, "weight_map": weight_map})
    )
    (tmp_path / "jang_config.json").write_text(json.dumps({"mxtq_seed": 42}))

    saved_argv = sys.argv
    sys.argv = ["build_jangtq_sidecar", str(tmp_path)]
    try:
        build_jangtq_sidecar()
    finally:
        sys.argv = saved_argv

    with safe_open(str(tmp_path / "jangtq_runtime.safetensors"), framework="numpy") as f:
        keys = set(f.keys())

    assert "signs.2048.42" in keys
    assert "codebook.2048.3" in keys
    assert "signs.2050.42" not in keys
    assert "codebook.2050.3" not in keys
