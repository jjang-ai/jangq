import json
import subprocess
import sys

import numpy as np
from safetensors.numpy import save_file


def test_qwen35_jangtq_converts_tiny_split_expert_review_fixture(tmp_path):
    src = tmp_path / "qwen-moe-bf16"
    out = tmp_path / "qwen-moe-bf16-JANGTQ3-review"
    src.mkdir()

    (src / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "num_hidden_layers": 1,
                "num_experts": 64,
                "num_local_experts": 64,
                "num_experts_per_tok": 8,
                "hidden_size": 8,
                "intermediate_size": 8,
                "max_position_embeddings": 1024,
            }
        ),
        encoding="utf-8",
    )
    (src / "tokenizer.json").write_text("{}", encoding="utf-8")
    save_file(
        {
            "model.layers.0.mlp.experts.gate_proj.weight": np.zeros((64, 8, 8), dtype=np.float32),
            "model.layers.0.mlp.gate.weight": np.zeros((64, 8), dtype=np.float32),
        },
        str(src / "model-00001-of-00001.safetensors"),
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jang_tools.convert_qwen35_jangtq",
            str(src),
            str(out),
            "JANGTQ3",
            "--progress=json",
            "--quiet-text",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (out / "tokenizer_config.json").is_file()
    assert (out / "jangtq_runtime.safetensors").is_file()

    index = json.loads((out / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    assert "model.layers.0.mlp.switch_mlp.gate_proj.tq_packed" in weight_map
    assert "model.layers.0.mlp.experts.gate_proj.weight.scales" not in weight_map

    validate = subprocess.run(
        [sys.executable, "-m", "jang_tools", "validate", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert validate.returncode == 0, validate.stdout + validate.stderr
