import json
import os
import subprocess
import sys
import struct
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


def _env() -> dict[str, str]:
    env = dict(os.environ)
    root = Path(__file__).resolve().parents[1]
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(root) if not existing else os.pathsep.join([str(root), existing])
    return env


def _make_qwen_moe_source(tmp_path: Path) -> Path:
    src = tmp_path / "qwen-src"
    src.mkdir()
    (src / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 1,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "hidden_size": 2,
            "moe_intermediate_size": 2,
        },
    }))
    (src / "tokenizer.json").write_text("{}")

    tensors = {
        "model.language_model.layers.0.mlp.gate.weight": np.asarray(
            [[0, 0], [1, 0], [3, 0], [2, 0]], dtype=np.float32
        ),
        "model.language_model.layers.0.mlp.experts.gate_up_proj": np.arange(
            16, dtype=np.float32
        ).reshape(4, 2, 2),
        "model.language_model.layers.0.mlp.experts.down_proj": np.arange(
            100, 124, dtype=np.float32
        ).reshape(4, 3, 2),
        "model.language_model.layers.0.mlp.shared_expert.up_proj.weight": np.asarray(
            [[7, 8], [9, 10]], dtype=np.float32
        ),
    }
    shard = src / "model-00001-of-00001.safetensors"
    save_file(tensors, str(shard))
    (src / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": shard.stat().st_size},
        "weight_map": {key: shard.name for key in tensors},
    }))
    return src


def _bf16_bytes(values) -> bytes:
    arr = np.asarray(values, dtype=np.float32)
    u16 = (arr.view(np.uint32) >> 16).astype("<u2")
    return u16.tobytes()


def _write_raw_safetensors(path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> None:
    header = {}
    chunks = []
    cursor = 0
    for name, (dtype, shape, payload) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [cursor, cursor + len(payload)],
        }
        chunks.append(payload)
        cursor += len(payload)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        for chunk in chunks:
            fh.write(chunk)


def _make_bf16_qwen36_like_source(tmp_path: Path) -> tuple[Path, dict[int, dict[str, np.ndarray]]]:
    src = tmp_path / "qwen36-bf16-src"
    src.mkdir()
    (src / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 2,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "hidden_size": 2,
            "moe_intermediate_size": 3,
            "dtype": "bfloat16",
        },
    }))
    (src / "tokenizer.json").write_text("{}")

    layer_values: dict[int, dict[str, np.ndarray]] = {}
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    for layer, router_scores in {
        0: [0.1, 3.0, 0.2, 2.0],
        1: [4.0, 1.0, 0.0, 3.0],
    }.items():
        router = np.asarray([[score, 0.0] for score in router_scores], dtype=np.float32)
        gate_up = (np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2) + layer * 100)
        down = (np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3) + layer * 200)
        layer_values[layer] = {"router": router, "gate_up": gate_up, "down": down}
        prefix = f"model.language_model.layers.{layer}.mlp"
        tensors[f"{prefix}.gate.weight"] = ("BF16", router.shape, _bf16_bytes(router))
        tensors[f"{prefix}.experts.gate_up_proj"] = ("BF16", gate_up.shape, _bf16_bytes(gate_up))
        tensors[f"{prefix}.experts.down_proj"] = ("BF16", down.shape, _bf16_bytes(down))

    shard = src / "model-00001-of-00001.safetensors"
    _write_raw_safetensors(shard, tensors)
    (src / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": shard.stat().st_size},
        "weight_map": {key: shard.name for key in tensors},
    }))
    return src, layer_values


def test_prequant_prune_qwen_moe_slices_source_before_quant(tmp_path):
    src = _make_qwen_moe_source(tmp_path)
    dst = tmp_path / "qwen-pruned"

    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "jang_tools",
            "--quiet-text",
            "prequant-prune-qwen-moe",
            str(src),
            str(dst),
            "--keep-experts",
            "2",
            "--json",
        ],
        env=_env(),
        capture_output=True,
        text=True,
        check=True,
    )
    summary = json.loads(r.stdout)
    assert summary["stage"] == "pre_quantization_bf16_source"
    assert summary["source_num_experts"] == 4
    assert summary["num_experts"] == 2

    config = json.loads((dst / "config.json").read_text())
    assert config["text_config"]["num_experts"] == 2
    assert config["jang_prequant_expert_pruning"]["source_num_experts"] == 4
    assert (dst / "tokenizer.json").exists()

    manifest = json.loads((dst / "expert_prune_manifest.json").read_text())
    assert manifest["layers"]["0"]["keep"] == [2, 3]
    assert manifest["layers"]["0"]["drop"] == [0, 1]

    with safe_open(str(dst / "model-00001-of-00001.safetensors"), framework="np") as handle:
        gate = handle.get_tensor("model.language_model.layers.0.mlp.gate.weight")
        gate_up = handle.get_tensor("model.language_model.layers.0.mlp.experts.gate_up_proj")
        down = handle.get_tensor("model.language_model.layers.0.mlp.experts.down_proj")
        shared = handle.get_tensor("model.language_model.layers.0.mlp.shared_expert.up_proj.weight")

    np.testing.assert_array_equal(gate, np.asarray([[3, 0], [2, 0]], dtype=np.float32))
    np.testing.assert_array_equal(gate_up, np.arange(16, dtype=np.float32).reshape(4, 2, 2)[[2, 3]])
    np.testing.assert_array_equal(down, np.arange(100, 124, dtype=np.float32).reshape(4, 3, 2)[[2, 3]])
    np.testing.assert_array_equal(shared, np.asarray([[7, 8], [9, 10]], dtype=np.float32))


def test_prequant_prune_slices_qwen36_bf16_rank3_experts(tmp_path):
    src, values = _make_bf16_qwen36_like_source(tmp_path)
    dst = tmp_path / "qwen36-bf16-pruned"

    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "jang_tools",
            "--quiet-text",
            "prequant-prune-qwen-moe",
            str(src),
            str(dst),
            "--keep-experts",
            "2",
            "--json",
        ],
        env=_env(),
        capture_output=True,
        text=True,
        check=True,
    )
    summary = json.loads(r.stdout)
    assert summary["verification"]["ok"] is True
    assert summary["num_layers"] == 2

    config = json.loads((dst / "config.json").read_text())
    assert config["text_config"]["num_experts"] == 2
    assert config["jang_prequant_expert_pruning"]["num_experts"] == 2

    module = __import__("jang_tools.prequant_prune_qwen_moe", fromlist=["_read_safetensors_header", "_read_tensor_bytes"])
    shard = dst / "model-00001-of-00001.safetensors"
    header, _ = module._read_safetensors_header(shard)
    expected_keeps = {0: [1, 3], 1: [0, 3]}
    for layer, keep in expected_keeps.items():
        prefix = f"model.language_model.layers.{layer}.mlp"
        assert header[f"{prefix}.gate.weight"]["shape"] == [2, 2]
        assert header[f"{prefix}.experts.gate_up_proj"]["shape"] == [2, 3, 2]
        assert header[f"{prefix}.experts.down_proj"]["shape"] == [2, 2, 3]
        assert module._read_tensor_bytes(shard, header[f"{prefix}.gate.weight"]) == _bf16_bytes(values[layer]["router"][keep])
        assert module._read_tensor_bytes(shard, header[f"{prefix}.experts.gate_up_proj"]) == _bf16_bytes(values[layer]["gate_up"][keep])
        assert module._read_tensor_bytes(shard, header[f"{prefix}.experts.down_proj"]) == _bf16_bytes(values[layer]["down"][keep])


def test_prequant_prune_accepts_prompt_trace_keep_map(tmp_path):
    src = _make_qwen_moe_source(tmp_path)
    keep_map = tmp_path / "expert-prune-plan.json"
    keep_map.write_text(json.dumps({
        "schema": "jang-expert-prune-plan-v1",
        "method": "prompt_trace_hits_mass_domain_lift_v1",
        "keepExpertsPerLayer": 2,
        "layers": {
            "0": {
                "keep": [0, 2],
                "drop": [1, 3],
                "evidence": [
                    {"expert": 0, "hits": 10, "probabilityMass": 3.5, "domains": {"coding": 7}, "label": "coding-specialist", "kept": True},
                    {"expert": 2, "hits": 8, "probabilityMass": 2.0, "domains": {"math": 4}, "label": "math-specialist", "kept": True},
                ],
            }
        },
    }))
    dst = tmp_path / "qwen-smart-pruned"

    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "jang_tools",
            "--quiet-text",
            "prequant-prune-qwen-moe",
            str(src),
            str(dst),
            "--keep-map",
            str(keep_map),
            "--json",
        ],
        env=_env(),
        capture_output=True,
        text=True,
        check=True,
    )
    summary = json.loads(r.stdout)
    assert summary["method"] == "prompt_trace_hits_mass_domain_lift_v1"
    assert summary["num_experts"] == 2

    config = json.loads((dst / "config.json").read_text())
    pruning = config["jang_prequant_expert_pruning"]
    assert pruning["method"] == "prompt_trace_hits_mass_domain_lift_v1"
    assert pruning["keep_map"] == str(keep_map.resolve())

    manifest = json.loads((dst / "expert_prune_manifest.json").read_text())
    assert manifest["layers"]["0"]["keep"] == [0, 2]
    assert manifest["layers"]["0"]["drop"] == [1, 3]
    assert manifest["keep_map"] == str(keep_map.resolve())

    with safe_open(str(dst / "model-00001-of-00001.safetensors"), framework="np") as handle:
        gate = handle.get_tensor("model.language_model.layers.0.mlp.gate.weight")
        gate_up = handle.get_tensor("model.language_model.layers.0.mlp.experts.gate_up_proj")
        down = handle.get_tensor("model.language_model.layers.0.mlp.experts.down_proj")

    np.testing.assert_array_equal(gate, np.asarray([[0, 0], [3, 0]], dtype=np.float32))
    np.testing.assert_array_equal(gate_up, np.arange(16, dtype=np.float32).reshape(4, 2, 2)[[0, 2]])
    np.testing.assert_array_equal(down, np.arange(100, 124, dtype=np.float32).reshape(4, 3, 2)[[0, 2]])


def test_prequant_prune_rejects_keep_map_missing_router_layer(tmp_path):
    src = _make_qwen_moe_source(tmp_path)
    keep_map = tmp_path / "bad-plan.json"
    keep_map.write_text(json.dumps({
        "schema": "jang-expert-prune-plan-v1",
        "method": "prompt_trace_hits_mass_domain_lift_v1",
        "layers": {"1": {"keep": [0, 2]}},
    }))
    dst = tmp_path / "bad-map"

    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "jang_tools",
            "--quiet-text",
            "prequant-prune-qwen-moe",
            str(src),
            str(dst),
            "--keep-map",
            str(keep_map),
            "--json",
        ],
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode != 0
    assert "missing router layer 0" in r.stderr


def test_prequant_prune_rejects_keep_below_router_topk(tmp_path):
    src = _make_qwen_moe_source(tmp_path)
    dst = tmp_path / "bad"

    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "jang_tools",
            "--quiet-text",
            "prequant-prune-qwen-moe",
            str(src),
            str(dst),
            "--keep-experts",
            "1",
            "--json",
        ],
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode != 0
    assert "below router top-k" in r.stderr


def test_prequant_prune_rejects_output_inside_or_above_source_tree(tmp_path):
    src = _make_qwen_moe_source(tmp_path)
    nested_output = src / "nested-pruned-output"
    cases = [
        src,
        nested_output,
        tmp_path,
    ]

    for dst in cases:
        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "jang_tools",
                "--quiet-text",
                "prequant-prune-qwen-moe",
                str(src),
                str(dst),
                "--keep-experts",
                "2",
                "--json",
            ],
            env=_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        assert r.returncode != 0, dst
        assert "output directory must be separate from the source model tree" in r.stderr
        assert (src / "config.json").exists()
        assert (src / "model.safetensors.index.json").exists()

    assert not nested_output.exists()


def test_prequant_prune_accepts_intent_plan_flat_keep_lists(tmp_path):
    """jang-intent-prune-plan-v1 uses flat keep lists per layer and scalar trained_top_k."""
    src = _make_qwen_moe_source(tmp_path)
    keep_map = tmp_path / "intent-plan.json"
    keep_map.write_text(json.dumps({
        "schema": "jang-intent-prune-plan-v1",
        "schema_version": 1,
        "scorer": "hybrid_v1",
        "source_model": str(src.resolve()),
        "keep_experts_per_layer": 2,
        "num_experts_source": 4,
        "num_layers": 1,
        "safety_stance": "balanced",
        "suite": {"name": "Reviewed Prune 50", "prompt_count": 50},
        "safety": {
            "passed": True,
            "minimum_active_experts_per_layer": 2,
            "trained_top_k": 1,
            "issues": [],
        },
        "layers": {
            "0": [0, 2],
        },
    }))
    dst = tmp_path / "qwen-intent-pruned"

    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "jang_tools",
            "--quiet-text",
            "prequant-prune-qwen-moe",
            str(src),
            str(dst),
            "--keep-map",
            str(keep_map),
            "--json",
        ],
        env=_env(),
        capture_output=True,
        text=True,
        check=True,
    )
    summary = json.loads(r.stdout)
    assert summary["method"] == "hybrid_v1"
    assert summary["num_experts"] == 2

    manifest = json.loads((dst / "expert_prune_manifest.json").read_text())
    assert manifest["layers"]["0"]["keep"] == [0, 2]

