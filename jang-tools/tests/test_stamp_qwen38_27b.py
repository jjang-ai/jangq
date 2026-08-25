from __future__ import annotations

import json
from pathlib import Path

import pytest

from jang_tools.capabilities import verify_directory
from jang_tools.stamp_qwen38_27b import resolve_mtp_tuning_depth, stamp


def _bundle(tmp_path: Path, *, video: bool = True) -> Path:
    bundle = tmp_path / "Qwen3.8-27B-JANG_4D"
    bundle.mkdir()
    (bundle / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5",
        "vision_config": {"hidden_size": 1152},
        "quantization": {"mode": "affine", "bits": 4},
    }))
    (bundle / "jang_config.json").write_text("{}\n")
    (bundle / "generation_config.json").write_text("{}\n")
    (bundle / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"mtp.fc.weight": "model-00001-of-00001.safetensors"},
    }))
    if video:
        (bundle / "video_preprocessor_config.json").write_text("{}\n")
    return bundle


def _validated_tuning(depth: int = 2) -> dict:
    speeds = {"1": 46.24, "2": 56.26, "3": 30.14}
    return {
        "best_depth": depth,
        "blocked": False,
        "validated": True,
        "baseline_tok_s": speeds["1"],
        "best_tok_s": speeds[str(depth)],
        "speedup_vs_baseline": speeds[str(depth)] / speeds["1"],
        "measured_tok_s_by_depth": speeds,
    }


def test_validated_wall_speed_drives_runtime_and_bundle_depth(tmp_path):
    bundle = _bundle(tmp_path)
    tuning_path = bundle / "vmlx_mtp_tuning.json"
    tuning_path.write_text(json.dumps(_validated_tuning()) + "\n")
    original_tuning = tuning_path.read_bytes()

    result = stamp(bundle)

    jang = json.loads((bundle / "jang_config.json").read_text())
    assert result["mtp_depth"] == 2
    assert result["mtp_depth_basis"] == "validated measured D2"
    assert jang["mtp"]["recommended_num_drafts"] == 2
    assert "recommended 2 draft(s)/step" in jang["runtime"]["mtp_status"]
    assert jang["capabilities"]["tool_parser"] == "qwen3_coder"
    assert jang["capabilities"]["family"] == "qwen3_5"
    assert jang["capabilities"]["modality"] == "multimodal"
    assert tuning_path.read_bytes() == original_tuning
    assert verify_directory(bundle)[0] is True


def test_unvalidated_deeper_depth_is_rejected_before_any_bundle_write(tmp_path):
    bundle = _bundle(tmp_path)
    tuning_path = bundle / "vmlx_mtp_tuning.json"
    tuning_path.write_text(json.dumps({
        "best_depth": 3,
        "blocked": False,
        "validated": False,
    }))
    before = {
        name: (bundle / name).read_bytes()
        for name in ("config.json", "jang_config.json", "generation_config.json")
    }

    with pytest.raises(ValueError, match="unvalidated.*depth 1"):
        stamp(bundle)

    assert all((bundle / name).read_bytes() == payload for name, payload in before.items())


def test_claimed_d3_is_rejected_when_its_speed_table_selects_d2(tmp_path):
    tuning = _validated_tuning(depth=3)

    with pytest.raises(ValueError, match="best_depth=3 contradicts.*D2 is fastest"):
        resolve_mtp_tuning_depth(tuning)


def test_deeper_depth_requires_wall_speedup_above_baseline():
    tuning = _validated_tuning(depth=2)
    tuning["speedup_vs_baseline"] = 1.0

    with pytest.raises(
        ValueError, match="D2/D3 speedup_vs_baseline must exceed 1.0"
    ):
        resolve_mtp_tuning_depth(tuning)


def test_missing_tuning_gets_atomic_conservative_d1_seed(tmp_path):
    bundle = _bundle(tmp_path, video=False)

    result = stamp(bundle)

    tuning = json.loads((bundle / "vmlx_mtp_tuning.json").read_text())
    jang = json.loads((bundle / "jang_config.json").read_text())
    assert result["mtp_depth"] == 1
    assert tuning["best_depth"] == 1
    assert tuning.get("validated") is None
    assert jang["mtp"]["recommended_num_drafts"] == 1
    assert jang["capabilities"]["modality"] == "vision"
    assert verify_directory(bundle)[0] is True
