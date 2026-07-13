import hashlib
import json
import importlib.util
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

MODULE_PATH = Path(__file__).resolve().parents[1] / "jang_tools" / "prequant_prune_qwen_moe.py"
SPEC = importlib.util.spec_from_file_location("prequant_prune_qwen_moe", MODULE_PATH)
prequant_prune_qwen_moe = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(prequant_prune_qwen_moe)
prune_prequant_qwen_moe = prequant_prune_qwen_moe.prune_prequant_qwen_moe
REQUIRED_REVIEWED_PRUNE_SEMANTIC_DOMAINS = sorted(
    prequant_prune_qwen_moe.REQUIRED_REVIEWED_PRUNE_SEMANTIC_DOMAINS
)


def _write_tiny_qwen_moe_source(path):
    shard = path / "model-00001-of-00001.safetensors"
    tensors = {
        "model.language_model.layers.0.mlp.gate.weight": np.array(
            [
                [0.1, 0.0],
                [3.0, 0.0],
                [0.2, 0.0],
                [2.0, 0.0],
            ],
            dtype=np.float16,
        ),
        "model.language_model.layers.0.mlp.experts.gate_up_proj": np.arange(4 * 3, dtype=np.float16).reshape(4, 3),
        "model.language_model.layers.0.mlp.experts.down_proj": np.arange(4 * 2, dtype=np.float16).reshape(4, 2),
    }
    save_file(tensors, str(shard))
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "num_experts": 4,
                "num_experts_per_tok": 2,
            }
        ),
        encoding="utf-8",
    )
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": shard.stat().st_size},
                "weight_map": {key: shard.name for key in tensors},
            }
        ),
        encoding="utf-8",
    )


def _write_tiny_text_qwen_split_moe_source(path):
    shard = path / "model-00001-of-00001.safetensors"
    tensors = {
        "model.layers.0.mlp.gate.weight": np.array(
            [
                [0.1, 0.0],
                [3.0, 0.0],
                [0.2, 0.0],
                [2.0, 0.0],
            ],
            dtype=np.float16,
        ),
        "model.layers.0.mlp.experts.gate_proj.weight": np.arange(4 * 3, dtype=np.float16).reshape(4, 3),
        "model.layers.0.mlp.experts.up_proj.weight": np.arange(4 * 5, dtype=np.float16).reshape(4, 5),
        "model.layers.0.mlp.experts.down_proj.weight": np.arange(4 * 2, dtype=np.float16).reshape(4, 2),
    }
    save_file(tensors, str(shard))
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "num_experts": 4,
                "num_experts_per_tok": 2,
            }
        ),
        encoding="utf-8",
    )
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": shard.stat().st_size},
                "weight_map": {key: shard.name for key in tensors},
            }
        ),
        encoding="utf-8",
    )


def _reviewed_suite_rows(*, include_required_semantic_probes=True):
    rows = []
    for index in range(50):
        prompt_id = f"p{index}"
        if include_required_semantic_probes and index == 0:
            rows.append(
                {
                    "id": prompt_id,
                    "domain": "math",
                    "subdomain": "arithmetic",
                    "prompt": "Return only the number: 17 * 23.",
                    "tags": ["arithmetic"],
                }
            )
        elif include_required_semantic_probes and index == 1:
            rows.append(
                {
                    "id": prompt_id,
                    "domain": "coding",
                    "subdomain": "swift",
                    "prompt": "Write a Swift function that groups strings.",
                    "tags": ["swift", "code"],
                }
            )
        elif include_required_semantic_probes and index == 2:
            rows.append(
                {
                    "id": prompt_id,
                    "domain": "structured",
                    "subdomain": "json",
                    "prompt": "Return JSON fields decision and rationale.",
                    "tags": ["json", "formatting"],
                }
            )
        elif include_required_semantic_probes and index == 3:
            rows.append(
                {
                    "id": prompt_id,
                    "domain": "instruction",
                    "subdomain": "hierarchy",
                    "prompt": "Explain how to follow instruction hierarchy.",
                    "tags": ["instruction-following"],
                }
            )
        elif include_required_semantic_probes and index == 4:
            rows.append(
                {
                    "id": prompt_id,
                    "domain": "reasoning",
                    "subdomain": "logic",
                    "prompt": "If A implies B and B is false, what follows about A?",
                    "tags": ["logic"],
                }
            )
        elif include_required_semantic_probes and index == 5:
            rows.append(
                {
                    "id": prompt_id,
                    "domain": "robustness",
                    "subdomain": "medicine-safety",
                    "prompt": "Explain safe next steps without diagnosing.",
                    "tags": ["safety", "medical", "legal"],
                }
            )
        elif include_required_semantic_probes and index == 6:
            rows.append(
                {
                    "id": prompt_id,
                    "domain": "multilingual",
                    "subdomain": "chinese",
                    "prompt": "Translate this status update into Simplified Chinese: Build succeeded.",
                    "tags": ["chinese", "translation", "non_english"],
                }
            )
        elif include_required_semantic_probes and index == 7:
            rows.append(
                {
                    "id": prompt_id,
                    "domain": "general",
                    "subdomain": "explanation",
                    "prompt": "Classify this prompt as English dominant.",
                    "tags": ["english_dominant"],
                }
            )
        elif include_required_semantic_probes and index == 8:
            rows.append(
                {
                    "id": prompt_id,
                    "domain": "multilingual",
                    "subdomain": "unknown-language-role",
                    "prompt": "Classify whether this text has an unknown language role: Bonjour.",
                    "tags": ["unknown_language_role", "non_english"],
                }
            )
        else:
            rows.append(
                {
                    "id": prompt_id,
                    "domain": "general",
                    "prompt": "Say hello.",
                    "tags": [],
                }
            )
    return rows


def _jsonl_text(rows):
    return "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n"


def _reviewed_suite_sha256(*, include_required_semantic_probes=True):
    return hashlib.sha256(
        _jsonl_text(
            _reviewed_suite_rows(
                include_required_semantic_probes=include_required_semantic_probes
            )
        ).encode("utf-8")
    ).hexdigest()


def _semantic_eval_index_fields():
    return {
        "semantic_coverage": REQUIRED_REVIEWED_PRUNE_SEMANTIC_DOMAINS,
        "missing_semantic_coverage": [],
    }


def _reviewed_keep_map_payload(src):
    return {
        "method": "prompt_trace_hits_mass_domain_lift_v1",
        "source_model": str(src.resolve()),
        "suite_jsonl": "suite.jsonl",
        "suite_sha256": _reviewed_suite_sha256(),
        "keepExpertsPerLayer": 2,
        "promptCount": 50,
        "safety": {
            "passed": True,
            "minimum_active_experts_per_layer": 2,
            "trained_top_k_by_layer": {"0": 2},
            "issues": [],
        },
        "comparison_summary": {
            "promptCount": 50,
            "passRateBaseline": 1.0,
            "passRateMasked": 1.0,
            "meanTextDelta": 0.0,
            "highRiskDomains": [],
            "safeDropCandidates": [{"layer": 0, "expert": 0}],
        },
        "eval_index": {
            "prompt_count": 50,
            "prompt_ids": [f"p{i}" for i in range(50)],
            "risky_prompt_ids": [],
            "high_risk_domains": [],
            "mean_baseline_tokens": 24.0,
            "mean_masked_tokens": 23.5,
            "runtime_mode": "bf16_vmlx",
            "runtime_backend": "vmlx",
            "runtime_device": "Device(gpu, 0)",
            "runtime_metal_enabled": True,
            "jang_tools_version": "2.5.31",
            "mlx_version": "0.31.2",
            "mlx_lm_version": "0.31.3",
            "source_model_path": str(src.resolve()),
            "baseline_route_record_count": 50,
            "masked_route_record_count": 50,
            "hooked_moe_layers": 1,
            "expected_moe_layers": 1,
            "hook_coverage_complete": True,
            "eval_jsonl": "eval.jsonl",
            "eval_trace_jsonl": "expert_lab_eval_trace.jsonl",
            "suite_sha256": _reviewed_suite_sha256(),
            **_semantic_eval_index_fields(),
            "mask": "mask.json",
            "mask_json": "mask.json",
            "mask_applied": True,
            "disabled_expert_count": 1,
        },
        "layers": {"0": {"keep": [1, 3]}},
    }


def _write_jsonl(path, rows):
    path.write_text(_jsonl_text(rows), encoding="utf-8")


def _write_reviewed_sidecars(
    root,
    src,
    *,
    omit_masked_trace=False,
    omit_mask_evidence=False,
    include_required_semantic_probes=True,
    eval_disabled_expert_count=1,
    eval_topk_override=None,
    mask_disabled_by_layer=None,
    trace_disabled_experts=None,
):
    mask_disabled_by_layer = mask_disabled_by_layer or {"0": [0]}
    trace_disabled_experts = [0] if trace_disabled_experts is None else trace_disabled_experts
    suite_rows = _reviewed_suite_rows(
        include_required_semantic_probes=include_required_semantic_probes
    )
    eval_rows = []
    trace_rows = []
    for index in range(50):
        prompt_id = f"p{index}"
        eval_row = {
            "promptID": prompt_id,
            "baselineText": "baseline output",
            "maskedText": "masked output",
            "textDelta": 0.0,
            "baselineTokenCount": 24,
            "maskedTokenCount": 23,
            "baselineRouteRecordCount": 1,
            "maskedRouteRecordCount": 1,
            "runtimeMode": "bf16_vmlx",
            "runtimeBackend": "vmlx",
            "runtimeDevice": "Device(gpu, 0)",
            "runtimeMetalEnabled": True,
            "jangToolsVersion": "2.5.31",
            "mlxVersion": "0.31.2",
            "mlxLMVersion": "0.31.3",
            "sourceModelPath": str(src.resolve()),
            "maskApplied": True,
            "risk": "none",
            "regressionSeverity": "none",
        }
        if eval_disabled_expert_count is not None:
            eval_row["disabledExpertCount"] = eval_disabled_expert_count
        if eval_topk_override is not None:
            eval_row["topKOverride"] = eval_topk_override
        eval_rows.append(eval_row)
        trace_rows.append(
            {
                "promptID": prompt_id,
                "variant": "baseline",
                "record": {"layer": 0, "selectedExperts": [1, 3], "topK": 2},
            }
        )
        if not omit_masked_trace:
            record = {"layer": 0, "selectedExperts": [1, 3], "topK": 2}
            if not omit_mask_evidence:
                record["disabledExperts"] = trace_disabled_experts
                record["disabledExpertCount"] = len(trace_disabled_experts)
            trace_rows.append({"promptID": prompt_id, "variant": "masked", "record": record})
    _write_jsonl(root / "suite.jsonl", suite_rows)
    _write_jsonl(root / "eval.jsonl", eval_rows)
    _write_jsonl(root / "expert_lab_eval_trace.jsonl", trace_rows)
    (root / "mask.json").write_text(
        json.dumps({"disabled_by_layer": mask_disabled_by_layer}),
        encoding="utf-8",
    )


def _assert_reviewed_keep_map_rejected(src, dst, keep_map, expected):
    try:
        prune_prequant_qwen_moe(
            src,
            dst,
            keep_experts=2,
            keep_map=keep_map,
            require_reviewed_comparison=True,
        )
    except RuntimeError as exc:
        assert "same-suite comparison gate" in str(exc)
        assert expected in str(exc)
    else:
        raise AssertionError("invalid reviewed keep-map should be rejected")


def test_prequant_prune_writes_verification_and_provenance_sidecars(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)

    result = prune_prequant_qwen_moe(src, dst, keep_experts=2)

    assert result["verification"]["ok"] is True
    assert (dst / "prune_manifest.json").exists()
    assert (dst / "expert_prune_manifest.json").exists()
    assert (dst / "prune_plan.json").exists()
    assert (dst / "source_fingerprint.json").exists()
    assert (dst / "verification.json").exists()

    verification = json.loads((dst / "verification.json").read_text(encoding="utf-8"))
    assert verification["checks"]["router_rows_match"] is True
    assert verification["checks"]["expert_rows_match"] is True


def test_prequant_prune_rejects_output_inside_source_even_with_force(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)

    for dst in [src, src / "nested-pruned-output"]:
        try:
            prune_prequant_qwen_moe(src, dst, keep_experts=2, force=True)
        except RuntimeError as exc:
            assert "separate from the source model tree" in str(exc)
        else:
            raise AssertionError(f"dangerous prune output was accepted: {dst}")

    assert (src / "model.safetensors.index.json").exists()
    assert not (src / "nested-pruned-output").exists()


def test_prequant_prune_rejects_output_parent_of_source_even_with_force(tmp_path):
    parent = tmp_path / "models"
    src = parent / "src"
    dst = parent
    src.mkdir(parents=True)
    _write_tiny_qwen_moe_source(src)
    sentinel = src / "model.safetensors.index.json"

    try:
        prune_prequant_qwen_moe(src, dst, keep_experts=2, force=True)
    except RuntimeError as exc:
        assert "separate from the source model tree" in str(exc)
    else:
        raise AssertionError("dangerous ancestor output was accepted")

    assert sentinel.exists()


def test_prequant_prune_accepts_text_qwen_split_expert_prefix(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_text_qwen_split_moe_source(src)

    result = prune_prequant_qwen_moe(src, dst, keep_experts=2)

    assert result["verification"]["ok"] is True
    manifest = json.loads((dst / "expert_prune_manifest.json").read_text(encoding="utf-8"))
    assert manifest["shape_update_count"] == 4
    verification = json.loads((dst / "verification.json").read_text(encoding="utf-8"))
    assert verification["checks"]["router_rows_match"] is True
    assert verification["checks"]["expert_rows_match"] is True


def test_prequant_prune_copies_reviewed_keep_map_as_prune_plan(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "reviewed-plan.json"
    _write_reviewed_sidecars(tmp_path, src)
    keep_map.write_text(
        json.dumps(_reviewed_keep_map_payload(src)),
        encoding="utf-8",
    )

    prune_prequant_qwen_moe(
        src,
        dst,
        keep_experts=2,
        keep_map=keep_map,
        require_reviewed_comparison=True,
    )

    copied = json.loads((dst / "prune_plan.json").read_text(encoding="utf-8"))
    assert copied["method"] == "prompt_trace_hits_mass_domain_lift_v1"
    assert copied["layers"]["0"]["keep"] == [1, 3]
    assert copied["suite_jsonl"] == "expert_lab_suite.jsonl"
    assert copied["suite_sha256"] == _reviewed_suite_sha256()
    assert copied["eval_index"]["suite_sha256"] == _reviewed_suite_sha256()
    assert copied["eval_index"]["semantic_coverage"] == REQUIRED_REVIEWED_PRUNE_SEMANTIC_DOMAINS
    assert copied["eval_index"]["missing_semantic_coverage"] == []
    assert copied["eval_index"]["eval_jsonl"] == "expert_lab_eval.jsonl"
    assert copied["eval_index"]["eval_trace_jsonl"] == "expert_lab_eval_trace.jsonl"
    assert copied["eval_index"]["comparison_summary"] == "expert_lab_comparison_summary.json"
    assert copied["eval_index"]["mask"] == "expert_lab_mask.json"
    assert copied["eval_index"]["mask_json"] == "expert_lab_mask.json"
    assert copied["reviewed_evidence_sidecars"]["eval_index"] == "expert_lab_eval_index.json"
    assert copied["reviewed_evidence_sidecars"]["mask_json"] == "expert_lab_mask.json"

    for name in [
        "expert_lab_suite.jsonl",
        "expert_lab_eval.jsonl",
        "expert_lab_eval_trace.jsonl",
        "expert_lab_mask.json",
        "expert_lab_eval_index.json",
        "expert_lab_comparison_summary.json",
        "expert_lab_review_summary.json",
    ]:
        assert (dst / name).exists()

    summary = json.loads((dst / "expert_lab_review_summary.json").read_text(encoding="utf-8"))
    assert summary["same_suite_verification_ready"] is True
    assert summary["review_sidecars_ready"] is True
    assert summary["pruned_suite_verification_ready"] is False
    assert "has not been run" in summary["pruned_suite_verification_issue"]
    assert summary["source_model_path"] == str(src.resolve())
    assert summary["pruned_source"] == str(dst.resolve())
    assert summary["suite_jsonl"] == str((dst / "expert_lab_suite.jsonl").resolve())
    assert summary["suite_sha256"] == _reviewed_suite_sha256()
    assert summary["mask_json"] == str((dst / "expert_lab_mask.json").resolve())
    assert summary["eval_index"] == str((dst / "expert_lab_eval_index.json").resolve())
    assert summary["prompt_count"] == 50

    eval_index = json.loads((dst / "expert_lab_eval_index.json").read_text(encoding="utf-8"))
    assert eval_index["suite_jsonl"] == "expert_lab_suite.jsonl"
    assert eval_index["suite_sha256"] == _reviewed_suite_sha256()
    assert eval_index["eval_jsonl"] == "expert_lab_eval.jsonl"
    assert eval_index["eval_trace_jsonl"] == "expert_lab_eval_trace.jsonl"
    assert eval_index["mask_json"] == "expert_lab_mask.json"

    manifest = json.loads((dst / "expert_prune_manifest.json").read_text(encoding="utf-8"))
    evidence = manifest["reviewed_evidence"]
    assert evidence["review_summary"] == str((dst / "expert_lab_review_summary.json").resolve())
    assert evidence["suite_sha256"] == _reviewed_suite_sha256()
    assert evidence["pruned_suite_verification_ready"] is False


def test_prequant_prune_rejects_reviewed_keep_map_without_runtime_evidence(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "runtime-free-reviewed-plan.json"
    keep_map.write_text(
        json.dumps(
            {
                "method": "prompt_trace_hits_mass_domain_lift_v1",
                "source_model": str(src.resolve()),
                "keepExpertsPerLayer": 2,
                "promptCount": 50,
                "safety": {
                    "passed": True,
                    "minimum_active_experts_per_layer": 2,
                    "trained_top_k_by_layer": {"0": 2},
                    "issues": [],
                },
                "comparison_summary": {
                    "promptCount": 50,
                    "passRateBaseline": 1.0,
                    "passRateMasked": 1.0,
                    "meanTextDelta": 0.0,
                    "highRiskDomains": [],
                    "safeDropCandidates": [{"layer": 0, "expert": 0}],
                },
                    "eval_index": {
                        "prompt_count": 50,
                        "prompt_ids": [f"p{i}" for i in range(50)],
                        "risky_prompt_ids": [],
                        "high_risk_domains": [],
                        **_semantic_eval_index_fields(),
                        "mean_baseline_tokens": 24.0,
                        "mean_masked_tokens": 23.5,
                    },
                "layers": {"0": {"keep": [1, 3]}},
            }
        ),
        encoding="utf-8",
    )

    try:
        prune_prequant_qwen_moe(
            src,
            dst,
            keep_experts=2,
            keep_map=keep_map,
            require_reviewed_comparison=True,
        )
    except RuntimeError as exc:
        assert "same-suite comparison gate" in str(exc)
        assert "runtime device evidence" in str(exc)
    else:
        raise AssertionError("runtime-free reviewed keep-map should be rejected")


def test_prequant_prune_rejects_reviewed_keep_map_without_vmlx_package_versions(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "version-free-reviewed-plan.json"
    plan = _reviewed_keep_map_payload(src)
    plan["eval_index"].pop("mlx_lm_version")
    keep_map.write_text(json.dumps(plan), encoding="utf-8")

    _assert_reviewed_keep_map_rejected(
        src,
        dst,
        keep_map,
        "eval_index is missing vMLX package version evidence",
    )


def test_prequant_prune_rejects_reviewed_keep_map_with_eval_index_source_mismatch(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "eval-source-mismatch-reviewed-plan.json"
    plan = _reviewed_keep_map_payload(src)
    plan["eval_index"]["source_model_path"] = str((tmp_path / "other-source").resolve())
    keep_map.write_text(json.dumps(plan), encoding="utf-8")

    _assert_reviewed_keep_map_rejected(
        src,
        dst,
        keep_map,
        "eval_index source model path does not match reviewed source",
    )


def test_prequant_prune_rejects_reviewed_keep_map_without_complete_route_evidence(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "sparse-route-reviewed-plan.json"
    plan = _reviewed_keep_map_payload(src)
    plan["eval_index"]["masked_route_record_count"] = 49
    keep_map.write_text(json.dumps(plan), encoding="utf-8")

    _assert_reviewed_keep_map_rejected(
        src,
        dst,
        keep_map,
        "eval_index is missing routing record evidence for every indexed prompt",
    )


def test_prequant_prune_rejects_reviewed_keep_map_with_eval_trace_route_count_mismatch(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "mismatched-route-reviewed-plan.json"
    plan = _reviewed_keep_map_payload(src)
    plan["eval_index"]["baseline_route_record_count"] = 51
    _write_reviewed_sidecars(tmp_path, src)
    keep_map.write_text(json.dumps(plan), encoding="utf-8")

    _assert_reviewed_keep_map_rejected(
        src,
        dst,
        keep_map,
        "eval_trace.jsonl has 50 baseline routing records for 51 indexed baseline route records",
    )


def test_prequant_prune_rejects_reviewed_keep_map_without_hook_coverage_evidence(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "hook-free-reviewed-plan.json"
    plan = _reviewed_keep_map_payload(src)
    plan["eval_index"].pop("hooked_moe_layers")
    keep_map.write_text(json.dumps(plan), encoding="utf-8")

    _assert_reviewed_keep_map_rejected(
        src,
        dst,
        keep_map,
        "eval_index is missing vMLX routed-layer hook evidence",
    )


def test_prequant_prune_rejects_reviewed_keep_map_with_incomplete_hook_coverage_flag(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "incomplete-hook-reviewed-plan.json"
    plan = _reviewed_keep_map_payload(src)
    plan["eval_index"]["hook_coverage_complete"] = False
    keep_map.write_text(json.dumps(plan), encoding="utf-8")

    _assert_reviewed_keep_map_rejected(
        src,
        dst,
        keep_map,
        "eval_index recorded incomplete vMLX routed-layer hook coverage",
    )


def test_prequant_prune_rejects_reviewed_keep_map_with_partial_hook_coverage(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "partial-hook-reviewed-plan.json"
    plan = _reviewed_keep_map_payload(src)
    plan["eval_index"]["hooked_moe_layers"] = 1
    plan["eval_index"]["expected_moe_layers"] = 2
    keep_map.write_text(json.dumps(plan), encoding="utf-8")

    _assert_reviewed_keep_map_rejected(
        src,
        dst,
        keep_map,
        "eval_index vMLX hook coverage 1 of 2 config-routed layers",
    )


def test_prequant_prune_rejects_reviewed_keep_map_without_eval_jsonl_sidecar(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "missing-eval-sidecar-reviewed-plan.json"
    plan = _reviewed_keep_map_payload(src)
    (tmp_path / "mask.json").write_text(
        json.dumps({"disabled_by_layer": {"0": [0]}}),
        encoding="utf-8",
    )
    keep_map.write_text(json.dumps(plan), encoding="utf-8")

    _assert_reviewed_keep_map_rejected(
        src,
        dst,
        keep_map,
        "eval.jsonl sidecar is missing",
    )


def test_prequant_prune_rejects_reviewed_keep_map_without_eval_trace_evidence(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "trace-free-reviewed-plan.json"
    plan = _reviewed_keep_map_payload(src)
    plan["eval_index"].pop("eval_trace_jsonl")
    keep_map.write_text(json.dumps(plan), encoding="utf-8")

    _assert_reviewed_keep_map_rejected(
        src,
        dst,
        keep_map,
        "eval_index is missing eval_trace.jsonl evidence",
    )


def test_prequant_prune_rejects_reviewed_keep_map_with_trace_missing_masked_variant(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "missing-masked-trace-reviewed-plan.json"
    plan = _reviewed_keep_map_payload(src)
    _write_reviewed_sidecars(tmp_path, src, omit_masked_trace=True)
    keep_map.write_text(json.dumps(plan), encoding="utf-8")

    _assert_reviewed_keep_map_rejected(
        src,
        dst,
        keep_map,
        "eval_trace.jsonl missing masked routing records for prompt IDs",
    )


def test_prequant_prune_rejects_reviewed_keep_map_with_masked_trace_without_mask_evidence(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "mask-evidence-free-trace-reviewed-plan.json"
    plan = _reviewed_keep_map_payload(src)
    _write_reviewed_sidecars(tmp_path, src, omit_mask_evidence=True)
    keep_map.write_text(json.dumps(plan), encoding="utf-8")

    _assert_reviewed_keep_map_rejected(
        src,
        dst,
        keep_map,
        "eval_trace.jsonl masked routing records are missing mask.json evidence",
    )


def test_prequant_prune_rejects_reviewed_keep_map_with_trace_that_does_not_match_mask_json(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "mask-json-mismatch-reviewed-plan.json"
    plan = _reviewed_keep_map_payload(src)
    _write_reviewed_sidecars(
        tmp_path,
        src,
        mask_disabled_by_layer={"0": [2]},
        trace_disabled_experts=[0],
    )
    keep_map.write_text(json.dumps(plan), encoding="utf-8")

    _assert_reviewed_keep_map_rejected(
        src,
        dst,
        keep_map,
        "eval_trace.jsonl masked routing records are missing mask.json evidence",
    )


def test_prequant_prune_accepts_reviewed_keep_map_with_eval_artifact_sidecars(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    eval_dir = tmp_path / "evals" / "reviewed-mask"
    src.mkdir()
    eval_dir.mkdir(parents=True)
    _write_tiny_qwen_moe_source(src)
    _write_reviewed_sidecars(tmp_path, src)
    (tmp_path / "eval.jsonl").rename(eval_dir / "eval.jsonl")
    (tmp_path / "expert_lab_eval_trace.jsonl").rename(eval_dir / "eval_trace.jsonl")
    (tmp_path / "mask.json").rename(eval_dir / "mask.json")

    keep_map = tmp_path / "swift-style-reviewed-plan.json"
    plan = _reviewed_keep_map_payload(src)
    plan["eval_artifact"] = str(eval_dir)
    plan.pop("suite_jsonl")
    plan["eval_index"].pop("suite_jsonl", None)
    plan["eval_index"].pop("eval_jsonl", None)
    plan["eval_index"].pop("eval_trace_jsonl", None)
    plan["eval_index"].pop("mask", None)
    plan["eval_index"].pop("mask_json", None)
    keep_map.write_text(json.dumps(plan), encoding="utf-8")

    prune_prequant_qwen_moe(
        src,
        dst,
        keep_experts=2,
        keep_map=keep_map,
        require_reviewed_comparison=True,
    )

    copied = json.loads((dst / "prune_plan.json").read_text(encoding="utf-8"))
    assert copied["suite_jsonl"] == "expert_lab_suite.jsonl"
    assert copied["eval_index"]["suite_sha256"] == _reviewed_suite_sha256()
    assert copied["eval_index"]["eval_jsonl"] == "expert_lab_eval.jsonl"
    assert copied["eval_index"]["eval_trace_jsonl"] == "expert_lab_eval_trace.jsonl"
    assert copied["eval_index"]["mask_json"] == "expert_lab_mask.json"
    assert (dst / "expert_lab_mask.json").is_file()


def test_prequant_prune_rejects_reviewed_keep_map_without_suite_sidecar_evidence(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "suite-free-reviewed-plan.json"
    plan = _reviewed_keep_map_payload(src)
    plan.pop("suite_jsonl")
    _write_reviewed_sidecars(tmp_path, src)
    keep_map.write_text(json.dumps(plan), encoding="utf-8")

    _assert_reviewed_keep_map_rejected(
        src,
        dst,
        keep_map,
        "plan is missing suite.jsonl evidence",
    )


def test_prequant_prune_rejects_reviewed_keep_map_without_suite_fingerprint(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "fingerprint-free-reviewed-plan.json"
    plan = _reviewed_keep_map_payload(src)
    plan.pop("suite_sha256")
    plan["eval_index"].pop("suite_sha256")
    _write_reviewed_sidecars(tmp_path, src)
    keep_map.write_text(json.dumps(plan), encoding="utf-8")

    _assert_reviewed_keep_map_rejected(
        src,
        dst,
        keep_map,
        "eval_index is missing suite.jsonl fingerprint evidence",
    )


def test_prequant_prune_rejects_reviewed_keep_map_without_eval_index_semantic_coverage(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "semantic-index-free-reviewed-plan.json"
    plan = _reviewed_keep_map_payload(src)
    plan["eval_index"].pop("semantic_coverage")
    keep_map.write_text(json.dumps(plan), encoding="utf-8")

    _assert_reviewed_keep_map_rejected(
        src,
        dst,
        keep_map,
        "eval_index is missing semantic coverage evidence",
    )


def test_prequant_prune_rejects_reviewed_keep_map_with_missing_eval_index_semantic_probe(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "semantic-index-missing-probe-reviewed-plan.json"
    plan = _reviewed_keep_map_payload(src)
    plan["eval_index"]["semantic_coverage"] = [
        domain
        for domain in REQUIRED_REVIEWED_PRUNE_SEMANTIC_DOMAINS
        if domain != "translation"
    ]
    plan["eval_index"]["missing_semantic_coverage"] = ["translation"]
    keep_map.write_text(json.dumps(plan), encoding="utf-8")

    _assert_reviewed_keep_map_rejected(
        src,
        dst,
        keep_map,
        "eval_index semantic coverage is missing required probes: translation",
    )


def test_prequant_prune_rejects_reviewed_keep_map_without_eval_index_missing_semantic_coverage_evidence(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "missing-semantic-index-free-reviewed-plan.json"
    plan = _reviewed_keep_map_payload(src)
    plan["eval_index"].pop("missing_semantic_coverage")
    keep_map.write_text(json.dumps(plan), encoding="utf-8")

    _assert_reviewed_keep_map_rejected(
        src,
        dst,
        keep_map,
        "eval_index is missing missing-semantic-coverage evidence",
    )


def test_prequant_prune_rejects_reviewed_keep_map_with_recorded_missing_eval_index_semantic_probe(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "recorded-missing-semantic-index-reviewed-plan.json"
    plan = _reviewed_keep_map_payload(src)
    plan["eval_index"]["missing_semantic_coverage"] = ["translation"]
    keep_map.write_text(json.dumps(plan), encoding="utf-8")

    _assert_reviewed_keep_map_rejected(
        src,
        dst,
        keep_map,
        "eval_index records missing semantic prompt probes: translation",
    )


def test_prequant_prune_rejects_reviewed_keep_map_with_suite_fingerprint_drift(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "fingerprint-drift-reviewed-plan.json"
    plan = _reviewed_keep_map_payload(src)
    plan["eval_index"]["suite_sha256"] = "0" * 64
    _write_reviewed_sidecars(tmp_path, src)
    keep_map.write_text(json.dumps(plan), encoding="utf-8")

    _assert_reviewed_keep_map_rejected(
        src,
        dst,
        keep_map,
        "eval_index suite.jsonl fingerprint does not match suite.jsonl",
    )


def test_prequant_prune_rejects_reviewed_keep_map_without_required_semantic_probes(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "semantic-sparse-reviewed-plan.json"
    plan = _reviewed_keep_map_payload(src)
    _write_reviewed_sidecars(tmp_path, src, include_required_semantic_probes=False)
    keep_map.write_text(json.dumps(plan), encoding="utf-8")

    _assert_reviewed_keep_map_rejected(
        src,
        dst,
        keep_map,
        "suite.jsonl is missing required semantic prompt probes",
    )


def test_prequant_prune_rejects_reviewed_keep_map_without_applied_mask_evidence(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "mask-free-reviewed-plan.json"
    plan = _reviewed_keep_map_payload(src)
    plan["eval_index"]["mask_applied"] = False
    keep_map.write_text(json.dumps(plan), encoding="utf-8")

    _assert_reviewed_keep_map_rejected(
        src,
        dst,
        keep_map,
        "eval_index did not record an applied BF16/vMLX mask",
    )


def test_prequant_prune_rejects_reviewed_keep_map_without_disabled_expert_evidence(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "disabled-expert-free-reviewed-plan.json"
    plan = _reviewed_keep_map_payload(src)
    plan["eval_index"].pop("disabled_expert_count")
    keep_map.write_text(json.dumps(plan), encoding="utf-8")

    _assert_reviewed_keep_map_rejected(
        src,
        dst,
        keep_map,
        "eval_index did not record disabled expert evidence",
    )


def test_prequant_prune_rejects_reviewed_keep_map_with_topk_only_mask_evidence(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "topk-only-reviewed-plan.json"
    plan = _reviewed_keep_map_payload(src)
    plan["eval_index"]["disabled_expert_count"] = 0
    plan["eval_index"]["top_k_override"] = 4
    keep_map.write_text(json.dumps(plan), encoding="utf-8")

    _assert_reviewed_keep_map_rejected(
        src,
        dst,
        keep_map,
        "top-k-only comparisons cannot authorize hard pruning",
    )


def test_prequant_prune_rejects_reviewed_keep_map_with_topk_only_eval_rows(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "topk-only-eval-rows-reviewed-plan.json"
    plan = _reviewed_keep_map_payload(src)
    _write_reviewed_sidecars(
        tmp_path,
        src,
        eval_disabled_expert_count=0,
        eval_topk_override=4,
    )
    keep_map.write_text(json.dumps(plan), encoding="utf-8")

    _assert_reviewed_keep_map_rejected(
        src,
        dst,
        keep_map,
        "eval.jsonl is missing per-prompt disabled expert evidence",
    )


def test_prequant_prune_rejects_reviewed_keep_map_without_clean_comparison(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "risky-reviewed-plan.json"
    keep_map.write_text(
        json.dumps(
            {
                "method": "prompt_trace_hits_mass_domain_lift_v1",
                "source_model": str(src.resolve()),
                "keepExpertsPerLayer": 2,
                "promptCount": 50,
                "safety": {
                    "passed": True,
                    "minimum_active_experts_per_layer": 2,
                    "trained_top_k_by_layer": {"0": 2},
                    "issues": [],
                },
                "comparison_summary": {
                    "promptCount": 50,
                    "passRateBaseline": 1.0,
                    "passRateMasked": 1.0,
                    "meanTextDelta": 0.7,
                    "highRiskDomains": ["math"],
                    "safeDropCandidates": [],
                },
                "eval_index": {
                    "prompt_count": 50,
                    "prompt_ids": [f"p{i}" for i in range(50)],
                    "risky_prompt_ids": [],
                    "high_risk_domains": [],
                },
                "layers": {"0": {"keep": [1, 3]}},
            }
        ),
        encoding="utf-8",
    )

    try:
        prune_prequant_qwen_moe(
            src,
            dst,
            keep_experts=2,
            keep_map=keep_map,
            require_reviewed_comparison=True,
        )
    except RuntimeError as exc:
        assert "same-suite comparison gate" in str(exc)
        assert "high-risk domains" in str(exc)
    else:
        raise AssertionError("risky reviewed keep-map should be rejected")


def test_prequant_prune_rejects_reviewed_keep_map_without_clean_eval_index(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "risky-eval-index-plan.json"
    keep_map.write_text(
        json.dumps(
            {
                "method": "prompt_trace_hits_mass_domain_lift_v1",
                "source_model": str(src.resolve()),
                "keepExpertsPerLayer": 2,
                "promptCount": 50,
                "safety": {
                    "passed": True,
                    "minimum_active_experts_per_layer": 2,
                    "trained_top_k_by_layer": {"0": 2},
                    "issues": [],
                },
                "comparison_summary": {
                    "promptCount": 50,
                    "passRateBaseline": 1.0,
                    "passRateMasked": 1.0,
                    "meanTextDelta": 0.0,
                    "highRiskDomains": [],
                    "safeDropCandidates": [{"layer": 0, "expert": 0}],
                },
                    "eval_index": {
                        "prompt_count": 50,
                        "prompt_ids": [f"p{i}" for i in range(50)],
                        "risky_prompt_ids": ["p13"],
                        "high_risk_domains": ["math"],
                        **_semantic_eval_index_fields(),
                    },
                "layers": {"0": {"keep": [1, 3]}},
            }
        ),
        encoding="utf-8",
    )

    try:
        prune_prequant_qwen_moe(
            src,
            dst,
            keep_experts=2,
            keep_map=keep_map,
            require_reviewed_comparison=True,
        )
    except RuntimeError as exc:
        assert "same-suite comparison gate" in str(exc)
        assert "eval_index still has risky prompt IDs" in str(exc)
    else:
        raise AssertionError("risky eval_index reviewed keep-map should be rejected")


def test_prequant_prune_rejects_reviewed_keep_map_with_eval_index_high_risk_domains(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "domain-risk-eval-index-plan.json"
    keep_map.write_text(
        json.dumps(
            {
                "method": "prompt_trace_hits_mass_domain_lift_v1",
                "source_model": str(src.resolve()),
                "keepExpertsPerLayer": 2,
                "promptCount": 50,
                "safety": {
                    "passed": True,
                    "minimum_active_experts_per_layer": 2,
                    "trained_top_k_by_layer": {"0": 2},
                    "issues": [],
                },
                "comparison_summary": {
                    "promptCount": 50,
                    "passRateBaseline": 1.0,
                    "passRateMasked": 1.0,
                    "meanTextDelta": 0.0,
                    "highRiskDomains": [],
                    "safeDropCandidates": [{"layer": 0, "expert": 0}],
                },
                    "eval_index": {
                        "prompt_count": 50,
                        "prompt_ids": [f"p{i}" for i in range(50)],
                        "risky_prompt_ids": [],
                        "high_risk_domains": ["medical"],
                        **_semantic_eval_index_fields(),
                        "mean_baseline_tokens": 24.0,
                        "mean_masked_tokens": 23.5,
                        "runtime_mode": "bf16_vmlx",
                    "runtime_backend": "vmlx",
                    "runtime_device": "Device(gpu, 0)",
                    "runtime_metal_enabled": True,
                },
                "layers": {"0": {"keep": [1, 3]}},
            }
        ),
        encoding="utf-8",
    )

    try:
        prune_prequant_qwen_moe(
            src,
            dst,
            keep_experts=2,
            keep_map=keep_map,
            require_reviewed_comparison=True,
        )
    except RuntimeError as exc:
        assert "same-suite comparison gate" in str(exc)
        assert "eval_index still has high-risk domains" in str(exc)
    else:
        raise AssertionError("domain-risk eval_index reviewed keep-map should be rejected")


def test_prequant_prune_rejects_reviewed_keep_map_without_prompt_id_coverage(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "sparse-eval-index-reviewed-plan.json"
    keep_map.write_text(
        json.dumps(
            {
                "method": "prompt_trace_hits_mass_domain_lift_v1",
                "source_model": str(src.resolve()),
                "keepExpertsPerLayer": 2,
                "promptCount": 50,
                "safety": {
                    "passed": True,
                    "minimum_active_experts_per_layer": 2,
                    "trained_top_k_by_layer": {"0": 2},
                    "issues": [],
                },
                "comparison_summary": {
                    "promptCount": 50,
                    "passRateBaseline": 1.0,
                    "passRateMasked": 1.0,
                    "meanTextDelta": 0.0,
                    "highRiskDomains": [],
                    "safeDropCandidates": [{"layer": 0, "expert": 0}],
                },
                "eval_index": {
                    "prompt_count": 50,
                    "prompt_ids": ["p0"],
                    "risky_prompt_ids": [],
                    "high_risk_domains": [],
                },
                "layers": {"0": {"keep": [1, 3]}},
            }
        ),
        encoding="utf-8",
    )

    try:
        prune_prequant_qwen_moe(
            src,
            dst,
            keep_experts=2,
            keep_map=keep_map,
            require_reviewed_comparison=True,
        )
    except RuntimeError as exc:
        assert "same-suite comparison gate" in str(exc)
        assert "eval_index lists 1 prompt IDs for 50 indexed prompts" in str(exc)
    else:
        raise AssertionError("under-enumerated eval_index reviewed keep-map should be rejected")


def test_prequant_prune_rejects_reviewed_keep_map_with_extra_eval_index_prompts(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "extra-eval-index-prompts-reviewed-plan.json"
    plan = _reviewed_keep_map_payload(src)
    plan["eval_index"]["prompt_count"] = 51
    plan["eval_index"]["prompt_ids"] = [f"p{i}" for i in range(51)]
    keep_map.write_text(json.dumps(plan), encoding="utf-8")

    _assert_reviewed_keep_map_rejected(
        src,
        dst,
        keep_map,
        "eval_index covers 51 of 50 compared prompts",
    )


def test_prequant_prune_rejects_reviewed_keep_map_when_comparison_exceeds_trace_count(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "comparison-over-trace-reviewed-plan.json"
    plan = _reviewed_keep_map_payload(src)
    plan["promptCount"] = 50
    plan["comparison_summary"]["promptCount"] = 51
    keep_map.write_text(json.dumps(plan), encoding="utf-8")

    _assert_reviewed_keep_map_rejected(
        src,
        dst,
        keep_map,
        "comparison summary covers 51 of 50 traced prompts",
    )


def test_prequant_prune_rejects_reviewed_keep_map_with_shallow_generation_depth(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "one-token-reviewed-plan.json"
    keep_map.write_text(
        json.dumps(
            {
                "method": "prompt_trace_hits_mass_domain_lift_v1",
                "source_model": str(src.resolve()),
                "keepExpertsPerLayer": 2,
                "promptCount": 50,
                "safety": {
                    "passed": True,
                    "minimum_active_experts_per_layer": 2,
                    "trained_top_k_by_layer": {"0": 2},
                    "issues": [],
                },
                "comparison_summary": {
                    "promptCount": 50,
                    "passRateBaseline": 1.0,
                    "passRateMasked": 1.0,
                    "meanTextDelta": 0.0,
                    "highRiskDomains": [],
                    "safeDropCandidates": [{"layer": 0, "expert": 0}],
                },
                    "eval_index": {
                        "prompt_count": 50,
                        "prompt_ids": [f"p{i}" for i in range(50)],
                        "risky_prompt_ids": [],
                        "high_risk_domains": [],
                        **_semantic_eval_index_fields(),
                        "mean_baseline_tokens": 1.0,
                        "mean_masked_tokens": 1.0,
                    },
                "layers": {"0": {"keep": [1, 3]}},
            }
        ),
        encoding="utf-8",
    )

    try:
        prune_prequant_qwen_moe(
            src,
            dst,
            keep_experts=2,
            keep_map=keep_map,
            require_reviewed_comparison=True,
        )
    except RuntimeError as exc:
        assert "same-suite comparison gate" in str(exc)
        assert "average generated depth 1.0 tokens is below 8" in str(exc)
    else:
        raise AssertionError("one-token reviewed keep-map should be rejected")


def test_prequant_prune_rejects_reviewed_keep_map_with_source_mismatch(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "wrong-source-reviewed-plan.json"
    keep_map.write_text(
        json.dumps(
            {
                "method": "prompt_trace_hits_mass_domain_lift_v1",
                "source_model": str((tmp_path / "other-source").resolve()),
                "keepExpertsPerLayer": 2,
                "promptCount": 50,
                "safety": {
                    "passed": True,
                    "minimum_active_experts_per_layer": 2,
                    "trained_top_k_by_layer": {"0": 2},
                    "issues": [],
                },
                "comparison_summary": {
                    "promptCount": 50,
                    "passRateBaseline": 1.0,
                    "passRateMasked": 1.0,
                    "meanTextDelta": 0.0,
                    "highRiskDomains": [],
                    "safeDropCandidates": [{"layer": 0, "expert": 0}],
                },
                "eval_index": {
                    "prompt_count": 50,
                    "prompt_ids": [f"p{i}" for i in range(50)],
                    "risky_prompt_ids": [],
                    "high_risk_domains": [],
                    "mean_baseline_tokens": 24.0,
                    "mean_masked_tokens": 23.5,
                    "runtime_mode": "bf16_vmlx",
                    "runtime_backend": "vmlx",
                    "runtime_device": "Device(gpu, 0)",
                    "runtime_metal_enabled": True,
                },
                "layers": {"0": {"keep": [1, 3]}},
            }
        ),
        encoding="utf-8",
    )

    try:
        prune_prequant_qwen_moe(
            src,
            dst,
            keep_experts=2,
            keep_map=keep_map,
            require_reviewed_comparison=True,
        )
    except RuntimeError as exc:
        assert "source gate" in str(exc)
        assert "does not match selected source" in str(exc)
    else:
        raise AssertionError("source-mismatched reviewed keep-map should be rejected")


def test_prequant_prune_rejects_reviewed_keep_map_without_topk_safety(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "unsafe-reviewed-plan.json"
    plan = _reviewed_keep_map_payload(src)
    plan.pop("safety")
    _write_reviewed_sidecars(tmp_path, src)
    keep_map.write_text(
        json.dumps(plan),
        encoding="utf-8",
    )

    try:
        prune_prequant_qwen_moe(
            src,
            dst,
            keep_experts=2,
            keep_map=keep_map,
            require_reviewed_comparison=True,
        )
    except RuntimeError as exc:
        assert "top-k safety gate" in str(exc)
        assert "missing top-k safety evidence" in str(exc)
    else:
        raise AssertionError("reviewed keep-map without safety should be rejected")


def test_prequant_prune_allows_external_keep_map_without_review_gate(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _write_tiny_qwen_moe_source(src)
    keep_map = tmp_path / "external-keep-map.json"
    keep_map.write_text(
        json.dumps(
            {
                "method": "external_reap_keep_map",
                "keepExpertsPerLayer": 2,
                "layers": {"0": {"keep": [1, 3]}},
            }
        ),
        encoding="utf-8",
    )

    result = prune_prequant_qwen_moe(src, dst, keep_experts=2, keep_map=keep_map)

    assert result["verification"]["ok"] is True
