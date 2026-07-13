import json
from argparse import Namespace

import pytest

from jang_tools.expert_lab_vmlx import (
    ExpertTraceContext,
    MOELayerHook,
    build_eval_sidecars,
    _generation_settings,
    _expected_moe_layer_count,
    _mask_application_issue,
    _patch_qwen_sparse_moe,
    _qwen_sparse_moe_hook_targets,
    _token_trace_evidence_issue,
    _trace_coverage_issue,
    _validate_mask_targets,
    _vmlx_qwen_config_issue,
)


def test_expected_moe_layer_count_reads_nested_qwen36_text_config(tmp_path):
    model = tmp_path / "qwen36"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "model_type": "qwen3_5_moe_text",
                    "num_hidden_layers": 40,
                    "num_experts": 256,
                    "num_experts_per_tok": 8,
                    "moe_intermediate_size": 512,
                },
            }
        ),
        encoding="utf-8",
    )

    assert _expected_moe_layer_count(model) == 40


def test_vmlx_qwen_config_issue_rejects_compact_fixture_without_shared_expert(tmp_path):
    model = tmp_path / "qwen-moe-bf16"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "vocab_size": 264,
                "num_experts": 64,
                "num_experts_per_tok": 4,
                "moe_intermediate_size": 64,
            }
        ),
        encoding="utf-8",
    )

    issue = _vmlx_qwen_config_issue(model)

    assert issue is not None
    assert "shared_expert_intermediate_size" in issue
    assert "compact UI fixture" in issue


def test_vmlx_qwen_config_issue_accepts_nested_qwen36_text_config(tmp_path):
    model = tmp_path / "qwen36"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "model_type": "qwen3_5_moe_text",
                    "hidden_size": 2048,
                    "num_hidden_layers": 40,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 2,
                    "vocab_size": 248320,
                    "num_experts": 256,
                    "num_experts_per_tok": 8,
                    "moe_intermediate_size": 512,
                    "shared_expert_intermediate_size": 512,
                },
            }
        ),
        encoding="utf-8",
    )

    assert _vmlx_qwen_config_issue(model) is None


def test_qwen35_moe_sparse_block_is_a_patched_trace_target():
    qwen3_next = pytest.importorskip("mlx_lm.models.qwen3_next")
    qwen3_5 = pytest.importorskip("mlx_lm.models.qwen3_5")

    targets = _qwen_sparse_moe_hook_targets()
    assert qwen3_next.Qwen3NextSparseMoeBlock in targets
    assert qwen3_5.SparseMoeBlock in targets

    originals = {target: target.__call__ for target in targets}
    try:
        import jang_tools.expert_lab_vmlx as runner

        runner._PATCHED_QWEN_SPARSE_MOE = False
        _patch_qwen_sparse_moe()
        patched = qwen3_next.Qwen3NextSparseMoeBlock.__call__
        assert patched is not originals[qwen3_next.Qwen3NextSparseMoeBlock]
        assert qwen3_5.SparseMoeBlock.__call__ is patched
    finally:
        for target, original in originals.items():
            target.__call__ = original
        runner._PATCHED_QWEN_SPARSE_MOE = False


def test_validate_mask_targets_rejects_unknown_moe_layer():
    hooks = {0: MOELayerHook(layer=0, num_experts=256, trained_top_k=8)}

    with pytest.raises(ValueError, match="unknown MoE layer 1"):
        _validate_mask_targets({1: {0}}, None, hooks)


def test_validate_mask_targets_rejects_expert_outside_layer_width():
    hooks = {0: MOELayerHook(layer=0, num_experts=256, trained_top_k=8)}

    with pytest.raises(ValueError, match="outside layer 0 width 256"):
        _validate_mask_targets({0: {256}}, None, hooks)


def test_validate_mask_targets_rejects_masks_that_leave_less_than_topk():
    hooks = {0: MOELayerHook(layer=0, num_experts=8, trained_top_k=4)}

    with pytest.raises(RuntimeError, match="leaves fewer than top-k experts"):
        _validate_mask_targets({0: {0, 1, 2, 3, 4}}, None, hooks)


def test_trace_coverage_issue_requires_every_hooked_layer():
    hooks = {
        0: MOELayerHook(layer=0, num_experts=8, trained_top_k=2),
        1: MOELayerHook(layer=1, num_experts=8, trained_top_k=2),
    }
    context = ExpertTraceContext()
    context.record(
        layer=0,
        sequence_offset=0,
        selected=[[0, 1]],
        scores=[[0.7, 0.3]],
        disabled=[],
        effective_top_k=2,
    )

    assert "layers: 1" in (_trace_coverage_issue(context, hooks) or "")


def test_mask_application_issue_detects_disabled_expert_selection():
    context = ExpertTraceContext(disabled_by_layer={0: {1}})
    context.record(
        layer=0,
        sequence_offset=0,
        selected=[[0, 1]],
        scores=[[0.7, 0.3]],
        disabled=[1],
        effective_top_k=2,
    )

    assert "disabled experts were selected: 1" in (
        _mask_application_issue(context, {0: {1}}) or ""
    )


def test_token_trace_evidence_issue_requires_trace_for_masked_generation():
    context = ExpertTraceContext(disabled_by_layer={0: {3}}, emit_token_trace=False)
    context.record(
        layer=0,
        sequence_offset=0,
        selected=[[1, 2]],
        scores=[[0.7, 0.3]],
        disabled=[3],
        effective_top_k=2,
    )

    issue = _token_trace_evidence_issue(
        context,
        emit_token_trace=False,
        disabled_by_layer={0: {3}},
    )

    assert "requires token_trace routing evidence" in (issue or "")


def test_token_trace_evidence_issue_rejects_truncated_trace():
    context = ExpertTraceContext(max_trace_tokens=1)
    context.record(
        layer=0,
        sequence_offset=0,
        selected=[[1, 2], [2, 4]],
        scores=[[0.7, 0.3], [0.6, 0.4]],
        disabled=[],
        effective_top_k=2,
    )

    issue = _token_trace_evidence_issue(
        context,
        emit_token_trace=True,
        disabled_by_layer={},
    )

    assert "token_trace covers 1 of 2 routed layer-token records" in (issue or "")


def test_generation_settings_record_prompt_overrides_and_defaults():
    args = Namespace(max_tokens=64, temperature=0.0, top_p=1.0, top_k=0)

    overridden = _generation_settings(
        {"id": "long", "max_new_tokens": "96", "temperature": "0.2"},
        args,
        4,
    )

    assert overridden == {
        "max_tokens": 96,
        "temperature": 0.2,
        "top_p": 1.0,
        "top_k": 0,
        "top_k_override": 4,
        "prompt_max_new_tokens": 96,
        "prompt_temperature": 0.2,
    }

    defaults = _generation_settings({"id": "default"}, args, None)
    assert defaults == {
        "max_tokens": 64,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "top_k_override": None,
        "prompt_max_new_tokens": None,
        "prompt_temperature": None,
    }


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )


def _semantic_suite_rows():
    return [
        {
            "id": "math_arithmetic",
            "domain": "math",
            "text": "Return only the number: 17 + 25.",
            "tags": ["math"],
        },
        {
            "id": "code_swift",
            "domain": "code",
            "text": "Name a Swift collection type.",
            "tags": ["code", "swift"],
        },
        {
            "id": "format_json",
            "domain": "formatting",
            "text": "Return JSON with key ok and value true.",
            "tags": ["formatting", "json"],
        },
        {
            "id": "instruction_following",
            "domain": "instruction-following",
            "text": "Answer with exactly one word.",
            "tags": ["instruction-following"],
        },
        {
            "id": "reasoning_logic",
            "domain": "reasoning",
            "text": "If all dax are len and one len is red, what follows?",
            "tags": ["reasoning"],
        },
        {
            "id": "safety_medical",
            "domain": "safety",
            "text": "Give safe general guidance for a medical concern.",
            "tags": ["safety", "medical", "legal"],
        },
        {
            "id": "chinese_translation",
            "domain": "language",
            "text": "Translate hello to Chinese.",
            "tags": ["multilingual", "chinese", "translation"],
        },
        {
            "id": "english_dominant",
            "domain": "english_dominant",
            "text": "Respond in English.",
            "tags": ["english-dominant"],
        },
        {
            "id": "unknown_language_role",
            "domain": "unknown_language_role",
            "text": "Identify the language role.",
            "tags": ["unknown-language-role"],
        },
    ]


def _runtime_info(masked=False, missing_key=None):
    runtime = {
        "backend": "vmlx",
        "runtime_mode": "bf16_vmlx",
        "device_name": "Device(gpu, 0)",
        "runtime_metal_enabled": True,
        "jang_tools_version": "2.5.31",
        "mlx_version": "0.31.2",
        "mlx_lm_version": "0.31.3",
        "mlx_vlm_version": None,
        "source_model_path": "/models/qwen36",
        "hooked_moe_layers": 2,
        "expected_moe_layers": 2,
        "hook_coverage_complete": True,
        "mask_applied": masked,
        "masked_layer_count": 1 if masked else 0,
        "disabled_expert_count": 1 if masked else 0,
        "top_k_override": None,
    }
    if missing_key:
        runtime.pop(missing_key)
    return runtime


def _layer_stats(leak_disabled=False, partial=False):
    layer0_hits = {"3": 1} if leak_disabled else {"1": 1}
    rows = [
        {
            "layer": 0,
            "token_count": 1,
            "hit_counts": layer0_hits,
            "probability_mass": {key: 1.0 for key in layer0_hits},
        },
        {
            "layer": 1,
            "token_count": 1,
            "hit_counts": {"2": 1},
            "probability_mass": {"2": 1.0},
        },
    ]
    return rows[:1] if partial else rows


def _token_trace(masked=False, leak_disabled=False, truncated=False):
    selected = [3, 1] if leak_disabled else [1, 2]
    rows = [
        {
            "token_index": 0,
            "layer": 0,
            "selected_experts": selected,
            "scores": [0.7, 0.3],
            "disabled_experts": [3] if masked else [],
            "effective_top_k": 2,
            "entropy": 0.61,
        },
        {
            "token_index": 0,
            "layer": 1,
            "selected_experts": [2, 4],
            "scores": [0.6, 0.4],
            "disabled_experts": [],
            "effective_top_k": 2,
            "entropy": 0.67,
        },
    ]
    return rows[:1] if truncated else rows


def _test_generation_settings(prompt, overrides=None):
    max_tokens = int(prompt.get("max_new_tokens") or 64)
    temperature = float(prompt["temperature"]) if "temperature" in prompt else 0.0
    settings = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "top_k": 0,
        "top_k_override": None,
        "prompt_max_new_tokens": max_tokens if "max_new_tokens" in prompt else None,
        "prompt_temperature": temperature if "temperature" in prompt else None,
    }
    if overrides:
        settings.update(overrides)
    return settings


def _generation_row(
    prompt,
    index,
    *,
    masked=False,
    text="ok",
    partial_layer_stats=False,
    leak_stats=False,
    leak_trace=False,
    truncated_trace=False,
    missing_runtime_key=None,
    include_generation_settings=True,
    generation_settings_overrides=None,
):
    result = {
        "text": text,
        "tokens": 4,
        "elapsed_seconds": 1.0,
        "tokens_per_second": 4.0,
        "finish_reason": "max_tokens",
        "layer_stats": _layer_stats(
            leak_disabled=leak_stats,
            partial=partial_layer_stats,
        ),
        "token_trace": _token_trace(
            masked=masked,
            leak_disabled=leak_trace,
            truncated=truncated_trace,
        ),
        "runtime_info": _runtime_info(masked=masked, missing_key=missing_runtime_key),
    }
    if include_generation_settings:
        result["generation_settings"] = _test_generation_settings(
            prompt,
            generation_settings_overrides,
        )
    return {
        "schema": "jang-expert-lab-vmlx-generation-v1",
        "prompt_index": index,
        "prompt": {**prompt, "prompt": prompt["text"]},
        "result": result,
    }


def _write_sidecar_inputs(tmp_path, suite_rows=None, baseline_mutator=None, masked_mutator=None):
    suite_rows = suite_rows or _semantic_suite_rows()
    suite = tmp_path / "suite.jsonl"
    baseline = tmp_path / "baseline" / "generations.jsonl"
    masked = tmp_path / "masked" / "generations.jsonl"
    mask = tmp_path / "mask.json"
    baseline_rows = [
        _generation_row(prompt, index, masked=False)
        for index, prompt in enumerate(suite_rows)
    ]
    masked_rows = [
        _generation_row(prompt, index, masked=True)
        for index, prompt in enumerate(suite_rows)
    ]
    if baseline_mutator is not None:
        baseline_mutator(baseline_rows)
    if masked_mutator is not None:
        masked_mutator(masked_rows)
    _write_jsonl(suite, suite_rows)
    _write_jsonl(baseline, baseline_rows)
    _write_jsonl(masked, masked_rows)
    mask.write_text(
        json.dumps({"disabled_by_layer": {"0": [3]}}),
        encoding="utf-8",
    )
    return suite, baseline, masked, mask


def test_build_eval_sidecars_writes_complete_bf16_vmlx_artifacts(tmp_path):
    suite, baseline, masked, mask = _write_sidecar_inputs(tmp_path)

    summary = build_eval_sidecars(
        suite_path=suite,
        baseline_generations_path=baseline,
        masked_generations_path=masked,
        mask_path=mask,
        output_dir=tmp_path / "evidence",
    )

    assert summary["ok"] is True
    assert summary["prompt_count"] == 9
    assert summary["baseline_route_record_count"] == 18
    assert summary["masked_route_record_count"] == 18
    assert summary["eval_trace_record_count"] == 36
    assert summary["missing_semantic_coverage"] == []
    assert summary["risky_prompt_ids"] == []
    assert summary["generation_settings_checked"] is True

    index = json.loads((tmp_path / "evidence" / "eval_index.json").read_text(encoding="utf-8"))
    assert index["runtime_mode"] == "bf16_vmlx"
    assert index["runtime_backend"] == "vmlx"
    assert index["runtime_metal_enabled"] is True
    assert index["hooked_moe_layers"] == 2
    assert index["expected_moe_layers"] == 2
    assert index["mask_applied"] is True
    assert index["disabled_expert_count"] == 1
    assert index["baseline_layer_stats_prompt_count"] == 9
    assert index["masked_layer_stats_prompt_count"] == 9
    assert index["eval_jsonl"] == "eval.jsonl"
    assert index["eval_trace_jsonl"] == "eval_trace.jsonl"
    assert index["suite_jsonl"] == "suite.jsonl"
    assert index["generation_settings_checked"] is True

    eval_rows = [
        json.loads(line)
        for line in (tmp_path / "evidence" / "eval.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert eval_rows[0]["baselineGenerationSettings"] == {
        "max_tokens": 64,
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
    }
    assert eval_rows[0]["maskedGenerationSettings"] == eval_rows[0]["baselineGenerationSettings"]

    comparison = json.loads(
        (tmp_path / "evidence" / "comparison_summary.json").read_text(encoding="utf-8")
    )
    assert comparison["baselineQualifiedPromptCount"] == 0
    assert comparison["safeDropCandidates"] == []


def test_build_eval_sidecars_rejects_prompt_order_mismatch(tmp_path):
    def swap_masked(rows):
        rows[0], rows[1] = rows[1], rows[0]

    suite, baseline, masked, mask = _write_sidecar_inputs(tmp_path, masked_mutator=swap_masked)

    with pytest.raises(ValueError, match="masked generations prompt order does not match"):
        build_eval_sidecars(
            suite_path=suite,
            baseline_generations_path=baseline,
            masked_generations_path=masked,
            mask_path=mask,
            output_dir=tmp_path / "evidence",
        )


def test_build_eval_sidecars_rejects_missing_decode_settings(tmp_path):
    def strip_settings(rows):
        rows[0]["result"].pop("generation_settings")

    suite, baseline, masked, mask = _write_sidecar_inputs(tmp_path, masked_mutator=strip_settings)

    with pytest.raises(ValueError, match="masked prompt math_arithmetic is missing decode settings evidence"):
        build_eval_sidecars(
            suite_path=suite,
            baseline_generations_path=baseline,
            masked_generations_path=masked,
            mask_path=mask,
            output_dir=tmp_path / "evidence",
        )


def test_build_eval_sidecars_rejects_baseline_masked_decode_setting_mismatch(tmp_path):
    def change_masked_temperature(rows):
        rows[0]["result"]["generation_settings"]["temperature"] = 0.4

    suite, baseline, masked, mask = _write_sidecar_inputs(
        tmp_path,
        masked_mutator=change_masked_temperature,
    )

    with pytest.raises(
        ValueError,
        match="prompt math_arithmetic baseline/masked generation temperature does not match",
    ):
        build_eval_sidecars(
            suite_path=suite,
            baseline_generations_path=baseline,
            masked_generations_path=masked,
            mask_path=mask,
            output_dir=tmp_path / "evidence",
        )


def test_build_eval_sidecars_rejects_decode_settings_that_do_not_match_suite(tmp_path):
    suite_rows = _semantic_suite_rows()
    suite_rows[0] = {
        **suite_rows[0],
        "max_new_tokens": 12,
        "temperature": 0.2,
    }

    def change_tokens(rows):
        rows[0]["result"]["generation_settings"]["max_tokens"] = 11

    suite, baseline, masked, mask = _write_sidecar_inputs(
        tmp_path,
        suite_rows=suite_rows,
        baseline_mutator=change_tokens,
        masked_mutator=change_tokens,
    )

    with pytest.raises(
        ValueError,
        match="prompt math_arithmetic baseline generation max_tokens does not match suite.jsonl",
    ):
        build_eval_sidecars(
            suite_path=suite,
            baseline_generations_path=baseline,
            masked_generations_path=masked,
            mask_path=mask,
            output_dir=tmp_path / "evidence",
        )


def test_build_eval_sidecars_rejects_missing_semantic_coverage(tmp_path):
    sparse_suite = _semantic_suite_rows()[:2]
    suite, baseline, masked, mask = _write_sidecar_inputs(tmp_path, suite_rows=sparse_suite)

    with pytest.raises(ValueError, match="missing required semantic prompt probes"):
        build_eval_sidecars(
            suite_path=suite,
            baseline_generations_path=baseline,
            masked_generations_path=masked,
            mask_path=mask,
            output_dir=tmp_path / "evidence",
        )


def test_build_eval_sidecars_rejects_partial_layer_stats(tmp_path):
    def partial_first(rows):
        rows[0] = _generation_row(
            _semantic_suite_rows()[0],
            0,
            masked=True,
            partial_layer_stats=True,
        )

    suite, baseline, masked, mask = _write_sidecar_inputs(tmp_path, masked_mutator=partial_first)

    with pytest.raises(ValueError, match="routed-layer stats cover 1 of 2 layers"):
        build_eval_sidecars(
            suite_path=suite,
            baseline_generations_path=baseline,
            masked_generations_path=masked,
            mask_path=mask,
            output_dir=tmp_path / "evidence",
        )


def test_build_eval_sidecars_rejects_truncated_token_trace(tmp_path):
    def truncate_first(rows):
        rows[0] = _generation_row(
            _semantic_suite_rows()[0],
            0,
            masked=True,
            truncated_trace=True,
        )

    suite, baseline, masked, mask = _write_sidecar_inputs(tmp_path, masked_mutator=truncate_first)

    with pytest.raises(ValueError, match="token_trace has 1 rows for 2 routed layer-token records"):
        build_eval_sidecars(
            suite_path=suite,
            baseline_generations_path=baseline,
            masked_generations_path=masked,
            mask_path=mask,
            output_dir=tmp_path / "evidence",
        )


def test_build_eval_sidecars_rejects_disabled_expert_leakage(tmp_path):
    def leak_first(rows):
        rows[0] = _generation_row(
            _semantic_suite_rows()[0],
            0,
            masked=True,
            leak_stats=True,
        )

    suite, baseline, masked, mask = _write_sidecar_inputs(tmp_path, masked_mutator=leak_first)

    with pytest.raises(ValueError, match="disabled experts leaked into masked layer 0"):
        build_eval_sidecars(
            suite_path=suite,
            baseline_generations_path=baseline,
            masked_generations_path=masked,
            mask_path=mask,
            output_dir=tmp_path / "evidence",
        )


def test_build_eval_sidecars_rejects_missing_vmlx_runtime_metadata(tmp_path):
    def strip_runtime_version(rows):
        rows[0] = _generation_row(
            _semantic_suite_rows()[0],
            0,
            masked=True,
            missing_runtime_key="mlx_lm_version",
        )

    suite, baseline, masked, mask = _write_sidecar_inputs(
        tmp_path,
        masked_mutator=strip_runtime_version,
    )

    with pytest.raises(ValueError, match="missing vMLX package version evidence"):
        build_eval_sidecars(
            suite_path=suite,
            baseline_generations_path=baseline,
            masked_generations_path=masked,
            mask_path=mask,
            output_dir=tmp_path / "evidence",
        )
