import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import mlx.core as mx

from examples.nemotron_ultra import run_runtime_candidate_suite as candidate_suite
from examples.nemotron_ultra.agent_handoff_report import _build_result as handoff_result
from examples.nemotron_ultra.component_budget_matrix import _build_result as component_matrix_result
from examples.nemotron_ultra.component_budget_matrix import _render as component_matrix_render
from examples.nemotron_ultra.compare_runtime_speed_logs import _compare_result as compare_result
from examples.nemotron_ultra.compare_runtime_speed_logs import _render as compare_render
from examples.nemotron_ultra.experiment_result_check import _build_result as check_result
from examples.nemotron_ultra.host_runtime_readiness import _parse_df as host_parse_df
from examples.nemotron_ultra.host_runtime_readiness import _parse_memory_pressure as host_parse_memory_pressure
from examples.nemotron_ultra.host_runtime_readiness import _parse_ps as host_parse_ps
from examples.nemotron_ultra.host_runtime_readiness import _parse_vm_stat as host_parse_vm_stat
from examples.nemotron_ultra.host_runtime_readiness import _render as host_render
from examples.nemotron_ultra.host_cleanup_runbook import _parse_ps as cleanup_parse_ps
from examples.nemotron_ultra.host_cleanup_runbook import _render as cleanup_render
from examples.nemotron_ultra.runtime_issue_ledger import _build_result as issue_ledger_result
from examples.nemotron_ultra.runtime_issue_ledger import _render as issue_ledger_render
from examples.nemotron_ultra.runtime_speed_gate import _gate as speed_gate
from examples.nemotron_ultra.runtime_speed_gate import _gate_result as speed_gate_result
from examples.nemotron_ultra.runtime_speed_fix_acceptance import _build_result as speed_acceptance_result
from examples.nemotron_ultra.runtime_speed_fix_acceptance import _render as speed_acceptance_render
from examples.nemotron_ultra.runtime_experiment_queue import _build_result as queue_result
from examples.nemotron_ultra.runtime_patch_spec import _build_result as patch_spec_result
from examples.nemotron_ultra.runtime_patch_spec import _render as patch_spec_render
from examples.nemotron_ultra.runtime_candidate_preflight import _build_result as preflight_result
from examples.nemotron_ultra.runtime_candidate_preflight import _render as preflight_render
from examples.nemotron_ultra.runtime_candidate_index import _build_result as candidate_index_result
from examples.nemotron_ultra.runtime_candidate_index import _render as candidate_index_render
from examples.nemotron_ultra.runtime_candidate_launch_guard import _build_result as launch_guard_result
from examples.nemotron_ultra.runtime_candidate_launch_guard import _render as launch_guard_render
from examples.nemotron_ultra.runtime_cache_parser_contract import _build_result as cache_parser_contract_result
from examples.nemotron_ultra.runtime_cache_parser_contract import _render as cache_parser_contract_render
from examples.nemotron_ultra.runtime_cleanup_ready_check import _build_result as cleanup_ready_result
from examples.nemotron_ultra.runtime_cleanup_ready_check import _render as cleanup_ready_render
from examples.nemotron_ultra.runtime_moe_candidate_contract import _build_result as moe_contract_result
from examples.nemotron_ultra.runtime_moe_candidate_contract import _render as moe_contract_render
from examples.nemotron_ultra.runtime_moe_execution_ticket import _build_result as moe_ticket_result
from examples.nemotron_ultra.runtime_moe_execution_ticket import _render as moe_ticket_render
from examples.nemotron_ultra.runtime_moe_delta_contract import _build_result as moe_delta_result
from examples.nemotron_ultra.runtime_moe_delta_contract import _render as moe_delta_render
from examples.nemotron_ultra.runtime_moe_patch_plan import _build_result as moe_patch_plan_result
from examples.nemotron_ultra.runtime_moe_patch_plan import _render as moe_patch_plan_render
from examples.nemotron_ultra.runtime_moe_surface_map import _build_result as moe_surface_result
from examples.nemotron_ultra.runtime_moe_surface_map import _render as moe_surface_render
from examples.nemotron_ultra.moe_component_probe import _profile_layer as moe_profile_layer
from examples.nemotron_ultra.runtime_mamba_candidate_contract import _build_result as mamba_contract_result
from examples.nemotron_ultra.runtime_mamba_candidate_contract import _render as mamba_contract_render
from examples.nemotron_ultra.runtime_lane_readiness_matrix import _build_result as lane_matrix_result
from examples.nemotron_ultra.runtime_lane_readiness_matrix import _render as lane_matrix_render
from examples.nemotron_ultra.runtime_next_runbook import _build_result as runbook_result
from examples.nemotron_ultra.runtime_next_runbook import _render as runbook_render
from examples.nemotron_ultra.runtime_proof_manifest import _build_result as manifest_result
from examples.nemotron_ultra.runtime_shape_contract import _build_result as shape_contract_result
from examples.nemotron_ultra.runtime_shape_contract import _render as shape_contract_render
from examples.nemotron_ultra.token_speed_budget import _build_result as budget_result
from examples.nemotron_ultra.validate_runtime_log_bundle import _validate as validate_bundle


def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _write_minimal_logs(root: Path, *, tps=8.5, moe=60.0, mamba=55.0, leaks=True):
    _write_json(
        root / "2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json",
        {
            "rows": [
                {
                    "id": "warm",
                    "decode_tps_excluding_first": tps,
                }
            ]
        },
    )
    _write_json(
        root / "2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json",
        {
            "manual_decode_total_ms": 130.0,
            "norm_lm_head_ms": 4.0,
            "summary_by_block_type": {
                "*": {"total_ms": 8.0},
                "E": {"total_ms": moe},
                "M": {"total_ms": mamba},
            },
        },
    )
    _write_json(
        root / "2026-06-04-nemotron-ultra-mamba-component-probe.json",
        {
            "layers": [
                {
                    "block_type": "M",
                    "cache_ordinal": 0,
                    "conv_dim": 18432,
                    "conv_input_shape": [1, 1, 18432],
                    "conv_output_shape": [1, 1, 18432],
                    "gate_shape": [1, 1, 16384],
                    "hidden_shape": [1, 1, 8192],
                    "intermediate_size": 16384,
                    "layer_index": 0,
                    "n_groups": 8,
                    "normed_shape": [1, 1, 8192],
                    "num_heads": 256,
                    "projected_shape": [1, 1, 35072],
                    "ssm_out_shape": [1, 1, 16384],
                    "ssm_state_size": 128,
                    "timings": [
                        {"label": "in_proj", "median_ms": 0.8},
                        {"label": "out_proj", "median_ms": 0.4},
                        {"label": "conv", "median_ms": 0.2},
                        {"label": "ssm_update", "median_ms": 0.18},
                        {"label": "full_mamba_mixer", "median_ms": 1.2},
                    ]
                }
            ]
        },
    )
    _write_json(
        root / "2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json",
        {
            "rows": [
                {
                    "id": "row",
                    "eos_reached": not leaks,
                    "visible_marker_leaks": ["</think>"] if leaks else [],
                    "ngram_repeat": {"repeat_fraction": 0.5 if leaks else 0.0},
                }
            ]
        },
    )
    _write_json(
        root / "2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json",
        {
            "layers": [
                {
                    "block_type": "E",
                    "hidden_shape": [1, 1, 8192],
                    "indices_shape": [1, 1, 22],
                    "latent_shape": [1, 1, 2048],
                    "layer_index": 1,
                    "routed_shape": [1, 1, 22, 2048],
                    "scores_shape": [1, 1, 22],
                    "timings": [
                        {"label": "switch_mlp", "median_ms": 1.1},
                        {"label": "shared_experts", "median_ms": 0.5},
                        {"label": "full_moe", "median_ms": 2.0},
                    ]
                }
            ]
        },
    )
    _write_json(
        root / "2026-06-04-nemotron-ultra-projection-tradeoff-probe.json",
        {
            "results": [
                {"name": "mamba_in_proj"},
                {"name": "mamba_out_proj"},
                {"name": "shared_up"},
                {"name": "shared_down"},
            ]
        },
    )
    _write_json(
        root / "2026-06-04-nemotron-ultra-jangtq1l-parser-probe.json",
        {
            "parser": "deepseek_r1 compatible <think> parser + Ultra XML function calls",
            "rows": [
                {
                    "id": "nt_math_default",
                    "visible_think_marker_leaks": [],
                    "truncated_reasoning": False,
                    "tool_calls": [],
                },
                {
                    "id": "nt_capital_default",
                    "visible_think_marker_leaks": ["</think>"] if leaks else [],
                    "truncated_reasoning": False,
                    "tool_calls": [],
                },
            ],
        },
    )


def _write_fake_bundle(root: Path):
    _write_json(
        root / "jang_config.json",
        {
            "format": "jangtq",
            "format_version": "2.0",
            "profile": "JANGTQ_1L",
            "capabilities": {
                "cache_type": "hybrid",
                "family": "nemotron_h",
                "modality": "text",
                "reasoning_parser": "deepseek_r1",
                "supports_thinking": True,
                "supports_tools": True,
                "think_in_template": True,
                "tool_parser": "nemotron",
            },
            "mxtq_bits": {
                "mamba_projection": 8,
                "routed_expert": {"down_proj": 1, "up_proj": 1},
                "shared_expert": 8,
            },
            "quantization": {"drops_mtp": True, "estimated_output_gib": 98.35},
            "runtime": {"shard_count": 51, "total_shard_bytes": 105603112752},
        },
    )
    _write_json(
        root / "config.json",
        {"layers_block_type": ["mamba", "moe", "attention"] * 2},
    )


def _write_queue(path: Path, baseline: Path, bundle: Path, candidate_root: Path):
    result = queue_result(
        argparse.Namespace(
            baseline_log_dir=baseline,
            bundle=bundle,
            candidate_root=candidate_root,
            speed_targets=[10.0, 12.0],
            wired_limit_gb=105,
            live_max_tokens=32,
            long_max_tokens=96,
            out=None,
            json_out=None,
        )
    )
    _write_json(path, result)
    return result


def _write_compare_gate_handoff(candidate: Path, baseline: Path, *, lane="moe", status_override=None):
    code, compare = compare_result(
        argparse.Namespace(
            baseline_log_dir=baseline,
            candidate_log_dir=candidate,
            out=None,
            json_out=None,
            max_repeat_fraction=0.25,
            max_tps_regression_pct=2.0,
            max_ms_regression_pct=5.0,
            min_tps_improvement_pct=2.0,
            min_ms_improvement_pct=5.0,
            strict=False,
        )
    )
    if status_override:
        compare["status"] = status_override
    _write_json(candidate / "2026-06-04-nemotron-ultra-runtime-speed-compare.json", compare)
    gate_args = argparse.Namespace(
        log_dir=candidate,
        min_live_tps=8.0,
        max_attention_ms=10.0,
        max_norm_lm_ms=5.0,
        min_bottleneck_ms=40.0,
        max_repeat_fraction=0.25,
        strict=False,
        out=None,
    )
    _, gate = speed_gate_result(gate_args)
    _write_json(candidate / "2026-06-04-nemotron-ultra-runtime-speed-gate.json", gate)
    _write_json(
        candidate / "2026-06-04-nemotron-ultra-token-speed-budget.json",
        budget_result(candidate, [10.0, 12.0]),
    )
    _write_json(
        candidate / "2026-06-04-nemotron-ultra-agent-handoff.json",
        {
            "artifact": {"drops_mtp": True},
            "cache_and_modality_gates": {
                "cache_type": "hybrid",
                "text_only": True,
            },
        },
    )
    _write_json(
        candidate / "2026-06-04-nemotron-ultra-runtime-cache-parser-contract.json",
        {
            "status": "PARTIAL",
            "cache_contract": {
                "cache_type": "hybrid",
                "cache_entries": 60,
                "mamba_companion_state_entries": 48,
                "attention_kv_cache_entries": 12,
            },
            "parser_contract": {
                "reasoning_parser": "deepseek_r1",
                "tool_parser": "nemotron",
            },
            "modality_contract": {
                "text_only": True,
                "drops_mtp": True,
            },
            "failures": [],
        },
    )
    return code


def test_runtime_speed_gate_reports_partial_for_remaining_bottlenecks(tmp_path):
    _write_minimal_logs(tmp_path)
    args = argparse.Namespace(
        log_dir=tmp_path,
        min_live_tps=8.0,
        max_attention_ms=10.0,
        max_norm_lm_ms=5.0,
        min_bottleneck_ms=40.0,
        max_repeat_fraction=0.25,
        strict=False,
        out=None,
    )
    code, report = speed_gate(args)
    json_code, result = speed_gate_result(args)

    assert code == 0
    assert json_code == 0
    assert "status: `PARTIAL`" in report
    assert result["status"] == "PARTIAL"
    assert result["metrics"]["best_live_tps"] == 8.5
    assert result["metrics"]["moe_ms"] == 60.0
    assert "best live speed 8.500 tok/s clears floor 8.000" in report
    assert "MoE remains a bottleneck at 60.000 ms" in report
    assert "coherence gate remains partial" in report


def test_runtime_speed_gate_reports_blocked_when_required_logs_are_missing(tmp_path):
    code, report = speed_gate(
        argparse.Namespace(
            log_dir=tmp_path,
            min_live_tps=8.0,
            max_attention_ms=10.0,
            max_norm_lm_ms=5.0,
            min_bottleneck_ms=40.0,
            max_repeat_fraction=0.25,
            strict=False,
            out=None,
        )
    )

    assert code == 2
    assert "status: `BLOCKED`" in report
    assert "missing live speed probe" in report
    assert "missing layer decode probe" in report
    assert "missing long coherence probe" in report


def test_runtime_speed_fix_acceptance_reports_partial_current_state(tmp_path):
    logs = tmp_path / "logs"
    bundle = tmp_path / "bundle"
    _write_preflight_inputs(logs, bundle, host_status="READY")
    _write_json(
        logs / "2026-06-04-nemotron-ultra-runtime-candidate-index.json",
        candidate_index_result(logs, logs / "2026-06-04-nemotron-ultra-runtime-experiment-queue.json"),
    )
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-issue-ledger.json", issue_ledger_result(logs))

    result = speed_acceptance_result(
        logs,
        target_tps=10.0,
        max_moe_ms=40.0,
        max_mamba_ms=40.0,
        require_speed_gate_fixed=True,
    )
    report = speed_acceptance_render(result)

    assert result["status"] == "PARTIAL"
    assert "speed gate is PARTIAL, not FIXED" in result["partial"]
    assert "no speed_candidate lane has ACCEPTED evidence" in result["partial"]
    assert "best live token/s 8.500 is below target 10.000" in result["partial"]
    assert "MoE bucket 60.000 ms exceeds acceptance ceiling 40.000" in result["partial"]
    assert "# Nemotron Ultra Runtime Speed Fix Acceptance" in report


def test_runtime_speed_fix_acceptance_reports_fixed_with_accepted_speed_lane(tmp_path):
    logs = tmp_path / "logs"
    bundle = tmp_path / "bundle"
    _write_minimal_logs(logs, tps=10.5, moe=35.0, mamba=35.0, leaks=False)
    _write_fake_bundle(bundle)
    _write_json(logs / "2026-06-04-nemotron-ultra-token-speed-budget.json", budget_result(logs, [10.0, 12.0]))
    _write_queue(logs / "2026-06-04-nemotron-ultra-runtime-experiment-queue.json", logs, bundle, logs)
    gate_args = argparse.Namespace(
        log_dir=logs,
        min_live_tps=8.0,
        max_attention_ms=10.0,
        max_norm_lm_ms=5.0,
        min_bottleneck_ms=40.0,
        max_repeat_fraction=0.25,
        strict=False,
        out=None,
    )
    _, gate = speed_gate_result(gate_args)
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-speed-gate.json", gate)
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-proof-manifest.json", {"status": "FIXED"})
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-issue-ledger.json", {"status": "FIXED"})

    candidate = logs / "candidate-moe-scheduling"
    candidate.mkdir(parents=True)
    _write_minimal_logs(candidate, tps=10.8, moe=30.0, mamba=35.0, leaks=False)
    _write_compare_gate_handoff(candidate, logs, lane="moe", status_override="IMPROVED")
    _write_json(
        candidate / "2026-06-04-nemotron-ultra-experiment-result-check.json",
        {
            "status": "ACCEPTED",
            "lane_id": "moe-routed-shared-scheduling",
            "lane_kind": "speed_candidate",
            "candidate_log_dir": str(candidate),
            "compare_status": "IMPROVED",
            "gate_status": "FIXED",
            "fixed": [],
            "failures": [],
            "missing_outputs": [],
        },
    )
    _write_json(
        logs / "2026-06-04-nemotron-ultra-runtime-candidate-index.json",
        candidate_index_result(logs, logs / "2026-06-04-nemotron-ultra-runtime-experiment-queue.json"),
    )

    result = speed_acceptance_result(
        logs,
        target_tps=10.0,
        max_moe_ms=40.0,
        max_mamba_ms=40.0,
        require_speed_gate_fixed=True,
    )

    assert result["status"] == "FIXED"
    assert result["partial"] == []
    assert result["blockers"] == []
    assert result["accepted_speed_lanes"][0]["id"] == "moe-routed-shared-scheduling"
    assert "best live token/s 10.500 meets target 10.000" in result["fixed"]


def test_host_runtime_readiness_parses_memory_disk_and_processes():
    vm = host_parse_vm_stat(
        """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               1000.
Pages active:                             2000.
Pages inactive:                           3000.
Pages speculative:                        4000.
Pages wired down:                         5000.
Pages occupied by compressor:             6000.
"""
    )
    pressure = host_parse_memory_pressure("System-wide memory free percentage: 42%\nSwapins: 7\nSwapouts: 9\n")
    disk = host_parse_df(
        """Filesystem 1G-blocks Used Available Capacity iused ifree %iused Mounted on
/dev/disk9s1 100 60 40 60% 1 2 33% /Volumes/EricsLLMDrive
"""
    )
    ps = host_parse_ps("1048576 /Applications/App.app\n524288 /usr/bin/python\n")

    assert round(vm["free_like_gib"], 3) == 0.122
    assert round(vm["wired_gib"], 3) == 0.076
    assert pressure["pressure"] == "OK"
    assert pressure["swapouts"] == 9
    assert disk["available_gib"] == 40.0
    assert ps[0]["rss_gib"] == 1.0


def test_host_runtime_readiness_render_includes_interpretation():
    report = host_render(
        {
            "status": "WATCH",
            "bundle": "/bundle",
            "log_dir": "logs",
            "memory": {
                "total_gib": 128.0,
                "vm_stat": {
                    "free_like_gib": 8.0,
                    "active_gib": 80.0,
                    "wired_gib": 30.0,
                    "compressed_gib": 4.0,
                },
                "memory_pressure": {"pressure": "OK", "swapins": 1, "swapouts": 2},
            },
            "disk": {
                "bundle": {"available_gib": 40.0, "mount": "/Volumes/EricsLLMDrive"},
                "log_dir": {"available_gib": 100.0, "mount": "/"},
            },
            "high_rss_processes": [{"rss_gib": 4.25, "command": "Example.app"}],
            "warnings": ["free/inactive/speculative/purgeable memory is only 8.0 GiB"],
            "interpretation": [
                "Saved speed logs still point at MoE and Mamba compute/dispatch as the primary bottlenecks."
            ],
            "commands": {"refresh": "python host_runtime_readiness.py"},
        }
    )

    assert "status: `WATCH`" in report
    assert "Example.app" in report
    assert "MoE and Mamba" in report


def test_host_cleanup_runbook_tags_model_servers_and_vms():
    rows = cleanup_parse_ps(
        """100 37045760 /Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3 -m vmlx_engine.cli serve model
200 7340032 /Applications/Parallels Desktop.app/Contents/MacOS//Parallels VM.app/Contents/MacOS/prl_vm_app --vm-name Windows
300 1048576 /Applications/App.app
""",
        min_rss_gib=2.0,
    )

    assert rows[0]["pid"] == 100
    assert rows[0]["likely_model_server"] is True
    assert rows[1]["likely_vm"] is True
    assert len(rows) == 2


def test_host_cleanup_runbook_render_is_non_destructive():
    report = cleanup_render(
        {
            "status": "WATCH",
            "log_dir": "logs",
            "min_rss_gib": 2.0,
            "processes": [
                {
                    "pid": 100,
                    "rss_gib": 35.3,
                    "command": "python -m vmlx_engine.cli serve model",
                    "likely_model_server": True,
                    "likely_vm": False,
                }
            ],
            "recommended_actions": ["Do not kill unknown processes blindly."],
            "follow_up_commands": {"host_readiness": "python host_runtime_readiness.py"},
        }
    )

    assert "pid `100`" in report
    assert "model_server" in report
    assert "Do not kill unknown processes blindly." in report
    assert "kill -9" not in report


def test_component_budget_matrix_projects_component_cut_gains(tmp_path):
    _write_minimal_logs(tmp_path, tps=8.5, moe=60.0, mamba=55.0, leaks=False)
    _write_json(tmp_path / "2026-06-04-nemotron-ultra-token-speed-budget.json", budget_result(tmp_path, [10.0]))

    result = component_matrix_result(tmp_path)
    report = component_matrix_render(result)

    assert result["current"]["manual_decode_total_ms"] == 130.0
    assert result["component_rows"][0]["family"] == "MoE"
    assert result["component_rows"][0]["label"] == "full_moe"
    assert result["component_rows"][0]["role"] == "inclusive_path"
    assert result["component_rows"][0]["projected_total_ms"] == 96.0
    assert result["target_hits"][0]["target_tps"] == 10.0
    assert "MoE:full_moe" in result["target_hits"][0]["single_component_can_cover"]
    assert "# Nemotron Ultra Component Budget Matrix" in report
    assert "25% cut tps" in report
    assert "`full_*` rows are inclusive path measurements" in report


def test_component_budget_matrix_classifies_weighted_decode_fast_path(tmp_path):
    _write_minimal_logs(tmp_path, tps=8.5, moe=60.0, mamba=55.0, leaks=False)
    _write_json(tmp_path / "2026-06-04-nemotron-ultra-token-speed-budget.json", budget_result(tmp_path, [10.0]))
    moe_path = tmp_path / "2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json"
    moe_log = json.loads(moe_path.read_text())
    moe_log["layers"][0]["timings"].append({"label": "weighted_decode", "median_ms": 0.95})
    _write_json(moe_path, moe_log)

    result = component_matrix_result(tmp_path)
    rows = {row["label"]: row for row in result["component_rows"]}

    assert rows["weighted_decode"]["role"] == "fused_fast_path"
    assert round(rows["weighted_decode"]["projected_total_ms"], 3) == 45.6


def test_moe_component_probe_times_weighted_decode_fast_path_when_available():
    class FakeSwitchMLP:
        def __call__(self, x, inds):
            return mx.ones((*inds.shape, x.shape[-1]), dtype=x.dtype)

        def _jangtq_weighted_decode(self, x, inds, scores):
            return mx.sum(self(x, inds) * scores[..., None], axis=-2)

    class FakeMoE:
        moe_latent_size = None
        config = argparse.Namespace(n_shared_experts=None)

        def __init__(self):
            self.switch_mlp = FakeSwitchMLP()

        def gate(self, x):
            inds = mx.array([[[0, 1]]], dtype=mx.uint32)
            scores = mx.array([[[0.25, 0.75]]], dtype=mx.float32)
            return inds, scores

        def __call__(self, x):
            inds, scores = self.gate(x)
            return self.switch_mlp._jangtq_weighted_decode(x, inds, scores)

    class FakeLayer:
        block_type = "E"

        def __init__(self):
            self.mixer = FakeMoE()

        def norm(self, hidden):
            return hidden

    result = moe_profile_layer(FakeLayer(), mx.ones((1, 1, 4), dtype=mx.bfloat16), repeats=1, warmup=0)
    labels = [timing["label"] for timing in result["timings"]]

    assert "weighted_decode" in labels


def test_runtime_issue_ledger_lists_open_speed_and_coherence_work(tmp_path):
    logs = tmp_path / "logs"
    bundle = tmp_path / "bundle"
    _write_preflight_inputs(logs, bundle, host_status="WATCH")
    _write_minimal_logs(logs, tps=8.5, moe=60.0, mamba=55.0, leaks=True)
    gate_args = argparse.Namespace(
        log_dir=logs,
        min_live_tps=8.0,
        max_attention_ms=10.0,
        max_norm_lm_ms=5.0,
        min_bottleneck_ms=40.0,
        max_repeat_fraction=0.25,
        strict=False,
        out=None,
    )
    _, gate = speed_gate_result(gate_args)
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-speed-gate.json", gate)
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-next-runbook.json", runbook_result(logs))

    result = issue_ledger_result(logs)
    report = issue_ledger_render(result)

    assert result["status"] == "OPEN"
    assert result["status_counts"]["OPEN"] == 4
    assert result["status_counts"]["FIXED"] == 1
    assert [issue["id"] for issue in result["issues"]] == [
        "NU-SPEED-001",
        "NU-SPEED-002",
        "NU-COHERENCE-001",
        "NU-HOST-001",
        "NU-FIXED-001",
    ]
    assert result["issues"][0]["status"] == "OPEN"
    assert "MoE bucket is 60.000 ms." in result["issues"][0]["evidence"]
    assert "Host readiness is READY" in result["issues"][3]["acceptance_checks"][0]
    assert "# Nemotron Ultra Runtime Issue Ledger" in report
    assert "NU-SPEED-001" in report


def test_runtime_candidate_index_reports_missing_and_accepted_lanes(tmp_path):
    logs = tmp_path / "logs"
    bundle = tmp_path / "bundle"
    _write_preflight_inputs(logs, bundle, host_status="READY")
    queue_path = logs / "2026-06-04-nemotron-ultra-runtime-experiment-queue.json"
    queue = json.loads(queue_path.read_text())
    candidate = logs / "candidate-moe-scheduling"
    candidate.mkdir(parents=True)
    _write_compare_gate_handoff(candidate, logs, lane="moe", status_override="IMPROVED")
    result_check = {
        "status": "ACCEPTED",
        "lane_id": "moe-routed-shared-scheduling",
        "lane_kind": "speed_candidate",
        "candidate_log_dir": str(candidate),
        "expected_compare_statuses": ["IMPROVED"],
        "compare_status": "IMPROVED",
        "gate_status": "PARTIAL",
        "fixed": ["compare status IMPROVED matches lane expectation"],
        "failures": [],
        "missing_outputs": [],
    }
    _write_json(candidate / "2026-06-04-nemotron-ultra-experiment-result-check.json", result_check)

    result = candidate_index_result(logs, queue_path)
    report = candidate_index_render(result)

    assert result["status"] == "OPEN"
    assert result["status_counts"]["ACCEPTED"] == 1
    assert result["status_counts"]["MISSING"] == 3
    assert result["lanes"][0]["status"] == "ACCEPTED"
    assert result["lanes"][0]["compare_status"] == "IMPROVED"
    assert result["lanes"][1]["status"] == "MISSING"
    assert "candidate-mamba-dispatch" in result["lanes"][1]["reason"]
    assert "# Nemotron Ultra Runtime Candidate Index" in report
    assert "moe-routed-shared-scheduling" in report


def test_runtime_candidate_launch_guard_blocks_watch_without_override(tmp_path):
    logs = tmp_path / "logs"
    bundle = tmp_path / "bundle"
    _write_preflight_inputs(logs, bundle, host_status="WATCH")
    _write_json(logs / "2026-06-04-nemotron-ultra-token-speed-budget.json", budget_result(logs, [10.0, 12.0]))
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-lane-readiness-matrix.json", lane_matrix_result(logs))
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-next-runbook.json", runbook_result(logs))
    _write_json(
        logs / "2026-06-04-nemotron-ultra-runtime-candidate-index.json",
        candidate_index_result(logs, logs / "2026-06-04-nemotron-ultra-runtime-experiment-queue.json"),
    )

    result = launch_guard_result(logs, allow_watch=False)
    report = launch_guard_render(result)

    assert result["status"] == "BLOCKED_BY_WATCH"
    assert result["lane"]["id"] == "moe-routed-shared-scheduling"
    assert "run_runtime_candidate_suite.py" in result["commands"]["candidate"]
    assert "experiment_result_check.py" in result["commands"]["post_check"]
    assert "# Nemotron Ultra Runtime Candidate Launch Guard" in report


def test_runtime_candidate_launch_guard_allows_watch_with_override(tmp_path):
    logs = tmp_path / "logs"
    bundle = tmp_path / "bundle"
    _write_preflight_inputs(logs, bundle, host_status="WATCH")
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-lane-readiness-matrix.json", lane_matrix_result(logs))
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-next-runbook.json", runbook_result(logs))
    _write_json(
        logs / "2026-06-04-nemotron-ultra-runtime-candidate-index.json",
        candidate_index_result(logs, logs / "2026-06-04-nemotron-ultra-runtime-experiment-queue.json"),
    )

    result = launch_guard_result(logs, allow_watch=True)

    assert result["status"] == "READY"
    assert result["allow_watch"] is True
    assert result["warnings"]


def test_runtime_cleanup_ready_check_reports_watch_blockers(tmp_path):
    logs = tmp_path / "logs"
    _write_json(
        logs / "2026-06-04-nemotron-ultra-host-runtime-readiness.json",
        {"status": "WATCH", "warnings": ["largest resident process is 35.9 GiB"]},
    )
    _write_json(
        logs / "2026-06-04-nemotron-ultra-host-cleanup-runbook.json",
        {
            "status": "WATCH",
            "processes": [
                {
                    "pid": 100,
                    "rss_gib": 35.9,
                    "command": "python -m vmlx_engine.cli serve model",
                    "likely_model_server": True,
                    "likely_vm": False,
                }
            ],
        },
    )
    _write_json(
        logs / "2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.json",
        {
            "status": "BLOCKED_BY_WATCH",
            "warnings": ["selected candidate is WATCH"],
        },
    )

    result = cleanup_ready_result(logs)
    report = cleanup_ready_render(result)

    assert result["status"] == "WATCH"
    assert "model server pid 100 uses 35.90 GiB RSS" in result["blockers"]
    assert "candidate launch guard is BLOCKED_BY_WATCH" in result["blockers"]
    assert "kill -9" not in report
    assert "# Nemotron Ultra Runtime Cleanup Ready Check" in report


def test_runtime_cleanup_ready_check_reports_ready_when_inputs_ready(tmp_path):
    logs = tmp_path / "logs"
    _write_json(logs / "2026-06-04-nemotron-ultra-host-runtime-readiness.json", {"status": "READY"})
    _write_json(
        logs / "2026-06-04-nemotron-ultra-host-cleanup-runbook.json",
        {"status": "READY", "processes": []},
    )
    _write_json(
        logs / "2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.json",
        {"status": "READY", "warnings": []},
    )

    result = cleanup_ready_result(logs)

    assert result["status"] == "READY"
    assert result["blockers"] == []


def test_runtime_moe_candidate_contract_records_targets_and_invariants(tmp_path):
    logs = tmp_path / "logs"
    bundle = tmp_path / "bundle"
    _write_preflight_inputs(logs, bundle, host_status="WATCH")
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-lane-readiness-matrix.json", lane_matrix_result(logs))
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-next-runbook.json", runbook_result(logs))
    _write_json(
        logs / "2026-06-04-nemotron-ultra-runtime-candidate-index.json",
        candidate_index_result(logs, logs / "2026-06-04-nemotron-ultra-runtime-experiment-queue.json"),
    )
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.json", launch_guard_result(logs, allow_watch=False))
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-issue-ledger.json", issue_ledger_result(logs))

    result = moe_contract_result(logs)
    report = moe_contract_render(result)

    assert result["status"] == "BLOCKED"
    assert result["lane_id"] == "moe-routed-shared-scheduling"
    assert result["target"]["target_tps"] == 10.0
    assert result["invariants"]["indices_shape"] == [1, 1, 22]
    assert result["invariants"]["routed_expert_bits"] == {"down_proj": 1, "up_proj": 1}
    assert "candidate launch guard is BLOCKED_BY_WATCH" in result["preconditions"]
    assert "run_runtime_candidate_suite.py" in result["candidate_command"]
    assert "# Nemotron Ultra MoE Candidate Contract" in report


def test_runtime_moe_execution_ticket_records_exact_ready_sequence(tmp_path):
    logs = tmp_path / "logs"
    bundle = tmp_path / "bundle"
    _write_preflight_inputs(logs, bundle, host_status="READY")
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-lane-readiness-matrix.json", lane_matrix_result(logs))
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-next-runbook.json", runbook_result(logs))
    _write_json(
        logs / "2026-06-04-nemotron-ultra-runtime-candidate-index.json",
        candidate_index_result(logs, logs / "2026-06-04-nemotron-ultra-runtime-experiment-queue.json"),
    )
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.json", launch_guard_result(logs, allow_watch=False))
    _write_json(
        logs / "2026-06-04-nemotron-ultra-runtime-cleanup-ready-check.json",
        {"status": "READY", "blockers": []},
    )
    _write_json(logs / "2026-06-04-nemotron-ultra-component-budget-matrix.json", component_matrix_result(logs))
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-issue-ledger.json", issue_ledger_result(logs))
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-moe-candidate-contract.json", moe_contract_result(logs))

    result = moe_ticket_result(logs)
    report = moe_ticket_render(result)

    assert result["status"] == "READY"
    assert result["lane_id"] == "moe-routed-shared-scheduling"
    assert result["candidate_status"] == "MISSING"
    assert result["guard_status"] == "READY"
    assert result["cleanup_status"] == "READY"
    assert result["contract_status"] == "READY"
    assert result["target"]["target_tps"] == 10.0
    assert result["invariants"]["indices_shape"] == [1, 1, 22]
    assert "run_runtime_candidate_suite.py" in result["commands"]["candidate"]
    assert "experiment_result_check.py" in result["commands"]["post_check"]
    assert "runtime_candidate_index.py" in result["commands"]["post_candidate_index"]
    assert "refresh_runtime_proof_bundle.py" in result["commands"]["post_candidate_refresh"]
    assert "do not run the Mamba lane" in result["do_not"][0]
    assert "# Nemotron Ultra MoE Execution Ticket" in report


def test_runtime_moe_execution_ticket_blocks_without_ready_inputs(tmp_path):
    logs = tmp_path / "logs"
    _write_json(
        logs / "2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.json",
        {"status": "BLOCKED_BY_WATCH", "lane": {"id": "moe-routed-shared-scheduling"}, "commands": {}},
    )
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-cleanup-ready-check.json", {"status": "WATCH"})
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-moe-candidate-contract.json", {"status": "BLOCKED"})
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-candidate-index.json", {"lanes": []})

    result = moe_ticket_result(logs)

    assert result["status"] == "BLOCKED"
    assert "candidate launch guard is BLOCKED_BY_WATCH" in result["failures"]
    assert "cleanup ready check is WATCH" in result["failures"]
    assert "MoE candidate contract is BLOCKED" in result["failures"]


def test_runtime_moe_surface_map_records_real_jang_symbols(tmp_path):
    logs = tmp_path / "logs"
    bundle = tmp_path / "bundle"
    _write_preflight_inputs(logs, bundle, host_status="READY")
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-lane-readiness-matrix.json", lane_matrix_result(logs))
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-next-runbook.json", runbook_result(logs))
    _write_json(
        logs / "2026-06-04-nemotron-ultra-runtime-candidate-index.json",
        candidate_index_result(logs, logs / "2026-06-04-nemotron-ultra-runtime-experiment-queue.json"),
    )
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.json", launch_guard_result(logs, allow_watch=False))
    _write_json(
        logs / "2026-06-04-nemotron-ultra-runtime-cleanup-ready-check.json",
        {"status": "READY", "blockers": []},
    )
    _write_json(logs / "2026-06-04-nemotron-ultra-component-budget-matrix.json", component_matrix_result(logs))
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-issue-ledger.json", issue_ledger_result(logs))
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-moe-candidate-contract.json", moe_contract_result(logs))
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-moe-execution-ticket.json", moe_ticket_result(logs))

    result = moe_surface_result(logs, Path("jang-tools"))
    report = moe_surface_render(result)

    assert result["status"] == "READY"
    assert result["lane_id"] == "moe-routed-shared-scheduling"
    assert result["ticket_status"] == "READY"
    assert result["contract_status"] == "READY"
    assert result["component_timings_ms"]["switch_mlp"] == 1.1
    moe_component_surface = next(row for row in result["surfaces"] if row["id"] == "moe-component-proof")
    assert "weighted_decode" in moe_component_surface["anchors"]
    assert {row["id"] for row in result["surfaces"]} >= {
        "loader-hydration",
        "nemotron-weighted-moe-patch",
        "routed-gather-kernel",
        "fused-gate-up-kernel",
        "grouped-nax-proof-surface",
    }
    assert all(row["status"] == "READY" for row in result["surfaces"])
    assert "JANGTQ_DISABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH=1" in result["runtime_controls"]["disable_weighted_moe_fastpath"]
    assert "do not edit vMLX or MLX Studio" in result["non_goals"][0]
    assert "# Nemotron Ultra MoE Runtime Surface Map" in report


def test_runtime_moe_surface_map_blocks_when_source_missing(tmp_path):
    logs = tmp_path / "logs"
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-moe-candidate-contract.json", {"lane_id": "moe-routed-shared-scheduling"})
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-moe-execution-ticket.json", {"status": "READY"})
    empty_source = tmp_path / "empty-source"

    result = moe_surface_result(logs, empty_source)

    assert result["status"] == "BLOCKED"
    assert any("missing source file" in item for item in result["missing"])


def test_runtime_moe_patch_plan_orders_implementation_steps(tmp_path):
    logs = tmp_path / "logs"
    bundle = tmp_path / "bundle"
    _write_preflight_inputs(logs, bundle, host_status="READY")
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-lane-readiness-matrix.json", lane_matrix_result(logs))
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-next-runbook.json", runbook_result(logs))
    _write_json(
        logs / "2026-06-04-nemotron-ultra-runtime-candidate-index.json",
        candidate_index_result(logs, logs / "2026-06-04-nemotron-ultra-runtime-experiment-queue.json"),
    )
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.json", launch_guard_result(logs, allow_watch=False))
    _write_json(
        logs / "2026-06-04-nemotron-ultra-runtime-cleanup-ready-check.json",
        {"status": "READY", "blockers": []},
    )
    _write_json(logs / "2026-06-04-nemotron-ultra-component-budget-matrix.json", component_matrix_result(logs))
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-issue-ledger.json", issue_ledger_result(logs))
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-moe-candidate-contract.json", moe_contract_result(logs))
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-moe-execution-ticket.json", moe_ticket_result(logs))
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-moe-surface-map.json", moe_surface_result(logs, Path("jang-tools")))
    _write_json(
        logs / "2026-06-04-nemotron-ultra-runtime-speed-fix-acceptance.json",
        speed_acceptance_result(
            logs,
            target_tps=10.0,
            max_moe_ms=40.0,
            max_mamba_ms=40.0,
            require_speed_gate_fixed=True,
        ),
    )

    result = moe_patch_plan_result(logs)
    report = moe_patch_plan_render(result)

    assert result["status"] == "READY"
    assert result["speed_acceptance_status"] == "PARTIAL"
    assert [step["id"] for step in result["steps"]] == [
        "moe-01-path-scheduling",
        "moe-02-switchmlp-routed-kernels",
        "moe-03-shared-experts-overlap",
    ]
    assert result["steps"][0]["component"]["label"] == "full_moe"
    assert result["steps"][1]["component"]["label"] == "switch_mlp"
    assert result["steps"][2]["component"]["label"] == "shared_experts"
    assert "run_runtime_candidate_suite.py" in result["candidate_command"]
    assert "runtime_speed_fix_acceptance.py" in result["acceptance_command"]
    assert "# Nemotron Ultra MoE Patch Plan" in report


def test_runtime_moe_patch_plan_prefers_weighted_decode_when_measured(tmp_path):
    logs = tmp_path / "logs"
    bundle = tmp_path / "bundle"
    _write_preflight_inputs(logs, bundle, host_status="READY")
    moe_path = logs / "2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json"
    moe_log = json.loads(moe_path.read_text())
    moe_log["layers"][0]["timings"].append({"label": "weighted_decode", "median_ms": 0.95})
    _write_json(moe_path, moe_log)
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-lane-readiness-matrix.json", lane_matrix_result(logs))
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-next-runbook.json", runbook_result(logs))
    _write_json(
        logs / "2026-06-04-nemotron-ultra-runtime-candidate-index.json",
        candidate_index_result(logs, logs / "2026-06-04-nemotron-ultra-runtime-experiment-queue.json"),
    )
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.json", launch_guard_result(logs, allow_watch=False))
    _write_json(
        logs / "2026-06-04-nemotron-ultra-runtime-cleanup-ready-check.json",
        {"status": "READY", "blockers": []},
    )
    _write_json(logs / "2026-06-04-nemotron-ultra-component-budget-matrix.json", component_matrix_result(logs))
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-issue-ledger.json", issue_ledger_result(logs))
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-moe-candidate-contract.json", moe_contract_result(logs))
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-moe-execution-ticket.json", moe_ticket_result(logs))
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-moe-surface-map.json", moe_surface_result(logs, Path("jang-tools")))
    _write_json(
        logs / "2026-06-04-nemotron-ultra-runtime-speed-fix-acceptance.json",
        speed_acceptance_result(
            logs,
            target_tps=10.0,
            max_moe_ms=40.0,
            max_mamba_ms=40.0,
            require_speed_gate_fixed=True,
        ),
    )

    result = moe_patch_plan_result(logs)

    assert result["steps"][1]["component"]["label"] == "weighted_decode"
    assert "weighted_decode" in result["steps"][1]["validation"][0]


def test_runtime_moe_patch_plan_blocks_when_inputs_are_missing(tmp_path):
    logs = tmp_path / "logs"
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-moe-surface-map.json", {"status": "BLOCKED"})

    result = moe_patch_plan_result(logs)

    assert result["status"] == "BLOCKED"
    assert "MoE surface map is BLOCKED" in result["failures"]
    assert "MoE candidate contract is MISSING" in result["failures"]
    assert "MoE execution ticket is MISSING" in result["failures"]


def test_runtime_moe_delta_contract_records_thresholds(tmp_path):
    logs = tmp_path / "logs"
    bundle = tmp_path / "bundle"
    _write_preflight_inputs(logs, bundle, host_status="READY")
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-lane-readiness-matrix.json", lane_matrix_result(logs))
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-next-runbook.json", runbook_result(logs))
    _write_json(
        logs / "2026-06-04-nemotron-ultra-runtime-candidate-index.json",
        candidate_index_result(logs, logs / "2026-06-04-nemotron-ultra-runtime-experiment-queue.json"),
    )
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.json", launch_guard_result(logs, allow_watch=False))
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-cleanup-ready-check.json", {"status": "READY", "blockers": []})
    _write_json(logs / "2026-06-04-nemotron-ultra-component-budget-matrix.json", component_matrix_result(logs))
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-issue-ledger.json", issue_ledger_result(logs))
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-moe-candidate-contract.json", moe_contract_result(logs))
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-moe-execution-ticket.json", moe_ticket_result(logs))
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-moe-surface-map.json", moe_surface_result(logs, Path("jang-tools")))
    _write_json(
        logs / "2026-06-04-nemotron-ultra-runtime-speed-fix-acceptance.json",
        speed_acceptance_result(
            logs,
            target_tps=10.0,
            max_moe_ms=40.0,
            max_mamba_ms=40.0,
            require_speed_gate_fixed=True,
        ),
    )
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-moe-patch-plan.json", moe_patch_plan_result(logs))

    result = moe_delta_result(logs)
    report = moe_delta_render(result)

    assert result["status"] == "READY"
    assert result["lane_id"] == "moe-routed-shared-scheduling"
    assert result["baseline"]["manual_decode_total_ms"] == 130.0
    assert result["baseline"]["moe_ms"] == 60.0
    assert result["target"]["target_tps"] == 10.0
    assert result["target"]["required_total_cut_ms"] == 30.0
    assert round(result["target"]["target_moe_ms_for_proportional_10tps"], 3) == 44.348
    assert result["target"]["acceptance_max_moe_ms"] == 40.0
    assert result["acceptance_thresholds"]["experiment_result_status"] == "ACCEPTED"
    assert result["acceptance_thresholds"]["compare_status"] == "IMPROVED"
    assert result["ordered_steps"][0]["id"] == "moe-01-path-scheduling"
    assert "weighted-moe-ablation" in result["negative_controls"][0]
    assert "run_runtime_candidate_suite.py" in result["commands"]["candidate"]
    assert "runtime_speed_fix_acceptance.py" in result["commands"]["acceptance_strict"]
    assert "# Nemotron Ultra MoE Delta Contract" in report


def test_runtime_moe_delta_contract_blocks_when_inputs_missing(tmp_path):
    logs = tmp_path / "logs"
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-moe-candidate-contract.json", {"status": "BLOCKED"})

    result = moe_delta_result(logs)

    assert result["status"] == "BLOCKED"
    assert "MoE candidate contract is BLOCKED" in result["failures"]
    assert "missing token speed budget" in result["failures"]
    assert "missing MoE patch plan" in result["failures"]


def test_runtime_mamba_candidate_contract_records_targets_and_invariants(tmp_path):
    logs = tmp_path / "logs"
    bundle = tmp_path / "bundle"
    _write_preflight_inputs(logs, bundle, host_status="WATCH")
    _write_json(logs / "2026-06-04-nemotron-ultra-token-speed-budget.json", budget_result(logs, [10.0, 12.0]))
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-lane-readiness-matrix.json", lane_matrix_result(logs))
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-next-runbook.json", runbook_result(logs))
    _write_json(
        logs / "2026-06-04-nemotron-ultra-runtime-candidate-index.json",
        candidate_index_result(logs, logs / "2026-06-04-nemotron-ultra-runtime-experiment-queue.json"),
    )
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.json", launch_guard_result(logs, allow_watch=False))
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-issue-ledger.json", issue_ledger_result(logs))

    result = mamba_contract_result(logs)
    report = mamba_contract_render(result)

    assert result["status"] == "BLOCKED"
    assert result["lane_id"] == "mamba-projection-dispatch"
    assert result["target"]["target_tps"] == 12.0
    assert result["invariants"]["projected_shape"] == [1, 1, 35072]
    assert result["invariants"]["gate_shape"] == [1, 1, 16384]
    assert result["invariants"]["ssm_state_size"] == 128
    assert result["invariants"]["mamba_projection_bits"] == 8
    assert "candidate launch guard is BLOCKED_BY_WATCH" in result["preconditions"]
    assert "MoE lane evidence is MISSING" in result["preconditions"][1]
    assert "run_runtime_candidate_suite.py" in result["candidate_command"]
    assert "# Nemotron Ultra Mamba Candidate Contract" in report


def test_runtime_patch_spec_builds_implementation_lanes(tmp_path):
    logs = tmp_path / "logs"
    bundle = tmp_path / "bundle"
    _write_minimal_logs(logs, tps=8.5, moe=60.0, mamba=55.0, leaks=False)
    _write_fake_bundle(bundle)
    _write_json(logs / "2026-06-04-nemotron-ultra-token-speed-budget.json", budget_result(logs, [10.0]))
    gate_args = argparse.Namespace(
        log_dir=logs,
        min_live_tps=8.0,
        max_attention_ms=10.0,
        max_norm_lm_ms=5.0,
        min_bottleneck_ms=40.0,
        max_repeat_fraction=0.25,
        strict=False,
        out=None,
    )
    _, gate = speed_gate_result(gate_args)
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-speed-gate.json", gate)
    _write_queue(logs / "2026-06-04-nemotron-ultra-runtime-experiment-queue.json", logs, bundle, logs)

    result = patch_spec_result(logs)
    report = patch_spec_render(result)

    assert result["status"] == "PARTIAL"
    assert result["lanes"][0]["id"] == "moe-routed-shared-scheduling"
    assert result["lanes"][1]["id"] == "mamba-projection-dispatch"
    assert "run_runtime_candidate_suite.py" in result["lanes"][0]["proof"]["candidate_command"]
    assert any("do not chase attention first" in item for item in result["global_non_goals"])
    assert "# Nemotron Ultra Runtime Patch Spec" in report
    assert "Implementation Lanes" in report


def test_runtime_shape_contract_records_bits_layers_and_shapes(tmp_path):
    logs = tmp_path / "logs"
    bundle = tmp_path / "bundle"
    _write_minimal_logs(logs, tps=8.5, moe=60.0, mamba=55.0, leaks=False)
    _write_fake_bundle(bundle)

    result = shape_contract_result(bundle, logs)
    report = shape_contract_render(result)

    assert result["status"] == "READY"
    assert result["architecture"]["layer_counts"] == {"total": 6, "mamba": 2, "moe": 2, "attention": 2}
    assert result["quantization"]["mxtq_bits"]["routed_expert"]["up_proj"] == 1
    assert result["quantization"]["mxtq_bits"]["mamba_projection"] == 8
    assert result["mamba_contract"]["projected_shape"] == [1, 1, 35072]
    assert result["moe_contract"]["indices_shape"] == [1, 1, 22]
    assert "MTP remains dropped" in report
    assert "# Nemotron Ultra Runtime Shape Contract" in report


def test_runtime_cache_parser_contract_records_hybrid_parser_and_modality_gates(tmp_path):
    logs = tmp_path / "logs"
    bundle = tmp_path / "bundle"
    _write_preflight_inputs(logs, bundle, host_status="READY")
    _write_json(
        logs / "2026-06-04-nemotron-ultra-runtime-candidate-index.json",
        candidate_index_result(logs, logs / "2026-06-04-nemotron-ultra-runtime-experiment-queue.json"),
    )
    _write_json(
        logs / "2026-06-04-nemotron-ultra-runtime-speed-fix-acceptance.json",
        speed_acceptance_result(
            logs,
            target_tps=10.0,
            max_moe_ms=40.0,
            max_mamba_ms=40.0,
            require_speed_gate_fixed=True,
        ),
    )

    result = cache_parser_contract_result(logs)
    report = cache_parser_contract_render(result)

    assert result["status"] == "PARTIAL"
    assert result["cache_contract"]["cache_type"] == "hybrid"
    assert result["cache_contract"]["cache_entries"] == 60
    assert result["cache_contract"]["mamba_companion_state_entries"] == 48
    assert result["cache_contract"]["attention_kv_cache_entries"] == 12
    assert result["parser_contract"]["reasoning_parser"] == "deepseek_r1"
    assert result["parser_contract"]["tool_parser"] == "nemotron"
    assert result["parser_contract"]["parser_probe"]["status"] == "PARTIAL"
    assert result["modality_contract"]["text_only"] is True
    assert result["modality_contract"]["drops_mtp"] is True
    assert "parser probe is PARTIAL" in result["partials"]
    assert "hybrid" in report
    assert "# Nemotron Ultra Cache Parser Contract" in report


def test_runtime_cache_parser_contract_blocks_when_core_inputs_missing(tmp_path):
    logs = tmp_path / "logs"

    result = cache_parser_contract_result(logs)

    assert result["status"] == "BLOCKED"
    assert any("missing agent handoff" in item for item in result["failures"])
    assert any("missing shape contract" in item for item in result["failures"])


def _write_preflight_inputs(logs: Path, bundle: Path, *, host_status="READY", manifest_status="PARTIAL"):
    _write_minimal_logs(logs, tps=8.5, moe=60.0, mamba=55.0, leaks=False)
    _write_fake_bundle(bundle)
    _write_json(
        bundle / "config.json",
        {
            "layers_block_type": ["mamba", "moe"] * 48 + ["attention"] * 12,
            "hidden_size": 8192,
            "vocab_size": 131072,
            "tie_word_embeddings": False,
            "num_hidden_layers": 108,
            "num_attention_heads": 64,
            "num_key_value_heads": 2,
            "num_experts_per_tok": 22,
            "moe_intermediate_size": 5120,
            "ssm_state_size": 128,
        },
    )
    _write_json(logs / "2026-06-04-nemotron-ultra-token-speed-budget.json", budget_result(logs, [10.0]))
    gate_args = argparse.Namespace(
        log_dir=logs,
        min_live_tps=8.0,
        max_attention_ms=10.0,
        max_norm_lm_ms=5.0,
        min_bottleneck_ms=40.0,
        max_repeat_fraction=0.25,
        strict=False,
        out=None,
    )
    _, gate = speed_gate_result(gate_args)
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-speed-gate.json", gate)
    _write_queue(logs / "2026-06-04-nemotron-ultra-runtime-experiment-queue.json", logs, bundle, logs)
    _write_json(
        logs / "2026-06-04-nemotron-ultra-agent-handoff.json",
        handoff_result(
            argparse.Namespace(
                bundle=bundle,
                log_dir=logs,
                out=None,
                json_out=None,
                min_live_tps=8.0,
                max_attention_ms=10.0,
                max_norm_lm_ms=5.0,
                min_bottleneck_ms=40.0,
                max_repeat_fraction=0.25,
                speed_targets=[10.0, 12.0],
            )
        ),
    )
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-patch-spec.json", patch_spec_result(logs))
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-shape-contract.json", shape_contract_result(bundle, logs))
    manifest = manifest_result(logs)
    manifest["status"] = manifest_status
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-proof-manifest.json", manifest)
    _write_json(
        logs / "2026-06-04-nemotron-ultra-host-runtime-readiness.json",
        {
            "status": host_status,
            "warnings": ["largest resident process is 35.3 GiB"] if host_status == "WATCH" else [],
        },
    )


def test_runtime_candidate_preflight_warns_for_host_watch(tmp_path):
    logs = tmp_path / "logs"
    bundle = tmp_path / "bundle"
    _write_preflight_inputs(logs, bundle, host_status="WATCH")

    result = preflight_result(logs, "moe-routed-shared-scheduling")
    report = preflight_render(result)

    assert result["status"] == "WATCH"
    assert result["failures"] == []
    assert any("host readiness is WATCH" in item for item in result["warnings"])
    assert "run_runtime_candidate_suite.py" in result["candidate_command"]
    assert result["candidate_log_dir"] == str(logs / "candidate-moe-scheduling")
    assert result["dry_run_command"] == result["candidate_command"] + " --dry-run"
    assert "# Nemotron Ultra Runtime Candidate Preflight" in report


def test_runtime_candidate_preflight_blocks_stale_manifest(tmp_path):
    logs = tmp_path / "logs"
    bundle = tmp_path / "bundle"
    _write_preflight_inputs(logs, bundle, host_status="READY")
    manifest_path = logs / "2026-06-04-nemotron-ultra-runtime-proof-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["stale_required_artifacts"] = ["docs/runtime/logs/stale.json"]
    _write_json(manifest_path, manifest)

    result = preflight_result(logs, "moe-routed-shared-scheduling")

    assert result["status"] == "BLOCKED"
    assert "stale required artifact: docs/runtime/logs/stale.json" in result["failures"]


def test_runtime_lane_readiness_matrix_rolls_up_all_lanes(tmp_path):
    logs = tmp_path / "logs"
    bundle = tmp_path / "bundle"
    _write_preflight_inputs(logs, bundle, host_status="WATCH")

    result = lane_matrix_result(logs)
    report = lane_matrix_render(result)

    assert result["status"] == "WATCH"
    assert [lane["id"] for lane in result["lanes"]] == [
        "moe-routed-shared-scheduling",
        "mamba-projection-dispatch",
        "weighted-moe-ablation",
        "activation-bf16-ablation",
    ]
    assert all(lane["status"] == "WATCH" for lane in result["lanes"])
    assert result["lanes"][0]["kind"] == "speed_candidate"
    assert result["lanes"][2]["kind"] == "negative_control"
    assert "# Nemotron Ultra Runtime Lane Readiness Matrix" in report


def test_runtime_lane_readiness_matrix_blocks_on_stale_manifest(tmp_path):
    logs = tmp_path / "logs"
    bundle = tmp_path / "bundle"
    _write_preflight_inputs(logs, bundle, host_status="READY")
    manifest_path = logs / "2026-06-04-nemotron-ultra-runtime-proof-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["stale_required_artifacts"] = ["docs/runtime/logs/stale.json"]
    _write_json(manifest_path, manifest)

    result = lane_matrix_result(logs)

    assert result["status"] == "BLOCKED"
    assert all(lane["status"] == "BLOCKED" for lane in result["lanes"])


def test_runtime_next_runbook_picks_first_speed_candidate(tmp_path):
    logs = tmp_path / "logs"
    bundle = tmp_path / "bundle"
    _write_preflight_inputs(logs, bundle, host_status="WATCH")
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-lane-readiness-matrix.json", lane_matrix_result(logs))

    result = runbook_result(logs)
    report = runbook_render(result)

    assert result["status"] == "WATCH"
    assert result["next_lane"]["id"] == "moe-routed-shared-scheduling"
    assert result["next_lane"]["kind"] == "speed_candidate"
    assert "run_runtime_candidate_suite.py" in result["next_lane"]["candidate_command"]
    assert any("vMLX server" in item for item in result["host_cleanup"])
    assert "# Nemotron Ultra Runtime Next Runbook" in report


def test_compare_runtime_speed_logs_detects_unchanged_baseline(tmp_path):
    _write_minimal_logs(tmp_path, leaks=False)
    code, report = compare_render(
        argparse.Namespace(
            baseline_log_dir=tmp_path,
            candidate_log_dir=tmp_path,
            out=None,
            max_repeat_fraction=0.25,
            max_tps_regression_pct=2.0,
            max_ms_regression_pct=5.0,
            min_tps_improvement_pct=2.0,
            min_ms_improvement_pct=5.0,
            strict=False,
        )
    )

    assert code == 0
    assert "status: `UNCHANGED`" in report
    assert "delta `+0.000`" in report
    assert "## Failures\n" in report
    json_code, result = compare_result(
        argparse.Namespace(
            baseline_log_dir=tmp_path,
            candidate_log_dir=tmp_path,
            out=None,
            json_out=None,
            max_repeat_fraction=0.25,
            max_tps_regression_pct=2.0,
            max_ms_regression_pct=5.0,
            min_tps_improvement_pct=2.0,
            min_ms_improvement_pct=5.0,
            strict=False,
        )
    )
    assert json_code == 0
    assert result["status"] == "UNCHANGED"
    assert result["metrics"]["deltas"]["best_tps"]["delta"] == 0.0


def test_compare_runtime_speed_logs_detects_candidate_improvement(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_minimal_logs(baseline, tps=8.0, moe=60.0, mamba=55.0, leaks=True)
    _write_minimal_logs(candidate, tps=8.5, moe=50.0, mamba=45.0, leaks=False)

    code, report = compare_render(
        argparse.Namespace(
            baseline_log_dir=baseline,
            candidate_log_dir=candidate,
            out=None,
            max_repeat_fraction=0.25,
            max_tps_regression_pct=2.0,
            max_ms_regression_pct=5.0,
            min_tps_improvement_pct=2.0,
            min_ms_improvement_pct=5.0,
            strict=False,
        )
    )

    assert code == 0
    assert "status: `IMPROVED`" in report
    assert "best_tps improved" in report
    assert "moe_ms improved" in report
    assert "coherence `leaks` count improved 1 -> 0" in report
    json_code, result = compare_result(
        argparse.Namespace(
            baseline_log_dir=baseline,
            candidate_log_dir=candidate,
            out=None,
            json_out=None,
            max_repeat_fraction=0.25,
            max_tps_regression_pct=2.0,
            max_ms_regression_pct=5.0,
            min_tps_improvement_pct=2.0,
            min_ms_improvement_pct=5.0,
            strict=False,
        )
    )
    assert json_code == 0
    assert result["status"] == "IMPROVED"
    assert result["coherence_counts"]["leaks"]["delta"] == -1


def test_refresh_runtime_proof_bundle_runs_no_load_on_minimal_logs(tmp_path):
    _write_minimal_logs(tmp_path, leaks=False)
    bundle = tmp_path / "bundle"
    _write_fake_bundle(bundle)
    _write_json(
        bundle / "config.json",
        {
            "layers_block_type": ["mamba", "moe"] * 48 + ["attention"] * 12,
            "hidden_size": 8192,
            "vocab_size": 131072,
            "tie_word_embeddings": False,
            "num_hidden_layers": 108,
            "num_attention_heads": 64,
            "num_key_value_heads": 2,
            "num_experts_per_tok": 22,
            "moe_intermediate_size": 5120,
            "ssm_state_size": 128,
        },
    )
    summary = tmp_path / "refresh.md"

    proc = subprocess.run(
        [
            sys.executable,
            "jang-tools/examples/nemotron_ultra/refresh_runtime_proof_bundle.py",
            "--log-dir",
            str(tmp_path),
            "--bundle",
            str(bundle),
            "--summary-out",
            str(summary),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    text = summary.read_text()
    assert "# Nemotron Ultra Runtime Proof Refresh" in text
    assert "Runtime status report" in text
    assert "Runtime speed gate" in text
    assert "Runtime speed fix acceptance" in text
    assert "Runtime speed compare" in text
    assert "Runtime cache parser contract" in text
    assert "Host cleanup runbook" in text
    assert "Runtime issue ledger" in text
    assert "Runtime candidate index" in text
    assert "Runtime candidate launch guard" in text
    assert "Runtime cleanup ready check" in text
    assert "Runtime MoE candidate contract" in text
    assert "Runtime MoE execution ticket" in text
    assert "Runtime MoE surface map" in text
    assert "Runtime MoE patch plan" in text
    assert "Runtime MoE delta contract" in text
    assert "Runtime Mamba candidate contract" in text
    assert "Runtime next runbook" in text
    assert "````text" in text
    gate_json = tmp_path / "2026-06-04-nemotron-ultra-runtime-speed-gate.json"
    assert gate_json.exists()
    assert json.loads(gate_json.read_text())["status"] == "PARTIAL"
    acceptance_json = tmp_path / "2026-06-04-nemotron-ultra-runtime-speed-fix-acceptance.json"
    assert acceptance_json.exists()
    assert json.loads(acceptance_json.read_text())["status"] == "PARTIAL"
    compare_json = tmp_path / "2026-06-04-nemotron-ultra-runtime-speed-compare.json"
    assert compare_json.exists()
    assert json.loads(compare_json.read_text())["status"] == "UNCHANGED"
    handoff_json = tmp_path / "2026-06-04-nemotron-ultra-agent-handoff.json"
    assert handoff_json.exists()
    assert json.loads(handoff_json.read_text())["artifact"]["profile"] == "JANGTQ_1L"
    cache_parser_json = tmp_path / "2026-06-04-nemotron-ultra-runtime-cache-parser-contract.json"
    assert cache_parser_json.exists()
    assert json.loads(cache_parser_json.read_text())["cache_contract"]["cache_type"] == "hybrid"
    budget_json = tmp_path / "2026-06-04-nemotron-ultra-token-speed-budget.json"
    assert budget_json.exists()
    assert json.loads(budget_json.read_text())["targets"][0]["target_tps"] == 10.0
    queue_json = tmp_path / "2026-06-04-nemotron-ultra-runtime-experiment-queue.json"
    assert queue_json.exists()
    assert json.loads(queue_json.read_text())["lanes"][0]["id"] == "moe-routed-shared-scheduling"
    manifest_json = tmp_path / "2026-06-04-nemotron-ultra-runtime-proof-manifest.json"
    assert manifest_json.exists()
    assert json.loads(manifest_json.read_text())["status"] == "PARTIAL"
    ledger_json = tmp_path / "2026-06-04-nemotron-ultra-runtime-issue-ledger.json"
    assert ledger_json.exists()
    assert json.loads(ledger_json.read_text())["status"] == "OPEN"
    next_json = tmp_path / "2026-06-04-nemotron-ultra-runtime-next-runbook.json"
    assert next_json.exists()
    assert json.loads(next_json.read_text())["next_lane"]["id"] == "moe-routed-shared-scheduling"
    candidate_index_json = tmp_path / "2026-06-04-nemotron-ultra-runtime-candidate-index.json"
    assert candidate_index_json.exists()
    assert json.loads(candidate_index_json.read_text())["status"] == "OPEN"
    launch_guard_json = tmp_path / "2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.json"
    assert launch_guard_json.exists()
    assert json.loads(launch_guard_json.read_text())["status"] in {"READY", "BLOCKED_BY_WATCH"}
    cleanup_ready_json = tmp_path / "2026-06-04-nemotron-ultra-runtime-cleanup-ready-check.json"
    assert cleanup_ready_json.exists()
    assert json.loads(cleanup_ready_json.read_text())["status"] in {"READY", "WATCH"}
    moe_contract_json = tmp_path / "2026-06-04-nemotron-ultra-runtime-moe-candidate-contract.json"
    assert moe_contract_json.exists()
    assert json.loads(moe_contract_json.read_text())["lane_id"] == "moe-routed-shared-scheduling"
    moe_ticket_json = tmp_path / "2026-06-04-nemotron-ultra-runtime-moe-execution-ticket.json"
    assert moe_ticket_json.exists()
    assert json.loads(moe_ticket_json.read_text())["lane_id"] == "moe-routed-shared-scheduling"
    moe_surface_json = tmp_path / "2026-06-04-nemotron-ultra-runtime-moe-surface-map.json"
    assert moe_surface_json.exists()
    assert json.loads(moe_surface_json.read_text())["lane_id"] == "moe-routed-shared-scheduling"
    moe_patch_plan_json = tmp_path / "2026-06-04-nemotron-ultra-runtime-moe-patch-plan.json"
    assert moe_patch_plan_json.exists()
    assert json.loads(moe_patch_plan_json.read_text())["lane_id"] == "moe-routed-shared-scheduling"
    moe_delta_json = tmp_path / "2026-06-04-nemotron-ultra-runtime-moe-delta-contract.json"
    assert moe_delta_json.exists()
    assert json.loads(moe_delta_json.read_text())["lane_id"] == "moe-routed-shared-scheduling"
    mamba_contract_json = tmp_path / "2026-06-04-nemotron-ultra-runtime-mamba-candidate-contract.json"
    assert mamba_contract_json.exists()
    assert json.loads(mamba_contract_json.read_text())["lane_id"] == "mamba-projection-dispatch"


def test_runtime_speed_gate_cli_writes_json_out(tmp_path):
    _write_minimal_logs(tmp_path)
    json_out = tmp_path / "gate.json"
    md_out = tmp_path / "gate.md"

    proc = subprocess.run(
        [
            sys.executable,
            "jang-tools/examples/nemotron_ultra/runtime_speed_gate.py",
            "--log-dir",
            str(tmp_path),
            "--out",
            str(md_out),
            "--json-out",
            str(json_out),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "status: `PARTIAL`" in md_out.read_text()
    data = json.loads(json_out.read_text())
    assert data["status"] == "PARTIAL"
    assert data["metrics"]["best_live_tps"] == 8.5
    assert data["thresholds"]["min_live_tps"] == 8.0


def test_runtime_proof_manifest_summarizes_artifacts_and_lanes(tmp_path):
    logs = tmp_path / "logs"
    bundle = tmp_path / "bundle"
    _write_minimal_logs(logs, tps=8.5, moe=60.0, mamba=55.0, leaks=False)
    _write_fake_bundle(bundle)
    gate_args = argparse.Namespace(
        log_dir=logs,
        min_live_tps=8.0,
        max_attention_ms=10.0,
        max_norm_lm_ms=5.0,
        min_bottleneck_ms=40.0,
        max_repeat_fraction=0.25,
        strict=False,
        out=None,
    )
    _, gate = speed_gate_result(gate_args)
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-speed-gate.json", gate)
    _write_json(logs / "2026-06-04-nemotron-ultra-token-speed-budget.json", budget_result(logs, [10.0, 12.0]))
    _write_json(
        logs / "2026-06-04-nemotron-ultra-agent-handoff.json",
        handoff_result(
            argparse.Namespace(
                bundle=bundle,
                log_dir=logs,
                out=None,
                json_out=None,
                min_live_tps=8.0,
                max_attention_ms=10.0,
                max_norm_lm_ms=5.0,
                min_bottleneck_ms=40.0,
                max_repeat_fraction=0.25,
                speed_targets=[10.0, 12.0],
            )
        ),
    )
    _write_json(
        logs / "2026-06-04-nemotron-ultra-runtime-experiment-queue.json",
        queue_result(
            argparse.Namespace(
                baseline_log_dir=logs,
                bundle=bundle,
                candidate_root=logs,
                speed_targets=[10.0, 12.0],
                wired_limit_gb=105,
                live_max_tokens=32,
                long_max_tokens=96,
                out=None,
                json_out=None,
            )
        ),
    )

    result = manifest_result(logs)

    assert result["status"] == "PARTIAL"
    assert result["current_metrics"]["best_live_tps"] == 8.5
    assert result["lanes"][0]["id"] == "moe-routed-shared-scheduling"
    assert any(item["role"] == "machine-readable speed gate" for item in result["artifacts"])
    assert any(item["role"] == "MoE runtime delta acceptance contract" for item in result["artifacts"])
    assert any(item["role"] == "cache/parser/runtime nuance contract" for item in result["artifacts"])
    assert "refresh_runtime_proof_bundle.py" in result["commands"]["refresh"]


def test_runtime_proof_manifest_blocks_stale_generated_artifacts(tmp_path):
    logs = tmp_path / "logs"
    bundle = tmp_path / "bundle"
    _write_minimal_logs(logs, tps=8.5, moe=60.0, mamba=55.0, leaks=False)
    _write_fake_bundle(bundle)
    gate_args = argparse.Namespace(
        log_dir=logs,
        min_live_tps=8.0,
        max_attention_ms=10.0,
        max_norm_lm_ms=5.0,
        min_bottleneck_ms=40.0,
        max_repeat_fraction=0.25,
        strict=False,
        out=None,
    )
    _, gate = speed_gate_result(gate_args)
    _write_json(logs / "2026-06-04-nemotron-ultra-runtime-speed-gate.json", gate)
    _write_json(logs / "2026-06-04-nemotron-ultra-token-speed-budget.json", budget_result(logs, [10.0, 12.0]))
    _write_json(
        logs / "2026-06-04-nemotron-ultra-agent-handoff.json",
        handoff_result(
            argparse.Namespace(
                bundle=bundle,
                log_dir=logs,
                out=None,
                json_out=None,
                min_live_tps=8.0,
                max_attention_ms=10.0,
                max_norm_lm_ms=5.0,
                min_bottleneck_ms=40.0,
                max_repeat_fraction=0.25,
                speed_targets=[10.0, 12.0],
            )
        ),
    )
    _write_json(
        logs / "2026-06-04-nemotron-ultra-runtime-experiment-queue.json",
        queue_result(
            argparse.Namespace(
                baseline_log_dir=logs,
                bundle=bundle,
                candidate_root=logs,
                speed_targets=[10.0, 12.0],
                wired_limit_gb=105,
                live_max_tokens=32,
                long_max_tokens=96,
                out=None,
                json_out=None,
            )
        ),
    )
    live_probe = logs / "2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json"
    os.utime(live_probe, (2_000_000_000, 2_000_000_000))

    result = manifest_result(logs)

    assert result["status"] == "BLOCKED"
    assert logs.joinpath("2026-06-04-nemotron-ultra-runtime-speed-gate.json").as_posix() in result[
        "stale_required_artifacts"
    ]


def test_compare_runtime_speed_logs_cli_writes_json_out(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_minimal_logs(baseline, tps=8.0, moe=60.0, mamba=55.0, leaks=True)
    _write_minimal_logs(candidate, tps=8.5, moe=50.0, mamba=45.0, leaks=False)
    json_out = tmp_path / "compare.json"
    md_out = tmp_path / "compare.md"

    proc = subprocess.run(
        [
            sys.executable,
            "jang-tools/examples/nemotron_ultra/compare_runtime_speed_logs.py",
            "--baseline-log-dir",
            str(baseline),
            "--candidate-log-dir",
            str(candidate),
            "--out",
            str(md_out),
            "--json-out",
            str(json_out),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "status: `IMPROVED`" in md_out.read_text()
    data = json.loads(json_out.read_text())
    assert data["status"] == "IMPROVED"
    assert data["metrics"]["deltas"]["best_tps"]["pct"] > 0
    assert data["coherence_counts"]["leaks"]["delta"] == -1


def test_agent_handoff_report_builds_machine_readable_state(tmp_path):
    logs = tmp_path / "logs"
    bundle = tmp_path / "bundle"
    _write_minimal_logs(logs)
    _write_fake_bundle(bundle)

    result = handoff_result(
        argparse.Namespace(
            bundle=bundle,
            log_dir=logs,
            out=None,
            json_out=None,
            min_live_tps=8.0,
            max_attention_ms=10.0,
            max_norm_lm_ms=5.0,
            min_bottleneck_ms=40.0,
            max_repeat_fraction=0.25,
            speed_targets=[10.0, 12.0],
        )
    )

    assert result["handoff_status"] == "PARTIAL"
    assert result["speed_gate"]["status"] == "PARTIAL"
    assert result["artifact"]["capabilities"]["cache_type"] == "hybrid"
    assert result["artifact"]["mxtq_bits"]["routed_expert"]["up_proj"] == 1
    assert result["topology"]["cache_entries"] == 4
    assert result["parser"]["marker_leak_rows"] == ["nt_capital_default"]
    assert result["speed_budget"]["targets"][0]["target_tps"] == 10.0
    assert "MoE routed/shared scheduling or fused decode kernel" in result["next_experiments"]


def test_agent_handoff_report_cli_writes_json_out(tmp_path):
    logs = tmp_path / "logs"
    bundle = tmp_path / "bundle"
    _write_minimal_logs(logs)
    _write_fake_bundle(bundle)
    md_out = tmp_path / "handoff.md"
    json_out = tmp_path / "handoff.json"

    proc = subprocess.run(
        [
            sys.executable,
            "jang-tools/examples/nemotron_ultra/agent_handoff_report.py",
            "--bundle",
            str(bundle),
            "--log-dir",
            str(logs),
            "--out",
            str(md_out),
            "--json-out",
            str(json_out),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "handoff_status: `PARTIAL`" in md_out.read_text()
    data = json.loads(json_out.read_text())
    assert data["cache_and_modality_gates"]["text_only"] is True
    assert data["topology"]["attention_kv_cache_entries"] == 2
    assert data["speed_budget"]["targets"][0]["required_total_cut_ms"] == 30.0


def test_token_speed_budget_calculates_target_ms_cuts(tmp_path):
    _write_minimal_logs(tmp_path, tps=8.5, moe=60.0, mamba=55.0, leaks=False)

    result = budget_result(tmp_path, [10.0, 12.5])

    assert result["current"]["manual_decode_total_ms"] == 130.0
    assert result["current"]["best_live_tps"] == 8.5
    ten_tps = result["targets"][0]
    assert ten_tps["target_tps"] == 10.0
    assert ten_tps["target_ms_per_token"] == 100.0
    assert ten_tps["required_total_cut_ms"] == 30.0
    assert round(ten_tps["moe_cut_ms_proportional"], 6) == round(30.0 * 60.0 / 115.0, 6)
    assert ten_tps["reachable_by_moe_mamba_only"] is True


def test_token_speed_budget_cli_writes_json_out(tmp_path):
    _write_minimal_logs(tmp_path, tps=8.5, moe=60.0, mamba=55.0, leaks=False)
    md_out = tmp_path / "budget.md"
    json_out = tmp_path / "budget.json"

    proc = subprocess.run(
        [
            sys.executable,
            "jang-tools/examples/nemotron_ultra/token_speed_budget.py",
            "--log-dir",
            str(tmp_path),
            "--targets",
            "10,12.5",
            "--out",
            str(md_out),
            "--json-out",
            str(json_out),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "`10.000`" in md_out.read_text()
    data = json.loads(json_out.read_text())
    assert data["targets"][1]["target_tps"] == 12.5
    assert data["targets"][1]["target_ms_per_token"] == 80.0


def test_runtime_experiment_queue_builds_candidate_lanes(tmp_path):
    logs = tmp_path / "logs"
    bundle = tmp_path / "bundle"
    candidate_root = tmp_path / "candidates"
    _write_minimal_logs(logs, tps=8.5, moe=60.0, mamba=55.0, leaks=False)
    _write_fake_bundle(bundle)

    result = queue_result(
        argparse.Namespace(
            baseline_log_dir=logs,
            bundle=bundle,
            candidate_root=candidate_root,
            speed_targets=[10.0, 12.0],
            wired_limit_gb=105,
            live_max_tokens=32,
            long_max_tokens=96,
            out=None,
            json_out=None,
        )
    )

    assert result["current"]["best_live_tps"] == 8.5
    assert [lane["id"] for lane in result["lanes"]][:2] == [
        "moe-routed-shared-scheduling",
        "mamba-projection-dispatch",
    ]
    assert "run_runtime_candidate_suite.py" in result["lanes"][0]["command"]
    assert "candidate-moe-scheduling" in result["lanes"][0]["command"]
    assert result["lanes"][0]["status"] == "OPEN"
    assert result["lanes"][0]["candidate_log_dir"] == str(candidate_root / "candidate-moe-scheduling")
    assert result["lanes"][0]["run_command"] == result["lanes"][0]["command"]
    assert result["lanes"][0]["dry_run_command"] == result["lanes"][0]["command"] + " --dry-run"
    assert "--lane-id moe-routed-shared-scheduling" in result["lanes"][0]["command"]
    assert "--queue-json" in result["lanes"][0]["command"]
    assert result["lanes"][0]["kind"] == "speed_candidate"
    assert result["lanes"][0]["expected_compare_statuses"] == ["IMPROVED"]
    assert result["lanes"][2]["env"]["JANGTQ_DISABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH"] == "1"
    assert result["lanes"][2]["kind"] == "negative_control"
    assert result["lanes"][2]["expected_compare_statuses"] == ["FAIL", "UNCHANGED"]
    assert result["lanes"][2]["command"].startswith("JANGTQ_DISABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH=1 PYTHONPATH=jang-tools")
    assert "experiment_result_check.py" in result["lanes"][0]["post_check_command"]
    assert "2026-06-04-nemotron-ultra-runtime-speed-compare.json" in result["lanes"][0]["required_outputs"]
    assert "2026-06-04-nemotron-ultra-runtime-cache-parser-contract.json" in result["lanes"][0]["required_outputs"]


def test_runtime_experiment_queue_cli_writes_json_out(tmp_path):
    logs = tmp_path / "logs"
    bundle = tmp_path / "bundle"
    _write_minimal_logs(logs, tps=8.5, moe=60.0, mamba=55.0, leaks=False)
    _write_fake_bundle(bundle)
    md_out = tmp_path / "queue.md"
    json_out = tmp_path / "queue.json"

    proc = subprocess.run(
        [
            sys.executable,
            "jang-tools/examples/nemotron_ultra/runtime_experiment_queue.py",
            "--baseline-log-dir",
            str(logs),
            "--bundle",
            str(bundle),
            "--candidate-root",
            str(tmp_path / "candidates"),
            "--out",
            str(md_out),
            "--json-out",
            str(json_out),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "MoE routed/shared scheduling" in md_out.read_text()
    data = json.loads(json_out.read_text())
    assert data["lanes"][1]["id"] == "mamba-projection-dispatch"
    assert data["lanes"][3]["env"]["JANGTQ_DISABLE_NEMOTRON_ACTIVATION_BF16"] == "1"
    assert data["lanes"][3]["expected_compare_statuses"] == ["FAIL"]
    assert data["lanes"][3]["command"].startswith("JANGTQ_DISABLE_NEMOTRON_ACTIVATION_BF16=1 PYTHONPATH=jang-tools")
    assert "activation-bf16-ablation" in data["lanes"][3]["post_check_command"]


def test_experiment_result_check_accepts_improved_moe_lane(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    bundle = tmp_path / "bundle"
    queue = tmp_path / "queue.json"
    _write_minimal_logs(baseline, tps=8.0, moe=60.0, mamba=55.0, leaks=False)
    _write_minimal_logs(candidate, tps=8.5, moe=50.0, mamba=55.0, leaks=False)
    _write_fake_bundle(bundle)
    _write_queue(queue, baseline, bundle, tmp_path)
    _write_compare_gate_handoff(candidate, baseline)

    code, result = check_result(
        argparse.Namespace(
            queue_json=queue,
            lane_id="moe-routed-shared-scheduling",
            candidate_log_dir=candidate,
            out=None,
            json_out=None,
            strict=True,
        )
    )

    assert code == 0
    assert result["status"] == "ACCEPTED"
    assert result["compare_status"] == "IMPROVED"


def test_experiment_result_check_rejects_wrong_lane_metric(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    bundle = tmp_path / "bundle"
    queue = tmp_path / "queue.json"
    _write_minimal_logs(baseline, tps=8.0, moe=60.0, mamba=55.0, leaks=False)
    _write_minimal_logs(candidate, tps=8.5, moe=60.0, mamba=45.0, leaks=False)
    _write_fake_bundle(bundle)
    _write_queue(queue, baseline, bundle, tmp_path)
    _write_compare_gate_handoff(candidate, baseline)

    code, result = check_result(
        argparse.Namespace(
            queue_json=queue,
            lane_id="moe-routed-shared-scheduling",
            candidate_log_dir=candidate,
            out=None,
            json_out=None,
            strict=True,
        )
    )

    assert code == 1
    assert result["status"] == "REJECTED"
    assert "MoE lane did not improve moe_ms" in result["failures"]


def test_experiment_result_check_blocks_missing_cache_parser_contract(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    bundle = tmp_path / "bundle"
    queue = tmp_path / "queue.json"
    _write_minimal_logs(baseline, tps=8.0, moe=60.0, mamba=55.0, leaks=False)
    _write_minimal_logs(candidate, tps=8.5, moe=50.0, mamba=55.0, leaks=False)
    _write_fake_bundle(bundle)
    _write_queue(queue, baseline, bundle, tmp_path)
    _write_compare_gate_handoff(candidate, baseline)
    (candidate / "2026-06-04-nemotron-ultra-runtime-cache-parser-contract.json").unlink()

    code, result = check_result(
        argparse.Namespace(
            queue_json=queue,
            lane_id="moe-routed-shared-scheduling",
            candidate_log_dir=candidate,
            out=None,
            json_out=None,
            strict=True,
        )
    )

    assert code == 2
    assert result["status"] == "BLOCKED"
    assert "2026-06-04-nemotron-ultra-runtime-cache-parser-contract.json" in result["missing_outputs"]


def test_experiment_result_check_blocks_missing_outputs(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    bundle = tmp_path / "bundle"
    queue = tmp_path / "queue.json"
    _write_minimal_logs(baseline, tps=8.0, moe=60.0, mamba=55.0, leaks=False)
    _write_minimal_logs(candidate, tps=8.5, moe=50.0, mamba=55.0, leaks=False)
    (candidate / "2026-06-04-nemotron-ultra-projection-tradeoff-probe.json").unlink()
    _write_fake_bundle(bundle)
    _write_queue(queue, baseline, bundle, tmp_path)

    code, result = check_result(
        argparse.Namespace(
            queue_json=queue,
            lane_id="moe-routed-shared-scheduling",
            candidate_log_dir=candidate,
            out=None,
            json_out=None,
            strict=True,
        )
    )

    assert code == 2
    assert result["status"] == "BLOCKED"
    assert "2026-06-04-nemotron-ultra-projection-tradeoff-probe.json" in result["missing_outputs"]


def test_candidate_runtime_suite_skip_model_probes_is_smoke_only(tmp_path):
    candidate = tmp_path / "candidate"

    proc = subprocess.run(
        [
            sys.executable,
            "jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py",
            "--candidate-log-dir",
            str(candidate),
            "--baseline-log-dir",
            str(tmp_path / "baseline"),
            "--skip-model-probes",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = candidate / "2026-06-04-nemotron-ultra-candidate-runtime-suite.md"
    text = summary.read_text()
    assert "skipped by --skip-model-probes" in text
    assert "report/compare steps also skipped" in text
    assert "Runtime speed gate" not in text


def test_candidate_runtime_suite_skip_model_probes_accepts_lane_args_without_checking(tmp_path):
    candidate = tmp_path / "candidate"
    queue = tmp_path / "queue.json"
    _write_json(queue, {"lanes": []})

    proc = subprocess.run(
        [
            sys.executable,
            "jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py",
            "--candidate-log-dir",
            str(candidate),
            "--baseline-log-dir",
            str(tmp_path / "baseline"),
            "--queue-json",
            str(queue),
            "--lane-id",
            "moe-routed-shared-scheduling",
            "--skip-model-probes",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    text = (candidate / "2026-06-04-nemotron-ultra-candidate-runtime-suite.md").read_text()
    assert "skipped by --skip-model-probes" in text
    assert "Experiment result check" not in text


def test_candidate_runtime_suite_passes_bundle_to_all_model_probes(tmp_path):
    bundle = tmp_path / "candidate-bundle"
    out = tmp_path / "candidate"
    args = argparse.Namespace(
        candidate_log_dir=out,
        bundle=bundle,
        wired_limit_gb=105,
        live_max_tokens=32,
        long_max_tokens=96,
    )

    commands = candidate_suite._model_commands(args)

    assert len(commands) == 6
    for label, command in commands:
        assert "--bundle" in command, label
        assert command[command.index("--bundle") + 1] == str(bundle), label


def test_candidate_runtime_suite_report_commands_include_compare_and_lane_check(tmp_path):
    args = argparse.Namespace(
        candidate_log_dir=tmp_path / "candidate",
        baseline_log_dir=tmp_path / "baseline",
        bundle=tmp_path / "bundle",
        queue_json=tmp_path / "queue.json",
        lane_id="moe-routed-shared-scheduling",
        strict_gate=False,
    )

    commands = candidate_suite._report_commands(args)

    labels = [label for label, _ in commands]
    assert "Baseline vs candidate compare" in labels
    assert "Experiment result check" in labels
    experiment = dict(commands)["Experiment result check"]
    assert "--queue-json" in experiment
    assert experiment[experiment.index("--queue-json") + 1] == str(args.queue_json)
    assert "--lane-id" in experiment
    assert experiment[experiment.index("--lane-id") + 1] == args.lane_id


def test_candidate_runtime_suite_dry_run_writes_planned_commands_without_running_model(tmp_path):
    candidate = tmp_path / "candidate"

    proc = subprocess.run(
        [
            sys.executable,
            "jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py",
            "--candidate-log-dir",
            str(candidate),
            "--baseline-log-dir",
            str(tmp_path / "baseline"),
            "--bundle",
            str(tmp_path / "bundle"),
            "--queue-json",
            str(tmp_path / "queue.json"),
            "--lane-id",
            "moe-routed-shared-scheduling",
            "--dry-run",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    text = (candidate / "2026-06-04-nemotron-ultra-candidate-runtime-suite.md").read_text()
    assert "dry run only; no commands executed" in text
    assert "live_speed_probe.py" in text
    assert "experiment_result_check.py" in text
    assert str(tmp_path / "bundle") in text


def test_validate_runtime_log_bundle_accepts_complete_minimal_logs(tmp_path):
    _write_minimal_logs(tmp_path)
    code, report = validate_bundle(tmp_path)

    assert code == 0
    assert "status: `FIXED`" in report
    assert "found live speed log" in report
    assert "## Failures\n" in report


def test_validate_runtime_log_bundle_blocks_missing_component_logs(tmp_path):
    _write_minimal_logs(tmp_path)
    (tmp_path / "2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json").unlink()

    code, report = validate_bundle(tmp_path)

    assert code == 2
    assert "status: `BLOCKED`" in report
    assert "missing moe component log" in report
