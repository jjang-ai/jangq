"""Generate a no-load manifest for Nemotron Ultra runtime speed proof files.

The manifest is the first file to open before runtime-speed work. It points to
the current status, core metrics, generated proof artifacts, candidate lanes,
and exact commands needed to refresh or continue the proof bundle.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_LOG_DIR = Path("docs/runtime/logs")
DEFAULT_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-proof-manifest.md")
DEFAULT_JSON_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-proof-manifest.json")


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _artifact(path: Path, role: str, required: bool = True) -> dict[str, Any]:
    mtime = path.stat().st_mtime if path.exists() else None
    return {
        "path": str(path),
        "role": role,
        "required": required,
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
        "mtime": mtime,
        "freshness": "missing" if not path.exists() else "current",
    }


def _mark_stale_generated_artifacts(artifacts: list[dict[str, Any]]) -> list[str]:
    source_roles = {
        "best live decode source",
        "layer decode bucket source",
        "long coherence source",
        "Mamba component source",
        "MoE component source",
        "projection tradeoff source",
    }
    generated_required_roles = {
        "target token/s millisecond budgets",
        "machine-readable experiment queue",
        "machine-readable speed gate",
        "machine-readable downstream agent handoff",
    }
    source_mtimes = [
        item["mtime"]
        for item in artifacts
        if item["role"] in source_roles and item["mtime"] is not None
    ]
    if not source_mtimes:
        return []
    latest_source_mtime = max(source_mtimes)
    stale: list[str] = []
    for item in artifacts:
        if item["role"] not in generated_required_roles or item["mtime"] is None:
            continue
        if item["mtime"] < latest_source_mtime:
            item["freshness"] = "stale"
            item["stale_after"] = latest_source_mtime
            stale.append(item["path"])
    return stale


def _build_result(log_dir: Path) -> dict[str, Any]:
    gate_path = log_dir / "2026-06-04-nemotron-ultra-runtime-speed-gate.json"
    budget_path = log_dir / "2026-06-04-nemotron-ultra-token-speed-budget.json"
    handoff_path = log_dir / "2026-06-04-nemotron-ultra-agent-handoff.json"
    queue_path = log_dir / "2026-06-04-nemotron-ultra-runtime-experiment-queue.json"
    compare_path = log_dir / "2026-06-04-nemotron-ultra-runtime-speed-compare.json"

    gate = _load(gate_path) or {}
    budget = _load(budget_path) or {}
    handoff = _load(handoff_path) or {}
    queue = _load(queue_path) or {}
    metrics = gate.get("metrics", {})
    current = budget.get("current", {})
    lanes = [
        {
            "id": lane.get("id"),
            "kind": lane.get("kind"),
            "title": lane.get("title"),
            "command": lane.get("command"),
            "post_check_command": lane.get("post_check_command"),
            "expected_compare_statuses": lane.get("expected_compare_statuses", []),
        }
        for lane in queue.get("lanes", [])
    ]
    artifacts = [
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-proof-refresh.md", "combined no-load refresh output", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-log-bundle-validation.md", "required log presence validation", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-status-report.md", "human-readable current status", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-host-runtime-readiness.json", "host memory/disk readiness snapshot", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-host-runtime-readiness.md", "human-readable host readiness snapshot", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-host-cleanup-runbook.json", "host cleanup runbook", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-host-cleanup-runbook.md", "human-readable host cleanup runbook", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-speed-experiment-plan.md", "ranked speed experiment plan", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-issue-ledger.json", "runtime issue ledger", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-issue-ledger.md", "human-readable runtime issue ledger", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-candidate-index.json", "runtime candidate index", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-candidate-index.md", "human-readable runtime candidate index", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.json", "runtime candidate launch guard", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.md", "human-readable runtime candidate launch guard", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-cleanup-ready-check.json", "runtime cleanup ready check", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-cleanup-ready-check.md", "human-readable runtime cleanup ready check", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-moe-candidate-contract.json", "MoE candidate contract", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-moe-candidate-contract.md", "human-readable MoE candidate contract", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-moe-execution-ticket.json", "MoE execution ticket", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-moe-execution-ticket.md", "human-readable MoE execution ticket", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-moe-surface-map.json", "MoE runtime source surface map", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-moe-surface-map.md", "human-readable MoE runtime source surface map", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-moe-patch-plan.json", "MoE runtime patch plan", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-moe-patch-plan.md", "human-readable MoE runtime patch plan", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-moe-delta-contract.json", "MoE runtime delta acceptance contract", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-moe-delta-contract.md", "human-readable MoE runtime delta acceptance contract", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-mamba-candidate-contract.json", "Mamba candidate contract", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-mamba-candidate-contract.md", "human-readable Mamba candidate contract", False),
        _artifact(budget_path, "target token/s millisecond budgets"),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-token-speed-budget.md", "human-readable target budgets", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-component-budget-matrix.json", "component-level token/s sensitivity matrix", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-component-budget-matrix.md", "human-readable component budget matrix", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-patch-spec.json", "implementation-facing runtime patch spec", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-patch-spec.md", "human-readable runtime patch spec", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-shape-contract.json", "runtime shape and bit contract", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-shape-contract.md", "human-readable runtime shape contract", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-candidate-preflight.json", "runtime candidate preflight", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-candidate-preflight.md", "human-readable candidate preflight", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-lane-readiness-matrix.json", "all-lane runtime readiness matrix", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-lane-readiness-matrix.md", "human-readable lane readiness matrix", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-next-runbook.json", "next runtime runbook", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-next-runbook.md", "human-readable next runtime runbook", False),
        _artifact(queue_path, "machine-readable experiment queue"),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-experiment-queue.md", "human-readable experiment queue", False),
        _artifact(gate_path, "machine-readable speed gate"),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-speed-gate.md", "human-readable speed gate", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-speed-fix-acceptance.json", "runtime speed fix acceptance audit", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-speed-fix-acceptance.md", "human-readable speed fix acceptance audit", False),
        _artifact(compare_path, "baseline-vs-baseline compare for current logs", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-speed-compare.md", "human-readable compare report", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-cache-parser-contract.json", "cache/parser/runtime nuance contract", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-runtime-cache-parser-contract.md", "human-readable cache/parser/runtime nuance contract", False),
        _artifact(handoff_path, "machine-readable downstream agent handoff"),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-agent-handoff.md", "human-readable downstream agent handoff", False),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json", "best live decode source"),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json", "layer decode bucket source"),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json", "long coherence source"),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-mamba-component-probe.json", "Mamba component source"),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json", "MoE component source"),
        _artifact(log_dir / "2026-06-04-nemotron-ultra-projection-tradeoff-probe.json", "projection tradeoff source"),
    ]
    missing_required = [item["path"] for item in artifacts if item["required"] and not item["exists"]]
    stale_required = _mark_stale_generated_artifacts(artifacts)
    return {
        "log_dir": str(log_dir),
        "status": "BLOCKED" if missing_required or stale_required else gate.get("status", "UNKNOWN"),
        "missing_required_artifacts": missing_required,
        "stale_required_artifacts": stale_required,
        "current_metrics": {
            "best_live_tps": metrics.get("best_live_tps") or current.get("best_live_tps"),
            "best_live_source": metrics.get("best_live_source") or current.get("best_live_source"),
            "manual_decode_total_ms": metrics.get("manual_decode_total_ms") or current.get("manual_decode_total_ms"),
            "manual_implied_tps": current.get("manual_implied_tps"),
            "moe_ms": metrics.get("moe_ms") or current.get("moe_ms"),
            "mamba_ms": metrics.get("mamba_ms") or current.get("mamba_ms"),
            "attention_ms": metrics.get("attention_ms") or current.get("attention_ms"),
            "norm_lm_head_ms": metrics.get("norm_lm_head_ms") or current.get("norm_lm_head_ms"),
            "moe_plus_mamba_pct_of_total": current.get("moe_plus_mamba_pct_of_total"),
        },
        "fixed": gate.get("fixed", []),
        "partial": gate.get("partial", []),
        "artifact_capabilities": handoff.get("artifact", {}).get("capabilities", {}),
        "topology": handoff.get("topology", {}),
        "lanes": lanes,
        "artifacts": artifacts,
        "commands": {
            "refresh": (
                "PYTHONPATH=jang-tools jang-tools/.venv/bin/python "
                "jang-tools/examples/nemotron_ultra/refresh_runtime_proof_bundle.py "
                f"--log-dir {log_dir} "
                f"--summary-out {log_dir / '2026-06-04-nemotron-ultra-runtime-proof-refresh.md'}"
            ),
            "strict_gate": (
                "PYTHONPATH=jang-tools jang-tools/.venv/bin/python "
                "jang-tools/examples/nemotron_ultra/runtime_speed_gate.py "
                f"--log-dir {log_dir} --strict"
            ),
        },
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _render(result: dict[str, Any]) -> str:
    metrics = result["current_metrics"]
    lines = [
        "# Nemotron Ultra Runtime Proof Manifest",
        "",
        f"log_dir: `{result['log_dir']}`",
        f"status: `{result['status']}`",
        "",
        "## Current Metrics",
    ]
    for key in (
        "best_live_tps",
        "manual_decode_total_ms",
        "manual_implied_tps",
        "moe_ms",
        "mamba_ms",
        "attention_ms",
        "norm_lm_head_ms",
        "moe_plus_mamba_pct_of_total",
    ):
        lines.append(f"- {key}: `{_fmt(metrics.get(key))}`")
    if metrics.get("best_live_source"):
        lines.append(f"- best_live_source: `{metrics['best_live_source']}`")
    lines.extend(["", "## Fixed Evidence"])
    lines.extend(f"- {item}" for item in result["fixed"])
    lines.extend(["", "## Partial Evidence"])
    lines.extend(f"- {item}" for item in result["partial"])
    lines.extend(["", "## Lanes"])
    for lane in result["lanes"]:
        lines.append(
            f"- `{lane['id']}` ({lane['kind']}): {lane['title']}; expected={lane['expected_compare_statuses']}"
        )
    lines.extend(["", "## Commands"])
    for name, command in result["commands"].items():
        lines.append(f"- {name}: `{command}`")
    lines.extend(["", "## Artifacts"])
    for item in result["artifacts"]:
        state = "present" if item["exists"] else "missing"
        if item["freshness"] == "stale":
            state = "stale"
        required = "required" if item["required"] else "optional"
        lines.append(f"- `{item['path']}`: {state}, {required}, {item['role']}")
    if result["missing_required_artifacts"]:
        lines.extend(["", "## Missing Required Artifacts"])
        lines.extend(f"- `{item}`" for item in result["missing_required_artifacts"])
    if result["stale_required_artifacts"]:
        lines.extend(["", "## Stale Required Artifacts"])
        lines.extend(f"- `{item}`" for item in result["stale_required_artifacts"])
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    result = _build_result(args.log_dir)
    report = _render(result)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    sys.stdout.write(report)


if __name__ == "__main__":
    main()
