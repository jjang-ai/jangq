"""Build a no-load runtime issue ledger for Nemotron Ultra JANGTQ_1L.

This reads saved proof JSON files and writes a compact issue list for runtime
agents. It does not load the model and does not edit any runtime implementation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_LOG_DIR = Path("docs/runtime/logs")
DEFAULT_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-issue-ledger.md")
DEFAULT_JSON_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-issue-ledger.json")


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def _fmt(value: Any, suffix: str = "") -> str:
    if isinstance(value, float):
        return f"{value:.3f}{suffix}"
    if isinstance(value, int):
        return f"{value}{suffix}"
    if value is None:
        return "unknown"
    return f"{value}{suffix}"


def _issue(
    *,
    issue_id: str,
    status: str,
    severity: str,
    title: str,
    evidence: list[str],
    next_actions: list[str],
    acceptance_checks: list[str],
    source_files: list[str],
) -> dict[str, Any]:
    return {
        "id": issue_id,
        "status": status,
        "severity": severity,
        "title": title,
        "evidence": evidence,
        "next_actions": next_actions,
        "acceptance_checks": acceptance_checks,
        "source_files": source_files,
    }


def _target_summary(budget: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for target in budget.get("targets", []):
        tps = target.get("target_tps")
        required = target.get("required_total_cut_ms")
        moe = target.get("moe_cut_ms_proportional")
        mamba = target.get("mamba_cut_ms_proportional")
        lines.append(
            f"{_fmt(tps)} tok/s needs {_fmt(required, ' ms')} total cut "
            f"(MoE {_fmt(moe, ' ms')}, Mamba {_fmt(mamba, ' ms')} proportional)."
        )
    return lines


def _build_result(log_dir: Path) -> dict[str, Any]:
    gate_path = log_dir / "2026-06-04-nemotron-ultra-runtime-speed-gate.json"
    budget_path = log_dir / "2026-06-04-nemotron-ultra-token-speed-budget.json"
    host_path = log_dir / "2026-06-04-nemotron-ultra-host-runtime-readiness.json"
    next_path = log_dir / "2026-06-04-nemotron-ultra-runtime-next-runbook.json"
    shape_path = log_dir / "2026-06-04-nemotron-ultra-runtime-shape-contract.json"

    gate = _load(gate_path)
    budget = _load(budget_path)
    host = _load(host_path)
    next_runbook = _load(next_path)
    shape = _load(shape_path)

    metrics = gate.get("metrics", {})
    next_lane = next_runbook.get("next_lane", {})
    issues: list[dict[str, Any]] = []

    moe_ms = metrics.get("moe_ms")
    mamba_ms = metrics.get("mamba_ms")
    attention_ms = metrics.get("attention_ms")
    norm_lm_ms = metrics.get("norm_lm_head_ms")

    issues.append(
        _issue(
            issue_id="NU-SPEED-001",
            status="OPEN" if isinstance(moe_ms, (int, float)) and moe_ms >= 40.0 else "FIXED",
            severity="critical",
            title="MoE routed/shared decode path dominates token latency",
            evidence=[
                f"MoE bucket is {_fmt(moe_ms, ' ms')}.",
                "Current next speed lane is "
                f"`{next_lane.get('id', 'unknown')}`: {next_lane.get('title', 'unknown')}.",
                *_target_summary(budget)[:1],
            ],
            next_actions=[
                "Run exactly one MoE scheduling candidate after host readiness is READY or WATCH is accepted.",
                "Preserve router top-k, weighted expert semantics, routed 1-bit experts, and shared 8-bit experts.",
                "Compare candidate logs with compare_runtime_speed_logs.py and experiment_result_check.py.",
            ],
            acceptance_checks=[
                "runtime-speed compare status is IMPROVED.",
                "MoE bucket drops enough to move target token/s budget without Mamba or coherence regression.",
                "long coherence leak/repeat/EOS counts do not regress.",
            ],
            source_files=[
                str(gate_path),
                str(budget_path),
                str(next_path),
            ],
        )
    )

    issues.append(
        _issue(
            issue_id="NU-SPEED-002",
            status="OPEN" if isinstance(mamba_ms, (int, float)) and mamba_ms >= 40.0 else "FIXED",
            severity="critical",
            title="Mamba projection/dispatch path dominates token latency",
            evidence=[
                f"Mamba bucket is {_fmt(mamba_ms, ' ms')}.",
                "Speed gate records projection/dispatch as the current Mamba target before conv rewrite.",
                *_target_summary(budget)[1:2],
            ],
            next_actions=[
                "Keep this as the second speed-candidate lane after MoE scheduling evidence is gathered.",
                "Preserve projected shape [1, 1, 35072], gate shape [1, 1, 16384], SSM state size 128, groups 8.",
                "Recheck mamba_component_probe.py and layer_decode_probe.py after any candidate.",
            ],
            acceptance_checks=[
                "Mamba bucket drops without changing cache cardinality or Mamba shape contract.",
                "Attention and norm/lm_head remain below current ceilings.",
                "long coherence leak/repeat/EOS counts do not regress.",
            ],
            source_files=[
                str(gate_path),
                str(budget_path),
                str(shape_path),
            ],
        )
    )

    partial_text = " ".join(str(item) for item in gate.get("partial", []))
    coherence_open = "coherence gate remains partial" in partial_text
    issues.append(
        _issue(
            issue_id="NU-COHERENCE-001",
            status="OPEN" if coherence_open else "FIXED",
            severity="high",
            title="Long decode still leaks/repeats or misses EOS",
            evidence=[partial_text or "Speed gate did not report coherence partials."],
            next_actions=[
                "Treat coherence as a regression gate for speed candidates, not a sampler/prompt masking target.",
                "Use long_decode_coherence_probe.py after any accepted speed candidate.",
            ],
            acceptance_checks=[
                "No visible thinking marker leaks.",
                "Repeat fraction stays below gate threshold.",
                "Expected rows reach EOS within the probe limit.",
            ],
            source_files=[
                str(gate_path),
                str(log_dir / "2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json"),
            ],
        )
    )

    host_status = host.get("status", "UNKNOWN")
    warnings = host.get("warnings", [])
    issues.append(
        _issue(
            issue_id="NU-HOST-001",
            status="OPEN" if host_status == "WATCH" else ("BLOCKED" if host_status == "BLOCKED" else "FIXED"),
            severity="medium",
            title="Host RAM/process state can add noise to expensive probes",
            evidence=[
                f"host_runtime_readiness status is `{host_status}`.",
                *(str(item) for item in warnings),
            ],
            next_actions=[
                "Use host_cleanup_runbook.py before loading the 98G bundle.",
                "Rerun host_runtime_readiness.py, runtime_lane_readiness_matrix.py, and runtime_next_runbook.py after cleanup.",
            ],
            acceptance_checks=[
                "Host readiness is READY, or WATCH is explicitly accepted before candidate timing.",
                "No unrelated high-RSS model server is competing with the Nemotron candidate run.",
            ],
            source_files=[
                str(host_path),
                str(log_dir / "2026-06-04-nemotron-ultra-host-cleanup-runbook.json"),
            ],
        )
    )

    fixed_evidence = gate.get("fixed", [])
    issues.append(
        _issue(
            issue_id="NU-FIXED-001",
            status="FIXED" if fixed_evidence else "OPEN",
            severity="info",
            title="Already-fixed runtime buckets must stay fixed",
            evidence=[
                f"attention bucket is {_fmt(attention_ms, ' ms')}.",
                f"norm/lm_head bucket is {_fmt(norm_lm_ms, ' ms')}.",
                *[str(item) for item in fixed_evidence],
            ],
            next_actions=[
                "Do not prioritize attention or lm_head while MoE/Mamba remain above bottleneck threshold.",
                "Keep these buckets in every compare report as regression checks.",
            ],
            acceptance_checks=[
                "Attention remains below 10 ms.",
                "norm/lm_head remains below 5 ms.",
            ],
            source_files=[str(gate_path)],
        )
    )

    status_counts: dict[str, int] = {}
    for item in issues:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1

    overall = "BLOCKED" if status_counts.get("BLOCKED") else ("OPEN" if status_counts.get("OPEN") else "FIXED")
    return {
        "log_dir": str(log_dir),
        "status": overall,
        "current_runtime_status": gate.get("status", "UNKNOWN"),
        "target_summary": _target_summary(budget),
        "issues": issues,
        "status_counts": status_counts,
        "commands": {
            "refresh_manifest": (
                "PYTHONPATH=jang-tools jang-tools/.venv/bin/python "
                "jang-tools/examples/nemotron_ultra/runtime_proof_manifest.py "
                f"--log-dir {log_dir}"
            ),
            "rerun_ledger": (
                "PYTHONPATH=jang-tools jang-tools/.venv/bin/python "
                "jang-tools/examples/nemotron_ultra/runtime_issue_ledger.py "
                f"--log-dir {log_dir}"
            ),
        },
    }


def _render(result: dict[str, Any]) -> str:
    lines = [
        "# Nemotron Ultra Runtime Issue Ledger",
        "",
        f"log_dir: `{result['log_dir']}`",
        f"status: `{result['status']}`",
        f"current_runtime_status: `{result['current_runtime_status']}`",
        "",
        "## Status Counts",
    ]
    for status in ("OPEN", "BLOCKED", "FIXED"):
        lines.append(f"- {status}: `{result['status_counts'].get(status, 0)}`")
    lines.extend(["", "## Target Summary"])
    lines.extend(f"- {item}" for item in result["target_summary"])
    lines.extend(["", "## Issues"])
    for issue in result["issues"]:
        lines.extend(
            [
                "",
                f"### {issue['id']}: {issue['title']}",
                f"- status: `{issue['status']}`",
                f"- severity: `{issue['severity']}`",
                "- evidence:",
            ]
        )
        lines.extend(f"  - {item}" for item in issue["evidence"] if item)
        lines.append("- next_actions:")
        lines.extend(f"  - {item}" for item in issue["next_actions"])
        lines.append("- acceptance_checks:")
        lines.extend(f"  - {item}" for item in issue["acceptance_checks"])
        lines.append("- source_files:")
        lines.extend(f"  - `{item}`" for item in issue["source_files"])
    lines.extend(["", "## Commands"])
    for name, command in result["commands"].items():
        lines.append(f"- {name}: `{command}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
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
