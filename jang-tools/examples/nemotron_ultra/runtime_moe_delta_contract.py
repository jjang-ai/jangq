"""No-load delta contract for the first Nemotron Ultra MoE speed candidate.

This report turns the current saved baseline into concrete pass/fail numbers
for the MoE lane. It does not load the model and does not edit runtime code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_LOG_DIR = Path("docs/runtime/logs")
DEFAULT_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-delta-contract.md")
DEFAULT_JSON_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-delta-contract.json")


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _target_for_tps(budget: dict[str, Any], target_tps: float) -> dict[str, Any]:
    targets = budget.get("targets", [])
    if not targets:
        return {}
    return min(targets, key=lambda row: abs(float(row.get("target_tps", 0.0)) - target_tps))


def _status_counts(index: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for lane in index.get("lanes", []):
        status = str(lane.get("status", "UNKNOWN"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _build_result(log_dir: Path, *, target_tps: float = 10.0) -> dict[str, Any]:
    contract_path = log_dir / "2026-06-04-nemotron-ultra-runtime-moe-candidate-contract.json"
    budget_path = log_dir / "2026-06-04-nemotron-ultra-token-speed-budget.json"
    acceptance_path = log_dir / "2026-06-04-nemotron-ultra-runtime-speed-fix-acceptance.json"
    patch_plan_path = log_dir / "2026-06-04-nemotron-ultra-runtime-moe-patch-plan.json"
    ticket_path = log_dir / "2026-06-04-nemotron-ultra-runtime-moe-execution-ticket.json"
    candidate_index_path = log_dir / "2026-06-04-nemotron-ultra-runtime-candidate-index.json"

    contract = _load(contract_path)
    budget = _load(budget_path)
    acceptance = _load(acceptance_path)
    patch_plan = _load(patch_plan_path)
    ticket = _load(ticket_path)
    candidate_index = _load(candidate_index_path)

    failures: list[str] = []
    for label, data in (
        ("MoE candidate contract", contract),
        ("token speed budget", budget),
        ("speed fix acceptance", acceptance),
        ("MoE patch plan", patch_plan),
        ("MoE execution ticket", ticket),
        ("candidate index", candidate_index),
    ):
        if data is None:
            failures.append(f"missing {label}")

    if contract is not None and contract.get("status") != "READY":
        failures.append(f"MoE candidate contract is {contract.get('status')}")
    if patch_plan is not None and patch_plan.get("status") != "READY":
        failures.append(f"MoE patch plan is {patch_plan.get('status')}")
    if ticket is not None and ticket.get("status") != "READY":
        failures.append(f"MoE execution ticket is {ticket.get('status')}")

    current = (budget or {}).get("current", {})
    target = _target_for_tps(budget or {}, target_tps)
    moe_ms = current.get("moe_ms")
    moe_cut = target.get("moe_cut_ms_proportional")
    target_moe_ms = None
    if isinstance(moe_ms, (int, float)) and isinstance(moe_cut, (int, float)):
        target_moe_ms = float(moe_ms) - float(moe_cut)

    candidate_command = (ticket or {}).get("commands", {}).get("candidate")
    post_check = (ticket or {}).get("commands", {}).get("post_check")
    post_candidate_index = (ticket or {}).get("commands", {}).get("post_candidate_index")
    post_candidate_refresh = (ticket or {}).get("commands", {}).get("post_candidate_refresh")
    acceptance_command = (
        "PYTHONPATH=jang-tools jang-tools/.venv/bin/python "
        "jang-tools/examples/nemotron_ultra/runtime_speed_fix_acceptance.py "
        f"--log-dir {log_dir} --strict"
    )

    status = "BLOCKED" if failures else "READY"
    return {
        "status": status,
        "log_dir": str(log_dir),
        "lane_id": (contract or {}).get("lane_id") or (ticket or {}).get("lane_id"),
        "baseline": {
            "best_live_tps": current.get("best_live_tps"),
            "manual_decode_total_ms": current.get("manual_decode_total_ms"),
            "manual_implied_tps": current.get("manual_implied_tps"),
            "moe_ms": current.get("moe_ms"),
            "mamba_ms": current.get("mamba_ms"),
            "attention_ms": current.get("attention_ms"),
            "norm_lm_head_ms": current.get("norm_lm_head_ms"),
        },
        "target": {
            "target_tps": target.get("target_tps", target_tps),
            "target_ms_per_token": target.get("target_ms_per_token"),
            "required_total_cut_ms": target.get("required_total_cut_ms"),
            "moe_cut_ms_proportional": target.get("moe_cut_ms_proportional"),
            "moe_cut_pct_of_current_moe": target.get("moe_cut_pct_of_current_moe"),
            "target_moe_ms_for_proportional_10tps": target_moe_ms,
            "acceptance_max_moe_ms": (acceptance or {}).get("max_moe_ms", 40.0),
            "acceptance_max_mamba_ms": (acceptance or {}).get("max_mamba_ms", 40.0),
        },
        "acceptance_thresholds": {
            "experiment_result_status": "ACCEPTED",
            "compare_status": "IMPROVED",
            "gate_status": "FIXED for final speed acceptance; PARTIAL only means candidate moved one bucket",
            "best_live_tps": f">= {target_tps:.3f} for final speed acceptance",
            "moe_ms": (
                "must improve versus baseline and should fall below "
                f"{target_moe_ms:.3f} ms for a 10 tok/s trajectory"
                if target_moe_ms is not None
                else "must improve versus baseline"
            ),
            "final_moe_ceiling_ms": (acceptance or {}).get("max_moe_ms", 40.0),
            "final_mamba_ceiling_ms": (acceptance or {}).get("max_mamba_ms", 40.0),
            "coherence": "no regression in leak, repeat, or EOS counts",
            "regression_guards": [
                "Mamba ms must not materially regress",
                "attention ms must stay under fixed gate ceiling",
                "norm/lm_head ms must stay under fixed gate ceiling",
                "parser/tool/reasoning behavior must not be hidden by prompt or sampler guards",
            ],
        },
        "negative_controls": [
            "weighted-moe-ablation is diagnostic only and must not be promoted as a speed fix",
            "activation-bf16-ablation is diagnostic only and must not be promoted as a speed fix",
        ],
        "invariants": (contract or {}).get("invariants", {}),
        "ordered_steps": [
            {
                "id": step.get("id"),
                "component": step.get("component", {}).get("label"),
                "projected_total_ms": step.get("component", {}).get("projected_total_ms"),
                "tps_after_25pct_cut": step.get("component", {}).get("tps_after_25pct_cut"),
                "tps_after_50pct_cut": step.get("component", {}).get("tps_after_50pct_cut"),
            }
            for step in (patch_plan or {}).get("steps", [])
        ],
        "candidate_index": {
            "status": (candidate_index or {}).get("status"),
            "status_counts": (candidate_index or {}).get("status_counts") or _status_counts(candidate_index or {}),
        },
        "commands": {
            "candidate": candidate_command,
            "post_check": post_check,
            "post_candidate_index": post_candidate_index,
            "post_candidate_refresh": post_candidate_refresh,
            "acceptance_strict": acceptance_command,
        },
        "failures": failures,
        "source_files": [
            str(contract_path),
            str(budget_path),
            str(acceptance_path),
            str(patch_plan_path),
            str(ticket_path),
            str(candidate_index_path),
        ],
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _render(result: dict[str, Any]) -> str:
    lines = [
        "# Nemotron Ultra MoE Delta Contract",
        "",
        f"log_dir: `{result['log_dir']}`",
        f"lane_id: `{result.get('lane_id')}`",
        f"status: `{result['status']}`",
        "",
        "## Baseline",
    ]
    for key, value in result["baseline"].items():
        lines.append(f"- {key}: `{_fmt(value)}`")
    lines.extend(["", "## Target"])
    for key, value in result["target"].items():
        lines.append(f"- {key}: `{_fmt(value)}`")
    lines.extend(["", "## Acceptance Thresholds"])
    for key, value in result["acceptance_thresholds"].items():
        if isinstance(value, list):
            lines.append(f"- {key}:")
            lines.extend(f"  - {item}" for item in value)
        else:
            lines.append(f"- {key}: `{_fmt(value)}`")
    lines.extend(["", "## Ordered MoE Steps"])
    for step in result["ordered_steps"]:
        lines.append(
            f"- `{step.get('id')}` component=`{step.get('component')}` "
            f"projected_total_ms=`{_fmt(step.get('projected_total_ms'))}` "
            f"25pct_tps=`{_fmt(step.get('tps_after_25pct_cut'))}` "
            f"50pct_tps=`{_fmt(step.get('tps_after_50pct_cut'))}`"
        )
    lines.extend(["", "## Negative Controls"])
    lines.extend(f"- {item}" for item in result["negative_controls"])
    lines.extend(["", "## Commands"])
    for key, value in result["commands"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Failures"])
    lines.extend(f"- {item}" for item in result["failures"] or ["none"])
    lines.extend(["", "## Source Files"])
    lines.extend(f"- `{item}`" for item in result["source_files"])
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--target-tps", type=float, default=10.0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    result = _build_result(args.log_dir, target_tps=args.target_tps)
    report = _render(result)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    sys.stdout.write(report)
    if args.strict and result["status"] != "READY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
