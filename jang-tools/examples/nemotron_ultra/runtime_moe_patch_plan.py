"""Generate the no-load MoE patch plan for Nemotron Ultra speed work.

This helper turns the current MoE surface map, component budget, ticket, and
candidate contract into a concrete implementation checklist. It does not edit
runtime code and does not load the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_LOG_DIR = Path("docs/runtime/logs")
DEFAULT_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-patch-plan.md")
DEFAULT_JSON_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-patch-plan.json")


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _surface(surface: dict[str, Any], surface_id: str) -> dict[str, Any]:
    for row in surface.get("surfaces", []):
        if row.get("id") == surface_id:
            return row
    return {}


def _moe_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in matrix.get("component_rows", []) if row.get("family") == "MoE"]
    return sorted(rows, key=lambda row: float(row.get("projected_total_ms") or 0.0), reverse=True)


def _scenario(row: dict[str, Any], cut_pct: float) -> dict[str, Any]:
    scenarios = row.get("scenarios", [])
    if not scenarios:
        return {}
    return min(scenarios, key=lambda item: abs(float(item.get("cut_pct", 0.0)) - cut_pct))


def _step(
    *,
    step_id: str,
    title: str,
    component: dict[str, Any],
    goal: str,
    surfaces: list[dict[str, Any]],
    validation: list[str],
    non_goals: list[str],
) -> dict[str, Any]:
    scenario_25 = _scenario(component, 25.0)
    scenario_50 = _scenario(component, 50.0)
    return {
        "id": step_id,
        "title": title,
        "component": {
            "label": component.get("label"),
            "role": component.get("role"),
            "median_ms_per_measured_layer": component.get("median_ms_per_measured_layer"),
            "projected_total_ms": component.get("projected_total_ms"),
            "coverage_pct_of_family_total": component.get("coverage_pct_of_family_total"),
            "tps_after_25pct_cut": scenario_25.get("new_manual_tps"),
            "tps_after_50pct_cut": scenario_50.get("new_manual_tps"),
        },
        "goal": goal,
        "surfaces": [
            {
                "id": surface.get("id"),
                "file": surface.get("file"),
                "anchors": surface.get("anchors", {}),
            }
            for surface in surfaces
        ],
        "validation": validation,
        "non_goals": non_goals,
    }


def _build_result(log_dir: Path) -> dict[str, Any]:
    surface_path = log_dir / "2026-06-04-nemotron-ultra-runtime-moe-surface-map.json"
    matrix_path = log_dir / "2026-06-04-nemotron-ultra-component-budget-matrix.json"
    contract_path = log_dir / "2026-06-04-nemotron-ultra-runtime-moe-candidate-contract.json"
    ticket_path = log_dir / "2026-06-04-nemotron-ultra-runtime-moe-execution-ticket.json"
    acceptance_path = log_dir / "2026-06-04-nemotron-ultra-runtime-speed-fix-acceptance.json"

    surface = _load(surface_path) or {}
    matrix = _load(matrix_path) or {}
    contract = _load(contract_path) or {}
    ticket = _load(ticket_path) or {}
    acceptance = _load(acceptance_path) or {}

    failures: list[str] = []
    if surface.get("status") != "READY":
        failures.append(f"MoE surface map is {surface.get('status', 'MISSING')}")
    if contract.get("status") != "READY":
        failures.append(f"MoE candidate contract is {contract.get('status', 'MISSING')}")
    if ticket.get("status") != "READY":
        failures.append(f"MoE execution ticket is {ticket.get('status', 'MISSING')}")

    rows = {row.get("label"): row for row in _moe_rows(matrix)}
    for label in ("full_moe", "switch_mlp", "shared_experts"):
        if label not in rows:
            failures.append(f"missing MoE component budget row: {label}")
    routed_label = "weighted_decode" if "weighted_decode" in rows else "switch_mlp"

    loader = _surface(surface, "nemotron-weighted-moe-patch")
    switch_toggle = _surface(surface, "switchmlp-fastpath-toggle")
    gather = _surface(surface, "routed-gather-kernel")
    fused = _surface(surface, "fused-gate-up-kernel")
    grouped = _surface(surface, "grouped-nax-proof-surface")
    component_probe = _surface(surface, "moe-component-proof")
    verdict = _surface(surface, "candidate-verdict-proof")

    steps: list[dict[str, Any]] = []
    if "full_moe" in rows:
        steps.append(
            _step(
                step_id="moe-01-path-scheduling",
                title="Reduce full MoE path scheduling overhead first",
                component=rows["full_moe"],
                goal="Attack the inclusive NemotronHMoE path before isolated micro-optimizations.",
                surfaces=[loader, switch_toggle, component_probe, verdict],
                validation=[
                    "rerun moe_component_probe.py and require full_moe plus switch_mlp timing to move",
                    "rerun layer_decode_probe.py and require E bucket improvement",
                    "rerun long_decode_coherence_probe.py and reject marker/repeat/EOS regression",
                ],
                non_goals=[
                    "do not lower top-k",
                    "do not bypass weighted expert scores",
                    "do not count a short live row as acceptance without experiment_result_check",
                ],
            )
        )
    if routed_label in rows:
        routed_validation = (
            [
                "compare weighted_decode against switch_mlp timings through moe_component_probe.py",
                "run candidate suite with weighted-MoE enabled, then compare against activation-BF16 negative controls",
                "preserve routed_expert_bits up/down=1 and indices shape [1,1,22]",
            ]
            if routed_label == "weighted_decode"
            else [
                "compare fused_gate_up and gather path timings through moe_component_probe.py",
                "run candidate suite with default env, then compare against weighted-MoE and activation-BF16 negative controls",
                "preserve routed_expert_bits up/down=1 and indices shape [1,1,22]",
            ]
        )
        steps.append(
            _step(
                step_id="moe-02-switchmlp-routed-kernels",
                title="Optimize SwitchMLP routed gate/up/down execution",
                component=rows[routed_label],
                goal="Reduce routed 1-bit expert dispatch around fused gate/up and gather down kernels.",
                surfaces=[fused, gather, grouped, component_probe, verdict],
                validation=routed_validation,
                non_goals=[
                    "do not expand routed experts to BF16",
                    "do not make grouped NAX the default without candidate proof",
                    "do not edit vMLX or MLX Studio for this JANG lane",
                ],
            )
        )
    if "shared_experts" in rows:
        steps.append(
            _step(
                step_id="moe-03-shared-experts-overlap",
                title="Measure shared expert overlap only after routed path moves",
                component=rows["shared_experts"],
                goal="Treat shared experts as secondary unless routed scheduling leaves them dominant.",
                surfaces=[loader, component_probe, verdict],
                validation=[
                    "require shared_experts timing to improve without increasing switch_mlp",
                    "preserve shared_expert_bits=8",
                    "keep speed acceptance PARTIAL unless token/s target, bucket ceilings, and candidate acceptance all pass",
                ],
                non_goals=[
                    "do not dequantize shared experts by default",
                    "do not optimize shared path before routed path evidence",
                ],
            )
        )

    status = "BLOCKED" if failures else "READY"
    return {
        "status": status,
        "log_dir": str(log_dir),
        "lane_id": contract.get("lane_id") or surface.get("lane_id"),
        "speed_acceptance_status": acceptance.get("status"),
        "current_speed": contract.get("current_speed", {}),
        "target": contract.get("target", {}),
        "invariants": contract.get("invariants", {}),
        "steps": steps,
        "failures": failures,
        "candidate_command": ticket.get("commands", {}).get("candidate") or contract.get("candidate_command"),
        "post_check_command": ticket.get("commands", {}).get("post_check") or contract.get("post_check_command"),
        "post_candidate_refresh": ticket.get("commands", {}).get("post_candidate_refresh"),
        "acceptance_command": (
            "PYTHONPATH=jang-tools jang-tools/.venv/bin/python "
            "jang-tools/examples/nemotron_ultra/runtime_speed_fix_acceptance.py "
            f"--log-dir {log_dir} --strict"
        ),
        "source_files": [
            str(surface_path),
            str(matrix_path),
            str(contract_path),
            str(ticket_path),
            str(acceptance_path),
        ],
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _render(result: dict[str, Any]) -> str:
    lines = [
        "# Nemotron Ultra MoE Patch Plan",
        "",
        f"log_dir: `{result['log_dir']}`",
        f"lane_id: `{result.get('lane_id')}`",
        f"status: `{result['status']}`",
        f"speed_acceptance_status: `{result.get('speed_acceptance_status')}`",
        "",
        "## Current Speed",
    ]
    for key, value in result["current_speed"].items():
        lines.append(f"- {key}: `{_fmt(value)}`")
    lines.extend(["", "## Target"])
    for key, value in result["target"].items():
        lines.append(f"- {key}: `{_fmt(value)}`")
    lines.extend(["", "## Ordered Steps"])
    for step in result["steps"]:
        component = step["component"]
        lines.extend(
            [
                f"### {step['id']}: {step['title']}",
                f"- component: `{component.get('label')}` ({component.get('role')})",
                f"- projected_total_ms: `{_fmt(component.get('projected_total_ms'))}`",
                f"- 25pct_cut_tps: `{_fmt(component.get('tps_after_25pct_cut'))}`",
                f"- 50pct_cut_tps: `{_fmt(component.get('tps_after_50pct_cut'))}`",
                f"- goal: {step['goal']}",
                "- surfaces:",
            ]
        )
        for surface in step["surfaces"]:
            anchors = ", ".join(f"{anchor}@{line}" for anchor, line in surface.get("anchors", {}).items())
            lines.append(f"  - `{surface.get('id')}`: `{surface.get('file')}` ({anchors})")
        lines.append("- validation:")
        lines.extend(f"  - {item}" for item in step["validation"])
        lines.append("- non_goals:")
        lines.extend(f"  - {item}" for item in step["non_goals"])
    lines.extend(["", "## Commands"])
    lines.append(f"- candidate: `{result.get('candidate_command')}`")
    lines.append(f"- post_check: `{result.get('post_check_command')}`")
    lines.append(f"- post_candidate_refresh: `{result.get('post_candidate_refresh')}`")
    lines.append(f"- acceptance: `{result.get('acceptance_command')}`")
    lines.extend(["", "## Failures"])
    lines.extend(f"- {item}" for item in result["failures"] or ["none"])
    lines.extend(["", "## Source Files"])
    lines.extend(f"- `{item}`" for item in result["source_files"])
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    ap.add_argument("--strict", action="store_true")
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
    if args.strict and result["status"] != "READY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
