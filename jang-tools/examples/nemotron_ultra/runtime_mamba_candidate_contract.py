"""Generate the Mamba candidate contract for Nemotron Ultra speed work.

This no-load helper narrows the broader patch spec to the second speed lane:
Mamba projection/dispatch. It records invariants, target cuts, blocked
preconditions, and acceptance checks for the runtime implementation pass after
MoE scheduling evidence exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_LOG_DIR = Path("docs/runtime/logs")
DEFAULT_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-mamba-candidate-contract.md")
DEFAULT_JSON_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-mamba-candidate-contract.json")
LANE_ID = "mamba-projection-dispatch"


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _find_lane(data: dict[str, Any], lane_id: str) -> dict[str, Any]:
    for lane in data.get("lanes", []):
        if lane.get("id") == lane_id:
            return lane
    return {}


def _build_result(log_dir: Path) -> dict[str, Any]:
    patch_path = log_dir / "2026-06-04-nemotron-ultra-runtime-patch-spec.json"
    shape_path = log_dir / "2026-06-04-nemotron-ultra-runtime-shape-contract.json"
    budget_path = log_dir / "2026-06-04-nemotron-ultra-token-speed-budget.json"
    issue_path = log_dir / "2026-06-04-nemotron-ultra-runtime-issue-ledger.json"
    guard_path = log_dir / "2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.json"
    candidate_index_path = log_dir / "2026-06-04-nemotron-ultra-runtime-candidate-index.json"

    patch = _load(patch_path) or {}
    shape = _load(shape_path) or {}
    budget = _load(budget_path) or {}
    issue_ledger = _load(issue_path) or {}
    guard = _load(guard_path) or {}
    candidate_index = _load(candidate_index_path) or {}

    lane = _find_lane(patch, LANE_ID)
    issue = next((item for item in issue_ledger.get("issues", []) if item.get("id") == "NU-SPEED-002"), {})
    current = budget.get("current", {})
    targets = budget.get("targets") or [{}, {}]
    second_target = targets[1] if len(targets) > 1 else targets[0]
    mamba_contract = shape.get("mamba_contract", {})
    quant = shape.get("quantization", {})
    moe_row = _find_lane(candidate_index, "moe-routed-shared-scheduling")

    preconditions = []
    if guard.get("status") != "READY":
        preconditions.append(f"candidate launch guard is {guard.get('status', 'UNKNOWN')}")
    if shape.get("status") != "READY":
        preconditions.append(f"shape contract is {shape.get('status', 'UNKNOWN')}")
    if moe_row.get("status") != "ACCEPTED":
        preconditions.append(f"MoE lane evidence is {moe_row.get('status', 'MISSING')}; run/accept MoE lane before Mamba lane")

    return {
        "status": "BLOCKED" if preconditions else "READY",
        "lane_id": LANE_ID,
        "log_dir": str(log_dir),
        "current_speed": {
            "mamba_ms": current.get("mamba_ms"),
            "moe_ms": current.get("moe_ms"),
            "manual_decode_total_ms": current.get("manual_decode_total_ms"),
            "best_live_tps": current.get("best_live_tps"),
        },
        "target": {
            "target_tps": second_target.get("target_tps"),
            "required_total_cut_ms": second_target.get("required_total_cut_ms"),
            "mamba_cut_ms_proportional": second_target.get("mamba_cut_ms_proportional"),
            "mamba_cut_pct_of_current_mamba": second_target.get("mamba_cut_pct_of_current_mamba"),
            "mamba_per_layer_cut_ms": second_target.get("mamba_per_layer_cut_ms"),
        },
        "invariants": {
            "hidden_shape": mamba_contract.get("hidden_shape"),
            "normed_shape": mamba_contract.get("normed_shape"),
            "projected_shape": mamba_contract.get("projected_shape"),
            "gate_shape": mamba_contract.get("gate_shape"),
            "conv_dim": mamba_contract.get("conv_dim"),
            "conv_input_shape": mamba_contract.get("conv_input_shape"),
            "conv_output_shape": mamba_contract.get("conv_output_shape"),
            "ssm_out_shape": mamba_contract.get("ssm_out_shape"),
            "intermediate_size": mamba_contract.get("intermediate_size"),
            "num_heads": mamba_contract.get("num_heads"),
            "n_groups": mamba_contract.get("n_groups"),
            "ssm_state_size": mamba_contract.get("ssm_state_size"),
            "mamba_projection_bits": quant.get("mxtq_bits", {}).get("mamba_projection"),
            "fp8_projection_affine_bits": quant.get("fp8_projection_affine_bits"),
            "fp8_projection_group_size": quant.get("fp8_projection_group_size"),
            "drops_mtp": quant.get("drops_mtp"),
        },
        "preconditions": preconditions,
        "do": lane.get("do", []),
        "do_not": lane.get("do_not", []),
        "acceptance_checks": issue.get("acceptance_checks", []),
        "candidate_command": lane.get("proof", {}).get("candidate_command"),
        "post_check_command": lane.get("proof", {}).get("post_check_command"),
        "required_outputs": lane.get("proof", {}).get("required_outputs", []),
        "source_files": [
            str(patch_path),
            str(shape_path),
            str(budget_path),
            str(issue_path),
            str(guard_path),
            str(candidate_index_path),
        ],
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _render(result: dict[str, Any]) -> str:
    lines = [
        "# Nemotron Ultra Mamba Candidate Contract",
        "",
        f"log_dir: `{result['log_dir']}`",
        f"lane_id: `{result['lane_id']}`",
        f"status: `{result['status']}`",
        "",
        "## Current Speed",
    ]
    for key, value in result["current_speed"].items():
        lines.append(f"- {key}: `{_fmt(value)}`")
    lines.extend(["", "## Target"])
    for key, value in result["target"].items():
        lines.append(f"- {key}: `{_fmt(value)}`")
    lines.extend(["", "## Mamba Invariants"])
    for key, value in result["invariants"].items():
        lines.append(f"- {key}: `{_fmt(value)}`")
    lines.extend(["", "## Preconditions"])
    lines.extend(f"- {item}" for item in result["preconditions"] or ["none"])
    lines.extend(["", "## Do"])
    lines.extend(f"- {item}" for item in result["do"])
    lines.extend(["", "## Do Not"])
    lines.extend(f"- {item}" for item in result["do_not"])
    lines.extend(["", "## Acceptance Checks"])
    lines.extend(f"- {item}" for item in result["acceptance_checks"])
    lines.extend(["", "## Commands"])
    lines.append(f"- candidate: `{result.get('candidate_command')}`")
    lines.append(f"- post_check: `{result.get('post_check_command')}`")
    lines.extend(["", "## Required Outputs"])
    lines.extend(f"- `{item}`" for item in result["required_outputs"])
    lines.extend(["", "## Source Files"])
    lines.extend(f"- `{item}`" for item in result["source_files"])
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
