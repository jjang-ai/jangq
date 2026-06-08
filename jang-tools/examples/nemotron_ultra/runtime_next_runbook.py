"""Generate the next-run runbook for Nemotron Ultra runtime speed work."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_LOG_DIR = Path("docs/runtime/logs")
DEFAULT_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-next-runbook.md")
DEFAULT_JSON_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-next-runbook.json")


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _first_speed_lane(matrix: dict[str, Any]) -> dict[str, Any] | None:
    for lane in matrix.get("lanes", []):
        if lane.get("kind") == "speed_candidate" and int(lane.get("failure_count", 0)) == 0:
            return lane
    return None


def _build_result(log_dir: Path) -> dict[str, Any]:
    matrix = _load(log_dir / "2026-06-04-nemotron-ultra-runtime-lane-readiness-matrix.json") or {}
    manifest = _load(log_dir / "2026-06-04-nemotron-ultra-runtime-proof-manifest.json") or {}
    host = _load(log_dir / "2026-06-04-nemotron-ultra-host-runtime-readiness.json") or {}
    patch = _load(log_dir / "2026-06-04-nemotron-ultra-runtime-patch-spec.json") or {}
    shape = _load(log_dir / "2026-06-04-nemotron-ultra-runtime-shape-contract.json") or {}
    lane = _first_speed_lane(matrix) or {}
    next_lane_id = lane.get("id")
    patch_lane = {}
    for item in patch.get("lanes", []):
        if item.get("id") == next_lane_id:
            patch_lane = item
            break
    status = "BLOCKED" if matrix.get("status") == "BLOCKED" else matrix.get("status", "UNKNOWN")
    return {
        "status": status,
        "log_dir": str(log_dir),
        "current_runtime_status": manifest.get("status"),
        "host_status": host.get("status"),
        "shape_status": shape.get("status"),
        "next_lane": {
            "id": next_lane_id,
            "kind": lane.get("kind"),
            "status": lane.get("status"),
            "title": lane.get("title"),
            "warnings": lane.get("warnings", []),
            "failures": lane.get("failures", []),
            "candidate_command": lane.get("candidate_command"),
            "post_check_command": lane.get("post_check_command"),
            "why": patch_lane.get("why", []),
            "do": patch_lane.get("do", []),
            "do_not": patch_lane.get("do_not", []),
        },
        "host_cleanup": [
            "Close or stop the high-RSS vMLX server before loading the 98G Nemotron bundle.",
            "Rerun host_runtime_readiness.py and runtime_lane_readiness_matrix.py after cleanup.",
            "Proceed with the candidate command only when the selected lane is READY, or consciously accept WATCH noise.",
        ],
        "proof_sequence": [
            "Refresh no-load proof bundle.",
            "Run runtime_lane_readiness_matrix.py.",
            "Run exactly one speed_candidate lane.",
            "Run that lane's post_check_command.",
            "Accept only IMPROVED compare status with no long-coherence/cache/modality regressions.",
        ],
    }


def _render(result: dict[str, Any]) -> str:
    lane = result["next_lane"]
    lines = [
        "# Nemotron Ultra Runtime Next Runbook",
        "",
        f"log_dir: `{result['log_dir']}`",
        f"runbook_status: `{result['status']}`",
        f"current_runtime_status: `{result['current_runtime_status']}`",
        f"host_status: `{result['host_status']}`",
        f"shape_status: `{result['shape_status']}`",
        "",
        "## Next Lane",
        f"- id: `{lane.get('id')}`",
        f"- kind: `{lane.get('kind')}`",
        f"- status: `{lane.get('status')}`",
        f"- title: {lane.get('title')}",
        "",
        "## Why This Lane",
    ]
    lines.extend(f"- {item}" for item in lane.get("why", []) or ["no patch-spec lane found"])
    lines.extend(["", "## Host Cleanup"])
    lines.extend(f"- {item}" for item in result["host_cleanup"])
    if lane.get("warnings"):
        lines.extend(["", "## Current Warnings"])
        lines.extend(f"- {item}" for item in lane["warnings"])
    if lane.get("failures"):
        lines.extend(["", "## Current Failures"])
        lines.extend(f"- {item}" for item in lane["failures"])
    lines.extend(["", "## Do"])
    lines.extend(f"- {item}" for item in lane.get("do", []))
    lines.extend(["", "## Do Not"])
    lines.extend(f"- {item}" for item in lane.get("do_not", []))
    lines.extend(["", "## Commands"])
    lines.append(f"- candidate: `{lane.get('candidate_command')}`")
    lines.append(f"- post_check: `{lane.get('post_check_command')}`")
    lines.extend(["", "## Proof Sequence"])
    lines.extend(f"- {item}" for item in result["proof_sequence"])
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
