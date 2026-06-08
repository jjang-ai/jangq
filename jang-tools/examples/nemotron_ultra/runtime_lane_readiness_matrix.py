"""Build a no-load readiness matrix for all Nemotron Ultra runtime lanes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from examples.nemotron_ultra.runtime_candidate_preflight import _build_result as _preflight_result


DEFAULT_LOG_DIR = Path("docs/runtime/logs")
DEFAULT_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-lane-readiness-matrix.md")
DEFAULT_JSON_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-lane-readiness-matrix.json")


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _build_result(log_dir: Path) -> dict[str, Any]:
    queue_path = log_dir / "2026-06-04-nemotron-ultra-runtime-experiment-queue.json"
    queue = _load(queue_path) or {}
    lanes = []
    for lane in queue.get("lanes", []):
        lane_id = str(lane.get("id"))
        preflight = _preflight_result(log_dir, lane_id)
        lanes.append(
            {
                "id": lane_id,
                "kind": lane.get("kind"),
                "title": lane.get("title"),
                "status": preflight["status"],
                "warning_count": len(preflight.get("warnings", [])),
                "failure_count": len(preflight.get("failures", [])),
                "expected_compare_statuses": lane.get("expected_compare_statuses", []),
                "candidate_command": preflight.get("candidate_command"),
                "post_check_command": preflight.get("post_check_command"),
                "warnings": preflight.get("warnings", []),
                "failures": preflight.get("failures", []),
            }
        )
    statuses = [lane["status"] for lane in lanes]
    status = "BLOCKED" if "BLOCKED" in statuses else ("WATCH" if "WATCH" in statuses else "READY")
    return {
        "status": status,
        "log_dir": str(log_dir),
        "queue": str(queue_path),
        "lanes": lanes,
        "interpretation": [
            "Run speed_candidate lanes for proposed runtime fixes; run negative_control lanes as guards after related changes.",
            "WATCH means proof wiring is usable but host/readiness warnings should be handled before loading the 98G bundle.",
            "BLOCKED means missing or stale proof files must be refreshed before a candidate suite.",
        ],
    }


def _render(result: dict[str, Any]) -> str:
    lines = [
        "# Nemotron Ultra Runtime Lane Readiness Matrix",
        "",
        f"log_dir: `{result['log_dir']}`",
        f"status: `{result['status']}`",
        f"queue: `{result['queue']}`",
        "",
        "## Lanes",
        "| lane | kind | status | warnings | failures | expected compare |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for lane in result["lanes"]:
        expected = ", ".join(f"`{item}`" for item in lane["expected_compare_statuses"])
        lines.append(
            f"| `{lane['id']}` | `{lane['kind']}` | `{lane['status']}` | `{lane['warning_count']}` | `{lane['failure_count']}` | {expected} |"
        )
    lines.extend(["", "## Commands"])
    for lane in result["lanes"]:
        lines.append(f"- `{lane['id']}` candidate: `{lane.get('candidate_command')}`")
        lines.append(f"- `{lane['id']}` post_check: `{lane.get('post_check_command')}`")
    lines.extend(["", "## Warnings And Failures"])
    for lane in result["lanes"]:
        lines.append(f"### {lane['id']}")
        if lane["warnings"]:
            lines.extend(f"- warning: {item}" for item in lane["warnings"])
        else:
            lines.append("- warning: none")
        if lane["failures"]:
            lines.extend(f"- failure: {item}" for item in lane["failures"])
        else:
            lines.append("- failure: none")
    lines.extend(["", "## Interpretation"])
    lines.extend(f"- {item}" for item in result["interpretation"])
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
        raise SystemExit(2 if result["status"] == "BLOCKED" else 1)


if __name__ == "__main__":
    main()
