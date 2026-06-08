"""Index Nemotron Ultra runtime candidate proof directories.

This no-load helper scans candidate log directories for compare, gate, and
experiment-result JSON files. It provides a single summary of which speed lanes
are accepted, rejected, blocked, or still missing proof.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_LOG_DIR = Path("docs/runtime/logs")
DEFAULT_QUEUE = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json")
DEFAULT_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-index.md")
DEFAULT_JSON_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-index.json")


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _lane_candidate_dir(log_dir: Path, command: str | None) -> Path | None:
    if not command:
        return None
    parts = command.split()
    for index, part in enumerate(parts):
        if part == "--candidate-log-dir" and index + 1 < len(parts):
            return Path(parts[index + 1])
    return None


def _find_result_json(candidate_dir: Path) -> Path | None:
    direct = candidate_dir / "2026-06-04-nemotron-ultra-experiment-result-check.json"
    if direct.exists():
        return direct
    matches = sorted(candidate_dir.glob("*experiment-result-check.json"))
    return matches[0] if matches else None


def _metric_delta(compare: dict[str, Any] | None, key: str) -> dict[str, Any] | None:
    if compare is None:
        return None
    delta = compare.get("metrics", {}).get("deltas", {}).get(key)
    return delta if isinstance(delta, dict) else None


def _row_for_lane(log_dir: Path, lane: dict[str, Any]) -> dict[str, Any]:
    candidate_dir = _lane_candidate_dir(log_dir, lane.get("command"))
    if candidate_dir is None:
        return {
            "id": lane.get("id"),
            "kind": lane.get("kind"),
            "title": lane.get("title"),
            "candidate_log_dir": None,
            "status": "MISSING",
            "reason": "lane command does not contain --candidate-log-dir",
        }

    compare_path = candidate_dir / "2026-06-04-nemotron-ultra-runtime-speed-compare.json"
    gate_path = candidate_dir / "2026-06-04-nemotron-ultra-runtime-speed-gate.json"
    result_path = _find_result_json(candidate_dir)
    compare = _load(compare_path)
    gate = _load(gate_path)
    result_check = _load(result_path) if result_path else None

    missing = []
    if not candidate_dir.exists():
        missing.append("candidate directory")
    if compare is None:
        missing.append("speed compare JSON")
    if gate is None:
        missing.append("speed gate JSON")
    if result_check is None:
        missing.append("experiment result check JSON")

    if missing:
        status = "MISSING"
        reason = f"{candidate_dir}: missing " + ", ".join(missing)
    else:
        status = str(result_check.get("status", "UNKNOWN"))
        reason = ""

    return {
        "id": lane.get("id"),
        "kind": lane.get("kind"),
        "title": lane.get("title"),
        "candidate_log_dir": str(candidate_dir),
        "status": status,
        "reason": reason,
        "expected_compare_statuses": lane.get("expected_compare_statuses", []),
        "compare_status": compare.get("status") if compare else None,
        "gate_status": gate.get("status") if gate else None,
        "best_tps_delta": _metric_delta(compare, "best_tps"),
        "moe_ms_delta": _metric_delta(compare, "moe_ms"),
        "mamba_ms_delta": _metric_delta(compare, "mamba_ms"),
        "coherence_counts": compare.get("coherence_counts") if compare else None,
        "failures": result_check.get("failures", []) if result_check else [],
        "missing_outputs": result_check.get("missing_outputs", []) if result_check else [],
        "result_json": str(result_path) if result_path else None,
    }


def _build_result(log_dir: Path, queue_json: Path) -> dict[str, Any]:
    queue = _load(queue_json)
    if queue is None:
        return {
            "status": "BLOCKED",
            "log_dir": str(log_dir),
            "queue_json": str(queue_json),
            "lanes": [],
            "status_counts": {"BLOCKED": 1},
            "failures": [f"missing queue JSON: {queue_json}"],
        }

    lanes = [_row_for_lane(log_dir, lane) for lane in queue.get("lanes", [])]
    counts: dict[str, int] = {}
    for lane in lanes:
        counts[lane["status"]] = counts.get(lane["status"], 0) + 1
    status = "BLOCKED" if counts.get("BLOCKED") else ("OPEN" if counts.get("MISSING") or counts.get("REJECTED") else "READY")
    if counts.get("ACCEPTED"):
        status = "READY" if status == "READY" else status
    return {
        "status": status,
        "log_dir": str(log_dir),
        "queue_json": str(queue_json),
        "lanes": lanes,
        "status_counts": counts,
        "failures": [],
        "interpretation": [
            "MISSING means the lane has not produced enough no-load proof to accept or reject.",
            "ACCEPTED speed_candidate lanes are the only lanes that should be promoted.",
            "Negative-control lanes should not be treated as speed wins even if their compare status is unusual.",
        ],
    }


def _fmt_delta(delta: dict[str, Any] | None) -> str:
    if not delta:
        return "missing"
    unit = delta.get("unit", "")
    return f"{delta.get('delta', 0.0):+.3f} {unit} ({delta.get('direction')})"


def _render(result: dict[str, Any]) -> str:
    lines = [
        "# Nemotron Ultra Runtime Candidate Index",
        "",
        f"log_dir: `{result['log_dir']}`",
        f"queue_json: `{result['queue_json']}`",
        f"status: `{result['status']}`",
        "",
        "## Status Counts",
    ]
    for key in sorted(result["status_counts"]):
        lines.append(f"- {key}: `{result['status_counts'][key]}`")
    if result.get("failures"):
        lines.extend(["", "## Failures"])
        lines.extend(f"- {item}" for item in result["failures"])
    lines.extend(
        [
            "",
            "## Lanes",
            "| lane | kind | status | compare | gate | best_tps delta | moe_ms delta | mamba_ms delta | reason |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for lane in result["lanes"]:
        lines.append(
            "| "
            f"`{lane['id']}` | `{lane['kind']}` | `{lane['status']}` | "
            f"`{lane.get('compare_status')}` | `{lane.get('gate_status')}` | "
            f"`{_fmt_delta(lane.get('best_tps_delta'))}` | "
            f"`{_fmt_delta(lane.get('moe_ms_delta'))}` | "
            f"`{_fmt_delta(lane.get('mamba_ms_delta'))}` | "
            f"{lane.get('reason') or 'none'} |"
        )
    lines.extend(["", "## Candidate Directories"])
    for lane in result["lanes"]:
        lines.append(f"- `{lane['id']}`: `{lane.get('candidate_log_dir')}`")
    if result.get("interpretation"):
        lines.extend(["", "## Interpretation"])
        lines.extend(f"- {item}" for item in result["interpretation"])
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--queue-json", type=Path, default=DEFAULT_QUEUE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    args = ap.parse_args()

    result = _build_result(args.log_dir, args.queue_json)
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
