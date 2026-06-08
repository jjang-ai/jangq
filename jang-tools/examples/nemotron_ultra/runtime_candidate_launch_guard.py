"""Guard the next Nemotron Ultra runtime candidate command.

This no-load helper reads the next-run runbook and candidate index. It writes
the exact candidate and post-check commands, but it does not execute them. By
default it blocks WATCH/BLOCKED lanes so expensive probes are not run while host
or proof state is noisy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_LOG_DIR = Path("docs/runtime/logs")
DEFAULT_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.md")
DEFAULT_JSON_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.json")


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _candidate_index_row(index: dict[str, Any], lane_id: str | None) -> dict[str, Any] | None:
    if lane_id is None:
        return None
    for row in index.get("lanes", []):
        if row.get("id") == lane_id:
            return row
    return None


def _build_result(log_dir: Path, *, allow_watch: bool) -> dict[str, Any]:
    runbook_path = log_dir / "2026-06-04-nemotron-ultra-runtime-next-runbook.json"
    index_path = log_dir / "2026-06-04-nemotron-ultra-runtime-candidate-index.json"
    runbook = _load(runbook_path)
    index = _load(index_path) or {}
    failures: list[str] = []
    warnings: list[str] = []

    if runbook is None:
        return {
            "status": "BLOCKED",
            "allow_watch": allow_watch,
            "log_dir": str(log_dir),
            "failures": [f"missing next runbook JSON: {runbook_path}"],
            "warnings": [],
            "lane": {},
            "commands": {},
        }

    lane = runbook.get("next_lane", {})
    lane_id = lane.get("id")
    lane_status = lane.get("status")
    runbook_status = runbook.get("status")
    host_status = runbook.get("host_status")
    candidate_row = _candidate_index_row(index, lane_id)

    if not lane_id:
        failures.append("next runbook has no selected lane")
    if not lane.get("candidate_command"):
        failures.append("next runbook has no candidate command")
    if not lane.get("post_check_command"):
        failures.append("next runbook has no post-check command")
    if runbook_status == "BLOCKED" or lane_status == "BLOCKED":
        failures.append(f"selected lane is blocked: runbook={runbook_status}, lane={lane_status}")
    if lane_status == "WATCH" or host_status == "WATCH" or runbook_status == "WATCH":
        warnings.extend(str(item) for item in lane.get("warnings", []))
        warnings.append("selected candidate is WATCH; rerun host cleanup/readiness before expensive timing")
    if candidate_row and candidate_row.get("status") not in (None, "MISSING"):
        warnings.append(
            f"candidate index already has status {candidate_row.get('status')} for lane {lane_id}; use a fresh candidate dir if rerunning"
        )

    if failures:
        status = "BLOCKED"
    elif warnings and not allow_watch:
        status = "BLOCKED_BY_WATCH"
    else:
        status = "READY"

    return {
        "status": status,
        "allow_watch": allow_watch,
        "log_dir": str(log_dir),
        "runbook_status": runbook_status,
        "host_status": host_status,
        "lane": {
            "id": lane_id,
            "kind": lane.get("kind"),
            "status": lane_status,
            "title": lane.get("title"),
            "candidate_index_status": candidate_row.get("status") if candidate_row else None,
        },
        "warnings": warnings,
        "failures": failures,
        "commands": {
            "candidate": lane.get("candidate_command"),
            "post_check": lane.get("post_check_command"),
            "refresh_after_cleanup": (
                "PYTHONPATH=jang-tools jang-tools/.venv/bin/python "
                "jang-tools/examples/nemotron_ultra/refresh_runtime_proof_bundle.py "
                f"--log-dir {log_dir} "
                f"--summary-out {log_dir / '2026-06-04-nemotron-ultra-runtime-proof-refresh.md'}"
            ),
        },
        "source_files": [str(runbook_path), str(index_path)],
    }


def _render(result: dict[str, Any]) -> str:
    lane = result.get("lane", {})
    lines = [
        "# Nemotron Ultra Runtime Candidate Launch Guard",
        "",
        f"log_dir: `{result['log_dir']}`",
        f"status: `{result['status']}`",
        f"allow_watch: `{result['allow_watch']}`",
        f"runbook_status: `{result.get('runbook_status')}`",
        f"host_status: `{result.get('host_status')}`",
        "",
        "## Selected Lane",
        f"- id: `{lane.get('id')}`",
        f"- kind: `{lane.get('kind')}`",
        f"- status: `{lane.get('status')}`",
        f"- candidate_index_status: `{lane.get('candidate_index_status')}`",
        f"- title: {lane.get('title')}",
        "",
        "## Warnings",
    ]
    lines.extend(f"- {item}" for item in result.get("warnings", []) or ["none"])
    lines.extend(["", "## Failures"])
    lines.extend(f"- {item}" for item in result.get("failures", []) or ["none"])
    lines.extend(["", "## Commands"])
    for name, command in result.get("commands", {}).items():
        lines.append(f"- {name}: `{command}`")
    lines.extend(["", "## Source Files"])
    lines.extend(f"- `{item}`" for item in result.get("source_files", []))
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--allow-watch", action="store_true")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    result = _build_result(args.log_dir, allow_watch=args.allow_watch)
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
