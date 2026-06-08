"""Check whether Nemotron Ultra host cleanup is ready for a candidate run.

This no-load helper combines host readiness, host cleanup, and launch guard
JSON into one readiness verdict. It never stops processes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_LOG_DIR = Path("docs/runtime/logs")
DEFAULT_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-cleanup-ready-check.md")
DEFAULT_JSON_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-cleanup-ready-check.json")


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _model_server_blockers(cleanup: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    for proc in cleanup.get("processes", []):
        if proc.get("likely_model_server") and float(proc.get("rss_gib", 0.0)) > 20.0:
            blockers.append(
                f"model server pid {proc.get('pid')} uses {float(proc.get('rss_gib', 0.0)):.2f} GiB RSS"
            )
    return blockers


def _build_result(log_dir: Path) -> dict[str, Any]:
    host_path = log_dir / "2026-06-04-nemotron-ultra-host-runtime-readiness.json"
    cleanup_path = log_dir / "2026-06-04-nemotron-ultra-host-cleanup-runbook.json"
    guard_path = log_dir / "2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.json"
    host = _load(host_path)
    cleanup = _load(cleanup_path)
    guard = _load(guard_path)

    failures: list[str] = []
    blockers: list[str] = []
    fixed: list[str] = []

    for label, path, data in (
        ("host readiness", host_path, host),
        ("host cleanup runbook", cleanup_path, cleanup),
        ("candidate launch guard", guard_path, guard),
    ):
        if data is None:
            failures.append(f"missing {label}: {path}")
        else:
            fixed.append(f"found {label}: {path}")

    if host:
        if host.get("status") == "READY":
            fixed.append("host readiness is READY")
        else:
            blockers.append(f"host readiness is {host.get('status')}")
            blockers.extend(str(item) for item in host.get("warnings", []))
    if cleanup:
        if cleanup.get("status") == "READY":
            fixed.append("host cleanup runbook is READY")
        else:
            blockers.append(f"host cleanup runbook is {cleanup.get('status')}")
        blockers.extend(_model_server_blockers(cleanup))
    if guard:
        if guard.get("status") == "READY":
            fixed.append("candidate launch guard is READY")
        else:
            blockers.append(f"candidate launch guard is {guard.get('status')}")
            blockers.extend(str(item) for item in guard.get("warnings", []))

    status = "BLOCKED" if failures else ("WATCH" if blockers else "READY")
    return {
        "status": status,
        "log_dir": str(log_dir),
        "fixed": fixed,
        "blockers": sorted(set(blockers)),
        "failures": failures,
        "manual_actions": [
            "Stop or close the owning app for unrelated high-RSS model servers before loading the 98G Nemotron bundle.",
            "Confirm the PID still matches the expected process before stopping anything.",
            "After cleanup, rerun the refresh command and require this check plus the launch guard to be READY.",
        ],
        "verify_commands": {
            "refresh": (
                "PYTHONPATH=jang-tools jang-tools/.venv/bin/python "
                "jang-tools/examples/nemotron_ultra/refresh_runtime_proof_bundle.py "
                f"--log-dir {log_dir} "
                f"--summary-out {log_dir / '2026-06-04-nemotron-ultra-runtime-proof-refresh.md'}"
            ),
            "strict_guard": (
                "PYTHONPATH=jang-tools jang-tools/.venv/bin/python "
                "jang-tools/examples/nemotron_ultra/runtime_candidate_launch_guard.py "
                f"--log-dir {log_dir} --strict"
            ),
            "strict_lane_matrix": (
                "PYTHONPATH=jang-tools jang-tools/.venv/bin/python "
                "jang-tools/examples/nemotron_ultra/runtime_lane_readiness_matrix.py "
                f"--log-dir {log_dir} --strict"
            ),
        },
        "source_files": [str(host_path), str(cleanup_path), str(guard_path)],
    }


def _render(result: dict[str, Any]) -> str:
    lines = [
        "# Nemotron Ultra Runtime Cleanup Ready Check",
        "",
        f"log_dir: `{result['log_dir']}`",
        f"status: `{result['status']}`",
        "",
        "## Fixed",
    ]
    lines.extend(f"- {item}" for item in result["fixed"] or ["none"])
    lines.extend(["", "## Blockers"])
    lines.extend(f"- {item}" for item in result["blockers"] or ["none"])
    lines.extend(["", "## Failures"])
    lines.extend(f"- {item}" for item in result["failures"] or ["none"])
    lines.extend(["", "## Manual Actions"])
    lines.extend(f"- {item}" for item in result["manual_actions"])
    lines.extend(["", "## Verify Commands"])
    for name, command in result["verify_commands"].items():
        lines.append(f"- {name}: `{command}`")
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
        raise SystemExit(2 if result["status"] == "BLOCKED" else 1)


if __name__ == "__main__":
    main()
