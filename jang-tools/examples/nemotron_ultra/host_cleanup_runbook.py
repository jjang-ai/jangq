"""Generate a safe host cleanup runbook before Nemotron Ultra speed probes.

This script does not stop processes. It lists high-RSS processes, identifies
likely model servers, and prints follow-up commands to rerun readiness checks.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_LOG_DIR = Path("docs/runtime/logs")
DEFAULT_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-host-cleanup-runbook.md")
DEFAULT_JSON_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-host-cleanup-runbook.json")


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    output = proc.stdout
    if proc.stderr:
        output += ("\n" if output else "") + proc.stderr
    return proc.returncode, output.strip()


def _parse_ps(output: str, *, min_rss_gib: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        rss_gib = int(parts[1]) / (1024**2)
        if rss_gib < min_rss_gib:
            continue
        command = parts[2]
        rows.append(
            {
                "pid": int(parts[0]),
                "rss_gib": rss_gib,
                "command": command,
                "likely_model_server": "vmlx_engine.cli serve" in command or "run_runtime_candidate_suite.py" in command,
                "likely_vm": "Parallels VM.app" in command or "prl_vm_app" in command,
            }
        )
    rows.sort(key=lambda item: item["rss_gib"], reverse=True)
    return rows


def _build_result(log_dir: Path, *, min_rss_gib: float) -> dict[str, Any]:
    code, output = _run(["ps", "-axo", "pid,rss,command"])
    processes = _parse_ps(output if code == 0 else "", min_rss_gib=min_rss_gib)
    status = "READY" if not any(item["likely_model_server"] and item["rss_gib"] > 20 for item in processes) else "WATCH"
    return {
        "status": status,
        "log_dir": str(log_dir),
        "min_rss_gib": min_rss_gib,
        "processes": processes,
        "recommended_actions": [
            "Use the owning app UI or service control to stop model servers before loading the 98G Nemotron bundle.",
            "Do not kill unknown processes blindly; confirm the PID still matches the command immediately before stopping it.",
            "After cleanup, rerun host_runtime_readiness.py, runtime_lane_readiness_matrix.py, and runtime_next_runbook.py.",
        ],
        "follow_up_commands": {
            "host_readiness": (
                "PYTHONPATH=jang-tools jang-tools/.venv/bin/python "
                "jang-tools/examples/nemotron_ultra/host_runtime_readiness.py "
                f"--log-dir {log_dir}"
            ),
            "lane_matrix": (
                "PYTHONPATH=jang-tools jang-tools/.venv/bin/python "
                "jang-tools/examples/nemotron_ultra/runtime_lane_readiness_matrix.py "
                f"--log-dir {log_dir}"
            ),
            "next_runbook": (
                "PYTHONPATH=jang-tools jang-tools/.venv/bin/python "
                "jang-tools/examples/nemotron_ultra/runtime_next_runbook.py "
                f"--log-dir {log_dir}"
            ),
        },
    }


def _render(result: dict[str, Any]) -> str:
    lines = [
        "# Nemotron Ultra Host Cleanup Runbook",
        "",
        f"status: `{result['status']}`",
        f"log_dir: `{result['log_dir']}`",
        f"min_rss_gib: `{result['min_rss_gib']:.1f}`",
        "",
        "## High RSS Processes",
    ]
    if result["processes"]:
        for process in result["processes"]:
            tags = []
            if process["likely_model_server"]:
                tags.append("model_server")
            if process["likely_vm"]:
                tags.append("vm")
            tag_text = ", ".join(tags) if tags else "process"
            lines.append(
                f"- pid `{process['pid']}` rss `{process['rss_gib']:.2f} GiB` tags `{tag_text}`: {process['command']}"
            )
    else:
        lines.append("- none above threshold")
    lines.extend(["", "## Recommended Actions"])
    lines.extend(f"- {item}" for item in result["recommended_actions"])
    lines.extend(["", "## Follow-Up Commands"])
    for name, command in result["follow_up_commands"].items():
        lines.append(f"- {name}: `{command}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--min-rss-gib", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    args = ap.parse_args()

    result = _build_result(args.log_dir, min_rss_gib=args.min_rss_gib)
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
