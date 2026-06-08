"""Generate a no-load execution ticket for the next MoE speed lane.

This helper does not load the model or run candidate probes. It consolidates
the launch guard, cleanup check, candidate index, and MoE contract into a
single handoff artifact for the next expensive runtime run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_LOG_DIR = Path("docs/runtime/logs")
DEFAULT_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-execution-ticket.md")
DEFAULT_JSON_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-execution-ticket.json")
LANE_ID = "moe-routed-shared-scheduling"


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _candidate_row(index: dict[str, Any]) -> dict[str, Any]:
    for row in index.get("lanes", []):
        if row.get("id") == LANE_ID:
            return row
    return {}


def _build_result(log_dir: Path) -> dict[str, Any]:
    guard_path = log_dir / "2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.json"
    cleanup_path = log_dir / "2026-06-04-nemotron-ultra-runtime-cleanup-ready-check.json"
    contract_path = log_dir / "2026-06-04-nemotron-ultra-runtime-moe-candidate-contract.json"
    index_path = log_dir / "2026-06-04-nemotron-ultra-runtime-candidate-index.json"
    manifest_path = log_dir / "2026-06-04-nemotron-ultra-runtime-proof-manifest.json"

    guard = _load(guard_path) or {}
    cleanup = _load(cleanup_path) or {}
    contract = _load(contract_path) or {}
    index = _load(index_path) or {}
    manifest = _load(manifest_path) or {}
    row = _candidate_row(index)

    failures: list[str] = []
    warnings: list[str] = []

    if guard.get("status") != "READY":
        failures.append(f"candidate launch guard is {guard.get('status', 'MISSING')}")
    if cleanup.get("status") != "READY":
        failures.append(f"cleanup ready check is {cleanup.get('status', 'MISSING')}")
    if contract.get("status") != "READY":
        failures.append(f"MoE candidate contract is {contract.get('status', 'MISSING')}")
    if guard.get("lane", {}).get("id") != LANE_ID:
        failures.append(f"launch guard selected lane is {guard.get('lane', {}).get('id')}")
    if contract.get("lane_id") != LANE_ID:
        failures.append(f"MoE contract lane is {contract.get('lane_id')}")
    if row.get("status") not in (None, "MISSING"):
        warnings.append(
            f"candidate index already has {row.get('status')} evidence for {LANE_ID}; use a fresh candidate directory"
        )
    if manifest.get("status") not in (None, "PARTIAL"):
        warnings.append(f"runtime manifest status is {manifest.get('status')}; expected PARTIAL before speed candidate")

    commands = dict(guard.get("commands", {}))
    commands["post_candidate_index"] = (
        "PYTHONPATH=jang-tools jang-tools/.venv/bin/python "
        "jang-tools/examples/nemotron_ultra/runtime_candidate_index.py "
        f"--log-dir {log_dir} "
        f"--queue-json {log_dir / '2026-06-04-nemotron-ultra-runtime-experiment-queue.json'} "
        f"--out {log_dir / '2026-06-04-nemotron-ultra-runtime-candidate-index.md'} "
        f"--json-out {log_dir / '2026-06-04-nemotron-ultra-runtime-candidate-index.json'}"
    )
    commands["post_candidate_refresh"] = (
        "PYTHONPATH=jang-tools jang-tools/.venv/bin/python "
        "jang-tools/examples/nemotron_ultra/refresh_runtime_proof_bundle.py "
        f"--log-dir {log_dir} "
        f"--summary-out {log_dir / '2026-06-04-nemotron-ultra-runtime-proof-refresh.md'}"
    )

    execution_order = [
        "confirm this ticket status is READY",
        "run candidate command exactly once for the selected MoE lane",
        "run post_check command to write ACCEPTED/REJECTED/BLOCKED verdict",
        "rerun candidate index to surface lane status",
        "rerun proof refresh to update manifest, ledger, and next runbook",
    ]

    return {
        "status": "BLOCKED" if failures else "READY",
        "lane_id": LANE_ID,
        "log_dir": str(log_dir),
        "candidate_status": row.get("status", "MISSING"),
        "guard_status": guard.get("status"),
        "cleanup_status": cleanup.get("status"),
        "contract_status": contract.get("status"),
        "manifest_status": manifest.get("status"),
        "failures": failures,
        "warnings": warnings,
        "target": contract.get("target", {}),
        "invariants": contract.get("invariants", {}),
        "acceptance_checks": contract.get("acceptance_checks", []),
        "required_outputs": contract.get("required_outputs", []),
        "execution_order": execution_order,
        "commands": commands,
        "do_not": [
            "do not run the Mamba lane until this MoE lane has accepted evidence",
            "do not treat an improved short live row as accepted without long coherence and experiment_result_check",
            "do not change MTP, VL/audio, parser, or hybrid cache assumptions for this speed lane",
        ],
        "source_files": [str(guard_path), str(cleanup_path), str(contract_path), str(index_path), str(manifest_path)],
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _render(result: dict[str, Any]) -> str:
    lines = [
        "# Nemotron Ultra MoE Execution Ticket",
        "",
        f"log_dir: `{result['log_dir']}`",
        f"lane_id: `{result['lane_id']}`",
        f"status: `{result['status']}`",
        f"candidate_status: `{result['candidate_status']}`",
        f"guard_status: `{result.get('guard_status')}`",
        f"cleanup_status: `{result.get('cleanup_status')}`",
        f"contract_status: `{result.get('contract_status')}`",
        f"manifest_status: `{result.get('manifest_status')}`",
        "",
        "## Failures",
    ]
    lines.extend(f"- {item}" for item in result["failures"] or ["none"])
    lines.extend(["", "## Warnings"])
    lines.extend(f"- {item}" for item in result["warnings"] or ["none"])
    lines.extend(["", "## Target"])
    for key, value in result["target"].items():
        lines.append(f"- {key}: `{_fmt(value)}`")
    lines.extend(["", "## Invariants"])
    for key, value in result["invariants"].items():
        lines.append(f"- {key}: `{_fmt(value)}`")
    lines.extend(["", "## Execution Order"])
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(result["execution_order"], start=1))
    lines.extend(["", "## Commands"])
    for name, command in result["commands"].items():
        lines.append(f"- {name}: `{command}`")
    lines.extend(["", "## Acceptance Checks"])
    lines.extend(f"- {item}" for item in result["acceptance_checks"])
    lines.extend(["", "## Required Outputs"])
    lines.extend(f"- `{item}`" for item in result["required_outputs"])
    lines.extend(["", "## Do Not"])
    lines.extend(f"- {item}" for item in result["do_not"])
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
