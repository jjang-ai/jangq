"""No-load acceptance audit for Nemotron Ultra token/s speed fixes.

This script answers a narrower question than the speed gate: can the current
saved evidence be called a fixed runtime-speed state? It requires an accepted
speed-candidate lane plus target token/s, bucket, and coherence gates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_LOG_DIR = Path("docs/runtime/logs")
DEFAULT_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-fix-acceptance.md")
DEFAULT_JSON_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-fix-acceptance.json")


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _accepted_speed_lanes(index: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        lane
        for lane in index.get("lanes", [])
        if lane.get("kind") == "speed_candidate" and lane.get("status") == "ACCEPTED"
    ]


def _build_result(
    log_dir: Path,
    *,
    target_tps: float,
    max_moe_ms: float,
    max_mamba_ms: float,
    require_speed_gate_fixed: bool,
) -> dict[str, Any]:
    speed_gate_path = log_dir / "2026-06-04-nemotron-ultra-runtime-speed-gate.json"
    candidate_index_path = log_dir / "2026-06-04-nemotron-ultra-runtime-candidate-index.json"
    budget_path = log_dir / "2026-06-04-nemotron-ultra-token-speed-budget.json"
    manifest_path = log_dir / "2026-06-04-nemotron-ultra-runtime-proof-manifest.json"
    ledger_path = log_dir / "2026-06-04-nemotron-ultra-runtime-issue-ledger.json"

    speed_gate = _load(speed_gate_path)
    candidate_index = _load(candidate_index_path)
    budget = _load(budget_path)
    manifest = _load(manifest_path) or {}
    ledger = _load(ledger_path) or {}

    blockers: list[str] = []
    partials: list[str] = []
    fixed: list[str] = []

    if speed_gate is None:
        blockers.append(f"missing speed gate JSON: {speed_gate_path}")
        metrics: dict[str, Any] = {}
    else:
        metrics = speed_gate.get("metrics", {})
        if require_speed_gate_fixed and speed_gate.get("status") != "FIXED":
            partials.append(f"speed gate is {speed_gate.get('status')}, not FIXED")
        else:
            fixed.append(f"speed gate status is {speed_gate.get('status')}")

    if candidate_index is None:
        blockers.append(f"missing candidate index JSON: {candidate_index_path}")
        accepted_lanes: list[dict[str, Any]] = []
    else:
        accepted_lanes = _accepted_speed_lanes(candidate_index)
        if not accepted_lanes:
            partials.append("no speed_candidate lane has ACCEPTED evidence")
        else:
            fixed.append(
                "accepted speed_candidate lanes: "
                + ", ".join(str(lane.get("id")) for lane in accepted_lanes)
            )

    if budget is None:
        blockers.append(f"missing token speed budget JSON: {budget_path}")
    else:
        current = budget.get("current", {})
        best_live_tps = current.get("best_live_tps")
        if not isinstance(best_live_tps, (int, float)):
            partials.append("best live token/s is missing")
        elif float(best_live_tps) < target_tps:
            partials.append(f"best live token/s {float(best_live_tps):.3f} is below target {target_tps:.3f}")
        else:
            fixed.append(f"best live token/s {float(best_live_tps):.3f} meets target {target_tps:.3f}")

    moe_ms = metrics.get("moe_ms")
    if isinstance(moe_ms, (int, float)):
        if float(moe_ms) > max_moe_ms:
            partials.append(f"MoE bucket {float(moe_ms):.3f} ms exceeds acceptance ceiling {max_moe_ms:.3f}")
        else:
            fixed.append(f"MoE bucket {float(moe_ms):.3f} ms clears ceiling {max_moe_ms:.3f}")
    elif speed_gate is not None:
        partials.append("MoE bucket is missing from speed gate metrics")

    mamba_ms = metrics.get("mamba_ms")
    if isinstance(mamba_ms, (int, float)):
        if float(mamba_ms) > max_mamba_ms:
            partials.append(f"Mamba bucket {float(mamba_ms):.3f} ms exceeds acceptance ceiling {max_mamba_ms:.3f}")
        else:
            fixed.append(f"Mamba bucket {float(mamba_ms):.3f} ms clears ceiling {max_mamba_ms:.3f}")
    elif speed_gate is not None:
        partials.append("Mamba bucket is missing from speed gate metrics")

    for item in speed_gate.get("partial", []) if speed_gate else []:
        if "coherence gate remains partial" in str(item):
            partials.append(str(item))
    for item in speed_gate.get("failures", []) if speed_gate else []:
        blockers.append(str(item))

    status = "BLOCKED" if blockers else ("PARTIAL" if partials else "FIXED")
    return {
        "status": status,
        "log_dir": str(log_dir),
        "target_tps": target_tps,
        "max_moe_ms": max_moe_ms,
        "max_mamba_ms": max_mamba_ms,
        "require_speed_gate_fixed": require_speed_gate_fixed,
        "fixed": fixed,
        "partial": partials,
        "blockers": blockers,
        "accepted_speed_lanes": [
            {
                "id": lane.get("id"),
                "candidate_log_dir": lane.get("candidate_log_dir"),
                "compare_status": lane.get("compare_status"),
                "gate_status": lane.get("gate_status"),
                "best_tps_delta": lane.get("best_tps_delta"),
                "moe_ms_delta": lane.get("moe_ms_delta"),
                "mamba_ms_delta": lane.get("mamba_ms_delta"),
            }
            for lane in accepted_lanes
        ],
        "current": {
            "manifest_status": manifest.get("status"),
            "ledger_status": ledger.get("status"),
            "speed_gate_status": speed_gate.get("status") if speed_gate else None,
            "candidate_index_status": candidate_index.get("status") if candidate_index else None,
            "best_live_tps": metrics.get("best_live_tps"),
            "manual_decode_total_ms": metrics.get("manual_decode_total_ms"),
            "moe_ms": metrics.get("moe_ms"),
            "mamba_ms": metrics.get("mamba_ms"),
            "attention_ms": metrics.get("attention_ms"),
            "norm_lm_head_ms": metrics.get("norm_lm_head_ms"),
        },
        "source_files": [
            str(speed_gate_path),
            str(candidate_index_path),
            str(budget_path),
            str(manifest_path),
            str(ledger_path),
        ],
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _render(result: dict[str, Any]) -> str:
    lines = [
        "# Nemotron Ultra Runtime Speed Fix Acceptance",
        "",
        f"log_dir: `{result['log_dir']}`",
        f"status: `{result['status']}`",
        f"target_tps: `{result['target_tps']:.3f}`",
        f"max_moe_ms: `{result['max_moe_ms']:.3f}`",
        f"max_mamba_ms: `{result['max_mamba_ms']:.3f}`",
        f"require_speed_gate_fixed: `{result['require_speed_gate_fixed']}`",
        "",
        "## Current",
    ]
    for key, value in result["current"].items():
        lines.append(f"- {key}: `{_fmt(value)}`")
    lines.extend(["", "## Fixed"])
    lines.extend(f"- {item}" for item in result["fixed"] or ["none"])
    lines.extend(["", "## Partial"])
    lines.extend(f"- {item}" for item in result["partial"] or ["none"])
    lines.extend(["", "## Blockers"])
    lines.extend(f"- {item}" for item in result["blockers"] or ["none"])
    lines.extend(["", "## Accepted Speed Lanes"])
    if result["accepted_speed_lanes"]:
        for lane in result["accepted_speed_lanes"]:
            lines.append(
                f"- `{lane.get('id')}` compare=`{lane.get('compare_status')}` "
                f"gate=`{lane.get('gate_status')}` dir=`{lane.get('candidate_log_dir')}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Source Files"])
    lines.extend(f"- `{item}`" for item in result["source_files"])
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--target-tps", type=float, default=10.0)
    ap.add_argument("--max-moe-ms", type=float, default=40.0)
    ap.add_argument("--max-mamba-ms", type=float, default=40.0)
    ap.add_argument("--allow-partial-speed-gate", action="store_true")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    result = _build_result(
        args.log_dir,
        target_tps=args.target_tps,
        max_moe_ms=args.max_moe_ms,
        max_mamba_ms=args.max_mamba_ms,
        require_speed_gate_fixed=not args.allow_partial_speed_gate,
    )
    report = _render(result)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    sys.stdout.write(report)
    if args.strict and result["status"] != "FIXED":
        raise SystemExit(2 if result["status"] == "BLOCKED" else 1)


if __name__ == "__main__":
    main()
