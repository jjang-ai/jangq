"""Preflight a Nemotron Ultra runtime candidate lane before loading the model.

This no-load checker reads the proof manifest, host readiness, shape contract,
patch spec, and experiment queue. It reports whether an expensive candidate
suite is ready to run, should be watched, or is blocked by stale/missing proof.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_LOG_DIR = Path("docs/runtime/logs")
DEFAULT_LANE_ID = "moe-routed-shared-scheduling"
DEFAULT_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-preflight.md")
DEFAULT_JSON_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-candidate-preflight.json")


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _find_lane(data: dict[str, Any], lane_id: str) -> dict[str, Any] | None:
    for lane in data.get("lanes", []):
        if lane.get("id") == lane_id:
            return lane
    return None


def _candidate_log_dir(queue_lane: dict[str, Any] | None) -> str | None:
    if not queue_lane:
        return None
    if queue_lane.get("candidate_log_dir"):
        return str(queue_lane["candidate_log_dir"])
    command = str(queue_lane.get("command") or queue_lane.get("run_command") or "")
    parts = command.split()
    for index, part in enumerate(parts):
        if part == "--candidate-log-dir" and index + 1 < len(parts):
            return parts[index + 1]
    return None


def _build_result(log_dir: Path, lane_id: str) -> dict[str, Any]:
    manifest_path = log_dir / "2026-06-04-nemotron-ultra-runtime-proof-manifest.json"
    host_path = log_dir / "2026-06-04-nemotron-ultra-host-runtime-readiness.json"
    shape_path = log_dir / "2026-06-04-nemotron-ultra-runtime-shape-contract.json"
    patch_path = log_dir / "2026-06-04-nemotron-ultra-runtime-patch-spec.json"
    queue_path = log_dir / "2026-06-04-nemotron-ultra-runtime-experiment-queue.json"

    manifest = _load(manifest_path)
    host = _load(host_path)
    shape = _load(shape_path)
    patch = _load(patch_path)
    queue = _load(queue_path)

    failures: list[str] = []
    warnings: list[str] = []
    fixed: list[str] = []

    for label, path, data in (
        ("manifest", manifest_path, manifest),
        ("host readiness", host_path, host),
        ("shape contract", shape_path, shape),
        ("patch spec", patch_path, patch),
        ("experiment queue", queue_path, queue),
    ):
        if data is None:
            failures.append(f"missing {label}: {path}")
        else:
            fixed.append(f"found {label}: {path}")

    manifest_lane = _find_lane(manifest or {}, lane_id)
    patch_lane = _find_lane(patch or {}, lane_id)
    queue_lane = _find_lane(queue or {}, lane_id)
    lane_kind = queue_lane.get("kind") if queue_lane else None
    if manifest_lane is None:
        failures.append(f"lane not present in manifest: {lane_id}")
    if lane_kind == "speed_candidate" and patch_lane is None:
        failures.append(f"lane not present in patch spec: {lane_id}")
    if queue_lane is None:
        failures.append(f"lane not present in experiment queue: {lane_id}")
    if manifest_lane and queue_lane and (patch_lane or lane_kind != "speed_candidate"):
        fixed.append(f"lane {lane_id} is present in required lane registries")

    if manifest:
        if manifest.get("missing_required_artifacts"):
            failures.extend(f"missing required artifact: {item}" for item in manifest["missing_required_artifacts"])
        if manifest.get("stale_required_artifacts"):
            failures.extend(f"stale required artifact: {item}" for item in manifest["stale_required_artifacts"])
        if manifest.get("status") == "BLOCKED":
            failures.append("manifest status is BLOCKED")
        elif manifest.get("status"):
            fixed.append(f"manifest status is {manifest['status']}")

    if shape:
        if shape.get("status") != "READY":
            failures.append(f"shape contract status is {shape.get('status')}")
        else:
            fixed.append("shape contract is READY")
        arch = shape.get("architecture", {})
        quant = shape.get("quantization", {})
        layer_counts = arch.get("layer_counts", {})
        if layer_counts != {"attention": 12, "mamba": 48, "moe": 48, "total": 108}:
            failures.append(f"unexpected layer counts: {layer_counts}")
        if quant.get("drops_mtp") is not True:
            failures.append("shape contract does not confirm drops_mtp=true")
        mxtq_bits = quant.get("mxtq_bits", {})
        if mxtq_bits.get("routed_expert", {}).get("up_proj") != 1:
            failures.append("routed up_proj bit contract is not 1")
        if mxtq_bits.get("routed_expert", {}).get("down_proj") != 1:
            failures.append("routed down_proj bit contract is not 1")
        if mxtq_bits.get("mamba_projection") != 8:
            failures.append("mamba projection bit contract is not 8")

    if host:
        host_status = host.get("status")
        if host_status == "READY":
            fixed.append("host readiness is READY")
        elif host_status == "WATCH":
            warnings.extend(str(item) for item in host.get("warnings", []))
            warnings.append("host readiness is WATCH; expensive probes may be noisy")
        else:
            failures.append(f"host readiness status is {host_status}")

    command = queue_lane.get("command") if queue_lane else None
    dry_run_command = queue_lane.get("dry_run_command") if queue_lane else None
    post_check = queue_lane.get("post_check_command") if queue_lane else None
    candidate_log_dir = _candidate_log_dir(queue_lane)
    status = "BLOCKED" if failures else ("WATCH" if warnings else "READY")
    return {
        "status": status,
        "lane_id": lane_id,
        "log_dir": str(log_dir),
        "fixed": fixed,
        "warnings": warnings,
        "failures": failures,
        "candidate_log_dir": candidate_log_dir,
        "candidate_command": command,
        "dry_run_command": dry_run_command,
        "post_check_command": post_check,
        "required_outputs": queue_lane.get("required_outputs", []) if queue_lane else [],
        "expected_compare_statuses": queue_lane.get("expected_compare_statuses", []) if queue_lane else [],
    }


def _render(result: dict[str, Any]) -> str:
    lines = [
        "# Nemotron Ultra Runtime Candidate Preflight",
        "",
        f"lane_id: `{result['lane_id']}`",
        f"log_dir: `{result['log_dir']}`",
        f"status: `{result['status']}`",
        "",
        "## Fixed",
    ]
    lines.extend(f"- {item}" for item in result["fixed"])
    lines.extend(["", "## Warnings"])
    lines.extend(f"- {item}" for item in result["warnings"] or ["none"])
    lines.extend(["", "## Failures"])
    lines.extend(f"- {item}" for item in result["failures"] or ["none"])
    lines.extend(["", "## Commands"])
    lines.append(f"- candidate: `{result.get('candidate_command')}`")
    lines.append(f"- dry_run: `{result.get('dry_run_command')}`")
    lines.append(f"- post_check: `{result.get('post_check_command')}`")
    lines.append("")
    lines.append("## Required Outputs")
    lines.extend(f"- `{item}`" for item in result.get("required_outputs", []))
    lines.append("")
    lines.append("## Expected Compare Statuses")
    lines.extend(f"- `{item}`" for item in result.get("expected_compare_statuses", []))
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--lane-id", default=DEFAULT_LANE_ID)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    result = _build_result(args.log_dir, args.lane_id)
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
