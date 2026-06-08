"""Check a completed Nemotron Ultra runtime experiment lane.

This is a no-load verifier. It reads the generated experiment queue plus a
candidate log directory and reports whether the lane met its own expected
compare status, required proof outputs, coherence non-regression, and
cache/modality handoff invariants.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_QUEUE = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json")
DEFAULT_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-experiment-result-check.md")
DEFAULT_JSON_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-experiment-result-check.json")


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _find_lane(queue: dict[str, Any], lane_id: str) -> dict[str, Any] | None:
    for lane in queue.get("lanes", []):
        if lane.get("id") == lane_id:
            return lane
    return None


def _direction(compare: dict[str, Any], key: str) -> str | None:
    delta = compare.get("metrics", {}).get("deltas", {}).get(key, {})
    direction = delta.get("direction")
    return str(direction) if direction is not None else None


def _coherence_non_regression(compare: dict[str, Any]) -> list[str]:
    failures = []
    for key, counts in compare.get("coherence_counts", {}).items():
        if int(counts.get("delta", 0)) > 0:
            failures.append(f"coherence {key} regressed by {counts['delta']}")
    return failures


def _handoff_failures(handoff: dict[str, Any] | None) -> list[str]:
    if handoff is None:
        return ["missing candidate handoff JSON"]
    failures = []
    gates = handoff.get("cache_and_modality_gates", {})
    artifact = handoff.get("artifact", {})
    if gates.get("text_only") is not True:
        failures.append("handoff text_only gate is not true")
    if gates.get("cache_type") != "hybrid":
        failures.append("handoff cache_type is not hybrid")
    if artifact.get("drops_mtp") is not True:
        failures.append("handoff artifact does not confirm drops_mtp=true")
    return failures


def _cache_parser_contract_failures(contract: dict[str, Any] | None) -> list[str]:
    if contract is None:
        return ["missing candidate cache/parser contract JSON"]
    failures: list[str] = []
    if contract.get("status") == "BLOCKED":
        failures.append("candidate cache/parser contract is BLOCKED")
    failures.extend(str(item) for item in contract.get("failures", []))

    cache = contract.get("cache_contract", {})
    parser = contract.get("parser_contract", {})
    modality = contract.get("modality_contract", {})
    if cache.get("cache_type") != "hybrid":
        failures.append("cache/parser contract cache_type is not hybrid")
    if cache.get("cache_entries") != 60:
        failures.append("cache/parser contract cache_entries is not 60")
    if cache.get("mamba_companion_state_entries") != 48:
        failures.append("cache/parser contract Mamba companion state entries are not 48")
    if cache.get("attention_kv_cache_entries") != 12:
        failures.append("cache/parser contract attention KV entries are not 12")
    if parser.get("reasoning_parser") != "deepseek_r1":
        failures.append("cache/parser contract reasoning_parser is not deepseek_r1")
    if parser.get("tool_parser") != "nemotron":
        failures.append("cache/parser contract tool_parser is not nemotron")
    if modality.get("text_only") is not True:
        failures.append("cache/parser contract text_only gate is not true")
    if modality.get("drops_mtp") is not True:
        failures.append("cache/parser contract drops_mtp gate is not true")
    return failures


def _build_result(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    queue_path = Path(args.queue_json)
    candidate_dir = Path(args.candidate_log_dir)
    queue = _load(queue_path)
    if not queue:
        return 2, {
            "status": "BLOCKED",
            "failures": [f"missing queue JSON: {queue_path}"],
            "fixed": [],
            "lane": None,
            "candidate_log_dir": str(candidate_dir),
        }
    lane = _find_lane(queue, args.lane_id)
    if lane is None:
        return 2, {
            "status": "BLOCKED",
            "failures": [f"lane not found in queue: {args.lane_id}"],
            "fixed": [],
            "lane": None,
            "candidate_log_dir": str(candidate_dir),
        }

    failures: list[str] = []
    fixed: list[str] = []
    missing_outputs = [name for name in lane.get("required_outputs", []) if not (candidate_dir / name).exists()]
    if missing_outputs:
        failures.extend(f"missing required output: {name}" for name in missing_outputs)
    else:
        fixed.append("all required lane outputs are present")

    compare = _load(candidate_dir / "2026-06-04-nemotron-ultra-runtime-speed-compare.json")
    gate = _load(candidate_dir / "2026-06-04-nemotron-ultra-runtime-speed-gate.json")
    handoff = _load(candidate_dir / "2026-06-04-nemotron-ultra-agent-handoff.json")
    cache_parser_contract = _load(candidate_dir / "2026-06-04-nemotron-ultra-runtime-cache-parser-contract.json")

    if compare is None:
        failures.append("missing candidate speed compare JSON")
    else:
        status = compare.get("status")
        expected = lane.get("expected_compare_statuses", [])
        if status not in expected:
            failures.append(f"compare status {status!r} not in expected {expected!r}")
        else:
            fixed.append(f"compare status {status} matches lane expectation")
        failures.extend(str(item) for item in compare.get("failures", []))
        failures.extend(_coherence_non_regression(compare))

        if lane.get("kind") == "speed_candidate":
            if lane.get("id") == "moe-routed-shared-scheduling" and _direction(compare, "moe_ms") != "better":
                failures.append("MoE lane did not improve moe_ms")
            if lane.get("id") == "mamba-projection-dispatch" and _direction(compare, "mamba_ms") != "better":
                failures.append("Mamba lane did not improve mamba_ms")
        elif lane.get("kind") == "negative_control":
            if status == "IMPROVED":
                failures.append("negative-control lane improved; preserve evidence before changing defaults")

    if gate is None:
        failures.append("missing candidate speed gate JSON")
    elif gate.get("status") == "BLOCKED":
        failures.append("candidate speed gate is BLOCKED")
    else:
        fixed.append(f"candidate speed gate status is {gate.get('status')}")

    handoff_issues = _handoff_failures(handoff)
    if handoff_issues:
        failures.extend(handoff_issues)
    else:
        fixed.append("candidate handoff preserves text-only hybrid-cache MTP-disabled invariants")

    cache_parser_issues = _cache_parser_contract_failures(cache_parser_contract)
    if cache_parser_issues:
        failures.extend(cache_parser_issues)
    else:
        fixed.append("candidate cache/parser contract preserves hybrid-cache parser MTP text-only invariants")

    status = "BLOCKED" if missing_outputs else ("REJECTED" if failures else "ACCEPTED")
    exit_code = 2 if status == "BLOCKED" else (1 if status == "REJECTED" and args.strict else 0)
    result = {
        "status": status,
        "lane_id": args.lane_id,
        "lane_kind": lane.get("kind"),
        "candidate_log_dir": str(candidate_dir),
        "expected_compare_statuses": lane.get("expected_compare_statuses", []),
        "compare_status": compare.get("status") if compare else None,
        "gate_status": gate.get("status") if gate else None,
        "cache_parser_contract_status": cache_parser_contract.get("status") if cache_parser_contract else None,
        "fixed": fixed,
        "failures": failures,
        "missing_outputs": missing_outputs,
    }
    return exit_code, result


def _render(result: dict[str, Any]) -> str:
    lines = [
        "# Nemotron Ultra Experiment Result Check",
        "",
        f"lane_id: `{result.get('lane_id')}`",
        f"lane_kind: `{result.get('lane_kind')}`",
        f"candidate_log_dir: `{result.get('candidate_log_dir')}`",
        f"status: `{result['status']}`",
        f"compare_status: `{result.get('compare_status')}`",
        f"gate_status: `{result.get('gate_status')}`",
        "",
        "## Fixed",
    ]
    lines.extend(f"- {item}" for item in result.get("fixed", []))
    lines.extend(["", "## Failures"])
    lines.extend(f"- {item}" for item in result.get("failures", []))
    if result.get("missing_outputs"):
        lines.extend(["", "## Missing Outputs"])
        lines.extend(f"- {item}" for item in result["missing_outputs"])
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue-json", type=Path, default=DEFAULT_QUEUE)
    ap.add_argument("--lane-id", required=True)
    ap.add_argument("--candidate-log-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    exit_code, result = _build_result(args)
    report = _render(result)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    sys.stdout.write(report)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
