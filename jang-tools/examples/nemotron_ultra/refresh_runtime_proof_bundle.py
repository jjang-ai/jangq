"""Regenerate the no-load Nemotron Ultra runtime proof bundle.

This script does not load the model. It reruns the saved-log report, speed
experiment plan, runtime speed gate, readiness runbooks, issue ledger, and
manifest with consistent output paths.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_LOG_DIR = Path("docs/runtime/logs")
DEFAULT_BUNDLE = Path("/Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L")


def _run(label: str, cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    output = proc.stdout
    if proc.stderr:
        output += ("\n" if output else "") + proc.stderr
    return proc.returncode, f"## {label}\n\n````text\n{output.rstrip()}\n````\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--strict-gate", action="store_true")
    ap.add_argument(
        "--summary-out",
        type=Path,
        default=DEFAULT_LOG_DIR / "2026-06-04-nemotron-ultra-runtime-proof-refresh.md",
    )
    args = ap.parse_args()

    py = sys.executable
    log_dir = str(args.log_dir)
    commands = [
        (
            "Runtime log bundle validation",
            [
                py,
                "jang-tools/examples/nemotron_ultra/validate_runtime_log_bundle.py",
                "--log-dir",
                log_dir,
                "--out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-log-bundle-validation.md"),
            ],
        ),
        (
            "Runtime status report",
            [
                py,
                "jang-tools/examples/nemotron_ultra/runtime_status_report.py",
                "--log-dir",
                log_dir,
                "--out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-status-report.md"),
            ],
        ),
        (
            "Host runtime readiness",
            [
                py,
                "jang-tools/examples/nemotron_ultra/host_runtime_readiness.py",
                "--bundle",
                str(args.bundle),
                "--log-dir",
                log_dir,
                "--out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-host-runtime-readiness.md"),
                "--json-out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-host-runtime-readiness.json"),
            ],
        ),
        (
            "Host cleanup runbook",
            [
                py,
                "jang-tools/examples/nemotron_ultra/host_cleanup_runbook.py",
                "--log-dir",
                log_dir,
                "--out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-host-cleanup-runbook.md"),
                "--json-out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-host-cleanup-runbook.json"),
            ],
        ),
        (
            "Speed experiment plan",
            [
                py,
                "jang-tools/examples/nemotron_ultra/speed_experiment_plan.py",
                "--log-dir",
                log_dir,
                "--out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-speed-experiment-plan.md"),
            ],
        ),
        (
            "Token speed budget",
            [
                py,
                "jang-tools/examples/nemotron_ultra/token_speed_budget.py",
                "--log-dir",
                log_dir,
                "--out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-token-speed-budget.md"),
                "--json-out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-token-speed-budget.json"),
            ],
        ),
        (
            "Component budget matrix",
            [
                py,
                "jang-tools/examples/nemotron_ultra/component_budget_matrix.py",
                "--log-dir",
                log_dir,
                "--out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-component-budget-matrix.md"),
                "--json-out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-component-budget-matrix.json"),
            ],
        ),
        (
            "Runtime experiment queue",
            [
                py,
                "jang-tools/examples/nemotron_ultra/runtime_experiment_queue.py",
                "--baseline-log-dir",
                log_dir,
                "--bundle",
                str(args.bundle),
                "--candidate-root",
                log_dir,
                "--out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-experiment-queue.md"),
                "--json-out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-experiment-queue.json"),
            ],
        ),
        (
            "Runtime shape contract",
            [
                py,
                "jang-tools/examples/nemotron_ultra/runtime_shape_contract.py",
                "--bundle",
                str(args.bundle),
                "--log-dir",
                log_dir,
                "--out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-shape-contract.md"),
                "--json-out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-shape-contract.json"),
            ],
        ),
        (
            "Runtime patch spec",
            [
                py,
                "jang-tools/examples/nemotron_ultra/runtime_patch_spec.py",
                "--log-dir",
                log_dir,
                "--out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-patch-spec.md"),
                "--json-out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-patch-spec.json"),
            ],
        ),
        (
            "Runtime speed gate",
            [
                py,
                "jang-tools/examples/nemotron_ultra/runtime_speed_gate.py",
                "--log-dir",
                log_dir,
                "--out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-speed-gate.md"),
                "--json-out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-speed-gate.json"),
                *(["--strict"] if args.strict_gate else []),
            ],
        ),
        (
            "Runtime speed compare",
            [
                py,
                "jang-tools/examples/nemotron_ultra/compare_runtime_speed_logs.py",
                "--baseline-log-dir",
                log_dir,
                "--candidate-log-dir",
                log_dir,
                "--out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-speed-compare.md"),
                "--json-out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-speed-compare.json"),
            ],
        ),
        (
            "Agent runtime handoff",
            [
                py,
                "jang-tools/examples/nemotron_ultra/agent_handoff_report.py",
                "--bundle",
                str(args.bundle),
                "--log-dir",
                log_dir,
                "--out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-agent-handoff.md"),
                "--json-out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-agent-handoff.json"),
            ],
        ),
        (
            "Runtime cache parser contract",
            [
                py,
                "jang-tools/examples/nemotron_ultra/runtime_cache_parser_contract.py",
                "--log-dir",
                log_dir,
                "--out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-cache-parser-contract.md"),
                "--json-out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-cache-parser-contract.json"),
            ],
        ),
        (
            "Runtime proof manifest preflight input",
            [
                py,
                "jang-tools/examples/nemotron_ultra/runtime_proof_manifest.py",
                "--log-dir",
                log_dir,
                "--out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-proof-manifest.md"),
                "--json-out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-proof-manifest.json"),
            ],
        ),
        (
            "Runtime candidate preflight",
            [
                py,
                "jang-tools/examples/nemotron_ultra/runtime_candidate_preflight.py",
                "--log-dir",
                log_dir,
                "--lane-id",
                "moe-routed-shared-scheduling",
                "--out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-candidate-preflight.md"),
                "--json-out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-candidate-preflight.json"),
            ],
        ),
        (
            "Runtime lane readiness matrix",
            [
                py,
                "jang-tools/examples/nemotron_ultra/runtime_lane_readiness_matrix.py",
                "--log-dir",
                log_dir,
                "--out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-lane-readiness-matrix.md"),
                "--json-out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-lane-readiness-matrix.json"),
            ],
        ),
        (
            "Runtime next runbook",
            [
                py,
                "jang-tools/examples/nemotron_ultra/runtime_next_runbook.py",
                "--log-dir",
                log_dir,
                "--out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-next-runbook.md"),
                "--json-out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-next-runbook.json"),
            ],
        ),
        (
            "Runtime issue ledger",
            [
                py,
                "jang-tools/examples/nemotron_ultra/runtime_issue_ledger.py",
                "--log-dir",
                log_dir,
                "--out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-issue-ledger.md"),
                "--json-out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-issue-ledger.json"),
            ],
        ),
        (
            "Runtime candidate index",
            [
                py,
                "jang-tools/examples/nemotron_ultra/runtime_candidate_index.py",
                "--log-dir",
                log_dir,
                "--queue-json",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-experiment-queue.json"),
                "--out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-candidate-index.md"),
                "--json-out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-candidate-index.json"),
            ],
        ),
        (
            "Runtime speed fix acceptance",
            [
                py,
                "jang-tools/examples/nemotron_ultra/runtime_speed_fix_acceptance.py",
                "--log-dir",
                log_dir,
                "--out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-speed-fix-acceptance.md"),
                "--json-out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-speed-fix-acceptance.json"),
            ],
        ),
        (
            "Runtime candidate launch guard",
            [
                py,
                "jang-tools/examples/nemotron_ultra/runtime_candidate_launch_guard.py",
                "--log-dir",
                log_dir,
                "--out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.md"),
                "--json-out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-candidate-launch-guard.json"),
            ],
        ),
        (
            "Runtime cleanup ready check",
            [
                py,
                "jang-tools/examples/nemotron_ultra/runtime_cleanup_ready_check.py",
                "--log-dir",
                log_dir,
                "--out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-cleanup-ready-check.md"),
                "--json-out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-cleanup-ready-check.json"),
            ],
        ),
        (
            "Runtime MoE candidate contract",
            [
                py,
                "jang-tools/examples/nemotron_ultra/runtime_moe_candidate_contract.py",
                "--log-dir",
                log_dir,
                "--out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-moe-candidate-contract.md"),
                "--json-out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-moe-candidate-contract.json"),
            ],
        ),
        (
            "Runtime MoE execution ticket",
            [
                py,
                "jang-tools/examples/nemotron_ultra/runtime_moe_execution_ticket.py",
                "--log-dir",
                log_dir,
                "--out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-moe-execution-ticket.md"),
                "--json-out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-moe-execution-ticket.json"),
            ],
        ),
        (
            "Runtime MoE surface map",
            [
                py,
                "jang-tools/examples/nemotron_ultra/runtime_moe_surface_map.py",
                "--log-dir",
                log_dir,
                "--source-root",
                "jang-tools",
                "--out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-moe-surface-map.md"),
                "--json-out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-moe-surface-map.json"),
            ],
        ),
        (
            "Runtime MoE patch plan",
            [
                py,
                "jang-tools/examples/nemotron_ultra/runtime_moe_patch_plan.py",
                "--log-dir",
                log_dir,
                "--out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-moe-patch-plan.md"),
                "--json-out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-moe-patch-plan.json"),
            ],
        ),
        (
            "Runtime MoE delta contract",
            [
                py,
                "jang-tools/examples/nemotron_ultra/runtime_moe_delta_contract.py",
                "--log-dir",
                log_dir,
                "--out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-moe-delta-contract.md"),
                "--json-out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-moe-delta-contract.json"),
            ],
        ),
        (
            "Runtime Mamba candidate contract",
            [
                py,
                "jang-tools/examples/nemotron_ultra/runtime_mamba_candidate_contract.py",
                "--log-dir",
                log_dir,
                "--out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-mamba-candidate-contract.md"),
                "--json-out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-mamba-candidate-contract.json"),
            ],
        ),
        (
            "Runtime proof manifest final",
            [
                py,
                "jang-tools/examples/nemotron_ultra/runtime_proof_manifest.py",
                "--log-dir",
                log_dir,
                "--out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-proof-manifest.md"),
                "--json-out",
                str(args.log_dir / "2026-06-04-nemotron-ultra-runtime-proof-manifest.json"),
            ],
        ),
    ]

    sections = ["# Nemotron Ultra Runtime Proof Refresh", ""]
    exit_code = 0
    for label, cmd in commands:
        code, section = _run(label, cmd)
        sections.append(section)
        if code != 0:
            exit_code = code

    text = "\n".join(sections)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(text)
    sys.stdout.write(text)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
