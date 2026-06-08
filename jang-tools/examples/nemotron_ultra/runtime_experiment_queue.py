"""Generate a no-load Nemotron Ultra runtime experiment queue.

The queue translates saved speed evidence into exact candidate-suite commands
and proof requirements. It does not run the model.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

from examples.nemotron_ultra.token_speed_budget import _build_result as _budget_result


DEFAULT_LOG_DIR = Path("docs/runtime/logs")
DEFAULT_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.md")
DEFAULT_JSON_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-experiment-queue.json")
DEFAULT_BUNDLE = Path("/Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L")


def _shell_join(items: list[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in items)


def _candidate_command(
    *,
    lane: str,
    lane_id: str,
    bundle: Path,
    baseline_log_dir: Path,
    candidate_root: Path,
    wired_limit_gb: int,
    live_max_tokens: int,
    long_max_tokens: int,
) -> list[str]:
    return [
        "PYTHONPATH=jang-tools",
        "jang-tools/.venv/bin/python",
        "jang-tools/examples/nemotron_ultra/run_runtime_candidate_suite.py",
        "--candidate-log-dir",
        str(candidate_root / lane),
        "--baseline-log-dir",
        str(baseline_log_dir),
        "--queue-json",
        str(candidate_root / "2026-06-04-nemotron-ultra-runtime-experiment-queue.json"),
        "--lane-id",
        lane_id,
        "--bundle",
        str(bundle),
        "--wired-limit-gb",
        str(wired_limit_gb),
        "--live-max-tokens",
        str(live_max_tokens),
        "--long-max-tokens",
        str(long_max_tokens),
    ]


def _lane(
    *,
    lane_id: str,
    kind: str,
    title: str,
    goal: str,
    evidence: str,
    patch_surface: str,
    env: dict[str, str],
    target_budget: dict[str, Any],
    command: list[str],
    post_check_command: list[str],
    expected_compare_statuses: list[str],
    acceptance: list[str],
) -> dict[str, Any]:
    env_prefix = [f"{key}={value}" for key, value in env.items()]
    candidate_log_dir = None
    for index, part in enumerate(command):
        if part == "--candidate-log-dir" and index + 1 < len(command):
            candidate_log_dir = command[index + 1]
            break
    run_command = _shell_join(env_prefix + command)
    return {
        "id": lane_id,
        "status": "OPEN",
        "kind": kind,
        "title": title,
        "goal": goal,
        "evidence": evidence,
        "patch_surface": patch_surface,
        "candidate_log_dir": candidate_log_dir,
        "env": env,
        "command": run_command,
        "run_command": run_command,
        "dry_run_command": run_command + " --dry-run",
        "post_check_command": _shell_join(post_check_command),
        "expected_compare_statuses": expected_compare_statuses,
        "target_budget": target_budget,
        "required_outputs": [
            "2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json",
            "2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json",
            "2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json",
            "2026-06-04-nemotron-ultra-mamba-component-probe.json",
            "2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json",
            "2026-06-04-nemotron-ultra-projection-tradeoff-probe.json",
            "2026-06-04-nemotron-ultra-runtime-speed-compare.json",
            "2026-06-04-nemotron-ultra-runtime-speed-gate.json",
            "2026-06-04-nemotron-ultra-token-speed-budget.json",
            "2026-06-04-nemotron-ultra-agent-handoff.json",
            "2026-06-04-nemotron-ultra-runtime-cache-parser-contract.json",
        ],
        "acceptance": acceptance,
    }


def _build_result(args: argparse.Namespace) -> dict[str, Any]:
    baseline_log_dir = Path(args.baseline_log_dir)
    bundle = Path(args.bundle)
    candidate_root = Path(args.candidate_root)
    budget = _budget_result(baseline_log_dir, args.speed_targets)
    target = budget["targets"][0]
    current = budget["current"]
    base_command_kwargs = {
        "bundle": bundle,
        "baseline_log_dir": baseline_log_dir,
        "candidate_root": candidate_root,
        "wired_limit_gb": args.wired_limit_gb,
        "live_max_tokens": args.live_max_tokens,
        "long_max_tokens": args.long_max_tokens,
    }

    def post_check(lane_id: str, lane_dir: str) -> list[str]:
        return [
            "PYTHONPATH=jang-tools",
            "jang-tools/.venv/bin/python",
            "jang-tools/examples/nemotron_ultra/experiment_result_check.py",
            "--queue-json",
            str(candidate_root / "2026-06-04-nemotron-ultra-runtime-experiment-queue.json"),
            "--lane-id",
            lane_id,
            "--candidate-log-dir",
            str(candidate_root / lane_dir),
            "--out",
            str(candidate_root / lane_dir / "2026-06-04-nemotron-ultra-experiment-result-check.md"),
            "--json-out",
            str(candidate_root / lane_dir / "2026-06-04-nemotron-ultra-experiment-result-check.json"),
            "--strict",
        ]

    lanes = [
        _lane(
            lane_id="moe-routed-shared-scheduling",
            kind="speed_candidate",
            title="MoE routed/shared scheduling",
            goal="Reduce MoE bucket without changing routing semantics or routed expert bit layout.",
            evidence=(
                f"MoE is {current['moe_ms']:.3f} ms; first target needs about "
                f"{target['moe_cut_ms_proportional']:.3f} ms MoE reduction for {target['target_tps']:.1f} tok/s."
            ),
            patch_surface="JANG loader/TurboQuant MoE scheduling only; no bundle expansion.",
            env={},
            target_budget=target,
            command=_candidate_command(
                lane="candidate-moe-scheduling",
                lane_id="moe-routed-shared-scheduling",
                **base_command_kwargs,
            ),
            post_check_command=post_check("moe-routed-shared-scheduling", "candidate-moe-scheduling"),
            expected_compare_statuses=["IMPROVED"],
            acceptance=[
                "candidate compare status is IMPROVED with no failures",
                "MoE bucket improves and Mamba/attention/lm_head do not materially regress",
                "long coherence leak/repeat/no_eos counts do not regress",
                "candidate handoff remains text-only, MTP-disabled, and hybrid-cache aware",
            ],
        ),
        _lane(
            lane_id="mamba-projection-dispatch",
            kind="speed_candidate",
            title="Mamba projection/dispatch fusion",
            goal="Reduce Mamba bucket by attacking projection and dispatch overhead before conv rewrites.",
            evidence=(
                f"Mamba is {current['mamba_ms']:.3f} ms; first target needs about "
                f"{target['mamba_cut_ms_proportional']:.3f} ms Mamba reduction for {target['target_tps']:.1f} tok/s."
            ),
            patch_surface="JANG loader/runtime Mamba path only; keep 8-bit affine projections unless new proof reverses it.",
            env={},
            target_budget=target,
            command=_candidate_command(
                lane="candidate-mamba-dispatch",
                lane_id="mamba-projection-dispatch",
                **base_command_kwargs,
            ),
            post_check_command=post_check("mamba-projection-dispatch", "candidate-mamba-dispatch"),
            expected_compare_statuses=["IMPROVED"],
            acceptance=[
                "candidate compare status is IMPROVED with no failures",
                "Mamba bucket improves and MoE/attention/lm_head do not materially regress",
                "long coherence leak/repeat/no_eos counts do not regress",
                "candidate handoff remains text-only, MTP-disabled, and hybrid-cache aware",
            ],
        ),
        _lane(
            lane_id="weighted-moe-ablation",
            kind="negative_control",
            title="Weighted MoE fast-path A/B",
            goal="Confirm the current weighted-MoE default remains beneficial after any MoE refactor.",
            evidence="Weighted MoE is a small positive/noisy improvement and must not silently regress.",
            patch_surface="A/B proof lane only; no code change implied.",
            env={"JANGTQ_DISABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH": "1"},
            target_budget=target,
            command=_candidate_command(
                lane="candidate-disable-weighted-moe",
                lane_id="weighted-moe-ablation",
                **base_command_kwargs,
            ),
            post_check_command=post_check("weighted-moe-ablation", "candidate-disable-weighted-moe"),
            expected_compare_statuses=["FAIL", "UNCHANGED"],
            acceptance=[
                "compare must not be treated as a speed fix",
                "if the lane is faster, preserve evidence before changing the default",
                "long coherence leak/repeat/no_eos counts do not regress",
                "candidate handoff remains text-only, MTP-disabled, and hybrid-cache aware",
            ],
        ),
        _lane(
            lane_id="activation-bf16-ablation",
            kind="negative_control",
            title="BF16 activation retention guard",
            goal="Guard the large lm_head/activation dtype speed fix from accidental rollback.",
            evidence="BF16 retention moved synchronized decode from about 320 ms/token to about 144 ms/token.",
            patch_surface="A/B proof lane only; should be slower and marked as a negative-control regression.",
            env={"JANGTQ_DISABLE_NEMOTRON_ACTIVATION_BF16": "1"},
            target_budget=target,
            command=_candidate_command(
                lane="candidate-disable-activation-bf16",
                lane_id="activation-bf16-ablation",
                **base_command_kwargs,
            ),
            post_check_command=post_check("activation-bf16-ablation", "candidate-disable-activation-bf16"),
            expected_compare_statuses=["FAIL"],
            acceptance=[
                "compare should fail or clearly regress speed versus baseline",
                "lm_head/norm and manual decode regressions confirm BF16 retention is still required",
                "do not promote this lane as a candidate fix",
                "candidate handoff remains text-only, MTP-disabled, and hybrid-cache aware",
            ],
        ),
    ]
    return {
        "baseline_log_dir": str(baseline_log_dir),
        "bundle": str(bundle),
        "candidate_root": str(candidate_root),
        "current": current,
        "speed_targets": args.speed_targets,
        "lanes": lanes,
        "notes": [
            "Run one candidate lane at a time; the model is about 98G and probes are expensive.",
            "Lane-specific env vars are already embedded in the generated command.",
            "Do not call a speed lane fixed until compare, gate, and long-coherence rows agree.",
            "Negative-control lanes are guards; expected regressions should not be promoted as fixes.",
        ],
    }


def _render(result: dict[str, Any]) -> str:
    current = result["current"]
    lines = [
        "# Nemotron Ultra Runtime Experiment Queue",
        "",
        f"baseline_log_dir: `{result['baseline_log_dir']}`",
        f"bundle: `{result['bundle']}`",
        f"candidate_root: `{result['candidate_root']}`",
        "",
        "## Current Baseline",
        f"- best_live_tps: `{current['best_live_tps']:.3f}`",
        f"- manual_decode_total_ms: `{current['manual_decode_total_ms']:.3f}`",
        f"- moe_ms: `{current['moe_ms']:.3f}`",
        f"- mamba_ms: `{current['mamba_ms']:.3f}`",
        f"- moe_plus_mamba_pct_of_total: `{current['moe_plus_mamba_pct_of_total']:.2f}%`",
        "",
        "## Lanes",
    ]
    for lane in result["lanes"]:
        lines.extend(
            [
                f"### {lane['title']}",
                f"- id: `{lane['id']}`",
                f"- kind: `{lane['kind']}`",
                f"- goal: {lane['goal']}",
                f"- evidence: {lane['evidence']}",
                f"- patch_surface: {lane['patch_surface']}",
            ]
        )
        if lane["env"]:
            env = " ".join(f"{key}={value}" for key, value in lane["env"].items())
            lines.append(f"- env: `{env}`")
        else:
            lines.append("- env: none")
        lines.append("- expected_compare_statuses: " + ", ".join(f"`{item}`" for item in lane["expected_compare_statuses"]))
        lines.append(f"- command: `{lane['command']}`")
        lines.append(f"- post_check_command: `{lane['post_check_command']}`")
        lines.append("- required_outputs: " + ", ".join(f"`{item}`" for item in lane["required_outputs"]))
        lines.append("- acceptance: " + "; ".join(lane["acceptance"]))
        lines.append("")
    lines.append("## Notes")
    lines.extend(f"- {item}" for item in result["notes"])
    return "\n".join(lines) + "\n"


def _parse_targets(raw: str) -> list[float]:
    targets = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not targets:
        raise argparse.ArgumentTypeError("at least one speed target is required")
    if any(item <= 0 for item in targets):
        raise argparse.ArgumentTypeError("speed targets must be positive")
    return targets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--candidate-root", type=Path, default=Path("docs/runtime/logs"))
    ap.add_argument("--speed-targets", type=_parse_targets, default=_parse_targets("10,12,15"))
    ap.add_argument("--wired-limit-gb", type=int, default=105)
    ap.add_argument("--live-max-tokens", type=int, default=32)
    ap.add_argument("--long-max-tokens", type=int, default=96)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    result = _build_result(args)
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
