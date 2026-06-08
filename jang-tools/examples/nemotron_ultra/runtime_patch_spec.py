"""Generate a concrete runtime patch spec from saved Nemotron Ultra speed proof.

This is a no-load planning artifact for the next implementation pass. It turns
the component budget matrix and experiment queue into implementation lanes,
acceptance gates, negative controls, and proof commands.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from examples.nemotron_ultra.component_budget_matrix import _build_result as _matrix_result


DEFAULT_LOG_DIR = Path("docs/runtime/logs")
DEFAULT_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-patch-spec.md")
DEFAULT_JSON_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-patch-spec.json")


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _queue(log_dir: Path) -> dict[str, Any]:
    return _load(log_dir / "2026-06-04-nemotron-ultra-runtime-experiment-queue.json") or {}


def _gate(log_dir: Path) -> dict[str, Any]:
    return _load(log_dir / "2026-06-04-nemotron-ultra-runtime-speed-gate.json") or {}


def _lane(queue: dict[str, Any], lane_id: str) -> dict[str, Any]:
    for lane in queue.get("lanes", []):
        if lane.get("id") == lane_id:
            return lane
    return {}


def _top_component(matrix: dict[str, Any], family: str, label: str) -> dict[str, Any]:
    for row in matrix.get("component_rows", []):
        if row.get("family") == family and row.get("label") == label:
            return row
    return {}


def _scenario_tps(row: dict[str, Any], cut_pct: float) -> float | None:
    for scenario in row.get("scenarios", []):
        if float(scenario.get("cut_pct", -1.0)) == cut_pct:
            return scenario.get("new_manual_tps")
    return None


def _fmt_tps(value: float | None) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "unknown"


def _build_result(log_dir: Path) -> dict[str, Any]:
    matrix = _matrix_result(log_dir)
    queue = _queue(log_dir)
    gate = _gate(log_dir)
    moe_lane = _lane(queue, "moe-routed-shared-scheduling")
    mamba_lane = _lane(queue, "mamba-projection-dispatch")
    switch = _top_component(matrix, "MoE", "switch_mlp")
    full_moe = _top_component(matrix, "MoE", "full_moe")
    full_mamba = _top_component(matrix, "Mamba", "full_mamba_mixer")
    mamba_in = _top_component(matrix, "Mamba", "in_proj")
    current = matrix["current"]
    targets = matrix.get("target_hits", [])

    lanes = [
        {
            "id": "moe-routed-shared-scheduling",
            "priority": 1,
            "title": "MoE routed/shared scheduling and switch_mlp path",
            "implementation_surface": moe_lane.get(
                "patch_surface", "JANG loader/TurboQuant MoE scheduling only; no bundle expansion."
            ),
            "why": [
                f"MoE bucket is {current['moe_ms']:.3f} ms across 48 layers.",
                f"`switch_mlp` projects to {switch.get('projected_total_ms', 0.0):.3f} ms; 50% cut implies {_fmt_tps(_scenario_tps(switch, 50.0))} tok/s synchronized.",
                f"`full_moe` is an inclusive path row at {full_moe.get('projected_total_ms', 0.0):.3f} ms, so path-level scheduling is the highest-leverage MoE target.",
            ],
            "do": [
                "preserve router top-k and weighted expert semantics",
                "preserve routed expert 1-bit layout and shared expert 8-bit layout",
                "reduce per-layer dispatch/synchronization around routed/shared expert execution",
                "measure `switch_mlp`, `shared_experts`, and layer E bucket after every candidate",
            ],
            "do_not": [
                "do not lower router top-k as the primary speed fix",
                "do not expand quantized experts to full precision",
                "do not promote a change unless long-coherence counts do not regress",
            ],
            "proof": {
                "candidate_command": moe_lane.get("command"),
                "post_check_command": moe_lane.get("post_check_command"),
                "expected_compare_statuses": moe_lane.get("expected_compare_statuses", []),
                "required_outputs": moe_lane.get("required_outputs", []),
            },
        },
        {
            "id": "mamba-projection-dispatch",
            "priority": 2,
            "title": "Mamba projection/dispatch path",
            "implementation_surface": mamba_lane.get(
                "patch_surface",
                "JANG loader/runtime Mamba path only; keep 8-bit affine projections unless new proof reverses it.",
            ),
            "why": [
                f"Mamba bucket is {current['mamba_ms']:.3f} ms across 48 layers.",
                f"`full_mamba_mixer` projects to {full_mamba.get('projected_total_ms', 0.0):.3f} ms; 50% cut implies {_fmt_tps(_scenario_tps(full_mamba, 50.0))} tok/s synchronized.",
                f"`in_proj` projects to {mamba_in.get('projected_total_ms', 0.0):.3f} ms; it is larger than conv/SSM update in the saved component probe.",
            ],
            "do": [
                "attack projection/dispatch overhead before grouped conv rewrites",
                "preserve 8-bit affine projection path unless a new projection tradeoff probe reverses the result",
                "preserve Mamba companion cache/state order for hybrid prefix cache compatibility",
                "measure M bucket plus `in_proj`, `out_proj`, `conv`, and `ssm_update` after every candidate",
            ],
            "do_not": [
                "do not dequantize Mamba projections to BF16 as a default speed fix",
                "do not treat attention KV cache work as a substitute for Mamba state proof",
                "do not change cache topology without rerunning cache/block handoff checks",
            ],
            "proof": {
                "candidate_command": mamba_lane.get("command"),
                "post_check_command": mamba_lane.get("post_check_command"),
                "expected_compare_statuses": mamba_lane.get("expected_compare_statuses", []),
                "required_outputs": mamba_lane.get("required_outputs", []),
            },
        },
    ]
    return {
        "log_dir": str(log_dir),
        "status": gate.get("status", "UNKNOWN"),
        "current": current,
        "target_hits": targets,
        "lanes": lanes,
        "global_non_goals": [
            "do not chase attention first while attention is 8.990 ms and below the gate ceiling",
            "do not hide parser/coherence failures with prompts, forced tags, or sampler tweaks",
            "do not enable MTP/speculative decode for this MTP-dropped bundle",
            "do not call a speed lane fixed without compare, gate, and long-coherence proof",
        ],
        "runtime_controls": {
            "disable_weighted_moe_fastpath": "JANGTQ_DISABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH=1",
            "disable_activation_bf16": "JANGTQ_DISABLE_NEMOTRON_ACTIVATION_BF16=1",
            "disable_switchmlp_fastpath": "JANGTQ_DISABLE_NEMOTRON_SWITCHMLP_FASTPATH=1",
        },
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _render(result: dict[str, Any]) -> str:
    current = result["current"]
    lines = [
        "# Nemotron Ultra Runtime Patch Spec",
        "",
        f"log_dir: `{result['log_dir']}`",
        f"current_status: `{result['status']}`",
        "",
        "## Current Speed State",
        f"- manual_decode_total_ms: `{_fmt(current['manual_decode_total_ms'])}`",
        f"- manual_implied_tps: `{_fmt(current['manual_implied_tps'])}`",
        f"- moe_ms: `{_fmt(current['moe_ms'])}`",
        f"- mamba_ms: `{_fmt(current['mamba_ms'])}`",
        f"- attention_ms: `{_fmt(current['attention_ms'])}`",
        f"- norm_lm_head_ms: `{_fmt(current['norm_lm_head_ms'])}`",
        f"- moe_mamba_pct: `{_fmt(current['moe_mamba_pct'])}`",
        "",
        "## Target Cuts",
    ]
    for target in result["target_hits"]:
        components = target.get("single_component_can_cover", [])
        suffix = ", ".join(f"`{item}`" for item in components) if components else "none"
        lines.append(
            f"- `{target['target_tps']:.3f}` tok/s needs `{target['required_total_cut_ms']:.3f}` ms cut; single measured row/path enough: {suffix}"
        )
    lines.extend(["", "## Implementation Lanes"])
    for lane in result["lanes"]:
        lines.extend(
            [
                f"### {lane['priority']}. {lane['title']}",
                f"- id: `{lane['id']}`",
                f"- implementation_surface: {lane['implementation_surface']}",
                "- why:",
            ]
        )
        lines.extend(f"  - {item}" for item in lane["why"])
        lines.append("- do:")
        lines.extend(f"  - {item}" for item in lane["do"])
        lines.append("- do_not:")
        lines.extend(f"  - {item}" for item in lane["do_not"])
        lines.append("- proof:")
        proof = lane["proof"]
        lines.append(f"  - candidate_command: `{proof.get('candidate_command')}`")
        lines.append(f"  - post_check_command: `{proof.get('post_check_command')}`")
        lines.append(
            "  - expected_compare_statuses: "
            + ", ".join(f"`{item}`" for item in proof.get("expected_compare_statuses", []))
        )
        lines.append("  - required_outputs: " + ", ".join(f"`{item}`" for item in proof.get("required_outputs", [])))
        lines.append("")
    lines.append("## Global Non-Goals")
    lines.extend(f"- {item}" for item in result["global_non_goals"])
    lines.extend(["", "## Runtime Controls"])
    lines.extend(f"- {name}: `{value}`" for name, value in result["runtime_controls"].items())
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
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


if __name__ == "__main__":
    main()
