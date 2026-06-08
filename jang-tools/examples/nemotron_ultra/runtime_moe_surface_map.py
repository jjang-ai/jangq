"""Map the JANG MoE runtime surfaces for the Nemotron Ultra speed lane.

This is a no-load documentation helper. It scans the local JANG source tree for
the concrete loader, routed-kernel, env-toggle, and proof symbols that matter
for the next MoE speed candidate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_LOG_DIR = Path("docs/runtime/logs")
DEFAULT_SOURCE_ROOT = Path("jang-tools")
DEFAULT_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-surface-map.md")
DEFAULT_JSON_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-moe-surface-map.json")


SURFACES = [
    {
        "id": "loader-hydration",
        "file": "jang_tools/load_jangtq.py",
        "role": "hydrates switch_mlp tensors into TurboQuantSwitchLinear modules",
        "anchors": ["def _hydrate_jangtq_model", "TurboQuantSwitchLinear", "switch_mlp"],
    },
    {
        "id": "nemotron-weighted-moe-patch",
        "file": "jang_tools/load_jangtq.py",
        "role": "patches NemotronHMoE decode to call weighted SwitchMLP and shared experts",
        "anchors": [
            "JANGTQ_DISABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH",
            "Nemotron-H MoE weighted SwitchMLP decode",
            "_switchmlp_weighted_decode",
        ],
    },
    {
        "id": "switchmlp-fastpath-toggle",
        "file": "jang_tools/load_jangtq.py",
        "role": "controls the SwitchMLP fast path and its negative-control env toggle",
        "anchors": [
            "JANGTQ_DISABLE_NEMOTRON_SWITCHMLP_FASTPATH",
            "JANGTQ_ENABLE_NEMOTRON_SWITCHMLP_FASTPATH",
        ],
    },
    {
        "id": "activation-bf16-toggle",
        "file": "jang_tools/load_jangtq.py",
        "role": "preserves or disables BF16 activation retention for negative-control proof",
        "anchors": ["JANGTQ_DISABLE_NEMOTRON_ACTIVATION_BF16"],
    },
    {
        "id": "routed-gather-kernel",
        "file": "jang_tools/turboquant/gather_tq_kernel.py",
        "role": "routed down/fc2 gather matmul kernel for selected experts",
        "anchors": ["def gather_tq_matmul", "make_gather_tq_decode_broadcast", "make_gather_tq_decode_per_row", "JANGTQ_GATHER_OPT"],
    },
    {
        "id": "fused-gate-up-kernel",
        "file": "jang_tools/turboquant/fused_gate_up_kernel.py",
        "role": "fused gate/up/SwiGLU routed expert path used by SwitchMLP",
        "anchors": ["def fused_gate_up_swiglu_matmul", "make_fused_gate_up_swiglu_decode", "JANGTQ_MPP_NAX"],
    },
    {
        "id": "grouped-nax-proof-surface",
        "file": "jang_tools/turboquant/mpp_nax_kernel.py",
        "role": "same-expert grouped tile helpers for possible routed scheduling work",
        "anchors": ["build_sorted_group_tiles", "gather_tq_matmul_mpp_nax_grouped_from_rot", "fused_gate_up_swiglu_mpp_nax_grouped_from_rot"],
    },
    {
        "id": "moe-component-proof",
        "file": "examples/nemotron_ultra/moe_component_probe.py",
        "role": "measures gate, switch_mlp, weighted_decode, shared_experts, weighted sum, and full_moe",
        "anchors": ["moe.switch_mlp", "weighted_decode", "shared_experts", "full_moe"],
    },
    {
        "id": "candidate-verdict-proof",
        "file": "examples/nemotron_ultra/experiment_result_check.py",
        "role": "accepts/rejects the MoE candidate using compare, speed gate, and handoff invariants",
        "anchors": ["moe-routed-shared-scheduling", "MoE lane did not improve moe_ms", "ACCEPTED"],
    },
]


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _anchor_lines(path: Path, anchors: list[str]) -> dict[str, int | None]:
    if not path.exists():
        return {anchor: None for anchor in anchors}
    lines = path.read_text(errors="replace").splitlines()
    found: dict[str, int | None] = {}
    for anchor in anchors:
        line_no = next((idx for idx, line in enumerate(lines, start=1) if anchor in line), None)
        found[anchor] = line_no
    return found


def _component_timings(log_dir: Path) -> dict[str, float]:
    data = _load(log_dir / "2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json") or {}
    rows = data.get("layers") or []
    if not rows:
        return {}
    return {
        str(item.get("label")): float(item.get("median_ms"))
        for item in rows[0].get("timings", [])
        if isinstance(item.get("median_ms"), (int, float))
    }


def _build_result(log_dir: Path, source_root: Path) -> dict[str, Any]:
    contract = _load(log_dir / "2026-06-04-nemotron-ultra-runtime-moe-candidate-contract.json") or {}
    ticket = _load(log_dir / "2026-06-04-nemotron-ultra-runtime-moe-execution-ticket.json") or {}
    budget = _load(log_dir / "2026-06-04-nemotron-ultra-token-speed-budget.json") or {}
    timings = _component_timings(log_dir)

    surface_rows = []
    missing: list[str] = []
    for surface in SURFACES:
        path = source_root / surface["file"]
        anchors = _anchor_lines(path, surface["anchors"])
        missing_anchors = [anchor for anchor, line_no in anchors.items() if line_no is None]
        if not path.exists():
            missing.append(f"missing source file: {path}")
        missing.extend(f"{surface['id']} missing anchor: {anchor}" for anchor in missing_anchors)
        surface_rows.append(
            {
                "id": surface["id"],
                "file": str(path),
                "role": surface["role"],
                "anchors": anchors,
                "status": "READY" if path.exists() and not missing_anchors else "MISSING",
            }
        )

    status = "READY" if not missing else "BLOCKED"
    first_target = (budget.get("targets") or [{}])[0]
    return {
        "status": status,
        "log_dir": str(log_dir),
        "source_root": str(source_root),
        "lane_id": contract.get("lane_id") or ticket.get("lane_id"),
        "ticket_status": ticket.get("status"),
        "contract_status": contract.get("status"),
        "current_speed": contract.get("current_speed", {}),
        "target": contract.get("target") or first_target,
        "invariants": contract.get("invariants", {}),
        "component_timings_ms": timings,
        "runtime_controls": {
            "disable_weighted_moe_fastpath": "JANGTQ_DISABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH=1",
            "disable_switchmlp_fastpath": "JANGTQ_DISABLE_NEMOTRON_SWITCHMLP_FASTPATH=1",
            "legacy_disable_switchmlp_fastpath": "JANGTQ_ENABLE_NEMOTRON_SWITCHMLP_FASTPATH=0",
            "disable_activation_bf16": "JANGTQ_DISABLE_NEMOTRON_ACTIVATION_BF16=1",
            "gather_opt": "JANGTQ_GATHER_OPT",
            "mpp_nax": "JANGTQ_MPP_NAX",
            "mpp_nax_strict": "JANGTQ_MPP_NAX_STRICT=1",
        },
        "surfaces": surface_rows,
        "missing": missing,
        "candidate_command": ticket.get("commands", {}).get("candidate") or contract.get("candidate_command"),
        "post_check_command": ticket.get("commands", {}).get("post_check") or contract.get("post_check_command"),
        "non_goals": [
            "do not edit vMLX or MLX Studio for this JANG handoff",
            "do not expand routed 1-bit or shared 8-bit tensors to full precision",
            "do not lower top-k or hide parser/coherence issues to make a speed row look better",
            "do not run the Mamba lane until the MoE lane has accepted evidence",
        ],
        "source_files": [
            str(log_dir / "2026-06-04-nemotron-ultra-runtime-moe-candidate-contract.json"),
            str(log_dir / "2026-06-04-nemotron-ultra-runtime-moe-execution-ticket.json"),
            str(log_dir / "2026-06-04-nemotron-ultra-token-speed-budget.json"),
            str(log_dir / "2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json"),
        ],
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _render(result: dict[str, Any]) -> str:
    lines = [
        "# Nemotron Ultra MoE Runtime Surface Map",
        "",
        f"log_dir: `{result['log_dir']}`",
        f"source_root: `{result['source_root']}`",
        f"lane_id: `{result.get('lane_id')}`",
        f"status: `{result['status']}`",
        f"ticket_status: `{result.get('ticket_status')}`",
        f"contract_status: `{result.get('contract_status')}`",
        "",
        "## Target",
    ]
    for key, value in result["target"].items():
        lines.append(f"- {key}: `{_fmt(value)}`")
    lines.extend(["", "## Current Speed"])
    for key, value in result["current_speed"].items():
        lines.append(f"- {key}: `{_fmt(value)}`")
    lines.extend(["", "## Component Timings"])
    for key, value in sorted(result["component_timings_ms"].items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- {key}: `{value:.3f} ms`")
    lines.extend(
        [
            "",
            "## Surfaces",
            "| id | status | file | role | anchors |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in result["surfaces"]:
        anchors = ", ".join(f"{anchor}@{line}" for anchor, line in row["anchors"].items())
        lines.append(f"| `{row['id']}` | `{row['status']}` | `{row['file']}` | {row['role']} | {anchors} |")
    lines.extend(["", "## Runtime Controls"])
    for key, value in result["runtime_controls"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Commands"])
    lines.append(f"- candidate: `{result.get('candidate_command')}`")
    lines.append(f"- post_check: `{result.get('post_check_command')}`")
    lines.extend(["", "## Non Goals"])
    lines.extend(f"- {item}" for item in result["non_goals"])
    lines.extend(["", "## Missing"])
    lines.extend(f"- {item}" for item in result["missing"] or ["none"])
    lines.extend(["", "## Source Files"])
    lines.extend(f"- `{item}`" for item in result["source_files"])
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    result = _build_result(args.log_dir, args.source_root)
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
