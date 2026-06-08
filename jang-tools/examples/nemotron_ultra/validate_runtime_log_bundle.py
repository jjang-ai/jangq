"""Validate a Nemotron Ultra runtime log bundle.

This script is no-load. It checks that a baseline or candidate log directory has
the JSON artifacts consumed by the status, plan, gate, and compare scripts, and
that those artifacts contain the required metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_FILES = {
    "live speed": "2026-06-04-nemotron-ultra-jangtq1l-bf16-activation-live-probe.json",
    "layer decode": "2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json",
    "long coherence": "2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json",
    "mamba component": "2026-06-04-nemotron-ultra-mamba-component-probe.json",
    "moe component": "2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json",
    "projection tradeoff": "2026-06-04-nemotron-ultra-projection-tradeoff-probe.json",
}


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _has_numeric(value: Any) -> bool:
    return isinstance(value, (int, float))


def _validate(log_dir: Path) -> tuple[int, str]:
    failures: list[str] = []
    fixed: list[str] = []
    data: dict[str, dict[str, Any]] = {}

    for label, filename in REQUIRED_FILES.items():
        path = log_dir / filename
        loaded = _load(path)
        if loaded is None:
            failures.append(f"missing {label} log: {filename}")
        else:
            fixed.append(f"found {label} log: {filename}")
            data[label] = loaded

    live = data.get("live speed")
    if live is not None:
        rows = live.get("rows", [])
        if not rows or not any(_has_numeric(row.get("decode_tps_excluding_first")) for row in rows):
            failures.append("live speed log has no numeric decode_tps_excluding_first row")

    layer = data.get("layer decode")
    if layer is not None:
        buckets = layer.get("summary_by_block_type", {})
        for key, name in (("E", "MoE"), ("M", "Mamba"), ("*", "attention")):
            if not _has_numeric(buckets.get(key, {}).get("total_ms")):
                failures.append(f"layer decode log missing {name} total_ms")
        if not _has_numeric(layer.get("manual_decode_total_ms")):
            failures.append("layer decode log missing manual_decode_total_ms")
        if not _has_numeric(layer.get("norm_lm_head_ms")):
            failures.append("layer decode log missing norm_lm_head_ms")

    coherence = data.get("long coherence")
    if coherence is not None:
        rows = coherence.get("rows", [])
        if not rows:
            failures.append("long coherence log has no rows")
        for row in rows:
            if "visible_marker_leaks" not in row:
                failures.append(f"long coherence row {row.get('id')} missing visible_marker_leaks")
            if "ngram_repeat" not in row:
                failures.append(f"long coherence row {row.get('id')} missing ngram_repeat")
            if "eos_reached" not in row:
                failures.append(f"long coherence row {row.get('id')} missing eos_reached")

    mamba = data.get("mamba component")
    if mamba is not None:
        labels = {
            item.get("label")
            for layer_row in mamba.get("layers", [])
            for item in layer_row.get("timings", [])
        }
        for label in ("in_proj", "out_proj", "conv", "ssm_update", "full_mamba_mixer"):
            if label not in labels:
                failures.append(f"mamba component log missing timing {label}")

    moe = data.get("moe component")
    if moe is not None:
        labels = {
            item.get("label")
            for layer_row in moe.get("layers", [])
            for item in layer_row.get("timings", [])
        }
        for label in ("switch_mlp", "shared_experts", "full_moe"):
            if label not in labels:
                failures.append(f"moe component log missing timing {label}")

    projection = data.get("projection tradeoff")
    if projection is not None:
        names = {row.get("name") for row in projection.get("results", [])}
        for name in ("mamba_in_proj", "mamba_out_proj", "shared_up", "shared_down"):
            if name not in names:
                failures.append(f"projection tradeoff log missing {name}")

    status = "BLOCKED" if failures else "FIXED"
    lines = [
        "# Nemotron Ultra Runtime Log Bundle Validation",
        "",
        f"log_dir: `{log_dir}`",
        f"status: `{status}`",
        "",
        "## Found",
    ]
    lines.extend(f"- {item}" for item in fixed)
    lines.extend(["", "## Failures"])
    lines.extend(f"- {item}" for item in failures)
    return (2 if failures else 0), "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    code, report = _validate(args.log_dir)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
    sys.stdout.write(report)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
