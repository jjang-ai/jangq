"""Compute token/s target budgets from saved Nemotron Ultra speed logs.

This is a no-load planning tool. It reads the current layer-decode split and
live-speed rows, then turns target token/s values into concrete millisecond
reduction budgets for MoE and Mamba work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_LOG_DIR = Path("docs/runtime/logs")
DEFAULT_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-token-speed-budget.md")
DEFAULT_JSON_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-token-speed-budget.json")


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _layer_log(log_dir: Path) -> dict[str, Any]:
    for name in (
        "2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json",
        "2026-06-04-nemotron-ultra-layer-decode-current-probe.json",
    ):
        data = _load(log_dir / name)
        if data:
            data["_file"] = name
            return data
    return {}


def _best_live_speed(log_dir: Path) -> tuple[str, str, float] | None:
    best: tuple[str, str, float] | None = None
    for path in sorted(log_dir.glob("*live-probe.json")):
        data = _load(path)
        if not data:
            continue
        for row in data.get("rows", []):
            value = row.get("decode_tps_excluding_first")
            if not isinstance(value, (int, float)):
                continue
            item = (path.name, str(row.get("id", "?")), float(value))
            if best is None or item[2] > best[2]:
                best = item
    return best


def _target_row(target_tps: float, total_ms: float, moe_ms: float, mamba_ms: float) -> dict[str, Any]:
    target_ms = 1000.0 / target_tps
    required_cut_ms = max(total_ms - target_ms, 0.0)
    combined_ms = moe_ms + mamba_ms
    reachable_by_moe_mamba = required_cut_ms <= combined_ms
    moe_share = moe_ms / combined_ms if combined_ms else 0.0
    mamba_share = mamba_ms / combined_ms if combined_ms else 0.0
    moe_cut = required_cut_ms * moe_share
    mamba_cut = required_cut_ms * mamba_share
    return {
        "target_tps": target_tps,
        "target_ms_per_token": target_ms,
        "required_total_cut_ms": required_cut_ms,
        "required_total_cut_pct": (required_cut_ms / total_ms) * 100.0 if total_ms else 0.0,
        "reachable_by_moe_mamba_only": reachable_by_moe_mamba,
        "moe_cut_ms_proportional": moe_cut,
        "mamba_cut_ms_proportional": mamba_cut,
        "moe_cut_pct_of_current_moe": (moe_cut / moe_ms) * 100.0 if moe_ms else 0.0,
        "mamba_cut_pct_of_current_mamba": (mamba_cut / mamba_ms) * 100.0 if mamba_ms else 0.0,
        "moe_per_layer_cut_ms": moe_cut / 48.0,
        "mamba_per_layer_cut_ms": mamba_cut / 48.0,
    }


def _build_result(log_dir: Path, targets: list[float]) -> dict[str, Any]:
    layer = _layer_log(log_dir)
    best = _best_live_speed(log_dir)
    buckets = layer.get("summary_by_block_type", {})
    total_ms = float(layer.get("manual_decode_total_ms") or 0.0)
    moe_ms = float(buckets.get("E", {}).get("total_ms", 0.0))
    mamba_ms = float(buckets.get("M", {}).get("total_ms", 0.0))
    attention_ms = float(buckets.get("*", {}).get("total_ms", 0.0))
    norm_lm_head_ms = float(layer.get("norm_lm_head_ms") or 0.0)
    other_ms = max(total_ms - moe_ms - mamba_ms - attention_ms - norm_lm_head_ms, 0.0)
    manual_tps = 1000.0 / total_ms if total_ms else None
    target_rows = [_target_row(target, total_ms, moe_ms, mamba_ms) for target in targets]
    return {
        "log_dir": str(log_dir),
        "layer_log": layer.get("_file"),
        "current": {
            "manual_decode_total_ms": total_ms or None,
            "manual_implied_tps": manual_tps,
            "best_live_tps": best[2] if best else None,
            "best_live_source": f"{best[0]}::{best[1]}" if best else None,
            "moe_ms": moe_ms,
            "mamba_ms": mamba_ms,
            "attention_ms": attention_ms,
            "norm_lm_head_ms": norm_lm_head_ms,
            "other_ms": other_ms,
            "moe_plus_mamba_ms": moe_ms + mamba_ms,
            "moe_plus_mamba_pct_of_total": ((moe_ms + mamba_ms) / total_ms) * 100.0 if total_ms else 0.0,
        },
        "targets": target_rows,
        "interpretation": [
            "Use manual synchronized decode for millisecond budgets; use live speed for user-visible baseline.",
            "If a target is not reachable by MoE/Mamba only, attention/lm_head/loop work also must move.",
            "Per-layer cuts are proportional planning numbers, not proof of an implementation strategy.",
        ],
    }


def _render(result: dict[str, Any]) -> str:
    current = result["current"]
    lines = [
        "# Nemotron Ultra Token Speed Budget",
        "",
        f"log_dir: `{result['log_dir']}`",
        f"layer_log: `{result.get('layer_log')}`",
        "",
        "## Current Baseline",
    ]
    if current["manual_decode_total_ms"] is not None:
        lines.append(f"- manual_decode_total_ms: `{current['manual_decode_total_ms']:.3f}`")
        lines.append(f"- manual_implied_tps: `{current['manual_implied_tps']:.3f}`")
    if current["best_live_tps"] is not None:
        lines.append(f"- best_live_tps: `{current['best_live_tps']:.3f}` from `{current['best_live_source']}`")
    lines.append(f"- moe_ms: `{current['moe_ms']:.3f}`")
    lines.append(f"- mamba_ms: `{current['mamba_ms']:.3f}`")
    lines.append(f"- attention_ms: `{current['attention_ms']:.3f}`")
    lines.append(f"- norm_lm_head_ms: `{current['norm_lm_head_ms']:.3f}`")
    lines.append(f"- other_ms: `{current['other_ms']:.3f}`")
    lines.append(f"- moe_plus_mamba: `{current['moe_plus_mamba_ms']:.3f}` ms (`{current['moe_plus_mamba_pct_of_total']:.2f}%` of manual decode)")
    lines.extend(
        [
            "",
            "## Target Budgets",
            "| target tok/s | target ms/token | total cut needed | total cut % | MoE cut | Mamba cut | per MoE layer | per Mamba layer | MoE/Mamba enough |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in result["targets"]:
        lines.append(
            "| `{target_tps:.3f}` | `{target_ms_per_token:.3f}` | `{required_total_cut_ms:.3f}` | "
            "`{required_total_cut_pct:.2f}%` | `{moe_cut_ms_proportional:.3f}` "
            "(`{moe_cut_pct_of_current_moe:.2f}%`) | `{mamba_cut_ms_proportional:.3f}` "
            "(`{mamba_cut_pct_of_current_mamba:.2f}%`) | `{moe_per_layer_cut_ms:.4f}` | "
            "`{mamba_per_layer_cut_ms:.4f}` | `{reachable_by_moe_mamba_only}` |".format(**row)
        )
    lines.extend(["", "## Interpretation"])
    lines.extend(f"- {item}" for item in result["interpretation"])
    return "\n".join(lines) + "\n"


def _parse_targets(raw: str) -> list[float]:
    targets = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value <= 0:
            raise argparse.ArgumentTypeError("targets must be positive token/s values")
        targets.append(value)
    if not targets:
        raise argparse.ArgumentTypeError("at least one target token/s value is required")
    return targets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--targets", type=_parse_targets, default=_parse_targets("10,12,15"))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    result = _build_result(args.log_dir, args.targets)
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
