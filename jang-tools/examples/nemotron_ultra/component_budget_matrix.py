"""Build a no-load component budget matrix for Nemotron Ultra decode speed.

The matrix combines saved layer-decode totals, component probe medians, and
token/s target budgets. It answers: which measured substeps are large enough to
matter, and how much token/s movement should a 25/50/100% cut buy?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_LOG_DIR = Path("docs/runtime/logs")
DEFAULT_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-component-budget-matrix.md")
DEFAULT_JSON_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-component-budget-matrix.json")


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _first_existing(log_dir: Path, names: tuple[str, ...]) -> dict[str, Any]:
    for name in names:
        data = _load(log_dir / name)
        if data:
            data["_file"] = name
            return data
    return {}


def _component_medians(data: dict[str, Any]) -> dict[str, float]:
    layers = data.get("layers", [])
    if not layers:
        return {}
    timings = layers[0].get("timings", [])
    return {
        str(item["label"]): float(item["median_ms"])
        for item in timings
        if isinstance(item.get("median_ms"), (int, float))
    }


def _projection_map(data: dict[str, Any]) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for item in data.get("results", []):
        name = str(item.get("name"))
        quantized = item.get("quantized", {})
        bf16 = item.get("bf16_dequantized", {})
        rows[name] = {
            "quantized_ms": float(quantized.get("median_ms", 0.0)),
            "bf16_ms": float(bf16.get("median_ms", 0.0)),
            "quantized_speedup_vs_bf16": float(item.get("quantized_speedup_vs_bf16", 0.0)),
        }
    return rows


def _tps(total_ms: float) -> float | None:
    return 1000.0 / total_ms if total_ms > 0 else None


def _component_role(label: str) -> str:
    if label.startswith("full_"):
        return "inclusive_path"
    if label == "weighted_decode":
        return "fused_fast_path"
    return "substep"


def _component_row(
    *,
    family: str,
    label: str,
    median_ms: float,
    layer_count: int,
    family_total_ms: float,
    manual_total_ms: float,
) -> dict[str, Any]:
    projected_total = median_ms * layer_count
    coverage_pct = (projected_total / family_total_ms) * 100.0 if family_total_ms else 0.0
    scenarios = []
    for cut_pct in (25.0, 50.0, 100.0):
        cut_ms = projected_total * (cut_pct / 100.0)
        new_total = max(manual_total_ms - cut_ms, 1e-9)
        scenarios.append(
            {
                "cut_pct": cut_pct,
                "cut_ms": cut_ms,
                "new_manual_ms": new_total,
                "new_manual_tps": _tps(new_total),
                "tps_gain": (_tps(new_total) or 0.0) - (_tps(manual_total_ms) or 0.0),
            }
        )
    return {
        "family": family,
        "label": label,
        "role": _component_role(label),
        "median_ms_per_measured_layer": median_ms,
        "layer_count": layer_count,
        "projected_total_ms": projected_total,
        "coverage_pct_of_family_total": coverage_pct,
        "scenarios": scenarios,
    }


def _target_hit_rows(current_total_ms: float, rows: list[dict[str, Any]], targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for target in targets:
        target_tps = float(target["target_tps"])
        target_ms = float(target["target_ms_per_token"])
        required_cut = max(current_total_ms - target_ms, 0.0)
        enough_single_components = [
            f"{row['family']}:{row['label']}"
            for row in rows
            if row["projected_total_ms"] >= required_cut
        ]
        hits.append(
            {
                "target_tps": target_tps,
                "target_ms_per_token": target_ms,
                "required_total_cut_ms": required_cut,
                "single_component_can_cover": enough_single_components,
            }
        )
    return hits


def _build_result(log_dir: Path) -> dict[str, Any]:
    layer = _first_existing(
        log_dir,
        (
            "2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json",
            "2026-06-04-nemotron-ultra-layer-decode-current-probe.json",
        ),
    )
    moe = _first_existing(
        log_dir,
        (
            "2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json",
            "2026-06-04-nemotron-ultra-moe-component-current-probe.json",
            "2026-06-04-nemotron-ultra-moe-component-probe.json",
        ),
    )
    mamba = _first_existing(log_dir, ("2026-06-04-nemotron-ultra-mamba-component-probe.json",))
    projection = _first_existing(log_dir, ("2026-06-04-nemotron-ultra-projection-tradeoff-probe.json",))
    token_budget = _first_existing(log_dir, ("2026-06-04-nemotron-ultra-token-speed-budget.json",))

    buckets = layer.get("summary_by_block_type", {})
    manual_total_ms = float(layer.get("manual_decode_total_ms") or 0.0)
    moe_total = float(buckets.get("E", {}).get("total_ms", 0.0))
    mamba_total = float(buckets.get("M", {}).get("total_ms", 0.0))
    attention_total = float(buckets.get("*", {}).get("total_ms", 0.0))
    norm_lm_head = float(layer.get("norm_lm_head_ms") or 0.0)
    moe_count = int(buckets.get("E", {}).get("count", 48) or 48)
    mamba_count = int(buckets.get("M", {}).get("count", 48) or 48)

    rows: list[dict[str, Any]] = []
    for label, value in sorted(_component_medians(moe).items(), key=lambda item: item[1], reverse=True):
        rows.append(
            _component_row(
                family="MoE",
                label=label,
                median_ms=value,
                layer_count=moe_count,
                family_total_ms=moe_total,
                manual_total_ms=manual_total_ms,
            )
        )
    for label, value in sorted(_component_medians(mamba).items(), key=lambda item: item[1], reverse=True):
        rows.append(
            _component_row(
                family="Mamba",
                label=label,
                median_ms=value,
                layer_count=mamba_count,
                family_total_ms=mamba_total,
                manual_total_ms=manual_total_ms,
            )
        )

    targets = token_budget.get("targets", [])
    return {
        "log_dir": str(log_dir),
        "sources": {
            "layer": layer.get("_file"),
            "moe": moe.get("_file"),
            "mamba": mamba.get("_file"),
            "projection": projection.get("_file"),
            "token_budget": token_budget.get("_file"),
        },
        "current": {
            "manual_decode_total_ms": manual_total_ms,
            "manual_implied_tps": _tps(manual_total_ms),
            "moe_ms": moe_total,
            "mamba_ms": mamba_total,
            "attention_ms": attention_total,
            "norm_lm_head_ms": norm_lm_head,
            "moe_mamba_pct": ((moe_total + mamba_total) / manual_total_ms) * 100.0 if manual_total_ms else 0.0,
        },
        "component_rows": rows,
        "target_hits": _target_hit_rows(manual_total_ms, rows, targets),
        "projection_tradeoff": _projection_map(projection),
        "interpretation": [
                "Component/path totals are projected from first measured layer medians; use them for ranking, not final proof.",
                "`full_*` rows are inclusive path measurements, not additive leaf substeps.",
            "Rows with large projected totals are plausible speed targets; small rows cannot move token/s enough alone.",
            "The current projection tradeoff says quantized 8-bit affine projections are faster than temporary BF16 copies.",
        ],
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _render(result: dict[str, Any]) -> str:
    current = result["current"]
    lines = [
        "# Nemotron Ultra Component Budget Matrix",
        "",
        f"log_dir: `{result['log_dir']}`",
        "",
        "## Sources",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in result["sources"].items())
    lines.extend(
        [
            "",
            "## Current Baseline",
            f"- manual_decode_total_ms: `{_fmt(current['manual_decode_total_ms'])}`",
            f"- manual_implied_tps: `{_fmt(current['manual_implied_tps'])}`",
            f"- moe_ms: `{_fmt(current['moe_ms'])}`",
            f"- mamba_ms: `{_fmt(current['mamba_ms'])}`",
            f"- attention_ms: `{_fmt(current['attention_ms'])}`",
            f"- norm_lm_head_ms: `{_fmt(current['norm_lm_head_ms'])}`",
            f"- moe_mamba_pct: `{_fmt(current['moe_mamba_pct'])}`",
            "",
            "## Component Cut Scenarios",
            "| family | role | component | per-layer median | projected total | family coverage | 25% cut tps | 50% cut tps | 100% cut tps |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in result["component_rows"]:
        scenarios = {item["cut_pct"]: item for item in row["scenarios"]}
        lines.append(
            "| {family} | `{role}` | `{label}` | `{median:.3f}` | `{total:.3f}` | `{coverage:.1f}%` | "
            "`{t25:.3f}` | `{t50:.3f}` | `{t100:.3f}` |".format(
                family=row["family"],
                role=row["role"],
                label=row["label"],
                median=row["median_ms_per_measured_layer"],
                total=row["projected_total_ms"],
                coverage=row["coverage_pct_of_family_total"],
                t25=scenarios[25.0]["new_manual_tps"],
                t50=scenarios[50.0]["new_manual_tps"],
                t100=scenarios[100.0]["new_manual_tps"],
            )
        )
    lines.extend(["", "## Target Coverage"])
    for hit in result["target_hits"]:
        components = hit["single_component_can_cover"]
        if components:
            suffix = ", ".join(f"`{item}`" for item in components)
        else:
            suffix = "none"
        lines.append(
            f"- `{hit['target_tps']:.3f}` tok/s needs `{hit['required_total_cut_ms']:.3f}` ms cut; single measured row/path enough: {suffix}"
        )
    if result["projection_tradeoff"]:
        lines.extend(["", "## Projection Tradeoff"])
        for name, row in result["projection_tradeoff"].items():
            lines.append(
                f"- `{name}`: quantized `{row['quantized_ms']:.3f}` ms, BF16 `{row['bf16_ms']:.3f}` ms, speedup `{row['quantized_speedup_vs_bf16']:.2f}x`"
            )
    lines.extend(["", "## Interpretation"])
    lines.extend(f"- {item}" for item in result["interpretation"])
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
