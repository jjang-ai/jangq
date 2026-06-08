"""Summarize Nemotron Ultra JANGTQ_1L runtime proof logs.

This script does not load the model. It reads existing JSON probe outputs and
prints a compact status report for speed, coherence, parser leakage, warmup,
and cache/VL readiness.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_LOG_DIR = Path("docs/runtime/logs")


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _row_speed(row: dict[str, Any]) -> float | None:
    for key in (
        "decode_tps_excluding_first",
        "decode_tps",
        "tok_per_s",
        "tokens_per_second",
    ):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _collect_live_speeds(log_dir: Path) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    for path in sorted(log_dir.glob("*live-probe.json")):
        data = _load(path)
        if not data:
            continue
        for row in data.get("rows", []):
            speed = _row_speed(row)
            if speed is not None:
                rows.append((path.name, str(row.get("id", "?")), speed))
    long_data = _load(log_dir / "2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json")
    if long_data:
        for row in long_data.get("rows", []):
            speed = _row_speed(row)
            if speed is not None:
                rows.append(("2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json", str(row.get("id", "?")), speed))
    return rows


def _best_live_speed(rows: list[tuple[str, str, float]]) -> tuple[str, str, float] | None:
    if not rows:
        return None
    return max(rows, key=lambda item: item[2])


def _coherence_status(log_dir: Path) -> dict[str, Any]:
    data = _load(log_dir / "2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json")
    if not data:
        return {"status": "MISSING", "reason": "long coherence log missing"}
    row_summaries = []
    any_leak = False
    any_repeat = False
    any_missing_expected = False
    any_no_eos = False
    for row in data.get("rows", []):
        expected_found = row.get("expected_found", {})
        leaks = row.get("visible_marker_leaks", [])
        repeat = row.get("ngram_repeat", {}).get("repeat_fraction", 0.0)
        missing_expected = not all(bool(v) for v in expected_found.values())
        no_eos = not bool(row.get("eos_reached"))
        any_leak = any_leak or bool(leaks)
        any_repeat = any_repeat or repeat > 0.25
        any_missing_expected = any_missing_expected or missing_expected
        any_no_eos = any_no_eos or no_eos
        row_summaries.append(
            {
                "id": row.get("id"),
                "eos": bool(row.get("eos_reached")),
                "speed": _row_speed(row),
                "leaks": leaks,
                "repeat_fraction": repeat,
                "expected_found": expected_found,
            }
        )
    status = "FIXED"
    reasons: list[str] = []
    if any_missing_expected:
        status = "PARTIAL"
        reasons.append("expected answer substring missing in at least one row")
    if any_leak:
        status = "PARTIAL"
        reasons.append("visible parser marker leakage")
    if any_repeat:
        status = "PARTIAL"
        reasons.append("high repeated n-gram fraction")
    if any_no_eos:
        status = "PARTIAL"
        reasons.append("at least one row did not reach EOS")
    return {"status": status, "reasons": reasons, "rows": row_summaries}


def _layer_split(log_dir: Path) -> dict[str, Any]:
    data = _load(log_dir / "2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json")
    if not data:
        data = _load(log_dir / "2026-06-04-nemotron-ultra-layer-decode-current-probe.json")
    if not data:
        return {"status": "MISSING", "reason": "layer decode split log missing"}
    buckets = data.get("summary_by_block_type", {})
    return {
        "status": "FOUND",
        "decode_token": data.get("decode_token"),
        "manual_decode_total_ms": data.get("manual_decode_total_ms"),
        "norm_lm_head_ms": data.get("norm_lm_head_ms"),
        "buckets": buckets,
    }


def _projection_tradeoffs(log_dir: Path) -> dict[str, Any]:
    data = _load(log_dir / "2026-06-04-nemotron-ultra-projection-tradeoff-probe.json")
    if not data:
        return {"status": "MISSING", "reason": "projection tradeoff log missing"}
    return {"status": "FOUND", "rows": data.get("results", [])}


def _mamba_components(log_dir: Path) -> dict[str, Any]:
    data = _load(log_dir / "2026-06-04-nemotron-ultra-mamba-component-probe.json")
    if not data:
        return {"status": "MISSING", "reason": "Mamba component log missing"}
    layers = data.get("layers", [])
    if not layers:
        return {"status": "MISSING", "reason": "Mamba component log has no layer rows"}
    timings = {
        str(item["label"]): float(item["median_ms"])
        for item in layers[0].get("timings", [])
        if isinstance(item.get("median_ms"), (int, float))
    }
    return {
        "status": "FOUND",
        "layer_index": layers[0].get("layer_index"),
        "cache_ordinal": layers[0].get("cache_ordinal"),
        "timings": timings,
    }


def _render_report(log_dir: Path) -> str:
    speeds = _collect_live_speeds(log_dir)
    best = _best_live_speed(speeds)
    coherence = _coherence_status(log_dir)
    layer_split = _layer_split(log_dir)
    projection = _projection_tradeoffs(log_dir)
    mamba = _mamba_components(log_dir)
    lines: list[str] = []

    lines.append("# Nemotron Ultra Runtime Status")
    lines.append("")
    lines.append(f"log_dir: {log_dir}")
    lines.append("")

    lines.append("## Speed")
    if best:
        lines.append(f"FIXED/PARTIAL: best observed warm decode row is {best[2]:.3f} tok/s")
        lines.append(f"source: {best[0]} :: {best[1]}")
        lines.append("remaining: MoE/Mamba forward overhead; not sampler or generic generation loop")
    else:
        lines.append("MISSING: no live speed rows found")
    lines.append("")

    lines.append("## Coherence")
    lines.append(f"{coherence['status']}: {', '.join(coherence.get('reasons', [])) or 'all rows passed'}")
    for row in coherence.get("rows", []):
        lines.append(
            "- {id}: eos={eos} speed={speed} leaks={leaks} repeat_fraction={repeat_fraction}".format(
                **row
            )
        )
    lines.append("")

    lines.append("## Layer Split")
    lines.append(layer_split["status"])
    if layer_split.get("buckets"):
        lines.append(f"manual_decode_total_ms: {layer_split.get('manual_decode_total_ms')}")
        lines.append(f"norm_lm_head_ms: {layer_split.get('norm_lm_head_ms')}")
        for block_type, name in (("E", "MoE"), ("M", "Mamba"), ("*", "Attention")):
            bucket = layer_split["buckets"].get(block_type, {})
            lines.append(
                "- {name}: total_ms={total_ms} count={count} median_ms={median_ms}".format(
                    name=name,
                    total_ms=round(float(bucket.get("total_ms", 0.0)), 3),
                    count=bucket.get("count", "?"),
                    median_ms=round(float(bucket.get("median_ms", 0.0)), 3),
                )
            )
    lines.append("")

    lines.append("## Projection Tradeoff")
    lines.append(projection["status"])
    if projection.get("rows"):
        lines.append("Use quantized 8-bit affine projections unless a new probe proves otherwise.")
        for row in projection["rows"]:
            lines.append(
                "- {name}: quantized_median_ms={q:.3f} bf16_median_ms={b:.3f} speedup={s:.2f}x".format(
                    name=row.get("name"),
                    q=float(row.get("quantized", {}).get("median_ms", 0.0)),
                    b=float(row.get("bf16_dequantized", {}).get("median_ms", 0.0)),
                    s=float(row.get("quantized_speedup_vs_bf16", 0.0)),
                )
            )
    lines.append("")

    lines.append("## Mamba Component")
    lines.append(mamba["status"])
    if mamba.get("timings"):
        for label in (
            "outer_norm",
            "in_proj",
            "conv",
            "ssm_update",
            "mamba_norm_gated",
            "out_proj",
            "full_mamba_mixer",
        ):
            if label in mamba["timings"]:
                lines.append(f"- {label}: median_ms={mamba['timings'][label]:.3f}")
        lines.append("Interpretation: projection/dispatch fusion is a better first target than a Python-level conv rewrite.")
    lines.append("")

    lines.append("## Cache / VL Gates")
    lines.append("PARTIAL: cache and VL gates are documented, not live-proven in vMLX.")
    lines.append("- TurboQuant KV only covers 12 attention layers.")
    lines.append("- Full prefix hit also requires 48 Mamba companion states.")
    lines.append("- Parser streaming state must be salted/restored.")
    lines.append("- This artifact is text-only; media requests must reject or reroute.")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    report = _render_report(args.log_dir)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
    sys.stdout.write(report)


if __name__ == "__main__":
    main()
