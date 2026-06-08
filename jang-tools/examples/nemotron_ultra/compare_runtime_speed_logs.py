"""Compare two Nemotron Ultra runtime proof log directories.

This is a no-load before/after comparator. Point `--baseline-log-dir` at the
current saved logs and `--candidate-log-dir` at logs from a runtime change. The
script reports token/s, layer-bucket, and coherence deltas, and flags material
regressions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_LOG_DIR = Path("docs/runtime/logs")
DEFAULT_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-compare.md")
DEFAULT_JSON_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-compare.json")


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _best_speed(log_dir: Path) -> tuple[str, str, float] | None:
    best: tuple[str, str, float] | None = None
    for path in sorted(log_dir.glob("*live-probe.json")):
        data = _load(path)
        if not data:
            continue
        for row in data.get("rows", []):
            speed = row.get("decode_tps_excluding_first")
            if not isinstance(speed, (int, float)):
                continue
            item = (path.name, str(row.get("id", "?")), float(speed))
            if best is None or item[2] > best[2]:
                best = item
    return best


def _layer(log_dir: Path) -> dict[str, Any]:
    for name in (
        "2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json",
        "2026-06-04-nemotron-ultra-layer-decode-current-probe.json",
    ):
        data = _load(log_dir / name)
        if data:
            buckets = data.get("summary_by_block_type", {})
            return {
                "file": name,
                "manual_decode_total_ms": float(data.get("manual_decode_total_ms", 0.0)),
                "norm_lm_head_ms": float(data.get("norm_lm_head_ms", 0.0)),
                "moe_ms": float(buckets.get("E", {}).get("total_ms", 0.0)),
                "mamba_ms": float(buckets.get("M", {}).get("total_ms", 0.0)),
                "attention_ms": float(buckets.get("*", {}).get("total_ms", 0.0)),
            }
    return {}


def _coherence(log_dir: Path, max_repeat_fraction: float) -> dict[str, Any]:
    data = _load(log_dir / "2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json")
    if not data:
        return {"missing": True, "leaks": [], "repeats": [], "no_eos": []}
    leaks = []
    repeats = []
    no_eos = []
    for row in data.get("rows", []):
        row_id = str(row.get("id"))
        if row.get("visible_marker_leaks"):
            leaks.append(row_id)
        repeat = row.get("ngram_repeat", {}).get("repeat_fraction", 0.0)
        if isinstance(repeat, (int, float)) and repeat > max_repeat_fraction:
            repeats.append(row_id)
        if not row.get("eos_reached"):
            no_eos.append(row_id)
    return {"missing": False, "leaks": leaks, "repeats": repeats, "no_eos": no_eos}


def _metrics(log_dir: Path, max_repeat_fraction: float) -> dict[str, Any]:
    best = _best_speed(log_dir)
    layer = _layer(log_dir)
    coherence = _coherence(log_dir, max_repeat_fraction)
    return {
        "best_file": best[0] if best else "",
        "best_row": best[1] if best else "",
        "best_tps": best[2] if best else 0.0,
        **layer,
        "coherence": coherence,
    }


def _pct(new: float, old: float) -> float:
    return ((new - old) / old) * 100.0 if old else 0.0


def _metric_delta(name: str, baseline: float, candidate: float, *, lower_is_better: bool) -> dict[str, Any]:
    delta = candidate - baseline
    pct = _pct(candidate, baseline)
    direction = "better" if (delta < 0 if lower_is_better else delta > 0) else "worse"
    if abs(delta) < 1e-9:
        direction = "same"
    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "pct": pct,
        "direction": direction,
        "lower_is_better": lower_is_better,
        "unit": "tok/s" if name == "best_tps" else "ms",
    }


def _delta_line(name: str, delta: dict[str, Any]) -> str:
    unit = "tok/s" if name == "best_tps" else "ms"
    return (
        f"- `{name}`: baseline `{delta['baseline']:.3f}` {unit}, "
        f"candidate `{delta['candidate']:.3f}` {unit}, "
        f"delta `{delta['delta']:+.3f}` ({delta['pct']:+.2f}%), {delta['direction']}"
    )


def _compare_result(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    baseline_dir = Path(args.baseline_log_dir)
    candidate_dir = Path(args.candidate_log_dir)
    baseline = _metrics(baseline_dir, args.max_repeat_fraction)
    candidate = _metrics(candidate_dir, args.max_repeat_fraction)
    failures: list[str] = []
    wins: list[str] = []
    metric_deltas: dict[str, dict[str, Any]] = {}

    tps_delta_pct = _pct(candidate["best_tps"], baseline["best_tps"])
    metric_deltas["best_tps"] = _metric_delta(
        "best_tps",
        baseline["best_tps"],
        candidate["best_tps"],
        lower_is_better=False,
    )
    if tps_delta_pct < -args.max_tps_regression_pct:
        failures.append(f"best_tps regressed by {tps_delta_pct:.2f}%")
    elif tps_delta_pct > args.min_tps_improvement_pct:
        wins.append(f"best_tps improved by {tps_delta_pct:.2f}%")

    for key in ("manual_decode_total_ms", "moe_ms", "mamba_ms", "attention_ms", "norm_lm_head_ms"):
        b = float(baseline.get(key, 0.0))
        c = float(candidate.get(key, 0.0))
        metric_deltas[key] = _metric_delta(key, b, c, lower_is_better=True)
        if not b or not c:
            failures.append(f"missing metric `{key}` in baseline or candidate")
            continue
        pct = _pct(c, b)
        if pct > args.max_ms_regression_pct:
            failures.append(f"{key} regressed by {pct:.2f}%")
        elif pct < -args.min_ms_improvement_pct:
            wins.append(f"{key} improved by {-pct:.2f}%")

    b_coh = baseline["coherence"]
    c_coh = candidate["coherence"]
    if c_coh.get("missing"):
        failures.append("candidate missing long coherence log")
    else:
        for key in ("leaks", "repeats", "no_eos"):
            if len(c_coh[key]) > len(b_coh[key]):
                failures.append(f"coherence `{key}` count regressed {len(b_coh[key])} -> {len(c_coh[key])}")
            elif len(c_coh[key]) < len(b_coh[key]):
                wins.append(f"coherence `{key}` count improved {len(b_coh[key])} -> {len(c_coh[key])}")

    status = "FAIL" if failures else ("IMPROVED" if wins else "UNCHANGED")
    exit_code = 1 if failures and args.strict else 0
    result = {
        "baseline_log_dir": str(baseline_dir),
        "candidate_log_dir": str(candidate_dir),
        "status": status,
        "wins": wins,
        "failures": failures,
        "metrics": {
            "baseline": baseline,
            "candidate": candidate,
            "deltas": metric_deltas,
        },
        "coherence_counts": {
            key: {
                "baseline": len(b_coh[key]),
                "candidate": len(c_coh[key]),
                "delta": len(c_coh[key]) - len(b_coh[key]),
            }
            for key in ("leaks", "repeats", "no_eos")
        },
        "thresholds": {
            "max_repeat_fraction": args.max_repeat_fraction,
            "max_tps_regression_pct": args.max_tps_regression_pct,
            "max_ms_regression_pct": args.max_ms_regression_pct,
            "min_tps_improvement_pct": args.min_tps_improvement_pct,
            "min_ms_improvement_pct": args.min_ms_improvement_pct,
        },
    }
    return exit_code, result


def _render_result(result: dict[str, Any]) -> str:
    deltas = result["metrics"]["deltas"]
    coherence_counts = result["coherence_counts"]

    lines = [
        "# Nemotron Ultra Runtime Speed Compare",
        "",
        f"baseline_log_dir: `{result['baseline_log_dir']}`",
        f"candidate_log_dir: `{result['candidate_log_dir']}`",
        f"status: `{result['status']}`",
        "",
        "## Metric Deltas",
        _delta_line("best_tps", deltas["best_tps"]),
        _delta_line("manual_decode_total_ms", deltas["manual_decode_total_ms"]),
        _delta_line("moe_ms", deltas["moe_ms"]),
        _delta_line("mamba_ms", deltas["mamba_ms"]),
        _delta_line("attention_ms", deltas["attention_ms"]),
        _delta_line("norm_lm_head_ms", deltas["norm_lm_head_ms"]),
        "",
        "## Coherence Counts",
    ]
    for key in ("leaks", "repeats", "no_eos"):
        counts = coherence_counts[key]
        lines.append(f"- `{key}`: baseline `{counts['baseline']}`, candidate `{counts['candidate']}`")
    lines.extend(["", "## Wins"])
    lines.extend(f"- {item}" for item in result["wins"])
    lines.extend(["", "## Failures"])
    lines.extend(f"- {item}" for item in result["failures"])
    return "\n".join(lines) + "\n"


def _render(args: argparse.Namespace) -> tuple[int, str]:
    exit_code, result = _compare_result(args)
    return exit_code, _render_result(result)


def _compare(args: argparse.Namespace) -> tuple[int, str, dict[str, Any]]:
    exit_code, result = _compare_result(args)
    return exit_code, _render_result(result), result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--candidate-log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--max-repeat-fraction", type=float, default=0.25)
    ap.add_argument("--max-tps-regression-pct", type=float, default=2.0)
    ap.add_argument("--max-ms-regression-pct", type=float, default=5.0)
    ap.add_argument("--min-tps-improvement-pct", type=float, default=2.0)
    ap.add_argument("--min-ms-improvement-pct", type=float, default=5.0)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    exit_code, report, result = _compare(args)
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
