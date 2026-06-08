"""No-load runtime speed gate for Nemotron Ultra JANGTQ_1L.

This is a regression/triage gate over saved probe logs. It does not load the
model. Use it after runtime changes to check whether token/s and known fixed
buckets still clear the current baseline, and to keep remaining bottlenecks
explicit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_LOG_DIR = Path("docs/runtime/logs")
DEFAULT_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-speed-gate.md")


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


def _layer(log_dir: Path) -> dict[str, Any] | None:
    for name in (
        "2026-06-04-nemotron-ultra-layer-decode-weighted-moe-probe.json",
        "2026-06-04-nemotron-ultra-layer-decode-current-probe.json",
    ):
        data = _load(log_dir / name)
        if data:
            data["_file"] = name
            return data
    return None


def _coherence(log_dir: Path) -> dict[str, Any] | None:
    return _load(log_dir / "2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json")


def _mamba(log_dir: Path) -> dict[str, float]:
    data = _load(log_dir / "2026-06-04-nemotron-ultra-mamba-component-probe.json")
    if not data or not data.get("layers"):
        return {}
    return {
        str(item["label"]): float(item["median_ms"])
        for item in data["layers"][0].get("timings", [])
        if isinstance(item.get("median_ms"), (int, float))
    }


def _gate_result(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    log_dir = Path(args.log_dir)
    best = _best_speed(log_dir)
    layer = _layer(log_dir)
    coherence = _coherence(log_dir)
    mamba = _mamba(log_dir)
    failures: list[str] = []
    partials: list[str] = []
    fixed: list[str] = []

    if best is None:
        failures.append("missing live speed probe")
    elif best[2] < args.min_live_tps:
        failures.append(f"best live speed {best[2]:.3f} tok/s is below floor {args.min_live_tps:.3f}")
    else:
        fixed.append(f"best live speed {best[2]:.3f} tok/s clears floor {args.min_live_tps:.3f}")

    if not layer:
        failures.append("missing layer decode probe")
        buckets = {}
        norm_lm = None
        total_ms = None
    else:
        buckets = layer.get("summary_by_block_type", {})
        norm_lm = layer.get("norm_lm_head_ms")
        total_ms = layer.get("manual_decode_total_ms")

    attention_ms = float(buckets.get("*", {}).get("total_ms", 0.0))
    moe_ms = float(buckets.get("E", {}).get("total_ms", 0.0))
    mamba_ms = float(buckets.get("M", {}).get("total_ms", 0.0))

    if attention_ms and attention_ms <= args.max_attention_ms:
        fixed.append(f"attention bucket {attention_ms:.3f} ms is below ceiling {args.max_attention_ms:.3f}")
    elif attention_ms:
        failures.append(f"attention bucket {attention_ms:.3f} ms exceeds ceiling {args.max_attention_ms:.3f}")

    if isinstance(norm_lm, (int, float)) and float(norm_lm) <= args.max_norm_lm_ms:
        fixed.append(f"norm/lm_head {float(norm_lm):.3f} ms is below ceiling {args.max_norm_lm_ms:.3f}")
    elif isinstance(norm_lm, (int, float)):
        failures.append(f"norm/lm_head {float(norm_lm):.3f} ms exceeds ceiling {args.max_norm_lm_ms:.3f}")

    if moe_ms >= args.min_bottleneck_ms:
        partials.append(f"MoE remains a bottleneck at {moe_ms:.3f} ms")
    if mamba_ms >= args.min_bottleneck_ms:
        partials.append(f"Mamba remains a bottleneck at {mamba_ms:.3f} ms")

    if coherence:
        leaks = []
        repeats = []
        no_eos = []
        for row in coherence.get("rows", []):
            if row.get("visible_marker_leaks"):
                leaks.append(str(row.get("id")))
            repeat = row.get("ngram_repeat", {}).get("repeat_fraction", 0.0)
            if isinstance(repeat, (int, float)) and repeat > args.max_repeat_fraction:
                repeats.append(str(row.get("id")))
            if not row.get("eos_reached"):
                no_eos.append(str(row.get("id")))
        if leaks or repeats or no_eos:
            partials.append(
                "coherence gate remains partial "
                f"(leaks={leaks}, repeats={repeats}, no_eos={no_eos})"
            )
        else:
            fixed.append("long coherence gate clears leak/repeat/EOS checks")
    else:
        failures.append("missing long coherence probe")

    if mamba:
        conv = mamba.get("conv", 0.0)
        ssm = mamba.get("ssm_update", 0.0)
        in_proj = mamba.get("in_proj", 0.0)
        out_proj = mamba.get("out_proj", 0.0)
        if conv and ssm and in_proj > conv and out_proj > ssm:
            fixed.append("Mamba component evidence points to projection/dispatch before conv rewrite")

    status = "BLOCKED" if failures else ("PARTIAL" if partials else "FIXED")
    exit_code = 2 if failures else (1 if partials and args.strict else 0)

    result = {
        "log_dir": str(log_dir),
        "status": status,
        "fixed": fixed,
        "partial": partials,
        "failures": failures,
        "metrics": {
            "best_live_tps": best[2] if best else None,
            "best_live_source": f"{best[0]}::{best[1]}" if best else None,
            "manual_decode_total_ms": float(total_ms) if isinstance(total_ms, (int, float)) else None,
            "moe_ms": moe_ms,
            "mamba_ms": mamba_ms,
            "attention_ms": attention_ms,
            "norm_lm_head_ms": float(norm_lm) if isinstance(norm_lm, (int, float)) else None,
        },
        "thresholds": {
            "min_live_tps": args.min_live_tps,
            "max_attention_ms": args.max_attention_ms,
            "max_norm_lm_ms": args.max_norm_lm_ms,
            "min_bottleneck_ms": args.min_bottleneck_ms,
            "max_repeat_fraction": args.max_repeat_fraction,
        },
    }
    return exit_code, result


def _render_result(result: dict[str, Any]) -> str:
    metrics = result["metrics"]

    lines = [
        "# Nemotron Ultra Runtime Speed Gate",
        "",
        f"log_dir: `{result['log_dir']}`",
        f"status: `{result['status']}`",
        "",
        "## Fixed Evidence",
    ]
    lines.extend(f"- {item}" for item in result["fixed"])
    lines.append("")
    lines.append("## Partial Evidence")
    lines.extend(f"- {item}" for item in result["partial"])
    lines.append("")
    lines.append("## Failures")
    lines.extend(f"- {item}" for item in result["failures"])
    lines.append("")
    lines.append("## Current Buckets")
    if metrics["best_live_tps"] is not None:
        lines.append(f"- best_live: `{metrics['best_live_tps']:.3f} tok/s` from `{metrics['best_live_source']}`")
    if metrics["manual_decode_total_ms"] is not None:
        lines.append(f"- manual_decode_total_ms: `{metrics['manual_decode_total_ms']:.3f}`")
    lines.append(f"- moe_ms: `{metrics['moe_ms']:.3f}`")
    lines.append(f"- mamba_ms: `{metrics['mamba_ms']:.3f}`")
    lines.append(f"- attention_ms: `{metrics['attention_ms']:.3f}`")
    if metrics["norm_lm_head_ms"] is not None:
        lines.append(f"- norm_lm_head_ms: `{metrics['norm_lm_head_ms']:.3f}`")
    return "\n".join(lines) + "\n"


def _gate(args: argparse.Namespace) -> tuple[int, str]:
    exit_code, result = _gate_result(args)
    return exit_code, _render_result(result)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--min-live-tps", type=float, default=8.0)
    ap.add_argument("--max-attention-ms", type=float, default=10.0)
    ap.add_argument("--max-norm-lm-ms", type=float, default=5.0)
    ap.add_argument("--min-bottleneck-ms", type=float, default=40.0)
    ap.add_argument("--max-repeat-fraction", type=float, default=0.25)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    exit_code, result = _gate_result(args)
    report = _render_result(result)
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
