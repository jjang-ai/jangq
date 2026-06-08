"""Summarize DSV4 Flash JANG quant matrix artifacts.

This is intentionally read-only. It turns matrix JSON from live vMLX probes
into a compact decision table so the next quant build is selected by measured
quality/speed deltas rather than broad trial-and-error.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Row:
    name: str
    profile: str
    du: str
    speed_tok_s: float | None
    peak_mb: float | None
    exact_count: int
    expect_count: int
    basic_count: int
    garbage_count: int
    quality_score: float
    plan: str
    math_text: str
    marker_text: str


def _load_json(path: Path) -> object:
    data = json.loads(path.read_text())
    if isinstance(data, str):
        data = json.loads(data)
    return data


def _entries_from_path(path: Path) -> list[dict]:
    data = _load_json(path)
    if isinstance(data, dict):
        if "entries" in data and isinstance(data["entries"], list):
            return [x for x in data["entries"] if isinstance(x, dict)]
        if "static" in data and "live" in data:
            return [data]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    raise ValueError(f"unsupported matrix JSON shape: {path}")


def _case_quality(case: dict) -> dict:
    quality = case.get("quality")
    return quality if isinstance(quality, dict) else {}


def _is_garbage(quality: dict) -> bool:
    return bool(
        quality.get("garbage")
        or int(quality.get("replacement_chars") or 0) > 0
        or int(quality.get("combining_marks") or 0) > 0
        or quality.get("raw_think_leak")
        or quality.get("role_leak")
    )


def _score_case(quality: dict) -> float:
    if _is_garbage(quality):
        return -2.0
    if quality.get("exact") is True:
        return 2.0
    if quality.get("expect_present") is True:
        return 1.0
    if quality.get("pass_basic") is True:
        return 0.5
    return 0.0


def _hot_speed(live: dict) -> float | None:
    for item in live.get("speed") or []:
        if item.get("name") == "speed_512_hot":
            value = item.get("wall_tok_s")
            return float(value) if value is not None else None
    return None


def _peak_mb(live: dict) -> float | None:
    for key in ("initial_health", "final_health"):
        health = live.get(key) or {}
        memory = health.get("memory") or {}
        value = memory.get("peak_mb") or memory.get("peak_memory_mb")
        if value is not None:
            return float(value)
    return None


def _plan(static: dict) -> str:
    cfg = static.get("config") or {}
    quant = cfg.get("quantization") or {}
    bookend_bits = quant.get("bookend_bits", quant.get("bits", "?"))
    bookend_group = quant.get("bookend_group_size", quant.get("group_size", "?"))
    token_bits = quant.get("token_bookend_bits", bookend_bits)
    token_group = quant.get("token_bookend_group_size", bookend_group)
    bit_plan = quant.get("routed_expert_bit_plan") or cfg.get("routed_expert_bit_plan") or {}
    routed_groups = bit_plan.get("routed_projection_group_sizes") or {}
    if routed_groups:
        group_text = ",".join(
            f"{proj}:g{routed_groups[proj]}" for proj in sorted(routed_groups)
        )
    else:
        group = quant.get("routed_expert_group_size") or cfg.get("routed_expert_group_size") or "?"
        group_text = f"default:g{group}"
    projection_bits = bit_plan.get("routed_projection_bits") or {}
    layer_bits = bit_plan.get("routed_projection_layer_bits") or {}
    extra = []
    if projection_bits:
        extra.append(
            "proj_bits="
            + ",".join(f"{k}:{projection_bits[k]}" for k in sorted(projection_bits))
        )
    if layer_bits:
        extra.append("selected_layer_bits=yes")
    suffix = "; " + "; ".join(extra) if extra else ""
    return (
        f"bookend={bookend_bits}b_g{bookend_group}; "
        f"token={token_bits}b_g{token_group}; "
        f"routed_gs={group_text}{suffix}"
    )


def summarize_entry(entry: dict) -> Row:
    static = entry.get("static") or {}
    live = entry.get("live") or {}
    cases = live.get("cases") or []
    exact_count = 0
    expect_count = 0
    basic_count = 0
    garbage_count = 0
    quality_score = 0.0
    text_by_case: dict[str, str] = {}
    for case in cases:
        if not isinstance(case, dict):
            continue
        name = str(case.get("name") or "")
        text_by_case[name] = str(case.get("text") or case.get("text_head") or "")
        quality = _case_quality(case)
        if quality.get("exact") is True:
            exact_count += 1
        if quality.get("expect_present") is True:
            expect_count += 1
        if quality.get("pass_basic") is True:
            basic_count += 1
        if _is_garbage(quality):
            garbage_count += 1
        quality_score += _score_case(quality)

    cfg = static.get("config") or {}
    profile = (
        (static.get("jang_config") or {}).get("profile")
        or cfg.get("_name_or_path")
        or "unknown"
    )
    return Row(
        name=str(static.get("name") or Path(str(static.get("path") or "")).name),
        profile=str(profile),
        du=str(static.get("du") or ""),
        speed_tok_s=_hot_speed(live),
        peak_mb=_peak_mb(live),
        exact_count=exact_count,
        expect_count=expect_count,
        basic_count=basic_count,
        garbage_count=garbage_count,
        quality_score=quality_score,
        plan=_plan(static),
        math_text=text_by_case.get("math_17_28", ""),
        marker_text=text_by_case.get("marker_single", ""),
    )


def summarize_paths(paths: Iterable[Path]) -> list[Row]:
    rows: list[Row] = []
    for path in paths:
        for entry in _entries_from_path(Path(path)):
            rows.append(summarize_entry(entry))
    return rows


def _fmt_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _find(rows: list[Row], needle: str) -> Row | None:
    for row in rows:
        if needle in row.name:
            return row
    return None


def _delta_line(label: str, before: Row, after: Row) -> str:
    speed_delta = (
        None
        if before.speed_tok_s is None or after.speed_tok_s is None
        else after.speed_tok_s - before.speed_tok_s
    )
    peak_delta = (
        None
        if before.peak_mb is None or after.peak_mb is None
        else after.peak_mb - before.peak_mb
    )
    return (
        f"| {label} | {_fmt_float(speed_delta)} | {_fmt_float(peak_delta, 1)} | "
        f"{after.exact_count - before.exact_count:+d} | "
        f"{after.expect_count - before.expect_count:+d} | "
        f"{after.garbage_count - before.garbage_count:+d} | "
        f"{after.quality_score - before.quality_score:+.2f} |"
    )


def _lever_deltas(rows: list[Row]) -> list[str]:
    pairs = [
        ("DQ2 -> Token8", "DQ2-Prestack", "DQ2-Token8-Prestack"),
        ("Token8 -> Bookend8", "DQ2-Token8-Prestack", "DQ2-Bookend8-Prestack"),
        ("Token8 -> DownG32", "DQ2-Token8", "DQ2-Token8-DownG32"),
        (
            "DownG32 -> Down3L7-11",
            "DQ2-Token8-DownG32-Prestack",
            "DQ2-Token8-DownG32-Down3L7-11",
        ),
        ("DownG32 -> G32All", "DQ2-Token8-DownG32", "DQ2-Token8-G32All"),
        ("DownG32 -> K", "DQ2-Token8-DownG32", "JANG_K-MTP"),
    ]
    lines = []
    for label, before_needle, after_needle in pairs:
        before = _find(rows, before_needle)
        after = _find(rows, after_needle)
        if before is None or after is None or before is after:
            continue
        lines.append(_delta_line(label, before, after))
    return lines


def _decision(row: Row) -> str:
    if row.speed_tok_s is not None and row.speed_tok_s < 15:
        return "reject-speed"
    if row.garbage_count > 0:
        return "reject-garbage"
    if (
        row.speed_tok_s is not None
        and row.speed_tok_s >= 22
        and row.expect_count >= 5
        and row.quality_score >= 8
    ):
        return "active-candidate"
    if row.speed_tok_s is not None and row.speed_tok_s >= 22:
        if row.quality_score < 5:
            return "reject-quality"
        return "speed-candidate"
    return "reference-only"


def render_markdown(rows: list[Row], title: str = "DSV4 Quant Matrix Analysis") -> str:
    rows = sorted(
        rows,
        key=lambda row: (
            row.speed_tok_s or 0.0,
            row.quality_score,
            -(row.peak_mb or 0.0),
        ),
        reverse=True,
    )
    out = [f"# {title}", ""]
    out.append("## Ranked Variants")
    out.append("")
    out.append(
        "| Model | du | speed tok/s | peak MB | exact | expect | garbage | score | decision |"
    )
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        out.append(
            f"| {row.name} | {row.du} | {_fmt_float(row.speed_tok_s)} | "
            f"{_fmt_float(row.peak_mb, 1)} | {row.exact_count} | "
            f"{row.expect_count} | {row.garbage_count} | "
            f"{row.quality_score:.2f} | {_decision(row)} |"
        )

    out.extend(["", "## Lever Deltas", ""])
    deltas = _lever_deltas(rows)
    if deltas:
        out.append("| Change | speed delta | peak MB delta | exact delta | expect delta | garbage delta | score delta |")
        out.append("|---|---:|---:|---:|---:|---:|---:|")
        out.extend(deltas)
    else:
        out.append("No recognized before/after lever pairs found.")

    out.extend(["", "## Plans", ""])
    for row in rows:
        out.append(f"- `{row.name}`: {row.plan}")

    out.extend(["", "## Residual Failures", ""])
    for row in rows:
        if row.math_text or row.marker_text:
            out.append(
                f"- `{row.name}`: math=`{row.math_text[:80]}`; "
                f"marker=`{row.marker_text[:80]}`"
            )
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("matrix_json", nargs="+", type=Path)
    parser.add_argument("--title", default="DSV4 Quant Matrix Analysis")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    markdown = render_markdown(summarize_paths(args.matrix_json), title=args.title)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown)
    else:
        print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
