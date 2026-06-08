"""Build a no-load Nemotron Ultra speed experiment plan from probe logs.

The output is intentionally operational: it ranks the next runtime experiments
by measured bucket size and simple token/s sensitivity, while also listing
negative controls that current evidence says not to chase.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_LOG_DIR = Path("docs/runtime/logs")
DEFAULT_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-speed-experiment-plan.md")


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


def _moe_log(log_dir: Path) -> dict[str, Any]:
    for name in (
        "2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json",
        "2026-06-04-nemotron-ultra-moe-component-current-probe.json",
        "2026-06-04-nemotron-ultra-moe-component-probe.json",
    ):
        data = _load(log_dir / name)
        if data:
            data["_file"] = name
            return data
    return {}


def _mamba_log(log_dir: Path) -> dict[str, Any]:
    data = _load(log_dir / "2026-06-04-nemotron-ultra-mamba-component-probe.json")
    if data:
        data["_file"] = "2026-06-04-nemotron-ultra-mamba-component-probe.json"
        return data
    return {}


def _projection_log(log_dir: Path) -> dict[str, Any]:
    data = _load(log_dir / "2026-06-04-nemotron-ultra-projection-tradeoff-probe.json")
    return data or {}


def _live_best_speed(log_dir: Path) -> float | None:
    best: float | None = None
    for path in sorted(log_dir.glob("*live-probe.json")):
        data = _load(path)
        if not data:
            continue
        for row in data.get("rows", []):
            value = row.get("decode_tps_excluding_first")
            if isinstance(value, (int, float)):
                best = float(value) if best is None else max(best, float(value))
    return best


def _moe_components(data: dict[str, Any]) -> dict[str, float]:
    layers = data.get("layers", [])
    if not layers:
        return {}
    timings = layers[0].get("timings", [])
    return {
        str(item["label"]): float(item["median_ms"])
        for item in timings
        if isinstance(item.get("median_ms"), (int, float))
    }


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


def _tps_after_ms_reduction(total_ms: float, reduction_ms: float) -> float:
    new_ms = max(total_ms - reduction_ms, 1e-9)
    return 1000.0 / new_ms


def _render(log_dir: Path) -> str:
    layer = _layer_log(log_dir)
    moe = _moe_log(log_dir)
    mamba = _mamba_log(log_dir)
    projection = _projection_log(log_dir)
    live_best = _live_best_speed(log_dir)

    total_ms = float(layer.get("manual_decode_total_ms") or 0.0)
    buckets = layer.get("summary_by_block_type", {})
    norm_lm = float(layer.get("norm_lm_head_ms") or 0.0)
    moe_total = float(buckets.get("E", {}).get("total_ms", 0.0))
    mamba_total = float(buckets.get("M", {}).get("total_ms", 0.0))
    attn_total = float(buckets.get("*", {}).get("total_ms", 0.0))
    component = _moe_components(moe)
    mamba_component = _component_medians(mamba)
    base_tps_from_manual = 1000.0 / total_ms if total_ms > 0 else None

    lines: list[str] = []
    lines.append("# Nemotron Ultra Speed Experiment Plan")
    lines.append("")
    lines.append(f"log_dir: `{log_dir}`")
    if layer:
        lines.append(f"layer_log: `{layer.get('_file')}`")
    if moe:
        lines.append(f"moe_log: `{moe.get('_file')}`")
    if mamba:
        lines.append(f"mamba_log: `{mamba.get('_file')}`")
    lines.append("")
    lines.append("## Current Bottleneck")
    if total_ms > 0:
        lines.append(f"- manual synchronized decode: `{total_ms:.3f} ms/token`")
        lines.append(f"- implied synchronized throughput: `{base_tps_from_manual:.3f} tok/s`")
    if live_best:
        lines.append(f"- best live generator row: `{live_best:.3f} tok/s`")
    lines.append(f"- MoE total: `{moe_total:.3f} ms` across 48 layers")
    lines.append(f"- Mamba total: `{mamba_total:.3f} ms` across 48 layers")
    lines.append(f"- attention total: `{attn_total:.3f} ms` across 12 layers")
    lines.append(f"- final norm/lm_head: `{norm_lm:.3f} ms`")
    lines.append("")

    lines.append("## Ranked Experiments")
    lines.append("| Rank | Experiment | Evidence | Target | Sensitivity |")
    lines.append("| --- | --- | --- | --- | --- |")
    if total_ms > 0:
        moe_10 = _tps_after_ms_reduction(total_ms, moe_total * 0.10)
        mamba_10 = _tps_after_ms_reduction(total_ms, mamba_total * 0.10)
        both_10 = _tps_after_ms_reduction(total_ms, (moe_total + mamba_total) * 0.10)
    else:
        moe_10 = mamba_10 = both_10 = 0.0
    switch_ms = component.get("switch_mlp", 0.0)
    shared_ms = component.get("shared_experts", 0.0)
    mamba_in_ms = mamba_component.get("in_proj", 0.0)
    mamba_out_ms = mamba_component.get("out_proj", 0.0)
    mamba_conv_ms = mamba_component.get("conv", 0.0)
    mamba_ssm_ms = mamba_component.get("ssm_update", 0.0)
    lines.append(
        "| 1 | MoE routed/shared scheduling or fused decode kernel | "
        f"`E={moe_total:.3f} ms`; single-layer `switch_mlp={switch_ms:.3f} ms`, "
        f"`shared_experts={shared_ms:.3f} ms` | reduce fixed per-layer MoE overhead | "
        f"10% MoE cut implies `{moe_10:.3f} tok/s` synchronized |"
    )
    lines.append(
        "| 2 | Mamba fused decode kernel / lower-overhead state update | "
        f"`M={mamba_total:.3f} ms`; first-layer `in_proj={mamba_in_ms:.3f} ms`, "
        f"`out_proj={mamba_out_ms:.3f} ms`, `conv={mamba_conv_ms:.3f} ms`, "
        f"`ssm={mamba_ssm_ms:.3f} ms` | reduce projection/dispatch overhead first | "
        f"10% Mamba cut implies `{mamba_10:.3f} tok/s` synchronized |"
    )
    lines.append(
        "| 3 | Joint MoE+Mamba scheduling path | "
        f"`E+M={moe_total + mamba_total:.3f} ms` dominates decode | attack Python/MLX dispatch and sync boundaries | "
        f"10% combined cut implies `{both_10:.3f} tok/s` synchronized |"
    )
    lines.append(
        "| 4 | Ahead-of-time warmup plan | cold JIT is about 33s, warmed TTFT about 1s | make startup predictable, not steady decode faster | improves TTFT, not tok/s |"
    )
    lines.append("")

    lines.append("## Negative Controls")
    lines.append("- Do not chase attention first: it is the smallest bucket after BF16 retention.")
    lines.append("- Do not dequantize 8-bit Mamba/shared projections for speed; current probe says quantized is faster.")
    lines.append("- Do not lower router top-k as the main fix; top-k 8 did not materially improve live decode.")
    lines.append("- Do not replace `mlx_lm.generate_step` with a Python argmax loop; manual loop was slower.")
    lines.append("- Do not hide parser/coherence problems with prompt suffixes, forced closing tags, or sampler tricks.")
    lines.append("")

    if projection.get("results"):
        lines.append("## Projection Evidence")
        for row in projection["results"]:
            lines.append(
                "- `{name}`: quantized `{q:.3f} ms`, BF16 dequantized `{b:.3f} ms`, quantized speedup `{s:.2f}x`".format(
                    name=row.get("name"),
                    q=float(row.get("quantized", {}).get("median_ms", 0.0)),
                    b=float(row.get("bf16_dequantized", {}).get("median_ms", 0.0)),
                    s=float(row.get("quantized_speedup_vs_bf16", 0.0)),
                )
            )
        lines.append("")

    if mamba_component:
        lines.append("## Mamba Component Evidence")
        for label in (
            "outer_norm",
            "in_proj",
            "conv",
            "ssm_update",
            "mamba_norm_gated",
            "out_proj",
            "full_mamba_mixer",
        ):
            if label in mamba_component:
                lines.append(f"- `{label}`: `{mamba_component[label]:.3f} ms`")
        lines.append(
            "Interpretation: the generic grouped conv and SSM update are not the largest "
            "isolated Mamba substeps in this probe; projection/dispatch fusion is the "
            "more credible first Mamba speed target."
        )
        lines.append("")

    lines.append("## Next Proof Rows")
    lines.append("- rerun layer decode after any MoE or Mamba runtime change")
    lines.append("- rerun live speed probe after warm compile")
    lines.append("- rerun long coherence probe; speed wins cannot regress parser/coherence")
    lines.append("- keep cache/VL proof separate from speed proof")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    text = _render(args.log_dir)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
