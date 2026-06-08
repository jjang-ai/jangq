"""Build a no-load Nemotron Ultra runtime handoff for downstream agents.

This script does not load model weights. It reads the JANGTQ bundle metadata and
saved probe logs, then emits a compact markdown and optional JSON handoff with
the current speed gate, bottlenecks, parser/cache/VL boundaries, runtime
toggles, and next proof rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from examples.nemotron_ultra.runtime_speed_gate import _gate_result
from examples.nemotron_ultra.token_speed_budget import _build_result as _speed_budget_result


DEFAULT_BUNDLE = Path("/Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L")
DEFAULT_LOG_DIR = Path("docs/runtime/logs")
DEFAULT_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-agent-handoff.md")
DEFAULT_JSON_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-agent-handoff.json")


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _layer_counts(config: dict[str, Any] | None) -> dict[str, int]:
    counts = {"mamba": 0, "moe": 0, "attention": 0}
    if not config:
        return counts
    for item in config.get("layers_block_type", []):
        name = str(item).lower()
        if name == "mamba":
            counts["mamba"] += 1
        elif name == "moe":
            counts["moe"] += 1
        elif name == "attention":
            counts["attention"] += 1
    return counts


def _component_timings(log_dir: Path, name: str) -> dict[str, float]:
    data = _load(log_dir / name)
    if not data or not data.get("layers"):
        return {}
    return {
        str(item["label"]): float(item["median_ms"])
        for item in data["layers"][0].get("timings", [])
        if isinstance(item.get("median_ms"), (int, float))
    }


def _parser_summary(log_dir: Path) -> dict[str, Any]:
    data = _load(log_dir / "2026-06-04-nemotron-ultra-jangtq1l-parser-probe.json")
    if not data:
        return {"status": "MISSING", "rows": 0, "marker_leak_rows": [], "truncated_reasoning_rows": [], "tool_rows": 0}
    marker_leak_rows = []
    truncated_reasoning_rows = []
    tool_rows = 0
    for row in data.get("rows", []):
        row_id = str(row.get("id"))
        if row.get("visible_think_marker_leaks"):
            marker_leak_rows.append(row_id)
        if row.get("truncated_reasoning"):
            truncated_reasoning_rows.append(row_id)
        if row.get("tool_calls"):
            tool_rows += 1
    status = "PARTIAL" if marker_leak_rows or truncated_reasoning_rows or tool_rows == 0 else "FIXED"
    return {
        "status": status,
        "parser": data.get("parser"),
        "rows": len(data.get("rows", [])),
        "marker_leak_rows": marker_leak_rows,
        "truncated_reasoning_rows": truncated_reasoning_rows,
        "tool_rows": tool_rows,
    }


def _coherence_summary(log_dir: Path, max_repeat_fraction: float) -> dict[str, Any]:
    data = _load(log_dir / "2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json")
    if not data:
        return {"status": "MISSING", "rows": 0, "leaks": [], "repeats": [], "no_eos": []}
    leaks = []
    repeats = []
    no_eos = []
    expected_missing = []
    for row in data.get("rows", []):
        row_id = str(row.get("id"))
        if row.get("visible_marker_leaks"):
            leaks.append(row_id)
        repeat = row.get("ngram_repeat", {}).get("repeat_fraction", 0.0)
        if isinstance(repeat, (int, float)) and repeat > max_repeat_fraction:
            repeats.append(row_id)
        if not row.get("eos_reached"):
            no_eos.append(row_id)
        expected = row.get("expected_found", {})
        if expected and not all(bool(v) for v in expected.values()):
            expected_missing.append(row_id)
    status = "PARTIAL" if leaks or repeats or no_eos or expected_missing else "FIXED"
    return {
        "status": status,
        "rows": len(data.get("rows", [])),
        "leaks": leaks,
        "repeats": repeats,
        "no_eos": no_eos,
        "expected_missing": expected_missing,
    }


def _gate_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        log_dir=args.log_dir,
        min_live_tps=args.min_live_tps,
        max_attention_ms=args.max_attention_ms,
        max_norm_lm_ms=args.max_norm_lm_ms,
        min_bottleneck_ms=args.min_bottleneck_ms,
        max_repeat_fraction=args.max_repeat_fraction,
        strict=False,
        out=None,
    )


def _build_result(args: argparse.Namespace) -> dict[str, Any]:
    bundle = Path(args.bundle)
    log_dir = Path(args.log_dir)
    jang_config = _load(bundle / "jang_config.json") or {}
    model_config = _load(bundle / "config.json") or {}
    layer_counts = _layer_counts(model_config)
    gate_exit, gate = _gate_result(_gate_args(args))
    metrics = gate["metrics"]
    mamba_components = _component_timings(log_dir, "2026-06-04-nemotron-ultra-mamba-component-probe.json")
    moe_components = _component_timings(log_dir, "2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json")
    parser = _parser_summary(log_dir)
    coherence = _coherence_summary(log_dir, args.max_repeat_fraction)
    speed_budget = _speed_budget_result(log_dir, args.speed_targets)
    capabilities = jang_config.get("capabilities", {})
    quant = jang_config.get("quantization", {})
    runtime = jang_config.get("runtime", {})

    next_experiments = []
    if metrics.get("moe_ms", 0.0) >= args.min_bottleneck_ms:
        next_experiments.append("MoE routed/shared scheduling or fused decode kernel")
    if metrics.get("mamba_ms", 0.0) >= args.min_bottleneck_ms:
        next_experiments.append("Mamba projection/dispatch fusion or fused decode state update")
    if next_experiments:
        next_experiments.append("Joint MoE+Mamba dispatch-boundary reduction")
    next_experiments.extend(
        [
            "rerun layer decode and live speed after any runtime change",
            "rerun long coherence; speed wins must not regress parser-visible output",
        ]
    )

    return {
        "bundle": str(bundle),
        "log_dir": str(log_dir),
        "handoff_status": "PARTIAL" if gate["status"] != "FIXED" or parser["status"] != "FIXED" else "FIXED",
        "speed_gate_exit_code": gate_exit,
        "speed_gate": gate,
        "artifact": {
            "profile": jang_config.get("profile"),
            "format": jang_config.get("format"),
            "format_version": jang_config.get("format_version"),
            "source_model": jang_config.get("source_model", {}),
            "capabilities": capabilities,
            "mxtq_bits": jang_config.get("mxtq_bits", {}),
            "estimated_output_gib": quant.get("estimated_output_gib"),
            "shard_count": runtime.get("shard_count"),
            "total_shard_bytes": runtime.get("total_shard_bytes"),
            "drops_mtp": quant.get("drops_mtp"),
        },
        "topology": {
            "layers_total": sum(layer_counts.values()) or model_config.get("num_hidden_layers"),
            "mamba_layers": layer_counts["mamba"] or 48,
            "moe_layers": layer_counts["moe"] or 48,
            "attention_layers": layer_counts["attention"] or 12,
            "cache_entries": (layer_counts["mamba"] or 48) + (layer_counts["attention"] or 12),
            "attention_kv_cache_entries": layer_counts["attention"] or 12,
            "mamba_companion_state_entries": layer_counts["mamba"] or 48,
        },
        "component_timings_ms": {
            "mamba": mamba_components,
            "moe": moe_components,
        },
        "speed_budget": speed_budget,
        "parser": parser,
        "coherence": coherence,
        "runtime_controls": {
            "disable_switchmlp_fastpath": "JANGTQ_DISABLE_NEMOTRON_SWITCHMLP_FASTPATH=1",
            "legacy_disable_switchmlp_fastpath": "JANGTQ_ENABLE_NEMOTRON_SWITCHMLP_FASTPATH=0",
            "disable_activation_bf16": "JANGTQ_DISABLE_NEMOTRON_ACTIVATION_BF16=1",
            "disable_weighted_moe_fastpath": "JANGTQ_DISABLE_NEMOTRON_WEIGHTED_MOE_FASTPATH=1",
        },
        "negative_controls": [
            "Do not chase attention first while it remains under the current ceiling.",
            "Do not dequantize 8-bit affine projections as a speed fix without new proof.",
            "Do not lower router top-k as the main fix; saved top-k probe did not materially improve decode.",
            "Do not hide parser/coherence failures with prompt suffixes or sampler tricks.",
            "Do not enable speculative/MTP decode for this MTP-dropped bundle.",
        ],
        "cache_and_modality_gates": {
            "cache_type": capabilities.get("cache_type"),
            "text_only": capabilities.get("modality") == "text",
            "kv_cache_boundary": "TurboQuant KV applies only to attention KV entries.",
            "mamba_state_boundary": "Prefix hits require matching Mamba companion states.",
            "vl_policy": "Reject or reroute media requests; this artifact has no VL/audio tensors or processor configs.",
            "mtp_policy": "Disabled for this bundle; draft KV/SSM state is out of scope.",
        },
        "next_experiments": next_experiments,
    }


def _render(result: dict[str, Any]) -> str:
    artifact = result["artifact"]
    topology = result["topology"]
    gate = result["speed_gate"]
    metrics = gate["metrics"]
    parser = result["parser"]
    coherence = result["coherence"]

    lines = [
        "# Nemotron Ultra Agent Runtime Handoff",
        "",
        f"bundle: `{result['bundle']}`",
        f"log_dir: `{result['log_dir']}`",
        f"handoff_status: `{result['handoff_status']}`",
        f"speed_gate: `{gate['status']}`",
        "",
        "## Artifact",
        f"- profile: `{artifact.get('profile')}`",
        f"- format: `{artifact.get('format')}` `{artifact.get('format_version')}`",
        f"- estimated_output_gib: `{artifact.get('estimated_output_gib')}`",
        f"- shard_count: `{artifact.get('shard_count')}`",
        f"- drops_mtp: `{artifact.get('drops_mtp')}`",
        f"- capabilities: `{json.dumps(artifact.get('capabilities', {}), sort_keys=True)}`",
        f"- mxtq_bits: `{json.dumps(artifact.get('mxtq_bits', {}), sort_keys=True)}`",
        "",
        "## Topology",
        f"- layers_total: `{topology['layers_total']}`",
        f"- mamba/moe/attention: `{topology['mamba_layers']}` / `{topology['moe_layers']}` / `{topology['attention_layers']}`",
        f"- cache_entries: `{topology['cache_entries']}` = `{topology['mamba_companion_state_entries']}` Mamba companion states + `{topology['attention_kv_cache_entries']}` attention KV entries",
        "",
        "## Current Speed Buckets",
    ]
    if metrics.get("best_live_tps") is not None:
        lines.append(f"- best_live_tps: `{metrics['best_live_tps']:.3f}` from `{metrics['best_live_source']}`")
    if metrics.get("manual_decode_total_ms") is not None:
        lines.append(f"- manual_decode_total_ms: `{metrics['manual_decode_total_ms']:.3f}`")
    for key in ("moe_ms", "mamba_ms", "attention_ms", "norm_lm_head_ms"):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            lines.append(f"- {key}: `{value:.3f}`")
    lines.extend(["", "## Fixed Evidence"])
    lines.extend(f"- {item}" for item in gate["fixed"])
    lines.extend(["", "## Partial Evidence"])
    lines.extend(f"- {item}" for item in gate["partial"])
    lines.extend(["", "## Parser And Coherence"])
    lines.append(f"- parser_status: `{parser['status']}` marker_leak_rows={parser['marker_leak_rows']} truncated_reasoning_rows={parser['truncated_reasoning_rows']} tool_rows={parser['tool_rows']}")
    lines.append(f"- long_coherence_status: `{coherence['status']}` leaks={coherence['leaks']} repeats={coherence['repeats']} no_eos={coherence['no_eos']}")
    lines.extend(["", "## Token Speed Budgets"])
    for row in result["speed_budget"]["targets"]:
        lines.append(
            "- target `{target_tps:.3f}` tok/s: cut `{required_total_cut_ms:.3f}` ms total; "
            "proportional MoE `{moe_cut_ms_proportional:.3f}` ms, Mamba `{mamba_cut_ms_proportional:.3f}` ms".format(
                **row
            )
        )
    lines.extend(["", "## Runtime Controls"])
    for name, value in result["runtime_controls"].items():
        lines.append(f"- {name}: `{value}`")
    lines.extend(["", "## Cache And Modality Gates"])
    for name, value in result["cache_and_modality_gates"].items():
        lines.append(f"- {name}: `{value}`")
    lines.extend(["", "## Next Experiments"])
    lines.extend(f"- {item}" for item in result["next_experiments"])
    lines.extend(["", "## Negative Controls"])
    lines.extend(f"- {item}" for item in result["negative_controls"])
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--min-live-tps", type=float, default=8.0)
    ap.add_argument("--max-attention-ms", type=float, default=10.0)
    ap.add_argument("--max-norm-lm-ms", type=float, default=5.0)
    ap.add_argument("--min-bottleneck-ms", type=float, default=40.0)
    ap.add_argument("--max-repeat-fraction", type=float, default=0.25)
    ap.add_argument(
        "--speed-targets",
        type=lambda raw: [float(item.strip()) for item in raw.split(",") if item.strip()],
        default=[10.0, 12.0, 15.0],
    )
    args = ap.parse_args()

    result = _build_result(args)
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
