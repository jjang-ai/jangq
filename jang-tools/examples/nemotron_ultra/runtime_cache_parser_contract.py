"""No-load cache/parser contract for Nemotron Ultra speed candidates.

Speed candidates must not regress reasoning parsing, tool-call parsing, hybrid
prefix-cache topology, Mamba companion states, MTP-disabled behavior, or
text-only modality gates. This helper records those invariants from saved logs
and bundle metadata without loading the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_LOG_DIR = Path("docs/runtime/logs")
DEFAULT_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-cache-parser-contract.md")
DEFAULT_JSON_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-cache-parser-contract.json")


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _parser_probe_summary(path: Path) -> dict[str, Any]:
    data = _load(path)
    if data is None:
        return {
            "status": "MISSING",
            "parser": None,
            "rows": 0,
            "marker_leak_rows": [],
            "truncated_reasoning_rows": [],
            "tool_rows": 0,
        }
    marker_leak_rows: list[str] = []
    truncated_reasoning_rows: list[str] = []
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


def _long_coherence_summary(path: Path) -> dict[str, Any]:
    data = _load(path)
    if data is None:
        return {"status": "MISSING", "rows": 0, "leak_rows": [], "repeat_rows": [], "no_eos_rows": []}
    leak_rows: list[str] = []
    repeat_rows: list[str] = []
    no_eos_rows: list[str] = []
    for row in data.get("rows", []):
        row_id = str(row.get("id"))
        if row.get("visible_marker_leaks"):
            leak_rows.append(row_id)
        repeat_fraction = row.get("ngram_repeat", {}).get("repeat_fraction", 0.0)
        if isinstance(repeat_fraction, (int, float)) and repeat_fraction > 0.25:
            repeat_rows.append(row_id)
        if not row.get("eos_reached"):
            no_eos_rows.append(row_id)
    status = "PARTIAL" if leak_rows or repeat_rows or no_eos_rows else "FIXED"
    return {
        "status": status,
        "rows": len(data.get("rows", [])),
        "leak_rows": leak_rows,
        "repeat_rows": repeat_rows,
        "no_eos_rows": no_eos_rows,
    }


def _build_result(log_dir: Path) -> dict[str, Any]:
    handoff_path = log_dir / "2026-06-04-nemotron-ultra-agent-handoff.json"
    shape_path = log_dir / "2026-06-04-nemotron-ultra-runtime-shape-contract.json"
    parser_path = log_dir / "2026-06-04-nemotron-ultra-jangtq1l-parser-probe.json"
    long_path = log_dir / "2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json"
    speed_acceptance_path = log_dir / "2026-06-04-nemotron-ultra-runtime-speed-fix-acceptance.json"
    candidate_index_path = log_dir / "2026-06-04-nemotron-ultra-runtime-candidate-index.json"

    handoff = _load(handoff_path)
    shape = _load(shape_path)
    speed_acceptance = _load(speed_acceptance_path) or {}
    candidate_index = _load(candidate_index_path) or {}
    parser_probe = _parser_probe_summary(parser_path)
    long_coherence = _long_coherence_summary(long_path)

    failures: list[str] = []
    if handoff is None:
        failures.append(f"missing agent handoff: {handoff_path}")
    if shape is None:
        failures.append(f"missing shape contract: {shape_path}")

    artifact = (handoff or {}).get("artifact", {})
    capabilities = artifact.get("capabilities", {})
    topology = (handoff or {}).get("topology", {})
    gates = (handoff or {}).get("cache_and_modality_gates", {})
    architecture = (shape or {}).get("architecture", {})
    quantization = (shape or {}).get("quantization", {})

    cache_contract = {
        "cache_type": capabilities.get("cache_type") or architecture.get("cache_type"),
        "cache_entries": topology.get("cache_entries"),
        "mamba_companion_state_entries": topology.get("mamba_companion_state_entries"),
        "attention_kv_cache_entries": topology.get("attention_kv_cache_entries"),
        "mamba_layers": topology.get("mamba_layers") or architecture.get("layer_counts", {}).get("mamba"),
        "attention_layers": topology.get("attention_layers") or architecture.get("layer_counts", {}).get("attention"),
        "kv_cache_boundary": gates.get("kv_cache_boundary"),
        "mamba_state_boundary": gates.get("mamba_state_boundary"),
        "prefix_cache_acceptance": [
            "attention KV hit is insufficient without the matching 48 Mamba companion states",
            "cache restore must preserve layer order and cache ordinal mapping",
            "parser streaming state must be salted/restored with the cache key",
        ],
    }
    parser_contract = {
        "reasoning_parser": capabilities.get("reasoning_parser"),
        "tool_parser": capabilities.get("tool_parser"),
        "supports_thinking": capabilities.get("supports_thinking"),
        "supports_tools": capabilities.get("supports_tools"),
        "think_in_template": capabilities.get("think_in_template"),
        "parser_probe": parser_probe,
        "long_coherence": long_coherence,
        "acceptance": [
            "no new visible <think>, </think>, <tool_call>, or <tool_response> leakage",
            "no truncated reasoning rows versus baseline",
            "tool-call parser remains Nemotron XML compatible",
            "do not hide parser failures with prompt suffixes, forced tags, or sampler penalties",
        ],
    }
    modality_contract = {
        "modality": capabilities.get("modality") or architecture.get("modality"),
        "text_only": gates.get("text_only"),
        "vl_policy": gates.get("vl_policy"),
        "audio_policy": "No audio tensors or processor configs are present; reject or reroute audio requests.",
        "mtp_policy": gates.get("mtp_policy"),
        "drops_mtp": artifact.get("drops_mtp") if artifact.get("drops_mtp") is not None else quantization.get("drops_mtp"),
    }

    partials: list[str] = []
    if cache_contract["cache_type"] != "hybrid":
        failures.append(f"cache_type is {cache_contract['cache_type']}, expected hybrid")
    if cache_contract["cache_entries"] not in (None, 60):
        failures.append(f"cache_entries is {cache_contract['cache_entries']}, expected 60")
    if cache_contract["mamba_companion_state_entries"] not in (None, 48):
        failures.append(
            f"mamba_companion_state_entries is {cache_contract['mamba_companion_state_entries']}, expected 48"
        )
    if cache_contract["attention_kv_cache_entries"] not in (None, 12):
        failures.append(f"attention_kv_cache_entries is {cache_contract['attention_kv_cache_entries']}, expected 12")
    if modality_contract["text_only"] is not True or modality_contract["modality"] != "text":
        failures.append("modality contract is not text-only")
    if modality_contract["drops_mtp"] is not True:
        failures.append("MTP is not marked dropped")
    if parser_contract["reasoning_parser"] != "deepseek_r1":
        failures.append(f"reasoning_parser is {parser_contract['reasoning_parser']}, expected deepseek_r1")
    if parser_contract["tool_parser"] != "nemotron":
        failures.append(f"tool_parser is {parser_contract['tool_parser']}, expected nemotron")

    if parser_probe["status"] != "FIXED":
        partials.append(f"parser probe is {parser_probe['status']}")
    if long_coherence["status"] != "FIXED":
        partials.append(f"long coherence is {long_coherence['status']}")
    if speed_acceptance.get("status") != "FIXED":
        partials.append(f"speed fix acceptance is {speed_acceptance.get('status', 'MISSING')}")
    if candidate_index.get("status") != "FIXED":
        partials.append(f"candidate index is {candidate_index.get('status', 'MISSING')}")

    status = "BLOCKED" if failures else ("PARTIAL" if partials else "FIXED")
    return {
        "status": status,
        "log_dir": str(log_dir),
        "cache_contract": cache_contract,
        "parser_contract": parser_contract,
        "modality_contract": modality_contract,
        "partials": partials,
        "failures": failures,
        "candidate_status": candidate_index.get("status"),
        "speed_acceptance_status": speed_acceptance.get("status"),
        "source_files": [
            str(handoff_path),
            str(shape_path),
            str(parser_path),
            str(long_path),
            str(speed_acceptance_path),
            str(candidate_index_path),
        ],
    }


def _fmt(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _render(result: dict[str, Any]) -> str:
    lines = [
        "# Nemotron Ultra Cache Parser Contract",
        "",
        f"log_dir: `{result['log_dir']}`",
        f"status: `{result['status']}`",
        f"speed_acceptance_status: `{result.get('speed_acceptance_status')}`",
        f"candidate_status: `{result.get('candidate_status')}`",
        "",
        "## Cache Contract",
    ]
    for key, value in result["cache_contract"].items():
        lines.append(f"- {key}: `{_fmt(value)}`")
    lines.extend(["", "## Parser Contract"])
    for key, value in result["parser_contract"].items():
        lines.append(f"- {key}: `{_fmt(value)}`")
    lines.extend(["", "## Modality And MTP"])
    for key, value in result["modality_contract"].items():
        lines.append(f"- {key}: `{_fmt(value)}`")
    lines.extend(["", "## Partial"])
    lines.extend(f"- {item}" for item in result["partials"] or ["none"])
    lines.extend(["", "## Failures"])
    lines.extend(f"- {item}" for item in result["failures"] or ["none"])
    lines.extend(["", "## Source Files"])
    lines.extend(f"- `{item}`" for item in result["source_files"])
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    ap.add_argument("--strict", action="store_true")
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
    if args.strict and result["status"] != "FIXED":
        raise SystemExit(2 if result["status"] == "BLOCKED" else 1)


if __name__ == "__main__":
    main()
