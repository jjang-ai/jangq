"""Pre-quantization expert pruning for raw Qwen 3.5/3.6 MoE HF bundles.

This command prunes the full-precision source before any JANG/JANGTQ
conversion. It streams safetensors byte ranges directly so BF16 tensors do not
require Torch, MLX, or loading the full model into memory.

Supported raw source layouts:
  model.language_model.layers.{L}.mlp.gate.weight
  model.language_model.layers.{L}.mlp.experts.gate_up_proj
  model.language_model.layers.{L}.mlp.experts.down_proj
  model.layers.{L}.mlp.gate.weight
  model.layers.{L}.mlp.experts.{gate_proj,up_proj,down_proj}.weight

Selection defaults to router-row L2 per layer only as a fallback. Prefer a
prompt-trace keep map exported from JANG Studio Expert Lab via ``--keep-map``;
the destructive rewrite path is the same, but the keep/drop decision is then
based on observed expert routing across prompt suites.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np


LAYER_PREFIX_RE = r"(?:model\.language_model\.layers|model\.layers|language_model\.layers)"
EXPERT_RE = re.compile(
    rf"^{LAYER_PREFIX_RE}\.(\d+)\.mlp\.experts\."
    r"(gate_up_proj|gate_proj|up_proj|down_proj)(?:\.weight)?$"
)
ROUTER_RE = re.compile(
    rf"^{LAYER_PREFIX_RE}\.(\d+)\.mlp\.gate\.weight$"
)
INDEX_NAME = "model.safetensors.index.json"
MIN_REVIEWED_PRUNE_PROMPTS = 50
MIN_REVIEWED_PRUNE_MEAN_TOKENS = 8.0
REQUIRED_REVIEWED_PRUNE_SEMANTIC_DOMAINS = {
    "math",
    "code",
    "formatting",
    "instruction_following",
    "reasoning",
    "safety_medical_legal_sensitive",
    "chinese",
    "non_english",
    "multilingual",
    "translation",
    "english_dominant",
    "unknown_language_role",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) != 8:
            raise RuntimeError(f"{path.name}: missing safetensors header length")
        header_len = struct.unpack("<Q", raw_len)[0]
        header = json.loads(fh.read(header_len))
    if not isinstance(header, dict):
        raise RuntimeError(f"{path.name}: safetensors header is not an object")
    return header, 8 + int(header_len)


def _tensor_items(header: dict[str, Any]):
    for key, info in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(info, dict):
            raise RuntimeError(f"safetensors entry {key} is not an object")
        yield key, info


def _data_offsets(info: dict[str, Any]) -> tuple[int, int]:
    offsets = info.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(v, int) for v in offsets)
    ):
        raise RuntimeError(f"tensor has invalid data_offsets: {offsets!r}")
    return int(offsets[0]), int(offsets[1])


def _shape(info: dict[str, Any]) -> list[int]:
    shape = info.get("shape")
    if not isinstance(shape, list) or not all(isinstance(v, int) for v in shape):
        raise RuntimeError(f"tensor has invalid shape: {shape!r}")
    return [int(v) for v in shape]


def _read_tensor_bytes(path: Path, info: dict[str, Any]) -> bytes:
    header, data_start = _read_safetensors_header(path)
    del header
    start, end = _data_offsets(info)
    with path.open("rb") as fh:
        fh.seek(data_start + start)
        data = fh.read(end - start)
    if len(data) != end - start:
        raise RuntimeError(f"{path.name}: short read for tensor data")
    return data


def _bf16_rows_to_f32(data: bytes, shape: list[int]) -> np.ndarray:
    u16 = np.frombuffer(data, dtype="<u2")
    expected = int(np.prod(shape))
    if u16.size != expected:
        raise RuntimeError(f"BF16 tensor byte count mismatch: got {u16.size}, expected {expected}")
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32).reshape(shape)


def _router_scores(data: bytes, dtype: str, shape: list[int]) -> np.ndarray:
    if len(shape) != 2:
        raise RuntimeError(f"router tensor must be rank-2, got shape {shape}")
    if dtype == "BF16":
        arr = _bf16_rows_to_f32(data, shape)
    elif dtype == "F16":
        arr = np.frombuffer(data, dtype="<f2").astype(np.float32).reshape(shape)
    elif dtype == "F32":
        arr = np.frombuffer(data, dtype="<f4").reshape(shape)
    else:
        raise RuntimeError(f"unsupported router dtype for scoring: {dtype}")
    return np.sum(arr * arr, axis=1)


def _router_keys(weight_map: dict[str, str]) -> list[str]:
    return sorted(
        [key for key in weight_map if ROUTER_RE.match(key)],
        key=lambda key: int(ROUTER_RE.match(key).group(1)),  # type: ignore[union-attr]
    )


def _compute_router_keep_indices(
    src: Path,
    weight_map: dict[str, str],
    keep_experts: int,
) -> tuple[dict[int, np.ndarray], int]:
    keys = _router_keys(weight_map)
    if not keys:
        raise RuntimeError("no Qwen MoE router weights found")

    keep_by_layer: dict[int, np.ndarray] = {}
    original_experts: int | None = None
    for key in keys:
        match = ROUTER_RE.match(key)
        assert match is not None
        layer = int(match.group(1))
        shard = src / weight_map[key]
        header, _ = _read_safetensors_header(shard)
        info = header[key]
        shape = _shape(info)
        if original_experts is None:
            original_experts = shape[0]
        elif shape[0] != original_experts:
            raise RuntimeError(
                f"router expert count mismatch: layer {layer} has {shape[0]}, "
                f"expected {original_experts}"
            )
        if keep_experts > shape[0]:
            raise RuntimeError(f"keep_experts={keep_experts} exceeds source experts={shape[0]}")
        data = _read_tensor_bytes(shard, info)
        scores = _router_scores(data, str(info.get("dtype")), shape)
        ranked = np.argsort(scores, kind="stable")
        keep_by_layer[layer] = np.sort(ranked[-keep_experts:]).astype(np.int64)

    assert original_experts is not None
    return keep_by_layer, original_experts


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and np.isfinite(value):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _float_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
        return number if np.isfinite(number) else None
    return None


def _bool_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def _string_value(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _path_value(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value).expanduser().resolve()


def _sidecar_path(value: Any, base_dir: Path | None) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def _plan_eval_artifact_dir(data: dict[str, Any], base_dir: Path | None) -> Path | None:
    path = _sidecar_path(
        data.get("eval_artifact")
        or data.get("evalArtifactPath")
        or data.get("eval_artifact_path"),
        base_dir,
    )
    if path is None:
        return None
    return path.parent if path.suffix else path


def _eval_artifact_sidecar(
    data: dict[str, Any],
    base_dir: Path | None,
    name: str,
) -> Path | None:
    directory = _plan_eval_artifact_dir(data, base_dir)
    if directory is None:
        return None
    return (directory / name).resolve()


def _review_suite_from_eval_artifact(
    data: dict[str, Any],
    base_dir: Path | None,
) -> Path | None:
    directory = _plan_eval_artifact_dir(data, base_dir)
    if directory is None:
        return None
    candidates = [
        directory / "suite.jsonl",
        directory.parent / "suite.jsonl",
        directory.parent.parent / "suite.jsonl",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[-1].resolve()


def _sidecar_path_with_fallback(
    value: Any,
    base_dir: Path | None,
    fallback: Path | None,
) -> Path | None:
    direct = _sidecar_path(value, base_dir)
    if direct is None:
        return fallback
    if direct.is_file() or fallback is None or not fallback.is_file():
        return direct
    return fallback


def _load_jsonl_objects(path: Path) -> list[dict[str, Any]] | None:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                parsed = json.loads(stripped)
                if not isinstance(parsed, dict):
                    return None
                rows.append(parsed)
    except (OSError, json.JSONDecodeError):
        return None
    return rows


def _prompt_id(row: dict[str, Any]) -> str | None:
    for key in ("promptID", "prompt_id", "id"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _preview_ids(ids: set[str]) -> str:
    ordered = sorted(ids)
    preview = ordered[:5]
    suffix = "" if len(ordered) <= len(preview) else f" (+{len(ordered) - len(preview)} more)"
    return ", ".join(preview) + suffix


def _row_text(row: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _int_set_map(value: Any) -> dict[int, set[int]] | None:
    if not isinstance(value, dict):
        return None
    out: dict[int, set[int]] = {}
    for raw_layer, raw_experts in value.items():
        layer = _int_value(raw_layer)
        if layer is None or layer < 0 or not isinstance(raw_experts, list):
            return None
        experts: set[int] = set()
        for raw_expert in raw_experts:
            expert = _int_value(raw_expert)
            if expert is None or expert < 0:
                return None
            experts.add(expert)
        out[layer] = experts
    return out


def _disabled_by_layer_from_mask(path: Path) -> dict[int, set[int]] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    for key in ("disabled_by_layer", "layers", "disabledExpertsByLayer"):
        if key not in data:
            continue
        disabled = _int_set_map(data.get(key))
        if disabled is None:
            return None
        return {layer: experts for layer, experts in disabled.items() if experts}
    return {}


def _preview_ints(values: set[int] | list[int], limit: int = 8) -> str:
    ordered = sorted(values)
    preview = ordered[:limit]
    suffix = "" if len(ordered) <= len(preview) else f" (+{len(ordered) - len(preview)} more)"
    return ", ".join(str(value) for value in preview) + suffix


def _normalized_semantic_slug(raw: str) -> str:
    return raw.strip().lower().replace("_", "-")


def _canonical_domain(raw: str) -> str:
    slug = _normalized_semantic_slug(raw)
    if slug in {"code", "coding", "swift", "python", "sql", "bugfinding", "concurrency", "syntax"}:
        return "coding"
    if slug in {"math", "arithmetic", "algebra", "estimation", "geometry", "statistics", "tensor"}:
        return "math"
    if slug in {"reasoning", "logic", "counterexample", "evidence", "verification"}:
        return "reasoning"
    if slug in {
        "multilingual",
        "language",
        "lang",
        "translation",
        "spanish",
        "french",
        "japanese",
        "chinese",
        "bilingual",
    }:
        return "language"
    if slug in {
        "robustness",
        "safety",
        "security",
        "refusal",
        "model-safety",
        "medical",
        "medicine-safety",
        "finance-safety",
    }:
        return "safety"
    if slug in {
        "agentic",
        "tools",
        "tool",
        "cli",
        "structured",
        "json",
        "table",
        "classification",
        "model-pruning",
        "expert-lab",
        "prune",
        "workflow",
        "recovery",
        "optimization",
        "planning",
    }:
        return "tools"
    if slug in {"instruction", "instruction-following", "hierarchy", "clarification"}:
        return "reasoning"
    known = {
        "general",
        "formatting",
        "instruction_following",
        "non_english",
        "chinese",
        "translation",
        "english_dominant",
        "unknown_language_role",
        "safety_medical_legal_sensitive",
        "safety_sensitive",
        "medical_sensitive",
        "legal_sensitive",
        "creative",
        "knowledge",
    }
    return slug.replace("-", "_") if slug.replace("-", "_") in known else "general"


def _canonical_semantic_domain(raw: str) -> str:
    slug = _normalized_semantic_slug(raw)
    if slug in {"code", "coding", "swift", "python", "sql", "bugfinding", "concurrency", "syntax"}:
        return "code"
    if slug in {"format", "formatting", "structured", "json", "table", "markdown"}:
        return "formatting"
    if slug in {"instruction", "instruction-following", "hierarchy", "clarification"}:
        return "instruction_following"
    if slug in {"multilingual", "language", "lang", "bilingual", "spanish", "french", "japanese"}:
        return "multilingual"
    if slug in {"non-english", "nonenglish", "romaji"}:
        return "non_english"
    if slug in {"chinese", "simplified-chinese", "traditional-chinese", "zh", "zh-cn", "zh-hans"}:
        return "chinese"
    if slug in {"translation", "translate", "back-translation"}:
        return "translation"
    if slug in {"english", "english-dominant", "english-dominant-role"}:
        return "english_dominant"
    if slug in {"unknown-language", "unknown-language-role", "language-id", "language-identification"}:
        return "unknown_language_role"
    if slug in {"safety-medical-legal-sensitive", "sensitive-domain", "sensitive"}:
        return "safety_medical_legal_sensitive"
    if slug in {"safety", "safety-sensitive", "robustness", "security", "refusal", "model-safety"}:
        return "safety_sensitive"
    if slug in {"medical", "medicine-safety", "medical-sensitive"}:
        return "medical_sensitive"
    if slug in {"legal", "law", "legal-sensitive"}:
        return "legal_sensitive"
    broad = _canonical_domain(slug)
    return "code" if broad == "coding" else broad


def _is_non_english_signal(slug: str) -> bool:
    return slug in {
        "multilingual",
        "bilingual",
        "spanish",
        "french",
        "japanese",
        "chinese",
        "simplified-chinese",
        "traditional-chinese",
        "zh",
        "zh-cn",
        "zh-hans",
        "non-english",
        "nonenglish",
        "romaji",
    }


def _is_translation_signal(slug: str) -> bool:
    return slug in {"translation", "translate", "back-translation"}


def _is_sensitive_signal(slug: str) -> bool:
    return slug in {
        "safety",
        "safety-sensitive",
        "robustness",
        "security",
        "refusal",
        "model-safety",
        "medical",
        "medicine-safety",
        "medical-sensitive",
        "legal",
        "law",
        "legal-sensitive",
        "finance-safety",
    }


def _suite_semantic_domains(row: dict[str, Any]) -> set[str]:
    domains: set[str] = set()

    def append(raw: str) -> None:
        canonical = _canonical_semantic_domain(raw)
        if canonical != "general":
            domains.add(canonical)
        slug = _normalized_semantic_slug(raw)
        if _is_non_english_signal(slug):
            domains.add("non_english")
        if _is_translation_signal(slug):
            domains.add("translation")
        if _is_sensitive_signal(slug):
            domains.add("safety_medical_legal_sensitive")

    for value in [
        _row_text(row, "domain"),
        _row_text(row, "subdomain", "sub_domain"),
    ]:
        if value is not None:
            append(value)
    for tag in _string_list(row.get("tags")):
        append(tag)
    if not domains:
        domain = _row_text(row, "domain")
        if domain is not None:
            domains.add(_canonical_domain(domain))
    return domains


def _suite_semantic_coverage_issue(rows: list[dict[str, Any]]) -> str | None:
    semantic_domains: set[str] = set()
    for row in rows:
        semantic_domains.update(_suite_semantic_domains(row))
    missing = sorted(REQUIRED_REVIEWED_PRUNE_SEMANTIC_DOMAINS.difference(semantic_domains))
    if missing:
        return "suite.jsonl is missing required semantic prompt probes: " + ", ".join(missing)
    return None


def _eval_index_semantic_coverage_issue(eval_index: dict[str, Any]) -> str | None:
    semantic_coverage = _string_list(
        _first_present(eval_index, "semantic_coverage", "semanticCoverage")
    )
    if not semantic_coverage:
        return "eval_index is missing semantic coverage evidence"
    coverage = {
        canonical
        for canonical in (_canonical_semantic_domain(value) for value in semantic_coverage)
        if canonical != "general"
    }
    missing = sorted(REQUIRED_REVIEWED_PRUNE_SEMANTIC_DOMAINS.difference(coverage))
    if missing:
        return "eval_index semantic coverage is missing required probes: " + ", ".join(missing)
    if (
        "missing_semantic_coverage" not in eval_index
        and "missingSemanticCoverage" not in eval_index
    ):
        return "eval_index is missing missing-semantic-coverage evidence"
    recorded_missing = {
        canonical
        for canonical in (
            _canonical_semantic_domain(value)
            for value in _string_list(
                _first_present(eval_index, "missing_semantic_coverage", "missingSemanticCoverage")
            )
        )
        if canonical != "general"
    }
    if recorded_missing:
        return "eval_index records missing semantic prompt probes: " + ", ".join(
            sorted(recorded_missing)
        )
    return None


def _eval_jsonl_issue(
    rows: list[dict[str, Any]],
    *,
    expected_prompt_ids: list[str],
    source_model_path: Path | None,
) -> str | None:
    if len(rows) != len(expected_prompt_ids):
        return f"eval.jsonl has {len(rows)} rows for {len(expected_prompt_ids)} indexed prompts"
    row_ids = [_prompt_id(row) for row in rows]
    if any(row_id is None for row_id in row_ids):
        return "eval.jsonl prompt IDs are unreadable"
    if row_ids != expected_prompt_ids:
        return "eval.jsonl prompt order does not match eval_index"
    for row in rows:
        if _row_text(row, "baselineText", "baseline_text") is None or _row_text(
            row, "maskedText", "masked_text"
        ) is None:
            return "eval.jsonl is missing per-prompt baseline/masked output text"
        if _float_value(_first_present(row, "textDelta", "text_delta")) is None:
            return "eval.jsonl is missing per-prompt text delta evidence"
        if (
            (_int_value(_first_present(row, "baselineTokenCount", "baseline_token_count")) or 0)
            <= 0
            or (_int_value(_first_present(row, "maskedTokenCount", "masked_token_count")) or 0)
            <= 0
        ):
            return "eval.jsonl is missing per-prompt token count evidence"
        if (
            (
                _int_value(
                    _first_present(
                        row,
                        "baselineRouteRecordCount",
                        "baseline_route_record_count",
                    )
                )
                or 0
            )
            <= 0
            or (
                _int_value(
                    _first_present(row, "maskedRouteRecordCount", "masked_route_record_count")
                )
                or 0
            )
            <= 0
        ):
            return "eval.jsonl is missing per-prompt routing record evidence"
        runtime_mode = _row_text(row, "runtimeMode", "runtime_mode")
        runtime_device = _row_text(row, "runtimeDevice", "runtime_device")
        runtime_metal_enabled = _bool_value(
            _first_present(row, "runtimeMetalEnabled", "runtime_metal_enabled")
        )
        if runtime_mode is None or runtime_device is None or runtime_metal_enabled is None:
            return "eval.jsonl is missing per-prompt runtime device evidence"
        if runtime_metal_enabled is not True:
            return "eval.jsonl did not record a Metal runtime"
        if runtime_mode != "bf16_vmlx":
            return "eval.jsonl did not record BF16/vMLX runtime evidence"
        if (
            _row_text(row, "jangToolsVersion", "jang_tools_version") is None
            or _row_text(row, "mlxVersion", "mlx_version") is None
            or _row_text(row, "mlxLMVersion", "mlx_lm_version") is None
        ):
            return "eval.jsonl is missing per-prompt vMLX package version evidence"
        row_source = _path_value(_first_present(row, "sourceModelPath", "source_model_path"))
        if row_source is None:
            return "eval.jsonl is missing per-prompt source model path evidence"
        if source_model_path is not None and row_source != source_model_path:
            return "eval.jsonl source model path does not match reviewed source"
        if _bool_value(_first_present(row, "maskApplied", "mask_applied")) is not True:
            return "eval.jsonl did not record an applied BF16/vMLX mask"
        row_disabled_expert_count = _int_value(
            _first_present(row, "disabledExpertCount", "disabled_expert_count")
        )
        if row_disabled_expert_count is None or row_disabled_expert_count <= 0:
            return (
                "eval.jsonl is missing per-prompt disabled expert evidence; "
                "top-k-only comparisons cannot authorize hard pruning"
            )
        if _row_text(row, "risk") is None or _row_text(
            row, "regressionSeverity", "regression_severity"
        ) is None:
            return "eval.jsonl is missing per-prompt regression flag evidence"
    return None


def _trace_row_has_mask_evidence(
    row: dict[str, Any],
    *,
    disabled_expert_count: int | None,
    topk_override: int | None,
) -> bool:
    record = row.get("record")
    if not isinstance(record, dict):
        return False
    if (disabled_expert_count or 0) > 0:
        disabled = _first_present(record, "disabledExperts", "disabled_experts")
        if isinstance(disabled, list) and disabled:
            return True
        return (
            _int_value(_first_present(record, "disabledExpertCount", "disabled_expert_count"))
            or 0
        ) > 0
    if topk_override is not None:
        return (
            _int_value(
                _first_present(record, "effectiveTopK", "effective_top_k", "topK", "top_k")
            )
            is not None
        )
    return True


def _trace_row_expected_mask_issue(
    row: dict[str, Any],
    *,
    prompt_id: str,
    expected_disabled_by_layer: dict[int, set[int]],
) -> tuple[int | None, str | None]:
    record = row.get("record")
    if not isinstance(record, dict):
        return None, None
    layer = _int_value(_first_present(record, "layer"))
    if layer is None or layer not in expected_disabled_by_layer:
        return None, None
    expected_disabled = expected_disabled_by_layer[layer]
    if not expected_disabled:
        return None, None

    selected = _first_present(record, "selectedExperts", "selected_experts")
    selected_experts: set[int] = set()
    if isinstance(selected, list):
        for raw_expert in selected:
            expert = _int_value(raw_expert)
            if expert is not None:
                selected_experts.add(expert)
    leaked = expected_disabled.intersection(selected_experts)
    if leaked:
        return (
            None,
            "eval_trace.jsonl masked routing record selected mask.json disabled experts "
            f"for prompt {prompt_id} layer {layer}: {_preview_ints(leaked)}",
        )

    disabled = _first_present(record, "disabledExperts", "disabled_experts")
    disabled_evidence: set[int] = set()
    if isinstance(disabled, list):
        for raw_expert in disabled:
            expert = _int_value(raw_expert)
            if expert is not None:
                disabled_evidence.add(expert)
    if expected_disabled.issubset(disabled_evidence):
        return layer, None
    return None, None


def _eval_trace_issue(
    rows: list[dict[str, Any]],
    *,
    expected_prompt_ids: list[str],
    disabled_expert_count: int | None,
    topk_override: int | None,
    expected_disabled_by_layer: dict[int, set[int]] | None = None,
    expected_baseline_route_record_count: int | None = None,
    expected_masked_route_record_count: int | None = None,
) -> str | None:
    if not rows:
        return "eval_trace.jsonl has no routing records"
    expected = set(expected_prompt_ids)
    seen = {_prompt_id(row) for row in rows}
    if None in seen:
        return "eval_trace.jsonl prompt IDs are unreadable"
    trace_ids = {str(prompt_id) for prompt_id in seen}
    missing = expected.difference(trace_ids)
    if missing:
        return "eval_index prompt IDs missing from eval_trace.jsonl: " + _preview_ids(missing)
    unexpected = trace_ids.difference(expected)
    if unexpected:
        return "eval_trace.jsonl prompt IDs outside eval_index: " + _preview_ids(unexpected)

    baseline_prompt_ids: set[str] = set()
    masked_prompt_ids: set[str] = set()
    masked_prompt_ids_with_mask_evidence: set[str] = set()
    baseline_trace_count = 0
    masked_trace_count = 0
    expected_mask_layers = set(
        layer for layer, disabled in (expected_disabled_by_layer or {}).items() if disabled
    )
    masked_prompt_layers_with_mask_json_evidence: dict[str, set[int]] = {}
    for row in rows:
        prompt_id = _prompt_id(row)
        assert prompt_id is not None
        variant = _row_text(row, "variant")
        variant = variant.lower() if variant is not None else None
        if variant == "baseline":
            baseline_prompt_ids.add(prompt_id)
            baseline_trace_count += 1
        elif variant == "masked":
            masked_prompt_ids.add(prompt_id)
            masked_trace_count += 1
            if expected_disabled_by_layer:
                layer, issue = _trace_row_expected_mask_issue(
                    row,
                    prompt_id=prompt_id,
                    expected_disabled_by_layer=expected_disabled_by_layer,
                )
                if issue is not None:
                    return issue
                if layer is not None:
                    masked_prompt_layers_with_mask_json_evidence.setdefault(prompt_id, set()).add(layer)
            if _trace_row_has_mask_evidence(
                row,
                disabled_expert_count=disabled_expert_count,
                topk_override=topk_override,
            ):
                masked_prompt_ids_with_mask_evidence.add(prompt_id)

    missing_baseline = expected.difference(baseline_prompt_ids)
    if missing_baseline:
        return "eval_trace.jsonl missing baseline routing records for prompt IDs: " + _preview_ids(
            missing_baseline
        )
    missing_masked = expected.difference(masked_prompt_ids)
    if missing_masked:
        return "eval_trace.jsonl missing masked routing records for prompt IDs: " + _preview_ids(
            missing_masked
        )
    if (
        expected_baseline_route_record_count is not None
        and baseline_trace_count != expected_baseline_route_record_count
    ):
        return (
            f"eval_trace.jsonl has {baseline_trace_count} baseline routing records for "
            f"{expected_baseline_route_record_count} indexed baseline route records"
        )
    if (
        expected_masked_route_record_count is not None
        and masked_trace_count != expected_masked_route_record_count
    ):
        return (
            f"eval_trace.jsonl has {masked_trace_count} masked routing records for "
            f"{expected_masked_route_record_count} indexed masked route records"
        )
    if (disabled_expert_count or 0) > 0 or topk_override is not None:
        if expected_mask_layers:
            for prompt_id in expected_prompt_ids:
                missing_layers = expected_mask_layers.difference(
                    masked_prompt_layers_with_mask_json_evidence.get(prompt_id, set())
                )
                if missing_layers:
                    return (
                        "eval_trace.jsonl masked routing records are missing mask.json evidence "
                        f"for prompt {prompt_id} layers: {_preview_ints(missing_layers)}"
                    )
        missing_mask_evidence = expected.difference(masked_prompt_ids_with_mask_evidence)
        if missing_mask_evidence:
            return (
                "eval_trace.jsonl masked routing records are missing mask evidence for prompt IDs: "
                + _preview_ids(missing_mask_evidence)
            )
    return None


def _reviewed_source_issue(data: dict[str, Any], src: Path) -> str | None:
    raw_source = data.get("source_model")
    if raw_source is None:
        raw_source = data.get("sourceModelPath")
    plan_source = _path_value(raw_source)
    if plan_source is None:
        return "plan is missing source_model path evidence"
    if plan_source != src.resolve():
        return f"plan source_model {plan_source} does not match selected source {src.resolve()}"
    return None


def _reviewed_safety_issue(
    data: dict[str, Any],
    *,
    keep_count: int,
    source_topk: int,
) -> str | None:
    safety = data.get("safety")
    if not isinstance(safety, dict):
        return "plan is missing top-k safety evidence"
    if _bool_value(safety.get("passed")) is not True:
        return "embedded top-k safety did not pass"
    issues = safety.get("issues")
    if isinstance(issues, list) and issues:
        return "embedded top-k safety issues remain: " + " ".join(str(issue) for issue in issues)
    raw_minimum = safety.get("minimum_active_experts_per_layer")
    if raw_minimum is None:
        raw_minimum = safety.get("minimumActiveExpertsPerLayer")
    minimum_active = _int_value(raw_minimum)
    if minimum_active is None:
        return "safety block is missing minimum active experts"
    if minimum_active != keep_count:
        return f"safety declares {minimum_active} active experts but keep-map keeps {keep_count}"
    raw_trained = safety.get("trained_top_k_by_layer")
    if raw_trained is None:
        raw_trained = safety.get("trainedTopKByLayer")
    if not isinstance(raw_trained, dict) or not raw_trained:
        return "safety block is missing trained top-k evidence"
    trained_values = [_int_value(value) for value in raw_trained.values()]
    trained_topk = max((value for value in trained_values if value is not None), default=None)
    if trained_topk is None:
        return "safety block has invalid trained top-k evidence"
    if keep_count < trained_topk:
        return f"plan keeps {keep_count} experts but embedded trained top-k is {trained_topk}"
    if source_topk and trained_topk != source_topk:
        return f"embedded trained top-k {trained_topk} does not match source router top-k {source_topk}"
    return None


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _reviewed_comparison_issue(
    data: dict[str, Any],
    *,
    src: Path | None = None,
    base_dir: Path | None = None,
) -> str | None:
    comparison = data.get("comparison_summary")
    if not isinstance(comparison, dict):
        return "plan is missing same-suite A/B comparison evidence"
    prompt_count = _int_value(comparison.get("promptCount") or comparison.get("prompt_count")) or 0
    if prompt_count < MIN_REVIEWED_PRUNE_PROMPTS:
        return f"compare at least {MIN_REVIEWED_PRUNE_PROMPTS} prompts before BF16/F16 prune"
    traced_prompt_count = _int_value(data.get("promptCount") or data.get("prompt_count")) or 0
    if traced_prompt_count > 0 and prompt_count != traced_prompt_count:
        return f"comparison summary covers {prompt_count} of {traced_prompt_count} traced prompts"
    baseline = _float_value(comparison.get("passRateBaseline") or comparison.get("pass_rate_baseline"))
    masked = _float_value(comparison.get("passRateMasked") or comparison.get("pass_rate_masked"))
    if baseline is not None and masked is not None and masked < baseline:
        return f"masked pass rate {masked:.0%} is below baseline {baseline:.0%}"
    high_risk = comparison.get("highRiskDomains")
    if high_risk is None:
        high_risk = comparison.get("high_risk_domains")
    if isinstance(high_risk, list) and high_risk:
        return "masked outputs regressed in high-risk domains: " + ", ".join(str(v) for v in sorted(high_risk))
    safe = comparison.get("safeDropCandidates")
    if safe is None:
        safe = comparison.get("safe_drop_candidates")
    if not isinstance(safe, list):
        return "comparison summary is missing A/B-safe candidates"
    if not safe:
        return "A/B comparison found no safe drop candidates"
    eval_index = data.get("eval_index")
    if not isinstance(eval_index, dict):
        return "plan is missing per-prompt eval_index evidence"
    eval_prompt_count = _int_value(eval_index.get("prompt_count") or eval_index.get("promptCount")) or 0
    prompt_ids = eval_index.get("prompt_ids")
    if prompt_ids is None:
        prompt_ids = eval_index.get("promptIDs")
    if not isinstance(prompt_ids, list):
        return "eval_index is missing prompt IDs"
    prompt_id_values = [str(prompt_id) for prompt_id in prompt_ids]
    if len(prompt_ids) != eval_prompt_count:
        return f"eval_index lists {len(prompt_ids)} prompt IDs for {eval_prompt_count} indexed prompts"
    if len(set(prompt_id_values)) < len(prompt_id_values):
        return "eval_index contains duplicate prompt IDs"
    issue = _eval_index_semantic_coverage_issue(eval_index)
    if issue is not None:
        return issue
    if eval_prompt_count != prompt_count:
        return f"eval_index covers {eval_prompt_count} of {prompt_count} compared prompts"
    if traced_prompt_count > 0 and eval_prompt_count != traced_prompt_count:
        return f"eval_index covers {eval_prompt_count} of {traced_prompt_count} traced prompts"
    risky = eval_index.get("risky_prompt_ids")
    if risky is None:
        risky = eval_index.get("riskyPromptIDs")
    if isinstance(risky, list) and risky:
        return "eval_index still has risky prompt IDs"
    index_high_risk = eval_index.get("high_risk_domains")
    if index_high_risk is None:
        index_high_risk = eval_index.get("highRiskDomains")
    if isinstance(index_high_risk, list) and index_high_risk:
        return "eval_index still has high-risk domains: " + ", ".join(
            str(v) for v in sorted(index_high_risk)
        )
    raw_mean_baseline_tokens = eval_index.get("mean_baseline_tokens")
    if raw_mean_baseline_tokens is None:
        raw_mean_baseline_tokens = eval_index.get("meanBaselineTokens")
    raw_mean_masked_tokens = eval_index.get("mean_masked_tokens")
    if raw_mean_masked_tokens is None:
        raw_mean_masked_tokens = eval_index.get("meanMaskedTokens")
    mean_baseline_tokens = _float_value(raw_mean_baseline_tokens)
    mean_masked_tokens = _float_value(raw_mean_masked_tokens)
    if mean_baseline_tokens is None or mean_masked_tokens is None:
        return "eval_index is missing generation-depth token evidence"
    shallow = min(mean_baseline_tokens, mean_masked_tokens)
    if shallow < MIN_REVIEWED_PRUNE_MEAN_TOKENS:
        return (
            f"eval_index average generated depth {shallow:.1f} tokens is below "
            f"{MIN_REVIEWED_PRUNE_MEAN_TOKENS:.0f}"
        )
    runtime_mode = eval_index.get("runtime_mode")
    if runtime_mode is None:
        runtime_mode = eval_index.get("runtimeMode")
    runtime_device = eval_index.get("runtime_device")
    if runtime_device is None:
        runtime_device = eval_index.get("runtimeDevice")
    runtime_metal_enabled = eval_index.get("runtime_metal_enabled")
    if runtime_metal_enabled is None:
        runtime_metal_enabled = eval_index.get("runtimeMetalEnabled")
    if not isinstance(runtime_mode, str) or not runtime_mode.strip():
        return "eval_index is missing runtime device evidence"
    if not isinstance(runtime_device, str) or not runtime_device.strip():
        return "eval_index is missing runtime device evidence"
    if _bool_value(runtime_metal_enabled) is None:
        return "eval_index is missing runtime device evidence"
    if _bool_value(runtime_metal_enabled) is not True:
        return "eval_index did not record a Metal runtime"
    if runtime_mode != "bf16_vmlx":
        return "eval_index did not record BF16/vMLX runtime evidence"

    jang_tools_version = _string_value(
        eval_index.get("jang_tools_version") or eval_index.get("jangToolsVersion")
    )
    mlx_version = _string_value(eval_index.get("mlx_version") or eval_index.get("mlxVersion"))
    mlx_lm_version = _string_value(eval_index.get("mlx_lm_version") or eval_index.get("mlxLMVersion"))
    if jang_tools_version is None or mlx_version is None or mlx_lm_version is None:
        return "eval_index is missing vMLX package version evidence"

    source_model_path = _path_value(
        eval_index.get("source_model_path") or eval_index.get("sourceModelPath")
    )
    if source_model_path is None:
        return "eval_index is missing source model path evidence"
    if src is not None and source_model_path != src.resolve():
        return "eval_index source model path does not match reviewed source"

    baseline_route_count = _int_value(
        eval_index.get("baseline_route_record_count") or eval_index.get("baselineRouteRecordCount")
    )
    masked_route_count = _int_value(
        eval_index.get("masked_route_record_count") or eval_index.get("maskedRouteRecordCount")
    )
    if (
        baseline_route_count is None
        or masked_route_count is None
        or baseline_route_count < eval_prompt_count
        or masked_route_count < eval_prompt_count
    ):
        return "eval_index is missing routing record evidence for every indexed prompt"
    hook_coverage_complete = _bool_value(
        _first_present(eval_index, "hook_coverage_complete", "hookCoverageComplete")
    )
    if hook_coverage_complete is False:
        return "eval_index recorded incomplete vMLX routed-layer hook coverage"
    hooked_moe_layers = _int_value(
        _first_present(eval_index, "hooked_moe_layers", "hookedMOELayers")
    )
    if hooked_moe_layers is None or hooked_moe_layers <= 0:
        return "eval_index is missing vMLX routed-layer hook evidence"
    expected_moe_layers = _int_value(
        _first_present(eval_index, "expected_moe_layers", "expectedMOELayers")
    )
    if expected_moe_layers is not None and hooked_moe_layers < expected_moe_layers:
        return (
            f"eval_index vMLX hook coverage {hooked_moe_layers} of "
            f"{expected_moe_layers} config-routed layers"
        )

    eval_jsonl = _sidecar_path_with_fallback(
        eval_index.get("eval_jsonl") or eval_index.get("evalJSONL"),
        base_dir,
        _eval_artifact_sidecar(data, base_dir, "eval.jsonl"),
    )
    if eval_jsonl is None:
        return "eval_index is missing eval.jsonl evidence"

    eval_trace_jsonl = _sidecar_path_with_fallback(
        eval_index.get("eval_trace_jsonl") or eval_index.get("evalTraceJSONL"),
        base_dir,
        _eval_artifact_sidecar(data, base_dir, "eval_trace.jsonl"),
    )
    if eval_trace_jsonl is None:
        return "eval_index is missing eval_trace.jsonl evidence"

    mask_applied = _bool_value(eval_index.get("mask_applied") or eval_index.get("maskApplied"))
    if mask_applied is not True:
        return "eval_index did not record an applied BF16/vMLX mask"
    disabled_expert_count = _int_value(
        eval_index.get("disabled_expert_count") or eval_index.get("disabledExpertCount")
    )
    topk_override = _int_value(eval_index.get("top_k_override") or eval_index.get("topKOverride"))
    if disabled_expert_count is None or disabled_expert_count <= 0:
        return (
            "eval_index did not record disabled expert evidence; "
            "top-k-only comparisons cannot authorize hard pruning"
        )
    mask_json = _sidecar_path_with_fallback(
        eval_index.get("mask_json")
        or eval_index.get("maskJSON")
        or eval_index.get("mask")
        or data.get("mask_json")
        or data.get("maskJSON")
        or data.get("mask"),
        base_dir,
        _eval_artifact_sidecar(data, base_dir, "mask.json"),
    )
    if mask_json is None:
        return "eval_index is missing mask.json evidence"
    if not mask_json.is_file():
        return "mask.json sidecar is missing"
    expected_disabled_by_layer = _disabled_by_layer_from_mask(mask_json)
    if expected_disabled_by_layer is None:
        return "mask.json is unreadable"
    mask_disabled_count = sum(len(experts) for experts in expected_disabled_by_layer.values())
    if mask_disabled_count <= 0:
        return (
            "mask.json does not disable any experts; "
            "top-k-only comparisons cannot authorize hard pruning"
        )
    if mask_disabled_count != disabled_expert_count:
        return (
            f"eval_index disabled expert count {disabled_expert_count} "
            f"does not match mask.json {mask_disabled_count}"
        )
    if not eval_jsonl.is_file():
        return "eval.jsonl sidecar is missing"
    eval_rows = _load_jsonl_objects(eval_jsonl)
    if eval_rows is None:
        return "eval.jsonl is unreadable"
    issue = _eval_jsonl_issue(
        eval_rows,
        expected_prompt_ids=prompt_id_values,
        source_model_path=source_model_path,
    )
    if issue is not None:
        return issue
    if not eval_trace_jsonl.is_file():
        return "eval_trace.jsonl sidecar is missing"
    eval_trace_rows = _load_jsonl_objects(eval_trace_jsonl)
    if eval_trace_rows is None:
        return "eval_trace.jsonl is unreadable"
    issue = _eval_trace_issue(
        eval_trace_rows,
        expected_prompt_ids=prompt_id_values,
        disabled_expert_count=disabled_expert_count,
        topk_override=topk_override,
        expected_disabled_by_layer=expected_disabled_by_layer,
        expected_baseline_route_record_count=baseline_route_count,
        expected_masked_route_record_count=masked_route_count,
    )
    if issue is not None:
        return issue

    suite_jsonl = _sidecar_path_with_fallback(
        data.get("suite_jsonl")
        or data.get("suiteJSONL")
        or eval_index.get("suite_jsonl")
        or eval_index.get("suiteJSONL"),
        base_dir,
        _review_suite_from_eval_artifact(data, base_dir),
    )
    if suite_jsonl is None:
        return "plan is missing suite.jsonl evidence"
    if not suite_jsonl.is_file():
        return "suite.jsonl sidecar is missing"
    suite_rows = _load_jsonl_objects(suite_jsonl)
    if suite_rows is None:
        return "suite.jsonl semantic prompt coverage is unreadable"
    suite_ids = [_prompt_id(row) for row in suite_rows]
    if any(prompt_id is None for prompt_id in suite_ids):
        return "suite.jsonl prompt IDs are unreadable"
    suite_id_values = [str(prompt_id) for prompt_id in suite_ids]
    if len(set(suite_id_values)) < len(suite_id_values):
        return "suite.jsonl contains duplicate prompt IDs"
    suite_set = set(suite_id_values)
    indexed_set = set(prompt_id_values)
    missing_suite_ids = suite_set.difference(indexed_set)
    if missing_suite_ids:
        return "eval_index prompt IDs missing suite.jsonl prompts: " + _preview_ids(
            missing_suite_ids
        )
    unexpected_suite_ids = indexed_set.difference(suite_set)
    if unexpected_suite_ids:
        return "eval_index prompt IDs outside suite.jsonl: " + _preview_ids(
            unexpected_suite_ids
        )
    if suite_id_values != prompt_id_values:
        return "eval_index prompt order does not match suite.jsonl"
    issue = _suite_semantic_coverage_issue(suite_rows)
    if issue is not None:
        return issue
    suite_sha256 = _file_sha256(suite_jsonl)
    recorded_suite_sha256 = _string_value(
        eval_index.get("suite_sha256")
        or eval_index.get("suiteSHA256")
        or data.get("suite_sha256")
        or data.get("suiteSHA256")
    )
    if recorded_suite_sha256 is None:
        return "eval_index is missing suite.jsonl fingerprint evidence"
    if recorded_suite_sha256 != suite_sha256:
        return "eval_index suite.jsonl fingerprint does not match suite.jsonl"
    return None


def _load_keep_map(
    path: Path,
    *,
    src: Path | None = None,
    source_topk: int = 0,
    require_reviewed_comparison: bool = False,
) -> tuple[dict[int, np.ndarray], int, str]:
    data = _load_json(path)
    if require_reviewed_comparison:
        if src is None:
            raise RuntimeError(f"{path}: source path is required for reviewed keep-map validation")
        issue = _reviewed_source_issue(data, src)
        if issue is not None:
            raise RuntimeError(f"{path}: reviewed keep-map failed source gate: {issue}")
        issue = _reviewed_comparison_issue(data, src=src, base_dir=path.parent)
        if issue is not None:
            raise RuntimeError(f"{path}: reviewed keep-map failed same-suite comparison gate: {issue}")
    layers = data.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise RuntimeError(f"{path}: keep-map must contain a non-empty layers object")

    keep_by_layer: dict[int, np.ndarray] = {}
    keep_count: int | None = None
    for raw_layer, layer_data in layers.items():
        try:
            layer = int(raw_layer)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{path}: invalid layer key {raw_layer!r}") from exc
        if not isinstance(layer_data, dict):
            raise RuntimeError(f"{path}: layer {layer} must be an object")
        keep = layer_data.get("keep")
        if not isinstance(keep, list) or not keep:
            raise RuntimeError(f"{path}: layer {layer} must contain a non-empty keep list")
        if not all(isinstance(v, int) for v in keep):
            raise RuntimeError(f"{path}: layer {layer} keep list must contain integers")
        unique = sorted(set(int(v) for v in keep))
        if len(unique) != len(keep):
            raise RuntimeError(f"{path}: layer {layer} keep list contains duplicates")
        if keep_count is None:
            keep_count = len(unique)
        elif keep_count != len(unique):
            raise RuntimeError(
                f"{path}: keep count mismatch; layer {layer} has {len(unique)}, "
                f"expected {keep_count}"
            )
        keep_by_layer[layer] = np.asarray(unique, dtype=np.int64)

    method = str(data.get("method") or "external_keep_map")
    assert keep_count is not None
    if require_reviewed_comparison:
        issue = _reviewed_safety_issue(data, keep_count=keep_count, source_topk=source_topk)
        if issue is not None:
            raise RuntimeError(f"{path}: reviewed keep-map failed top-k safety gate: {issue}")
    return keep_by_layer, keep_count, method


def _compute_keep_indices(
    src: Path,
    weight_map: dict[str, str],
    keep_experts: int | None,
    keep_map: Path | None,
    source_topk: int = 0,
    require_reviewed_comparison: bool = False,
) -> tuple[dict[int, np.ndarray], int, int, str]:
    if keep_map is None:
        if keep_experts is None:
            raise RuntimeError("keep_experts is required when --keep-map is not provided")
        keep_by_layer, original_experts = _compute_router_keep_indices(src, weight_map, keep_experts)
        return keep_by_layer, original_experts, keep_experts, "router_row_l2_topk_per_layer"

    keys = _router_keys(weight_map)
    if not keys:
        raise RuntimeError("no Qwen MoE router weights found")

    mapped, mapped_keep_count, method = _load_keep_map(
        keep_map,
        src=src,
        source_topk=source_topk,
        require_reviewed_comparison=require_reviewed_comparison,
    )
    if keep_experts is not None and keep_experts != mapped_keep_count:
        raise RuntimeError(
            f"--keep-experts={keep_experts} does not match keep-map count={mapped_keep_count}"
        )

    keep_by_layer: dict[int, np.ndarray] = {}
    original_experts: int | None = None
    required_layers: set[int] = set()
    for key in keys:
        match = ROUTER_RE.match(key)
        assert match is not None
        layer = int(match.group(1))
        required_layers.add(layer)
        shard = src / weight_map[key]
        header, _ = _read_safetensors_header(shard)
        info = header[key]
        shape = _shape(info)
        if original_experts is None:
            original_experts = shape[0]
        elif shape[0] != original_experts:
            raise RuntimeError(
                f"router expert count mismatch: layer {layer} has {shape[0]}, "
                f"expected {original_experts}"
            )
        keep = mapped.get(layer)
        if keep is None:
            raise RuntimeError(f"keep-map {keep_map} is missing router layer {layer}")
        if len(keep) > shape[0]:
            raise RuntimeError(f"keep-map layer {layer} keeps {len(keep)} experts, source has {shape[0]}")
        out_of_range = [int(v) for v in keep.tolist() if int(v) < 0 or int(v) >= shape[0]]
        if out_of_range:
            raise RuntimeError(f"keep-map layer {layer} has out-of-range experts: {out_of_range[:8]}")
        keep_by_layer[layer] = keep

    extra_layers = sorted(set(mapped) - required_layers)
    if extra_layers:
        raise RuntimeError(f"keep-map has layers not present in source routers: {extra_layers[:8]}")

    assert original_experts is not None
    return keep_by_layer, original_experts, mapped_keep_count, method


def _copy_range(src_fh: BinaryIO, dst_fh: BinaryIO, start: int, size: int, chunk: int = 8 << 20) -> None:
    src_fh.seek(start)
    remaining = size
    while remaining:
        data = src_fh.read(min(chunk, remaining))
        if not data:
            raise RuntimeError("short read while copying safetensors data")
        dst_fh.write(data)
        remaining -= len(data)


def _copy_sliced_axis0(
    src_fh: BinaryIO,
    dst_fh: BinaryIO,
    *,
    data_start: int,
    offsets: tuple[int, int],
    shape: list[int],
    keep: np.ndarray,
) -> None:
    start, end = offsets
    if not shape or shape[0] <= 0:
        raise RuntimeError(f"cannot slice tensor with shape {shape}")
    tensor_bytes = end - start
    if tensor_bytes % shape[0] != 0:
        raise RuntimeError(f"axis-0 row size is not integral for shape {shape}")
    row_bytes = tensor_bytes // shape[0]
    base = data_start + start
    for expert in keep.tolist():
        if expert < 0 or expert >= shape[0]:
            raise RuntimeError(f"keep expert {expert} out of range for shape {shape}")
        _copy_range(src_fh, dst_fh, base + int(expert) * row_bytes, row_bytes)


def _slice_plan_for_key(key: str, keep_by_layer: dict[int, np.ndarray]) -> np.ndarray | None:
    match = EXPERT_RE.match(key) or ROUTER_RE.match(key)
    if match is None:
        return None
    return keep_by_layer.get(int(match.group(1)))


def _write_pruned_shard(
    src_shard: Path,
    dst_shard: Path,
    keep_by_layer: dict[int, np.ndarray],
) -> tuple[int, dict[str, list[int]]]:
    header, data_start = _read_safetensors_header(src_shard)
    new_header: dict[str, Any] = {}
    actions: list[tuple[str, dict[str, Any], np.ndarray | None]] = []
    cursor = 0
    shape_updates: dict[str, list[int]] = {}

    for key, value in header.items():
        if key == "__metadata__":
            new_header[key] = value
            continue
        info = dict(value)
        shape = _shape(info)
        start, end = _data_offsets(info)
        keep = _slice_plan_for_key(key, keep_by_layer)
        if keep is not None:
            if not shape or shape[0] <= 0:
                raise RuntimeError(f"{key}: cannot prune shape {shape}")
            old_size = end - start
            if old_size % shape[0] != 0:
                raise RuntimeError(f"{key}: axis-0 row size is not integral")
            row_bytes = old_size // shape[0]
            new_shape = [len(keep), *shape[1:]]
            new_size = row_bytes * len(keep)
            info["shape"] = new_shape
            shape_updates[key] = new_shape
        else:
            new_size = end - start
        info["data_offsets"] = [cursor, cursor + new_size]
        new_header[key] = info
        actions.append((key, value, keep))
        cursor += new_size

    dst_shard.parent.mkdir(parents=True, exist_ok=True)
    header_bytes = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    with src_shard.open("rb") as src_fh, dst_shard.open("wb") as dst_fh:
        dst_fh.write(struct.pack("<Q", len(header_bytes)))
        dst_fh.write(header_bytes)
        for key, old_info, keep in actions:
            old_shape = _shape(old_info)
            offsets = _data_offsets(old_info)
            if keep is None:
                start, end = offsets
                _copy_range(src_fh, dst_fh, data_start + start, end - start)
            else:
                _copy_sliced_axis0(
                    src_fh,
                    dst_fh,
                    data_start=data_start,
                    offsets=offsets,
                    shape=old_shape,
                    keep=keep,
                )

    return dst_shard.stat().st_size, shape_updates


def _copy_sidecars(src: Path, dst: Path) -> None:
    for path in src.iterdir():
        if path.is_dir():
            continue
        if path.suffix == ".safetensors" or path.name == INDEX_NAME:
            continue
        shutil.copy2(path, dst / path.name)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _reviewed_prompt_count(data: dict[str, Any]) -> int | None:
    comparison = data.get("comparison_summary")
    eval_index = data.get("eval_index")
    for value in [
        comparison.get("promptCount") if isinstance(comparison, dict) else None,
        comparison.get("prompt_count") if isinstance(comparison, dict) else None,
        eval_index.get("prompt_count") if isinstance(eval_index, dict) else None,
        eval_index.get("promptCount") if isinstance(eval_index, dict) else None,
        data.get("promptCount"),
        data.get("prompt_count"),
    ]:
        prompt_count = _int_value(value)
        if prompt_count is not None:
            return prompt_count
    return None


def _copy_optional_review_sidecar(
    *,
    src: Path | None,
    dst: Path,
    name: str,
) -> str | None:
    if src is None or not src.is_file():
        return None
    target = dst / name
    shutil.copy2(src, target)
    return target.name


def _materialize_reviewed_evidence_bundle(
    *,
    src: Path,
    dst: Path,
    keep_map: Path,
) -> dict[str, Any]:
    data = _load_json(keep_map)
    eval_index = data.get("eval_index")
    comparison = data.get("comparison_summary")
    if not isinstance(eval_index, dict) or not isinstance(comparison, dict):
        raise RuntimeError("reviewed keep-map is missing validated comparison/eval_index evidence")

    base_dir = keep_map.parent
    suite_src = _sidecar_path_with_fallback(
        data.get("suite_jsonl")
        or data.get("suiteJSONL")
        or eval_index.get("suite_jsonl")
        or eval_index.get("suiteJSONL"),
        base_dir,
        _review_suite_from_eval_artifact(data, base_dir),
    )
    eval_src = _sidecar_path_with_fallback(
        eval_index.get("eval_jsonl") or eval_index.get("evalJSONL"),
        base_dir,
        _eval_artifact_sidecar(data, base_dir, "eval.jsonl"),
    )
    trace_src = _sidecar_path_with_fallback(
        eval_index.get("eval_trace_jsonl") or eval_index.get("evalTraceJSONL"),
        base_dir,
        _eval_artifact_sidecar(data, base_dir, "eval_trace.jsonl"),
    )
    for label, path in [
        ("suite.jsonl", suite_src),
        ("eval.jsonl", eval_src),
        ("eval_trace.jsonl", trace_src),
    ]:
        if path is None or not path.is_file():
            raise RuntimeError(f"reviewed keep-map {label} sidecar disappeared after validation")

    suite_name = _copy_optional_review_sidecar(
        src=suite_src,
        dst=dst,
        name="expert_lab_suite.jsonl",
    )
    eval_name = _copy_optional_review_sidecar(
        src=eval_src,
        dst=dst,
        name="expert_lab_eval.jsonl",
    )
    trace_name = _copy_optional_review_sidecar(
        src=trace_src,
        dst=dst,
        name="expert_lab_eval_trace.jsonl",
    )
    assert suite_name is not None and eval_name is not None and trace_name is not None
    suite_sha256 = _file_sha256(dst / suite_name)

    comparison_name = "expert_lab_comparison_summary.json"
    eval_index_name = "expert_lab_eval_index.json"
    _write_json(dst / comparison_name, comparison)

    eval_index_out = dict(eval_index)
    eval_index_out["suite_jsonl"] = suite_name
    eval_index_out["suite_sha256"] = suite_sha256
    eval_index_out["comparison_summary"] = comparison_name
    eval_index_out["eval_jsonl"] = eval_name
    eval_index_out["eval_trace_jsonl"] = trace_name

    mask_src = _sidecar_path_with_fallback(
        eval_index.get("mask_json")
        or eval_index.get("maskJSON")
        or eval_index.get("mask")
        or data.get("mask_json")
        or data.get("maskJSON")
        or data.get("mask"),
        base_dir,
        _eval_artifact_sidecar(data, base_dir, "mask.json"),
    )
    mask_name = _copy_optional_review_sidecar(
        src=mask_src,
        dst=dst,
        name="expert_lab_mask.json",
    )
    if mask_name is not None:
        eval_index_out["mask"] = mask_name
        eval_index_out["mask_json"] = mask_name

    _write_json(dst / eval_index_name, eval_index_out)

    plan = dict(data)
    plan["suite_jsonl"] = suite_name
    plan["suite_sha256"] = suite_sha256
    plan["eval_index"] = eval_index_out
    plan["reviewed_evidence_sidecars"] = {
        "suite_jsonl": suite_name,
        "comparison_summary": comparison_name,
        "eval_jsonl": eval_name,
        "eval_trace_jsonl": trace_name,
        "eval_index": eval_index_name,
    }
    if mask_name is not None:
        plan["reviewed_evidence_sidecars"]["mask"] = mask_name
        plan["reviewed_evidence_sidecars"]["mask_json"] = mask_name

    prompt_count = _reviewed_prompt_count(data)
    summary: dict[str, Any] = {
        "schema": "jang-expert-lab-pruned-source-review-v1",
        "generated_at": _utc_now_iso(),
        "same_suite_verification_ready": True,
        "review_sidecars_ready": True,
        "review_sidecars_issue": None,
        "pruned_suite_verification_ready": False,
        "pruned_suite_verification_issue": (
            "pruned-source same-suite vMLX generation has not been run for this "
            "pruned BF16/F16 output"
        ),
        "source_model_path": str(src),
        "source_model": str(src),
        "pruned_source": str(dst),
        "reviewed_prune_plan": str(dst / "prune_plan.json"),
        "suite_jsonl": str(dst / suite_name),
        "suite_sha256": suite_sha256,
        "comparison_summary": str(dst / comparison_name),
        "eval_jsonl": str(dst / eval_name),
        "eval_trace_jsonl": str(dst / trace_name),
        "eval_index": str(dst / eval_index_name),
    }
    if prompt_count is not None:
        summary["prompt_count"] = prompt_count
    if mask_name is not None:
        summary["mask"] = str(dst / mask_name)
        summary["mask_json"] = str(dst / mask_name)
    for key in ("run_id", "atlas_id", "review_run_directory", "review_eval_directory"):
        if data.get(key) is not None:
            summary[key] = data[key]
    _write_json(dst / "expert_lab_review_summary.json", summary)

    manifest = {
        "reviewed_prune_plan": str(dst / "prune_plan.json"),
        "review_summary": str(dst / "expert_lab_review_summary.json"),
        "suite_jsonl": str(dst / suite_name),
        "suite_sha256": suite_sha256,
        "comparison_summary": str(dst / comparison_name),
        "eval_jsonl": str(dst / eval_name),
        "eval_trace_jsonl": str(dst / trace_name),
        "eval_index": str(dst / eval_index_name),
        "pruned_suite_verification_ready": False,
    }
    if mask_name is not None:
        manifest["mask"] = str(dst / mask_name)
        manifest["mask_json"] = str(dst / mask_name)
    return {
        "plan": plan,
        "summary": summary,
        "manifest": manifest,
    }


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    text = config.get("text_config")
    if isinstance(text, dict):
        return text
    return config


def _config_int(config: dict[str, Any], *keys: str) -> int:
    text = _text_config(config)
    for key in keys:
        value = config.get(key)
        if value is None:
            value = text.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _update_config(
    dst: Path,
    *,
    keep_experts: int,
    original_experts: int,
    method: str,
    keep_map: Path | None,
) -> None:
    config_path = dst / "config.json"
    config = _load_json(config_path)
    text = _text_config(config)
    for target in (config, text):
        if not isinstance(target, dict):
            continue
        for key in ("num_experts", "n_routed_experts", "num_local_experts", "moe_num_experts"):
            if key in target:
                target[key] = keep_experts
    text["num_experts"] = keep_experts
    config["jang_prequant_expert_pruning"] = {
        "source_num_experts": original_experts,
        "num_experts": keep_experts,
        "dropped_experts_per_layer": original_experts - keep_experts,
        "method": method,
        "stage": "pre_quantization_bf16_source",
        "manifest": "expert_prune_manifest.json",
    }
    if keep_map is not None:
        config["jang_prequant_expert_pruning"]["keep_map"] = str(keep_map)
    _write_json(config_path, config)


def _write_index(
    dst: Path,
    index: dict[str, Any],
    shard_sizes: dict[str, int],
) -> None:
    new_index = dict(index)
    metadata = dict(new_index.get("metadata") or {})
    metadata["total_size"] = sum(shard_sizes.values())
    new_index["metadata"] = metadata
    _write_json(dst / INDEX_NAME, new_index)


def _manifest(
    *,
    src: Path,
    dst: Path,
    keep_by_layer: dict[int, np.ndarray],
    original_experts: int,
    keep_experts: int,
    method: str,
    keep_map: Path | None,
    shape_updates: dict[str, list[int]],
) -> dict[str, Any]:
    layers: dict[str, Any] = {}
    for layer, keep in sorted(keep_by_layer.items()):
        kept = {int(v) for v in keep.tolist()}
        layers[str(layer)] = {
            "keep_count": keep_experts,
            "drop_count": original_experts - keep_experts,
            "keep": [int(v) for v in keep.tolist()],
            "drop": [idx for idx in range(original_experts) if idx not in kept],
        }
    return {
        "schema": "qwen-moe-prequant-expert-prune-v1",
        "source": str(src),
        "output": str(dst),
        "stage": "pre_quantization_bf16_source",
        "method": method,
        "source_num_experts": original_experts,
        "num_experts": keep_experts,
        "num_layers": len(keep_by_layer),
        "shape_update_count": len(shape_updates),
        "caveat": (
            "Router-row-L2 is a structural proxy. Prefer prompt routing or "
            "REAP saliency plans when available, then convert/quantize the "
            "resulting pruned source bundle."
            if keep_map is None
            else "Keep/drop map came from an external prompt-trace or saliency plan. "
            "Quality still depends on the prompt suite used to collect evidence."
        ),
        "keep_map": str(keep_map) if keep_map is not None else None,
        "layers": layers,
    }


def _source_fingerprint(src: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for name in ("config.json", INDEX_NAME, "tokenizer.json", "tokenizer_config.json"):
        path = src / name
        if not path.exists() or not path.is_file():
            continue
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        files[name] = {
            "bytes": path.stat().st_size,
            "sha256": h.hexdigest(),
        }
    return {
        "source": str(src),
        "files": files,
    }


def _verify_pruned_output(
    dst: Path,
    *,
    keep_by_layer: dict[int, np.ndarray],
    keep_experts: int,
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "config_parses": False,
        "index_parses": False,
        "index_covers_tensors": False,
        "router_rows_match": False,
        "expert_rows_match": False,
    }
    errors: list[str] = []

    try:
        config = _load_json(dst / "config.json")
        checks["config_parses"] = isinstance(config, dict)
    except Exception as exc:  # pragma: no cover - returned to caller
        errors.append(f"config parse failed: {exc}")

    try:
        index = _load_json(dst / INDEX_NAME)
        weight_map: dict[str, str] = dict(index.get("weight_map") or {})
        checks["index_parses"] = True
    except Exception as exc:  # pragma: no cover - returned to caller
        weight_map = {}
        errors.append(f"index parse failed: {exc}")

    seen_tensors: set[str] = set()
    router_ok = True
    expert_ok = True
    for shard_name in sorted(set(weight_map.values())):
        try:
            header, _ = _read_safetensors_header(dst / shard_name)
        except Exception as exc:
            errors.append(f"{shard_name}: header read failed: {exc}")
            router_ok = False
            expert_ok = False
            continue
        for key, info in _tensor_items(header):
            seen_tensors.add(key)
            shape = _shape(info)
            if ROUTER_RE.match(key):
                if not shape or shape[0] != keep_experts:
                    router_ok = False
                    errors.append(f"{key}: router rows {shape[:1]} != {keep_experts}")
            if EXPERT_RE.match(key):
                if not shape or shape[0] != keep_experts:
                    expert_ok = False
                    errors.append(f"{key}: expert rows {shape[:1]} != {keep_experts}")

    checks["index_covers_tensors"] = set(weight_map.keys()) == seen_tensors
    if not checks["index_covers_tensors"]:
        missing = sorted(set(weight_map.keys()) - seen_tensors)[:8]
        extra = sorted(seen_tensors - set(weight_map.keys()))[:8]
        errors.append(f"index/header mismatch missing={missing} extra={extra}")
    checks["router_rows_match"] = router_ok
    checks["expert_rows_match"] = expert_ok

    missing_layers = sorted(set(keep_by_layer) - {
        int(match.group(1))
        for key in weight_map
        for match in [ROUTER_RE.match(key)]
        if match is not None
    })
    if missing_layers:
        checks["router_rows_match"] = False
        errors.append(f"missing pruned router layers: {missing_layers[:8]}")

    return {
        "ok": all(checks.values()),
        "checks": checks,
        "errors": errors,
    }


def prune_prequant_qwen_moe(
    src: Path,
    dst: Path,
    keep_experts: int | None,
    *,
    keep_map: Path | None = None,
    require_reviewed_comparison: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    src = src.resolve()
    dst = dst.resolve()
    if dst == src or _is_relative_to(dst, src) or _is_relative_to(src, dst):
        raise RuntimeError(
            f"output directory must be separate from the source model tree: src={src}, dst={dst}"
        )
    index_path = src / INDEX_NAME
    if not index_path.exists():
        raise RuntimeError(f"missing source index: {index_path}")
    config = _load_json(src / "config.json")
    topk = _config_int(config, "num_experts_per_tok", "top_k_experts", "moe_router_topk")
    if dst.exists():
        if not force:
            raise RuntimeError(f"output exists: {dst}")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    index = _load_json(index_path)
    weight_map: dict[str, str] = dict(index.get("weight_map") or {})
    resolved_keep_map = keep_map.resolve() if keep_map is not None else None
    keep_by_layer, original_experts, keep_experts, method = _compute_keep_indices(
        src,
        weight_map,
        keep_experts,
        resolved_keep_map,
        source_topk=topk,
        require_reviewed_comparison=require_reviewed_comparison,
    )
    if keep_experts <= 0:
        raise RuntimeError("keep_experts must be positive")
    if topk and keep_experts < topk:
        raise RuntimeError(f"keep_experts={keep_experts} is below router top-k={topk}")
    if keep_experts >= original_experts:
        raise RuntimeError(
            f"keep_experts={keep_experts} must be less than source experts={original_experts}"
        )

    _copy_sidecars(src, dst)
    shard_sizes: dict[str, int] = {}
    shape_updates: dict[str, list[int]] = {}
    for shard_name in sorted(set(weight_map.values())):
        size, updates = _write_pruned_shard(src / shard_name, dst / shard_name, keep_by_layer)
        shard_sizes[shard_name] = size
        shape_updates.update(updates)

    _write_index(dst, index, shard_sizes)
    _update_config(
        dst,
        keep_experts=keep_experts,
        original_experts=original_experts,
        method=method,
        keep_map=resolved_keep_map,
    )
    reviewed_evidence = (
        _materialize_reviewed_evidence_bundle(src=src, dst=dst, keep_map=resolved_keep_map)
        if resolved_keep_map is not None and require_reviewed_comparison
        else None
    )
    manifest = _manifest(
        src=src,
        dst=dst,
        keep_by_layer=keep_by_layer,
        original_experts=original_experts,
        keep_experts=keep_experts,
        method=method,
        keep_map=resolved_keep_map,
        shape_updates=shape_updates,
    )
    if reviewed_evidence is not None:
        manifest["reviewed_evidence"] = reviewed_evidence["manifest"]
    _write_json(dst / "prune_manifest.json", manifest)
    _write_json(dst / "expert_prune_manifest.json", manifest)
    if resolved_keep_map is not None:
        if reviewed_evidence is not None:
            _write_json(dst / "prune_plan.json", reviewed_evidence["plan"])
        else:
            shutil.copy2(resolved_keep_map, dst / "prune_plan.json")
    else:
        _write_json(dst / "prune_plan.json", manifest)
    _write_json(dst / "source_fingerprint.json", _source_fingerprint(src))
    verification = _verify_pruned_output(
        dst,
        keep_by_layer=keep_by_layer,
        keep_experts=keep_experts,
    )
    _write_json(dst / "verification.json", verification)
    if not verification["ok"]:
        raise RuntimeError(f"pruned output verification failed: {verification['errors'][:4]}")
    return {k: v for k, v in manifest.items() if k != "layers"} | {
        "shard_count": len(shard_sizes),
        "total_size": sum(shard_sizes.values()),
        "verification": verification,
    }


def cmd_prequant_prune_qwen_moe(args) -> None:
    result = prune_prequant_qwen_moe(
        args.src,
        args.dst,
        args.keep_experts,
        keep_map=args.keep_map,
        require_reviewed_comparison=args.require_reviewed_comparison,
        force=args.force,
    )
    if args.json:
        print(json.dumps(result, separators=(",", ":")))
    else:
        print(json.dumps(result, indent=2))


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "prequant-prune-qwen-moe",
        help="Prune raw BF16 Qwen MoE experts before JANG/JANGTQ conversion",
    )
    p.add_argument("src", type=Path, help="Raw HuggingFace Qwen MoE source directory")
    p.add_argument("dst", type=Path, help="Output pruned HuggingFace source directory")
    p.add_argument("--keep-experts", type=int, help="Experts per layer to keep")
    p.add_argument("--keep-map", type=Path, help="Expert Lab/REAP JSON keep map with per-layer keep lists")
    p.add_argument(
        "--require-reviewed-comparison",
        action="store_true",
        help="Require keep-map to embed clean same-suite Expert Lab A/B comparison evidence",
    )
    p.add_argument("--force", action="store_true", help="Replace output directory if it exists")
    p.add_argument("--json", action="store_true", help="Print compact JSON summary")
    p.set_defaults(func=cmd_prequant_prune_qwen_moe)
