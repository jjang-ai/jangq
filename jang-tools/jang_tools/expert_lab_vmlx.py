"""BF16/vMLX Expert Lab runner.

This module is intentionally a small CLI adapter around the installed
``mlx_lm`` runtime. It keeps Expert Lab's behavioral authority on the original
BF16/F16 source model while exporting the same trace shape the Swift UI already
uses for JANGTQ review runs.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import re
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

try:
    from . import __version__ as _JANG_TOOLS_VERSION
except Exception:
    _JANG_TOOLS_VERSION = None


_ACTIVE_TRACE: "ExpertTraceContext | None" = None
_PATCHED_QWEN_SPARSE_MOE = False
_QWEN35_VMLX_MODEL_TYPES = {"qwen3_5_moe", "qwen3_5_moe_text"}


class VMLXSourceConfigError(ValueError):
    """Raised when a selected BF16/F16 source cannot be loaded by the vMLX runner."""


def _positive_int(raw: Any) -> int | None:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _package_version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def _runtime_versions() -> dict[str, str | None]:
    return {
        "jang_tools_version": _JANG_TOOLS_VERSION or _package_version("jang"),
        "mlx_version": _package_version("mlx"),
        "mlx_lm_version": _package_version("mlx-lm") or _package_version("mlx_lm"),
        "mlx_vlm_version": _package_version("mlx-vlm") or _package_version("mlx_vlm"),
    }


def _nested_text_config(config: dict[str, Any]) -> dict[str, Any]:
    text_config = config.get("text_config")
    return text_config if isinstance(text_config, dict) else config


def _config_string_value(data: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _positive_config_int(data: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = _positive_int(data.get(key))
        if value is not None:
            return value
    return None


def _vmlx_qwen_config_issue(model_path: Path) -> str | None:
    config_path = model_path / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return f"BF16/vMLX source config is missing: {config_path}"
    except json.JSONDecodeError as exc:
        return f"BF16/vMLX source config is unreadable: {config_path}: {exc}"
    if not isinstance(config, dict):
        return f"BF16/vMLX source config must be a JSON object: {config_path}"

    text_config = _nested_text_config(config)
    model_type = (_config_string_value(text_config, "model_type") or "").lower()
    if model_type not in _QWEN35_VMLX_MODEL_TYPES:
        return None

    required_positive_fields = [
        ("hidden_size", ("hidden_size",)),
        ("num_hidden_layers", ("num_hidden_layers", "n_layer", "num_layers")),
        ("num_attention_heads", ("num_attention_heads",)),
        ("num_key_value_heads", ("num_key_value_heads",)),
        ("vocab_size", ("vocab_size",)),
        ("num_experts", ("n_routed_experts", "num_experts", "num_local_experts")),
        ("num_experts_per_tok", ("num_experts_per_tok", "top_k_experts", "moe_router_topk")),
        ("moe_intermediate_size", ("moe_intermediate_size",)),
        ("shared_expert_intermediate_size", ("shared_expert_intermediate_size",)),
    ]
    for label, keys in required_positive_fields:
        if _positive_config_int(text_config, *keys) is None:
            return (
                "BF16/vMLX tracing requires a complete Qwen3.5 MoE source config; "
                f"{label} is missing or zero in {config_path}. Reopen the original "
                "BF16/F16 source rather than a compact UI fixture."
            )

    experts = _positive_config_int(text_config, "n_routed_experts", "num_experts", "num_local_experts")
    top_k = _positive_config_int(text_config, "num_experts_per_tok", "top_k_experts", "moe_router_topk")
    if experts is not None and top_k is not None and top_k > experts:
        return (
            "BF16/vMLX tracing requires num_experts_per_tok to be no larger "
            f"than the routed expert count in {config_path}."
        )
    return None


def _validate_vmlx_source_config(model_path: Path) -> None:
    issue = _vmlx_qwen_config_issue(model_path)
    if issue:
        raise VMLXSourceConfigError(issue)


@dataclass
class ExpertTraceContext:
    disabled_by_layer: dict[int, set[int]] = field(default_factory=dict)
    top_k_override: int | None = None
    emit_token_trace: bool = True
    max_trace_tokens: int = 32768
    token_trace: list[dict[str, Any]] = field(default_factory=list)
    layer_token_positions: dict[int, set[int]] = field(default_factory=lambda: defaultdict(set))
    layer_hits: dict[int, Counter[int]] = field(default_factory=lambda: defaultdict(Counter))
    layer_mass: dict[int, defaultdict[int, float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(float))
    )
    _last_layer: int | None = None
    _current_forward_base: int = 0
    _total_positions: int = 0

    def disabled_for(self, layer: int) -> set[int]:
        return self.disabled_by_layer.get(layer, set())

    def effective_top_k(self, trained_top_k: int) -> int:
        if self.top_k_override and self.top_k_override > 0:
            return max(1, min(int(self.top_k_override), int(trained_top_k)))
        return int(trained_top_k)

    def base_for_layer_call(self, layer: int, sequence_length: int) -> int:
        if self._last_layer is None or layer <= self._last_layer:
            self._current_forward_base = self._total_positions
            self._total_positions += int(sequence_length)
        self._last_layer = layer
        return self._current_forward_base

    def record(
        self,
        *,
        layer: int,
        sequence_offset: int,
        selected: list[list[int]],
        scores: list[list[float]],
        disabled: list[int],
        effective_top_k: int,
    ) -> None:
        for position, experts in enumerate(selected):
            token_index = sequence_offset + position
            self.layer_token_positions[layer].add(token_index)
            score_row = scores[position] if position < len(scores) else []
            for slot, expert in enumerate(experts):
                score = float(score_row[slot]) if slot < len(score_row) else 1.0
                self.layer_hits[layer][int(expert)] += 1
                self.layer_mass[layer][int(expert)] += score
            if self.emit_token_trace and len(self.token_trace) < self.max_trace_tokens:
                entropy = _selected_entropy(score_row)
                self.token_trace.append(
                    {
                        "token_index": token_index,
                        "layer": int(layer),
                        "selected_experts": [int(e) for e in experts],
                        "scores": [float(s) for s in score_row],
                        "disabled_experts": disabled,
                        "effective_top_k": int(effective_top_k),
                        "entropy": entropy,
                    }
                )

    def layer_stats(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for layer in sorted(self.layer_token_positions):
            hits = self.layer_hits[layer]
            mass = self.layer_mass[layer]
            rows.append(
                {
                    "layer": int(layer),
                    "token_count": len(self.layer_token_positions[layer]),
                    "hit_counts": {str(k): int(v) for k, v in sorted(hits.items())},
                    "probability_mass": {
                        str(k): float(v) for k, v in sorted(mass.items())
                    },
                }
            )
        return rows


@dataclass(frozen=True)
class MOELayerHook:
    layer: int
    num_experts: int
    trained_top_k: int


def _selected_entropy(scores: list[float]) -> float | None:
    total = sum(max(float(s), 0.0) for s in scores)
    if total <= 0:
        return None
    entropy = 0.0
    for raw in scores:
        p = max(float(raw), 0.0) / total
        if p > 0:
            entropy -= p * math.log(p)
    return float(entropy)


def _coerce_non_negative_int(raw: Any, label: str) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{label} must be non-negative, got {value}")
    return value


def _coerce_positive_int(raw: Any, label: str) -> int:
    value = _coerce_non_negative_int(raw, label)
    if value <= 0:
        raise ValueError(f"{label} must be positive, got {value}")
    return value


def _coerce_int_set_map(raw: Any, field_name: str) -> dict[int, set[int]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{field_name} must be an object keyed by layer")
    out: dict[int, set[int]] = {}
    for key, value in raw.items():
        layer = _coerce_non_negative_int(key, f"{field_name} layer")
        if not isinstance(value, list):
            raise ValueError(f"{field_name}[{layer}] must be a list of expert IDs")
        experts: set[int] = set()
        for index, expert in enumerate(value):
            experts.add(
                _coerce_non_negative_int(
                    expert,
                    f"{field_name}[{layer}][{index}]",
                )
            )
        out[layer] = experts
    return out


def _read_mask(path: str | None) -> tuple[dict[int, set[int]], int | None]:
    if not path:
        return {}, None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    disabled: dict[int, set[int]] = {}
    for field_name in ("disabled_by_layer", "layers", "disabledExpertsByLayer"):
        if field_name in data:
            disabled = _coerce_int_set_map(data.get(field_name), field_name)
            break
    top_k = data.get("top_k_override") or data.get("topKOverride")
    return disabled, _coerce_positive_int(top_k, "top_k_override") if top_k else None


def _qwen_sparse_moe_hook_targets() -> list[type[Any]]:
    from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

    targets: list[type[Any]] = [Qwen3NextSparseMoeBlock]
    try:
        from mlx_lm.models import qwen3_5
    except Exception:
        qwen3_5 = None
    qwen35_block = getattr(qwen3_5, "SparseMoeBlock", None)
    if isinstance(qwen35_block, type) and qwen35_block not in targets:
        targets.append(qwen35_block)
    return targets


def _patch_qwen_sparse_moe() -> None:
    global _PATCHED_QWEN_SPARSE_MOE
    if _PATCHED_QWEN_SPARSE_MOE:
        return

    import mlx.core as mx
    from mlx.nn.layers.distributed import sum_gradients

    def traced_call(self, x):
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        context = _ACTIVE_TRACE
        layer = int(getattr(self, "_expert_lab_layer_idx", -1))
        disabled = context.disabled_for(layer) if context and layer >= 0 else set()
        trained_top_k = int(self.top_k)
        effective_top_k = (
            context.effective_top_k(trained_top_k) if context else trained_top_k
        )
        if disabled and len(disabled) > int(self.num_experts) - effective_top_k:
            raise RuntimeError(
                f"expert mask leaves fewer than top-k experts at layer {layer}: "
                f"disabled={len(disabled)} experts={self.num_experts} top_k={effective_top_k}"
            )

        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)
        if disabled:
            available = mx.array(
                [i not in disabled for i in range(int(self.num_experts))],
                dtype=mx.bool_,
            )
            gates = mx.where(available, gates, 0)
            denom = gates.sum(axis=-1, keepdims=True)
            gates = mx.where(denom > 0, gates / denom, gates)

        inds = mx.argpartition(gates, kth=-effective_top_k, axis=-1)[
            ..., -effective_top_k:
        ]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
        y = y + shared_y

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        if context and layer >= 0:
            import numpy as np

            # Generation uses batch size 1; keep the code explicit so accidental
            # batching fails visibly instead of silently mixing prompt evidence.
            inds_np = np.array(inds.astype(mx.int32))
            scores_np = np.array(scores.astype(mx.float32))
            if inds_np.ndim != 3 or inds_np.shape[0] != 1:
                raise RuntimeError(
                    f"Expert Lab vMLX runner expected batch=1 router output, got {inds_np.shape}"
                )
            base = context.base_for_layer_call(layer, int(inds_np.shape[1]))
            context.record(
                layer=layer,
                sequence_offset=base,
                selected=inds_np[0].astype(int).tolist(),
                scores=scores_np[0].astype(float).tolist(),
                disabled=sorted(int(v) for v in disabled),
                effective_top_k=effective_top_k,
            )

        return y

    for target in _qwen_sparse_moe_hook_targets():
        target.__call__ = traced_call
    _PATCHED_QWEN_SPARSE_MOE = True


def _install_layer_indices(model: Any) -> dict[int, MOELayerHook]:
    seen: set[int] = set()
    hooks: dict[int, MOELayerHook] = {}
    candidates = [
        model,
        getattr(model, "language_model", None),
        getattr(model, "model", None),
        getattr(getattr(model, "language_model", None), "model", None),
    ]
    for owner in candidates:
        layers = getattr(owner, "layers", None)
        if not isinstance(layers, list):
            continue
        for index, layer in enumerate(layers):
            block = getattr(layer, "mlp", None)
            if block is None or not hasattr(block, "switch_mlp") or not hasattr(block, "gate"):
                continue
            ident = id(block)
            if ident in seen:
                continue
            seen.add(ident)
            setattr(block, "_expert_lab_layer_idx", int(index))
            num_experts = _positive_int(getattr(block, "num_experts", None))
            trained_top_k = _positive_int(getattr(block, "top_k", None))
            if num_experts is None or trained_top_k is None:
                raise RuntimeError(
                    f"MoE router block at layer {index} is missing expert/top-k metadata"
                )
            hooks[int(index)] = MOELayerHook(
                layer=int(index),
                num_experts=int(num_experts),
                trained_top_k=int(trained_top_k),
            )
    return hooks


def _validate_mask_targets(
    disabled_by_layer: dict[int, set[int]],
    top_k_override: int | None,
    layer_hooks: dict[int, MOELayerHook],
) -> None:
    if not disabled_by_layer and not top_k_override:
        return
    if not layer_hooks:
        raise RuntimeError("Cannot apply a BF16/vMLX mask before MoE router hooks are installed")

    hooked_layers = set(layer_hooks)
    for layer, disabled in sorted(disabled_by_layer.items()):
        hook = layer_hooks.get(layer)
        if hook is None:
            known = _preview_ints(sorted(hooked_layers))
            raise ValueError(
                f"BF16/vMLX mask targets unknown MoE layer {layer}; hooked layers: {known}"
            )
        invalid = sorted(expert for expert in disabled if expert >= hook.num_experts)
        if invalid:
            raise ValueError(
                f"BF16/vMLX mask targets expert outside layer {layer} width "
                f"{hook.num_experts}: {_preview_ints(invalid)}"
            )
        effective_top_k = (
            max(1, min(int(top_k_override), hook.trained_top_k))
            if top_k_override
            else hook.trained_top_k
        )
        if len(disabled) > hook.num_experts - effective_top_k:
            raise RuntimeError(
                f"expert mask leaves fewer than top-k experts at layer {layer}: "
                f"disabled={len(disabled)} experts={hook.num_experts} top_k={effective_top_k}"
            )


def _trace_coverage_issue(
    context: ExpertTraceContext,
    layer_hooks: dict[int, MOELayerHook],
) -> str | None:
    recorded_layers = set(context.layer_token_positions)
    missing = sorted(set(layer_hooks) - recorded_layers)
    if missing:
        return (
            "BF16/vMLX generation did not record routed-layer evidence for hooked "
            f"layers: {_preview_ints(missing)}"
        )
    for layer in sorted(layer_hooks):
        if not context.layer_token_positions.get(layer):
            return f"BF16/vMLX generation layer {layer} is missing token-position depth"
        if not context.layer_hits.get(layer):
            return f"BF16/vMLX generation layer {layer} is missing expert hit counts"
        if not context.layer_mass.get(layer):
            return f"BF16/vMLX generation layer {layer} is missing expert gate-mass evidence"
    return None


def _mask_application_issue(
    context: ExpertTraceContext,
    disabled_by_layer: dict[int, set[int]],
) -> str | None:
    for layer, disabled in sorted(disabled_by_layer.items()):
        if not disabled:
            continue
        hits = context.layer_hits.get(layer)
        if not hits:
            return f"BF16/vMLX masked layer {layer} produced no routing evidence"
        leaked = sorted(set(hits).intersection(disabled))
        if leaked:
            return (
                f"BF16/vMLX mask failed at layer {layer}; disabled experts were selected: "
                f"{_preview_ints(leaked)}"
            )
    return None


def _token_trace_evidence_issue(
    context: ExpertTraceContext,
    *,
    emit_token_trace: bool,
    disabled_by_layer: dict[int, set[int]],
    top_k_override: int | None = None,
) -> str | None:
    mask_applied = bool(disabled_by_layer or top_k_override)
    if not emit_token_trace:
        if mask_applied:
            return (
                "BF16/vMLX masked generation requires token_trace routing evidence; "
                "pass --emit-token-trace"
            )
        return None

    layer_stats = context.layer_stats()
    expected_routes = sum(int(row.get("token_count") or 0) for row in layer_stats)
    trace_count = len(context.token_trace)
    if expected_routes <= 0:
        return "BF16/vMLX generation is missing routed layer-token records"
    if trace_count <= 0:
        return "BF16/vMLX generation is missing token_trace routing evidence"
    if trace_count != expected_routes:
        return (
            f"BF16/vMLX token_trace covers {trace_count} of "
            f"{expected_routes} routed layer-token records; increase --max-trace-tokens"
        )
    if disabled_by_layer:
        leakage = _disabled_expert_leakage_issue(
            layer_stats=layer_stats,
            token_trace=context.token_trace,
            disabled_by_layer=disabled_by_layer,
        )
        if leakage is not None:
            return f"BF16/vMLX masked generation token_trace evidence failed: {leakage}"
    return None


def _preview_ints(values: list[int], limit: int = 8) -> str:
    if not values:
        return "none"
    prefix = ", ".join(str(value) for value in values[:limit])
    if len(values) > limit:
        return f"{prefix}, ... (+{len(values) - limit} more)"
    return prefix



def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    text_config = config.get("text_config")
    return text_config if isinstance(text_config, dict) else config


def _config_positive_int(config: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = _positive_int(config.get(key))
        if value is not None:
            return value
    return None


def _config_int_set(config: dict[str, Any], key: str) -> set[int]:
    raw = config.get(key)
    if not isinstance(raw, list):
        return set()
    values: set[int] = set()
    for item in raw:
        try:
            value = int(item)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            values.add(value)
    return values


def _expected_moe_layer_count(model_path: Path) -> int | None:
    config_path = model_path / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(config, dict):
        return None

    text_config = _text_config(config)
    layer_count = _config_positive_int(
        text_config,
        "num_hidden_layers",
        "n_layer",
        "num_layers",
    )
    expert_count = _config_positive_int(
        text_config,
        "num_experts",
        "n_routed_experts",
        "num_local_experts",
    )
    moe_width = _config_positive_int(text_config, "moe_intermediate_size")
    if layer_count is None or (expert_count is None and moe_width is None):
        return None

    try:
        first_dense = int(text_config.get("first_k_dense_replace") or 0)
    except (TypeError, ValueError):
        first_dense = 0
    first_dense = max(0, min(first_dense, layer_count))
    sparse_step = _config_positive_int(text_config, "decoder_sparse_step") or 1
    mlp_only_layers = _config_int_set(text_config, "mlp_only_layers")

    routed_layers = [
        layer
        for layer in range(first_dense, layer_count)
        if (layer - first_dense) % sparse_step == 0 and layer not in mlp_only_layers
    ]
    return len(routed_layers) if routed_layers else None


def _validate_moe_hook_coverage(model_path: Path, hooked_layers: int) -> int | None:
    expected = _expected_moe_layer_count(model_path)
    if expected is not None and hooked_layers < expected:
        raise RuntimeError(
            "Incomplete vMLX MoE router hook coverage for BF16 Expert Lab: "
            f"hooked {hooked_layers} of {expected} config-routed layers."
        )
    return expected


def _load_suite(path: str) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        prompt = json.loads(line)
        if not isinstance(prompt, dict):
            raise ValueError(f"suite line {line_number} must be a JSON object")
        if "prompt" not in prompt and "text" in prompt:
            prompt["prompt"] = prompt["text"]
        prompt_id = str(prompt.get("id") or "").strip()
        if not prompt_id:
            raise ValueError(f"suite line {line_number} is missing a non-empty prompt id")
        if prompt_id in seen_ids:
            raise ValueError(f"suite line {line_number} duplicate prompt id {prompt_id!r}")
        prompt_text = str(prompt.get("prompt") or "").strip()
        if not prompt_text:
            raise ValueError(f"suite line {line_number} prompt {prompt_id!r} has empty text")
        seen_ids.add(prompt_id)
        prompts.append(prompt)
    if not prompts:
        raise ValueError("prompt suite is empty")
    return prompts


def _prompt_tokens(tokenizer: Any, prompt: str) -> list[int] | str:
    messages = [{"role": "user", "content": prompt}]
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_template):
        try:
            tokens = apply_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            if isinstance(tokens, dict):
                tokens = tokens.get("input_ids", tokens)
            return tokens
        except Exception:
            pass
    return prompt


def _mask_info(
    disabled_by_layer: dict[int, set[int]],
    top_k_override: int | None,
) -> dict[str, Any]:
    disabled_count = sum(len(values) for values in disabled_by_layer.values())
    return {
        "mask_applied": bool(disabled_count > 0 or top_k_override),
        "masked_layer_count": len(disabled_by_layer),
        "disabled_expert_count": disabled_count,
        "top_k_override": int(top_k_override) if top_k_override else None,
        "disabled_by_layer": {
            str(layer): sorted(int(expert) for expert in experts)
            for layer, experts in sorted(disabled_by_layer.items())
        },
    }


def _runtime_info(
    *,
    model_path: Path,
    hooked_layers: int,
    expected_moe_layers: int | None,
    disabled_by_layer: dict[int, set[int]],
    top_k_override: int | None,
) -> dict[str, Any]:
    import mlx.core as mx

    device = mx.default_device()
    mask = _mask_info(disabled_by_layer, top_k_override)
    return {
        "backend": "vmlx",
        "runtime_mode": "bf16_vmlx",
        "device_name": str(device),
        "runtime_metal_enabled": "gpu" in str(device).lower(),
        **_runtime_versions(),
        "source_model_path": str(model_path),
        "hooked_moe_layers": int(hooked_layers),
        "expected_moe_layers": int(expected_moe_layers) if expected_moe_layers else None,
        "hook_coverage_complete": (
            True if expected_moe_layers is None else hooked_layers >= expected_moe_layers
        ),
        "mask_applied": mask["mask_applied"],
        "masked_layer_count": mask["masked_layer_count"],
        "disabled_expert_count": mask["disabled_expert_count"],
        "top_k_override": mask["top_k_override"],
        "mask": mask,
        "notes": [
            "Original BF16/F16 source loaded through mlx_lm/vMLX.",
            "Qwen MoE router hooks record selected experts and apply masks before top-k.",
            (
                f"Hooked {hooked_layers} of {expected_moe_layers} config-routed MoE layers."
                if expected_moe_layers
                else f"Hooked {hooked_layers} Qwen MoE router layers."
            ),
        ],
    }


def _iso8601_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    )
    path.write_text(text + ("\n" if rows else ""), encoding="utf-8")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _generation_settings(
    prompt: dict[str, Any],
    args: argparse.Namespace,
    top_k_override: int | None,
) -> dict[str, Any]:
    prompt_max_tokens = prompt.get("max_new_tokens")
    prompt_temperature = prompt.get("temperature")
    return {
        "max_tokens": int(prompt_max_tokens or args.max_tokens),
        "temperature": float(
            prompt_temperature
            if prompt_temperature is not None
            else args.temperature
        ),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
        "top_k_override": int(top_k_override) if top_k_override else None,
        "prompt_max_new_tokens": int(prompt_max_tokens) if prompt_max_tokens else None,
        "prompt_temperature": (
            float(prompt_temperature) if prompt_temperature is not None else None
        ),
    }


def _read_jsonl_objects(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} line {line_number} is not valid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{label} line {line_number} must be a JSON object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{label} is empty")
    return rows


def _generations_jsonl_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "generations.jsonl"
    return candidate


def _first_value(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _string_value(raw: Any) -> str | None:
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _int_value(raw: Any) -> int | None:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value


def _float_value(raw: Any) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _prompt_id_from_suite(row: dict[str, Any], label: str = "suite row") -> str:
    prompt_id = _string_value(_first_value(row, "id", "prompt_id", "promptID"))
    if prompt_id is None:
        raise ValueError(f"{label} is missing a prompt id")
    return prompt_id


def _prompt_id_from_generation(row: dict[str, Any], label: str) -> str:
    prompt = row.get("prompt")
    if not isinstance(prompt, dict):
        raise ValueError(f"{label} is missing embedded prompt metadata")
    return _prompt_id_from_suite(prompt, f"{label} embedded prompt")


def _generation_result(row: dict[str, Any], label: str) -> dict[str, Any]:
    result = row.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"{label} is missing generation result")
    return result


def _generation_ids(rows: list[dict[str, Any]], label: str) -> list[str]:
    ids = [_prompt_id_from_generation(row, f"{label} line {index + 1}") for index, row in enumerate(rows)]
    if len(set(ids)) < len(ids):
        raise ValueError(f"{label} contains duplicate prompt ids")
    return ids


def _suite_decode_settings(row: dict[str, Any]) -> dict[str, int | float]:
    settings: dict[str, int | float] = {}
    max_tokens = _int_value(_first_value(row, "max_new_tokens", "maxTokens", "max_tokens"))
    if max_tokens is not None:
        settings["max_tokens"] = max_tokens
    temperature = _float_value(row.get("temperature"))
    if temperature is not None:
        settings["temperature"] = temperature
    return settings


def _result_generation_settings(result: dict[str, Any], label: str) -> dict[str, int | float]:
    settings = result.get("generation_settings")
    if not isinstance(settings, dict):
        raise ValueError(f"{label} is missing decode settings evidence")
    max_tokens = _int_value(_first_value(settings, "max_tokens", "maxTokens"))
    temperature = _float_value(settings.get("temperature"))
    top_p = _float_value(_first_value(settings, "top_p", "topP"))
    top_k = _int_value(_first_value(settings, "top_k", "topK"))
    if max_tokens is None or max_tokens <= 0 or temperature is None or top_p is None or top_k is None:
        raise ValueError(f"{label} has unreadable decode settings evidence")
    return {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }


def _validate_generation_settings(
    *,
    prompt_id: str,
    suite_row: dict[str, Any],
    baseline_result: dict[str, Any],
    masked_result: dict[str, Any],
) -> tuple[dict[str, int | float], dict[str, int | float]]:
    baseline_settings = _result_generation_settings(baseline_result, f"baseline prompt {prompt_id}")
    masked_settings = _result_generation_settings(masked_result, f"masked prompt {prompt_id}")
    suite_settings = _suite_decode_settings(suite_row)

    for key in ("max_tokens", "temperature", "top_p", "top_k"):
        if baseline_settings[key] != masked_settings[key]:
            raise ValueError(
                f"prompt {prompt_id} baseline/masked generation {key} does not match"
            )

    for key, expected in suite_settings.items():
        if baseline_settings[key] != expected:
            raise ValueError(
                f"prompt {prompt_id} baseline generation {key} does not match suite.jsonl"
            )
        if masked_settings[key] != expected:
            raise ValueError(
                f"prompt {prompt_id} masked generation {key} does not match suite.jsonl"
            )

    return baseline_settings, masked_settings


def _reviewed_prune_semantic_domains(row: dict[str, Any]) -> set[str]:
    from .prequant_prune_qwen_moe import _suite_semantic_domains

    return _suite_semantic_domains(row)


def _required_reviewed_prune_semantic_domains() -> set[str]:
    from .prequant_prune_qwen_moe import REQUIRED_REVIEWED_PRUNE_SEMANTIC_DOMAINS

    return set(REQUIRED_REVIEWED_PRUNE_SEMANTIC_DOMAINS)


def _suite_semantic_coverage(
    suite_rows: list[dict[str, Any]],
) -> tuple[dict[str, list[str]], list[str], list[str]]:
    per_prompt: dict[str, list[str]] = {}
    coverage: set[str] = set()
    for row in suite_rows:
        prompt_id = _prompt_id_from_suite(row)
        domains = sorted(_reviewed_prune_semantic_domains(row))
        per_prompt[prompt_id] = domains
        coverage.update(domains)
    missing = sorted(_required_reviewed_prune_semantic_domains().difference(coverage))
    return per_prompt, sorted(coverage), missing


def _expected_kind(row: dict[str, Any]) -> str:
    raw = _string_value(_first_value(row, "expectedKind", "expected_kind")) or "freeform"
    if raw not in {"freeform", "exact", "regex", "unit_test", "judge"}:
        return "freeform"
    return raw


def _expected_value(row: dict[str, Any]) -> str | None:
    return _string_value(row.get("expected"))


def _validator_object(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("validator", "expected_behavior", "expectedBehavior"):
        value = row.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _validator_expected(row: dict[str, Any], validator: dict[str, Any]) -> str | None:
    for key in ("expected", "pattern", "value", "contains"):
        raw = validator.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw
    return _expected_value(row)


def _validator_kind(row: dict[str, Any], validator: dict[str, Any]) -> str:
    raw = (
        _string_value(
            _first_value(
                validator,
                "kind",
                "type",
                "expectedKind",
                "expected_kind",
            )
        )
        or _expected_kind(row)
    )
    normalized = raw.strip().lower().replace("-", "_")
    aliases = {
        "normalized_exact": "exact",
        "equals": "exact",
        "match": "exact",
        "regexp": "regex",
        "unit_test_expected_regex": "unit_test",
        "substring": "contains",
        "json_schema": "json",
    }
    return aliases.get(normalized, normalized)


def _validator_spec(row: dict[str, Any]) -> dict[str, Any]:
    validator = _validator_object(row)
    kind = _validator_kind(row, validator)
    expected = _validator_expected(row, validator)
    has_expected = expected is not None and expected.strip() != ""
    has_json_schema = isinstance(
        _first_value(validator, "schema", "json_schema", "jsonSchema"),
        dict,
    )
    mechanically_scored = (
        kind in {"exact", "regex", "unit_test", "contains"} and has_expected
    ) or (kind == "json" and (has_expected or has_json_schema))
    if kind in {"freeform", "judge"}:
        mechanically_scored = False
    reason: str | None = None
    if not mechanically_scored:
        if kind in {"freeform", "judge"}:
            reason = f"{kind} requires external rubric/judge evidence"
        else:
            reason = "validator is missing expected behavior metadata"
    return {
        "schema": "jang-expert-lab-validator-v1",
        "kind": kind,
        "source": "validator" if validator else "suite_expected",
        "available": mechanically_scored,
        "expected": expected,
        "reason": reason,
    }


def _evaluation_passed(text: str, validator: dict[str, Any]) -> bool | None:
    if not validator.get("available"):
        return None
    expected = _string_value(validator.get("expected"))
    kind = str(validator.get("kind") or "freeform")
    if kind == "exact":
        if expected is None:
            return None
        return text.strip() == expected.strip()
    if kind in {"regex", "unit_test"}:
        if expected is None:
            return None
        try:
            return re.search(expected, text, flags=re.MULTILINE) is not None
        except re.error:
            return None
    if kind == "contains":
        if expected is None:
            return None
        return expected.strip() in text
    if kind == "json":
        try:
            parsed = json.loads(_extract_json_object(text))
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        if expected is None:
            return isinstance(parsed, dict)
        try:
            expected_obj = json.loads(expected)
        except json.JSONDecodeError:
            return expected.strip() in text
        if isinstance(expected_obj, dict):
            required = expected_obj.get("required")
            if isinstance(required, list):
                return all(isinstance(key, str) and key in parsed for key in required)
        return True
    return None


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return stripped


def _prompt_classification(
    *,
    baseline_passed: bool | None,
    candidate_passed: bool | None,
) -> str:
    if baseline_passed is None or candidate_passed is None:
        return "inconclusive"
    if not baseline_passed:
        return "baseline_invalid"
    if candidate_passed:
        return "preserved"
    return "degraded"


def _text_delta(baseline: str, masked: str) -> float:
    if baseline == masked:
        return 0.0
    ratio = difflib.SequenceMatcher(a=baseline, b=masked).ratio()
    return float(max(0.0, min(1.0, 1.0 - ratio)))


def _regression_severity(
    *,
    classification: str,
    text_delta: float,
) -> str:
    if classification == "degraded":
        return "critical"
    if text_delta > 0.20:
        return "watch"
    return "none"


def _risk_label(
    *,
    classification: str,
    severity: str,
) -> str:
    if classification == "degraded":
        return "regression"
    if classification == "baseline_invalid":
        return "baseline_invalid"
    if classification == "inconclusive":
        return "inconclusive"
    if severity == "watch":
        return "watch"
    return "preserved"


def _classification_counts(records: list[dict[str, Any]], field: str = "promptClassification") -> dict[str, int]:
    counts = {key: 0 for key in ("baseline_invalid", "preserved", "degraded", "inconclusive")}
    for record in records:
        classification = str(record.get(field) or "inconclusive")
        counts[classification] = counts.get(classification, 0) + 1
    return counts


def _semantic_coverage_for_records(records: list[dict[str, Any]]) -> list[str]:
    coverage: set[str] = set()
    for record in records:
        semantic = record.get("semanticDomains")
        if isinstance(semantic, list):
            coverage.update(str(value) for value in semantic if str(value))
        else:
            domain = _string_value(record.get("domain"))
            if domain:
                coverage.add(domain)
    return sorted(coverage)


def _runtime_info_from_result(result: dict[str, Any], label: str) -> dict[str, Any]:
    runtime = result.get("runtime_info")
    if not isinstance(runtime, dict):
        raise ValueError(f"{label} is missing BF16/vMLX runtime_info")
    return runtime


def _runtime_common_key(runtime: dict[str, Any]) -> tuple[Any, ...]:
    return (
        runtime.get("runtime_mode"),
        runtime.get("backend"),
        runtime.get("device_name"),
        runtime.get("runtime_metal_enabled"),
        runtime.get("jang_tools_version"),
        runtime.get("mlx_version"),
        runtime.get("mlx_lm_version"),
        runtime.get("mlx_vlm_version"),
        runtime.get("source_model_path"),
        runtime.get("hooked_moe_layers"),
        runtime.get("expected_moe_layers"),
        runtime.get("hook_coverage_complete"),
    )


def _validate_runtime_info(
    runtime: dict[str, Any],
    *,
    label: str,
    require_mask: bool,
    expected_disabled_count: int,
) -> None:
    if runtime.get("runtime_mode") != "bf16_vmlx":
        raise ValueError(f"{label} did not record BF16/vMLX runtime evidence")
    if runtime.get("backend") != "vmlx":
        raise ValueError(f"{label} did not record vMLX backend evidence")
    if not _string_value(runtime.get("device_name")):
        raise ValueError(f"{label} is missing runtime device evidence")
    if runtime.get("runtime_metal_enabled") is not True:
        raise ValueError(f"{label} did not record a Metal runtime")
    if (
        not _string_value(runtime.get("jang_tools_version"))
        or not _string_value(runtime.get("mlx_version"))
        or not _string_value(runtime.get("mlx_lm_version"))
    ):
        raise ValueError(f"{label} is missing vMLX package version evidence")
    if not _string_value(runtime.get("source_model_path")):
        raise ValueError(f"{label} is missing source model path evidence")
    if runtime.get("hook_coverage_complete") is not True:
        raise ValueError(f"{label} recorded incomplete vMLX routed-layer hook coverage")
    hooked_layers = _int_value(runtime.get("hooked_moe_layers"))
    if hooked_layers is None or hooked_layers <= 0:
        raise ValueError(f"{label} is missing vMLX routed-layer hook evidence")
    expected_layers = _int_value(runtime.get("expected_moe_layers"))
    if expected_layers is not None and hooked_layers < expected_layers:
        raise ValueError(
            f"{label} vMLX hook coverage {hooked_layers} of {expected_layers} config-routed layers"
        )
    if require_mask:
        if runtime.get("mask_applied") is not True:
            raise ValueError(f"{label} did not record an applied BF16/vMLX mask")
        disabled_count = _int_value(runtime.get("disabled_expert_count"))
        if disabled_count is None or disabled_count <= 0:
            raise ValueError(f"{label} did not record disabled expert evidence")
        if expected_disabled_count and disabled_count != expected_disabled_count:
            raise ValueError(
                f"{label} disabled expert count {disabled_count} does not match mask {expected_disabled_count}"
            )
    elif runtime.get("mask_applied") is True:
        raise ValueError(f"{label} unexpectedly recorded an applied mask")


def _validate_layer_stats(
    stats: Any,
    *,
    runtime: dict[str, Any],
    label: str,
) -> int:
    if not isinstance(stats, list) or not stats:
        raise ValueError(f"{label} is missing routed-layer stats")
    seen_layers: set[int] = set()
    total_tokens = 0
    for index, row in enumerate(stats):
        if not isinstance(row, dict):
            raise ValueError(f"{label} layer_stats[{index}] must be an object")
        layer = _int_value(row.get("layer"))
        if layer is None or layer < 0:
            raise ValueError(f"{label} layer_stats[{index}] has an invalid layer")
        if layer in seen_layers:
            raise ValueError(f"{label} routed-layer stats contain duplicate layer {layer}")
        seen_layers.add(layer)
        token_count = _int_value(_first_value(row, "token_count", "tokenCount"))
        if token_count is None or token_count <= 0:
            raise ValueError(f"{label} routed-layer stats are missing token-position depth")
        total_tokens += token_count
        hit_counts = _first_value(row, "hit_counts", "hitCounts")
        if not isinstance(hit_counts, dict) or not hit_counts:
            raise ValueError(f"{label} routed-layer stats are missing expert hit counts")
        probability_mass = _first_value(row, "probability_mass", "probabilityMass")
        if not isinstance(probability_mass, dict) or not probability_mass:
            raise ValueError(f"{label} routed-layer stats are missing expert gate-mass evidence")

    hooked_layers = _int_value(runtime.get("hooked_moe_layers"))
    expected_layers = _int_value(runtime.get("expected_moe_layers"))
    required_layers = expected_layers or hooked_layers
    if required_layers is not None and len(seen_layers) < required_layers:
        raise ValueError(
            f"{label} routed-layer stats cover {len(seen_layers)} of {required_layers} layers"
        )
    return total_tokens


def _token_trace_from_result(
    result: dict[str, Any],
    *,
    expected_route_records: int,
    label: str,
) -> list[dict[str, Any]]:
    trace = result.get("token_trace")
    if not isinstance(trace, list) or not trace:
        raise ValueError(f"{label} is missing token_trace routing evidence")
    if len(trace) != expected_route_records:
        raise ValueError(
            f"{label} token_trace has {len(trace)} rows for "
            f"{expected_route_records} routed layer-token records"
        )
    for index, row in enumerate(trace):
        if not isinstance(row, dict):
            raise ValueError(f"{label} token_trace[{index}] must be an object")
        if _int_value(row.get("layer")) is None or _int_value(row.get("token_index")) is None:
            raise ValueError(f"{label} token_trace[{index}] is missing layer/token evidence")
        selected = row.get("selected_experts")
        if not isinstance(selected, list) or not selected:
            raise ValueError(f"{label} token_trace[{index}] is missing selected experts")
    return trace


def _disabled_expert_leakage_issue(
    *,
    layer_stats: list[dict[str, Any]],
    token_trace: list[dict[str, Any]],
    disabled_by_layer: dict[int, set[int]],
) -> str | None:
    if not disabled_by_layer:
        return None
    for row in layer_stats:
        layer = _int_value(row.get("layer"))
        if layer is None or layer not in disabled_by_layer:
            continue
        hit_counts = _first_value(row, "hit_counts", "hitCounts")
        if not isinstance(hit_counts, dict):
            continue
        hit_experts = {_int_value(expert) for expert in hit_counts}
        leaked = sorted(
            expert for expert in disabled_by_layer[layer] if expert in hit_experts
        )
        if leaked:
            return (
                f"disabled experts leaked into masked layer {layer} stats: "
                f"{_preview_ints(leaked)}"
            )

    evidence_layers: set[int] = set()
    for row in token_trace:
        layer = _int_value(row.get("layer"))
        if layer is None or layer not in disabled_by_layer:
            continue
        selected = {
            int(expert)
            for expert in row.get("selected_experts", [])
            if _int_value(expert) is not None
        }
        leaked = sorted(disabled_by_layer[layer].intersection(selected))
        if leaked:
            return (
                f"disabled experts leaked into masked layer {layer} token_trace: "
                f"{_preview_ints(leaked)}"
            )
        disabled_evidence = {
            int(expert)
            for expert in row.get("disabled_experts", [])
            if _int_value(expert) is not None
        }
        if disabled_by_layer[layer].issubset(disabled_evidence):
            evidence_layers.add(layer)
    missing_evidence = sorted(set(disabled_by_layer).difference(evidence_layers))
    if missing_evidence:
        return (
            "masked token_trace is missing disabled-expert evidence for layers: "
            f"{_preview_ints(missing_evidence)}"
        )
    return None


def _copy_json_artifact(src: Path, dst: Path, fallback: dict[str, Any] | None = None) -> None:
    if src.is_file():
        data = json.loads(src.read_text(encoding="utf-8"))
    else:
        data = fallback or {}
    _write_json(dst, data)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _pass_rate(values: list[bool]) -> float | None:
    return sum(1 for value in values if value) / len(values) if values else None


def _severity_for_records(records: list[dict[str, Any]]) -> str:
    severities = {str(record.get("regressionSeverity") or "none") for record in records}
    if "critical" in severities:
        return "critical"
    if "high" in severities:
        return "high"
    if "watch" in severities:
        return "watch"
    return "none"


def _safe_drop_candidates(
    *,
    disabled_by_layer: dict[int, set[int]],
    risky_prompt_ids: list[str],
    high_risk_domains: list[str],
) -> list[dict[str, int]]:
    if risky_prompt_ids or high_risk_domains:
        return []
    return [
        {"layer": int(layer), "expert": int(expert)}
        for layer, experts in sorted(disabled_by_layer.items())
        for expert in sorted(experts)
    ]


def build_eval_sidecars(
    *,
    suite_path: str | Path,
    baseline_generations_path: str | Path,
    masked_generations_path: str | Path,
    mask_path: str | Path,
    output_dir: str | Path,
    run_id: str | None = None,
    mask_id: str | None = None,
) -> dict[str, Any]:
    suite = _load_suite(str(suite_path))
    suite_ids = [_prompt_id_from_suite(row, f"suite line {index + 1}") for index, row in enumerate(suite)]
    if len(set(suite_ids)) < len(suite_ids):
        raise ValueError("suite.jsonl contains duplicate prompt IDs")

    semantic_by_prompt, semantic_coverage, missing_semantic = _suite_semantic_coverage(suite)
    if missing_semantic:
        raise ValueError(
            "suite.jsonl is missing required semantic prompt probes: "
            + ", ".join(missing_semantic)
        )

    baseline_path = _generations_jsonl_path(baseline_generations_path)
    masked_path = _generations_jsonl_path(masked_generations_path)
    baseline_rows = _read_jsonl_objects(baseline_path, "baseline generations.jsonl")
    masked_rows = _read_jsonl_objects(masked_path, "masked generations.jsonl")
    baseline_ids = _generation_ids(baseline_rows, "baseline generations.jsonl")
    masked_ids = _generation_ids(masked_rows, "masked generations.jsonl")
    if baseline_ids != suite_ids:
        raise ValueError("baseline generations prompt order does not match suite.jsonl")
    if masked_ids != suite_ids:
        raise ValueError("masked generations prompt order does not match suite.jsonl")
    if baseline_ids != masked_ids:
        raise ValueError("baseline and masked generations prompt order does not match")

    disabled_by_layer, mask_top_k = _read_mask(str(mask_path))
    disabled_count = sum(len(experts) for experts in disabled_by_layer.values())
    if disabled_count <= 0:
        raise ValueError("mask does not disable any experts; top-k-only comparisons cannot authorize hard pruning")

    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    suite_out = out_dir / "suite.jsonl"
    mask_out = out_dir / "mask.json"
    shutil.copyfile(Path(suite_path).expanduser(), suite_out)
    suite_sha256 = _file_sha256(suite_out)
    _copy_json_artifact(
        Path(mask_path).expanduser(),
        mask_out,
        fallback={
            "disabled_by_layer": {
                str(layer): sorted(experts) for layer, experts in sorted(disabled_by_layer.items())
            },
            "top_k_override": mask_top_k,
        },
    )

    records: list[dict[str, Any]] = []
    eval_trace_rows: list[dict[str, Any]] = []
    baseline_route_counts: list[int] = []
    masked_route_counts: list[int] = []
    baseline_token_counts: list[int] = []
    masked_token_counts: list[int] = []
    common_runtime_key: tuple[Any, ...] | None = None
    baseline_runtime_for_index: dict[str, Any] | None = None
    masked_runtime_for_index: dict[str, Any] | None = None

    for index, (suite_row, baseline_row, masked_row) in enumerate(
        zip(suite, baseline_rows, masked_rows, strict=True)
    ):
        prompt_id = suite_ids[index]
        domain = _string_value(suite_row.get("domain")) or "general"
        semantic_domains = semantic_by_prompt.get(prompt_id, [])

        baseline_result = _generation_result(baseline_row, f"baseline prompt {prompt_id}")
        masked_result = _generation_result(masked_row, f"masked prompt {prompt_id}")
        baseline_generation_settings, masked_generation_settings = _validate_generation_settings(
            prompt_id=prompt_id,
            suite_row=suite_row,
            baseline_result=baseline_result,
            masked_result=masked_result,
        )
        baseline_runtime = _runtime_info_from_result(baseline_result, f"baseline prompt {prompt_id}")
        masked_runtime = _runtime_info_from_result(masked_result, f"masked prompt {prompt_id}")
        _validate_runtime_info(
            baseline_runtime,
            label=f"baseline prompt {prompt_id}",
            require_mask=False,
            expected_disabled_count=0,
        )
        _validate_runtime_info(
            masked_runtime,
            label=f"masked prompt {prompt_id}",
            require_mask=True,
            expected_disabled_count=disabled_count,
        )
        if baseline_runtime.get("source_model_path") != masked_runtime.get("source_model_path"):
            raise ValueError(f"prompt {prompt_id} baseline/masked source_model_path does not match")
        if _runtime_common_key(baseline_runtime) != _runtime_common_key(masked_runtime):
            raise ValueError(f"prompt {prompt_id} baseline/masked runtime metadata does not match")
        current_key = _runtime_common_key(baseline_runtime)
        if common_runtime_key is None:
            common_runtime_key = current_key
            baseline_runtime_for_index = baseline_runtime
            masked_runtime_for_index = masked_runtime
        elif current_key != common_runtime_key:
            raise ValueError(f"prompt {prompt_id} runtime metadata does not match earlier prompts")

        baseline_layer_stats = baseline_result.get("layer_stats")
        masked_layer_stats = masked_result.get("layer_stats")
        baseline_routes = _validate_layer_stats(
            baseline_layer_stats,
            runtime=baseline_runtime,
            label=f"baseline prompt {prompt_id}",
        )
        masked_routes = _validate_layer_stats(
            masked_layer_stats,
            runtime=masked_runtime,
            label=f"masked prompt {prompt_id}",
        )
        baseline_trace = _token_trace_from_result(
            baseline_result,
            expected_route_records=baseline_routes,
            label=f"baseline prompt {prompt_id}",
        )
        masked_trace = _token_trace_from_result(
            masked_result,
            expected_route_records=masked_routes,
            label=f"masked prompt {prompt_id}",
        )
        leakage = _disabled_expert_leakage_issue(
            layer_stats=masked_layer_stats,
            token_trace=masked_trace,
            disabled_by_layer=disabled_by_layer,
        )
        if leakage is not None:
            raise ValueError(f"prompt {prompt_id}: {leakage}")

        baseline_text = str(baseline_result.get("text") or "")
        masked_text = str(masked_result.get("text") or "")
        expected_kind = _expected_kind(suite_row)
        expected = _expected_value(suite_row)
        validator = _validator_spec(suite_row)
        baseline_passed = _evaluation_passed(baseline_text, validator)
        masked_passed = _evaluation_passed(masked_text, validator)
        text_delta = _text_delta(baseline_text, masked_text)
        classification = _prompt_classification(
            baseline_passed=baseline_passed,
            candidate_passed=masked_passed,
        )
        severity = _regression_severity(
            classification=classification,
            text_delta=text_delta,
        )

        baseline_tokens = _int_value(baseline_result.get("tokens")) or 0
        masked_tokens = _int_value(masked_result.get("tokens")) or 0
        baseline_token_counts.append(baseline_tokens)
        masked_token_counts.append(masked_tokens)
        baseline_route_counts.append(baseline_routes)
        masked_route_counts.append(masked_routes)

        records.append(
            {
                "promptID": prompt_id,
                "domain": domain,
                "semanticDomains": semantic_domains,
                "expectedKind": expected_kind,
                "expected": expected,
                "validator": validator,
                "validatorKind": validator.get("kind"),
                "validatorAvailable": validator.get("available"),
                "validatorSource": validator.get("source"),
                "validatorReason": validator.get("reason"),
                "baselineText": baseline_text,
                "maskedText": masked_text,
                "textDelta": text_delta,
                "baselineGenerationSettings": baseline_generation_settings,
                "maskedGenerationSettings": masked_generation_settings,
                "baselineTokenCount": baseline_tokens,
                "maskedTokenCount": masked_tokens,
                "baselineRouteRecordCount": baseline_routes,
                "maskedRouteRecordCount": masked_routes,
                "baselineLayerStats": baseline_layer_stats,
                "maskedLayerStats": masked_layer_stats,
                "baselineTokensPerSecond": _float_value(baseline_result.get("tokens_per_second")) or 0.0,
                "maskedTokensPerSecond": _float_value(masked_result.get("tokens_per_second")) or 0.0,
                "latencyDeltaPct": 0.0,
                "baselinePassed": baseline_passed,
                "maskedPassed": masked_passed,
                "baselineQualified": baseline_passed is True,
                "promptClassification": classification,
                "safeDropEvidenceEligible": classification == "preserved",
                "adapter": "bf16_vmlx",
                "risk": _risk_label(
                    classification=classification,
                    severity=severity,
                ),
                "regressionSeverity": severity,
                "runtimeMode": baseline_runtime.get("runtime_mode"),
                "runtimeBackend": baseline_runtime.get("backend"),
                "runtimeDevice": baseline_runtime.get("device_name"),
                "runtimeMetalEnabled": baseline_runtime.get("runtime_metal_enabled"),
                "jangToolsVersion": baseline_runtime.get("jang_tools_version"),
                "mlxVersion": baseline_runtime.get("mlx_version"),
                "mlxLMVersion": baseline_runtime.get("mlx_lm_version"),
                "mlxVLMVersion": baseline_runtime.get("mlx_vlm_version"),
                "sourceModelPath": baseline_runtime.get("source_model_path"),
                "hookedMOELayers": baseline_runtime.get("hooked_moe_layers"),
                "expectedMOELayers": baseline_runtime.get("expected_moe_layers"),
                "hookCoverageComplete": baseline_runtime.get("hook_coverage_complete"),
                "maskApplied": masked_runtime.get("mask_applied"),
                "disabledExpertCount": masked_runtime.get("disabled_expert_count"),
                "topKOverride": masked_runtime.get("top_k_override"),
            }
        )

        for trace in baseline_trace:
            eval_trace_rows.append(
                {
                    "promptID": prompt_id,
                    "domain": domain,
                    "variant": "baseline",
                    "record": trace,
                }
            )
        for trace in masked_trace:
            eval_trace_rows.append(
                {
                    "promptID": prompt_id,
                    "domain": domain,
                    "variant": "masked",
                    "record": trace,
                }
            )

    assert baseline_runtime_for_index is not None
    assert masked_runtime_for_index is not None

    degraded_prompt_ids = [
        str(record["promptID"])
        for record in records
        if record.get("promptClassification") == "degraded"
    ]
    baseline_invalid_prompt_ids = [
        str(record["promptID"])
        for record in records
        if record.get("promptClassification") == "baseline_invalid"
    ]
    inconclusive_prompt_ids = [
        str(record["promptID"])
        for record in records
        if record.get("promptClassification") == "inconclusive"
    ]
    preserved_prompt_ids = [
        str(record["promptID"])
        for record in records
        if record.get("promptClassification") == "preserved"
    ]
    baseline_qualified_records = [
        record for record in records if record.get("baselineQualified") is True
    ]
    baseline_qualified_semantic_coverage = _semantic_coverage_for_records(
        baseline_qualified_records
    )
    missing_baseline_qualified_semantic_coverage = sorted(
        _required_reviewed_prune_semantic_domains().difference(
            set(baseline_qualified_semantic_coverage)
        )
    )
    validator_available_prompt_count = sum(
        1 for record in records if record.get("validatorAvailable") is True
    )
    classification_counts = _classification_counts(records)
    risky_prompt_ids = degraded_prompt_ids
    high_risk_domains = sorted(
        {
            domain
            for record in records
            if record["promptID"] in degraded_prompt_ids
            for domain in (record.get("semanticDomains") or [record.get("domain", "general")])
        }
    )
    severity = _severity_for_records(records)
    safe_candidates = _safe_drop_candidates(
        disabled_by_layer=disabled_by_layer,
        risky_prompt_ids=(
            degraded_prompt_ids
            + missing_baseline_qualified_semantic_coverage
            + (["no_baseline_qualified_prompts"] if not baseline_qualified_records else [])
        ),
        high_risk_domains=high_risk_domains,
    )
    prompt_count = len(records)
    run_name = run_id or baseline_path.parent.name or "ad-hoc"
    mask_name = mask_id or Path(mask_path).expanduser().stem or "mask"
    generated_at = _iso8601_now()
    baseline_route_total = sum(baseline_route_counts)
    masked_route_total = sum(masked_route_counts)

    comparison_summary = {
        "schema": "jang-expert-lab-comparison-summary-v1",
        "baselineRunID": run_name,
        "maskID": mask_name,
        "promptCount": prompt_count,
        "passRateBaseline": _pass_rate(
            [
                bool(record["baselinePassed"])
                for record in records
                if isinstance(record.get("baselinePassed"), bool)
            ]
        ),
        "passRateMasked": _pass_rate(
            [
                bool(record["maskedPassed"])
                for record in records
                if isinstance(record.get("maskedPassed"), bool)
            ]
        ),
        "baselineQualifiedPromptCount": len(baseline_qualified_records),
        "baselineQualifiedMaskedPassRate": _pass_rate(
            [
                bool(record["maskedPassed"])
                for record in baseline_qualified_records
                if isinstance(record.get("maskedPassed"), bool)
            ]
        ),
        "validatorAvailablePromptCount": validator_available_prompt_count,
        "classificationCounts": classification_counts,
        "baselineQualifiedPromptIDs": [str(record["promptID"]) for record in baseline_qualified_records],
        "baselineInvalidPromptIDs": baseline_invalid_prompt_ids,
        "inconclusivePromptIDs": inconclusive_prompt_ids,
        "preservedPromptIDs": preserved_prompt_ids,
        "degradedPromptIDs": degraded_prompt_ids,
        "baselineQualifiedSemanticCoverage": baseline_qualified_semantic_coverage,
        "missingBaselineQualifiedSemanticCoverage": missing_baseline_qualified_semantic_coverage,
        "meanTextDelta": _mean([float(record["textDelta"]) for record in records]) or 0.0,
        "meanLatencyDeltaPct": 0.0,
        "regression_severity": severity,
        "highRiskDomains": high_risk_domains,
        "safeDropCandidates": safe_candidates,
        "semanticCoverage": semantic_coverage,
        "missingSemanticCoverage": missing_semantic,
        "maxTextDelta": max((float(record["textDelta"]) for record in records), default=0.0),
    }

    eval_index = {
        "schema": "jang-expert-lab-eval-index-v1",
        "generated_at": generated_at,
        "run_id": run_name,
        "mask_id": mask_name,
        "prompt_count": prompt_count,
        "risky_prompt_ids": risky_prompt_ids,
        "prompt_ids": [str(record["promptID"]) for record in records],
        "high_risk_domains": high_risk_domains,
        "pass_rate_baseline": comparison_summary["passRateBaseline"],
        "pass_rate_masked": comparison_summary["passRateMasked"],
        "validator_schema": "jang-expert-lab-validator-v1",
        "validator_available_prompt_count": validator_available_prompt_count,
        "prompt_classification_counts": classification_counts,
        "baseline_qualified_prompt_count": len(baseline_qualified_records),
        "baseline_qualified_prompt_ids": comparison_summary["baselineQualifiedPromptIDs"],
        "baseline_invalid_prompt_ids": baseline_invalid_prompt_ids,
        "inconclusive_prompt_ids": inconclusive_prompt_ids,
        "preserved_prompt_ids": preserved_prompt_ids,
        "degraded_prompt_ids": degraded_prompt_ids,
        "baseline_qualified_masked_pass_rate": comparison_summary[
            "baselineQualifiedMaskedPassRate"
        ],
        "baseline_qualified_semantic_coverage": baseline_qualified_semantic_coverage,
        "missing_baseline_qualified_semantic_coverage": missing_baseline_qualified_semantic_coverage,
        "mean_text_delta": comparison_summary["meanTextDelta"],
        "regression_severity": severity,
        "min_baseline_tokens": min(baseline_token_counts) if baseline_token_counts else None,
        "min_masked_tokens": min(masked_token_counts) if masked_token_counts else None,
        "mean_baseline_tokens": _mean([float(value) for value in baseline_token_counts]),
        "mean_masked_tokens": _mean([float(value) for value in masked_token_counts]),
        "baseline_route_record_count": baseline_route_total,
        "masked_route_record_count": masked_route_total,
        "baseline_layer_stats_prompt_count": prompt_count,
        "masked_layer_stats_prompt_count": prompt_count,
        "eval_jsonl": "eval.jsonl",
        "eval_trace_jsonl": "eval_trace.jsonl",
        "comparison_summary": "comparison_summary.json",
        "suite_jsonl": "suite.jsonl",
        "suite_sha256": suite_sha256,
        "mask": "mask.json",
        "mask_json": "mask.json",
        "runtime_mode": baseline_runtime_for_index.get("runtime_mode"),
        "runtime_backend": baseline_runtime_for_index.get("backend"),
        "runtime_device": baseline_runtime_for_index.get("device_name"),
        "runtime_metal_enabled": baseline_runtime_for_index.get("runtime_metal_enabled"),
        "jang_tools_version": baseline_runtime_for_index.get("jang_tools_version"),
        "mlx_version": baseline_runtime_for_index.get("mlx_version"),
        "mlx_lm_version": baseline_runtime_for_index.get("mlx_lm_version"),
        "mlx_vlm_version": baseline_runtime_for_index.get("mlx_vlm_version"),
        "source_model_path": baseline_runtime_for_index.get("source_model_path"),
        "hooked_moe_layers": baseline_runtime_for_index.get("hooked_moe_layers"),
        "expected_moe_layers": baseline_runtime_for_index.get("expected_moe_layers"),
        "hook_coverage_complete": baseline_runtime_for_index.get("hook_coverage_complete"),
        "mask_applied": masked_runtime_for_index.get("mask_applied"),
        "disabled_expert_count": masked_runtime_for_index.get("disabled_expert_count"),
        "top_k_override": masked_runtime_for_index.get("top_k_override"),
        "generation_settings_checked": True,
        "semantic_coverage": semantic_coverage,
        "missing_semantic_coverage": missing_semantic,
    }

    comparison_path = out_dir / "comparison_summary.json"
    eval_path = out_dir / "eval.jsonl"
    trace_path = out_dir / "eval_trace.jsonl"
    index_path = out_dir / "eval_index.json"
    _write_json(comparison_path, comparison_summary)
    _write_jsonl(eval_path, records)
    _write_jsonl(trace_path, eval_trace_rows)
    _write_json(index_path, eval_index)

    return {
        "ok": True,
        "schema": "jang-expert-lab-vmlx-eval-build-v1",
        "output": str(out_dir),
        "suite_jsonl": str(suite_out),
        "suite_sha256": suite_sha256,
        "mask": str(mask_out),
        "mask_json": str(mask_out),
        "comparison_summary": str(comparison_path),
        "eval_jsonl": str(eval_path),
        "eval_trace_jsonl": str(trace_path),
        "eval_index": str(index_path),
        "prompt_count": prompt_count,
        "baseline_route_record_count": baseline_route_total,
        "masked_route_record_count": masked_route_total,
        "eval_trace_record_count": len(eval_trace_rows),
        "semantic_coverage": semantic_coverage,
        "missing_semantic_coverage": missing_semantic,
        "risky_prompt_ids": risky_prompt_ids,
        "high_risk_domains": high_risk_domains,
        "regression_severity": severity,
        "safe_drop_candidates": safe_candidates,
        "prompt_classification_counts": classification_counts,
        "baseline_qualified_prompt_count": len(baseline_qualified_records),
        "baseline_invalid_prompt_ids": baseline_invalid_prompt_ids,
        "inconclusive_prompt_ids": inconclusive_prompt_ids,
        "preserved_prompt_ids": preserved_prompt_ids,
        "degraded_prompt_ids": degraded_prompt_ids,
        "missing_baseline_qualified_semantic_coverage": missing_baseline_qualified_semantic_coverage,
        "generation_settings_checked": True,
    }


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import load

    _patch_qwen_sparse_moe()

    model_path = Path(args.model).expanduser()
    out_dir = Path(args.output).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = _load_suite(args.suite)
    disabled_by_layer, mask_top_k = _read_mask(args.mask)
    if args.top_k_override < 0:
        raise ValueError(f"--top-k-override must be non-negative, got {args.top_k_override}")
    top_k_override = args.top_k_override or mask_top_k

    _validate_vmlx_source_config(model_path)
    model, tokenizer = load(str(model_path))
    layer_hooks = _install_layer_indices(model)
    hooked_layers = len(layer_hooks)
    if hooked_layers <= 0:
        raise RuntimeError(
            "No Qwen/vMLX MoE router blocks were found; cannot collect BF16 Expert Lab traces."
        )
    expected_moe_layers = _validate_moe_hook_coverage(model_path, hooked_layers)
    _validate_mask_targets(disabled_by_layer, top_k_override, layer_hooks)

    runtime = _runtime_info(
        model_path=model_path,
        hooked_layers=hooked_layers,
        expected_moe_layers=expected_moe_layers,
        disabled_by_layer=disabled_by_layer,
        top_k_override=top_k_override,
    )
    generations_path = out_dir / "generations.jsonl"
    summary_path = out_dir / "summary.json"
    # Path transitions for Intent Prune hybrid scoring (PR-IP0).
    # --emit-transitions is opt-in (does not auto-fire from --emit-token-trace).
    # Transitions still require collecting token_trace in memory for path build.
    emit_transitions = bool(getattr(args, "emit_transitions", False))
    persist_token_trace = bool(getattr(args, "emit_token_trace", False))
    collect_token_trace = bool(persist_token_trace or emit_transitions)
    transitions_path = out_dir / "expert_transitions.jsonl"
    transition_records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    if emit_transitions:
        from .intent_prune.transitions import (
            build_transition_records_for_prompt,
            write_transitions_jsonl,
        )

    with generations_path.open("w", encoding="utf-8") as fh:
        for index, prompt in enumerate(prompts):
            prompt_text = str(prompt.get("prompt") or prompt.get("text") or "")
            context = ExpertTraceContext(
                disabled_by_layer=disabled_by_layer,
                top_k_override=top_k_override,
                emit_token_trace=collect_token_trace,
                max_trace_tokens=args.max_trace_tokens,
            )
            global _ACTIVE_TRACE
            _ACTIVE_TRACE = context
            started = time.perf_counter()
            text = ""
            final = None
            generation_settings = _generation_settings(prompt, args, top_k_override)
            sampler = make_sampler(
                temp=float(generation_settings["temperature"]),
                top_p=float(generation_settings["top_p"]),
                top_k=int(generation_settings["top_k"]),
            )
            try:
                for response in stream_generate(
                    model,
                    tokenizer,
                    _prompt_tokens(tokenizer, prompt_text),
                    max_tokens=int(generation_settings["max_tokens"]),
                    sampler=sampler,
                ):
                    text += response.text
                    final = response
            finally:
                _ACTIVE_TRACE = None
                mx.clear_cache()

            if let_issue := _trace_coverage_issue(context, layer_hooks):
                raise RuntimeError(f"{let_issue} for prompt {prompt.get('id')!r}")
            if let_issue := _mask_application_issue(context, disabled_by_layer):
                raise RuntimeError(f"{let_issue} for prompt {prompt.get('id')!r}")
            if let_issue := _token_trace_evidence_issue(
                context,
                emit_token_trace=collect_token_trace,
                disabled_by_layer=disabled_by_layer,
                top_k_override=top_k_override,
            ):
                raise RuntimeError(f"{let_issue} for prompt {prompt.get('id')!r}")

            elapsed = max(time.perf_counter() - started, 0.000001)
            generation_tokens = int(getattr(final, "generation_tokens", 0) or 0)
            collected_trace = context.token_trace if collect_token_trace else None
            # Persist full token_trace only when explicitly requested; transitions
            # alone keep paths in expert_transitions.jsonl to limit disk use.
            token_trace_out = collected_trace if persist_token_trace else None
            if emit_transitions and collected_trace:
                fallback_id = f"prompt-{index}"
                transition_records.extend(
                    build_transition_records_for_prompt(
                        prompt,
                        collected_trace,
                        fallback_id=fallback_id,
                    )
                )
            row = {
                "schema": "jang-expert-lab-vmlx-generation-v1",
                "prompt_index": index,
                "prompt": prompt,
                "result": {
                    "text": text,
                    "tokens": generation_tokens,
                    "elapsed_seconds": elapsed,
                    "tokens_per_second": generation_tokens / elapsed,
                    "generation_settings": generation_settings,
                    "finish_reason": (
                        "stop"
                        if getattr(final, "finish_reason", None) == "stop"
                        else "max_tokens"
                    ),
                    "layer_stats": context.layer_stats(),
                    "token_trace": token_trace_out,
                    "runtime_info": runtime,
                },
            }
            rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            fh.flush()

    if emit_transitions:
        write_transitions_jsonl(transitions_path, transition_records)

    summary = {
        "schema": "jang-expert-lab-vmlx-run-v1",
        "ok": True,
        "model": str(model_path),
        "suite": str(Path(args.suite).expanduser()),
        "output": str(out_dir),
        "generations_jsonl": str(generations_path),
        "summary_json": str(summary_path),
        "expert_transitions_jsonl": str(transitions_path) if emit_transitions else None,
        "expert_transition_record_count": len(transition_records) if emit_transitions else 0,
        "suite_sha256": _file_sha256(Path(args.suite).expanduser()),
        "generation_defaults": {
            "max_tokens": int(args.max_tokens),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "top_k": int(args.top_k),
            "top_k_override": int(top_k_override) if top_k_override else None,
        },
        "prompt_count": len(rows),
        "hooked_moe_layers": hooked_layers,
        "expected_moe_layers": expected_moe_layers,
        "hook_coverage_complete": (
            True if expected_moe_layers is None else hooked_layers >= expected_moe_layers
        ),
        "mask": _mask_info(disabled_by_layer, top_k_override),
        "runtime_info": runtime,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def cmd_expert_lab_vmlx(args: argparse.Namespace) -> None:
    try:
        summary = run_suite(args)
    except VMLXSourceConfigError as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps(summary, sort_keys=True))


def cmd_expert_lab_vmlx_build_eval(args: argparse.Namespace) -> None:
    summary = build_eval_sidecars(
        suite_path=args.suite,
        baseline_generations_path=args.baseline_generations,
        masked_generations_path=args.masked_generations,
        mask_path=args.mask,
        output_dir=args.output,
        run_id=args.run_id,
        mask_id=args.mask_id,
    )
    print(json.dumps(summary, sort_keys=True))


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "expert-lab-vmlx",
        help="Run BF16/F16 Expert Lab prompt tracing through mlx_lm/vMLX",
    )
    parser.add_argument("model", help="Original BF16/F16 HuggingFace model directory")
    parser.add_argument("--suite", required=True, help="Expert prompt suite JSONL")
    parser.add_argument("--output", required=True, help="Directory for run artifacts")
    parser.add_argument("--mask", help="ExpertLab mask artifact JSON")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--emit-token-trace",
        action="store_true",
        help=(
            "Persist full per-layer token_trace inside generations.jsonl. "
            "Does not write expert_transitions.jsonl; use --emit-transitions for path scoring."
        ),
    )
    parser.add_argument(
        "--emit-transitions",
        action="store_true",
        help=(
            "Write expert_transitions.jsonl (ordered per-token layer paths for Intent Prune). "
            "Collects token_trace in memory for path build; does not embed it in "
            "generations.jsonl unless --emit-token-trace is also set."
        ),
    )
    parser.add_argument("--max-trace-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-k-override", type=int, default=0)
    parser.set_defaults(func=cmd_expert_lab_vmlx)

    eval_parser = subparsers.add_parser(
        "expert-lab-vmlx-build-eval",
        help="Build Expert Lab same-suite eval sidecars from baseline/masked vMLX generations",
    )
    eval_parser.add_argument("--suite", required=True, help="Expert prompt suite JSONL")
    eval_parser.add_argument(
        "--baseline-generations",
        required=True,
        help="Baseline generations.jsonl or its parent directory",
    )
    eval_parser.add_argument(
        "--masked-generations",
        required=True,
        help="Masked generations.jsonl or its parent directory",
    )
    eval_parser.add_argument("--mask", required=True, help="ExpertLab mask artifact JSON")
    eval_parser.add_argument("--output", required=True, help="Directory for eval sidecars")
    eval_parser.add_argument("--run-id", help="Run identifier stored in eval_index.json")
    eval_parser.add_argument("--mask-id", help="Mask identifier stored in eval_index.json")
    eval_parser.set_defaults(func=cmd_expert_lab_vmlx_build_eval, json=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    register(p.add_subparsers(dest="command", required=True))
    cmd_args = p.parse_args()
    cmd_args.func(cmd_args)
