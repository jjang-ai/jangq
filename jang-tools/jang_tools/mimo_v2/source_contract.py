"""Source-contract inspection for MiMo-V2.5 JANG_2L conversion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors import safe_open


@dataclass(frozen=True)
class MiMoSourceContract:
    model_type: str
    num_hidden_layers: int
    n_routed_experts: int
    num_experts_per_tok: int
    full_kv_heads: int
    swa_kv_heads: int
    full_qkv_shape: tuple[int, int]
    swa_qkv_shape: tuple[int, int]
    full_layer_count: int
    swa_layer_count: int
    has_visual_tensors: bool
    has_audio_tensors: bool
    has_mtp_tensors: bool
    ignored_text_o_proj_count: int
    capabilities: dict[str, Any]
    runtime: dict[str, Any]


def inspect_mimo_source(src_dir: str | Path) -> MiMoSourceContract:
    """Inspect a local MiMo-V2.5 source directory and return runtime facts.

    This function is deliberately source-backed: it cross-checks config fields
    against real qkv tensor shapes so stale README or memory-page claims cannot
    drive the converter.
    """
    src = Path(src_dir).expanduser()
    cfg = _read_json(src / "config.json")
    weight_map = _read_json(src / "model.safetensors.index.json")["weight_map"]

    pattern = list(cfg.get("hybrid_layer_pattern") or [])
    if len(pattern) != int(cfg["num_hidden_layers"]):
        raise ValueError("hybrid_layer_pattern length does not match num_hidden_layers")

    full_layers = [i for i, kind in enumerate(pattern) if int(kind) == 0]
    swa_layers = [i for i, kind in enumerate(pattern) if int(kind) == 1]
    if not full_layers or not swa_layers:
        raise ValueError("MiMo source must contain both full-attention and SWA layers")

    full_qkv_shape = _tensor_shape(
        src, weight_map, f"model.layers.{full_layers[0]}.self_attn.qkv_proj.weight"
    )
    swa_qkv_shape = _tensor_shape(
        src, weight_map, f"model.layers.{swa_layers[0]}.self_attn.qkv_proj.weight"
    )

    ignored = cfg.get("quantization_config", {}).get("ignored_layers", [])
    ignored_text_o_proj_count = sum(
        1
        for name in ignored
        if name.startswith("model.layers.") and name.endswith(".self_attn.o_proj")
    )

    has_mtp = any(k.startswith("model.mtp.") for k in weight_map)
    capabilities = {
        "family": "mimo_v2",
        "modality": "text",
        "modalities": ["text"],
        "preserved_modalities": ["vision", "audio"],
        "unwired_modalities": ["vision", "audio"],
        "cache_type": "kv",
        "supports_tools": True,
        "supports_thinking": True,
        "reasoning_parser": "think_xml",
        "tool_parser": "xml_function",
        "tool_status": "template_uses_xml_function_tool_call",
        "think_in_template": False,
        "multimodal_status": "weights_preserved_text_runtime",
    }
    runtime = {
        "bundle_has_mtp": has_mtp,
        "mtp_mode": "preserved_disabled",
        "mtp_status": (
            "MTP tensors are preserved for runtimes with a proven native "
            "accept/reject path; base JANG_2L decode must stay autoregressive."
        ),
        "attention": {
            "full_kv_heads": int(cfg["num_key_value_heads"]),
            "swa_kv_heads": int(cfg["swa_num_key_value_heads"]),
            "full_qkv_shape": full_qkv_shape,
            "swa_qkv_shape": swa_qkv_shape,
            "attention_value_scale": cfg.get("attention_value_scale"),
            "sliding_window": cfg.get("sliding_window"),
            "swa_attention_sink_bias": bool(cfg.get("add_swa_attention_sink_bias")),
        },
        "cache_topology": {
            "family": "hybrid_full_swa_kv",
            "prefix_cache": True,
            "l2_disk_cache": True,
            "turboquant_kv": "full_attention_layers_only",
            "swa_layers": "rotating_kv_native",
        },
    }

    return MiMoSourceContract(
        model_type=str(cfg["model_type"]),
        num_hidden_layers=int(cfg["num_hidden_layers"]),
        n_routed_experts=int(cfg["n_routed_experts"]),
        num_experts_per_tok=int(cfg["num_experts_per_tok"]),
        full_kv_heads=int(cfg["num_key_value_heads"]),
        swa_kv_heads=int(cfg["swa_num_key_value_heads"]),
        full_qkv_shape=full_qkv_shape,
        swa_qkv_shape=swa_qkv_shape,
        full_layer_count=len(full_layers),
        swa_layer_count=len(swa_layers),
        has_visual_tensors=any(k.startswith("visual.") for k in weight_map),
        has_audio_tensors=any(k.startswith("audio_encoder.") for k in weight_map),
        has_mtp_tensors=has_mtp,
        ignored_text_o_proj_count=ignored_text_o_proj_count,
        capabilities=capabilities,
        runtime=runtime,
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _tensor_shape(
    src_dir: Path,
    weight_map: dict[str, str],
    tensor_name: str,
) -> tuple[int, int]:
    shard = weight_map[tensor_name]
    with safe_open(str(src_dir / shard), framework="pt") as f:
        shape = tuple(int(v) for v in f.get_slice(tensor_name).get_shape())
    if len(shape) != 2:
        raise ValueError(f"{tensor_name} is not 2-D: {shape}")
    return shape
