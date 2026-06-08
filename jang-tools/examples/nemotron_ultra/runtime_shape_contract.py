"""Generate a no-load shape and bit contract for Nemotron Ultra runtime work.

This artifact is meant to sit next to the speed patch spec. It records the
current bundle metadata and saved probe tensor shapes that a MoE/Mamba speed
patch must preserve.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_BUNDLE = Path("/Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L")
DEFAULT_LOG_DIR = Path("docs/runtime/logs")
DEFAULT_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-shape-contract.md")
DEFAULT_JSON_OUT = Path("docs/runtime/logs/2026-06-04-nemotron-ultra-runtime-shape-contract.json")


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _first_layer(path: Path) -> dict[str, Any]:
    data = _load(path) or {}
    layers = data.get("layers", [])
    return dict(layers[0]) if layers else {}


def _layer_counts(config: dict[str, Any]) -> dict[str, int]:
    raw = config.get("layers_block_type", [])
    counts = Counter(str(item) for item in raw)
    return {
        "total": len(raw),
        "mamba": counts.get("mamba", 0),
        "moe": counts.get("moe", 0),
        "attention": counts.get("attention", 0),
    }


def _build_result(bundle: Path, log_dir: Path) -> dict[str, Any]:
    config = _load(bundle / "config.json") or {}
    jang_config = _load(bundle / "jang_config.json") or {}
    mamba_layer = _first_layer(log_dir / "2026-06-04-nemotron-ultra-mamba-component-probe.json")
    moe_layer = _first_layer(log_dir / "2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json")
    layer_counts = _layer_counts(config)
    return {
        "bundle": str(bundle),
        "log_dir": str(log_dir),
        "status": "READY" if config and jang_config and mamba_layer and moe_layer else "PARTIAL",
        "sources": {
            "config": str(bundle / "config.json"),
            "jang_config": str(bundle / "jang_config.json"),
            "mamba_component": str(log_dir / "2026-06-04-nemotron-ultra-mamba-component-probe.json"),
            "moe_component": str(log_dir / "2026-06-04-nemotron-ultra-moe-weighted-decode-component-probe.json"),
        },
        "architecture": {
            "family": jang_config.get("capabilities", {}).get("family"),
            "modality": jang_config.get("capabilities", {}).get("modality"),
            "cache_type": jang_config.get("capabilities", {}).get("cache_type"),
            "hidden_size": config.get("hidden_size"),
            "vocab_size": config.get("vocab_size"),
            "tie_word_embeddings": config.get("tie_word_embeddings"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "layer_counts": layer_counts,
            "num_attention_heads": config.get("num_attention_heads"),
            "num_key_value_heads": config.get("num_key_value_heads"),
            "num_experts_per_tok": config.get("num_experts_per_tok"),
            "moe_intermediate_size": config.get("moe_intermediate_size"),
            "ssm_state_size": config.get("ssm_state_size"),
        },
        "quantization": {
            "mxtq_bits": jang_config.get("mxtq_bits") or config.get("mxtq_bits"),
            "method": jang_config.get("quantization", {}).get("method"),
            "drops_mtp": jang_config.get("quantization", {}).get("drops_mtp"),
            "estimated_output_gib": jang_config.get("quantization", {}).get("estimated_output_gib"),
            "fp8_projection_affine_bits": jang_config.get("quantization", {}).get("fp8_projection_affine_bits"),
            "fp8_projection_group_size": jang_config.get("quantization", {}).get("fp8_projection_group_size"),
            "keeps_attention_bf16": jang_config.get("runtime", {}).get("keeps_attention_bf16"),
            "keeps_latent_moe_bf16": jang_config.get("runtime", {}).get("keeps_latent_moe_bf16"),
            "keeps_router_gates_source_precision": jang_config.get("runtime", {}).get("keeps_router_gates_source_precision"),
            "shard_count": jang_config.get("runtime", {}).get("shard_count"),
            "total_shard_bytes": jang_config.get("runtime", {}).get("total_shard_bytes"),
        },
        "mamba_contract": {
            key: mamba_layer.get(key)
            for key in (
                "layer_index",
                "cache_ordinal",
                "hidden_shape",
                "normed_shape",
                "projected_shape",
                "gate_shape",
                "conv_dim",
                "conv_input_shape",
                "conv_output_shape",
                "ssm_out_shape",
                "intermediate_size",
                "num_heads",
                "n_groups",
                "ssm_state_size",
            )
            if key in mamba_layer
        },
        "moe_contract": {
            key: moe_layer.get(key)
            for key in (
                "layer_index",
                "hidden_shape",
                "scores_shape",
                "indices_shape",
                "latent_shape",
                "routed_shape",
            )
            if key in moe_layer
        },
        "preserve": [
            "48 Mamba companion cache entries plus 12 attention KV entries for hybrid prefix cache",
            "MTP remains dropped; speculative draft KV/SSM state is out of scope for this bundle",
            "routed expert up/down projections remain 1-bit according to mxtq_bits",
            "shared expert and Mamba projection paths remain 8-bit unless new proof reverses the projection tradeoff",
            "router gates and attention BF16 retention remain source-precision/preserved runtime surfaces",
            "text-only modality remains explicit; media requests must reject or reroute",
        ],
    }


def _fmt(value: Any) -> str:
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _render(result: dict[str, Any]) -> str:
    arch = result["architecture"]
    quant = result["quantization"]
    lines = [
        "# Nemotron Ultra Runtime Shape Contract",
        "",
        f"bundle: `{result['bundle']}`",
        f"log_dir: `{result['log_dir']}`",
        f"status: `{result['status']}`",
        "",
        "## Architecture",
        f"- family: `{arch.get('family')}`",
        f"- modality: `{arch.get('modality')}`",
        f"- cache_type: `{arch.get('cache_type')}`",
        f"- hidden_size: `{arch.get('hidden_size')}`",
        f"- vocab_size: `{arch.get('vocab_size')}`",
        f"- tie_word_embeddings: `{arch.get('tie_word_embeddings')}`",
        f"- num_hidden_layers: `{arch.get('num_hidden_layers')}`",
        f"- layer_counts: `{_fmt(arch.get('layer_counts'))}`",
        f"- attention_heads: `{arch.get('num_attention_heads')}`",
        f"- key_value_heads: `{arch.get('num_key_value_heads')}`",
        f"- num_experts_per_tok: `{arch.get('num_experts_per_tok')}`",
        f"- moe_intermediate_size: `{arch.get('moe_intermediate_size')}`",
        f"- ssm_state_size: `{arch.get('ssm_state_size')}`",
        "",
        "## Quantization",
        f"- mxtq_bits: `{_fmt(quant.get('mxtq_bits'))}`",
        f"- method: `{quant.get('method')}`",
        f"- drops_mtp: `{quant.get('drops_mtp')}`",
        f"- estimated_output_gib: `{quant.get('estimated_output_gib')}`",
        f"- fp8_projection_affine_bits: `{quant.get('fp8_projection_affine_bits')}`",
        f"- fp8_projection_group_size: `{quant.get('fp8_projection_group_size')}`",
        f"- keeps_attention_bf16: `{quant.get('keeps_attention_bf16')}`",
        f"- keeps_latent_moe_bf16: `{quant.get('keeps_latent_moe_bf16')}`",
        f"- keeps_router_gates_source_precision: `{quant.get('keeps_router_gates_source_precision')}`",
        f"- shard_count: `{quant.get('shard_count')}`",
        "",
        "## Mamba Decode Contract",
    ]
    lines.extend(f"- {key}: `{_fmt(value)}`" for key, value in result["mamba_contract"].items())
    lines.extend(["", "## MoE Decode Contract"])
    lines.extend(f"- {key}: `{_fmt(value)}`" for key, value in result["moe_contract"].items())
    lines.extend(["", "## Preserve"])
    lines.extend(f"- {item}" for item in result["preserve"])
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    args = ap.parse_args()

    result = _build_result(args.bundle, args.log_dir)
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
