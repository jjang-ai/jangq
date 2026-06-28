"""
JANG Model Loader — Load JANG quantized models into MLX for inference.
Created by Jinho Jang (eric@jangq.ai)

v2 models: MLX-native safetensors — load via mx.load() mmap in seconds.
v1 models: Legacy format — repacks JANG uint8 to MLX uint32 (slow, 5-10 min).

v2 is the default format for new conversions. v1 backward compat is preserved
so existing models on HuggingFace continue to work.
"""

import gc
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
from .ssm_layout import sanitize_grouped_conv1d_layout

try:
    import mlx.core as mx
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False

logger = logging.getLogger(__name__)

# Support current "jang_config.json" and legacy names
JANG_CONFIG_FILENAMES = ["jang_config.json", "jjqf_config.json", "jang_cfg.json", "mxq_config.json"]
JANG_FORMAT_VALUES = ["jang", "jjqf", "mxq"]

# Shard flush threshold for v1 streaming repack (~2 GB)
_SHARD_FLUSH_BYTES = 2_000_000_000  # Made by Jinho Jang — jangq.ai


def _jang_quant_block_size(jang_cfg: dict, default: int = 64) -> int:
    """Resolve runtime quant group size from old and new sidecar keys."""
    quant = jang_cfg.get("quantization") or {}
    return int(quant.get("block_size") or quant.get("group_size") or default)


def _jang_default_bits(jang_cfg: dict, fallback: list[int] | None = None) -> int:
    """Resolve the default affine bit width for JANG runtime modules."""
    quant = jang_cfg.get("quantization") or {}
    if quant.get("bits") is not None:
        return int(quant["bits"])
    bit_widths = quant.get("bit_widths_used", fallback or [4])
    return int(min(bit_widths))


def _module_can_quantize_with_group_size(module, group_size: int) -> bool:
    """Whether an affine module's input dimension can use this group size."""
    input_dims = getattr(module, "input_dims", None)
    if input_dims is None:
        weight = getattr(module, "weight", None)
        shape = getattr(weight, "shape", None)
        if shape is not None and len(shape) >= 2:
            input_dims = shape[-1]
    if input_dims is None:
        return True
    try:
        return int(input_dims) % int(group_size) == 0
    except Exception:
        return True


def _module_path_is_raw_adapter(path: str) -> bool:
    """Adapters such as vision-LoRA are stored as raw float weights."""
    return any(part.startswith("lora") for part in str(path).split("."))


def _sanitize_grouped_conv1d_layout(weights: dict) -> dict:
    """Force leftover grouped Conv1d weights into MLX layout.

    Some mlx-lm sanitize paths return successfully without touching dense or
    already-stacked converted bundles, leaving HF Conv1d layout `(out, 1,
    kernel)` on disk. MLX Conv1d expects `(out, kernel, 1)`, so keep this
    idempotent pass after model-specific sanitize too.
    """
    def _transpose(value):
        if _MLX_AVAILABLE and type(value).__module__.startswith("mlx."):
            return mx.transpose(value, axes=(0, 2, 1))
        return np.transpose(value, (0, 2, 1))

    return sanitize_grouped_conv1d_layout(weights, _transpose)


def _sanitize_qwen3_next_conv1d_layout(weights: dict) -> dict:
    """Backward-compatible alias for older callers/tests."""
    return _sanitize_grouped_conv1d_layout(weights)


def _find_config_path(model_path: Path) -> Optional[Path]:
    for name in JANG_CONFIG_FILENAMES:
        p = model_path / name
        if p.exists():
            return p
    return None


# M152 (iter 75): migrated from local M151 copies to the shared helpers
# in _json_utils. Keep these aliases for minimum diff to the call sites
# (4 sites refer to _read_config_or_raise / _read_config_or_none).
from ._json_utils import (
    read_json_object as _read_config_or_raise_base,
    read_json_object_safe as _read_config_or_safe,
)


def _read_config_or_raise(path: Path, *, purpose: str) -> dict:
    """Raise-contract thin alias → shared _json_utils.read_json_object."""
    return _read_config_or_raise_base(path, purpose=purpose)


def _read_config_or_none(path: Path) -> Optional[dict]:
    """Detection-helper variant: returns None on ANY read/parse failure
    instead of raising. Used by ``_is_v2_model`` / ``_is_vlm_config``
    which are part of the "can we handle this at all?" probe surface —
    they must tolerate corrupt files and return False upstream, not
    crash the entire wizard's source-detection step.
    """
    data, _err = _read_config_or_safe(path, purpose="config")
    return data


def is_jang_model(model_path: str | Path) -> bool:
    """Check if a directory contains a JANG model."""
    return _find_config_path(Path(model_path)) is not None


def _is_v2_model(model_path: Path) -> bool:
    """Check if a JANG model uses v2 format (MLX-native safetensors).

    v2 detection: has model.safetensors.index.json OR standard .safetensors
    files (not .jang.safetensors). Also verified via format_version in config.
    """
    # Check for v2 index file
    if (model_path / "model.safetensors.index.json").exists():
        return True
    # Check for standard safetensors (not .jang.safetensors)
    has_standard = any(model_path.glob("model-*.safetensors"))
    has_jang = any(model_path.glob("*.jang.safetensors"))
    if has_standard and not has_jang:
        return True
    # Check format_version in config. M151 (iter 74): tolerate corrupt
    # config so detection can return False cleanly instead of crashing the
    # caller with a cryptic JSONDecodeError.
    config_path = _find_config_path(model_path)
    if config_path:
        cfg = _read_config_or_none(config_path)
        if cfg is not None:
            version = cfg.get("format_version", "1.0")
            if version.startswith("2"):
                return True
    return False


def _is_vlm_config(model_path: Path) -> bool:
    """Check if a model has vision_config in its config.json (i.e., is a VL model)."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        return False
    # M151 (iter 74): detection must not crash on corrupt config — return
    # False so callers (Swift SourceStep, is_jang_model probes) handle the
    # corrupt-file case via the standard "not a VLM" path.
    config = _read_config_or_none(config_path)
    if config is None:
        return False
    return "vision_config" in config


# ─── v2 loader (instant) ────────────────────────────────────────────


def _strip_runtime_ignored_weights(weights: dict) -> dict:
    """Drop preserved-sidecar tensors that current MLX decoders do not consume."""
    return {
        k: v
        for k, v in weights.items()
        if not k.endswith(".importance") and not k.startswith("mtp.")
    }


def _first_safetensors_metadata(path: Path) -> dict:
    weight_files = _get_v2_weight_files(path)
    if not weight_files:
        return {}
    try:
        import safetensors
    except ImportError:
        return {}
    try:
        with safetensors.safe_open(str(weight_files[0]), framework="np") as f:
            return dict(f.metadata() or {})
    except (OSError, RuntimeError, ValueError):
        return {}


def _has_indexed_mtp_weights(path: Path) -> bool:
    index_path = path / "model.safetensors.index.json"
    if not index_path.exists():
        return False
    try:
        index = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return any(name.startswith("mtp.") for name in index.get("weight_map", {}))


def _load_jang_v2_vlm_native_mlx(path: Path, jang_cfg: dict):
    """Load native MLX VLM weights while filtering preserved MTP tensors.

    Preserved MTP tensors are part of the bundle contract, but current mlx-vlm
    model classes do not expose MTP modules. This mirrors mlx_vlm.load_model for
    native `format=mlx` shards and removes only the tensors that are not part of
    today's autoregressive runtime surface.
    """
    import mlx.nn as nn
    from mlx_vlm.utils import (
        get_model_and_args,
        load_config as vlm_load_config,
        load_image_processor,
        load_processor,
        sanitize_weights,
        skip_multimodal_module,
        update_module_configs,
    )

    start = time.perf_counter()
    config = vlm_load_config(path)
    weight_files = _get_v2_weight_files(path)
    if not weight_files:
        raise FileNotFoundError(f"No safetensors weights found in {path}")

    weights = {}
    for weight_file in weight_files:
        weights.update(mx.load(str(weight_file)))
    weights = _strip_runtime_ignored_weights(weights)

    metadata = _first_safetensors_metadata(path)
    is_mlx_format = metadata.get("format") == "mlx"

    model_class, _ = get_model_and_args(config=config)
    config.setdefault("text_config", config.pop("llm_config", {}))
    config.setdefault("vision_config", {})
    config.setdefault("audio_config", {})

    has_quantization = "quantization" in config
    model_config = model_class.ModelConfig.from_dict(config)
    modules = ["text", "vision", "perceiver", "projector", "audio"]
    model_config = update_module_configs(model_config, model_class, config, modules)
    model = model_class.Model(model_config)

    if not is_mlx_format:
        weights = sanitize_weights(model, weights)
        if hasattr(model_class, "VisionModel"):
            weights = sanitize_weights(
                model_class.VisionModel, weights, model_config.vision_config
            )
        if hasattr(model_class, "LanguageModel"):
            weights = sanitize_weights(
                model_class.LanguageModel, weights, model_config.text_config
            )
        if hasattr(model_class, "AudioModel"):
            weights = sanitize_weights(
                model_class.AudioModel, weights, model_config.audio_config
            )

    if has_quantization:
        for quantization_key in ("quantization", "quantization_config"):
            quantization_value = getattr(model_config, quantization_key, None)
            if quantization_value is not None:
                config[quantization_key] = quantization_value

    quantization = config.get("quantization")
    if quantization is not None:
        skip_vision = config.get("vision_config", {}).get("skip_vision", False)

        def get_class_predicate(p, m):
            if skip_multimodal_module(p) and skip_vision:
                return False
            if p in config["quantization"]:
                return config["quantization"][p]
            if not hasattr(m, "to_quantized"):
                return False
            if hasattr(m, "weight") and m.weight.size % 64 != 0:
                return False
            return f"{p}.scales" in weights

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization.get("mode", "affine"),
            class_predicate=get_class_predicate,
        )

    model.load_weights(list(weights.items()))
    _fix_quantized_bits(model, {})
    mx.eval(model.parameters())
    model.eval()

    elapsed = time.perf_counter() - start
    logger.info(f"JANG native MLX VLM loaded in {elapsed:.1f}s")

    image_processor = load_image_processor(path)
    eos_token_id = getattr(model.config, "eos_token_id", None)
    try:
        processor = load_processor(path, True, eos_token_ids=eos_token_id)
    except (ImportError, ValueError, OSError, KeyError, TypeError):
        processor = _build_vlm_processor(path, eos_token_id)
    if image_processor is not None:
        processor.image_processor = image_processor

    return model, processor


def _load_jang_v2(path: Path, jang_cfg: dict):
    """
    Load a JANG v2 model — instant via mx.load() mmap.

    v2 models store weights in MLX-native format (uint32 packed weights,
    float16 scales/biases) in standard safetensors. No repacking needed.
    """
    from mlx_lm.utils import load_config, load_model as _load_model_skeleton, load_tokenizer

    start = time.perf_counter()
    config = load_config(path)

    # config.json already has quantization key (written by v2 converter)
    # but ensure it exists for older v2 models
    if "quantization" not in config:
        block_size = _jang_quant_block_size(jang_cfg)
        default_bits = _jang_default_bits(jang_cfg, [4])
        config["quantization"] = {"group_size": block_size, "bits": default_bits}

    model, config = _load_model_skeleton(
        path, lazy=True, strict=False, model_config=config
    )
    _upgrade_switch_to_quantized(
        model,
        config["quantization"]["bits"],
        config["quantization"]["group_size"],
    )

    # Load weights via mmap — this is instant
    weight_files = _get_v2_weight_files(path)
    logger.info(f"  Loading {len(weight_files)} safetensors shards via mmap")

    # Detect model-specific weight transforms
    _model_cfg = json.loads((path / "config.json").read_text())
    _top_type = _model_cfg.get("model_type", "")
    _text_type = _model_cfg.get("text_config", {}).get("model_type", "")
    _is_nemotron = _top_type == "nemotron_h"
    _is_mistral4 = _text_type == "mistral4"
    # Nemotron + Mistral 4 need gate dequantization
    _needs_gate_dequant = _is_nemotron or _is_mistral4
    _nemotron_renames = {
        "switch_mlp.up_proj": "switch_mlp.fc1",
        "switch_mlp.down_proj": "switch_mlp.fc2",
    }
    # Mistral 4 MLA: kv_b_proj must be split into embed_q + unembed_out
    _mistral4_cfg = None
    if _is_mistral4:
        _t_cfg = _model_cfg.get("text_config", _model_cfg)
        _mistral4_cfg = {
            "nheads": _t_cfg.get("num_attention_heads", 32),
            "qk_nope": _t_cfg.get("qk_nope_head_dim", 64),
            "v_head": _t_cfg.get("v_head_dim", 128),
            "kv_rank": _t_cfg.get("kv_lora_rank", 256),
            "nlayers": _t_cfg.get("num_hidden_layers", 36),
        }
        _mistral4_cfg["head_dim"] = _mistral4_cfg["qk_nope"] + _mistral4_cfg["v_head"]

    for sf in weight_files:
        weights = mx.load(str(sf))
        weights = {k: v for k, v in weights.items()
                   if not k.endswith(".importance") and not k.startswith("mtp.")}
        if hasattr(model, "sanitize"):
            weights = model.sanitize(weights)
        weights = _sanitize_qwen3_next_conv1d_layout(weights)

        # Nemotron-H: rename switch_mlp keys + dequantize gate weights
        if _needs_gate_dequant:
            renamed = {}
            gate_parts = {}  # prefix -> {scales, biases}
            for k, v in weights.items():
                new_k = k
                # Collect MoE ROUTER gate scales/biases for dequantization
                # Matches: .mlp.gate. (Mistral 4) and .mixer.gate. (Nemotron-H)
                # Must NOT match gate_proj or switch_mlp.gate_proj
                _is_gate_meta = ((".mlp.gate." in k or ".mixer.gate." in k)
                                 and "gate_proj" not in k
                                 and (k.endswith(".scales") or k.endswith(".biases")))
                if _is_gate_meta:
                    prefix = k.rsplit(".", 1)[0]
                    gate_parts.setdefault(prefix, {})[k.rsplit(".", 1)[1]] = v
                    continue
                # Apply fc1/fc2 rename (Nemotron-H only)
                if _is_nemotron:
                    for old, new in _nemotron_renames.items():
                        if old in k:
                            new_k = k.replace(old, new)
                            break
                renamed[new_k] = v
            # Dequantize gate weights (uint32 packed → bfloat16 float)
            # Gate is MoEGate with raw weight param. bfloat16 preserves routing
            # precision that float16 cannot (MoE routing is extremely sensitive).
            # Note: bf16 gate causes float32 promotion (~23 tok/s instead of ~40).
            # Speed fix requires re-conversion with f16-compatible gate quantization.
            for prefix, parts in gate_parts.items():
                wkey = f"{prefix}.weight"
                if wkey in renamed and "scales" in parts:
                    qw = renamed[wkey]
                    scales = parts["scales"]
                    biases = parts.get("biases", mx.zeros_like(scales))
                    for bits in [8, 6, 4, 3, 2]:
                        elem_per_u32 = 32 // bits
                        real_cols = qw.shape[-1] * elem_per_u32
                        gs = real_cols // scales.shape[-1] if scales.shape[-1] > 0 else 0
                        if gs > 0 and gs * scales.shape[-1] == real_cols:
                            try:
                                dq = mx.dequantize(qw, scales, biases, gs, bits)
                                mx.eval(dq)
                                # Keep float32 for maximum routing precision.
                                # bfloat16 loses 3 mantissa bits → breaks MoE expert selection.
                                # float16 also breaks on some models.
                                # float32 is safe at cost of some speed (float32 promotion).
                                renamed[wkey] = dq
                                logger.info(f"  Dequantized gate: {wkey} bits={bits} gs={gs} → {dq.shape} (f32)")
                                break
                            except Exception:
                                continue
            weights = renamed

        # Mistral 4 MLA: split kv_b_proj into embed_q + unembed_out
        if _mistral4_cfg:
            mc = _mistral4_cfg
            keys_to_remove = []
            new_weights = {}
            kv_count = sum(1 for k in weights if "kv_b_proj" in k)
            kv_w = sum(1 for k in weights if ".kv_b_proj.weight" in k)
            if kv_count > 0:
                print(f"  MLA: {kv_count} kv_b_proj keys ({kv_w} weights) in this shard")
                sample = [k for k in weights if "kv_b_proj" in k][:3]
                print(f"    Sample: {sample}")
            for k, v in weights.items():
                if ".kv_b_proj.weight" not in k:
                    continue
                pfx = k.replace(".kv_b_proj.weight", "")
                # Dequantize if quantized
                s_key = f"{pfx}.kv_b_proj.scales"
                b_key = f"{pfx}.kv_b_proj.biases"
                if s_key in weights:
                    s, b = weights[s_key], weights.get(b_key, mx.zeros_like(weights[s_key]))
                    for try_bits in [8, 6, 4, 3, 2]:
                        elem = 32 // try_bits
                        real = v.shape[-1] * elem
                        gs = real // s.shape[-1] if s.shape[-1] > 0 else 0
                        if gs > 0 and gs * s.shape[-1] == real:
                            try:
                                v = mx.dequantize(v, s, b, gs, try_bits)
                                break
                            except Exception:
                                continue
                    keys_to_remove.extend([s_key, b_key])
                # Reshape: (nheads*head_dim, kv_rank) → (nheads, head_dim, kv_rank)
                v = v.reshape(mc["nheads"], mc["head_dim"], mc["kv_rank"])
                # embed_q: key projection (nheads, kv_rank, qk_nope)
                wk = mx.contiguous(v[:, :mc["qk_nope"], :].swapaxes(-1, -2))
                # unembed_out: value projection (nheads, v_head, kv_rank)
                wv = mx.contiguous(v[:, mc["qk_nope"]:, :])
                new_weights[f"{pfx}.embed_q.weight"] = wk.astype(mx.float16)
                new_weights[f"{pfx}.unembed_out.weight"] = wv.astype(mx.float16)
                keys_to_remove.append(k)
                logger.info(f"  Split kv_b_proj: embed_q={wk.shape}, unembed_out={wv.shape}")
            for k in keys_to_remove:
                weights.pop(k, None)
            weights.update(new_weights)

        model.load_weights(list(weights.items()), strict=False)
        del weights
        gc.collect()

    _fix_quantized_bits(model, {})

    if not hasattr(model, "config"):
        model.config = config

    # ── bfloat16 compute for 512+ expert models ──────────────────
    # Models with 512+ experts and hidden_size>=4096 overflow float16
    # at the shared expert down_proj (SiLU*up product → 4096-dim dot product
    # exceeds float16 max 65504). bfloat16 has same range as float32
    # (max 3.4e38) so it handles this without any quality loss.
    # (float16 overflow on 512+ expert models; bf16 required).
    _model_cfg = json.loads((path / "config.json").read_text())
    _text_cfg = _model_cfg.get("text_config", _model_cfg)
    _n_experts = (_text_cfg.get("num_experts") or
                    _text_cfg.get("num_local_experts") or
                    _text_cfg.get("n_routed_experts") or 0)
    _hidden = _text_cfg.get("hidden_size", 0)
    if _n_experts >= 512 and _hidden >= 4096:
        model.set_dtype(mx.bfloat16)
        logger.info(f"  bfloat16 enabled: {_n_experts} experts, hidden={_hidden} (float16 overflow prevention)")
    # Mistral 4 (and models with MLA + MoE): gate weight is bfloat16 from dequant.
    # Setting entire model to bfloat16 prevents mixed-dtype float32 promotion
    # that halves throughput (23 tok/s → 75+ tok/s). Gate routing requires bf16
    # precision — even f16 breaks expert selection.
    elif _text_cfg.get("model_type") == "mistral4" or _text_cfg.get("kv_lora_rank", 0) > 0:
        model.set_dtype(mx.bfloat16)
        logger.info(f"  bfloat16 enabled: MLA model with bf16 gate (speed optimization)")

    mx.eval(model.parameters())

    # ── TurboQuant KV cache (JANG-exclusive) ──────────────────
    _tq_cfg = jang_cfg.get("turboquant")
    if _tq_cfg and _tq_cfg.get("enabled", False):
        if os.environ.get("VMLX_DISABLE_TQ_KV") == "1":
            logger.info("  TurboQuant KV disabled by VMLX_DISABLE_TQ_KV=1")
        else:
            try:
                from .turboquant.config import TurboQuantConfig, make_turboquant_cache
                try:
                    _native_cache = model.make_cache()
                    _native_cache_types = [type(c).__name__ for c in _native_cache]
                    n_layers = len(_native_cache)
                    del _native_cache
                except Exception:
                    n_layers = len(model.layers)
                    _native_cache_types = []
                tq_config = TurboQuantConfig.from_jang_config(jang_cfg, n_layers)
                if tq_config:
                    _key_dim = _text_cfg.get("head_dim", 128)
                    _val_dim = _text_cfg.get("head_dim", 128)
                    if _text_cfg.get("kv_lora_rank", 0) > 0:
                        _key_dim = _text_cfg.get("qk_nope_head_dim", 128) + _text_cfg.get("qk_rope_head_dim", 64)
                        _val_dim = _text_cfg.get("v_head_dim", 128)
                    # Detect hybrid layer types from config
                    _layer_type_list = _text_cfg.get("layer_types", [])
                    _hybrid_pattern = _text_cfg.get("hybrid_override_pattern",
                                        _model_cfg.get("hybrid_override_pattern", ""))
                    try:
                        _logical_layers = int(
                            _text_cfg.get("num_hidden_layers")
                            or _model_cfg.get("num_hidden_layers")
                            or len(getattr(model, "layers", []) or [])
                            or n_layers
                        )
                    except Exception:
                        _logical_layers = n_layers
                    if _layer_type_list:
                        # Qwen3.5 style: explicit layer_types list
                        _layer_types = [
                            "attention" if lt == "full_attention" else "ssm"
                            for lt in _layer_type_list[:_logical_layers]
                        ]
                        while len(_layer_types) < _logical_layers:
                            _layer_types.append("attention")
                    elif _hybrid_pattern:
                        # Nemotron pattern: M=Mamba(SSM), *=attention, E=MoE(MLP), -=MLP
                        # Only M and * get cache entries. E and - are pure MLP (no cache).
                        # Must match model's make_cache() which skips E/- layers.
                        _layer_types = []
                        for ch in _hybrid_pattern[:_logical_layers]:
                            if ch == "M":
                                _layer_types.append("ssm")
                            elif ch == "*":
                                _layer_types.append("attention")
                            # E and - layers: no cache entry (skip)
                    else:
                        _layer_types = ["attention"] * n_layers
                    if len(_layer_types) != n_layers:
                        if _native_cache_types:
                            _layer_types = [
                                "ssm" if t in ("ArraysCache", "MambaCache", "BatchMambaCache")
                                else "attention"
                                for t in _native_cache_types
                            ]
                            logger.warning(
                                "  TurboQuant cache layout inferred from native "
                                "make_cache types (%d slots)",
                                n_layers,
                            )
                        else:
                            logger.warning(
                                "  TurboQuant cache layout mismatch: detector produced "
                                "%d entries for %d native cache slots; falling back "
                                "to all-attention",
                                len(_layer_types),
                                n_layers,
                            )
                            _layer_types = ["attention"] * n_layers
                    _n_attn = sum(1 for t in _layer_types if t == "attention")
                    _n_ssm = sum(1 for t in _layer_types if t == "ssm")
                    _n_cache = len(_layer_types)  # may be < n_layers for Nemotron
                    _n_skip = max(0, _logical_layers - _n_cache)
                    if _n_ssm > 0 or _n_skip > 0:
                        logger.info(f"  Hybrid model: {_n_attn} attention + {_n_ssm} SSM"
                                    + (f" + {_n_skip} no-cache" if _n_skip else "") + " layers")
                    def _turboquant_make_cache(
                        _cfg=tq_config, _n=_n_cache, _kd=_key_dim, _vd=_val_dim, _lt=_layer_types
                    ):
                        return make_turboquant_cache(_cfg, _n, [_kd]*_n, [_vd]*_n, _lt)
                    model.make_cache = _turboquant_make_cache
                    logger.info(f"  TurboQuant enabled: {tq_config.default_key_bits}-bit keys, "
                                f"{tq_config.default_value_bits}-bit values, "
                                f"{len(tq_config.critical_layers)} critical layers")
            except ImportError:
                logger.warning("  TurboQuant config found but turboquant module not available")

    # ── Hadamard rotation (input pre-rotation for rotated weights) ──
    # If model was converted with --hadamard, weight rows are Hadamard-rotated.
    # At inference, we must pre-rotate each linear layer's input with the same
    # transform. Math: y = hadamard_rotate(x, signs) @ W_rot^T = x @ W^T (exact)
    # The rotation cancels (signs^2 = 1), but quantization error is smaller
    # because the rotated weights have fewer outliers.
    _has_hadamard = jang_cfg.get("quantization", {}).get("hadamard_rotation", False)
    if _has_hadamard:
        _patch_hadamard_rotation(model)
        logger.info("  Hadamard rotation enabled: input pre-rotation on quantized layers")

    elapsed = time.perf_counter() - start

    actual_bits = jang_cfg.get("quantization", {}).get("actual_bits", 0)
    source_model = jang_cfg.get("source_model", {}).get("name", "unknown")
    logger.info(
        f"JANG v2 loaded in {elapsed:.1f}s: {source_model} "
        f"({actual_bits:.1f}-bit avg)"
    )

    tokenizer = load_tokenizer(
        path, eos_token_ids=config.get("eos_token_id", None)
    )
    return model, tokenizer


def _load_jang_v2_vlm(path: Path, jang_cfg: dict):
    """Load a JANG v2 Vision-Language model via mmap — instant."""
    import mlx.nn as nn
    from mlx_vlm.utils import (
        get_model_and_args, load_config as vlm_load_config,
        update_module_configs, load_image_processor,
        load_processor, skip_multimodal_module,
    )

    start = time.perf_counter()

    block_size = _jang_quant_block_size(jang_cfg)
    default_bits = _jang_default_bits(jang_cfg, [4])

    config = vlm_load_config(path)
    quant_mode = (
        (config.get("quantization") or {}).get("mode")
        or (jang_cfg.get("quantization") or {}).get("mode")
        or "affine"
    )
    if (
        quant_mode in {"mxfp4", "mxfp8", "nvfp4"}
        and jang_cfg.get("runtime", {}).get("bundle_has_mtp")
        and _has_indexed_mtp_weights(path)
        and _first_safetensors_metadata(path).get("format") == "mlx"
    ):
        return _load_jang_v2_vlm_native_mlx(path, jang_cfg)

    model_class, _ = get_model_and_args(config=config)

    config.setdefault("text_config", {})
    config.setdefault("vision_config", {})
    config.setdefault("audio_config", {})

    model_config = model_class.ModelConfig.from_dict(config)
    modules = ["text", "vision", "perceiver", "projector", "audio"]
    model_config = update_module_configs(model_config, model_class, config, modules)
    model = model_class.Model(model_config)

    # Quantize ALL layers that support it (same approach as mlx_lm).
    # Use model's quant_predicate if available (e.g., Mistral 4 needs 8-bit gate)
    _lang_model = getattr(model, 'language_model', model)
    _quant_pred = getattr(_lang_model, 'quant_predicate', None)

    def get_class_predicate(p, m):
        if skip_multimodal_module(p):
            return False
        if _module_path_is_raw_adapter(p):
            return False
        if not hasattr(m, "to_quantized"):
            return False
        if not _module_can_quantize_with_group_size(m, block_size):
            return False
        # Model-specific quantization (e.g., Mistral 4: gate at 8-bit)
        if _quant_pred is not None:
            result = _quant_pred(p, m)
            if isinstance(result, dict):
                return result  # e.g., {"group_size": 64, "bits": 8}
            return result
        return True

    nn.quantize(model, group_size=block_size, bits=default_bits, mode=quant_mode,
                class_predicate=get_class_predicate)

    # Load weights via mmap, stripping .importance keys (calibration-only)
    weight_files = _get_v2_weight_files(path)
    from mlx_vlm.utils import sanitize_weights
    for sf in weight_files:
        shard_weights = mx.load(str(sf))
        shard_weights = {k: v for k, v in shard_weights.items()
                         if not k.endswith(".importance")}
        # JANG v2 weights are already in MLX naming (switch_mlp, split gate/up).
        # model.sanitize() expects HuggingFace naming and will crash on MoE models
        # because it tries to split gate_up_proj which is already split.
        # On failure, apply a minimal sanitize (rename + transpose + norm fix)
        # that skips the gate_up splitting.
        # JANG MoE expert remap: switch_mlp.* → experts.switch_glu.*
        # Only for models that use experts.switch_glu (Gemma 4).
        # Mistral 4 uses mlp.switch_mlp directly — do NOT remap.
        _text_mt_vlm = config.get("text_config", {}).get("model_type", "")
        _needs_expert_remap = _text_mt_vlm != "mistral4"
        remapped = {}
        for k, v in shard_weights.items():
            if _needs_expert_remap and ".switch_mlp." in k:
                remapped[k.replace(".switch_mlp.", ".experts.switch_glu.")] = v
            else:
                remapped[k] = v
        shard_weights = remapped

        sanitize_ok = False
        if hasattr(model, "sanitize"):
            try:
                shard_weights = model.sanitize(shard_weights)
                sanitize_ok = True
            except (KeyError, ValueError):
                pass
        if not sanitize_ok:
            # Minimal sanitize: rename keys, transpose conv1d, fix norms
            norm_suffixes = (
                ".input_layernorm.weight", ".post_attention_layernorm.weight",
                "model.norm.weight", ".q_norm.weight", ".k_norm.weight",
            )
            fixed = {}
            for k, v in shard_weights.items():
                # Strip mtp weights
                if "mtp." in k:
                    continue
                # Rename HF → MLX conventions
                if "model.language_model" in k:
                    k = k.replace("model.language_model", "language_model.model")
                elif "model.visual" in k:
                    k = k.replace("model.visual", "vision_tower")
                elif "lm_head" in k and "language_model" not in k:
                    k = k.replace("lm_head", "language_model.lm_head")
                # Transpose conv1d from HF (ch,1,kW) to MLX (ch,kW,1)
                if "conv1d.weight" in k and v.ndim == 3 and v.shape[-1] != 1:
                    v = mx.transpose(v, axes=(0, 2, 1))
                # Add 1.0 to norm weights (MLX convention)
                if any(k.endswith(s) for s in norm_suffixes) and v.ndim == 1:
                    v = v + 1.0
                fixed[k] = v
            shard_weights = fixed
        shard_weights = _sanitize_qwen3_next_conv1d_layout(shard_weights)
        try:
            shard_weights = sanitize_weights(
                model_class.VisionModel, shard_weights, model_config.vision_config)
            shard_weights = sanitize_weights(
                model_class.LanguageModel, shard_weights, model_config.text_config)
        except (KeyError, ValueError, AttributeError):
            pass  # Some model classes don't have VisionModel/LanguageModel
        # Dequantize vision conv weights that were incorrectly quantized by converter.
        # Conv layers (patch_embed, temporal_embed) need float weights, not uint32.
        for k in list(shard_weights.keys()):
            if ("patch_embed" in k or "temporal_embed" in k) and k.endswith(".weight"):
                w = shard_weights[k]
                if w.dtype == mx.uint32:
                    base = k[:-7]
                    s_key = f"{base}.scales"
                    b_key = f"{base}.biases"
                    if s_key in shard_weights and b_key in shard_weights:
                        s = shard_weights[s_key]
                        b = shard_weights[b_key]
                        # Infer bits and gs: w_cols * 32 / bits = s_cols * gs
                        # Try common combos
                        w_cols, s_cols = w.shape[-1], s.shape[-1]
                        dq = None
                        for try_bits in (2, 3, 4, 6, 8):
                            in_dim = w_cols * 32 // try_bits
                            if w_cols * 32 % try_bits != 0 or in_dim % s_cols != 0:
                                continue
                            try_gs = in_dim // s_cols
                            if try_gs >= 2:
                                try:
                                    dq = mx.dequantize(w, s, b, group_size=try_gs, bits=try_bits)
                                    break
                                except Exception:
                                    continue
                        if dq is not None:
                            shard_weights[k] = dq.astype(mx.float16)
                            del shard_weights[s_key], shard_weights[b_key]
        # Mistral 4 MLA: split kv_b_proj into embed_q + unembed_out
        # DISABLED: mlx_vlm now has native mistral4 model (Mistral4Model in
        # mlx_vlm/models/mistral4/language.py) that handles kv_b_proj natively.
        # Our split conflicts with the native model's quantized expectations.
        _text_mt = config.get("text_config", {}).get("model_type", "")
        if False and _text_mt == "mistral4":
            _t_cfg = config.get("text_config", config)
            _nheads = _t_cfg.get("num_attention_heads", 32)
            _qk_nope = _t_cfg.get("qk_nope_head_dim", 64)
            _v_head = _t_cfg.get("v_head_dim", 128)
            _kv_rank = _t_cfg.get("kv_lora_rank", 256)
            _head_dim = _qk_nope + _v_head
            new_kv = {}
            rm_kv = []
            for k, v in shard_weights.items():
                if ".kv_b_proj.weight" not in k:
                    continue
                pfx = k.replace(".kv_b_proj.weight", "")
                s_key = f"{pfx}.kv_b_proj.scales"
                b_key = f"{pfx}.kv_b_proj.biases"
                if s_key in shard_weights:
                    s = shard_weights[s_key]
                    b = shard_weights.get(b_key, mx.zeros_like(s))
                    for bits in [8, 6, 4, 3, 2]:
                        elem = 32 // bits
                        real = v.shape[-1] * elem
                        gs = real // s.shape[-1] if s.shape[-1] > 0 else 0
                        if gs > 0 and gs * s.shape[-1] == real:
                            try:
                                v = mx.dequantize(v, s, b, gs, bits)
                                break
                            except Exception:
                                continue
                    rm_kv.extend([s_key, b_key])
                v = v.reshape(_nheads, _head_dim, _kv_rank)
                wk = mx.contiguous(v[:, :_qk_nope, :].swapaxes(-1, -2))
                wv = mx.contiguous(v[:, _qk_nope:, :])
                new_kv[f"{pfx}.embed_q.weight"] = wk.astype(mx.float16)
                new_kv[f"{pfx}.unembed_out.weight"] = wv.astype(mx.float16)
                rm_kv.append(k)
                logger.info(f"  Split kv_b_proj: embed_q={wk.shape}, unembed_out={wv.shape}")
            for k in rm_kv:
                shard_weights.pop(k, None)
            shard_weights.update(new_kv)

        # MoE gate dequantization (Nemotron, Mistral 4, etc.)
        # Gate is nn.Linear quantized at 8-bit by nn.quantize, but JANG stores
        # gate as uint32+scales+biases. Dequant here so it loads as float.
        # After loading, we fix the gate module type (see post-load fixup below).
        _n_exp = config.get("text_config", config).get("n_routed_experts", 0) or config.get("text_config", config).get("num_local_experts", 0)
        if _n_exp > 0:
            gate_parts = {}
            gate_rm = []
            for k, v in shard_weights.items():
                if ".gate." in k and (k.endswith(".scales") or k.endswith(".biases")) and "gate_proj" not in k:
                    prefix = k.rsplit(".", 1)[0]
                    gate_parts.setdefault(prefix, {})[k.rsplit(".", 1)[1]] = v
                    gate_rm.append(k)
            for k in gate_rm:
                del shard_weights[k]
            for prefix, parts in gate_parts.items():
                wkey = f"{prefix}.weight"
                if wkey in shard_weights and "scales" in parts:
                    qw = shard_weights[wkey]
                    scales = parts["scales"]
                    biases = parts.get("biases", mx.zeros_like(scales))
                    for bits in [8, 6, 4, 3, 2]:
                        elem = 32 // bits
                        real = qw.shape[-1] * elem
                        gs = real // scales.shape[-1] if scales.shape[-1] > 0 else 0
                        if gs > 0 and gs * scales.shape[-1] == real:
                            try:
                                dq = mx.dequantize(qw, scales, biases, gs, bits)
                                mx.eval(dq)
                                shard_weights[wkey] = dq.astype(mx.bfloat16)
                                logger.info(f"  Dequantized gate: {wkey} bits={bits} gs={gs}")
                                break
                            except Exception:
                                continue

        model.load_weights(list(shard_weights.items()), strict=False)
        del shard_weights
        gc.collect()

    _fix_quantized_bits(model, {})

    # Post-load fixup: replace QuantizedLinear gates with plain Linear
    # when gate weights were dequantized to float by our loader.
    # nn.quantize created QuantizedLinear but we loaded float weights.
    if _n_exp > 0:
        _lang = getattr(model, 'language_model', model)
        _mdl = getattr(_lang, 'model', _lang)
        _layers = getattr(_mdl, 'layers', [])
        for _layer in _layers:
            _mlp = getattr(_layer, 'mlp', None)
            if _mlp is None:
                continue
            _gate = getattr(_mlp, 'gate', None)
            if _gate is None:
                continue
            _gw = getattr(_gate, 'weight', None)
            if _gw is not None and _gw.dtype != mx.uint32:
                # Gate was dequanted to float — replace QuantizedLinear with Linear
                new_gate = nn.Linear(_gw.shape[1], _gw.shape[0], bias=False)
                new_gate.weight = _gw
                _mlp.gate = new_gate

    if not hasattr(model, "config"):
        model.config = model_config

    # ── bfloat16 for 512+ expert models (same as text loader) ──
    _model_cfg = json.loads((path / "config.json").read_text())
    _text_cfg = _model_cfg.get("text_config", _model_cfg)
    _n_experts = (_text_cfg.get("num_experts") or
                    _text_cfg.get("num_local_experts") or
                    _text_cfg.get("n_routed_experts") or 0)
    _hidden = _text_cfg.get("hidden_size", 0)
    if _n_experts >= 512 and _hidden >= 4096:
        model.set_dtype(mx.bfloat16)
        logger.info(f"  bfloat16 enabled: {_n_experts} experts, hidden={_hidden}")

    mx.eval(model.parameters())
    # ── TurboQuant for VLM (patch language_model, not top-level model) ──
    _tq_cfg_vlm = jang_cfg.get("turboquant")
    if _tq_cfg_vlm and _tq_cfg_vlm.get("enabled", False):
        if os.environ.get("VMLX_DISABLE_TQ_KV") == "1":
            logger.info("  TurboQuant VLM KV disabled by VMLX_DISABLE_TQ_KV=1")
        else:
            try:
                from .turboquant.config import TurboQuantConfig, make_turboquant_cache
                _lang_model = getattr(model, "language_model", None)
                if _lang_model is not None and hasattr(_lang_model, "layers"):
                    _vlm_n_layers = len(_lang_model.layers)
                    _vlm_tq = TurboQuantConfig.from_jang_config(jang_cfg, _vlm_n_layers)
                    if _vlm_tq:
                        _vlm_model_cfg = json.loads((path / "config.json").read_text())
                        _vlm_text_cfg = _vlm_model_cfg.get("text_config", _vlm_model_cfg)
                        _vlm_kd = _vlm_text_cfg.get("head_dim", 128)
                        _vlm_vd = _vlm_text_cfg.get("head_dim", 128)
                        if _vlm_text_cfg.get("kv_lora_rank", 0) > 0:
                            _vlm_kd = _vlm_text_cfg.get("qk_nope_head_dim", 128) + _vlm_text_cfg.get("qk_rope_head_dim", 64)
                            _vlm_vd = _vlm_text_cfg.get("v_head_dim", 128)
                        _vlm_lt = ["attention"] * _vlm_n_layers
                        def _vlm_tq_make_cache(_c=_vlm_tq, _n=_vlm_n_layers, _k=_vlm_kd, _v=_vlm_vd, _t=_vlm_lt):
                            return make_turboquant_cache(_c, _n, [_k]*_n, [_v]*_n, _t)
                        _lang_model.make_cache = _vlm_tq_make_cache
                        logger.info(f"  TurboQuant VLM enabled: {_vlm_tq.default_key_bits}-bit keys")
            except ImportError:
                pass

    elapsed = time.perf_counter() - start
    logger.info(f"JANG v2 VLM loaded in {elapsed:.1f}s")

    image_processor = load_image_processor(path)
    eos_token_id = getattr(model.config, "eos_token_id", None)
    try:
        processor = load_processor(path, True, eos_token_ids=eos_token_id)
    except (ImportError, ValueError, OSError, KeyError, TypeError):
        processor = _build_vlm_processor(path, eos_token_id)
    if image_processor is not None:
        processor.image_processor = image_processor

    return model, processor


def _get_v2_weight_files(path: Path) -> list[Path]:
    """Get safetensors weight files for a v2 model."""
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        return [path / sf for sf in sorted(set(index["weight_map"].values()))]

    # Fallback: glob for standard safetensors
    files = sorted(path.glob("model-*.safetensors"))
    if not files:
        files = sorted(path.glob("*.safetensors"))
    return files


# ─── Public API ──────────────────────────────────────────────────────


def load_jang_vlm_model(model_path: str | Path):
    """
    Load a JANG Vision-Language model into mlx-vlm for multimodal inference.

    Automatically detects v2 (instant) or v1 (repack) format.

    Returns:
        Tuple of (model, processor) compatible with mlx-vlm.generate()
    """
    if not _MLX_AVAILABLE:
        raise ImportError(
            "Loading JANG VLM models requires MLX (Apple Silicon only). "
            "Install with: pip install 'jang[mlx]'"
        )
    path = Path(model_path)
    config_path = _find_config_path(path)
    if not config_path:
        raise FileNotFoundError(f"No JANG config found in {path}")

    # M151 (iter 74): use _read_config_or_raise so corrupt jang_config
    # produces a clean ValueError naming the file + decode location.
    jang_cfg = _read_config_or_raise(config_path, purpose="JANG config")
    fmt = jang_cfg.get("format")
    # M130 (iter 52): peer-helper parity with load_jang_model's format guard.
    # The text path uses a split check that gives a clearer "missing field"
    # vs "not a JANG model" message and adds a format_version sanity check.
    # Mirror it here so a future v3 artifact (or a bad format tag) produces
    # the same actionable message instead of confusing mlx_vlm internals.
    if not fmt:
        raise ValueError(
            f"JANG config {config_path.name} is missing 'format' field. "
            f"Expected one of: {', '.join(JANG_FORMAT_VALUES)}"
        )
    if fmt not in JANG_FORMAT_VALUES:
        raise ValueError(f"Not a JANG model: format='{fmt}' (expected {', '.join(JANG_FORMAT_VALUES)})")

    version = jang_cfg.get("format_version", "1.0")
    major = int(version.split(".")[0])
    if major > 2:
        raise ValueError(
            f"Unsupported JANG format version: {version} (this loader supports 1.x and 2.x)"
        )

    # v2: instant load
    if _is_v2_model(path):
        logger.info(f"JANG v2 VLM detected — loading via mmap (instant)")
        return _load_jang_v2_vlm(path, jang_cfg)

    # v1: repack path (legacy)
    logger.info(f"JANG v1 VLM detected — repacking (this takes a few minutes)")
    return _load_jang_v1_vlm(path, jang_cfg, config_path)


def load_jang_model(model_path: str | Path):
    """
    Load a JANG model for inference.

    Automatically detects v2 (instant) or v1 (repack) format.
    v2 loads in seconds via mx.load() mmap.
    v1 repacks JANG uint8 → MLX uint32 (takes 5-10 minutes for large models).

    Returns:
        Tuple of (model, tokenizer) compatible with mlx-lm
    """
    if not _MLX_AVAILABLE:
        raise ImportError(
            "Loading JANG models requires MLX (Apple Silicon only). "
            "Install with: pip install 'jang[mlx]'"
        )
    path = Path(model_path)
    config_path = _find_config_path(path)
    if not config_path:
        raise FileNotFoundError(f"No JANG config found in {path}")

    # M151 (iter 74): use _read_config_or_raise so corrupt jang_config
    # produces a clean ValueError naming the file + decode location.
    jang_cfg = _read_config_or_raise(config_path, purpose="JANG config")
    fmt = jang_cfg.get("format")
    if not fmt:
        raise ValueError(
            f"JANG config {config_path.name} is missing 'format' field. "
            f"Expected one of: {', '.join(JANG_FORMAT_VALUES)}"
        )
    if fmt not in JANG_FORMAT_VALUES:
        raise ValueError(f"Not a JANG model: format='{fmt}' (expected {', '.join(JANG_FORMAT_VALUES)})")

    version = jang_cfg.get("format_version", "1.0")
    major = int(version.split(".")[0])
    if major > 2:
        raise ValueError(
            f"Unsupported JANG format version: {version} (this loader supports 1.x and 2.x)"
        )

    # v2: instant load via mmap
    if _is_v2_model(path):
        logger.info(f"JANG v2 detected — loading via mmap (instant)")
        if _is_vlm_config(path):
            logger.info(f"  VLM model detected — using mlx_vlm loader")
            model, processor = _load_jang_v2_vlm(path, jang_cfg)
            tokenizer = getattr(processor, 'tokenizer', processor)
            return model, tokenizer
        try:
            return _load_jang_v2(path, jang_cfg)
        except ValueError as e:
            if "not supported" in str(e).lower():
                logger.info(f"  mlx_lm failed ({e}), trying mlx_vlm loader")
                model, processor = _load_jang_v2_vlm(path, jang_cfg)
                tokenizer = getattr(processor, 'tokenizer', processor)
                return model, tokenizer
            raise

    # v1: repack path (legacy)
    logger.info(f"JANG v1 detected — repacking to MLX format (this may take a few minutes)")
    return _load_jang_v1(path, jang_cfg, config_path)


# ─── v1 loader (legacy, repack) ─────────────────────────────────────


def _load_jang_v1(path: Path, jang_cfg: dict, config_path: Path):
    """Load a JANG v1 model by repacking weights from uint8 to uint32."""
    from mlx_lm.utils import load_config, load_model as _load_model_skeleton, load_tokenizer

    start = time.perf_counter()

    block_size = _jang_quant_block_size(jang_cfg)
    target_bits = jang_cfg.get("quantization", {}).get("target_bits", 4)
    actual_bits = jang_cfg.get("quantization", {}).get("actual_bits", target_bits)
    source_model = jang_cfg.get("source_model", {}).get("name", "unknown")

    logger.info(
        f"Loading JANG v1 model: {source_model} "
        f"({actual_bits:.1f}-bit avg, block_size={block_size})"
    )

    config = load_config(path)
    default_bits = _jang_default_bits(jang_cfg, [2, 4, 6, 8])
    config.pop("quantization", None)
    config.pop("quantization_config", None)
    config["quantization"] = {"group_size": block_size, "bits": default_bits}

    model, config = _load_model_skeleton(
        path, lazy=True, strict=False, model_config=config
    )
    _upgrade_switch_to_quantized(model, default_bits, block_size)

    result, tmp_dir = _repack_jang_to_mlx(path, block_size, config)

    try:
        if tmp_dir is not None:
            logger.info(f"  Loading {len(result)} repacked shards via mmap")
            for sf in result:
                shard_weights = mx.load(sf)
                if hasattr(model, "sanitize"):
                    shard_weights = model.sanitize(shard_weights)
                shard_weights = _sanitize_qwen3_next_conv1d_layout(shard_weights)
                model.load_weights(list(shard_weights.items()), strict=False)
                del shard_weights
                gc.collect()
        else:
            weights = result
            if hasattr(model, "sanitize"):
                weights = model.sanitize(weights)
            weights = _sanitize_qwen3_next_conv1d_layout(weights)
            model.load_weights(list(weights.items()), strict=False)
            del weights
            gc.collect()

        _fix_quantized_bits(model, {})
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if not hasattr(model, "config"):
        model.config = config

    mx.eval(model.parameters())
    elapsed = time.perf_counter() - start
    n_params = sum(
        p.size for p in model.parameters().values() if isinstance(p, mx.array)
    )
    logger.info(
        f"JANG v1 model loaded in {elapsed:.1f}s: "
        f"{n_params / 1e9:.1f}B params, {actual_bits:.1f}-bit avg"
    )

    tokenizer = load_tokenizer(
        path, eos_token_ids=config.get("eos_token_id", None)
    )
    return model, tokenizer


def _load_jang_v1_vlm(path: Path, jang_cfg: dict, config_path: Path):
    """Load a JANG v1 VLM model by repacking (legacy)."""
    import mlx.nn as nn
    from mlx_vlm.utils import (
        get_model_and_args, load_config as vlm_load_config,
        update_module_configs, load_image_processor,
        load_processor, skip_multimodal_module,
    )

    start = time.perf_counter()

    block_size = _jang_quant_block_size(jang_cfg)
    default_bits = _jang_default_bits(jang_cfg, [2, 4, 6, 8])
    source_model = jang_cfg.get("source_model", {}).get("name", "unknown")

    logger.info(f"Loading JANG v1 VLM: {source_model}")

    config = vlm_load_config(path)
    model_class, _ = get_model_and_args(config=config)

    config.setdefault("text_config", {})
    config.setdefault("vision_config", {})
    config.setdefault("audio_config", {})

    model_config = model_class.ModelConfig.from_dict(config)
    modules = ["text", "vision", "perceiver", "projector", "audio"]
    model_config = update_module_configs(model_config, model_class, config, modules)
    model = model_class.Model(model_config)

    result, tmp_dir = _repack_jang_to_mlx(path, block_size, config)

    try:
        # Quantize ALL layers that support it (same fix as v2 VLM path).
        # Don't check weight keys — model param names differ from file keys.
        def get_class_predicate(p, m):
            if skip_multimodal_module(p):
                return False
            if _module_path_is_raw_adapter(p):
                return False
            return (
                hasattr(m, "to_quantized")
                and _module_can_quantize_with_group_size(m, block_size)
            )

        nn.quantize(model, group_size=block_size, bits=default_bits,
                    class_predicate=get_class_predicate)

        from mlx_vlm.utils import sanitize_weights

        # Handle both streaming (list of file paths) and in-memory (dict) results
        if tmp_dir is not None:
            # Streaming: result is a list of shard file paths
            for sf in result:
                shard_weights = mx.load(sf)
                shard_weights = {k: v for k, v in shard_weights.items()
                                 if not k.endswith(".importance")}
                if hasattr(model, "sanitize"):
                    shard_weights = model.sanitize(shard_weights)
                shard_weights = _sanitize_qwen3_next_conv1d_layout(shard_weights)
                shard_weights = sanitize_weights(
                    model_class.VisionModel, shard_weights, model_config.vision_config)
                shard_weights = sanitize_weights(
                    model_class.LanguageModel, shard_weights, model_config.text_config)
                model.load_weights(list(shard_weights.items()), strict=False)
                del shard_weights
                gc.collect()
        else:
            # In-memory: result is a dict of weights
            weights = result
            weights = {k: v for k, v in weights.items()
                       if not k.endswith(".importance")}
            if hasattr(model, "sanitize"):
                weights = model.sanitize(weights)
            weights = _sanitize_qwen3_next_conv1d_layout(weights)
            weights = sanitize_weights(
                model_class.VisionModel, weights, model_config.vision_config)
            weights = sanitize_weights(
                model_class.LanguageModel, weights, model_config.text_config)
            model.load_weights(list(weights.items()), strict=False)
            del weights
            gc.collect()

        _fix_quantized_bits(model, {})
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if not hasattr(model, "config"):
        model.config = model_config

    mx.eval(model.parameters())
    elapsed = time.perf_counter() - start
    logger.info(f"JANG v1 VLM loaded in {elapsed:.1f}s")

    image_processor = load_image_processor(path)
    eos_token_id = getattr(model.config, "eos_token_id", None)
    try:
        processor = load_processor(path, True, eos_token_ids=eos_token_id)
    except (ImportError, ValueError, OSError, KeyError, TypeError):
        processor = _build_vlm_processor(path, eos_token_id)
    if image_processor is not None:
        processor.image_processor = image_processor

    return model, processor


# ─── v1 repack engine (unchanged from original) ─────────────────────


def _repack_jang_to_mlx(
    model_path: Path,
    block_size: int,
    config: dict,
) -> tuple[list[str], str]:
    """
    Load JANG v1 shards and repack quantized tensors into MLX format.
    Returns (shard_file_paths, tmp_dir_path) or (weights_dict, None).
    """
    from safetensors import safe_open

    INDEX_NAMES = ["model.jang.index.json", "model.jjqf.index.json", "model.mxq.index.json"]
    SHARD_GLOBS = ["*.jang.safetensors", "*.jjqf.safetensors", "*.mxq.safetensors"]
    SUFFIXES = (".qweight", ".scales", ".zeros", ".biases", ".bit_map", ".block_offsets", ".shape", ".bits")

    index_path = None
    for name in INDEX_NAMES:
        p = model_path / name
        if p.exists():
            index_path = p
            break

    shard_files = []
    if index_path:
        index = json.loads(index_path.read_text())
        shard_files = [model_path / sf for sf in sorted(set(index["weight_map"].values()))]
    else:
        for pattern in SHARD_GLOBS:
            shard_files.extend(sorted(model_path.glob(pattern)))

    shard_handles = {}
    tensor_to_shard = {}
    all_tensor_names = []

    for sf in shard_files:
        sf_str = str(sf)
        logger.info(f"  Indexing shard: {sf.name if hasattr(sf, 'name') else sf}")
        handle = safe_open(sf_str, framework="numpy")
        shard_handles[sf_str] = handle
        for key in handle.keys():
            tensor_to_shard[key] = sf_str
            all_tensor_names.append(key)

    class LazyTensors:
        def __getitem__(self, key):
            sf_str = tensor_to_shard[key]
            return shard_handles[sf_str].get_tensor(key)
        def __contains__(self, key):
            return key in tensor_to_shard
        def keys(self):
            return all_tensor_names
        def __iter__(self):
            return iter(all_tensor_names)
        def __len__(self):
            return len(all_tensor_names)

    raw_tensors = LazyTensors()

    if not raw_tensors:
        raise FileNotFoundError(f"No JANG weight files found in {model_path}")

    quantized_bases = set()
    non_quantized_names = []

    for name in raw_tensors:
        matched = False
        for suffix in SUFFIXES:
            if name.endswith(suffix):
                quantized_bases.add(name[: -len(suffix)])
                matched = True
                break
        if not matched:
            non_quantized_names.append(name)

    logger.info(
        f"  {len(quantized_bases)} quantized tensors, "
        f"{len(non_quantized_names)} non-quantized tensors"
    )

    import os
    try:
        total_ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except (ValueError, AttributeError):
        import subprocess
        total_ram = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize']).strip())

    model_disk_bytes = sum(sf.stat().st_size for sf in shard_files if sf.exists())
    ram_threshold = int(total_ram * 0.50)
    use_streaming = model_disk_bytes > ram_threshold

    if use_streaming:
        logger.info(f"  Streaming mode: model {model_disk_bytes/1e9:.0f} GB > 50% of {total_ram/1e9:.0f} GB RAM")
    else:
        logger.info(f"  In-memory mode: model {model_disk_bytes/1e9:.0f} GB fits in {total_ram/1e9:.0f} GB RAM")

    tmp_dir = None
    output_shards = []
    current_shard = {}
    current_bytes = 0
    shard_idx = 0
    bit_counts = {}

    if use_streaming:
        for candidate_dir in [str(model_path.parent), str(model_path), None]:
            try:
                tmp_dir = tempfile.mkdtemp(prefix=".jang_repack_", dir=candidate_dir)
                test_f = Path(tmp_dir) / ".write_test"
                test_f.write_text("ok")
                test_f.unlink()
                break
            except (OSError, PermissionError):
                if tmp_dir and Path(tmp_dir).exists():
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                tmp_dir = None
        if tmp_dir is None:
            tmp_dir = tempfile.mkdtemp(prefix="jang_repack_")

    import re
    _per_expert_2d_pattern = re.compile(
        r".+\.experts\.(\d+)\.(w[123]|gate_proj|up_proj|down_proj)\."
    )
    expert_buffer = {}

    def _flush_shard():
        nonlocal current_shard, current_bytes, shard_idx
        if not current_shard:
            return
        if not use_streaming:
            return
        shard_path = f"{tmp_dir}/shard_{shard_idx:04d}.safetensors"
        mx.eval(*current_shard.values())
        mx.save_safetensors(shard_path, current_shard)
        output_shards.append(shard_path)
        logger.info(f"  Flushed shard {shard_idx} ({current_bytes / 1e9:.1f} GB, {len(current_shard)} tensors)")
        shard_idx += 1
        current_shard = {}
        current_bytes = 0
        gc.collect()

    def _add_to_shard(key, arr):
        nonlocal current_bytes
        current_shard[key] = arr
        current_bytes += arr.nbytes
        if current_bytes >= _SHARD_FLUSH_BYTES:
            _flush_shard()

    for base in sorted(quantized_bases):
        qweight_raw = raw_tensors[f"{base}.qweight"]
        jang_scales = raw_tensors[f"{base}.scales"].astype(np.float32)
        biases_key = f"{base}.biases"
        zeros_key = f"{base}.zeros"
        if biases_key in raw_tensors:
            jang_biases_raw = raw_tensors[biases_key].astype(np.float32)
        elif zeros_key in raw_tensors:
            jang_zeros = raw_tensors[zeros_key].astype(np.float32)
            jang_biases_raw = -jang_scales * jang_zeros
        else:
            jang_biases_raw = np.zeros_like(jang_scales)

        n_blocks = len(jang_scales)

        bits_key = f"{base}.bits"
        if bits_key in raw_tensors:
            bits = int(raw_tensors[bits_key][0])
        elif f"{base}.bit_map" in raw_tensors:
            bits = int(raw_tensors[f"{base}.bit_map"][0])
        else:
            logger.warning(f"  No bits info for {base}, assuming 4-bit")
            bits = 4

        bit_counts[bits] = bit_counts.get(bits, 0) + n_blocks

        shape_key = f"{base}.shape"
        if shape_key in raw_tensors:
            shape = tuple(int(x) for x in raw_tensors[shape_key])
        else:
            total_weights = n_blocks * block_size
            shape = _infer_weight_shape(base, config, total_weights)

        is_3d = shape is not None and len(shape) >= 3
        if is_3d:
            num_experts = shape[0]
            expert_out = shape[1]
            in_dim = shape[-1]
            out_dim = num_experts * expert_out
        elif shape is not None:
            num_experts = 0
            expert_out = 0
            out_dim, in_dim = shape
        else:
            num_experts = 0
            expert_out = 0
            out_dim = n_blocks
            in_dim = block_size

        packed_bytes = qweight_raw.tobytes()
        pad_needed = (4 - len(packed_bytes) % 4) % 4
        if pad_needed:
            packed_bytes += b'\x00' * pad_needed
        mlx_qweight = np.frombuffer(packed_bytes, dtype=np.uint32)

        packed_per_row = (in_dim * bits + 31) // 32
        expected_len = out_dim * packed_per_row
        if len(mlx_qweight) < expected_len:
            mlx_qweight = np.pad(mlx_qweight, (0, expected_len - len(mlx_qweight)))
        mlx_qweight = mlx_qweight[:expected_len]

        if is_3d:
            mlx_qweight = mlx_qweight.reshape(num_experts, expert_out, packed_per_row)
        else:
            mlx_qweight = mlx_qweight.reshape(out_dim, packed_per_row)

        n_groups_per_row = (in_dim + block_size - 1) // block_size
        expected_groups = out_dim * n_groups_per_row
        jang_biases = jang_biases_raw

        if n_blocks < expected_groups:
            pad = expected_groups - n_blocks
            jang_scales = np.pad(jang_scales, (0, pad), constant_values=1.0)
            jang_biases = np.pad(jang_biases, (0, pad), constant_values=0.0)

        if is_3d:
            mlx_scales = jang_scales[:expected_groups].reshape(num_experts, expert_out, n_groups_per_row)
            mlx_biases = jang_biases[:expected_groups].reshape(num_experts, expert_out, n_groups_per_row)
        else:
            mlx_scales = jang_scales[:expected_groups].reshape(out_dim, n_groups_per_row)
            mlx_biases = jang_biases[:expected_groups].reshape(out_dim, n_groups_per_row)

        if shape is not None and len(shape) >= 3:
            weight_key = base
        else:
            weight_key = f"{base}.weight"

        if is_3d and "gate_up_proj" in base:
            mid = expert_out // 2
            gate_w = mlx_qweight[:, :mid, :]
            up_w = mlx_qweight[:, mid:, :]
            gate_s = mlx_scales[:, :mid, :]
            up_s = mlx_scales[:, mid:, :]
            gate_b = mlx_biases[:, :mid, :]
            up_b = mlx_biases[:, mid:, :]

            sw_prefix = base.replace("experts.gate_up_proj", "switch_mlp")
            _add_to_shard(f"{sw_prefix}.gate_proj.weight", mx.array(gate_w))
            _add_to_shard(f"{sw_prefix}.gate_proj.scales", mx.array(gate_s))
            _add_to_shard(f"{sw_prefix}.gate_proj.biases", mx.array(gate_b))
            _add_to_shard(f"{sw_prefix}.up_proj.weight", mx.array(up_w))
            _add_to_shard(f"{sw_prefix}.up_proj.scales", mx.array(up_s))
            _add_to_shard(f"{sw_prefix}.up_proj.biases", mx.array(up_b))
        elif is_3d and "down_proj" in base:
            sw_prefix = base.replace("experts.down_proj", "switch_mlp")
            _add_to_shard(f"{sw_prefix}.down_proj.weight", mx.array(mlx_qweight))
            _add_to_shard(f"{sw_prefix}.down_proj.scales", mx.array(mlx_scales))
            _add_to_shard(f"{sw_prefix}.down_proj.biases", mx.array(mlx_biases))
        elif not is_3d and "gate_up_proj" in base:
            mid = out_dim // 2
            gate_w = mlx_qweight[:mid, :]
            up_w = mlx_qweight[mid:, :]
            gate_s = mlx_scales[:mid, :]
            up_s = mlx_scales[mid:, :]
            gate_b = mlx_biases[:mid, :]
            up_b = mlx_biases[mid:, :]

            gate_base = base.replace("gate_up_proj", "gate_proj")
            up_base = base.replace("gate_up_proj", "up_proj")
            _add_to_shard(f"{gate_base}.weight", mx.array(gate_w))
            _add_to_shard(f"{gate_base}.scales", mx.array(gate_s))
            _add_to_shard(f"{gate_base}.biases", mx.array(gate_b))
            _add_to_shard(f"{up_base}.weight", mx.array(up_w))
            _add_to_shard(f"{up_base}.scales", mx.array(up_s))
            _add_to_shard(f"{up_base}.biases", mx.array(up_b))
        else:
            if _per_expert_2d_pattern.search(weight_key):
                scale_key = weight_key.replace('.weight', '') if '.weight' in weight_key else weight_key
                expert_buffer[weight_key] = mx.array(mlx_qweight)
                expert_buffer[f"{scale_key}.scales"] = mx.array(mlx_scales)
                expert_buffer[f"{scale_key}.biases"] = mx.array(mlx_biases)
            else:
                _add_to_shard(weight_key, mx.array(mlx_qweight))
                scale_key = weight_key.replace('.weight', '') if '.weight' in weight_key else weight_key
                _add_to_shard(f"{scale_key}.scales", mx.array(mlx_scales))
                _add_to_shard(f"{scale_key}.biases", mx.array(mlx_biases))

        del qweight_raw, jang_scales, jang_biases_raw, jang_biases, packed_bytes
        del mlx_qweight, mlx_scales, mlx_biases

    if expert_buffer:
        _stack_per_expert_weights(expert_buffer, config)
        for k, v in expert_buffer.items():
            _add_to_shard(k, v)
        expert_buffer.clear()
        gc.collect()

    for name in non_quantized_names:
        arr = raw_tensors[name]
        if arr.dtype == np.float32:
            _add_to_shard(name, mx.array(arr))
        elif arr.dtype == np.float16:
            _add_to_shard(name, mx.array(arr))
        else:
            _add_to_shard(name, mx.array(arr.astype(np.float16)))

    for handle in shard_handles.values():
        del handle
    shard_handles.clear()
    gc.collect()

    rename_keys = []
    rename_keys += [(k, "vision_tower" + k[len("model.visual"):]) for k in list(current_shard.keys()) if k.startswith("model.visual")]
    rename_keys += [(k, "language_model.model" + k[len("model.language_model"):]) for k in list(current_shard.keys()) if k.startswith("model.language_model")]
    for old_k, new_k in rename_keys:
        current_shard[new_k] = current_shard.pop(old_k)

    _flush_shard()
    _rename_keys_in_flushed_shards(output_shards, tmp_dir)

    total_blocks = sum(bit_counts.values())
    if total_blocks > 0:
        dist_str = ", ".join(
            f"{b}-bit: {c} ({100 * c // total_blocks}%)"
            for b, c in sorted(bit_counts.items())
        )
        logger.info(f"  Bit distribution: {dist_str}")

    if use_streaming:
        logger.info(f"  Repacked into {len(output_shards)} temp shards in {tmp_dir}")
        return output_shards, tmp_dir
    else:
        logger.info(f"  Repacked {len(current_shard)} tensors in memory")
        return current_shard, None


# ─── Shared helpers ──────────────────────────────────────────────────


def _rename_keys_in_flushed_shards(shard_paths, tmp_dir):
    for shard_path in shard_paths:
        data = mx.load(shard_path)
        needs_rewrite = False
        renamed = {}
        for k, v in data.items():
            if k.startswith("model.visual"):
                new_k = "vision_tower" + k[len("model.visual"):]
                renamed[new_k] = v
                needs_rewrite = True
            elif k.startswith("model.language_model"):
                new_k = "language_model.model" + k[len("model.language_model"):]
                renamed[new_k] = v
                needs_rewrite = True
            else:
                renamed[k] = v
        if needs_rewrite:
            mx.save_safetensors(shard_path, renamed)
        del data, renamed
        gc.collect()


def _stack_per_expert_weights(weights, config):
    import re
    expert_pattern = re.compile(
        r"(.+)\.experts\.(\d+)\.(w[123]|gate_proj|up_proj|down_proj)\.weight$"
    )
    expert_groups = {}
    for key in list(weights.keys()):
        m = expert_pattern.match(key)
        if m:
            prefix, expert_id, wtype = m.group(1), int(m.group(2)), m.group(3)
            group_key = (prefix, wtype)
            if group_key not in expert_groups:
                expert_groups[group_key] = {}
            expert_groups[group_key][expert_id] = key

    if not expert_groups:
        return

    name_map = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}

    for (prefix, wtype), experts in expert_groups.items():
        if len(experts) < 2:
            continue
        num_experts = max(experts.keys()) + 1
        new_name = name_map.get(wtype, wtype)
        sw_key = f"{prefix}.switch_mlp.{new_name}"

        to_stack = [weights.pop(experts[e]) for e in range(num_experts)]
        weights[f"{sw_key}.weight"] = mx.stack(to_stack)

        for suffix in [".scales", ".biases"]:
            parts = []
            found = True
            for e in range(num_experts):
                sk = experts.get(e, "").replace(".weight", "") + suffix
                if sk in weights:
                    parts.append(weights.pop(sk))
                else:
                    found = False
                    break
            if found and parts:
                weights[f"{sw_key}{suffix}"] = mx.stack(parts)

    if expert_groups:
        logger.info(f"  Stacked {len(expert_groups)} expert groups into QuantizedSwitchLinear format")


def _patch_hadamard_rotation(model):
    """
    Wrap quantized linear layers with Hadamard input pre-rotation.

    For models converted with --hadamard, the weight rows were rotated before
    quantization. At inference, we pre-rotate the input with the same Hadamard
    transform so the rotation cancels and the output is correct — but with
    less quantization error because the rotated weights have fewer outliers.

    Math: y = hadamard_rotate(x, signs) @ W_rot^T = x @ W^T
    The signs^2 cancel (±1² = 1), giving the exact identity.
    """
    import mlx.nn as nn
    from .turboquant.rotation import generate_random_signs, hadamard_rotate

    # Cache signs by dimension (deterministic from seed=42)
    _signs_cache = {}

    def _get_signs(dim):
        if dim not in _signs_cache:
            _signs_cache[dim] = generate_random_signs(dim, seed=42)
        return _signs_cache[dim]

    _wrapped_classes = {}

    def _make_wrapped_class(cls):
        if getattr(cls, "_jang_hadamard_wrapper", False):
            return cls
        if cls in _wrapped_classes:
            return _wrapped_classes[cls]

        class HadamardQuantizedLinear(cls):
            _jang_hadamard_wrapper = True

            def __call__(self, x, *args, **kwargs):
                signs = getattr(self, "_hadamard_signs", None)
                if signs is None:
                    return super().__call__(x, *args, **kwargs)
                return super().__call__(hadamard_rotate(x, signs), *args, **kwargs)

        HadamardQuantizedLinear.__name__ = f"Hadamard{cls.__name__}"
        HadamardQuantizedLinear.__qualname__ = f"Hadamard{cls.__qualname__}"
        _wrapped_classes[cls] = HadamardQuantizedLinear
        return HadamardQuantizedLinear

    def _iter_children(module):
        if isinstance(module, (list, tuple)):
            for idx, child in enumerate(module):
                yield str(idx), child
            return
        if isinstance(module, dict):
            for key, child in module.items():
                yield str(key), child
            return
        if hasattr(module, "items"):
            try:
                for key, child in module.items():
                    yield str(key), child
            except Exception:
                return

    def _wrap_linear(module, name_path):
        """Recursively find and wrap QuantizedLinear layers."""
        for attr_name, child in _iter_children(module):
            child_path = f"{name_path}.{attr_name}" if name_path else attr_name
            child_path_l = child_path.lower()

            # Skip: embeddings, norms, gates (not rotated during conversion)
            if (
                attr_name == "gate"
                or ".gate." in child_path_l
                or any(skip in child_path_l for skip in ["embed", "lm_head", "norm"])
            ):
                continue

            if isinstance(child, nn.QuantizedLinear):
                # Only rotate 3-bit+ layers (2-bit has too few levels to benefit)
                if child.bits < 3:
                    continue
                in_dim = child.weight.shape[-1] * (32 // child.bits)
                signs = _get_signs(in_dim)
                object.__setattr__(child, "_hadamard_signs", signs)
                object.__setattr__(child, "__class__", _make_wrapped_class(type(child)))
            elif hasattr(child, "items") or isinstance(child, (list, tuple, dict)):
                # Recurse into submodules
                _wrap_linear(child, child_path)

    # Walk model tree
    layers = model.model.layers if hasattr(model, "model") else getattr(model, "layers", [])
    for i, layer in enumerate(layers):
        if layer is not None:
            _wrap_linear(layer, f"layers.{i}")

    logger.info(f"  Hadamard signs cached for dims: {sorted(_signs_cache.keys())}")


def _upgrade_switch_to_quantized(model, bits, group_size):
    try:
        from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear
    except ImportError:
        return

    for name, module in model.named_modules():
        if not isinstance(module, SwitchLinear):
            continue
        ql = QuantizedSwitchLinear(
            module.input_dims, module.output_dims, module.num_experts,
            bias="bias" in module and module.bias is not None,
            group_size=group_size, bits=bits,
        )
        parts = name.rsplit('.', 1)
        if len(parts) == 2:
            parent = model
            for p in parts[0].split('.'):
                if p.isdigit():
                    parent = parent[int(p)]
                else:
                    parent = getattr(parent, p)
            setattr(parent, parts[1], ql)


def _fix_quantized_bits(model, weights):
    import mlx.nn as nn
    import mlx.core as mx
    try:
        from mlx_lm.models.switch_layers import QuantizedSwitchLinear
        quant_types = (nn.QuantizedLinear, nn.QuantizedEmbedding, QuantizedSwitchLinear)
    except ImportError:
        quant_types = (nn.QuantizedLinear, nn.QuantizedEmbedding)
    # MLA models (GLM-5.1, Mistral 4, DeepSeek V3) use QuantizedMultiLinear for embed_q/unembed_out.
    # sanitize() re-quantizes the split kv_b_proj at bits/group_size inferred from its raw shape
    # (bits=8 gs=64 for GLM-5.1) but never updates the module's runtime bits/group_size, which
    # stays at the global default (bits=2 for JANG_1L). Without this import, _fix_quantized_bits
    # silently skips the mismatch and L>1 prefill crashes at mla.py:76 quantized_matmul.
    try:
        from mlx_lm.models.mla import QuantizedMultiLinear
        quant_types = quant_types + (QuantizedMultiLinear,)
    except ImportError:
        pass

    for name, module in model.named_modules():
        if not isinstance(module, quant_types):
            continue
        if not hasattr(module, 'scales') or not hasattr(module, 'weight'):
            continue
        # MXFP4/MXFP8 detection: scales stored as uint8 (UE8M0 exponent).
        # Standard affine has float16/float32 scales. When we see uint8,
        # reconfigure the module for mxfp4 (if bits=4) or mxfp8 (if bits=8).
        try:
            if module.scales.dtype == mx.uint8:
                # Infer bits from weight/scale shape ratio. Native MXFP4/MXFP8
                # bundles store UE8M0 scales as uint8 and have no real affine
                # biases on disk. However nn.quantize() initializes QuantizedLinear
                # modules with a random affine `biases` placeholder before weights
                # are loaded. Treat uint8 scales as authoritative and remove that
                # placeholder, otherwise generation dispatches the affine uint8
                # kernel instead of the MXFP kernel.
                w_cols = module.weight.shape[-1]
                s_cols = module.scales.shape[-1]
                # mxfp4: weight is packed 8x per uint32 → in_dim = w_cols * 8
                # mxfp8: weight is packed 4x per uint32 → in_dim = w_cols * 4
                # scale block_size = 32 for both mxfp4 and mxfp8
                # in_dim = s_cols * 32
                in_dim_mxfp4 = w_cols * 8
                in_dim_mxfp8 = w_cols * 4
                if s_cols * 32 == in_dim_mxfp4:
                    module.mode = 'mxfp4'
                    module.bits = 4
                    module.group_size = 32
                elif s_cols * 32 == in_dim_mxfp8:
                    module.mode = 'mxfp8'
                    module.bits = 8
                    module.group_size = 32
                else:
                    continue
                # Ensure affine biases placeholder is removed so
                # quantized_matmul skips affine dequantization.
                if hasattr(module, 'biases'):
                    try:
                        del module.biases
                    except Exception:
                        module.biases = None
                continue  # done with this module
        except Exception as exc:
            logger.debug("QuantizedLinear metadata fast-path skipped for %s: %s", name, exc)
        try:
            # Infer actual bits and group_size from tensor shapes.
            # weight: (out, in_dim * bits / 32), scales: (out, in_dim / gs)
            # Equation: w_cols * 32 / bits == s_cols * gs
            # Multiple (bits, gs) pairs can satisfy this. Strategy:
            # 1. Try the module's current gs first (from config.json)
            # 2. Then try common gs values (64, 128)
            w_cols = module.weight.shape[-1]
            s_cols = module.scales.shape[-1]
            fixed = False
            logical_input_dims = getattr(module, "dims", None)
            if logical_input_dims is None:
                logical_input_dims = getattr(module, "input_dims", None)
            if logical_input_dims is not None:
                try:
                    logical_input_dims = int(logical_input_dims)
                except Exception:
                    logical_input_dims = None
            # Try group sizes in preference order.
            # Router/gate tensors prefer gs=64 (precision-critical).
            # Everything else prefers the module's initialized gs (from config.json).
            name_lower = name.lower()
            is_router = (".gate." in name_lower or name_lower.endswith(".gate")
                         or "shared_expert_gate" in name_lower)
            if logical_input_dims:
                gs_candidates = []
                for gs in (module.group_size, 64, 128, 32):
                    if (
                        gs not in gs_candidates
                        and logical_input_dims % int(gs) == 0
                        and s_cols * int(gs) == logical_input_dims
                    ):
                        gs_candidates.append(int(gs))
                for gs in (module.group_size, 64, 128, 32):
                    if gs not in gs_candidates:
                        gs_candidates.append(int(gs))
            elif is_router:
                gs_candidates = [64, module.group_size, 128]
            else:
                gs_candidates = [module.group_size]
                for gs in (64, 128):
                    if gs not in gs_candidates:
                        gs_candidates.append(gs)
            for try_gs in gs_candidates:
                in_dim = s_cols * try_gs
                if in_dim <= 0:
                    continue
                if (w_cols * 32) % in_dim != 0:
                    continue
                try_bits = (w_cols * 32) // in_dim
                if try_bits in (2, 3, 4, 5, 6, 8):
                    if logical_input_dims and in_dim != logical_input_dims:
                        continue
                    if try_bits != module.bits:
                        module.bits = try_bits
                    if try_gs != module.group_size:
                        module.group_size = try_gs
                    fixed = True
                    break
            if not fixed:
                # Last resort: try all valid bit widths with current gs
                in_dim = s_cols * module.group_size
                if in_dim > 0:
                    actual_bits = (w_cols * 32) // in_dim
                    if actual_bits != module.bits and actual_bits in (2, 3, 4, 5, 6, 8):
                        module.bits = actual_bits
        except Exception as exc:
            logger.debug("QuantizedLinear bits/group inference skipped for %s: %s", name, exc)


def _build_vlm_processor(model_path: Path, eos_token_id=None):
    from transformers import AutoTokenizer, AutoImageProcessor
    from transformers.processing_utils import ProcessorMixin
    from mlx_vlm.tokenizer_utils import load_tokenizer as vlm_load_tokenizer
    from mlx_vlm.utils import StoppingCriteria

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    image_processor = AutoImageProcessor.from_pretrained(model_path)

    config = json.loads((model_path / "config.json").read_text())
    model_type = config.get("model_type", "")

    tok_config_path = model_path / "tokenizer_config.json"
    chat_template = None
    if tok_config_path.exists():
        chat_template = json.loads(tok_config_path.read_text()).get("chat_template")

    processor = None
    try:
        from transformers.video_processing_utils import BaseVideoProcessor
        video_stub = BaseVideoProcessor()

        processor_classes = {}
        try:
            from transformers import Qwen3VLProcessor
            processor_classes["qwen3_5"] = Qwen3VLProcessor
            processor_classes["qwen3_5_moe"] = Qwen3VLProcessor
            processor_classes["qwen3_vl"] = Qwen3VLProcessor
        except ImportError:
            pass
        try:
            from transformers import Qwen2VLProcessor
            processor_classes["qwen2_vl"] = Qwen2VLProcessor
            processor_classes["qwen2_5_vl"] = Qwen2VLProcessor
        except ImportError:
            pass

        proc_class = processor_classes.get(model_type)
        if proc_class is not None:
            _orig = ProcessorMixin.check_argument_for_proper_class
            def _permissive(self, name, arg):
                if name == "video_processor":
                    return type(arg)
                return _orig(self, name, arg)
            ProcessorMixin.check_argument_for_proper_class = _permissive
            try:
                processor = proc_class(
                    image_processor=image_processor,
                    tokenizer=tokenizer,
                    video_processor=video_stub,
                    chat_template=chat_template,
                )
            finally:
                ProcessorMixin.check_argument_for_proper_class = _orig
    except Exception as exc:
        logger.warning(f"Could not construct VL processor: {exc}")

    if processor is None:
        class _SimpleVLMProcessor:
            def __init__(self, tok, ip):
                self.tokenizer = tok
                self.image_processor = ip
            def __call__(self, *a, **kw):
                return self.tokenizer(*a, **kw)
        processor = _SimpleVLMProcessor(tokenizer, image_processor)

    detokenizer_class = vlm_load_tokenizer(model_path, return_tokenizer=False)
    tokenizer_obj = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    processor.detokenizer = detokenizer_class(tokenizer_obj)

    final_eos = eos_token_id if eos_token_id is not None else getattr(tokenizer_obj, "eos_token_ids", None)
    criteria = StoppingCriteria(final_eos, tokenizer_obj)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.stopping_criteria = criteria
    else:
        processor.stopping_criteria = criteria

    return processor


def _infer_weight_shape(base_name, config, n_elements):
    tc = config.get("text_config", {})
    def _get(key, default=0):
        return config.get(key, tc.get(key, default))

    hidden = _get("hidden_size", 0)
    intermediate = _get("intermediate_size", 0)
    moe_intermediate = _get("moe_intermediate_size", intermediate)
    shared_expert_intermediate = _get("shared_expert_intermediate_size", moe_intermediate)
    num_heads = _get("num_attention_heads", 0)
    num_kv_heads = _get("num_key_value_heads", num_heads)
    head_dim = _get("head_dim", hidden // num_heads if num_heads else 0)
    vocab_size = _get("vocab_size", 0)

    name = base_name.lower()

    if "qkv_proj" in name:
        out = (num_heads + 2 * num_kv_heads) * head_dim
        return (out, hidden)
    elif "q_proj" in name:
        return (num_heads * head_dim, hidden)
    elif "k_proj" in name:
        return (num_kv_heads * head_dim, hidden)
    elif "v_proj" in name:
        return (num_kv_heads * head_dim, hidden)
    elif "o_proj" in name:
        return (hidden, num_heads * head_dim)
    elif ".experts." in name or ".shared_expert." in name:
        ei = shared_expert_intermediate if ".shared_expert." in name else (moe_intermediate if moe_intermediate else intermediate)
        if "gate_proj" in name or "up_proj" in name or "w1" in name or "w3" in name:
            return (ei, hidden)
        elif "down_proj" in name or "w2" in name:
            return (hidden, ei)
    elif "gate_up_proj" in name:
        return (2 * intermediate, hidden)
    elif "gate_proj" in name or "up_proj" in name or "w1" in name or "w3" in name:
        return (intermediate, hidden)
    elif "down_proj" in name or "w2" in name:
        return (hidden, intermediate)
    elif "embed_tokens" in name:
        return (vocab_size, hidden)
    elif "lm_head" in name:
        return (vocab_size, hidden)

    if n_elements > 0 and hidden > 0 and n_elements % hidden == 0:
        return (n_elements // hidden, hidden)

    logger.warning(f"  Could not infer shape for {base_name} ({n_elements} elements)")
    return None


# ─── Upgrade v1 → v2 ────────────────────────────────────────────────


def upgrade_v1_to_v2(model_path: str | Path) -> None:
    """
    Upgrade a JANG v1 model to v2 format in-place.

    Repacks uint8 qweight → uint32 MLX-native, then replaces
    the .jang.safetensors files with standard .safetensors files.
    After upgrade, the model loads instantly via mx.load() mmap.

    Args:
        model_path: path to JANG v1 model directory
    """
    if not _MLX_AVAILABLE:
        raise ImportError(
            "Upgrading JANG models requires MLX (Apple Silicon only). "
            "Install with: pip install 'jang[mlx]'"
        )
    path = Path(model_path)
    config_path = _find_config_path(path)
    if not config_path:
        raise FileNotFoundError(f"No JANG config found in {path}")

    if _is_v2_model(path):
        print(f"  Already v2 format: {path}")
        return

    jang_cfg = json.loads(config_path.read_text())
    block_size = _jang_quant_block_size(jang_cfg)

    # Load model config
    model_config = json.loads((path / "config.json").read_text())
    config = dict(model_config)
    tc = config.get("text_config", {})
    for key in ["hidden_size", "intermediate_size", "num_attention_heads",
                 "num_key_value_heads", "head_dim", "vocab_size",
                 "moe_intermediate_size", "shared_expert_intermediate_size"]:
        if key in tc and key not in config:
            config[key] = tc[key]

    print(f"  Upgrading JANG v1 → v2: {path}")
    print(f"  This repacks {sum(1 for _ in path.glob('*.jang.safetensors'))} JANG shards to MLX-native format...")

    # Run the v1 repack to get MLX-format tensors
    result, tmp_dir = _repack_jang_to_mlx(path, block_size, config)

    try:
        # Collect all repacked tensors, stripping .importance (calibration-only)
        all_tensors = {}
        if tmp_dir is not None:
            for sf in result:
                data = mx.load(sf)
                for k, v in data.items():
                    if not k.endswith(".importance"):
                        all_tensors[k] = np.array(v)
                del data
                gc.collect()
        else:
            for k, v in result.items():
                if not k.endswith(".importance"):
                    all_tensors[k] = np.array(v)
            del result
            gc.collect()

        print(f"  Repacked {len(all_tensors)} tensors to MLX-native format")

        # Write v2 safetensors
        from safetensors.numpy import save_file

        # Shard into ~5 GB files
        max_shard = 5 * 1024 ** 3
        shards = []
        current_shard = {}
        current_size = 0
        weight_map = {}
        total_size = 0
        shard_idx = 0

        for name in sorted(all_tensors.keys()):
            arr = all_tensors[name]
            arr_bytes = arr.nbytes
            if current_size + arr_bytes > max_shard and current_shard:
                n_shards_est = max(1, sum(a.nbytes for a in all_tensors.values()) // max_shard + 1)
                shard_name = f"model-{shard_idx + 1:05d}-of-{n_shards_est:05d}.safetensors"
                shards.append((shard_name, current_shard))
                shard_idx += 1
                current_shard = {}
                current_size = 0
            current_shard[name] = arr
            current_size += arr_bytes
            total_size += arr_bytes

        if current_shard:
            shard_idx += 1
            shards.append((f"placeholder", current_shard))

        # Fix shard names with correct total count
        n_shards = len(shards)
        final_shards = []
        for i, (_, shard_data) in enumerate(shards):
            shard_name = f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors"
            final_shards.append((shard_name, shard_data))
            for tensor_name in shard_data:
                weight_map[tensor_name] = shard_name

        # Write new safetensors files
        for shard_name, shard_data in final_shards:
            save_file(shard_data, str(path / shard_name), metadata={"format": "mlx"})
            print(f"  Wrote {shard_name} ({sum(a.nbytes for a in shard_data.values()) / 1e9:.1f} GB)")

        # Write v2 index
        index = {
            "metadata": {
                "format": "jang",
                "format_version": "2.0",
                "total_size": total_size,
            },
            "weight_map": weight_map,
        }
        (path / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2) + "\n"
        )

        # Update config.json with quantization key
        model_config["quantization"] = {
            "group_size": block_size,
            "bits": _jang_default_bits(jang_cfg, [4]),
        }
        (path / "config.json").write_text(
            json.dumps(model_config, indent=2, ensure_ascii=False) + "\n"
        )

        # Update jang_config.json version
        jang_cfg["format_version"] = "2.0"
        config_path.write_text(
            json.dumps(jang_cfg, indent=2, ensure_ascii=False) + "\n"
        )

        # Remove old v1 files
        old_files = list(path.glob("*.jang.safetensors"))
        old_files += list(path.glob("*.jjqf.safetensors"))
        old_files += list(path.glob("*.mxq.safetensors"))
        for old_idx_name in ["model.jang.index.json", "model.jjqf.index.json", "model.mxq.index.json"]:
            old_idx = path / old_idx_name
            if old_idx.exists():
                old_files.append(old_idx)

        for f in old_files:
            f.unlink()
            print(f"  Removed {f.name}")

        print(f"\n  Upgrade complete! Model now loads instantly via mx.load() mmap.")
        print(f"  v2 shards: {n_shards}, total: {total_size / 1e9:.1f} GB")

    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
