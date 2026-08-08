"""
JANG Format Writer — Write quantized models to JANG format.
Created by Jinho Jang (eric@jangq.ai)

v2.0: Writes MLX-native safetensors (uint32 weights, float16 scales/biases).
      Models load instantly via mx.load() mmap — no repack needed.
v1.x: Legacy format with uint8 qweight packing (kept for backward compat).
"""

import json
import os
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from .spec import (
    FORMAT_NAME,
    FORMAT_VERSION,
    FORMAT_VERSION_V1,
    JANG_CONFIG_FILENAME,
    JANG_IMATRIX_FILENAME,
    JANG_INDEX_FILENAME,
    JANG_V2_INDEX_FILENAME,
)
from ..quantize import QuantizedTensor


def write_jang_v2_model(
    output_dir: str | Path,
    tensors: dict[str, np.ndarray],
    model_config: dict,
    jang_config: dict,
    tokenizer_files: dict[str, str | dict] | None = None,
    importance_data: dict[str, np.ndarray] | None = None,
    max_shard_bytes: int = 5 * 1024 ** 3,  # 5 GB per shard
    preflushed_map: dict[str, str] | None = None,
) -> None:
    """
    Write a JANG v2 model — MLX-native safetensors format.

    Tensors are stored in the exact format MLX expects:
    - .weight: uint32 packed quantized data (shaped, not flat)
    - .scales: float16 per-group scale factors (shaped)
    - .biases: float16 per-group biases (shaped)

    Loading is instant via mx.load() mmap — no repacking step.

    Args:
        output_dir: output directory path
        tensors: flat dict of tensor_name -> numpy array (MLX-format)
                 e.g. {"layers.0.self_attn.q_proj.weight": uint32_array,
                        "layers.0.self_attn.q_proj.scales": float16_array, ...}
        model_config: HuggingFace model config dict
        jang_config: JANG quantization config dict
        tokenizer_files: dict of filename → content (str or dict for JSON)
        importance_data: dict of tensor_name → importance scores array
        max_shard_bytes: maximum bytes per shard file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add quantization key to model config so mlx_lm creates QuantizedLinear layers.
    # Mixed-bit JANG bundles must expose per-module overrides here. mlx-lm's
    # quantization predicate reads config["quantization"][module_path] directly;
    # a uniform top-level bit width forces loader-side shape repair and can build
    # the wrong QuantizedLinear shape before weights are loaded.
    bit_widths = jang_config.get("quantization", {}).get("bit_widths_used", [4])
    block_size = jang_config.get("quantization", {}).get("block_size", 64)
    default_bits = min(bit_widths)
    manifest = jang_config.get("quantization", {}).get("tensor_quantization_manifest", {})

    model_config_out = dict(model_config)
    quantization = {
        "group_size": block_size,
        "bits": default_bits,
        "mode": "affine",
    }
    if isinstance(manifest, dict):
        for module_path, spec in manifest.items():
            if not isinstance(module_path, str) or not isinstance(spec, dict):
                continue
            bits = spec.get("bits")
            group_size = spec.get("group_size")
            if isinstance(bits, int) and isinstance(group_size, int):
                module_quantization = {
                    "group_size": group_size,
                    "bits": bits,
                    "mode": "affine",
                }
                storage_bits = spec.get("storage_bits")
                if isinstance(storage_bits, int):
                    module_quantization["storage_bits"] = storage_bits
                quantization[module_path] = module_quantization
    model_config_out["quantization"] = quantization
    _write_json(output_dir / "config.json", model_config_out)

    # Write JANG config
    jang_config.setdefault("format", FORMAT_NAME)
    jang_config["format_version"] = FORMAT_VERSION  # 2.0
    _write_json(output_dir / JANG_CONFIG_FILENAME, jang_config)

    # Write tokenizer files. mlxstudio#74: tolerate three content types
    #   - dict        → JSON (parsed earlier, may have eos fixups applied)
    #   - bytes       → binary protobuf (sentencepiece tokenizer.model)
    #   - str         → utf-8 text (merges.txt etc., possibly with
    #                   surrogateescape codepoints from non-utf8 sources)
    # Any of these previously fell through to `write_text(content)` which
    # raises if `content` isn't a string. The new binary path for
    # tokenizer.model would have crashed the WHOLE conversion at the
    # very last step, after quantizing 100%. Now handled explicitly.
    if tokenizer_files:
        for filename, content in tokenizer_files.items():
            dst = output_dir / filename
            if isinstance(content, dict):
                _write_json(dst, content)
            elif isinstance(content, (bytes, bytearray)):
                dst.write_bytes(bytes(content))
            elif isinstance(content, str):
                # Use surrogateescape to round-trip non-utf8 bytes from
                # the source merges.txt (rare but seen on some Asian
                # tokenizer dumps). Default `errors=` would crash here.
                dst.write_text(content, encoding="utf-8", errors="surrogateescape")
            else:
                # Unknown type — best-effort string conversion + warn
                print(f"  WARNING: tokenizer file {filename} has unexpected content type {type(content).__name__}; coercing to str")
                dst.write_text(str(content), encoding="utf-8")

    # Include pre-flushed shards in the weight map (incremental flush for large models)
    weight_map = {}
    total_size = 0
    preflushed_count = 0
    if preflushed_map:
        weight_map.update(preflushed_map)
        preflushed_count = len(set(preflushed_map.values()))

    # Shard remaining tensors into files — standard safetensors naming
    shards = _shard_tensors(tensors, max_shard_bytes)

    for shard_idx, shard_tensor_names in enumerate(shards):
        shard_num = preflushed_count + shard_idx + 1
        # Use placeholder total — will rename after all shards written
        shard_name = f"model-{shard_num:05d}-of-NNNNN.safetensors"
        shard_path = output_dir / shard_name

        shard_data = {}
        for tensor_name in shard_tensor_names:
            arr = tensors[tensor_name]
            shard_data[tensor_name] = arr
            weight_map[tensor_name] = shard_name
            total_size += arr.nbytes

        # CRITICAL: metadata={"format":"mlx"} tells mlx_lm to use the fast
        # loader path. Without this, speed drops 67% (50→15 tok/s).
        save_file(shard_data, str(shard_path), metadata={"format": "mlx"})

    # Rename all shard files with correct total count
    n_total_shards = preflushed_count + len(shards)
    for old_path in sorted(output_dir.glob("model-*-of-NNNNN.safetensors")):
        new_name = old_path.name.replace("NNNNN", f"{n_total_shards:05d}")
        new_path = old_path.parent / new_name
        old_path.rename(new_path)
        # Update weight_map references
        for k, v in weight_map.items():
            if v == old_path.name:
                weight_map[k] = new_name

    # Compute actual indexed shard bytes after final shard renaming. For large
    # conversions, most tensors may have been pre-flushed before this writer
    # sees the final in-memory tail, so summing only `tensors` undercounts the
    # bundle and emits misleading runtime metadata.
    total_size = 0
    for shard_name in set(weight_map.values()):
        shard_path = output_dir / shard_name
        if shard_path.exists():
            total_size += shard_path.stat().st_size

    # Write standard safetensors index (model.safetensors.index.json)
    index = {
        "metadata": {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "total_size": total_size,
        },
        "weight_map": weight_map,
    }
    _write_json(output_dir / JANG_V2_INDEX_FILENAME, index)

    # Write importance matrix if provided
    if importance_data:
        save_file(importance_data, str(output_dir / JANG_IMATRIX_FILENAME))


def write_jang_model(
    output_dir: str | Path,
    quantized_tensors: dict[str, QuantizedTensor],
    model_config: dict,
    jang_config: dict,
    tokenizer_files: dict[str, str | dict] | None = None,
    importance_data: dict[str, np.ndarray] | None = None,
    passthrough_tensors: dict[str, np.ndarray] | None = None,
    max_shard_bytes: int = 5 * 1024 ** 3,  # 5 GB per shard
) -> None:
    """
    Write a JANG v1 model (legacy format with uint8 qweight packing).
    Kept for backward compatibility. New conversions use write_jang_v2_model().
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_json(output_dir / "config.json", model_config)

    jang_config.setdefault("format", FORMAT_NAME)
    jang_config.setdefault("format_version", FORMAT_VERSION_V1)
    _write_json(output_dir / JANG_CONFIG_FILENAME, jang_config)

    if tokenizer_files:
        for filename, content in tokenizer_files.items():
            if isinstance(content, dict):
                _write_json(output_dir / filename, content)
            else:
                (output_dir / filename).write_text(content)

    all_tensors = {}
    for name, qt in quantized_tensors.items():
        all_tensors[f"{name}.qweight"] = qt.qweight
        all_tensors[f"{name}.scales"] = qt.scales
        all_tensors[f"{name}.biases"] = qt.biases
        all_tensors[f"{name}.bits"] = np.array([qt.bits], dtype=np.uint8)
        if qt.shape is not None:
            all_tensors[f"{name}.shape"] = np.array(qt.shape, dtype=np.int64)

    if passthrough_tensors:
        for name, tensor in passthrough_tensors.items():
            all_tensors[name] = tensor

    shards = _shard_tensors(all_tensors, max_shard_bytes)
    weight_map = {}
    total_size = 0

    for shard_idx, shard_tensors in enumerate(shards):
        n_shards = len(shards)
        shard_name = f"model-{shard_idx + 1:05d}-of-{n_shards:05d}.jang.safetensors"
        shard_path = output_dir / shard_name

        shard_data = {}
        for tensor_name in shard_tensors:
            arr = all_tensors[tensor_name]
            shard_data[tensor_name] = arr
            weight_map[tensor_name] = shard_name
            total_size += arr.nbytes

        save_file(shard_data, str(shard_path))

    index = {
        "metadata": {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION_V1,
            "total_size": total_size,
        },
        "weight_map": weight_map,
    }
    _write_json(output_dir / JANG_INDEX_FILENAME, index)

    if importance_data:
        save_file(importance_data, str(output_dir / JANG_IMATRIX_FILENAME))


def _shard_tensors(
    tensors: dict[str, np.ndarray],
    max_bytes: int,
) -> list[list[str]]:
    """Split tensors into shards by size."""
    shards = []
    current_shard = []
    current_size = 0

    # Group companion tensors together (same base name)
    base_names = {}
    for name in tensors:
        base = name.rsplit(".", 1)[0]
        if base not in base_names:
            base_names[base] = []
        base_names[base].append(name)

    for base, group_names in base_names.items():
        group_size = sum(tensors[n].nbytes for n in group_names)

        if current_size + group_size > max_bytes and current_shard:
            shards.append(current_shard)
            current_shard = []
            current_size = 0

        current_shard.extend(group_names)
        current_size += group_size

    if current_shard:
        shards.append(current_shard)

    return shards


def _write_json(path: Path, data: dict) -> None:
    """Write dict as formatted JSON. mlxstudio#74: force utf-8 + fsync
    so the JSON files are durable on external drives where the OS
    write cache may not flush cleanly on process exit. Critical for
    config.json, jang_config.json, and model.safetensors.index.json
    — without these, the loader can't resolve any tensors.
    """
    payload = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
