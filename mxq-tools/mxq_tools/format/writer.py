"""
MXQ Format Writer — Write quantized models to .mxq format.
Created by Eric Jang (eric@vmlx.net)
"""

import json
import os
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from .spec import (
    FORMAT_NAME,
    FORMAT_VERSION,
    MXQ_CONFIG_FILENAME,
    MXQ_IMATRIX_FILENAME,
    MXQ_INDEX_FILENAME,
)
from ..quantize import QuantizedTensor


def write_mxq_model(
    output_dir: str | Path,
    quantized_tensors: dict[str, QuantizedTensor],
    model_config: dict,
    mxq_config: dict,
    tokenizer_files: dict[str, str | dict] | None = None,
    importance_data: dict[str, np.ndarray] | None = None,
    passthrough_tensors: dict[str, np.ndarray] | None = None,
    max_shard_bytes: int = 5 * 1024 ** 3,  # 5 GB per shard
) -> None:
    """
    Write a complete MXQ model directory.

    Args:
        output_dir: output directory path
        quantized_tensors: dict of tensor_name -> QuantizedTensor
        model_config: HuggingFace model config dict
        mxq_config: MXQ quantization config dict
        tokenizer_files: dict of filename → content (str or dict for JSON)
        importance_data: dict of tensor_name → importance scores array
        max_shard_bytes: maximum bytes per shard file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write model config
    _write_json(output_dir / "config.json", model_config)

    # Write MXQ config
    mxq_config.setdefault("format", FORMAT_NAME)
    mxq_config.setdefault("format_version", FORMAT_VERSION)
    _write_json(output_dir / MXQ_CONFIG_FILENAME, mxq_config)

    # Write tokenizer files
    if tokenizer_files:
        for filename, content in tokenizer_files.items():
            if isinstance(content, dict):
                _write_json(output_dir / filename, content)
            else:
                (output_dir / filename).write_text(content)

    # Build flat tensor dict for safetensors
    all_tensors = {}
    for name, qt in quantized_tensors.items():
        all_tensors[f"{name}.qweight"] = qt.qweight
        all_tensors[f"{name}.scales"] = qt.scales
        all_tensors[f"{name}.zeros"] = qt.zeros
        all_tensors[f"{name}.bit_map"] = qt.bit_map
        all_tensors[f"{name}.block_offsets"] = qt.block_offsets

    # Add non-quantized tensors (norms, biases, etc.)
    if passthrough_tensors:
        for name, tensor in passthrough_tensors.items():
            all_tensors[name] = tensor

    # Shard tensors into files
    shards = _shard_tensors(all_tensors, max_shard_bytes)
    weight_map = {}
    total_size = 0

    for shard_idx, shard_tensors in enumerate(shards):
        n_shards = len(shards)
        shard_name = f"model-{shard_idx + 1:05d}-of-{n_shards:05d}.mxq.safetensors"
        shard_path = output_dir / shard_name

        # Build shard data
        shard_data = {}
        for tensor_name in shard_tensors:
            arr = all_tensors[tensor_name]
            shard_data[tensor_name] = arr
            weight_map[tensor_name] = shard_name
            total_size += arr.nbytes

        save_file(shard_data, str(shard_path))

    # Write shard index
    index = {
        "metadata": {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "total_size": total_size,
        },
        "weight_map": weight_map,
    }
    _write_json(output_dir / MXQ_INDEX_FILENAME, index)

    # Write importance matrix if provided
    if importance_data:
        save_file(importance_data, str(output_dir / MXQ_IMATRIX_FILENAME))


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
        # Strip suffix to get base: "layers.0.q_proj.qweight" → "layers.0.q_proj"
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
    """Write dict as formatted JSON."""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
