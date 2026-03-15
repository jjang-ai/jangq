"""
MLXQ Format Reader — Load quantized models from .mxq format.
Created by Eric Jang (eric@vmlx.net)
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file

from .spec import (
    FORMAT_NAME,
    MXQ_CONFIG_FILENAME,
    MXQ_IMATRIX_FILENAME,
    MXQ_INDEX_FILENAME,
    MXQ_SUFFIXES,
    DEFAULT_BLOCK_SIZE,
)
from ..quantize import QuantizedTensor


class MXQModel:
    """Loaded MXQ model — provides access to quantized tensors and metadata."""

    def __init__(
        self,
        path: Path,
        mxq_config: dict,
        model_config: dict,
        tensors: dict[str, np.ndarray],
    ):
        self.path = path
        self.mxq_config = mxq_config
        self.model_config = model_config
        self._tensors = tensors
        self._quantized_cache: dict[str, QuantizedTensor] = {}

    @property
    def target_bits(self) -> float:
        return self.mxq_config["quantization"]["target_bits"]

    @property
    def actual_bits(self) -> float:
        return self.mxq_config["quantization"].get("actual_bits", self.target_bits)

    @property
    def block_size(self) -> int:
        return self.mxq_config["quantization"].get("block_size", DEFAULT_BLOCK_SIZE)

    @property
    def source_model(self) -> str:
        return self.mxq_config.get("source_model", {}).get("name", "unknown")

    @property
    def weight_names(self) -> list[str]:
        """Get base weight names (without .qweight/.scales etc suffixes)."""
        names = set()
        for key in self._tensors:
            for suffix in MXQ_SUFFIXES:
                if key.endswith(suffix):
                    names.add(key[: -len(suffix)])
                    break
        return sorted(names)

    def get_quantized_tensor(self, name: str) -> QuantizedTensor:
        """Get a QuantizedTensor by base name (e.g., 'layers.0.self_attn.q_proj')."""
        if name in self._quantized_cache:
            return self._quantized_cache[name]

        qt = QuantizedTensor(
            qweight=self._tensors[f"{name}.qweight"],
            scales=self._tensors[f"{name}.scales"],
            zeros=self._tensors[f"{name}.zeros"],
            bit_map=self._tensors[f"{name}.bit_map"],
            block_offsets=self._tensors[f"{name}.block_offsets"],
            shape=None,  # shape not stored in format yet — infer from config
        )
        self._quantized_cache[name] = qt
        return qt

    def get_raw_tensor(self, name: str) -> Optional[np.ndarray]:
        """Get a non-quantized tensor by exact name."""
        return self._tensors.get(name)

    def summary(self) -> dict:
        """Generate a model summary."""
        bit_maps = []
        for name in self.weight_names:
            qt = self.get_quantized_tensor(name)
            bit_maps.append(qt.bit_map)

        all_bits = np.concatenate(bit_maps) if bit_maps else np.array([])
        total_blocks = len(all_bits)

        histogram = {}
        for bw in sorted(set(int(b) for b in all_bits)):
            count = int(np.sum(all_bits == bw))
            histogram[f"{bw}-bit"] = {
                "count": count,
                "percent": round(100 * count / total_blocks, 1),
            }

        total_qweight_bytes = sum(
            self.get_quantized_tensor(n).qweight.nbytes for n in self.weight_names
        )

        return {
            "source_model": self.source_model,
            "target_bits": self.target_bits,
            "actual_bits": round(float(np.mean(all_bits)), 2) if len(all_bits) > 0 else 0,
            "block_size": self.block_size,
            "total_blocks": total_blocks,
            "total_weight_names": len(self.weight_names),
            "total_qweight_bytes": total_qweight_bytes,
            "total_qweight_gb": round(total_qweight_bytes / (1024 ** 3), 2),
            "histogram": histogram,
        }


def load_mxq_model(path: str | Path) -> MXQModel:
    """
    Load an MXQ model from a directory.

    Args:
        path: path to MXQ model directory

    Returns:
        MXQModel instance with all tensors loaded
    """
    path = Path(path)

    if not path.is_dir():
        raise FileNotFoundError(f"Not a directory: {path}")

    config_path = path / MXQ_CONFIG_FILENAME
    if not config_path.exists():
        raise FileNotFoundError(
            f"Not an MXQ model directory: {path} (missing {MXQ_CONFIG_FILENAME})"
        )

    # Load MXQ config
    mxq_config = json.loads(config_path.read_text())
    if mxq_config.get("format") != FORMAT_NAME:
        raise ValueError(f"Invalid format: expected '{FORMAT_NAME}', got '{mxq_config.get('format')}'")

    # Load model config
    model_config_path = path / "config.json"
    model_config = {}
    if model_config_path.exists():
        model_config = json.loads(model_config_path.read_text())

    # Load tensors from shard files
    index_path = path / MXQ_INDEX_FILENAME
    tensors = {}

    if index_path.exists():
        index = json.loads(index_path.read_text())
        shard_files = set(index["weight_map"].values())

        for shard_file in sorted(shard_files):
            shard_path = path / shard_file
            shard_tensors = load_file(str(shard_path))
            tensors.update(shard_tensors)
    else:
        # Try loading any .mxq.safetensors files
        for sf in sorted(path.glob("*.mxq.safetensors")):
            shard_tensors = load_file(str(sf))
            tensors.update(shard_tensors)

    return MXQModel(
        path=path,
        mxq_config=mxq_config,
        model_config=model_config,
        tensors=tensors,
    )


def is_mxq_model(path: str | Path) -> bool:
    """Check if a directory contains an MXQ model."""
    path = Path(path)
    return path.is_dir() and (path / MXQ_CONFIG_FILENAME).exists()


def load_importance_matrix(path: str | Path) -> dict[str, np.ndarray]:
    """Load an importance matrix from safetensors file."""
    path = Path(path)
    if path.is_dir():
        path = path / MXQ_IMATRIX_FILENAME
    return load_file(str(path))
