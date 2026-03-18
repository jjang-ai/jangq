"""
JANG Format Reader — Load quantized models from .jang format.
Created by Jinho Jang (eric@jangq.ai)
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file

from .spec import (
    FORMAT_NAME,
    JANG_CONFIG_FILENAME,
    JANG_IMATRIX_FILENAME,
    JANG_INDEX_FILENAME,
    JANG_SUFFIXES,
    DEFAULT_BLOCK_SIZE,
)
from ..quantize import QuantizedTensor


class JANGModel:
    """Loaded JANG model — provides access to quantized tensors and metadata."""

    def __init__(
        self,
        path: Path,
        jang_config: dict,
        model_config: dict,
        tensors: dict[str, np.ndarray],
    ):
        self.path = path
        self.jang_config = jang_config
        self.model_config = model_config
        self._tensors = tensors
        self._quantized_cache: dict[str, QuantizedTensor] = {}

    @property
    def target_bits(self) -> float:
        return self.jang_config["quantization"]["target_bits"]

    @property
    def actual_bits(self) -> float:
        return self.jang_config["quantization"].get("actual_bits", self.target_bits)

    @property
    def block_size(self) -> int:
        return self.jang_config["quantization"].get("block_size", DEFAULT_BLOCK_SIZE)

    @property
    def source_model(self) -> str:
        return self.jang_config.get("source_model", {}).get("name", "unknown")

    @property
    def weight_names(self) -> list[str]:
        """Get base weight names (without .qweight/.scales etc suffixes)."""
        names = set()
        for key in self._tensors:
            for suffix in JANG_SUFFIXES:
                if key.endswith(suffix):
                    names.add(key[: -len(suffix)])
                    break
        return sorted(names)

    def get_quantized_tensor(self, name: str) -> QuantizedTensor:
        """Get a QuantizedTensor by base name (e.g., 'layers.0.self_attn.q_proj')."""
        if name in self._quantized_cache:
            return self._quantized_cache[name]

        shape_key = f"{name}.shape"
        stored_shape = tuple(self._tensors[shape_key].tolist()) if shape_key in self._tensors else None

        # v1.1 format: single .bits per tensor (no bit_map/block_offsets)
        bits_key = f"{name}.bits"
        if bits_key in self._tensors:
            bits = int(self._tensors[bits_key][0])
        else:
            # v1.0 fallback: read from bit_map (all values should be same)
            bit_map = self._tensors[f"{name}.bit_map"]
            bits = int(bit_map[0])

        # v1.2+: biases stored directly; v1.0-1.1: zeros stored
        scales = self._tensors[f"{name}.scales"]
        biases_key = f"{name}.biases"
        zeros_key = f"{name}.zeros"
        if biases_key in self._tensors:
            biases = self._tensors[biases_key]
        elif zeros_key in self._tensors:
            zeros = self._tensors[zeros_key]
            biases = -(scales.astype(np.float32) * zeros.astype(np.float32)).astype(np.float16)
        else:
            biases = np.zeros_like(scales)

        qt = QuantizedTensor(
            qweight=self._tensors[f"{name}.qweight"],
            scales=scales,
            biases=biases,
            bits=bits,
            shape=stored_shape,
        )
        self._quantized_cache[name] = qt
        return qt

    def get_raw_tensor(self, name: str) -> Optional[np.ndarray]:
        """Get a non-quantized tensor by exact name."""
        return self._tensors.get(name)

    def summary(self) -> dict:
        """Generate a model summary."""
        total_blocks = 0
        bit_block_counts = {}
        for name in self.weight_names:
            qt = self.get_quantized_tensor(name)
            n_blocks = len(qt.scales)
            total_blocks += n_blocks
            bit_block_counts[qt.bits] = bit_block_counts.get(qt.bits, 0) + n_blocks

        histogram = {}
        for bw in sorted(bit_block_counts.keys()):
            count = bit_block_counts[bw]
            histogram[f"{bw}-bit"] = {
                "count": count,
                "percent": round(100 * count / total_blocks, 1) if total_blocks else 0,
            }

        total_qweight_bytes = sum(
            self.get_quantized_tensor(n).qweight.nbytes for n in self.weight_names
        )

        return {
            "source_model": self.source_model,
            "target_bits": self.target_bits,
            "actual_bits": round(sum(bw * bit_block_counts[bw] for bw in bit_block_counts) / total_blocks, 2) if total_blocks > 0 else 0,
            "block_size": self.block_size,
            "total_blocks": total_blocks,
            "total_weight_names": len(self.weight_names),
            "total_qweight_bytes": total_qweight_bytes,
            "total_qweight_gb": round(total_qweight_bytes / (1024 ** 3), 2),
            "histogram": histogram,
        }


def load_jang_model(path: str | Path) -> JANGModel:
    """
    Load a JANG model from a directory.

    Args:
        path: path to JANG model directory

    Returns:
        JANGModel instance with all tensors loaded
    """
    path = Path(path)

    if not path.is_dir():
        raise FileNotFoundError(f"Not a directory: {path}")

    config_path = path / JANG_CONFIG_FILENAME
    if not config_path.exists():
        raise FileNotFoundError(
            f"Not a JANG model directory: {path} (missing {JANG_CONFIG_FILENAME})"
        )

    # Load JANG config
    jang_config = json.loads(config_path.read_text())
    if jang_config.get("format") != FORMAT_NAME:
        raise ValueError(f"Invalid format: expected '{FORMAT_NAME}', got '{jang_config.get('format')}'")

    # Load model config
    model_config_path = path / "config.json"
    model_config = {}
    if model_config_path.exists():
        model_config = json.loads(model_config_path.read_text())

    # Load tensors from shard files
    index_path = path / JANG_INDEX_FILENAME
    tensors = {}

    if index_path.exists():
        index = json.loads(index_path.read_text())
        shard_files = set(index["weight_map"].values())

        for shard_file in sorted(shard_files):
            # Sanitize: reject paths with directory traversal
            if ".." in shard_file or shard_file.startswith("/"):
                raise ValueError(f"Invalid shard path in index: {shard_file}")
            shard_path = path / shard_file
            shard_tensors = load_file(str(shard_path))
            tensors.update(shard_tensors)
    else:
        # Try loading any .jang.safetensors files
        for sf in sorted(path.glob("*.jang.safetensors")):
            shard_tensors = load_file(str(sf))
            tensors.update(shard_tensors)

    return JANGModel(
        path=path,
        jang_config=jang_config,
        model_config=model_config,
        tensors=tensors,
    )


def is_jang_model(path: str | Path) -> bool:
    """Check if a directory contains a JANG model."""
    path = Path(path)
    return path.is_dir() and (path / JANG_CONFIG_FILENAME).exists()


def load_importance_matrix(path: str | Path) -> dict[str, np.ndarray]:
    """Load an importance matrix from safetensors file."""
    path = Path(path)
    if path.is_dir():
        path = path / JANG_IMATRIX_FILENAME
    return load_file(str(path))
