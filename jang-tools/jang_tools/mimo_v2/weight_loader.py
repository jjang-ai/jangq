"""Streaming weight loader for MiMo-V2 source checkpoints.

Hides shard boundaries and transparently dequantizes FP8 e4m3fn weights
(with their fp32 ``*_weight_scale_inv`` companions) into fp32. Plain bf16
weights pass through unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from .fp8_block_codec import dequant_fp8_e4m3_scale_inv


class MiMoShardIndex:
    """Lookup table from tensor name → source shard, with cached safe_open handles."""

    def __init__(self, src: str | Path):
        self.src = Path(src).expanduser()
        idx = json.loads((self.src / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = idx["weight_map"]
        # All tensor names visible in the source (incl. *_weight_scale_inv).
        self.keys: list[str] = sorted(self.weight_map.keys())
        # Weight names that have a companion FP8 scale.
        self.fp8_weight_names: set[str] = {
            k[: -len(".weight_scale_inv")] + ".weight"
            for k in self.keys
            if k.endswith(".weight_scale_inv")
        }
        # Names yielded to callers — drop bare scale tensors, they are read
        # internally when their companion weight is requested.
        self.weight_keys: list[str] = [k for k in self.keys if not k.endswith(".weight_scale_inv")]

    # ------------------------------------------------------------------
    # Tensor reads
    # ------------------------------------------------------------------

    def is_fp8_weight(self, name: str) -> bool:
        return name in self.fp8_weight_names

    def read_passthrough(self, name: str, *, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        """Read a tensor as-is from its shard. No dequantization."""
        with safe_open(str(self.src / self.weight_map[name]), framework="pt", device="cpu") as f:
            t = f.get_tensor(name)
        if out_dtype is not None and t.dtype != out_dtype:
            t = t.to(out_dtype)
        return t

    def read_tensor(self, name: str, *, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Read `name` and dequantize from FP8 + block scale_inv if applicable.

        Plain bf16/fp32 tensors are cast to ``out_dtype`` if needed.
        """
        if name not in self.fp8_weight_names:
            return self.read_passthrough(name, out_dtype=out_dtype)

        scale_name = name[: -len(".weight")] + ".weight_scale_inv"
        weight_shard = self.weight_map[name]
        scale_shard = self.weight_map[scale_name]
        with safe_open(str(self.src / weight_shard), framework="pt", device="cpu") as f:
            w = f.get_tensor(name)
        if scale_shard == weight_shard:
            with safe_open(str(self.src / weight_shard), framework="pt", device="cpu") as f:
                s = f.get_tensor(scale_name)
        else:
            with safe_open(str(self.src / scale_shard), framework="pt", device="cpu") as f:
                s = f.get_tensor(scale_name)
        return dequant_fp8_e4m3_scale_inv(w, s, out_dtype=out_dtype)
