"""Mistral 3.5 weight loader. Handles bf16 / fp8 / jangtq / mxfp4 sources."""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open

from .config import Mistral3Config
from .fp8_per_tensor_codec import dequant_fp8_per_tensor


def load_weights(src: str, cfg: Mistral3Config, fmt: str) -> dict:
    src = Path(src)
    idx = json.loads((src / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    by_shard: dict = {}
    for k, sh in wm.items():
        by_shard.setdefault(sh, []).append(k)

    out: dict = {}
    ignored = set(cfg.fp8_ignored_modules)

    def is_ignored(key: str) -> bool:
        base = key.rsplit(".weight", 1)[0]
        return any(base == ig or base.startswith(ig + ".") for ig in ignored)

    for shard, keys in by_shard.items():
        with safe_open(str(src / shard), framework="numpy") as f:
            for key in keys:
                if key.endswith("_scale") or key.endswith("_scale_inv"):
                    continue
                arr = f.get_tensor(key)
                if fmt == "fp8" and key.endswith(".weight") and not is_ignored(key):
                    scale_key = key.replace(".weight", ".weight_scale")
                    if scale_key in wm:
                        with safe_open(str(src / wm[scale_key]), framework="numpy") as g:
                            scale = g.get_tensor(scale_key)
                        if arr.dtype != np.uint8:
                            arr = arr.view(np.uint8)
                        out[key] = dequant_fp8_per_tensor(arr, scale)
                        continue
                out[key] = mx.array(arr)
    return out
