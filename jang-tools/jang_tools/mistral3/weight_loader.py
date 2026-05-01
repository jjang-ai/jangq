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

    # 2026-04-30 fix: numpy safe_open raises on bf16 shards (Mistral-Medium-3.5
    # source ships bf16). Stage-A: read bf16/fp16/fp32 keys with `mx.load`
    # which understands bf16 natively. Stage-B: fp8 keys still need the
    # numpy view+dequant path (fp8 → uint8 → dequant_fp8_per_tensor),
    # so we open the same shard a second time with framework="numpy"
    # but ONLY for keys whose dtype isn't bf16. This keeps fp8 quant
    # working AND lets bf16 source bundles load.
    for shard, keys in by_shard.items():
        shard_path = str(src / shard)
        try:
            shard_mx = mx.load(shard_path)
        except Exception:
            shard_mx = {}
        for key in keys:
            if key.endswith("_scale") or key.endswith("_scale_inv"):
                continue
            if key in shard_mx and shard_mx[key].dtype != mx.uint8:
                # Pure-tensor case: bf16/fp16/fp32 — copy directly.
                if fmt == "fp8" and key.endswith(".weight") and not is_ignored(key):
                    pass  # Fall through to fp8 dequant below
                else:
                    out[key] = shard_mx[key]
                    continue
            with safe_open(shard_path, framework="numpy") as f:
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
