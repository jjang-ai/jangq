"""Bf16 / affine / JANGTQ weight loaders for Laguna.

bf16: vanilla source from poolside/Laguna-XS.2 — straight read.
affine: mx.quantize bf16 + scales + biases (JANG_2L or MXFP4 bundles).
jangtq: TurboQuant-packed routed experts + affine non-experts.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open

from .config import LagunaConfig


def _read_all(src: Path) -> dict:
    idx = json.loads((src / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    by_shard: dict = {}
    for k, sh in wm.items():
        by_shard.setdefault(sh, []).append(k)
    out: dict = {}
    for shard, keys in by_shard.items():
        # 2026-04-30 fix: safe_open(framework="numpy") raises
        # `TypeError: data type 'bfloat16' not understood` on safetensors
        # shards stored in bf16 (Laguna-XS.2 source ships bf16). Use
        # `mx.load` which understands bf16 natively. Slight memory
        # increase since the whole shard goes through MLX in one pass,
        # but the alternative of pulling torch in for `safe_open(framework="pt")`
        # is heavier and we already require torch elsewhere only for the
        # Stage-1 Omni bridge — not for Laguna.
        path = str(src / shard)
        shard_data = mx.load(path)
        for k in keys:
            if k in shard_data:
                out[k] = shard_data[k]
    return out


def load_bf16(src: str, cfg: LagunaConfig) -> dict:
    raw = _read_all(Path(src))
    # mx.load returns mx.array directly — no need to wrap.
    return raw


def load_affine(src: str, cfg: LagunaConfig) -> dict:
    """JANG affine and MXFP4 share key layout: .weight / .scales / .biases."""
    raw = _read_all(Path(src))
    return {k: mx.array(v) for k, v in raw.items()}


def load_jangtq(src: str, cfg: LagunaConfig) -> dict:
    """JANGTQ: routed experts have .tq_packed/.tq_norms/.tq_bits;
    other modules have .weight/.scales/.biases (affine 8-bit).

    For now we just return all tensors raw; the runtime LagunaForCausalLM
    will be wrapped by jang_tools.jangrt.linear.JANGTQLinear / JANGLinear
    when the matching layers are constructed. End-to-end wiring still
    pending — this loader produces the right tensor names so the next
    iteration can finish that.
    """
    raw = _read_all(Path(src))
    return {k: mx.array(v) for k, v in raw.items()}
