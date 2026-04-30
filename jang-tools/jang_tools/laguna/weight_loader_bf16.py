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
        with safe_open(str(src / shard), framework="numpy") as f:
            for k in keys:
                out[k] = f.get_tensor(k)
    return out


def load_bf16(src: str, cfg: LagunaConfig) -> dict:
    raw = _read_all(Path(src))
    return {k: mx.array(v) for k, v in raw.items()}


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
