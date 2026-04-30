"""Load MiMo-V2.5-Pro FP8 source weights into MLX bf16.

Maps HF safetensors keys → MLX module attribute paths and dequantizes any
FP8 tensor to bf16 via `fp8_codec.dequant_fp8_block`. Tensors in
`quantization_config.ignored_layers` are loaded as-is (bf16).

Streaming: opens shards lazily and frees per-tensor numpy buffers after each
mx.array conversion. Total RAM during load ≈ size of one shard (~31 GB) plus
the running MLX state. Use this on the Mac Studio for the JANG_2L oracle path.

For distributed sharded load, see `jang_tools.distributed.mimo_v2_dist`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

import mlx.core as mx
import numpy as np

try:
    from safetensors import safe_open
except ImportError as e:  # pragma: no cover
    raise SystemExit("pip install safetensors") from e

from .config import MiMoV2Config
from .fp8_codec import dequant_fp8_block, fp8_e4m3_to_fp32


def _iter_keys(index_path: Path) -> Iterator[tuple[str, str]]:
    with open(index_path) as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    for key, shard in weight_map.items():
        yield key, shard


def _hf_to_mlx_key(key: str) -> str:
    """Strip the leading `model.` prefix where appropriate; lm_head stays.

    HF stores `model.layers.X.self_attn.qkv_proj.weight`, MLX wants
    `model.layers.X.self_attn.qkv_proj.weight` too — same path, OK as-is.
    `model.embed_tokens.weight` stays. `lm_head.weight` stays.
    `model.mtp.*` is loaded into a separate MTP head, skipped here.
    """
    return key


def load_fp8_to_bf16(
    src_dir: str,
    config: MiMoV2Config,
    *,
    skip_mtp: bool = True,
    progress: bool = True,
) -> dict[str, mx.array]:
    """Stream all weights from `src_dir/model-*.safetensors` and return a
    flat {key: mx.array} dict ready for `mx.tree_unflatten`.
    """
    src = Path(src_dir)
    index = src / "model.safetensors.index.json"
    weights: dict[str, mx.array] = {}
    ignored = set(config.fp8_ignored_layers)
    block = config.fp8_weight_block_size

    seen_shards: dict[str, "safe_open"] = {}

    def _open(shard_name: str):
        if shard_name not in seen_shards:
            seen_shards[shard_name] = safe_open(src / shard_name, framework="numpy")
        return seen_shards[shard_name]

    by_shard: dict[str, list[str]] = {}
    for k, s in _iter_keys(index):
        by_shard.setdefault(s, []).append(k)

    n_total = sum(len(v) for v in by_shard.values())
    n_done = 0
    for shard_name, keys in by_shard.items():
        f = _open(shard_name)
        for key in keys:
            if skip_mtp and key.startswith("model.mtp."):
                n_done += 1
                continue
            mlx_key = _hf_to_mlx_key(key)
            base = key[: -len(".weight")] if key.endswith(".weight") else key
            base = base[: -len(".bias")] if key.endswith(".bias") else base
            scale_key = key.replace(".weight", ".weight_scale_inv")
            if scale_key in f.keys() and base not in ignored:
                w = f.get_tensor(key)        # uint8 view of fp8 (or fp8 native)
                s = f.get_tensor(scale_key)  # float32
                if w.dtype != np.uint8:
                    w = w.view(np.uint8)
                arr = dequant_fp8_block(w, s, block=block)
            else:
                w = f.get_tensor(key)
                if str(w.dtype).startswith("float8"):
                    w = fp8_e4m3_to_fp32(w.view(np.uint8)).astype(np.float32)
                arr = mx.array(w).astype(mx.bfloat16)
            weights[mlx_key] = arr
            n_done += 1
            if progress and n_done % 50 == 0:
                print(f"  loaded {n_done}/{n_total} tensors", flush=True)
    return weights
