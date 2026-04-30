"""Load a JANGTQ (TurboQuant) MiMo-V2.5-Pro bundle.

Bundle layout (matches DSV4/Kimi/MiniMax JANGTQ convention):
    config.json
        ...
        "mxtq_bits": 2,
        "routed_expert_bits": 2,
        "quantization": { per-module overrides },
    model-*.safetensors
        layer.weight_packed   — uint8 packed codes (TQ codebook indices)
        layer.codebook        — fp16/bf16 (n_codewords, group_size)
        layer.bias            — optional
    sidecar.json or sidecar.safetensors
        Hadamard rotations + per-channel scales when present

Loader maps these to MLX TurboQuantLinear (see jang_tools.turboquant.linear).
For distributed: only loads experts in plan.my_experts.

This file is a thin wrapper around jang_tools.turboquant.linear.load_block;
keep this file model-aware (key -> module path) and turboquant.linear
codec-aware.
"""

from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Optional

import mlx.core as mx

try:
    from safetensors import safe_open
except ImportError as e:
    raise SystemExit("pip install safetensors") from e

from .config import MiMoV2Config


def _expert_index_from_key(key: str) -> Optional[int]:
    bits = key.split(".")
    if "experts" in bits:
        i = bits.index("experts")
        if i + 1 < len(bits) and bits[i + 1].isdigit():
            return int(bits[i + 1])
    return None


def load_jangtq(src: str, cfg: MiMoV2Config, *, plan=None) -> dict:
    src = Path(src)
    cfg_json = _json.loads((src / "config.json").read_text())
    bits = cfg_json.get("mxtq_bits") or cfg_json.get("routed_expert_bits")
    if not bits:
        raise ValueError(
            f"{src}/config.json missing mxtq_bits / routed_expert_bits "
            "(2026-04-25 invariant). Patch the bundle before loading."
        )
    print(f"[jangtq] bundle bits={bits}")

    idx = _json.loads((src / "model.safetensors.index.json").read_text())
    weight_map = idx["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for k, s in weight_map.items():
        by_shard.setdefault(s, []).append(k)

    my_experts = set(plan.my_experts) if plan is not None else None
    weights: dict[str, mx.array] = {}
    for shard_name, keys in by_shard.items():
        with safe_open(src / shard_name, framework="numpy") as f:
            for key in keys:
                if my_experts is not None:
                    e = _expert_index_from_key(key)
                    if e is not None and e not in my_experts:
                        continue
                arr = f.get_tensor(key)
                weights[key] = mx.array(arr)
    return weights
