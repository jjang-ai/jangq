"""Load a JANG-format MiMo-V2.5-Pro bundle (mx.quantize affine).

Bundle layout:
    config.json                 — vanilla MiMoV2 config + quantization sidecar
    model-*.safetensors         — per-shard quantized + scale + bias tensors
    model.safetensors.index.json
    tokenizer.json + tokenizer_config.json

Quant convention (JANG):
    - mx.quantize affine, group_size=64, bits=2|3|4
    - Per-module overrides recorded in config["quantization"]:
          {"layers.X.self_attn.o_proj": {"bits": null}, ...}
      `null` = dense bf16 (matches upstream ignored_layers)
    - Routed experts MUST set per-layer bits >= 2 with the
      gate=4-bit / down=3-bit floor for 2-bit profiles (project_mlp_asymmetry)

Distributed: only loads experts owned by `plan.my_experts`. Other expert
weights are skipped (replaced by None) and never realized.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

try:
    from safetensors import safe_open
except ImportError as e:
    raise SystemExit("pip install safetensors") from e

from .config import MiMoV2Config


def _expert_index_from_key(key: str) -> Optional[int]:
    # e.g. model.layers.5.mlp.experts.123.gate_proj.weight  ->  123
    bits = key.split(".")
    if "experts" in bits:
        i = bits.index("experts")
        if i + 1 < len(bits) and bits[i + 1].isdigit():
            return int(bits[i + 1])
    return None


def load_jang(src: str, cfg: MiMoV2Config, *, plan=None) -> dict:
    """Stream all JANG tensors. If `plan` is given, skip experts not owned
    by `plan.rank`."""
    src = Path(src)
    import json as _json
    idx = _json.loads((src / "model.safetensors.index.json").read_text())
    weight_map = idx["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for k, s in weight_map.items():
        by_shard.setdefault(s, []).append(k)

    weights: dict[str, mx.array] = {}
    my_experts = set(plan.my_experts) if plan is not None else None
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
