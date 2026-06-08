"""Static MiMo-V2 JANG bundle audit.

Reads safetensor headers only. Use this to prove payload size, per-layer byte
layout, metadata, and routed bit/group policy without loading the model.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from safetensors import safe_open


DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "U32": 4,
    "I32": 4,
    "I64": 8,
}

LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


def _gb(n: int) -> str:
    return f"{n / 1e9:.3f} GB"


def _role(key: str) -> str:
    if key.startswith("model.mtp."):
        return "mtp"
    if key.startswith("visual."):
        return "visual"
    if key.startswith("audio_encoder.") or key.startswith("speech_embeddings."):
        return "audio"
    if key == "model.embed_tokens.weight":
        return "embed_tokens"
    if key == "lm_head.weight":
        return "lm_head"
    if ".mlp.experts." in key:
        return "routed_experts"
    if key.endswith(".mlp.gate.weight") and ".experts." not in key:
        return "router"
    if key.endswith(".e_score_correction_bias"):
        return "router"
    if ".self_attn.qkv_proj." in key:
        return "attention_qkv"
    if ".self_attn.o_proj." in key:
        return "attention_o_proj"
    if ".mlp." in key and ".experts." not in key:
        return "dense_mlp"
    if key.endswith("attention_sink_bias"):
        return "attention_sink"
    if key.endswith(".norm.weight") or key.endswith("layernorm.weight"):
        return "norms"
    if key.endswith(".bias"):
        return "biases"
    return "other"


def _shape_numel(shape: list[int]) -> int:
    n = 1
    for dim in shape:
        n *= int(dim)
    return n


def _tensor_bytes(dtype: str, shape: list[int]) -> int:
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"unknown safetensors dtype {dtype!r}")
    return DTYPE_BYTES[dtype] * _shape_numel(shape)


def _iter_headers(bundle: Path):
    index = json.loads((bundle / "model.safetensors.index.json").read_text())
    seen: set[Path] = set()
    for shard_name in sorted(set(index["weight_map"].values())):
        shard = bundle / shard_name
        if shard in seen:
            continue
        seen.add(shard)
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                sl = f.get_slice(key)
                yield key, sl.get_dtype(), sl.get_shape(), shard_name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--top-layers", type=int, default=8)
    args = parser.parse_args()

    cfg: dict[str, Any] = json.loads((args.bundle / "config.json").read_text())
    by_role: Counter[str] = Counter()
    by_dtype: Counter[str] = Counter()
    by_layer: dict[int, Counter[str]] = defaultdict(Counter)
    tensor_count = 0

    for key, dtype, shape, _shard_name in _iter_headers(args.bundle):
        nbytes = _tensor_bytes(dtype, shape)
        role = _role(key)
        tensor_count += 1
        by_role[role] += nbytes
        by_dtype[dtype] += nbytes
        m = LAYER_RE.match(key)
        if m:
            by_layer[int(m.group(1))][role] += nbytes

    total = sum(by_role.values())
    qcfg = cfg.get("quantization", {})
    runtime = cfg.get("runtime", {})
    print(f"bundle={args.bundle}")
    print(f"jang_profile={cfg.get('jang_profile')}")
    print(f"mtp_mode={runtime.get('mtp_mode')} bundle_has_mtp={runtime.get('bundle_has_mtp')}")
    print(f"model_type={cfg.get('model_type')} cache_type={runtime.get('cache_type') or cfg.get('capabilities', {}).get('cache_type')}")
    print(f"routed_expert_bits={cfg.get('routed_expert_bits')}")
    print(f"routed_expert_group_size={cfg.get('routed_expert_group_size')}")
    print(f"default_quant_bits={qcfg.get('bits')} default_group_size={qcfg.get('group_size')}")
    print(f"tensor_count={tensor_count}")
    print(f"payload={_gb(total)}")

    print("\nby_role:")
    for role, nbytes in by_role.most_common():
        print(f"  {role:20s} {_gb(nbytes)}")

    print("\nby_dtype:")
    for dtype, nbytes in by_dtype.most_common():
        print(f"  {dtype:5s} {_gb(nbytes)}")

    if by_layer:
        print("\nper_layer:")
        for layer in sorted(by_layer):
            parts = by_layer[layer]
            line = (
                f"  layer {layer:02d} total={_gb(sum(parts.values()))}"
                f" experts={_gb(parts['routed_experts'])}"
                f" qkv={_gb(parts['attention_qkv'])}"
                f" o_proj={_gb(parts['attention_o_proj'])}"
                f" dense_mlp={_gb(parts['dense_mlp'])}"
            )
            extras = []
            for role in ("router", "norms", "attention_sink", "biases", "other"):
                if parts[role]:
                    extras.append(f"{role}={_gb(parts[role])}")
            if extras:
                line += " " + " ".join(extras)
            print(line)

    overrides = {
        k: v for k, v in qcfg.items()
        if isinstance(v, dict) and "bits" in v and "group_size" in v
    }
    if overrides:
        counts = Counter((int(v["bits"]), int(v["group_size"])) for v in overrides.values())
        print("\nquant_override_modules:")
        for (bits, group), count in sorted(counts.items()):
            print(f"  bits={bits} group={group}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
