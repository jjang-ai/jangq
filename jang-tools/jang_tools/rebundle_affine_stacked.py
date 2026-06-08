"""Rebundle split affine DSV4 experts into pre-stacked switch_mlp tensors.

This is the affine counterpart to ``rebundle_jangtq_stacked``. It does not
requantize tensors. It only changes the main-model routed expert layout from

  layers.N.ffn.experts.E.{w1,w2,w3}.{weight,scales,biases}

to

  layers.N.mlp.switch_mlp.{gate_proj,down_proj,up_proj}.{weight,scales,biases}

stacked along axis 0. ``mtp.*`` tensors are copied through unchanged so future
DSV4 MTP work can load them separately.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


EXPERT_RE = re.compile(
    r"^(layers\.(\d+)\.ffn\.)experts\.(\d+)\.(w[123])\."
    r"(weight|scales|biases)$"
)
PROJ_MAP = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}


def _stack_base(layer_prefix: str, proj: str) -> str:
    return f"{layer_prefix.replace('.ffn.', '.mlp.')}switch_mlp.{PROJ_MAP[proj]}"


def patch_prestacked_affine_config(bundle: str | Path) -> None:
    """Patch config metadata so MLX quantizes pre-stacked switch modules.

    The split-expert converter records source-path overrides such as
    ``layers.0.ffn.experts.7.w1``. Once experts are pre-stacked, MLX needs the
    module-path override ``model.layers.0.mlp.switch_mlp.gate_proj`` instead.
    """
    bundle = Path(bundle)
    cfg_path = bundle / "config.json"
    jang_path = bundle / "jang_config.json"
    if not cfg_path.exists():
        return
    cfg = json.loads(cfg_path.read_text())
    jang = json.loads(jang_path.read_text()) if jang_path.exists() else {}

    quant = cfg.setdefault("quantization", {})
    num_layers = int(cfg.get("num_hidden_layers") or cfg.get("n_layers") or 0)
    profile_bits = int(
        quant.get("routed_expert_bits")
        or cfg.get("routed_expert_bits")
        or ((jang.get("quantization") or {}).get("routed_experts") or {}).get("bits")
        or 2
    )
    routed_group_size = int(
        quant.get("routed_expert_group_size")
        or cfg.get("routed_expert_group_size")
        or ((jang.get("quantization") or {}).get("routed_experts") or {}).get("group_size")
        or 64
    )
    bookend_bits = int(quant.get("bookend_bits") or quant.get("bits") or 8)
    bookend_group_size = int(quant.get("bookend_group_size") or quant.get("group_size") or 64)

    bit_plan = quant.get("routed_expert_bit_plan") or cfg.get("routed_expert_bit_plan") or {}
    projection_bits = {
        str(k): int(v)
        for k, v in (bit_plan.get("routed_projection_bits") or {}).items()
    }
    projection_layer_bits = {
        str(proj): {int(layer): int(bits) for layer, bits in layer_bits.items()}
        for proj, layer_bits in (
            bit_plan.get("routed_projection_layer_bits") or {}
        ).items()
        if isinstance(layer_bits, dict)
    }
    projection_group_sizes = {
        str(k): int(v)
        for k, v in (bit_plan.get("routed_projection_group_sizes") or {}).items()
    }

    # Default quantization should match non-routed/bookend modules. Switch MLP
    # modules get explicit overrides below.
    quant["bits"] = bookend_bits
    quant["group_size"] = bookend_group_size
    quant["mode"] = "affine"
    quant["routed_expert_bits"] = profile_bits
    quant["routed_expert_group_size"] = routed_group_size
    quant["bookend_bits"] = bookend_bits
    quant["bookend_group_size"] = bookend_group_size

    for layer in range(num_layers):
        for source_proj, module_proj in PROJ_MAP.items():
            module = f"model.layers.{layer}.mlp.switch_mlp.{module_proj}"
            bits = int(projection_bits.get(source_proj, profile_bits))
            bits = int(projection_layer_bits.get(source_proj, {}).get(layer, bits))
            quant[module] = {
                "bits": bits,
                "group_size": int(projection_group_sizes.get(source_proj, routed_group_size)),
                "mode": "affine",
            }

    cfg["routed_expert_layout"] = "prestacked_affine"
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    if jang_path.exists():
        jang["routed_expert_layout"] = "prestacked_affine"
        jang_path.write_text(json.dumps(jang, indent=2, ensure_ascii=False))


def rebundle(src: str | Path, dst: str | Path, *, shard_bytes: int = 1_000_000_000) -> None:
    src = Path(src).expanduser().resolve()
    dst = Path(dst).expanduser().resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"source bundle not found: {src}")
    if src == dst:
        raise ValueError("dst must differ from src")
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    src_shards = sorted(src.glob("model-*.safetensors"))
    if not src_shards:
        raise FileNotFoundError(f"no model shards found in {src}")

    experts: dict[tuple[str, int], dict[str, tuple[Path, str]]] = defaultdict(dict)
    passthrough: list[tuple[str, Path]] = []
    for shard in src_shards:
        with safe_open(str(shard), framework="numpy") as handle:
            for key in handle.keys():
                match = EXPERT_RE.match(key)
                if not match:
                    passthrough.append((key, shard))
                    continue
                layer_prefix, _layer, expert_id, proj, kind = match.groups()
                base = _stack_base(layer_prefix, proj)
                experts[(base, int(expert_id))][kind] = (shard, key)

    if not experts:
        raise RuntimeError("found 0 split affine DSV4 experts to stack")

    by_base: dict[str, dict[int, dict[str, tuple[Path, str]]]] = defaultdict(dict)
    for (base, expert_id), parts in experts.items():
        by_base[base][expert_id] = parts

    print("=" * 70)
    print("  affine rebundle (split experts -> pre-stacked switch_mlp)")
    print(f"  src: {src}")
    print(f"  dst: {dst}")
    print(f"  output groups: {len(by_base)}")
    print(f"  passthrough keys: {len(passthrough)}")
    print("=" * 70, flush=True)

    shard_idx = 0
    shard_buf: dict[str, np.ndarray] = {}
    current_bytes = 0
    weight_map: dict[str, str] = {}

    def flush_shard() -> None:
        nonlocal shard_idx, shard_buf, current_bytes
        if not shard_buf:
            return
        shard_idx += 1
        name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        save_file(shard_buf, str(dst / name))
        for key in shard_buf:
            weight_map[key] = name
        print(f"    shard {shard_idx}: {len(shard_buf)} tensors, {current_bytes / 1e9:.2f} GB", flush=True)
        shard_buf = {}
        current_bytes = 0

    def emit(key: str, arr: np.ndarray) -> None:
        nonlocal current_bytes
        shard_buf[key] = arr
        current_bytes += arr.nbytes
        if current_bytes >= shard_bytes:
            flush_shard()

    for base in sorted(by_base):
        group = by_base[base]
        n_experts = max(group) + 1
        missing = [idx for idx in range(n_experts) if idx not in group]
        if missing:
            raise RuntimeError(f"non-contiguous experts for {base}: missing {missing[:8]}")
        for kind in ("weight", "scales", "biases"):
            arrays = []
            for expert_id in range(n_experts):
                parts = group[expert_id]
                if kind not in parts:
                    raise RuntimeError(f"missing {kind} for {base} expert {expert_id}")
                shard, key = parts[kind]
                with safe_open(str(shard), framework="numpy") as handle:
                    arrays.append(handle.get_tensor(key))
            stacked = np.stack(arrays, axis=0)
            emit(f"{base}.{kind}", stacked)
            del arrays, stacked
            gc.collect()

    seen: set[str] = set()
    for key, shard in passthrough:
        if key in seen:
            continue
        seen.add(key)
        with safe_open(str(shard), framework="numpy") as handle:
            arr = handle.get_tensor(key)
        emit(key, arr)
        del arr
    flush_shard()

    for idx in range(1, shard_idx + 1):
        old = dst / f"model-{idx:05d}-of-XXXXX.safetensors"
        new = dst / f"model-{idx:05d}-of-{shard_idx:05d}.safetensors"
        if old.exists():
            old.rename(new)
    weight_map = {k: v.replace("XXXXX", f"{shard_idx:05d}") for k, v in weight_map.items()}
    total_size = sum((dst / name).stat().st_size for name in set(weight_map.values()))
    (dst / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {
            "format": "affine",
            "total_size": total_size,
            "rebundled_from": str(src),
            "rebundled_layout": "prestacked-switch_mlp-affine",
        },
        "weight_map": weight_map,
    }, indent=2))

    for item in src.iterdir():
        if item.name == "model.safetensors.index.json":
            continue
        if item.is_file() and item.name.startswith("model-") and item.suffix == ".safetensors":
            continue
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)

    try:
        patch_prestacked_affine_config(dst)
    except Exception as exc:
        print(f"warn: failed to patch prestacked affine config: {exc}", flush=True)

    print(f"  output shards: {shard_idx}")
    print(f"  output size: {total_size / 1e9:.3f} GB")
    print(f"  output: {dst}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--shard-bytes", type=int, default=1_000_000_000)
    args = parser.parse_args()
    rebundle(args.src, args.dst, shard_bytes=args.shard_bytes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
