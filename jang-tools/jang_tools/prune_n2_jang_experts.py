"""Prune Nex/N2 JANG/JANGTQ routed experts from an already-converted bundle.

This is a streaming artifact pruner for the qwen3_5_moe/Nex layout produced by
the classic affine converter and Qwen3.5 JANGTQ converter. It slices routed
expert axis 0 and matching router gate rows, preserving all other tensors and
sidecars.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


SWITCH_RE = re.compile(
    r"^language_model\.model\.layers\.(\d+)\.mlp\.switch_mlp\."
    r"(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$"
)
TQ_SWITCH_RE = re.compile(
    r"^language_model\.model\.layers\.(\d+)\.mlp\.switch_mlp\."
    r"(gate_proj|up_proj|down_proj)\.(tq_packed|tq_norms|tq_bits)$"
)
ROUTER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.gate\.weight$")
SANITIZED_ROUTER_RE = re.compile(r"^language_model\.model\.layers\.(\d+)\.mlp\.gate\.weight$")


SIDECARS = (
    "config.json",
    "jang_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "generation_config.json",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _copy_sidecars(
    src: Path,
    dst: Path,
    *,
    keep_experts: int,
    original_experts: int,
    method: str,
) -> None:
    for name in SIDECARS:
        source = src / name
        if source.exists():
            shutil.copy2(source, dst / name)

    config_path = dst / "config.json"
    config = _load_json(config_path)
    text_config = config.setdefault("text_config", {})
    text_config["num_experts"] = keep_experts
    if "n_routed_experts" in text_config:
        text_config["n_routed_experts"] = keep_experts
    config["jang_expert_pruning"] = {
        "source_num_experts": original_experts,
        "num_experts": keep_experts,
        "dropped_experts_per_layer": original_experts - keep_experts,
        "method": method,
        "manifest": "expert_prune_manifest.json",
    }
    _write_json(config_path, config)

def _router_match(key: str) -> re.Match[str] | None:
    return ROUTER_RE.match(key) or SANITIZED_ROUTER_RE.match(key)


def _compute_keep_indices(src: Path, weight_map: dict[str, str], keep_experts: int) -> dict[int, np.ndarray]:
    router_keys = sorted(
        (key for key in weight_map if _router_match(key)),
        key=lambda key: int(_router_match(key).group(1)),  # type: ignore[union-attr]
    )
    if not router_keys:
        raise RuntimeError("No router gate weights found for expert scoring.")

    keep_by_layer: dict[int, np.ndarray] = {}
    for key in router_keys:
        match = _router_match(key)
        assert match is not None
        layer = int(match.group(1))
        with safe_open(str(src / weight_map[key]), framework="np") as handle:
            router = handle.get_tensor(key)
        if router.ndim != 2:
            raise RuntimeError(f"Router tensor {key} has unexpected shape {router.shape}")
        if keep_experts > router.shape[0]:
            raise RuntimeError(
                f"keep_experts={keep_experts} exceeds router experts={router.shape[0]}"
            )
        scores = np.sum(router.astype(np.float32) * router.astype(np.float32), axis=1)
        ranked = np.argsort(scores, kind="stable")
        keep = np.sort(ranked[-keep_experts:]).astype(np.int64)
        keep_by_layer[layer] = keep
    return keep_by_layer


def _load_keep_indices_from_profile(
    src: Path,
    weight_map: dict[str, str],
    profile_path: Path,
    keep_experts: int,
    *,
    coverage_key: str = "keep_by_probability_coverage",
) -> dict[int, np.ndarray]:
    """Build a keep map from a router profile, filling gaps by router-row L2.

    Existing N2 router profiles were collected at several smaller keep counts
    (for example 384), not exactly 435. For a light prune we keep every ranked
    profile expert available, then fill the remaining slots with highest router
    row-L2 experts that were not already selected.
    """
    profile = _load_json(profile_path)
    layers = profile.get("layers")
    if not isinstance(layers, dict):
        raise RuntimeError(f"profile missing layers dict: {profile_path}")

    router_l2 = _compute_keep_indices(src, weight_map, original_num_experts(weight_map))
    keep_by_layer: dict[int, np.ndarray] = {}
    for layer_s, layer_data in layers.items():
        layer = int(layer_s)
        if not isinstance(layer_data, dict):
            raise RuntimeError(f"profile layer {layer} is not a dict")
        coverage = layer_data.get(coverage_key)
        if not isinstance(coverage, dict):
            raise RuntimeError(f"profile layer {layer} missing {coverage_key}")
        candidates = sorted((int(k) for k in coverage.keys()), reverse=True)
        ranked: list[int] = []
        for k in candidates:
            experts = coverage[str(k)].get("experts") if isinstance(coverage[str(k)], dict) else None
            if isinstance(experts, list):
                ranked = [int(v) for v in experts]
                break
        if not ranked:
            raise RuntimeError(f"profile layer {layer} has no expert ranking")

        selected: list[int] = []
        seen: set[int] = set()
        for expert in ranked:
            if 0 <= expert < original_num_experts(weight_map) and expert not in seen:
                selected.append(expert)
                seen.add(expert)
            if len(selected) >= keep_experts:
                break

        if len(selected) < keep_experts:
            # _compute_keep_indices returns sorted top-L2 experts. Reverse the
            # router score sort by recomputing a full ranking for this layer.
            full_rank = _router_l2_rank_for_layer(src, weight_map, layer)
            for expert in full_rank:
                if expert not in seen:
                    selected.append(int(expert))
                    seen.add(int(expert))
                if len(selected) >= keep_experts:
                    break

        if len(selected) != keep_experts:
            raise RuntimeError(
                f"profile layer {layer} produced {len(selected)} experts, expected {keep_experts}"
            )
        keep_by_layer[layer] = np.sort(np.asarray(selected, dtype=np.int64))

    return keep_by_layer


def original_num_experts(weight_map: dict[str, str]) -> int:
    # N2 Pro currently uses 512 routed experts. Keep this helper separate so
    # future variants can replace it without touching profile logic.
    return 512


def _router_l2_rank_for_layer(src: Path, weight_map: dict[str, str], layer: int) -> np.ndarray:
    wanted = [
        key for key in weight_map
        if (m := _router_match(key)) is not None and int(m.group(1)) == layer
    ]
    if not wanted:
        raise RuntimeError(f"No router gate weight found for layer {layer}")
    key = wanted[0]
    with safe_open(str(src / weight_map[key]), framework="np") as handle:
        router = handle.get_tensor(key)
    scores = np.sum(router.astype(np.float32) * router.astype(np.float32), axis=1)
    return np.argsort(-scores, kind="stable").astype(np.int64)


def _slice_tensor(name: str, tensor: np.ndarray, keep_by_layer: dict[int, np.ndarray]) -> np.ndarray:
    switch = SWITCH_RE.match(name)
    if switch:
        layer = int(switch.group(1))
        keep = keep_by_layer[layer]
        return tensor[keep, ...]
    tq_switch = TQ_SWITCH_RE.match(name)
    if tq_switch:
        suffix = tq_switch.group(3)
        if suffix == "tq_bits":
            return tensor
        layer = int(tq_switch.group(1))
        keep = keep_by_layer[layer]
        return tensor[keep, ...]
    router = ROUTER_RE.match(name)
    if router is None:
        router = SANITIZED_ROUTER_RE.match(name)
    if router:
        layer = int(router.group(1))
        keep = keep_by_layer[layer]
        return tensor[keep, ...]
    return tensor


def _write_pruned_shards(src: Path, dst: Path, index: dict[str, Any], keep_by_layer: dict[int, np.ndarray]) -> dict[str, Any]:
    weight_map: dict[str, str] = dict(index["weight_map"])
    shards = sorted(set(weight_map.values()))
    total_tensor_bytes = 0
    tensor_count = 0
    sliced_count = 0
    shape_updates: dict[str, list[int]] = {}
    for shard_name in shards:
        in_path = src / shard_name
        out_path = dst / shard_name
        tensors: dict[str, np.ndarray] = {}
        metadata = None
        with safe_open(str(in_path), framework="np") as handle:
            metadata = handle.metadata()
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                pruned = _slice_tensor(key, tensor, keep_by_layer)
                if pruned.shape != tensor.shape:
                    sliced_count += 1
                    shape_updates[key] = [int(dim) for dim in pruned.shape]
                tensors[key] = pruned
                total_tensor_bytes += int(pruned.nbytes)
                tensor_count += 1
        save_file(tensors, str(out_path), metadata=metadata)
        print(f"wrote {shard_name}: tensors={len(tensors)}", flush=True)

    total_size = sum((dst / shard_name).stat().st_size for shard_name in shards)
    new_index = dict(index)
    new_metadata = dict(new_index.get("metadata") or {})
    new_metadata["total_size"] = total_size
    new_metadata["expert_pruned_from"] = "512"
    new_metadata["expert_pruned_to"] = str(len(next(iter(keep_by_layer.values()))))
    new_index["metadata"] = new_metadata
    (dst / "model.safetensors.index.json").write_text(
        json.dumps(new_index, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "tensor_count": tensor_count,
        "sliced_tensor_count": sliced_count,
        "total_tensor_bytes": total_tensor_bytes,
        "index_total_size": total_size,
        "shard_count": len(shards),
        "shape_updates": shape_updates,
    }


def _update_jang_metadata(
    dst: Path,
    *,
    keep_experts: int,
    original_experts: int,
    shape_updates: dict[str, list[int]],
    method: str = "router_row_l2_topk_per_layer",
    evidence_level: str = "router_weight_proxy",
) -> None:
    jang_path = dst / "jang_config.json"
    if not jang_path.exists():
        return

    jang = _load_json(jang_path)
    jang["expert_pruning"] = {
        "source_num_experts": original_experts,
        "num_experts": keep_experts,
        "dropped_experts_per_layer": original_experts - keep_experts,
        "method": method,
        "manifest": "expert_prune_manifest.json",
        "evidence_level": evidence_level,
        "note": (
            "Built as a light N2 routed-expert prune candidate. Expert "
            "selection should use activation/router probability coverage when "
            "available, with router-row-L2 only as a fill strategy."
        ),
    }

    manifest = (
        jang.get("quantization", {})
        .get("tensor_quantization_manifest", {})
    )
    if isinstance(manifest, dict):
        for entry in manifest.values():
            if not isinstance(entry, dict):
                continue
            for field, shape_field in (
                ("weight_key", "weight_shape"),
                ("scales_key", "scales_shape"),
                ("biases_key", "biases_shape"),
            ):
                key = entry.get(field)
                if isinstance(key, str) and key in shape_updates:
                    entry[shape_field] = shape_updates[key]
        q = jang.setdefault("quantization", {})
        q["tensor_quantization_manifest_count"] = len(manifest)

    _write_json(jang_path, jang)


def _manifest(
    *,
    src: Path,
    dst: Path,
    keep_by_layer: dict[int, np.ndarray],
    stats: dict[str, Any],
    method: str,
    evidence_level: str,
    profile: Path | None = None,
) -> dict[str, Any]:
    layers = {}
    for layer, keep in sorted(keep_by_layer.items()):
        kept = set(int(x) for x in keep.tolist())
        dropped = [idx for idx in range(512) if idx not in kept]
        layers[str(layer)] = {
            "keep_count": len(keep),
            "drop_count": len(dropped),
            "keep": [int(x) for x in keep.tolist()],
            "drop": dropped,
        }
    return {
        "schema": "nex-n2-jang-expert-prune-v1",
        "source": str(src),
        "output": str(dst),
        "method": method,
        "evidence_level": evidence_level,
        "profile": str(profile) if profile else None,
        "source_num_experts": 512,
        "num_experts": len(next(iter(keep_by_layer.values()))),
        "num_layers": len(keep_by_layer),
        "stats": stats,
        "caveat": (
            "This is a light routed-expert prune candidate. It must pass live "
            "runtime coherency before it is treated as a keepable model."
        ),
        "layers": layers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--keep-experts", type=int, default=416)
    parser.add_argument("--expert-prune-map", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    src = args.src
    dst = args.dst
    if not (src / "model.safetensors.index.json").exists():
        raise SystemExit(f"missing source index: {src}")
    if dst.exists():
        if not args.force:
            raise SystemExit(f"output exists: {dst}")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    index = _load_json(src / "model.safetensors.index.json")
    weight_map: dict[str, str] = dict(index["weight_map"])
    if args.expert_prune_map is not None:
        keep_by_layer = _load_keep_indices_from_profile(
            src,
            weight_map,
            args.expert_prune_map,
            args.keep_experts,
        )
        method = "activation_probability_coverage_fill_router_l2"
        evidence_level = "router_profile_probability_plus_l2_fill"
    else:
        keep_by_layer = _compute_keep_indices(src, weight_map, args.keep_experts)
        method = "router_row_l2_topk_per_layer"
        evidence_level = "router_weight_proxy"
    original_experts = 512
    _copy_sidecars(
        src,
        dst,
        keep_experts=args.keep_experts,
        original_experts=original_experts,
        method=method,
    )
    stats = _write_pruned_shards(src, dst, index, keep_by_layer)
    shape_updates = stats.pop("shape_updates")
    _update_jang_metadata(
        dst,
        keep_experts=args.keep_experts,
        original_experts=original_experts,
        shape_updates=shape_updates,
        method=method,
        evidence_level=evidence_level,
    )
    manifest = _manifest(
        src=src,
        dst=dst,
        keep_by_layer=keep_by_layer,
        stats=stats,
        method=method,
        evidence_level=evidence_level,
        profile=args.expert_prune_map,
    )
    _write_json(dst / "expert_prune_manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "layers"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
