"""Prune Nex/N2 JANG affine routed experts from an already-converted bundle.

This is a streaming artifact pruner for the qwen3_5_moe/Nex layout produced by
the classic affine converter. It slices routed expert axis 0 and matching router
gate rows, preserving all other tensors and sidecars.
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
ROUTER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.gate\.weight$")


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


def _copy_sidecars(src: Path, dst: Path, *, keep_experts: int, original_experts: int) -> None:
    for name in SIDECARS:
        source = src / name
        if source.exists():
            shutil.copy2(source, dst / name)

    config_path = dst / "config.json"
    config = _load_json(config_path)
    text_config = config.setdefault("text_config", {})
    text_config["num_experts"] = keep_experts
    config["jang_expert_pruning"] = {
        "source_num_experts": original_experts,
        "num_experts": keep_experts,
        "dropped_experts_per_layer": original_experts - keep_experts,
        "method": "router_row_l2_topk_per_layer",
        "manifest": "expert_prune_manifest.json",
    }
    _write_json(config_path, config)

    jang_path = dst / "jang_config.json"
    if jang_path.exists():
        jang = _load_json(jang_path)
        jang["expert_pruning"] = {
            "source_num_experts": original_experts,
            "num_experts": keep_experts,
            "dropped_experts_per_layer": original_experts - keep_experts,
            "method": "router_row_l2_topk_per_layer",
            "manifest": "expert_prune_manifest.json",
            "evidence_level": "router_weight_proxy",
            "note": (
                "Built because the unpruned repaired JANG_1L-outproj8 artifact "
                "OOMs on the 128GB vMLX affine runtime before activation traces "
                "can be collected."
            ),
        }
        _write_json(jang_path, jang)


def _compute_keep_indices(src: Path, weight_map: dict[str, str], keep_experts: int) -> dict[int, np.ndarray]:
    router_keys = sorted(
        (key for key in weight_map if ROUTER_RE.match(key)),
        key=lambda key: int(ROUTER_RE.match(key).group(1)),  # type: ignore[union-attr]
    )
    if not router_keys:
        raise RuntimeError("No router gate weights found for expert scoring.")

    keep_by_layer: dict[int, np.ndarray] = {}
    for key in router_keys:
        match = ROUTER_RE.match(key)
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


def _slice_tensor(name: str, tensor: np.ndarray, keep_by_layer: dict[int, np.ndarray]) -> np.ndarray:
    switch = SWITCH_RE.match(name)
    if switch:
        layer = int(switch.group(1))
        keep = keep_by_layer[layer]
        return tensor[keep, ...]
    router = ROUTER_RE.match(name)
    if router:
        layer = int(router.group(1))
        keep = keep_by_layer[layer]
        return tensor[keep, ...]
    return tensor


def _write_pruned_shards(src: Path, dst: Path, index: dict[str, Any], keep_by_layer: dict[int, np.ndarray]) -> dict[str, Any]:
    weight_map: dict[str, str] = dict(index["weight_map"])
    shards = sorted(set(weight_map.values()))
    total_size = 0
    tensor_count = 0
    sliced_count = 0
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
                tensors[key] = pruned
                total_size += int(pruned.nbytes)
                tensor_count += 1
        save_file(tensors, str(out_path), metadata=metadata)
        print(f"wrote {shard_name}: tensors={len(tensors)}", flush=True)

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
        "index_total_size": total_size,
        "shard_count": len(shards),
    }


def _manifest(
    *,
    src: Path,
    dst: Path,
    keep_by_layer: dict[int, np.ndarray],
    stats: dict[str, Any],
    method: str,
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
        "evidence_level": "router_weight_proxy",
        "source_num_experts": 512,
        "num_experts": len(next(iter(keep_by_layer.values()))),
        "num_layers": len(keep_by_layer),
        "stats": stats,
        "caveat": (
            "This is a first buildable 128GB-lane candidate. It uses router row "
            "L2 as a proxy because the unpruned repaired affine artifact OOMs "
            "before live router activation traces can be collected."
        ),
        "layers": layers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--keep-experts", type=int, default=416)
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
    keep_by_layer = _compute_keep_indices(src, weight_map, args.keep_experts)
    original_experts = 512
    _copy_sidecars(src, dst, keep_experts=args.keep_experts, original_experts=original_experts)
    stats = _write_pruned_shards(src, dst, index, keep_by_layer)
    manifest = _manifest(
        src=src,
        dst=dst,
        keep_by_layer=keep_by_layer,
        stats=stats,
        method="router_row_l2_topk_per_layer",
    )
    _write_json(dst / "expert_prune_manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "layers"}, indent=2))
    return 0


def dispatch(argv) -> int:
    """Run the pruner from a raw argv slice (post `prune-n2-experts`).

    Called via early-intercept in __main__ so the tool's own --flags forward
    correctly (argparse REMAINDER can't capture a leading --flag). Keeps the
    tool identical to `python -m jang_tools.prune_n2_jang_experts`.
    """
    import sys as _sys
    _sys.argv = ["jang prune-n2-experts"] + list(argv or [])
    return main()


def register(subparsers) -> None:
    """Register `prune-n2-experts` for `jang --help` discoverability.

    Execution is handled by `dispatch()` via early-intercept in __main__.
    """
    p = subparsers.add_parser(
        "prune-n2-experts", add_help=False,
        help="Streaming post-hoc router-L2 expert pruner for converted "
             "Nex/N2 qwen3_5_moe affine bundles "
             "(<src> <dst> [--keep-experts N] [--force])",
    )
    p.set_defaults(func=lambda a: __import__("sys").exit(dispatch(getattr(a, "_n2_rest", []) or [])))


if __name__ == "__main__":
    raise SystemExit(main())
