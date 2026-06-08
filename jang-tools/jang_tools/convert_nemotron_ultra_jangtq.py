"""Nemotron 3 Ultra NVFP4 -> smallest coherent JANGTQ converter.

The target profile is deliberately not uniform precision. It keeps the
Nemotron-H control plane coherent while compressing the parameter-dominant
routed expert bank:

* routed backbone experts: ModelOpt NVFP4 -> 1-bit TurboQuant
* source FP8 always-on/shared projections: dequant -> 8-bit MLX affine
* BF16/F32 control tensors, attention, embeddings, lm_head: passthrough
* MTP tensors: dropped for the hard under-128 GiB runtime target

This module streams one source tensor at a time and one routed expert at a time.
It must not stack full 512-expert layers during conversion.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from dataclasses import asdict, dataclass
from math import prod
from pathlib import Path
import re

import mlx.core as mx
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from jang_tools.step37.nvfp4_codec import dequant_nvfp4_modelopt
from jang_tools.turboquant.linear import tq_quantize_weight


SEED = 42
PROFILE = "JANGTQ_1L"
ROUTED_BITS = 1
FP8_AFFINE_BITS = 8
FP8_AFFINE_GROUP = 128
MAX_SHARD_BYTES_DEFAULT = int(2.0 * 1024**3)
JSON_SAFE_INFINITY = 1.0e20


@dataclass(frozen=True)
class TensorPlan:
    name: str
    dtype: str
    shape: tuple[int, ...]
    shard: str
    action: str
    bits: int
    estimated_bytes: int


def _read_header(path: Path) -> dict:
    with path.open("rb") as f:
        hsize = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(hsize))


def _load_index(src: Path) -> dict:
    path = src / "model.safetensors.index.json"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_indexed_tensors(src: Path) -> list[tuple[str, str, tuple[int, ...], str]]:
    index = _load_index(src)
    by_shard: dict[str, list[str]] = {}
    for name, shard in index["weight_map"].items():
        by_shard.setdefault(shard, []).append(name)

    out: list[tuple[str, str, tuple[int, ...], str]] = []
    for shard, names in sorted(by_shard.items()):
        header = _read_header(src / shard)
        for name in sorted(names):
            meta = header[name]
            out.append((name, str(meta["dtype"]), tuple(meta["shape"]), shard))
    return out


def _dtype_bytes(dtype: str) -> int:
    return {
        "BF16": 2,
        "F16": 2,
        "F32": 4,
        "F8_E4M3": 1,
        "U8": 1,
        "U32": 4,
    }.get(dtype, 0)


def _is_routed_backbone_expert_weight(name: str, dtype: str) -> bool:
    return (
        dtype == "U8"
        and name.startswith("backbone.layers.")
        and ".mixer.experts." in name
        and (name.endswith(".up_proj.weight") or name.endswith(".down_proj.weight"))
    )


def _is_source_fp8_projection_weight(name: str, dtype: str) -> bool:
    if dtype != "F8_E4M3" or not name.endswith(".weight"):
        return False
    return (
        (name.startswith("backbone.layers.") and (".mixer.in_proj.weight" in name or ".mixer.out_proj.weight" in name))
        or (name.startswith("backbone.layers.") and ".mixer.shared_experts." in name)
    )


def _is_modelopt_scale_sidecar(name: str) -> bool:
    return (
        name.endswith(".input_scale")
        or name.endswith(".weight_scale")
        or name.endswith(".weight_scale_2")
        or name.endswith(".k_scale")
        or name.endswith(".v_scale")
    )


def _is_mtp_tensor(name: str) -> bool:
    return name.startswith("mtp.")


def classify_tensor(name: str, dtype: str) -> tuple[str, int]:
    """Return conversion action and nominal bits for the smallest coherent profile."""
    if _is_mtp_tensor(name):
        return "drop_mtp", 0
    if _is_routed_backbone_expert_weight(name, dtype):
        return "routed_tq", ROUTED_BITS
    if _is_source_fp8_projection_weight(name, dtype):
        return "fp8_dequant_affine", FP8_AFFINE_BITS
    if _is_modelopt_scale_sidecar(name):
        return "skip_source_scale", 0
    if dtype == "F32":
        return "passthrough", 32
    if dtype == "BF16":
        return "passthrough", 16
    if dtype == "F8_E4M3":
        return "passthrough", 8
    if dtype == "U8":
        return "passthrough", 8
    return "passthrough", 0


def planned_output_keys(name: str, action: str) -> list[str]:
    if action == "routed_tq":
        base = _routed_output_base(name)
        return [f"{base}.tq_packed", f"{base}.tq_norms", f"{base}.tq_bits"]
    if action == "fp8_dequant_affine":
        base = name[: -len(".weight")]
        return [f"{base}.weight", f"{base}.scales", f"{base}.biases"]
    if action in {"skip_source_scale", "drop_mtp"}:
        return []
    return [name]


_ROUTED_RE = re.compile(
    r"^(backbone\.layers\.\d+\.mixer\.)experts\.(\d+)\.(up_proj|down_proj)\.weight$"
)


def _routed_group(name: str) -> tuple[str, int] | None:
    m = _ROUTED_RE.match(name)
    if not m:
        return None
    prefix, expert, proj = m.groups()
    mapped = {"up_proj": "fc1", "down_proj": "fc2"}[proj]
    return f"{prefix}switch_mlp.{mapped}", int(expert)


def _routed_output_base(name: str) -> str:
    group = _routed_group(name)
    if group is None:
        return name[: -len(".weight")]
    return group[0]


def _estimate_tq_bytes(shape: tuple[int, ...], bits: int) -> int:
    rows = prod(shape[:-1])
    in_dim = shape[-1] * 2
    packed_cols = (in_dim + (32 // bits) - 1) // (32 // bits)
    return int(rows * packed_cols * 4 + rows * 2 + 1)


def _estimate_affine_bytes(shape: tuple[int, ...], bits: int, group_size: int) -> int:
    rows = prod(shape[:-1])
    in_dim = shape[-1]
    qweight = rows * ((in_dim * bits) // 8)
    scales_biases = rows * (in_dim // group_size) * 4
    return int(qweight + scales_biases)


def build_plan(src: Path) -> tuple[list[TensorPlan], dict]:
    plans: list[TensorPlan] = []
    stats = {"by_action": {}, "estimated_output_gib": 0.0}
    estimated = 0
    for name, dtype, shape, shard in _iter_indexed_tensors(src):
        action, bits = classify_tensor(name, dtype)
        if action == "routed_tq":
            est = _estimate_tq_bytes(shape, bits)
        elif action == "fp8_dequant_affine":
            est = _estimate_affine_bytes(shape, bits, FP8_AFFINE_GROUP)
        elif action in {"skip_source_scale", "drop_mtp"}:
            est = 0
        else:
            est = prod(shape) * _dtype_bytes(dtype)
        plans.append(TensorPlan(name, dtype, shape, shard, action, bits, int(est)))
        stats["by_action"][action] = stats["by_action"].get(action, 0) + 1
        estimated += est
    stats["estimated_output_gib"] = round(estimated / 1024**3, 3)
    return plans, stats


class TorchShardWriter:
    def __init__(self, out: Path, max_shard_bytes: int) -> None:
        self.out = out
        self.max_shard_bytes = max_shard_bytes
        self.shard_idx = 0
        self.tensors: dict[str, torch.Tensor] = {}
        self.shard_bytes = 0
        self.weight_map: dict[str, str] = {}

    def add(self, name: str, tensor: torch.Tensor | np.ndarray) -> None:
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(np.ascontiguousarray(tensor))
        tensor = tensor.detach().cpu().contiguous()
        nbytes = tensor.numel() * tensor.element_size()
        if self.tensors and self.shard_bytes + nbytes > self.max_shard_bytes:
            self.flush()
        self.tensors[name] = tensor
        self.shard_bytes += nbytes
        if self.shard_bytes >= self.max_shard_bytes:
            self.flush()

    def flush(self) -> None:
        if not self.tensors:
            return
        self.shard_idx += 1
        name = f"model-{self.shard_idx:05d}-of-XXXXX.safetensors"
        save_file(self.tensors, str(self.out / name), metadata={"format": "jangtq"})
        for key in self.tensors:
            self.weight_map[key] = name
        print(f"  wrote {name}: {len(self.tensors)} tensors, {self.shard_bytes / 1024**3:.3f} GiB", flush=True)
        self.tensors = {}
        self.shard_bytes = 0

    def finalize(self) -> None:
        self.flush()
        final_map: dict[str, str] = {}
        for old in sorted(set(self.weight_map.values())):
            idx = int(old.split("-")[1])
            new = f"model-{idx:05d}-of-{self.shard_idx:05d}.safetensors"
            (self.out / old).rename(self.out / new)
        for key, old in self.weight_map.items():
            idx = int(old.split("-")[1])
            final_map[key] = f"model-{idx:05d}-of-{self.shard_idx:05d}.safetensors"
        self.weight_map = final_map
        total = sum((self.out / f).stat().st_size for f in set(final_map.values()))
        (self.out / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"format": "jangtq", "total_size": total}, "weight_map": final_map}, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _fp8_dequant_affine(weight: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    decoded = (weight.float() * scale.float()).to(torch.float16).cpu().numpy()
    w = mx.array(decoded)
    qw, scales, biases = mx.quantize(w, group_size=FP8_AFFINE_GROUP, bits=FP8_AFFINE_BITS)
    mx.eval(qw, scales, biases)
    return (
        torch.from_numpy(np.array(qw)),
        torch.from_numpy(np.array(scales).astype(np.float16)),
        torch.from_numpy(np.array(biases).astype(np.float16)),
    )


def _nvfp4_to_tq(weight: torch.Tensor, scale: torch.Tensor, scale2: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    decoded = dequant_nvfp4_modelopt(weight, scale, scale2, out_dtype=torch.float16).cpu().numpy()
    result = tq_quantize_weight(decoded, bits=bits, seed=SEED)
    return (
        torch.from_numpy(np.asarray(result["packed"])),
        torch.from_numpy(np.asarray(result["norms"]).astype(np.float16)),
        torch.tensor([bits], dtype=torch.uint8),
    )


def _flush_routed_groups(
    writer: TorchShardWriter,
    groups: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
) -> None:
    for base, by_expert in sorted(groups.items()):
        expert_ids = sorted(by_expert)
        if expert_ids != list(range(len(expert_ids))):
            raise RuntimeError(f"{base} has non-contiguous expert ids: first={expert_ids[:5]} last={expert_ids[-5:]}")
        packed = torch.stack([by_expert[i][0] for i in expert_ids], dim=0)
        norms = torch.stack([by_expert[i][1] for i in expert_ids], dim=0)
        bits = by_expert[expert_ids[0]][2]
        writer.add(f"{base}.tq_packed", packed)
        writer.add(f"{base}.tq_norms", norms)
        writer.add(f"{base}.tq_bits", bits)


def _copy_metadata(src: Path, out: Path) -> None:
    skip = {".safetensors", ".json"}
    for p in src.iterdir():
        if p.name.startswith("model-") or p.name == "model.safetensors.index.json":
            continue
        if p.suffix == ".safetensors":
            continue
        if p.is_file():
            shutil.copy2(p, out / p.name)


def _write_configs(src: Path, out: Path, plans: list[TensorPlan], writer: TorchShardWriter) -> None:
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    cfg["weight_format"] = "mxtq"
    if "num_hidden_layers" not in cfg and isinstance(cfg.get("layers_block_type"), list):
        cfg["num_hidden_layers"] = len(cfg["layers_block_type"])
    if isinstance(cfg.get("time_step_limit"), list):
        cfg["time_step_limit"] = [
            JSON_SAFE_INFINITY
            if isinstance(x, dict) and x.get("__float__") == "Infinity"
            else x
            for x in cfg["time_step_limit"]
        ]
    cfg["num_nextn_predict_layers"] = 0
    cfg["mtp_layers_block_type"] = []
    cfg["mxtq_bits"] = {
        "routed_expert": {"up_proj": ROUTED_BITS, "down_proj": ROUTED_BITS},
        "mamba_projection": FP8_AFFINE_BITS,
        "shared_expert": FP8_AFFINE_BITS,
    }
    cfg["quantization"] = {"group_size": FP8_AFFINE_GROUP, "bits": FP8_AFFINE_BITS}
    cfg["_jang_source"] = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
    cfg["_jang_profile"] = PROFILE
    (out / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")

    by_action: dict[str, int] = {}
    estimated = 0
    for p in plans:
        by_action[p.action] = by_action.get(p.action, 0) + 1
        estimated += p.estimated_bytes
    jang = {
        "format": "jangtq",
        "format_version": "2.0",
        "profile": PROFILE,
        "mxtq_seed": SEED,
        "mxtq_bits": cfg["mxtq_bits"],
        "source_model": {
            "name": "NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
            "hub_id": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
            "architecture": "nemotron_h",
            "dtype": "mixed: BF16/F32/F8_E4M3/NVFP4",
        },
        "quantization": {
            "method": "routed-tq1-control-plane-preserved",
            "profile": PROFILE,
            "drops_mtp": True,
            "routed_expert_bits": {"up_proj": ROUTED_BITS, "down_proj": ROUTED_BITS},
            "fp8_projection_affine_bits": FP8_AFFINE_BITS,
            "fp8_projection_group_size": FP8_AFFINE_GROUP,
            "estimated_output_gib": round(estimated / 1024**3, 3),
            "actions": by_action,
        },
        "runtime": {
            "shard_count": writer.shard_idx,
            "total_shard_bytes": sum((out / name).stat().st_size for name in set(writer.weight_map.values())),
            "keeps_mtp": False,
            "keeps_attention_bf16": True,
            "keeps_router_gates_source_precision": True,
            "keeps_latent_moe_bf16": True,
        },
        "capabilities": {
            "family": "nemotron_h",
            "modality": "text",
            "cache_type": "hybrid",
            "reasoning_parser": "deepseek_r1",
            "tool_parser": "nemotron",
            "think_in_template": True,
            "supports_tools": True,
            "supports_thinking": True,
        },
    }
    (out / "jang_config.json").write_text(json.dumps(jang, indent=2, sort_keys=True), encoding="utf-8")


def convert(src: Path, out: Path, *, max_shard_bytes: int, force: bool = False) -> None:
    plans, stats = build_plan(src)
    if out.exists() and any(out.glob("model-*.safetensors")):
        if not force:
            raise SystemExit(f"{out} already contains model shards; pass --force")
        for p in out.glob("model-*.safetensors"):
            p.unlink()
        index = out / "model.safetensors.index.json"
        if index.exists():
            index.unlink()
    out.mkdir(parents=True, exist_ok=True)
    _copy_metadata(src, out)

    by_name = {p.name: p for p in plans}
    index = _load_index(src)["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for name, shard in index.items():
        by_shard.setdefault(shard, []).append(name)

    writer = TorchShardWriter(out, max_shard_bytes)
    total = sum(1 for p in plans if p.action not in {"skip_source_scale", "drop_mtp"})
    done = 0
    print(json.dumps({"source": str(src), "output": str(out), "profile": PROFILE, "stats": stats}, indent=2), flush=True)
    for shard, names in sorted(by_shard.items()):
        print(f"reading {shard}", flush=True)
        routed_groups: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
        with safe_open(str(src / shard), framework="pt", device="cpu") as f:
            keys = set(f.keys())

            def load(name: str) -> torch.Tensor:
                owner = index[name]
                if owner == shard and name in keys:
                    return f.get_tensor(name)
                with safe_open(str(src / owner), framework="pt", device="cpu") as side:
                    return side.get_tensor(name)

            for name in sorted(names):
                plan = by_name[name]
                if plan.action in {"skip_source_scale", "drop_mtp"}:
                    continue
                if plan.action == "routed_tq":
                    base = name[: -len(".weight")]
                    packed, norms, bits = _nvfp4_to_tq(
                        load(name),
                        load(f"{base}.weight_scale"),
                        load(f"{base}.weight_scale_2"),
                        plan.bits,
                    )
                    group = _routed_group(name)
                    if group is None:
                        raise RuntimeError(f"could not route TQ tensor {name}")
                    out_base, expert_id = group
                    routed_groups.setdefault(out_base, {})[expert_id] = (packed, norms, bits)
                elif plan.action == "fp8_dequant_affine":
                    base = name[: -len(".weight")]
                    qw, scales, biases = _fp8_dequant_affine(load(name), load(f"{base}.weight_scale"))
                    for key, tensor in zip(planned_output_keys(name, plan.action), (qw, scales, biases), strict=True):
                        writer.add(key, tensor)
                else:
                    writer.add(name, load(name))
                done += 1
                if done % 1000 == 0 or done == total:
                    print(f"  progress {done}/{total}: {name}", flush=True)
        _flush_routed_groups(writer, routed_groups)
    writer.finalize()
    _write_configs(src, out, plans, writer)

    from jang_tools.build_jangtq_sidecar import main as build_sidecar
    import sys

    old_argv = sys.argv[:]
    try:
        sys.argv = ["build_jangtq_sidecar", str(out)]
        build_sidecar()
    finally:
        sys.argv = old_argv
    print(f"done: {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-shard-gb", type=float, default=2.0)
    args = parser.parse_args()

    src = args.src.expanduser()
    out = args.out.expanduser()
    plans, stats = build_plan(src)
    if args.dry_run:
        by_action_gib: dict[str, float] = {}
        for p in plans:
            by_action_gib[p.action] = by_action_gib.get(p.action, 0.0) + p.estimated_bytes / 1024**3
        print(json.dumps({
            "source": str(src),
            "output": str(out),
            "profile": PROFILE,
            "stats": stats,
            "by_action_gib": {k: round(v, 3) for k, v in sorted(by_action_gib.items())},
            "sample": [asdict(p) for p in plans[:20]],
        }, indent=2))
        return
    convert(src, out, max_shard_bytes=int(args.max_shard_gb * 1024**3), force=args.force)


if __name__ == "__main__":
    main()
