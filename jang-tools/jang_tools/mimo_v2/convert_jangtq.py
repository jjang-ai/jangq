"""MiMo-V2.5 -> JANGTQ bundle converter.

This is separate from ``convert_jang``: routed experts are emitted as
pre-stacked TurboQuant triplets under ``switch_mlp`` runtime module names,
while the non-routed control plane stays on the proven MiMo JANG policy.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file as sf_save_torch

from jang_tools.turboquant.linear import tq_quantize_weight

from .convert_jang import (
    _copy_aux_files,
    _normalize_rope,
    classify,
    is_routed_expert_weight,
)
from .weight_loader import MiMoShardIndex


_EXPERT_RUNTIME_PAT = re.compile(
    r"^(model\.layers\.(?P<layer>\d+)\.mlp)\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)
_MIMO_NUM_EXPERTS = 256


@dataclass(frozen=True)
class JANGTQProfile:
    name: str
    routed_expert_bits: dict[str, int]
    default_bits: int = 8
    default_group_size: int = 64
    mxtq_seed: int = 42

    @classmethod
    def parse(cls, raw: str | int) -> "JANGTQProfile":
        key = str(raw).strip().lower().replace("_", "").replace("-", "")
        if key in {"2", "2l", "jangtq2", "jangtq2l"}:
            return cls("JANGTQ_2", {"gate_proj": 2, "up_proj": 2, "down_proj": 2})
        if key in {"224", "2k", "k", "jangtq2k"}:
            return cls("JANGTQ_2K", {"gate_proj": 2, "up_proj": 2, "down_proj": 4})
        if key in {"322", "jangtq322"}:
            return cls("JANGTQ_322", {"gate_proj": 3, "up_proj": 2, "down_proj": 2})
        if key in {"333", "3", "jangtq3"}:
            return cls("JANGTQ_3", {"gate_proj": 3, "up_proj": 3, "down_proj": 3})
        raise ValueError("unknown MiMo JANGTQ profile; use 2, 2k/224, 322, or 333")

    def bits_for_expert_name(self, name: str) -> int:
        m = _EXPERT_RUNTIME_PAT.match(name)
        if m is None:
            raise ValueError(f"not a MiMo routed expert weight: {name}")
        return self.routed_expert_bits[m.group("proj")]


def jangtq_runtime_base_for_expert_weight(name: str) -> str:
    m = _EXPERT_RUNTIME_PAT.match(name)
    if m is None:
        raise ValueError(f"not a MiMo routed expert weight: {name}")
    return f"{m.group(1)}.switch_mlp.{m.group('proj')}"


def tq_tensor_keys_for_expert_weight(name: str) -> tuple[str, str, str]:
    base = jangtq_runtime_base_for_expert_weight(name)
    return f"{base}.tq_packed", f"{base}.tq_norms", f"{base}.tq_bits"


def _write_config_json(src: Path, dst: Path, profile: JANGTQProfile, include_mtp: bool) -> None:
    cfg = json.loads((src / "config.json").read_text())
    cfg.pop("quantization_config", None)
    tokenizer_config = json.loads((src / "tokenizer_config.json").read_text())
    if tokenizer_config.get("chat_template"):
        cfg["chat_template"] = tokenizer_config["chat_template"]
    _normalize_rope(cfg)
    cfg["format"] = "jangtq"
    cfg["quantization"] = {
        "bits": profile.default_bits,
        "group_size": profile.default_group_size,
        "quant_method": "affine",
        "mode": "affine",
        "routed_experts": "tq_prestacked_switch_mlp",
    }
    for layer in range(48):
        cfg["quantization"][f"model.layers.{layer}.self_attn.qkv_proj"] = {
            "bits": 4,
            "group_size": 64,
            "mode": "affine",
        }
        cfg["quantization"][f"model.layers.{layer}.self_attn.o_proj"] = {
            "bits": 4,
            "group_size": 64,
            "mode": "affine",
        }
    cfg["mxtq_bits"] = {"routed_expert": profile.routed_expert_bits}
    cfg["mxtq_seed"] = profile.mxtq_seed
    cfg["routed_expert_bits"] = profile.routed_expert_bits
    cfg["jang_profile"] = profile.name
    cfg["jang_version"] = "v2"
    cfg["capabilities"] = {
        "family": "mimo_v2",
        "modalities": ["text", "vision", "audio"],
        "cache_type": "kv",
        "attention": {
            "full": True,
            "sliding_window": True,
            "sliding_window_size": cfg.get("sliding_window"),
        },
        "reasoning": {"supported": True, "default": True, "parser": "think_xml"},
        "tools": {"supported": True, "parser": "xml_function"},
    }
    cfg["runtime"] = {
        "cache_type": "kv",
        "attention_impl": "hybrid_full_swa_sink",
        "cache_topology": {
            "family": "hybrid_full_swa_kv",
            "prefix_cache": True,
            "l2_disk_cache": True,
            "turboquant_kv": "full_attention_layers_only",
            "swa_layers": "rotating_kv_native",
        },
        "mtp_mode": "preserved_disabled" if include_mtp else "absent",
        "bundle_has_mtp": include_mtp,
        "multimodal_mode": "weights_preserved_text_runtime",
        "quantization_profile": profile.name,
        "routed_expert_bits": profile.routed_expert_bits,
        "tq_layout": "prestacked_switch_mlp",
    }
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))


def write_jangtq_metadata(src: Path, dst: Path, profile: JANGTQProfile, include_mtp: bool = False) -> None:
    _write_config_json(src, dst, profile, include_mtp=include_mtp)
    jang_cfg = {
        "format": "jangtq",
        "family": "mimo_v2",
        "profile": profile.name,
        "mxtq_seed": profile.mxtq_seed,
        "mxtq_bits": {"routed_expert": profile.routed_expert_bits},
        "routed_expert_bits": profile.routed_expert_bits,
        "tq_layout": "prestacked_switch_mlp",
        "num_experts": _MIMO_NUM_EXPERTS,
    }
    (dst / "jang_config.json").write_text(json.dumps(jang_cfg, indent=2))


def _quantize_affine_cpu_u32(t: torch.Tensor, *, group_size: int, bits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if bits not in {2, 3, 4, 5, 6, 8}:
        raise ValueError(f"unsupported affine bits={bits}")
    x = t.detach().cpu().float().numpy()
    if x.ndim != 2:
        raise ValueError(f"expected 2D weight for CPU affine pack, got shape {x.shape}")
    rows, cols = x.shape
    if cols % group_size != 0:
        raise ValueError(f"weight cols {cols} not divisible by group_size {group_size}")
    groups = cols // group_size
    if (group_size * bits) % 32 != 0:
        raise ValueError(f"group_size={group_size} and bits={bits} do not pack into whole uint32 words")
    words_per_group = (group_size * bits) // 32
    xr = x.reshape(rows, groups, group_size)
    maxv = xr.max(axis=2).astype(np.float32)
    minv = xr.min(axis=2).astype(np.float32)
    scale = ((minv - maxv) / float((1 << bits) - 1)).astype(np.float32)
    scale = np.minimum(scale, np.float32(-1e-7))
    bias = maxv.astype(np.float32)
    q = np.rint((xr - bias[:, :, None]) / scale[:, :, None])
    q = np.clip(q, 0, (1 << bits) - 1).astype(np.uint32)
    packed_words = np.zeros((rows, groups, words_per_group), dtype=np.uint32)
    mask = np.uint32((1 << bits) - 1)
    for i_col in range(group_size):
        bit_offset = i_col * bits
        word_idx = bit_offset // 32
        shift = bit_offset % 32
        vals = q[:, :, i_col] & mask
        packed_words[:, :, word_idx] |= vals << np.uint32(shift)
        spill = shift + bits - 32
        if spill > 0:
            packed_words[:, :, word_idx + 1] |= vals >> np.uint32(bits - spill)
    return (
        torch.from_numpy(packed_words.reshape(rows, groups * words_per_group)),
        torch.from_numpy(scale),
        torch.from_numpy(bias),
    )


def _torch_from_np(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(arr))


def convert(
    src: Path,
    dst: Path,
    profile_bits: str | int = "2",
    *,
    max_shard_bytes: int = 1_000_000_000,
    include_mtp: bool = False,
    dry_run: bool = False,
) -> None:
    profile = JANGTQProfile.parse(profile_bits)
    idx = MiMoShardIndex(src)
    keys = idx.weight_keys

    print(f"[mimo-jangtq] source: {src}")
    print(f"[mimo-jangtq] target: {dst}")
    print(f"[mimo-jangtq] profile: {profile.name} routed={profile.routed_expert_bits} seed={profile.mxtq_seed}")
    print(f"[mimo-jangtq] MTP tensors: {'preserve' if include_mtp else 'drop'}")
    if dry_run:
        routed = sum(1 for k in keys if is_routed_expert_weight(k))
        non_routed = sum(1 for k in keys if k.endswith(".weight") and not is_routed_expert_weight(k))
        print(f"[mimo-jangtq] dry-run: routed expert weights={routed}, non-routed weights={non_routed}")
        return

    dst.mkdir(parents=True, exist_ok=True)
    shard_idx = 1
    shard_bytes = 0
    shard_buf: dict[str, torch.Tensor] = {}
    shard_map: dict[str, str] = {}
    expert_stash: dict[tuple[int, str], dict[int, tuple[np.ndarray, np.ndarray, int]]] = {}
    method_totals = {"tq_expert": 0, "affine": 0, "passthrough_bf16": 0, "passthrough_fp32": 0}
    t_start = time.time()

    def flush_shard() -> None:
        nonlocal shard_idx, shard_bytes, shard_buf
        if not shard_buf:
            return
        shard_name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        sf_save_torch(shard_buf, str(dst / shard_name))
        for k in shard_buf:
            shard_map[k] = shard_name
        print(f"    shard {shard_idx}: {len(shard_buf)} tensors, {shard_bytes / 1e9:.2f} GB", flush=True)
        shard_buf = {}
        shard_bytes = 0
        shard_idx += 1
        gc.collect()

    def add_tensor(name: str, tensor: torch.Tensor) -> None:
        nonlocal shard_bytes
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        shard_buf[name] = tensor.cpu()
        shard_bytes += tensor.element_size() * tensor.numel()
        if shard_bytes >= max_shard_bytes:
            flush_shard()

    def flush_expert_group(layer: int, proj: str) -> None:
        group = expert_stash.get((layer, proj))
        if group is None or len(group) != _MIMO_NUM_EXPERTS:
            return
        packed = np.stack([group[i][0] for i in range(_MIMO_NUM_EXPERTS)], axis=0)
        norms = np.stack([group[i][1] for i in range(_MIMO_NUM_EXPERTS)], axis=0)
        bits = group[0][2]
        base = f"model.layers.{layer}.mlp.switch_mlp.{proj}"
        add_tensor(f"{base}.tq_packed", _torch_from_np(packed).to(torch.uint32))
        add_tensor(f"{base}.tq_norms", _torch_from_np(norms).to(torch.float16))
        add_tensor(f"{base}.tq_bits", torch.tensor([bits], dtype=torch.int32))
        del expert_stash[(layer, proj)]

    for i, name in enumerate(keys):
        if not include_mtp and name.startswith("model.mtp."):
            continue
        if is_routed_expert_weight(name):
            m = _EXPERT_RUNTIME_PAT.match(name)
            if m is None:
                raise RuntimeError(f"routed expert pattern mismatch: {name}")
            layer = int(m.group("layer"))
            expert = int(m.group("expert"))
            proj = m.group("proj")
            bits = profile.bits_for_expert_name(name)
            w = idx.read_tensor(name, out_dtype=torch.float32).numpy()
            q = tq_quantize_weight(w, bits=bits, seed=profile.mxtq_seed)
            expert_stash.setdefault((layer, proj), {})[expert] = (q["packed"], q["norms"], bits)
            method_totals["tq_expert"] += 1
            flush_expert_group(layer, proj)
            del w, q
        else:
            bits, method, group_size = classify(name, "2")
            if name.startswith("model.layers.") and name.endswith(".self_attn.qkv_proj.weight"):
                bits, method, group_size = 4, "affine", 64
            if name.startswith("model.layers.") and name.endswith(".self_attn.o_proj.weight"):
                bits, method, group_size = 4, "affine", 64
            if method == "skip":
                continue
            if method == "passthrough_bf16":
                add_tensor(name, idx.read_passthrough(name).to(torch.bfloat16))
                method_totals["passthrough_bf16"] += 1
            elif method == "passthrough_fp32":
                add_tensor(name, idx.read_passthrough(name, out_dtype=torch.float32))
                method_totals["passthrough_fp32"] += 1
            elif method == "affine":
                t = idx.read_tensor(name, out_dtype=torch.float32)
                qw, qs, qb = _quantize_affine_cpu_u32(t, group_size=group_size, bits=bits)
                base = name[: -len(".weight")]
                add_tensor(f"{base}.weight", qw)
                add_tensor(f"{base}.scales", qs.to(torch.float16))
                add_tensor(f"{base}.biases", qb.to(torch.float16))
                method_totals["affine"] += 1
                del t, qw, qs, qb
            else:
                raise RuntimeError(f"unknown method {method!r} for {name}")
        if (i + 1) % 250 == 0:
            elapsed = time.time() - t_start
            print(f"    [{i+1:6d}/{len(keys)}] elapsed={elapsed:.0f}s totals={method_totals}", flush=True)

    if expert_stash:
        missing = [f"layer={l} proj={p} got={len(v)}" for (l, p), v in sorted(expert_stash.items())[:8]]
        raise RuntimeError(f"incomplete expert groups: {', '.join(missing)}")

    flush_shard()
    total_shards = shard_idx - 1
    for k in range(1, shard_idx):
        old = dst / f"model-{k:05d}-of-XXXXX.safetensors"
        new = dst / f"model-{k:05d}-of-{total_shards:05d}.safetensors"
        if old.exists():
            old.rename(new)
    final_map = {k: v.replace("XXXXX", f"{total_shards:05d}") for k, v in shard_map.items()}
    total_bytes = sum((dst / fn).stat().st_size for fn in set(final_map.values()))
    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": final_map}, indent=2)
    )
    _copy_aux_files(src, dst)
    write_jangtq_metadata(src, dst, profile, include_mtp=include_mtp)
    subprocess.run([sys.executable, "-m", "jang_tools.build_jangtq_sidecar", str(dst)], check=True)
    print(f"[mimo-jangtq] DONE: {total_shards} shards, {total_bytes / 1e9:.2f} GB, totals={method_totals}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert MiMo-V2.5 source checkpoint to JANGTQ.")
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--dst", required=True, type=Path)
    p.add_argument("--profile", default="2", help="2, 2k/224, 322, or 333")
    p.add_argument("--max-shard-bytes", type=int, default=1_000_000_000)
    p.add_argument("--include-mtp", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    convert(
        args.src.expanduser(),
        args.dst.expanduser(),
        args.profile,
        max_shard_bytes=args.max_shard_bytes,
        include_mtp=args.include_mtp,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
