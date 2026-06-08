"""Tencent Hy3-preview -> all-affine JANG conversion."""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from jang_tools.calibrate import _load_bf16_tensor


DEFAULT_SHARD_BYTES = 1_000_000_000
SEED = 42
SIDECARS = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "chat_template.jinja",
    "chat_template.json",
    "merges.txt",
    "vocab.json",
]


@dataclass(frozen=True)
class Hy3JangPolicy:
    profile: str
    group_size: int
    routed_bits: dict[str, int]
    attention_bits: int
    shared_expert_bits: int
    dense_ffn_bits: int
    embed_bits: int
    lm_head_bits: int
    mtp_policy: str
    num_hidden_layers: int = 80


def profile_policy(
    profile: str = "JANG_2L",
    *,
    mtp_policy: str = "drop",
    num_hidden_layers: int = 80,
) -> Hy3JangPolicy:
    profile_norm = profile.upper()
    if profile_norm not in {"JANG_2L", "JANG_2K"}:
        raise ValueError("Hy3 all-affine converter currently supports JANG_2L and JANG_2K")
    if mtp_policy not in {"drop", "preserve-affine8"}:
        raise ValueError("mtp_policy must be 'drop' or 'preserve-affine8'")
    routed_bits = (
        {"gate_proj": 2, "up_proj": 2, "down_proj": 2}
        if profile_norm == "JANG_2L"
        else {"gate_proj": 2, "up_proj": 2, "down_proj": 3}
    )
    return Hy3JangPolicy(
        profile=profile_norm,
        group_size=128,
        routed_bits=routed_bits,
        attention_bits=8,
        shared_expert_bits=8,
        dense_ffn_bits=8,
        embed_bits=6,
        lm_head_bits=8,
        mtp_policy=mtp_policy,
        num_hidden_layers=num_hidden_layers,
    )


def _layer_index(name: str) -> int | None:
    match = re.search(r"model\.layers\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def classify_tensor(name: str, policy: Hy3JangPolicy) -> tuple[int, str]:
    """Return ``(bits, method)`` where method is affine/passthrough/drop."""
    layer = _layer_index(name)
    if layer is not None and layer >= policy.num_hidden_layers:
        if policy.mtp_policy == "drop":
            return 0, "drop"

    n = name.lower()
    if (
        "norm" in n
        or name.endswith(".bias")
        or name.endswith(".expert_bias")
        or ".expert_bias" in name
        or ".mlp.router.gate.weight" in name
    ):
        return 16, "passthrough"

    if name == "model.embed_tokens.weight":
        return policy.embed_bits, "affine"
    if name == "lm_head.weight":
        return policy.lm_head_bits, "affine"

    if "self_attn" in name and any(
        proj in name for proj in (".q_proj.weight", ".k_proj.weight", ".v_proj.weight", ".o_proj.weight")
    ):
        return policy.attention_bits, "affine"

    if ".mlp.shared_mlp." in name and name.endswith(".weight"):
        return policy.shared_expert_bits, "affine"

    if re.search(r"model\.layers\.0\.mlp\.(gate|up|down)_proj\.weight$", name):
        return policy.dense_ffn_bits, "affine"

    if ".mlp.experts." in name and name.endswith(".weight"):
        if layer is not None and layer >= policy.num_hidden_layers:
            return 8, "affine"
        for proj, bits in policy.routed_bits.items():
            if f".{proj}.weight" in name:
                return bits, "affine"

    if name.endswith(".weight"):
        return 8, "affine"
    return 16, "passthrough"


class ShardedWriter:
    def __init__(self, out_dir: Path, shard_bytes: int):
        self.out_dir = out_dir
        self.shard_bytes = shard_bytes
        self.shard_idx = 0
        self.bytes_in_shard = 0
        self.tensors: dict[str, np.ndarray] = {}
        self.placeholder_map: dict[str, str] = {}

    def add(self, name: str, arr: np.ndarray) -> None:
        self.tensors[name] = arr
        self.bytes_in_shard += arr.nbytes
        if self.bytes_in_shard >= self.shard_bytes:
            self.flush()

    def flush(self) -> None:
        if not self.tensors:
            return
        self.shard_idx += 1
        fname = f"model-{self.shard_idx:05d}-of-XXXXX.safetensors"
        save_file(self.tensors, str(self.out_dir / fname))
        for key in self.tensors:
            self.placeholder_map[key] = fname
        print(
            f"    shard {self.shard_idx}: {len(self.tensors)} tensors, "
            f"{self.bytes_in_shard / 1e9:.2f} GB",
            flush=True,
        )
        self.tensors = {}
        self.bytes_in_shard = 0

    def finalize(self) -> tuple[int, int, dict[str, str]]:
        self.flush()
        total_shards = self.shard_idx
        weight_map: dict[str, str] = {}
        for i in range(1, total_shards + 1):
            old_name = f"model-{i:05d}-of-XXXXX.safetensors"
            new_name = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
            old = self.out_dir / old_name
            if old.exists():
                old.rename(self.out_dir / new_name)
            for key, value in self.placeholder_map.items():
                if value == old_name:
                    weight_map[key] = new_name
        total_size = sum(
            (self.out_dir / shard).stat().st_size for shard in set(weight_map.values())
        )
        return total_shards, total_size, weight_map


def _load_one(src: Path, weight_map: dict[str, str], name: str) -> np.ndarray:
    sf_path = src / weight_map[name]
    with safe_open(str(sf_path), framework="numpy") as f:
        shape = list(f.get_slice(name).get_shape())
        try:
            tensor = f.get_tensor(name)
            if isinstance(tensor, np.ndarray):
                return tensor
            return np.asarray(tensor)
        except Exception:
            return _load_bf16_tensor(sf_path, name, shape)


def _quantize(weight: np.ndarray, *, bits: int, group_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = mx.array(weight.astype(np.float16, copy=False))
    qw, qs, qb = mx.quantize(w, group_size=group_size, bits=bits)
    out = (
        np.array(qw),
        np.array(qs).astype(np.float16),
        np.array(qb).astype(np.float16),
    )
    del w, qw, qs, qb
    mx.metal.clear_cache()
    return out


def _quantize_to_writer(
    writer: ShardedWriter,
    name: str,
    tensor: np.ndarray,
    *,
    bits: int,
    group_size: int,
) -> int:
    qw, qs, qb = _quantize(tensor, bits=bits, group_size=group_size)
    base = name[: -len(".weight")] if name.endswith(".weight") else name
    writer.add(f"{base}.weight", qw)
    writer.add(f"{base}.scales", qs)
    writer.add(f"{base}.biases", qb)
    nbytes = qw.nbytes + qs.nbytes + qb.nbytes
    del qw, qs, qb
    return nbytes


def _copy_sidecars(src: Path, out: Path) -> None:
    for fname in SIDECARS:
        p = src / fname
        if p.exists():
            shutil.copy2(p, out / fname)
    tok_cfg = out / "tokenizer_config.json"
    template = out / "chat_template.jinja"
    if tok_cfg.exists() and template.exists():
        cfg = json.loads(tok_cfg.read_text())
        if not cfg.get("chat_template"):
            cfg["chat_template"] = template.read_text(encoding="utf-8")
            tok_cfg.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))


def _quant_overrides(config: dict, policy: Hy3JangPolicy) -> dict:
    q: dict[str, object] = {
        "bits": 8,
        "group_size": policy.group_size,
        "mode": "affine",
    }
    q["model.embed_tokens"] = {
        "bits": policy.embed_bits,
        "group_size": policy.group_size,
        "mode": "affine",
    }
    n_layers = int(config["num_hidden_layers"])
    first_dense = int(config.get("first_k_dense_replace", 1))
    for layer in range(first_dense, n_layers):
        for proj, bits in policy.routed_bits.items():
            q[f"model.layers.{layer}.mlp.switch_mlp.{proj}"] = {
                "bits": bits,
                "group_size": policy.group_size,
                "mode": "affine",
            }
    return q


def _dry_run(src: Path, policy: Hy3JangPolicy) -> None:
    counts: dict[str, int] = {"affine": 0, "passthrough": 0, "drop": 0}
    samples: dict[str, list[str]] = {"affine": [], "passthrough": [], "drop": []}
    index = json.loads((src / "model.safetensors.index.json").read_text())
    for name in index["weight_map"]:
        _bits, method = classify_tensor(name, policy)
        counts[method] = counts.get(method, 0) + 1
        if len(samples[method]) < 8:
            samples[method].append(name)
    print(json.dumps({"profile": policy.profile, "counts": counts, "sample": samples}, indent=2))


def convert(src: Path, out: Path, policy: Hy3JangPolicy, *, shard_bytes: int) -> None:
    cfg = json.loads((src / "config.json").read_text())
    if cfg.get("model_type") != "hy_v3":
        raise SystemExit(f"expected model_type='hy_v3', got {cfg.get('model_type')!r}")
    index_path = src / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    missing = sorted({s for s in weight_map.values() if not (src / s).exists()})
    if missing:
        raise SystemExit(f"source incomplete: {len(missing)} missing shards; first={missing[0]}")

    out.mkdir(parents=True, exist_ok=True)
    writer = ShardedWriter(out, shard_bytes)
    bytes_in = 0
    bytes_out = 0
    t0 = time.time()
    n_layers = int(cfg["num_hidden_layers"])
    first_dense = int(cfg.get("first_k_dense_replace", 1))
    n_experts = int(cfg["num_experts"])

    print("=" * 64)
    print(f"  Hy3-preview -> {policy.profile} all-affine")
    print("=" * 64)
    print(f"  source: {src}")
    print(f"  output: {out}")
    print(f"  group_size={policy.group_size} routed_bits={policy.routed_bits}")
    print(f"  mtp_policy={policy.mtp_policy}")

    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        bits, method = classify_tensor(name, policy)
        if method == "affine":
            tensor = _load_one(src, weight_map, name)
            bytes_in += tensor.nbytes
            bytes_out += _quantize_to_writer(
                writer, name, tensor, bits=bits, group_size=policy.group_size
            )
            del tensor

    for name in ("model.norm.weight",):
        if name in weight_map:
            tensor = _load_one(src, weight_map, name)
            arr = tensor.astype(np.float16)
            writer.add(name, arr)
            bytes_in += tensor.nbytes
            bytes_out += arr.nbytes
            del tensor, arr

    for layer in range(n_layers):
        prefix = f"model.layers.{layer}"
        t_layer = time.time()

        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            name = f"{prefix}.self_attn.{proj}.weight"
            if name in weight_map:
                bits, method = classify_tensor(name, policy)
                if method == "affine":
                    tensor = _load_one(src, weight_map, name)
                    bytes_in += tensor.nbytes
                    bytes_out += _quantize_to_writer(
                        writer, name, tensor, bits=bits, group_size=policy.group_size
                    )
                    del tensor

        for suffix in (
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "mlp.router.gate.weight",
            "mlp.expert_bias",
        ):
            name = f"{prefix}.{suffix}"
            if name in weight_map:
                bits, method = classify_tensor(name, policy)
                if method == "passthrough":
                    tensor = _load_one(src, weight_map, name)
                    arr = tensor.astype(np.float16)
                    writer.add(name, arr)
                    bytes_in += tensor.nbytes
                    bytes_out += arr.nbytes
                    del tensor, arr

        if layer < first_dense:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                name = f"{prefix}.mlp.{proj}.weight"
                if name in weight_map:
                    bits, method = classify_tensor(name, policy)
                    if method == "affine":
                        tensor = _load_one(src, weight_map, name)
                        bytes_in += tensor.nbytes
                        bytes_out += _quantize_to_writer(
                            writer, name, tensor, bits=bits, group_size=policy.group_size
                        )
                        del tensor
        else:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                shared = f"{prefix}.mlp.shared_mlp.{proj}.weight"
                if shared in weight_map:
                    bits, method = classify_tensor(shared, policy)
                    if method == "affine":
                        tensor = _load_one(src, weight_map, shared)
                        bytes_in += tensor.nbytes
                        bytes_out += _quantize_to_writer(
                            writer, shared, tensor, bits=bits, group_size=policy.group_size
                        )
                        del tensor

            for proj in ("gate_proj", "up_proj", "down_proj"):
                bits = policy.routed_bits[proj]
                first_key = f"{prefix}.mlp.experts.0.{proj}.weight"
                if first_key not in weight_map:
                    continue
                first = _load_one(src, weight_map, first_key)
                stack_shape = (n_experts, *first.shape)
                stack = np.empty(stack_shape, dtype=np.float16)
                stack[0] = first.astype(np.float16)
                bytes_in += first.nbytes
                del first
                for expert in range(1, n_experts):
                    name = f"{prefix}.mlp.experts.{expert}.{proj}.weight"
                    tensor = _load_one(src, weight_map, name)
                    stack[expert] = tensor.astype(np.float16)
                    bytes_in += tensor.nbytes
                    del tensor
                target = f"{prefix}.mlp.switch_mlp.{proj}.weight"
                bytes_out += _quantize_to_writer(
                    writer, target, stack, bits=bits, group_size=policy.group_size
                )
                print(
                    f"    L{layer:02d} {proj}: bits={bits} stacked={stack_shape}",
                    flush=True,
                )
                del stack
                gc.collect()
                mx.metal.clear_cache()

        print(
            f"    L{layer:02d} done in {time.time() - t_layer:.1f}s "
            f"out={bytes_out / 1e9:.2f}GB",
            flush=True,
        )

    total_shards, total_size, final_map = writer.finalize()
    (out / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"format": "jang", "total_size": total_size}, "weight_map": final_map},
            indent=2,
        )
    )

    out_cfg = dict(cfg)
    out_cfg.pop("quantization_config", None)
    out_cfg["weight_format"] = "affine"
    out_cfg["quantization"] = _quant_overrides(cfg, policy)
    out_cfg["_name_or_path"] = f"Hy3-preview-{policy.profile}"
    out_cfg["runtime"] = {
        "bundle_has_mtp": policy.mtp_policy != "drop",
        "mtp_layers": int(cfg.get("num_nextn_predict_layers", 0)),
        "mtp_mode": "dropped_for_smallest_affine" if policy.mtp_policy == "drop" else "preserved_disabled",
        "mtp_status": (
            "MTP tensors were explicitly dropped for the smallest all-affine "
            "runtime target; current Hy3 runtime decodes autoregressively."
            if policy.mtp_policy == "drop"
            else "MTP tensors are preserved but speculative decode is disabled."
        ),
    }
    out_cfg["capabilities"] = {
        "reasoning_parser": "qwen3",
        "tool_parser": "hunyuan",
        "think_in_template": False,
        "supports_tools": True,
        "supports_thinking": True,
        "family": "hy_v3",
        "modality": "text",
        "cache_type": "kv",
    }
    (out / "config.json").write_text(json.dumps(out_cfg, indent=2))

    jang_cfg = {
        "format": "jang",
        "format_version": "2.0",
        "weight_format": "affine",
        "profile": policy.profile,
        "source_model": {
            "name": "Hy3-preview",
            "org": "tencent",
            "architecture": "hy_v3",
        },
        "affine_bits": {
            "routed_expert": dict(policy.routed_bits),
            "attention": policy.attention_bits,
            "shared_expert": policy.shared_expert_bits,
            "dense_ffn": policy.dense_ffn_bits,
            "embed_tokens": policy.embed_bits,
            "lm_head": policy.lm_head_bits,
            "norms_router_biases": 16,
        },
        "quantization": {
            "method": "jang-affine",
            "group_size": policy.group_size,
            "bits_default": 8,
            "mode": "affine",
        },
        "runtime": out_cfg["runtime"],
        "capabilities": out_cfg["capabilities"],
    }
    (out / "jang_config.json").write_text(json.dumps(jang_cfg, indent=2))
    _copy_sidecars(src, out)

    elapsed = time.time() - t0
    print(f"\n  bytes_in:  {bytes_in / 1e9:.2f} GB")
    print(f"  bytes_out: {bytes_out / 1e9:.2f} GB")
    print(f"  shards:    {total_shards}")
    print(f"  on_disk:   {total_size / 1e9:.2f} GB")
    print(f"  elapsed:   {elapsed / 60:.1f} min")
    print(f"  DONE -> {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("profile", nargs="?", default="JANG_2L")
    parser.add_argument("--mtp-policy", choices=["drop", "preserve-affine8"], default="drop")
    parser.add_argument("--shard-bytes", type=int, default=DEFAULT_SHARD_BYTES)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = args.src.expanduser()
    out = args.out.expanduser()
    cfg = json.loads((src / "config.json").read_text())
    policy = profile_policy(
        args.profile,
        mtp_policy=args.mtp_policy,
        num_hidden_layers=int(cfg.get("num_hidden_layers", 80)),
    )
    if args.dry_run:
        _dry_run(src, policy)
        return
    convert(src, out, policy, shard_bytes=args.shard_bytes)


if __name__ == "__main__":
    main()
