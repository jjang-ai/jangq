"""Step 3.7 Flash NVFP4/BF16 source -> JANG/JANGTQ converter.

This module intentionally starts with source inventory and dry-run planning.
The full writer streams ModelOpt NVFP4 routed experts through
``nvfp4_codec.dequant_nvfp4_modelopt`` before JANG quantization; it must never
feed packed uint8 payloads directly into the generic JANG converter.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from dataclasses import dataclass
from math import prod
from pathlib import Path

import numpy as np
import torch
import mlx.core as mx
from safetensors import safe_open
from safetensors.numpy import save_file

from jang_tools.allocate import classify_tensor, JANG_PROFILES, Tier
from jang_tools.capabilities import build_capabilities
from jang_tools.step37.nvfp4_codec import dequant_nvfp4_modelopt
from jang_tools.turboquant.linear import tq_quantize_experts


SIDE_CARS = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "configuration_step3p7.py",
    "modeling_step3p7.py",
    "processing_step3.py",
    "step3p7_mlx.py",
    "vision_encoder.py",
    "hf_quant_config.json",
    "README.md",
]

STEP_PROFILES = {
    "JANG_2L": {
        "format": "jang",
        "routed_expert_bits": {"gate_proj": 4, "up_proj": 2, "down_proj": 3},
    },
    "JANG_K": {
        "format": "jang",
        "routed_expert_bits": {"gate_proj": 4, "up_proj": 2, "down_proj": 2},
    },
    "JANGTQ_2K": {
        "format": "jangtq",
        "routed_expert_bits": {"gate_proj": 4, "up_proj": 2, "down_proj": 2},
    },
    "JANGTQ_K": {
        "format": "jangtq",
        "routed_expert_bits": {"gate_proj": 2, "up_proj": 2, "down_proj": 4},
    },
    "JANGTQ_4K": {
        "format": "jangtq",
        "routed_expert_bits": {"gate_proj": 4, "up_proj": 4, "down_proj": 4},
    },
}

SEED = 42

STEP37_CHAT_SAMPLING_DEFAULTS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
}

STEP37_CHAT_METADATA = {
    "reasoning": {
        "supported": True,
        "parser": "qwen3",
        "default_mode": "no_think",
        "modes": ["no_think", "thinking"],
    },
    "tool_calling": {
        "supported": True,
        "parser": "step3p5",
    },
    "sampling_defaults": STEP37_CHAT_SAMPLING_DEFAULTS,
}


@dataclass(frozen=True)
class TensorPlan:
    name: str
    dtype: str
    shape: tuple[int, ...]
    shard: str
    action: str
    bits: int
    group_size: int
    expanded_shape: tuple[int, ...]
    estimated_bytes: int


def _read_header(path: Path) -> dict:
    with path.open("rb") as f:
        hsize = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(hsize))


def _iter_indexed_tensors(src: Path) -> list[tuple[str, str, tuple[int, ...], Path]]:
    index_path = src / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    by_shard: dict[str, list[str]] = {}
    for name, shard in index.get("weight_map", {}).items():
        by_shard.setdefault(shard, []).append(name)

    out: list[tuple[str, str, tuple[int, ...], Path]] = []
    for shard, names in sorted(by_shard.items()):
        shard_path = src / shard
        header = _read_header(shard_path)
        for name in sorted(names):
            meta = header[name]
            out.append((name, str(meta["dtype"]), tuple(meta["shape"]), shard_path))
    return out


def _load_index(src: Path) -> dict:
    index_path = src / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing {index_path}")
    return json.loads(index_path.read_text(encoding="utf-8"))


def _is_nvfp4_payload(name: str, dtype: str) -> bool:
    return dtype == "U8" and name.endswith(".weight") and ".moe." in name


def _is_nvfp4_sidecar(name: str) -> bool:
    return name.endswith(".input_scale") or name.endswith(".weight_scale") or name.endswith(".weight_scale_2")


def _is_vision_or_projector(name: str) -> bool:
    lower = name.lower()
    return "vision_model" in lower or "vit_large_projector" in lower


def _is_audio(name: str) -> bool:
    lower = name.lower()
    return "audio" in lower or "whisper" in lower or "speech" in lower


def _is_mtp(name: str) -> bool:
    lower = name.lower()
    return "mtp" in lower or "nextn" in lower or "next" in lower


def _component(name: str) -> str:
    lower = name.lower()
    if _is_vision_or_projector(name):
        return "vision_projector"
    if ".moe." in lower:
        return "routed_moe"
    if ".self_attn." in lower:
        return "attention"
    if ".mlp." in lower or "share_expert" in lower:
        return "dense_mlp_shared"
    if "embed_tokens" in lower or "lm_head" in lower:
        return "embed_lm_head"
    return "other_text"


def _profile_spec(profile: str) -> dict:
    if profile not in STEP_PROFILES:
        raise ValueError(f"unsupported Step profile {profile!r}; choices={sorted(STEP_PROFILES)}")
    return STEP_PROFILES[profile]


def _routed_projection(name: str) -> str | None:
    lower = name.lower()
    for proj in ("gate_proj", "up_proj", "down_proj"):
        if f".moe.{proj}." in lower:
            return proj
    return None


def _profile_bits(name: str, profile: str, num_experts: int) -> int:
    proj = _routed_projection(name)
    if proj is not None:
        return int(_profile_spec(profile)["routed_expert_bits"][proj])

    critical, important, compress = JANG_PROFILES["JANG_2L"]
    tier = classify_tensor(name, num_experts=num_experts)
    if tier == Tier.CRITICAL:
        bits = critical
    elif tier == Tier.IMPORTANT:
        bits = important
    else:
        bits = compress

    lower = name.lower()
    if num_experts >= 256 and "shared_expert" not in lower:
        if "gate_proj" in lower:
            bits = max(bits, 4)
        elif "down_proj" in lower:
            bits = max(bits, 3)
    return int(bits)


def _default_group_size(num_experts: int) -> int:
    return 128 if num_experts >= 150 else 64


def _compatible_group_size(in_dim: int, requested: int) -> int:
    for gs in (128, 64, 32):
        if gs <= requested and in_dim % gs == 0:
            return gs
    raise ValueError(f"in_dim={in_dim} is not compatible with MLX group sizes 32/64/128")


def _tensor_group_size(name: str, shape: tuple[int, ...], num_experts: int) -> int:
    if len(shape) < 2:
        return 0
    lower = name.lower()
    requested = _default_group_size(num_experts)
    if ".gate." in lower or lower.endswith(".gate") or "shared_expert_gate" in lower:
        requested = 64
    return _compatible_group_size(int(shape[-1]), requested)


def _expanded_shape(name: str, dtype: str, shape: tuple[int, ...]) -> tuple[int, ...]:
    if _is_nvfp4_payload(name, dtype):
        return tuple(shape[:-1]) + (int(shape[-1]) * 2,)
    return shape


def _estimate_quantized_bytes(shape: tuple[int, ...], bits: int, group_size: int) -> int:
    rows = prod(shape[:-1])
    in_dim = int(shape[-1])
    qweight_bytes = rows * ((in_dim * bits) // 8)
    scale_bias_bytes = rows * (in_dim // group_size) * 4
    return int(qweight_bytes + scale_bias_bytes)


def _estimate_tq_bytes(shape: tuple[int, ...], bits: int) -> int:
    rows = prod(shape[:-1])
    in_dim = int(shape[-1])
    packed_cols = (in_dim + (32 // bits) - 1) // (32 // bits)
    packed_bytes = rows * packed_cols * 4
    norm_bytes = rows * 2
    bit_scalar_bytes = 1
    return int(packed_bytes + norm_bytes + bit_scalar_bytes)


def _dtype_bytes(dtype: str) -> int:
    if dtype in {"BF16", "F16", "U16"}:
        return 2
    if dtype in {"F32", "U32", "I32"}:
        return 4
    if dtype in {"F8_E4M3", "U8", "I8", "BOOL"}:
        return 1
    return 0


def _plan_tensor(
    name: str,
    dtype: str,
    shape: tuple[int, ...],
    shard: str,
    action: str,
    bits: int,
    num_experts: int,
) -> TensorPlan:
    expanded = _expanded_shape(name, dtype, shape)
    group_size = _tensor_group_size(name, expanded, num_experts) if action in {"bf16-jang", "nvfp4-dequant-then-jang"} else 0
    if action in {"bf16-jang", "nvfp4-dequant-then-jang"}:
        estimated = _estimate_quantized_bytes(expanded, bits, group_size)
    elif action == "nvfp4-dequant-then-tq":
        estimated = _estimate_tq_bytes(expanded, bits)
    elif action == "nvfp4-sidecar-skip":
        estimated = 0
    else:
        estimated = prod(expanded) * _dtype_bytes(dtype)
    return TensorPlan(name, dtype, shape, shard, action, bits, group_size, expanded, int(estimated))


def build_plan(src: Path, profile: str = "JANG_2L") -> tuple[list[TensorPlan], dict]:
    spec = _profile_spec(profile)
    config = json.loads((src / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config", {})
    num_experts = int(text_config.get("moe_num_experts", 0) or 0)
    plans: list[TensorPlan] = []
    stats = {
        "nvfp4_payloads": 0,
        "nvfp4_sidecars": 0,
        "bf16_passthrough": 0,
        "bf16_quantized": 0,
        "vision_tensors": 0,
        "audio_tensors": 0,
        "mtp_tensors": 0,
    }

    for name, dtype, shape, shard_path in _iter_indexed_tensors(src):
        if _is_audio(name):
            stats["audio_tensors"] += 1
        if _is_mtp(name):
            stats["mtp_tensors"] += 1
        if _is_vision_or_projector(name):
            stats["vision_tensors"] += 1

        if _is_nvfp4_sidecar(name):
            stats["nvfp4_sidecars"] += 1
            plans.append(_plan_tensor(name, dtype, shape, shard_path.name, "nvfp4-sidecar-skip", 0, num_experts))
            continue

        if _is_nvfp4_payload(name, dtype):
            stats["nvfp4_payloads"] += 1
            bits = _profile_bits(name, profile, num_experts)
            action = "nvfp4-dequant-then-tq" if spec["format"] == "jangtq" else "nvfp4-dequant-then-jang"
            plans.append(_plan_tensor(name, dtype, shape, shard_path.name, action, bits, num_experts))
            continue

        if dtype in {"BF16", "F32"} and (len(shape) < 2 or _is_vision_or_projector(name) or not name.endswith(".weight")):
            stats["bf16_passthrough"] += 1
            plans.append(_plan_tensor(name, dtype, shape, shard_path.name, "passthrough", 16 if dtype == "BF16" else 32, num_experts))
            continue

        if dtype == "BF16" and name.endswith(".weight"):
            stats["bf16_quantized"] += 1
            bits = _profile_bits(name, profile, num_experts)
            plans.append(_plan_tensor(name, dtype, shape, shard_path.name, "bf16-jang", bits, num_experts))
            continue

        plans.append(_plan_tensor(name, dtype, shape, shard_path.name, "passthrough", 16 if dtype == "BF16" else 32, num_experts))

    return plans, stats


class ShardWriter:
    def __init__(self, out: Path, max_shard_bytes: int) -> None:
        self.out = out
        self.max_shard_bytes = max_shard_bytes
        self.shard_idx = 0
        self.tensors: dict[str, np.ndarray] = {}
        self.shard_bytes = 0
        self.weight_map: dict[str, str] = {}

    def add(self, name: str, arr: np.ndarray) -> None:
        arr = np.ascontiguousarray(arr)
        if self.tensors and self.shard_bytes + arr.nbytes > self.max_shard_bytes:
            self.flush()
        self.tensors[name] = arr
        self.shard_bytes += arr.nbytes
        if self.shard_bytes >= self.max_shard_bytes:
            self.flush()

    def flush(self) -> None:
        if not self.tensors:
            return
        self.shard_idx += 1
        shard_name = f"model-{self.shard_idx:05d}-of-XXXXX.safetensors"
        save_file(self.tensors, str(self.out / shard_name), metadata={"format": "mlx"})
        for key in self.tensors:
            self.weight_map[key] = shard_name
        print(
            f"  wrote {shard_name}: {len(self.tensors)} tensors, {self.shard_bytes / (1024 ** 3):.3f} GiB",
            flush=True,
        )
        self.tensors = {}
        self.shard_bytes = 0

    def finalize(self) -> None:
        self.flush()
        if self.shard_idx == 0:
            return
        final_map: dict[str, str] = {}
        for old_name in sorted({v for v in self.weight_map.values()}):
            idx = int(old_name.split("-")[1])
            final_name = f"model-{idx:05d}-of-{self.shard_idx:05d}.safetensors"
            (self.out / old_name).rename(self.out / final_name)
        for key, old_name in self.weight_map.items():
            idx = int(old_name.split("-")[1])
            final_map[key] = f"model-{idx:05d}-of-{self.shard_idx:05d}.safetensors"
        self.weight_map = final_map
        total_size = sum((self.out / name).stat().st_size for name in sorted(set(final_map.values())))
        index = {"metadata": {"format": "mlx", "total_size": total_size}, "weight_map": final_map}
        (self.out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")


def _torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float16)
    elif tensor.dtype == torch.float8_e4m3fn:
        tensor = tensor.float()
    return tensor.detach().cpu().numpy()


def _quantize_array(arr: np.ndarray, *, bits: int, group_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if arr.ndim not in (2, 3):
        raise ValueError(f"expected 2D or 3D quantized array, got shape {arr.shape}")
    original_shape = arr.shape
    flat = arr.reshape(-1, arr.shape[-1]) if arr.ndim == 3 else arr
    chunk_rows = max(1, 100_000_000 // int(flat.shape[-1]))
    q_parts: list[np.ndarray] = []
    s_parts: list[np.ndarray] = []
    b_parts: list[np.ndarray] = []
    for start in range(0, flat.shape[0], chunk_rows):
        end = min(start + chunk_rows, flat.shape[0])
        chunk = mx.array(flat[start:end].astype(np.float16, copy=False))
        qw, scales, biases = mx.quantize(chunk, group_size=group_size, bits=bits)
        q_parts.append(np.array(qw))
        s_parts.append(np.array(scales).astype(np.float16))
        b_parts.append(np.array(biases).astype(np.float16))
        mx.synchronize()
    q = np.concatenate(q_parts, axis=0)
    s = np.concatenate(s_parts, axis=0)
    b = np.concatenate(b_parts, axis=0)
    if len(original_shape) == 3:
        q = q.reshape(original_shape[0], original_shape[1], -1)
        s = s.reshape(original_shape[0], original_shape[1], -1)
        b = b.reshape(original_shape[0], original_shape[1], -1)
    return q, s, b


def _tq_base_name(source_weight_name: str) -> str:
    base = source_weight_name[:-len(".weight")]
    if base.startswith("model.language_model."):
        base = "model." + base[len("model.language_model."):]
    base = base.replace(".moe.gate_proj", ".mlp.switch_mlp.gate_proj")
    base = base.replace(".moe.up_proj", ".mlp.switch_mlp.up_proj")
    base = base.replace(".moe.down_proj", ".mlp.switch_mlp.down_proj")
    return base


def _quantize_tq_expert_chunk(arr: np.ndarray, *, bits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = tq_quantize_experts(arr.astype(np.float16, copy=False), bits=bits, seed=SEED)
    packed = np.asarray(result["packed"])
    norms = np.asarray(result["norms"]).astype(np.float16, copy=False)
    bit_tensor = np.array([bits], dtype=np.uint8)
    return packed, norms, bit_tensor


def _copy_sidecars(src: Path, out: Path) -> None:
    for name in SIDE_CARS:
        source = src / name
        if source.exists():
            shutil.copy2(source, out / name)
    bridge = Path(__file__).with_name("step3p7_mlx.py")
    if bridge.exists():
        shutil.copy2(bridge, out / "step3p7_mlx.py")


def _write_configs(src: Path, out: Path, profile: str, plans: list[TensorPlan], writer: ShardWriter) -> None:
    spec = _profile_spec(profile)
    fmt = str(spec["format"])
    routed_bits = dict(spec["routed_expert_bits"])
    config = json.loads((src / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config", {})
    group_sizes = sorted({p.group_size for p in plans if p.group_size})
    default_group_size = 128 if 128 in group_sizes else (group_sizes[-1] if group_sizes else 64)
    quantization: dict[str, object] = {"group_size": default_group_size, "bits": 2, "format": fmt}
    for p in plans:
        if p.action not in {"bf16-jang", "nvfp4-dequant-then-jang", "nvfp4-dequant-then-tq"} or not p.name.endswith(".weight"):
            continue
        base = p.name[:-len(".weight")]
        module = base
        if module.startswith("model.language_model."):
            module = "model." + module[len("model.language_model."):]
        module = module.replace(".moe.gate_proj", ".mlp.switch_mlp.gate_proj")
        module = module.replace(".moe.up_proj", ".mlp.switch_mlp.up_proj")
        module = module.replace(".moe.down_proj", ".mlp.switch_mlp.down_proj")
        module = module.replace(".moe.gate", ".mlp.gate.gate")
        module = module.replace(".share_expert.", ".mlp.share_expert.")
        if module.startswith("model.vision_model") or module.startswith("model.vit_large_projector"):
            continue
        entry = {"bits": p.bits}
        if p.group_size:
            entry["group_size"] = p.group_size
        quantization[module] = entry
    config["model_file"] = "step3p7_mlx.py"
    config["quantization"] = quantization
    config["mxtq_seed"] = SEED
    config["mxtq_bits"] = {
        "attention": 8,
        "embedding": 6,
        "routed_expert": routed_bits,
    }
    config["routed_expert_bits"] = routed_bits
    config["use_cache"] = True
    if isinstance(text_config, dict):
        text_config["use_cache"] = True
    (out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    tokenizer_config_path = out / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        tokenizer_config["backend"] = "tokenizers"
        tokenizer_config["tokenizer_class"] = "PreTrainedTokenizerFast"
        tokenizer_config_path.write_text(json.dumps(tokenizer_config, indent=2, sort_keys=True), encoding="utf-8")

    quantized = [p for p in plans if p.action in {"bf16-jang", "nvfp4-dequant-then-jang", "nvfp4-dequant-then-tq"}]
    total_source_bits = sum(prod(p.expanded_shape) * 16 for p in quantized)
    total_quant_bits = sum(p.estimated_bytes * 8 for p in quantized)
    actual_bits = round(total_quant_bits / max(1, sum(prod(p.expanded_shape) for p in quantized)), 3)
    bits_used = sorted({p.bits for p in quantized})
    passthrough_bits = sorted({p.bits for p in plans if p.action == "passthrough"})
    jang = {
        "format": fmt,
        "format_version": "2.0",
        "mxtq_seed": SEED,
        "routed_expert_bits": routed_bits,
        "source_model": {
            "name": "Step-3.7-Flash-NVFP4",
            "hub_id": "stepfun-ai/Step-3.7-Flash-NVFP4",
            "dtype": "nvfp4+bf16",
        },
        "architecture": {
            "type": "step3p7",
            "text_model_type": text_config.get("model_type", "step3p5") if isinstance(text_config, dict) else "step3p5",
            "has_vision": True,
            "has_audio": False,
            "has_mtp_tensors": False,
        },
        "quantization": {
            "method": "jang-importance",
            "profile": profile,
            "target_bits": 2.0,
            "actual_bits": actual_bits,
            "block_size": default_group_size,
            "quantization_backend": "mx.quantize+tq_quantize_experts" if fmt == "jangtq" else "mx.quantize",
            "source_weight_decode": "modelopt-nvfp4-for-routed-moe",
            "bit_widths_used": bits_used,
            "passthrough_bit_widths_used": passthrough_bits,
            "total_source_bits": total_source_bits,
            "total_quantized_bits": total_quant_bits,
            "routed_expert_bits": routed_bits,
            "mxtq_bits": {
                "attention": 8,
                "embedding": 6,
                "routed_expert": routed_bits,
            },
        },
        "runtime": {
            "total_shard_bytes": sum((out / name).stat().st_size for name in sorted(set(writer.weight_map.values()))),
            "shard_count": writer.shard_idx,
            "requires": [
                "step3p7-vlm-wrapper",
                "step3p5-text-runtime",
                "full-and-sliding-kv-cache",
                "head-wise-attention-gate",
                "qk-rmsnorm",
                "kv-scale-sidecars",
                "image-patch-processing",
            ],
        },
        "mxtq_bits": {
            "attention": 8,
            "embedding": 6,
            "routed_expert": routed_bits,
        },
        "chat": STEP37_CHAT_METADATA,
    }
    caps = build_capabilities({"source_model": {"architecture": "step3p7"}, "has_vision": True}, config, out)
    if caps is not None:
        jang["capabilities"] = caps
    (out / "jang_config.json").write_text(json.dumps(jang, indent=2, sort_keys=True), encoding="utf-8")


def convert(src: Path, out: Path, *, profile: str, max_shard_gb: float, force: bool) -> None:
    plans, stats = build_plan(src, profile)
    if out.exists():
        existing = list(out.glob("model*.safetensors")) + list(out.glob("model.safetensors.index.json"))
        if existing and not force:
            raise SystemExit(f"{out} already contains model files; pass --force to replace them")
        if force:
            for p in out.glob("model*.safetensors"):
                p.unlink()
            index = out / "model.safetensors.index.json"
            if index.exists():
                index.unlink()
    out.mkdir(parents=True, exist_ok=True)
    _copy_sidecars(src, out)

    by_name = {p.name: p for p in plans}
    index = _load_index(src)
    weight_map: dict[str, str] = index.get("weight_map", {})
    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(name)

    writer = ShardWriter(out, int(max_shard_gb * 1024 ** 3))
    total = sum(1 for p in plans if p.action != "nvfp4-sidecar-skip")
    done = 0
    print(json.dumps({"source": str(src), "output": str(out), "profile": profile, "stats": stats}, indent=2), flush=True)
    for shard, names in sorted(by_shard.items()):
        shard_path = src / shard
        print(f"reading {shard}", flush=True)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            current_keys = set(f.keys())

            def load_tensor(tensor_name: str) -> torch.Tensor:
                if tensor_name in current_keys:
                    return f.get_tensor(tensor_name)
                side_shard = weight_map.get(tensor_name)
                if side_shard is None:
                    raise KeyError(f"{tensor_name} is not present in model.safetensors.index.json")
                with safe_open(src / side_shard, framework="pt", device="cpu") as side_f:
                    return side_f.get_tensor(tensor_name)

            for name in sorted(names):
                plan = by_name[name]
                if plan.action == "nvfp4-sidecar-skip":
                    continue
                if plan.action == "nvfp4-dequant-then-jang":
                    base = name[:-len(".weight")]
                    weight_u8 = load_tensor(name)
                    weight_scale = load_tensor(f"{base}.weight_scale")
                    weight_scale_2 = load_tensor(f"{base}.weight_scale_2")
                    q_parts: list[np.ndarray] = []
                    s_parts: list[np.ndarray] = []
                    b_parts: list[np.ndarray] = []
                    for start in range(0, int(weight_u8.shape[0]), 8):
                        end = min(start + 8, int(weight_u8.shape[0]))
                        decoded = dequant_nvfp4_modelopt(
                            weight_u8[start:end],
                            weight_scale[start:end],
                            weight_scale_2[start:end],
                            out_dtype=torch.float16,
                        ).cpu().numpy()
                        qw, scales, biases = _quantize_array(decoded, bits=plan.bits, group_size=plan.group_size)
                        q_parts.append(qw)
                        s_parts.append(scales)
                        b_parts.append(biases)
                        del decoded
                    writer.add(name, np.concatenate(q_parts, axis=0))
                    writer.add(f"{base}.scales", np.concatenate(s_parts, axis=0))
                    writer.add(f"{base}.biases", np.concatenate(b_parts, axis=0))
                elif plan.action == "nvfp4-dequant-then-tq":
                    tq_base = _tq_base_name(name)
                    weight_u8 = load_tensor(name)
                    weight_scale = load_tensor(f"{name[:-len('.weight')]}.weight_scale")
                    weight_scale_2 = load_tensor(f"{name[:-len('.weight')]}.weight_scale_2")
                    p_parts: list[np.ndarray] = []
                    n_parts: list[np.ndarray] = []
                    bits_arr: np.ndarray | None = None
                    for start in range(0, int(weight_u8.shape[0]), 8):
                        end = min(start + 8, int(weight_u8.shape[0]))
                        decoded = dequant_nvfp4_modelopt(
                            weight_u8[start:end],
                            weight_scale[start:end],
                            weight_scale_2[start:end],
                            out_dtype=torch.float16,
                        ).cpu().numpy()
                        packed, norms, bit_tensor = _quantize_tq_expert_chunk(decoded, bits=plan.bits)
                        p_parts.append(packed)
                        n_parts.append(norms)
                        bits_arr = bit_tensor
                        del decoded
                    writer.add(f"{tq_base}.tq_packed", np.concatenate(p_parts, axis=0))
                    writer.add(f"{tq_base}.tq_norms", np.concatenate(n_parts, axis=0))
                    writer.add(f"{tq_base}.tq_bits", bits_arr if bits_arr is not None else np.array([plan.bits], dtype=np.uint8))
                elif plan.action == "bf16-jang":
                    arr = _torch_to_numpy(load_tensor(name))
                    qw, scales, biases = _quantize_array(arr, bits=plan.bits, group_size=plan.group_size)
                    base = name[:-len(".weight")]
                    writer.add(name, qw)
                    writer.add(f"{base}.scales", scales)
                    writer.add(f"{base}.biases", biases)
                else:
                    arr = _torch_to_numpy(load_tensor(name))
                    writer.add(name, arr)
                done += 1
                if done % 25 == 0 or done == total:
                    print(f"  progress {done}/{total}: {name}", flush=True)
    writer.finalize()
    _write_configs(src, out, profile, plans, writer)
    if _profile_spec(profile)["format"] == "jangtq":
        from jang_tools.build_jangtq_sidecar import main as build_jangtq_sidecar

        import sys

        old_argv = sys.argv[:]
        try:
            sys.argv = ["build_jangtq_sidecar", str(out)]
            build_jangtq_sidecar()
        finally:
            sys.argv = old_argv
        if not (out / "jangtq_runtime.safetensors").is_file():
            raise RuntimeError("missing jangtq_runtime.safetensors after sidecar build")
    print(f"done: {out} ({writer.shard_idx} shards)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3.7 Flash -> JANG_2L converter")
    parser.add_argument("src", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--profile", default="JANG_2L", choices=sorted(STEP_PROFILES))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-shard-gb", type=float, default=1.5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    src = args.src.expanduser()
    plans, stats = build_plan(src, args.profile)
    print(json.dumps({"source": str(src), "output": str(args.out), "profile": args.profile, "stats": stats}, indent=2))
    if args.dry_run:
        by_action: dict[str, int] = {}
        by_bits: dict[str, int] = {}
        by_group_size: dict[str, int] = {}
        by_component_gib: dict[str, float] = {}
        estimated_bytes = 0
        for p in plans:
            by_action[p.action] = by_action.get(p.action, 0) + 1
            by_bits[str(p.bits)] = by_bits.get(str(p.bits), 0) + 1
            if p.group_size:
                by_group_size[str(p.group_size)] = by_group_size.get(str(p.group_size), 0) + 1
            estimated_bytes += p.estimated_bytes
            comp = _component(p.name)
            by_component_gib[comp] = by_component_gib.get(comp, 0.0) + p.estimated_bytes / (1024 ** 3)
        by_component_gib = {
            k: round(v, 3)
            for k, v in sorted(by_component_gib.items(), key=lambda item: item[1], reverse=True)
        }
        print(json.dumps({
            "by_action": by_action,
            "by_bits": by_bits,
            "by_group_size": by_group_size,
            "by_component_gib": by_component_gib,
            "estimated_output_gib": round(estimated_bytes / (1024 ** 3), 3),
            "sample": [p.__dict__ for p in plans[:20]],
        }, indent=2))
        return

    convert(src, args.out.expanduser(), profile=args.profile, max_shard_gb=args.max_shard_gb, force=args.force)


if __name__ == "__main__":
    main()
