"""Build proper discrete-text JANG-affine Bonsai-27B bundles.

The two supported source mirrors are already dequantized BF16 tensors:

- Bonsai binary groups contain exactly two values per 128-weight group.
- Ternary Bonsai groups contain exactly ``{-s, 0, +s}`` per group.

This converter restores those tensors to proper groupwise JANG affine storage
without JANGTQ/MXTQ/codebooks or a runtime sidecar. Text matrices, embeddings,
and the untied language-model head use 1-bit or 2-bit affine storage. Eligible
vision Linear layers use native 4-bit affine storage; state tensors, norms,
the grouped GatedDeltaNet convolution, and unsupported vision projections stay
float16. Binary/ternary levels must be exact before standard float16 affine
sidecar rounding, whose residual is measured and reported.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open
from tqdm import tqdm

from .affine import quantize_discrete_affine, quantize_native_affine_numpy
from .format.writer import write_jang_v2_model


PROFILE_BY_BITS = {
    1: "JANG_AFFINE_1BIT",
    2: "JANG_AFFINE_TERNARY_2BIT",
}
TEXT_GROUP_SIZE = 128
VISION_BITS = 4
VISION_GROUP_SIZE = 64
EOS_FROM = 248044
EOS_TO = 248046


@dataclass(frozen=True)
class TensorPolicy:
    method: str
    bits: int
    group_size: int
    reason: str


@dataclass
class SizePlan:
    tensor_count: int = 0
    quantized_tensor_count: int = 0
    passthrough_tensor_count: int = 0
    text_quantized_parameters: int = 0
    vision_quantized_parameters: int = 0
    passthrough_parameters: int = 0
    packed_bytes: int = 0
    affine_metadata_bytes: int = 0
    passthrough_bytes: int = 0

    @property
    def projected_bytes(self) -> int:
        return self.packed_bytes + self.affine_metadata_bytes + self.passthrough_bytes

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["projected_bytes"] = self.projected_bytes
        result["projected_gib"] = round(self.projected_bytes / (1024**3), 3)
        result["projected_gb"] = round(self.projected_bytes / 1_000_000_000, 3)
        return result


def sanitize_key(name: str) -> str:
    """Map raw Qwen3.5 VLM keys to the vMLX/MLX-VLM module tree."""
    if name.startswith("model.language_model."):
        return name.replace("model.language_model", "language_model.model", 1)
    if name.startswith("model.visual."):
        return name.replace("model.visual", "vision_tower", 1)
    if name == "lm_head.weight" or name.startswith("lm_head."):
        return f"language_model.{name}"
    return name


def _is_text_matrix(name: str, shape: tuple[int, ...]) -> bool:
    return (
        name.endswith(".weight")
        and len(shape) == 2
        and (
            name.startswith("model.language_model.")
            or name.startswith("language_model.model.")
            or name == "lm_head.weight"
            or name.startswith("language_model.lm_head.")
        )
    )


def _is_vision_linear(name: str, shape: tuple[int, ...]) -> bool:
    if not (
        name.endswith(".weight")
        and len(shape) == 2
        and (name.startswith("model.visual.") or name.startswith("vision_tower."))
    ):
        return False
    return any(
        marker in name
        for marker in (
            ".attn.qkv.weight",
            ".attn.proj.weight",
            ".mlp.linear_fc1.weight",
            ".mlp.linear_fc2.weight",
            ".merger.linear_fc1.weight",
            ".merger.linear_fc2.weight",
        )
    )


def classify_tensor(
    name: str,
    shape: tuple[int, ...],
    *,
    text_bits: int,
) -> TensorPolicy:
    """Return the complete Bonsai storage policy for one source tensor."""
    if _is_text_matrix(name, shape):
        input_dim = int(shape[-1])
        if input_dim % TEXT_GROUP_SIZE:
            raise ValueError(
                f"text matrix {name} has input_dim={input_dim}, not divisible by "
                f"the required group_size={TEXT_GROUP_SIZE}"
            )
        return TensorPolicy(
            "discrete_affine",
            text_bits,
            TEXT_GROUP_SIZE,
            "binary/ternary text matrix, embedding, or untied lm_head",
        )

    if _is_vision_linear(name, shape):
        input_dim = int(shape[-1])
        for group_size in (VISION_GROUP_SIZE, 32):
            if input_dim % group_size == 0:
                return TensorPolicy(
                    "mlx_affine",
                    VISION_BITS,
                    group_size,
                    "eligible vision Linear",
                )
        return TensorPolicy(
            "passthrough",
            16,
            0,
            "vision Linear input dimension is incompatible with MLX group sizes",
        )

    return TensorPolicy(
        "passthrough",
        16,
        0,
        "norm/state/bias/convolution/position embedding passthrough",
    )


def _read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    return 8 + header_size, header


def _load_tensor(
    shard_path: Path,
    handle,
    header: dict[str, Any],
    data_start: int,
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    dtype = str(handle.get_slice(name).get_dtype()).upper()
    if dtype != "BF16":
        return np.asarray(handle.get_tensor(name))

    info = header[name]
    start, end = (int(value) for value in info["data_offsets"])
    raw = np.memmap(
        shard_path,
        dtype=np.uint16,
        mode="r",
        offset=data_start + start,
        shape=((end - start) // 2,),
    )
    float_bits = raw.astype(np.uint32) << np.uint32(16)
    return float_bits.view(np.float32).reshape(shape)


def _prepare_passthrough(name: str, tensor: np.ndarray) -> np.ndarray:
    output_name = sanitize_key(name)
    result = tensor
    if (
        output_name.endswith("vision_tower.patch_embed.proj.weight")
        and result.ndim == 5
        and result.shape[1] in (1, 3)
    ):
        result = np.ascontiguousarray(np.transpose(result, (0, 2, 3, 4, 1)))
    return np.asarray(result, dtype=np.float16)


def _quantize_vision_native(
    tensor: np.ndarray,
    *,
    bits: int,
    group_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    return quantize_native_affine_numpy(
        tensor,
        bits=bits,
        group_size=group_size,
        chunk_rows=256,
    )


def _source_index(model_path: Path) -> dict[str, Any]:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing source index: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(index.get("weight_map"), dict):
        raise ValueError(f"invalid source weight_map: {index_path}")
    return index


def _source_tensor_metadata(model_path: Path) -> dict[str, tuple[tuple[int, ...], str]]:
    index = _source_index(model_path)
    weight_map = index["weight_map"]
    result: dict[str, tuple[tuple[int, ...], str]] = {}
    for shard_name in sorted(set(weight_map.values())):
        shard_path = model_path / shard_name
        with safe_open(str(shard_path), framework="numpy") as handle:
            for name in handle.keys():
                tensor_slice = handle.get_slice(name)
                result[name] = (
                    tuple(int(value) for value in tensor_slice.get_shape()),
                    str(tensor_slice.get_dtype()).upper(),
                )
    if set(result) != set(weight_map):
        missing = sorted(set(weight_map) - set(result))
        extra = sorted(set(result) - set(weight_map))
        raise ValueError(
            f"source index/header mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    return result


def plan_conversion(model_path: str | Path, *, text_bits: int) -> SizePlan:
    model_path = Path(model_path)
    if text_bits not in PROFILE_BY_BITS:
        raise ValueError(f"text_bits must be 1 or 2, got {text_bits}")

    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    _validate_bonsai_source(config, _source_index(model_path))
    metadata = _source_tensor_metadata(model_path)
    plan = SizePlan(tensor_count=len(metadata))

    for name, (shape, _dtype) in metadata.items():
        elements = int(np.prod(shape, dtype=np.int64))
        policy = classify_tensor(name, shape, text_bits=text_bits)
        if policy.method in {"discrete_affine", "mlx_affine"}:
            rows, columns = int(shape[0]), int(shape[1])
            plan.quantized_tensor_count += 1
            plan.packed_bytes += elements * policy.bits // 8
            plan.affine_metadata_bytes += rows * (columns // policy.group_size) * 4
            if policy.method == "discrete_affine":
                plan.text_quantized_parameters += elements
            else:
                plan.vision_quantized_parameters += elements
        else:
            plan.passthrough_tensor_count += 1
            plan.passthrough_parameters += elements
            plan.passthrough_bytes += elements * 2
    return plan


def _validate_bonsai_source(config: dict[str, Any], index: dict[str, Any]) -> None:
    if config.get("model_type") != "qwen3_5":
        raise ValueError(f"expected model_type=qwen3_5, got {config.get('model_type')!r}")
    text_config = config.get("text_config") or {}
    if text_config.get("model_type") != "qwen3_5_text":
        raise ValueError(
            f"expected text_config.model_type=qwen3_5_text, "
            f"got {text_config.get('model_type')!r}"
        )
    if int(text_config.get("num_hidden_layers", 0)) != 64:
        raise ValueError("Bonsai converter expects the dense 64-layer 27B checkpoint")
    if config.get("tie_word_embeddings") is not False:
        raise ValueError("Bonsai must keep its untied lm_head")
    keys = set(index["weight_map"])
    if "lm_head.weight" not in keys:
        raise ValueError("source is missing the required bare untied lm_head.weight")
    if not any(name.startswith("model.visual.") for name in keys):
        raise ValueError("source config declares vision but no model.visual tensors exist")
    routed = [
        name
        for name in keys
        if ".experts." in name or ".router." in name or ".shared_expert." in name
    ]
    if routed:
        raise ValueError(
            "Bonsai-27B is expected to be dense; routed expert tensors were found: "
            f"{routed[:5]}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _patch_eos_mapping(value: Any) -> Any:
    if isinstance(value, int) and value == EOS_FROM:
        return EOS_TO
    if isinstance(value, list):
        patched = [EOS_TO if item == EOS_FROM else item for item in value]
        return list(dict.fromkeys(patched))
    return value


def _patch_model_config_tokens(config: dict[str, Any]) -> dict[str, Any]:
    output = json.loads(json.dumps(config))
    _drop_unbacked_mtp_metadata(output)
    text_config = output.setdefault("text_config", {})
    text_config["eos_token_id"] = _patch_eos_mapping(
        text_config.get("eos_token_id", EOS_FROM)
    )
    output["eos_token_id"] = _patch_eos_mapping(
        output.get("eos_token_id")
        if output.get("eos_token_id") is not None
        else text_config["eos_token_id"]
    )
    for key in ("bos_token_id", "pad_token_id"):
        if output.get(key) is None and text_config.get(key) is not None:
            output[key] = text_config[key]
    output.pop("quantization_config", None)
    output.pop("quantization", None)
    return output


def _drop_unbacked_mtp_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Remove source MTP claims when the verified Bonsai index has no MTP."""
    for owner in (config, config.get("text_config")):
        if not isinstance(owner, dict):
            continue
        for key in list(owner):
            if key.startswith("mtp_") or key in {
                "num_nextn_predict_layers",
                "num_next_n_predict_layers",
            }:
                owner.pop(key, None)
    return config


def _load_tokenizer_files(model_path: Path) -> dict[str, str | bytes | dict]:
    result: dict[str, str | bytes | dict] = {}
    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "merges.txt",
        "vocab.json",
        "added_tokens.json",
    ):
        path = model_path / filename
        if not path.is_file():
            continue
        if filename.endswith(".json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and "eos_token_id" in value:
                value["eos_token_id"] = _patch_eos_mapping(value["eos_token_id"])
            result[filename] = value
        elif filename == "tokenizer.model":
            result[filename] = path.read_bytes()
        else:
            result[filename] = path.read_text(
                encoding="utf-8", errors="surrogateescape"
            )
    return result


def _copy_extra_files(model_path: Path, output_path: Path) -> list[str]:
    copied = []
    for filename in (
        "preprocessor_config.json",
        "processor_config.json",
        "video_preprocessor_config.json",
        "chat_template.json",
        "chat_template.jinja",
        "generation_config.json",
        "LICENSE",
        "LICENSE.txt",
        "NOTICE.txt",
    ):
        source = model_path / filename
        if not source.is_file():
            continue
        destination = output_path / filename
        if filename == "generation_config.json":
            value = json.loads(source.read_text(encoding="utf-8"))
            if "eos_token_id" in value:
                value["eos_token_id"] = _patch_eos_mapping(value["eos_token_id"])
            destination.write_text(
                json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        else:
            shutil.copy2(source, destination)
        copied.append(filename)
    for source in model_path.glob("*.py"):
        shutil.copy2(source, output_path / source.name)
        copied.append(source.name)
    return copied


def _module_base(name: str) -> str:
    output_name = sanitize_key(name)
    return output_name[:-7] if output_name.endswith(".weight") else output_name


def _verify_bundle(
    output_path: Path,
    *,
    manifest: dict[str, dict[str, Any]],
    text_bits: int,
) -> dict[str, Any]:
    index = json.loads(
        (output_path / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map = index["weight_map"]
    missing = []
    for module_path in manifest:
        for suffix in ("weight", "scales", "biases"):
            key = f"{module_path}.{suffix}"
            if key not in weight_map:
                missing.append(key)
    if missing:
        raise RuntimeError(f"output is missing affine tensors: {missing[:10]}")
    if any(".tq_" in name for name in weight_map):
        raise RuntimeError("JANG-affine output unexpectedly contains TurboQuant tensors")
    if (output_path / "jangtq_runtime.safetensors").exists():
        raise RuntimeError("JANG-affine output unexpectedly contains a JANGTQ sidecar")
    if "lm_head.weight" in weight_map:
        raise RuntimeError("VLM output must not contain a bare lm_head.weight")
    if "language_model.lm_head.weight" not in weight_map:
        raise RuntimeError("VLM output is missing language_model.lm_head.weight")
    config = json.loads((output_path / "config.json").read_text(encoding="utf-8"))
    if config.get("eos_token_id") != EOS_TO:
        raise RuntimeError(f"top-level eos_token_id is not {EOS_TO}")
    if (config.get("text_config") or {}).get("eos_token_id") != EOS_TO:
        raise RuntimeError(f"text_config.eos_token_id is not {EOS_TO}")
    onebit_count = sum(
        1
        for spec in manifest.values()
        if int(spec.get("storage_bits", spec["bits"])) == 1
    )
    if text_bits == 1 and onebit_count == 0:
        raise RuntimeError("1-bit profile has no 1-bit affine storage modules")
    shard_bytes = sum(
        (output_path / filename).stat().st_size
        for filename in set(weight_map.values())
    )
    return {
        "status": "ok",
        "indexed_tensor_count": len(weight_map),
        "quantized_module_count": len(manifest),
        "affine1_module_count": onebit_count,
        "indexed_shard_bytes": shard_bytes,
        "indexed_shard_gib": round(shard_bytes / (1024**3), 3),
    }


def convert_bonsai_jang_affine(
    model_path: str | Path,
    output_path: str | Path,
    *,
    text_bits: int,
    chunk_rows: int = 256,
) -> dict[str, Any]:
    model_path = Path(model_path)
    output_path = Path(output_path)
    if text_bits not in PROFILE_BY_BITS:
        raise ValueError(f"text_bits must be 1 or 2, got {text_bits}")

    source_index = _source_index(model_path)
    source_config = json.loads(
        (model_path / "config.json").read_text(encoding="utf-8")
    )
    _validate_bonsai_source(source_config, source_index)
    plan = plan_conversion(model_path, text_bits=text_bits)
    output_path.mkdir(parents=True, exist_ok=True)

    stale_patterns = (
        "model-*-of-*.safetensors",
        "model.safetensors.index.json",
        "jang_config.json",
        "jang_affine_report.json",
    )
    for pattern in stale_patterns:
        for stale in output_path.glob(pattern):
            stale.unlink()

    weight_map = source_index["weight_map"]
    tensors: dict[str, np.ndarray] = {}
    manifest: dict[str, dict[str, Any]] = {}
    vision_rel_l1: dict[str, float] = {}
    text_rel_l1: dict[str, float] = {}
    text_max_abs_error: dict[str, float] = {}
    method_counts: dict[str, int] = {}

    shard_names = sorted(set(weight_map.values()))
    total_tensors = len(weight_map)
    progress = tqdm(total=total_tensors, desc="  Bonsai JANG affine")
    for shard_name in shard_names:
        shard_path = model_path / shard_name
        data_start, header = _read_safetensors_header(shard_path)
        with safe_open(str(shard_path), framework="numpy") as handle:
            for name in handle.keys():
                shape = tuple(int(value) for value in handle.get_slice(name).get_shape())
                policy = classify_tensor(name, shape, text_bits=text_bits)
                tensor = _load_tensor(
                    shard_path, handle, header, data_start, name, shape
                )
                output_name = sanitize_key(name)
                method_counts[policy.method] = method_counts.get(policy.method, 0) + 1

                if policy.method == "discrete_affine":
                    try:
                        packed, scales, biases, rel_l1, max_abs_error = (
                            quantize_discrete_affine(
                                tensor,
                                bits=policy.bits,
                                group_size=policy.group_size,
                                chunk_rows=chunk_rows,
                                validate=True,
                            )
                        )
                    except ValueError as error:
                        raise ValueError(f"{name}: {error}") from error
                    base = _module_base(name)
                    tensors[f"{base}.weight"] = packed
                    tensors[f"{base}.scales"] = scales
                    tensors[f"{base}.biases"] = biases
                    manifest[base] = {
                        "bits": policy.bits,
                        "storage_bits": policy.bits,
                        "group_size": policy.group_size,
                        "mode": "affine",
                        "source_tensor": name,
                        "discrete_levels_exact": True,
                        "sidecar_dtype": "float16",
                        "source_rel_l1": rel_l1,
                        "source_max_abs_error": max_abs_error,
                    }
                    text_rel_l1[base] = rel_l1
                    text_max_abs_error[base] = max_abs_error
                elif policy.method == "mlx_affine":
                    packed, scales, biases, rel_l1 = _quantize_vision_native(
                        tensor,
                        bits=policy.bits,
                        group_size=policy.group_size,
                    )
                    base = _module_base(name)
                    tensors[f"{base}.weight"] = packed
                    tensors[f"{base}.scales"] = scales
                    tensors[f"{base}.biases"] = biases
                    manifest[base] = {
                        "bits": policy.bits,
                        "storage_bits": policy.bits,
                        "group_size": policy.group_size,
                        "mode": "affine",
                        "source_tensor": name,
                        "lossless_source_reconstruction": False,
                    }
                    vision_rel_l1[base] = rel_l1
                else:
                    tensors[output_name] = _prepare_passthrough(name, tensor)

                del tensor
                progress.update(1)
                if progress.n % 32 == 0:
                    gc.collect()
        del header
    progress.close()

    model_config = _patch_model_config_tokens(source_config)
    quantized_parameters = (
        plan.text_quantized_parameters + plan.vision_quantized_parameters
    )
    total_parameters = quantized_parameters + plan.passthrough_parameters
    nominal_storage_bits = (
        plan.text_quantized_parameters * text_bits
        + plan.vision_quantized_parameters * VISION_BITS
        + plan.passthrough_parameters * 16
    ) / max(total_parameters, 1)
    effective_storage_bits = plan.projected_bytes * 8 / max(total_parameters, 1)

    profile = PROFILE_BY_BITS[text_bits]
    jang_config: dict[str, Any] = {
        "format": "jang",
        "format_version": "2.0",
        "weight_format": "affine",
        "profile": profile,
        "quantization": {
            "method": "jang-affine-discrete",
            "profile": profile,
            "bits": text_bits,
            "block_size": TEXT_GROUP_SIZE,
            "group_size": TEXT_GROUP_SIZE,
            "mode": "affine",
            "quantization_scheme": "asymmetric",
            "quantization_backend": "jang-affine+mlx-compatible-cpu-affine",
            "bit_widths_used": sorted({text_bits, VISION_BITS}),
            "actual_bits": round(nominal_storage_bits, 4),
            "effective_bits_per_parameter": round(effective_storage_bits, 4),
            "tensor_quantization_manifest_schema": 2,
            "tensor_quantization_manifest_count": len(manifest),
            "tensor_quantization_manifest": manifest,
            "passthrough_tensor_count": plan.passthrough_tensor_count,
            "passthrough_parameter_count": plan.passthrough_parameters,
            "affine1_runtime_expansion": (
                {
                    "storage_bits": 1,
                    "runtime_bits": 2,
                    "lossless": True,
                    "scales_biases_unchanged": True,
                }
                if text_bits == 1
                else None
            ),
        },
        "source_model": {
            "name": model_path.name,
            "model_type": source_config.get("model_type"),
            "config_sha256": _sha256(model_path / "config.json"),
            "index_sha256": _sha256(model_path / "model.safetensors.index.json"),
            "dtype": "bfloat16-dequantized-binary"
            if text_bits == 1
            else "bfloat16-dequantized-ternary",
        },
        "architecture": {
            "type": "qwen3_5",
            "text_model_type": "qwen3_5_text",
            "has_vision": True,
            "has_ssm": True,
            "has_moe": False,
            "num_hidden_layers": 64,
            "attn_output_gate": bool(
                (source_config.get("text_config") or {}).get("attn_output_gate")
            ),
        },
        "layout": {
            "key_layout": "mlx-vlm-qwen3_5",
            "language_norms": "zero-centered-runtime-plus-one",
            "linear_attn_conv1d": "hf-out-one-kernel-runtime-moveaxis",
            "vision_patch_embed": "mlx-channel-last",
        },
        "runtime": {
            "target": "vmlx-python",
            "requires_jang_affine1_expansion": text_bits == 1,
            "bundle_has_mtp": False,
        },
        "conversion_plan": plan.to_dict(),
    }

    try:
        from .capabilities import build_capabilities

        capabilities = build_capabilities(
            jang_config, model_config, tensor_names=list(weight_map)
        )
        if capabilities is not None:
            jang_config["capabilities"] = capabilities
    except Exception:
        pass

    tokenizer_files = _load_tokenizer_files(model_path)
    write_jang_v2_model(
        output_dir=output_path,
        tensors=tensors,
        model_config=model_config,
        jang_config=jang_config,
        tokenizer_files=tokenizer_files,
        max_shard_bytes=2 * 1024**3,
    )
    copied_files = _copy_extra_files(model_path, output_path)
    verification = _verify_bundle(
        output_path, manifest=manifest, text_bits=text_bits
    )

    final_jang_config = json.loads(
        (output_path / "jang_config.json").read_text(encoding="utf-8")
    )
    final_jang_config["runtime"]["total_weight_bytes"] = verification[
        "indexed_shard_bytes"
    ]
    final_jang_config["runtime"]["total_weight_gib"] = verification[
        "indexed_shard_gib"
    ]
    (output_path / "jang_config.json").write_text(
        json.dumps(final_jang_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report = {
        "status": "converted_not_runtime_verified",
        "profile": profile,
        "source": str(model_path),
        "output": str(output_path),
        "text_storage_bits": text_bits,
        "text_group_size": TEXT_GROUP_SIZE,
        "vision_storage_bits": VISION_BITS,
        "plan": plan.to_dict(),
        "methods": method_counts,
        "text_affine_error": {
            "tensor_count": len(text_rel_l1),
            "mean_rel_l1": round(float(np.mean(list(text_rel_l1.values()))), 12)
            if text_rel_l1
            else None,
            "max_rel_l1": round(float(max(text_rel_l1.values())), 12)
            if text_rel_l1
            else None,
            "max_abs_error": float(max(text_max_abs_error.values()))
            if text_max_abs_error
            else None,
            "boundary": "exact discrete levels; error is float16 scale/bias rounding",
        },
        "vision_rel_l1": {
            "tensor_count": len(vision_rel_l1),
            "mean": round(float(np.mean(list(vision_rel_l1.values()))), 8)
            if vision_rel_l1
            else None,
            "max": round(float(max(vision_rel_l1.values())), 8)
            if vision_rel_l1
            else None,
        },
        "copied_auxiliary_files": copied_files,
        "verification": verification,
    }
    (output_path / "jang_affine_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert binary/ternary Bonsai-27B BF16 mirrors to JANG affine"
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--text-bits", type=int, choices=(1, 2), required=True)
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect tensor headers and print the exact projected storage budget",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = plan_conversion(args.source, text_bits=args.text_bits)
    if args.dry_run:
        print(json.dumps(plan.to_dict(), indent=2))
        return 0
    report = convert_bonsai_jang_affine(
        args.source,
        args.output,
        text_bits=args.text_bits,
        chunk_rows=args.chunk_rows,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
