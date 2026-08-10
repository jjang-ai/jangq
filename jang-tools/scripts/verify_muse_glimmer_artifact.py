#!/usr/bin/env python3
"""Structural and dequantization verifier for Muse Glimmer JANG bundles.

This is intentionally a quant-artifact gate, not a generation claim.  The
five-layer assistant/DFlash checkpoint is a separate artifact and must never
appear in either base-model JANG bundle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from jang_tools.calibrate import _load_bf16_tensor


PINNED_SOURCE_REVISION = "f84ecc3a0ea984a4c04542a84269e3d065350a6e"
PINNED_ASSISTANT_REVISION = "2c86316d689027b91123638739743fef1d425233"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _hf_revision(path: Path) -> str | None:
    metadata = path / ".cache/huggingface/download/config.json.metadata"
    return metadata.read_text(encoding="utf-8").splitlines()[0] if metadata.exists() else None


def _index(path: Path) -> dict[str, str]:
    return _json(path / "model.safetensors.index.json")["weight_map"]


def _headers(path: Path) -> tuple[dict[str, tuple[list[int], str]], list[str]]:
    entries: dict[str, tuple[list[int], str]] = {}
    shards = sorted(path.glob("model-*-of-*.safetensors"))
    for shard in shards:
        with safe_open(shard, framework="numpy") as handle:
            for key in handle.keys():
                if key in entries:
                    raise AssertionError(f"duplicate tensor key: {key}")
                view = handle.get_slice(key)
                entries[key] = ([int(v) for v in view.get_shape()], view.get_dtype())
    return entries, [shard.name for shard in shards]


def _tensor(path: Path, index: dict[str, str], key: str) -> np.ndarray:
    with safe_open(path / index[key], framework="numpy") as handle:
        return handle.get_tensor(key)


def _source_tensor(path: Path, index: dict[str, str], key: str) -> np.ndarray:
    shard = path / index[key]
    with safe_open(shard, framework="numpy") as handle:
        shape = tuple(handle.get_slice(key).get_shape())
        try:
            return handle.get_tensor(key).astype(np.float32)
        except TypeError:
            return _load_bf16_tensor(shard, key, shape)


def _dequant(path: Path, index: dict[str, str], manifest: dict, base: str) -> np.ndarray:
    import mlx.core as mx

    entry = manifest[base]
    weight = _tensor(path, index, entry["weight_key"])
    scales = _tensor(path, index, entry["scales_key"])
    biases = _tensor(path, index, entry["biases_key"])
    value = mx.dequantize(
        mx.array(weight),
        mx.array(scales),
        mx.array(biases),
        group_size=int(entry["group_size"]),
        bits=int(entry["bits"]),
        mode="affine",
    )
    mx.eval(value)
    return np.asarray(value).astype(np.float32)


def _rel_l1(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - expected)) / max(float(np.mean(np.abs(expected))), 1e-12))


def verify(source: Path, artifact: Path, profile: str, dequant: bool) -> dict:
    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    source_config = _json(source / "config.json")
    jang = _json(artifact / "jang_config.json")
    output_config = _json(artifact / "config.json")
    source_index = _index(source)
    output_index = _index(artifact)
    headers, shards = _headers(artifact)

    require(_hf_revision(source) == PINNED_SOURCE_REVISION, "source revision is not pinned Muse Glimmer commit")
    require(source_config.get("model_type") == "muse_glimmer", "source model_type != muse_glimmer")
    require(output_config.get("model_type") == "muse_glimmer", "output model_type != muse_glimmer")
    require(output_config.get("weight_format") == "jang_affine", "config weight_format != jang_affine")
    require(jang.get("weight_format") == "jang_affine", "jang_config weight_format != jang_affine")
    require(jang.get("source_model", {}).get("revision") == PINNED_SOURCE_REVISION, "missing source revision stamp")
    require(jang.get("quantization", {}).get("profile") == profile, f"profile stamp != {profile}")
    require(jang.get("quantization", {}).get("source_qat") == "not_present", "QAT provenance is not explicit")
    require(jang.get("quantization", {}).get("imatrix_applied") is False, "fixed baseline falsely claims imatrix")
    require(jang.get("quantization", {}).get("awq_applied") is False, "fixed baseline falsely claims AWQ")

    caps = jang.get("capabilities", {})
    require(caps.get("family") == "muse_glimmer", "wrong capability family")
    require(caps.get("reasoning_parser") == "muse_glimmer", "wrong reasoning parser stamp")
    require(caps.get("tool_parser") == "atem", "wrong ATEM tool parser stamp")
    require(caps.get("has_vision") is True and caps.get("has_video") is True, "vision/video capability missing")
    require(output_config.get("capabilities") == caps, "config/jang capability stamps differ")

    chat = jang.get("chat", {})
    require(chat.get("reasoning_default") == "high", "reasoning default != high")
    require(chat.get("reasoning", {}).get("default_mode") == "high", "structured reasoning default != high")
    require(chat.get("tool_calling", {}).get("parser") == "atem", "structured ATEM tool stamp missing")
    require(chat.get("generation_defaults") == _json(source / "generation_config.json"), "generation defaults drifted")
    require(chat.get("sampling_defaults") == {"do_sample": False}, "sampling defaults are not source-exact")

    runtime = jang.get("runtime", {})
    require(runtime.get("sliding_window") == 2048, "sliding window != 2048")
    require(runtime.get("full_attention_layers") == list(range(3, 52, 4)), "full-attention schedule mismatch")
    require(len(runtime.get("sliding_attention_layers", [])) == 39, "sliding-attention layer count != 39")

    require(set(output_index) == set(headers), "artifact index/header key mismatch")
    require(all("NNNNN" not in name for name in shards), "unfinished shard filename remains")
    require(all(output_index[key] in shards for key in output_index), "index references a missing shard")
    require(not any("assistant" in key.lower() or "mtp" in key.lower() for key in output_index), "assistant/MTP tensors leaked into base bundle")

    for sidecar in ("chat_template.jinja", "processor_config.json", "generation_config.json"):
        require((artifact / sidecar).exists(), f"missing {sidecar}")
        if (artifact / sidecar).exists():
            require((artifact / sidecar).read_bytes() == (source / sidecar).read_bytes(), f"{sidecar} bytes drifted")

    vision_keys = [key for key in source_index if key.startswith("model.vision_")]
    require(bool(vision_keys), "source vision namespace not found")
    require(all(key in headers for key in vision_keys), "one or more vision tensors are missing")
    require(all(headers.get(key, ([], ""))[1] == "F16" for key in vision_keys), "vision tensors are not FP16 passthrough")

    manifest = jang.get("quantization", {}).get("tensor_quantization_manifest", {})
    require("language_model.lm_head" in manifest, "wrapped VL lm_head manifest missing")
    if "language_model.lm_head" in manifest:
        require(manifest["language_model.lm_head"].get("source_tensor") == "lm_head.weight", "lm_head source mapping is wrong")
    allowed_bits = {
        "JANG_2L": {2, 6, 8},
        "JANG_4M": {4, 8},
        "JANG_6M": {6, 8},
    }[profile]
    require({entry.get("bits") for entry in manifest.values()} <= allowed_bits, "unexpected profile bit width")
    quant_keys = {
        entry[key]
        for entry in manifest.values()
        for key in ("weight_key", "scales_key", "biases_key")
    }
    passthrough_count = len(set(output_index) - quant_keys)
    require(
        jang.get("quantization", {}).get("passthrough_tensor_count") == passthrough_count,
        "passthrough tensor count stamp is inaccurate",
    )

    for publication_file in ("README.md", "LICENSE", "USAGE_POLICY.md", "osaurus-x-banner.png"):
        require((artifact / publication_file).is_file(), f"missing publication file {publication_file}")
    for inherited_file in ("LICENSE", "USAGE_POLICY.md"):
        if (artifact / inherited_file).exists():
            require((artifact / inherited_file).read_bytes() == (source / inherited_file).read_bytes(), f"{inherited_file} drifted")

    rel_l1: dict[str, float] = {}
    if dequant and not failures:
        samples = {
            "language_model.model.layers.0.self_attn.k_proj":
                "model.language_model.layers.0.self_attn.k_proj.weight",
            "language_model.model.layers.0.mlp.down_proj":
                "model.language_model.layers.0.mlp.down_proj.weight",
        }
        # Artifact sanity ceilings, not language-coherence acceptance gates.
        thresholds = {"JANG_2L": 0.50, "JANG_4M": 0.16, "JANG_6M": 0.08}
        for base, source_key in samples.items():
            actual = _dequant(artifact, output_index, manifest, base)
            expected = _source_tensor(source, source_index, source_key)
            score = _rel_l1(actual, expected)
            rel_l1[base] = score
            require(score < thresholds[profile], f"{base} rel-L1 {score:.6f} exceeds threshold")
            del actual, expected

    result = {
        "status": "PASS" if not failures else "FAIL",
        "artifact": str(artifact),
        "profile": profile,
        "source_revision": jang.get("source_model", {}).get("revision"),
        "shards": len(shards),
        "tensor_keys": len(headers),
        "quantized_modules": len(manifest),
        "vision_passthrough_tensors": len(vision_keys),
        "all_passthrough_tensors": passthrough_count,
        "assistant_revision_preserved_separately": PINNED_ASSISTANT_REVISION,
        "dequant_rel_l1": rel_l1,
        "failures": failures,
        "proof_scope": "artifact structure and dequantization; generation deferred",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--source", type=Path, default=Path("~/models/meta-models/Muse-Glimmer-30B").expanduser())
    parser.add_argument("--profile", required=True, choices=("JANG_2L", "JANG_4M", "JANG_6M"))
    parser.add_argument("--dequant", action="store_true")
    args = parser.parse_args()
    result = verify(args.source.expanduser(), args.artifact.expanduser(), args.profile, args.dequant)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
