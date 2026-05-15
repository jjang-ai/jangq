"""Shared conversion helpers for Zyphra/ZAYA1-8B.

ZAYA uses alternating CCA-attention and top-1 MoE layers. The source stores
each expert as Megatron-style fused `linear_fc1` (gate/up) plus `linear_fc2`.
The JANG/JANGTQ runtime-facing layout splits and pre-stacks those experts under
`zaya_block.experts.switch_mlp.{gate_proj,up_proj,down_proj}`.
"""

from __future__ import annotations

import json
import re
import shutil
import struct
from pathlib import Path
from typing import Any


EXPERT_KEY_RE = re.compile(
    # ZAYA-text source emits `model.layers.N.zaya_block...`; ZAYA1-VL source
    # emits `model.layers.N.mlp.zaya_block...` (extra `.mlp.` segment). The
    # loader's _vlm_quant_weight_key_candidates handles both prefixes at load
    # time, so the converter strips the `.mlp.` here and emits a unified output
    # key under `expert_output_base` (no mlp).
    r"^model\.layers\.(\d+)\.(?:mlp\.)?zaya_block\.experts\.local_experts\.(\d+)\.(linear_fc1|linear_fc2)\.weight$"
)

PROFILE_BITS = {
    "JANGTQ2": 2,
    "JANGTQ_2L": 2,
    "JANGTQ_2S": 2,
    "JANGTQ3": 3,
    "JANGTQ_3L": 3,
    "JANGTQ_3S": 3,
    "JANGTQ4": 4,
    "JANGTQ_4M": 4,
    "JANGTQ_4K": 4,
    # Mixed-bit. Sentinel "mixed" tells convert_zaya_jangtq.py to consult
    # ZAYA_JANGTQ_K_BITS for per-projection allocation. Same scheme that
    # MiniMax-M2.7-JANGTQ_K ships (down=4, gate/up=2). Targets ZAYA's
    # top-1 + MOD-passthrough sensitivity that breaks plain JANGTQ2.
    "JANGTQ_K": "mixed",
    "JANGTQK": "mixed",
}

ZAYA_JANGTQ_K_BITS = {
    "gate_proj": 2,
    "up_proj": 2,
    "down_proj": 4,
}

CAPABILITIES = {
    # ZAYA reasons by default — measured live: enable_thinking=False (the
    # template's default closed </think> block) still produces chain-of-thought
    # output ("Okay, I need to calculate... step by step..."). Mark
    # supports_thinking=True; keep think_in_template=False because the
    # template's default render is a closed </think>, not an open one.
    # Callers can opt into open-reasoning rendering via enable_thinking=True
    # in apply_chat_template, but model behavior reasons either way.
    "reasoning_parser": "qwen3",
    "tool_parser": "zaya_xml",
    "think_in_template": False,
    "supports_tools": True,
    "supports_thinking": True,
    "family": "zaya",
    "modality": "text",
    "cache_type": "hybrid",
}

SIDECAR_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "chat_template.jinja",
    "chat_template.json",
    "merges.txt",
    "vocab.json",
    "README.md",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_tensor(sf_path: Path, tensor_name: str, shape: list[int] | tuple[int, ...]) -> np.ndarray:
    import numpy as np
    from safetensors import safe_open

    try:
        with safe_open(str(sf_path), framework="numpy") as f:
            arr = f.get_tensor(tensor_name)
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
    except Exception:
        from jang_tools.calibrate import _load_bf16_tensor

        arr = _load_bf16_tensor(sf_path, tensor_name, tuple(shape))
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr


def read_safetensor_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(header_size))


def _classify_tensor(
    key: str,
    shape: list[int],
    sf_path: Path,
    regular: list[tuple[str, list[int], Path]],
    experts: dict[tuple[int, int], dict[str, tuple[list[int], Path, str]]],
) -> None:
    m = EXPERT_KEY_RE.match(key)
    if m:
        layer = int(m.group(1))
        expert = int(m.group(2))
        kind = m.group(3)
        # Preserve the source key so callers can load the tensor without
        # reconstructing the path themselves (VL bundles use a `.mlp.`
        # segment that the text bundle doesn't).
        experts.setdefault((layer, expert), {})[kind] = (shape, sf_path, key)
    else:
        regular.append((key, shape, sf_path))


def _scan_source_from_headers(src: Path) -> tuple[list[tuple[str, list[int], Path]], dict[tuple[int, int], dict[str, tuple[list[int], Path, str]]]]:
    regular: list[tuple[str, list[int], Path]] = []
    experts: dict[tuple[int, int], dict[str, tuple[list[int], Path, str]]] = {}
    index_path = src / "model.safetensors.index.json"
    if index_path.exists():
        index = load_json(index_path)
        by_shard: dict[str, list[str]] = {}
        for key, shard in index.get("weight_map", {}).items():
            by_shard.setdefault(shard, []).append(key)
        for shard, keys in sorted(by_shard.items()):
            sf_path = src / shard
            header = read_safetensor_header(sf_path)
            for key in sorted(keys):
                meta = header[key]
                _classify_tensor(key, list(meta.get("shape", [])), sf_path, regular, experts)
        return regular, experts

    for sf_path in sorted(src.glob("model-*.safetensors")):
        header = read_safetensor_header(sf_path)
        for key, meta in sorted(header.items()):
            if key == "__metadata__":
                continue
            _classify_tensor(key, list(meta.get("shape", [])), sf_path, regular, experts)
    return regular, experts


def scan_source(src: Path) -> tuple[list[tuple[str, list[int], Path]], dict[tuple[int, int], dict[str, tuple[list[int], Path, str]]]]:
    try:
        from safetensors import safe_open
    except Exception:
        return _scan_source_from_headers(src)

    regular: list[tuple[str, list[int], Path]] = []
    experts: dict[tuple[int, int], dict[str, tuple[list[int], Path, str]]] = {}
    for sf_path in sorted(src.glob("model-*.safetensors")):
        with safe_open(str(sf_path), framework="numpy") as f:
            for key in f.keys():
                shape = list(f.get_slice(key).get_shape())
                _classify_tensor(key, shape, sf_path, regular, experts)
    return regular, experts


def is_passthrough(name: str) -> bool:
    if "norm" in name:
        return True
    if ".res_scale." in name:
        return True
    if ".router." in name:
        return True
    if ".conv_qk." in name:
        return True
    if name.endswith(".temp"):
        return True
    if name.endswith(".bias"):
        return True
    if name.endswith(".balancing_biases"):
        return True
    if name.endswith(".router_states_scale"):
        return True
    if name.endswith(".hidden_states_scale") or name.endswith(".hidden_states_bias"):
        return True
    if name.endswith(".residual_scale") or name.endswith(".residual_bias"):
        return True
    if name.endswith(".weight") and len(name.split(".")) < 2:
        return True
    return False


def regular_bits(name: str, default_bits: int, embed_bits: int) -> int:
    if name == "model.embed_tokens.weight" or name == "lm_head.weight":
        return embed_bits
    return default_bits


def affine_quantize(weight: np.ndarray, bits: int, group_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import mlx.core as mx
    import numpy as np

    w = mx.array(weight.astype(np.float16))
    qw, qs, qb = mx.quantize(w, group_size=group_size, bits=bits)
    out = (np.array(qw), np.array(qs).astype(np.float16), np.array(qb).astype(np.float16))
    del w, qw, qs, qb
    return out


def tq_quantize_weight_rowwise(weight: np.ndarray, bits: int, seed: int) -> dict[str, np.ndarray]:
    """TurboQuant a 2-D weight matrix with per-row bit packing.

    Existing flat packing is only byte-compatible when `in_features` is
    divisible by `32 // bits`. ZAYA's 2048-wide experts break that for 3-bit
    (vals_per_u32=10), so converters use row-wise packing.
    """

    import mlx.core as mx
    import numpy as np

    from jang_tools.turboquant.codebook import compute_codebook
    from jang_tools.turboquant.rotation import generate_random_signs, hadamard_rotate

    out_features, in_features = weight.shape
    w = mx.array(weight.astype(np.float32))
    signs = mx.array(generate_random_signs(in_features, seed=seed))
    w_rot = hadamard_rotate(w, signs)
    norms = mx.sqrt(mx.sum(w_rot * w_rot, axis=1, keepdims=True))
    norms_safe = mx.maximum(norms, mx.array(1e-10))
    w_normed = w_rot / norms_safe

    cb = mx.array(compute_codebook(in_features, bits))
    boundaries = (cb[:-1] + cb[1:]) / 2.0
    indices = mx.zeros(w_normed.shape, dtype=mx.uint8)
    for boundary in boundaries:
        indices = indices + (w_normed > boundary).astype(mx.uint8)

    vals_per_u32 = 32 // bits
    pad = (vals_per_u32 - (in_features % vals_per_u32)) % vals_per_u32
    if pad:
        indices = mx.concatenate(
            [indices, mx.zeros((out_features, pad), dtype=mx.uint8)],
            axis=1,
        )
    rows = indices.reshape(out_features, -1, vals_per_u32).astype(mx.uint32)
    packed = mx.zeros((out_features, rows.shape[1]), dtype=mx.uint32)
    for idx in range(vals_per_u32):
        packed = packed | (rows[:, :, idx] << (idx * bits))
    mx.eval(packed, norms)
    return {
        "packed": np.array(packed),
        "norms": np.array(norms.squeeze(-1).astype(mx.float16)),
    }


def tq_quantize_experts_rowwise(weights: np.ndarray, bits: int, seed: int) -> dict[str, np.ndarray]:
    import numpy as np

    packed = []
    norms = []
    for expert_idx in range(weights.shape[0]):
        result = tq_quantize_weight_rowwise(weights[expert_idx], bits=bits, seed=seed)
        packed.append(result["packed"])
        norms.append(result["norms"])
    return {
        "packed": np.stack(packed, axis=0),
        "norms": np.stack(norms, axis=0).astype(np.float16),
    }


def split_expert_fc1(fc1: np.ndarray, hidden_size: int) -> tuple[np.ndarray, np.ndarray]:
    if fc1.shape[0] != 2 * hidden_size:
        raise ValueError(f"expected linear_fc1 rows={2 * hidden_size}, got {fc1.shape}")
    return fc1[:hidden_size], fc1[hidden_size:]


def expert_output_base(layer: int, proj: str) -> str:
    return f"model.layers.{layer}.zaya_block.experts.switch_mlp.{proj}"


def copy_sidecars_with_template(src: Path, out: Path) -> None:
    for file_name in SIDECAR_FILES:
        src_file = src / file_name
        if src_file.exists():
            shutil.copy2(str(src_file), str(out / file_name))

    tok_cfg = out / "tokenizer_config.json"
    template_path = out / "chat_template.jinja"
    if tok_cfg.exists() and template_path.exists():
        cfg = load_json(tok_cfg)
        if not cfg.get("chat_template"):
            cfg["chat_template"] = template_path.read_text(encoding="utf-8")
            write_json(tok_cfg, cfg)


def finalize_shards(out: Path, shard_idx: int, shard_map: dict[str, str]) -> dict[str, str]:
    for idx in range(1, shard_idx + 1):
        old = out / f"model-{idx:05d}-of-XXXXX.safetensors"
        new = out / f"model-{idx:05d}-of-{shard_idx:05d}.safetensors"
        if old.exists():
            old.rename(new)
    return {key: value.replace("XXXXX", f"{shard_idx:05d}") for key, value in shard_map.items()}


def total_shard_size(out: Path, shard_map: dict[str, str]) -> int:
    total = 0
    for file_name in set(shard_map.values()):
        path = out / file_name
        if path.exists():
            total += path.stat().st_size
    return total
