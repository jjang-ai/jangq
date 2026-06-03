"""Gemma 4 (gemma4_unified) omni-modal -> JANG (MLX-native affine) conversion.

Produces a JANG_4M dense bundle: **8-bit affine attention (q/k/v/o) + 4-bit
affine MLP (gate/up/down) + 4-bit affine tied token embedding**, with the thin
multimodal embedders / norms / layer_scalar kept fp16.

This reuses the verified gemma4 plumbing from `convert_gemma4_mxfp.py`
(`sanitize_key`, multimodal passthrough fragments, single-file scan, no-+1 norm
passthrough, k_eq_v missing-v_proj tolerance) and differs only in:

  * affine quantization (`mode="affine"`) at per-tier bit widths, and
  * writing a CORRECT mixed-bit `config.json["quantization"]` block —
    top-level `{group_size, bits=8, mode="affine"}` PLUS a per-module override
    `{bits=4,...}` for every 4-bit module. Without the per-module overrides a
    loader dequantizes the 8-bit attention with the 4-bit kernel and emits
    garbage (the "config-metadata bit bug",
    research/JANGTQ-CONFIG-METADATA-BUG-2026-04-24.md).

Dense note: the MoE-only allocation rules (MLP gate/down asymmetry, bf16
activations, expert codebooks) do NOT apply here — gemma4-12B is dense.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import mlx.core as mx
from safetensors.numpy import save_file

from jang_tools.capabilities import build_capabilities, verify_directory
from jang_tools.convert import _remove_stale_jang_artifacts
from jang_tools.progress import ProgressEmitter
from jang_tools.convert_gemma4_mxfp import (
    MAX_SHARD,
    SIDECAR_FILES,
    _MULTIMODAL_FRAGMENTS,
    _copy_sidecars,
    _load_tensor,
    _prepare_passthrough,
    _scan_source,
    sanitize_key,
)

# JANG_4M dense tier widths.
ATTN_BITS = 8       # CRITICAL: self_attn q/k/v/o
MLP_BITS = 4        # COMPRESS: mlp gate/up/down
EMBED_BITS = 4      # IMPORTANT: tied token embedding
DEFAULT_TOP_BITS = 8  # config.json top-level default (matches the 8-bit majority)


def jang_bits(tensor_name: str) -> int | None:
    """Return affine bit width for a tensor, or None for fp16 passthrough."""
    name = tensor_name.lower()
    if tensor_name.endswith("_scale_inv"):
        return None
    if any(frag in name for frag in _MULTIMODAL_FRAGMENTS):
        return None
    if (
        "norm" in name
        or tensor_name.endswith(".bias")
        or tensor_name.endswith("layer_scalar")
        or tensor_name.endswith("pos_embedding")
        or tensor_name.endswith("embed_scale")
    ):
        return None
    if len(tensor_name.split(".")) == 1:
        return None
    if ".self_attn." in name and any(
        name.endswith(f"{p}.weight") for p in ("q_proj", "k_proj", "v_proj", "o_proj")
    ):
        return ATTN_BITS
    if tensor_name.endswith("embed_tokens.weight"):
        return EMBED_BITS
    # mlp gate/up/down + any other 2D decoder linear
    return MLP_BITS


def _affine_quantize(tensor: np.ndarray, *, bits: int, group_size: int):
    if tensor.ndim >= 3:
        original_shape = tensor.shape
        tensor = tensor.reshape(-1, tensor.shape[-1])
    else:
        original_shape = None
    q_w, q_s, q_b = [], [], []
    chunk_rows = max(1, min(tensor.shape[0], 100_000_000 // max(1, tensor.shape[1])))
    for start in range(0, tensor.shape[0], chunk_rows):
        chunk = mx.array(tensor[start : start + chunk_rows].astype(np.float16))
        qw, qs, qb = mx.quantize(chunk, group_size=group_size, bits=bits, mode="affine")
        q_w.append(np.array(qw)); q_s.append(np.array(qs)); q_b.append(np.array(qb))
        mx.eval(qw, qs, qb)
        del chunk, qw, qs, qb
    weight = np.concatenate(q_w, axis=0)
    scales = np.concatenate(q_s, axis=0)
    biases = np.concatenate(q_b, axis=0)
    if original_shape is not None:
        weight = weight.reshape(original_shape[0], original_shape[1], -1)
        scales = scales.reshape(original_shape[0], original_shape[1], -1)
        biases = biases.reshape(original_shape[0], original_shape[1], -1)
    return weight, scales, biases


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Gemma 4 (gemma4_unified) source to JANG_4M.")
    p.add_argument("src", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quiet-text", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gs = args.group_size
    profile = "JANG_4M"
    progress = ProgressEmitter(json_to_stderr=False, quiet_text=args.quiet_text)
    src = args.src.expanduser()
    out = args.out.expanduser()

    config = json.loads((src / "config.json").read_text(encoding="utf-8"))
    text_cfg = config.get("text_config", config)
    n_layers = int(text_cfg.get("num_hidden_layers", 0))
    layer_types = text_cfg.get("layer_types") or []
    n_full = sum(1 for t in layer_types if t == "full_attention")

    print("=" * 70)
    print(f"  Gemma 4 (gemma4_unified) -> {profile} (dense, affine)")
    print("=" * 70)
    print(f"  Source:  {src}")
    print(f"  Output:  {out}")
    print(f"  Layers:  {n_layers}  (full-attn {n_full} / sliding {n_layers - n_full})")
    print(f"  Bits:    attn={ATTN_BITS}  mlp={MLP_BITS}  embed={EMBED_BITS}  (group_size {gs})")
    print(f"  Norm:    scale_shift=0 (NO +1) ; multimodal embedders fp16 ; MTP none")

    progress.phase(1, 3, "scan")
    tensors = _scan_source(src)
    print(f"  Found {len(tensors)} tensors")
    if args.dry_run:
        counts: dict[str, int] = {}
        for name, _shape, _sf in tensors:
            b = jang_bits(name)
            k = "fp16" if b is None else f"affine-{b}"
            counts[k] = counts.get(k, 0) + 1
        print(json.dumps(counts, indent=2, sort_keys=True))
        progress.done(ok=True, output="dry-run")
        return

    out.mkdir(parents=True, exist_ok=True)
    removed = _remove_stale_jang_artifacts(out)
    if removed:
        print(f"  Removed {len(removed)} stale output file(s)")

    shard_idx = 0
    shard_tensors: dict[str, np.ndarray] = {}
    shard_bytes = 0
    shard_map: dict[str, str] = {}
    overrides: dict[str, dict] = {}
    n_affine = n_pass = 0

    def flush_shard() -> None:
        nonlocal shard_idx, shard_tensors, shard_bytes
        if not shard_tensors:
            return
        shard_idx += 1
        name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        save_file(shard_tensors, str(out / name))
        for k in shard_tensors:
            shard_map[k] = name
        print(f"    Shard {shard_idx}: {len(shard_tensors)} tensors, {shard_bytes / 1e9:.2f} GB")
        shard_tensors = {}
        shard_bytes = 0

    def add_tensor(name: str, arr: np.ndarray) -> None:
        nonlocal shard_bytes
        shard_tensors[name] = arr
        shard_bytes += arr.nbytes
        if shard_bytes >= MAX_SHARD:
            flush_shard()

    progress.phase(2, 3, "convert")
    from tqdm import tqdm
    for tensor_name, shape, sf_path in tqdm(tensors, desc="  Processing"):
        bits = jang_bits(tensor_name)
        out_name = sanitize_key(tensor_name)
        tensor = _load_tensor(sf_path, tensor_name, shape)
        if bits is None or tensor.ndim < 2:
            tensor = _prepare_passthrough(out_name, tensor)
            add_tensor(out_name, tensor.astype(np.float16))
            n_pass += 1
        else:
            qw, qs, qb = _affine_quantize(tensor, bits=bits, group_size=gs)
            base = out_name[: -len(".weight")] if out_name.endswith(".weight") else out_name
            add_tensor(f"{base}.weight", qw)
            add_tensor(f"{base}.scales", qs)
            add_tensor(f"{base}.biases", qb)
            # Per-module override ONLY when this module differs from the 8-bit
            # top-level default (i.e. the 4-bit MLP/embed modules).
            if bits != DEFAULT_TOP_BITS:
                overrides[base] = {"group_size": gs, "bits": bits, "mode": "affine"}
            n_affine += 1
            del qw, qs, qb
        del tensor
        if (n_affine + n_pass) % 200 == 0:
            gc.collect(); mx.clear_cache()

    flush_shard()

    progress.phase(3, 3, "write")
    for idx in range(1, shard_idx + 1):
        old = out / f"model-{idx:05d}-of-XXXXX.safetensors"
        new = out / f"model-{idx:05d}-of-{shard_idx:05d}.safetensors"
        if old.exists():
            old.rename(new)
    shard_map = {k: v.replace("XXXXX", f"{shard_idx:05d}") for k, v in shard_map.items()}
    total_size = sum((out / name).stat().st_size for name in set(shard_map.values()))
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"format": "jang_affine", "total_size": total_size}, "weight_map": shard_map}, indent=2),
        encoding="utf-8",
    )

    # config.json: mlx-loadable mixed-bit quantization block.
    config.pop("quantization_config", None)
    config["weight_format"] = "jang_affine"
    quant_block: dict = {"group_size": gs, "bits": DEFAULT_TOP_BITS, "mode": "affine"}
    quant_block.update(overrides)
    config["quantization"] = quant_block
    caps = build_capabilities({}, config, out)
    if caps is not None:
        config["capabilities"] = caps
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    jang_config = {
        "version": 2,
        "weight_format": "jang_affine",
        "profile": profile,
        "source_model": {
            "name": src.name,
            "architecture": text_cfg.get("model_type", config.get("model_type", "gemma4_unified_text")),
        },
        "has_vision": True,
        "has_audio": True,
        "quantization": {
            "method": "jang_affine",
            "quantization_backend": "mx.quantize",
            "mode": "affine",
            "group_size": gs,
            "tier_bits": {"attention": ATTN_BITS, "mlp": MLP_BITS, "embed": EMBED_BITS},
            "norm_convention": "gemma4_scale_shift_zero",
            "multimodal": "fp16_passthrough_embedders_early_fusion",
            "mtp_policy": "none",
            "per_module_override_count": len(overrides),
            "passthrough_tensor_count": n_pass,
        },
        "runtime": {
            "total_weight_bytes": total_size,
            "total_weight_gb": round(total_size / (1024 ** 3), 2),
            "attention": "hybrid_swa_full_5to1",
            "sliding_window": text_cfg.get("sliding_window"),
            "attention_k_eq_v_on_full_layers": bool(text_cfg.get("attention_k_eq_v")),
            "full_attention_layers": [i for i, t in enumerate(layer_types) if t == "full_attention"],
        },
    }
    caps = build_capabilities(jang_config, config, out)
    if caps is not None:
        jang_config["capabilities"] = caps
    (out / "jang_config.json").write_text(json.dumps(jang_config, indent=2), encoding="utf-8")
    _copy_sidecars(src, out)

    ok, msg = verify_directory(out)
    print(f"  verify: {msg}")
    if not ok:
        raise SystemExit(f"capabilities verify failed: {msg}")

    print("\n  Done!")
    print(f"  Affine tensors:      {n_affine}  (per-module 4-bit overrides: {len(overrides)})")
    print(f"  Passthrough tensors: {n_pass}")
    print(f"  Output:              {out}")
    progress.done(ok=True, output=str(out))


if __name__ == "__main__":
    main()
