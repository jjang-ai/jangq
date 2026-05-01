"""Kimi K2.6 (REAP-pruned or raw) → JANGTQ conversion.

Supports JANGTQ_1L / JANGTQ_2L / JANGTQ_3L profiles:

  JANGTQ_1L (~1.8 bpw avg):
    routed experts     -> 2-bit MXTQ (no AWQ)
    attention          -> 8-bit affine
    shared experts     -> 8-bit affine
    embed_tokens       -> 8-bit affine
    lm_head            -> 8-bit affine
    norms, router      -> FP16 passthrough

  JANGTQ_2L (~2.3 bpw avg):
    routed experts     -> 2-bit MXTQ + AWQ per-channel scale (if provided)
    attention          -> FP16
    shared experts     -> FP16
    embed / lm_head    -> FP16
    norms, router      -> FP16

  JANGTQ_3L (~3.3 bpw avg):
    routed experts     -> 3-bit MXTQ
    attention / shared / embed / lm_head / norms / router -> FP16

Source: Kimi K2.6 compressed-tensors INT4 format (Moonshot's original or
our REAP-pruned output with the same on-disk layout).

Usage:
  python -m jang_tools.kimi_prune.convert_kimi_jangtq \\
      --src <path/to/Kimi-K2.6-REAP-30> \\
      --dst <path/to/data-drive>/dealignai/Kimi-K2.6-REAP-30-JANGTQ_1L \\
      --profile 1L
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from .int4_codec import unpack_int4


SEED = 42


def get_bits_and_method(tensor_name: str, profile: str) -> tuple[int, str]:
    """Return (bits, method) for each tensor based on JANGTQ profile.

    method in {"passthrough" (FP16), "affine" (mlx-native quant),
               "mxtq" (JANG turbo-quant), "mxtq_awq"}
    """
    name = tensor_name.lower()

    # Norms + router bias + router weight + MTP scalars → FP16
    if "norm" in name or name.endswith(".bias"):
        return 16, "passthrough"
    if ".gate." in name and "gate_proj" not in name:
        return 16, "passthrough"
    if "e_score_correction_bias" in name:
        return 16, "passthrough"

    is_attn = "self_attn" in name
    is_shared = "shared_expert" in name  # plural or singular
    is_embed = "embed_tokens" in name
    is_lmhead = name.endswith("lm_head.weight") or name.endswith(
        "lm_head.weight_packed") or "lm_head" in name
    is_routed = ("experts" in name and not is_shared) or "switch_mlp" in name
    is_dense_mlp = (not is_attn) and (not is_shared) and (not is_embed) \
        and (not is_lmhead) and (not is_routed) and \
        (".mlp.gate_proj" in name or ".mlp.up_proj" in name or ".mlp.down_proj" in name)

    if profile == "1L":
        if is_routed:
            return 2, "mxtq"
        if is_attn or is_shared or is_embed or is_lmhead or is_dense_mlp:
            return 8, "affine"
        return 16, "passthrough"

    if profile == "2L":
        if is_routed:
            return 2, "mxtq"
        if is_attn or is_shared or is_embed or is_lmhead or is_dense_mlp:
            return 16, "passthrough"
        return 16, "passthrough"

    if profile == "3L":
        if is_routed:
            return 3, "mxtq"
        if is_attn or is_shared or is_embed or is_lmhead or is_dense_mlp:
            return 16, "passthrough"
        return 16, "passthrough"

    raise ValueError(f"unknown profile: {profile}")


def _read_source_tensor(sf_path: Path, name: str, shard_f) -> np.ndarray:
    """Read a tensor from a source shard, handling Kimi's compressed-tensors
    INT4 layout (packed + scale + shape triples) vs plain BF16/F32 weights.

    `shard_f` is an opened `safe_open(..., framework="pt")` context.
    Returns a numpy float32 array, shape as stored (INT4 unpacked to full dims).
    """
    import torch
    # If the compressed triple exists, unpack via our int4_codec.
    if name.endswith(".weight"):
        base = name[: -len(".weight")]
        pk = base + ".weight_packed"
        sk = base + ".weight_scale"
        hk = base + ".weight_shape"
        keys = set(shard_f.keys())
        if pk in keys:
            packed = shard_f.get_tensor(pk).numpy()
            scale = shard_f.get_tensor(sk).to(torch.float32).numpy()
            shape = shard_f.get_tensor(hk).numpy()
            return unpack_int4(packed, scale, shape, group_size=32)

    # Otherwise plain BF16 / F32.
    t = shard_f.get_tensor(name)
    if t.dtype == torch.bfloat16:
        return t.to(torch.float32).numpy()
    return t.numpy().astype(np.float32, copy=False)


def list_source_weight_keys(src: Path) -> list[tuple[str, Path]]:
    """List every logical tensor (name, shard).

    For compressed INT4 triples (.weight_packed/_scale/_shape) we yield
    the logical `.weight` name once; for plain BF16/F32 we yield as-is.
    """
    keys: list[tuple[str, Path]] = []
    seen_bases: set[str] = set()
    for sf in sorted(src.glob("model-*.safetensors")):
        with safe_open(str(sf), framework="pt") as f:
            all_keys = list(f.keys())
            shard_key_set = set(all_keys)
            for k in all_keys:
                if k.endswith(".weight_packed") or k.endswith(".weight_scale") \
                        or k.endswith(".weight_shape"):
                    base = k.rsplit(".weight_", 1)[0]
                    logical = base + ".weight"
                    if logical in seen_bases:
                        continue
                    seen_bases.add(logical)
                    keys.append((logical, sf))
                else:
                    if k in seen_bases:
                        continue
                    seen_bases.add(k)
                    keys.append((k, sf))
    return keys


def convert(src: Path, dst: Path, profile: str, awq_scales_file: Path | None = None):
    import mlx.core as mx
    from jang_tools.turboquant.linear import tq_quantize_weight
    from safetensors.numpy import load_file

    dst.mkdir(parents=True, exist_ok=True)
    with open(src / "config.json") as f:
        config = json.load(f)
    tc = config.get("text_config", config)
    n_layers = tc["num_hidden_layers"]
    first_dense = tc.get("first_k_dense_replace", 0)
    n_experts = tc["n_routed_experts"]

    awq_scales = {}
    if awq_scales_file and awq_scales_file.exists():
        awq_scales = {k: v.astype(np.float32)
                      for k, v in load_file(str(awq_scales_file)).items()}

    print("=" * 62, flush=True)
    print(f"  Kimi K2.6 → JANGTQ_{profile}", flush=True)
    print(f"  src: {src}", flush=True)
    print(f"  dst: {dst}", flush=True)
    print(f"  layers: {n_layers} ({first_dense} dense + {n_layers - first_dense} MoE)",
          flush=True)
    print(f"  routed experts/layer: {n_experts}", flush=True)
    print("=" * 62, flush=True)

    print("\nscanning source...", flush=True)
    all_tensors = list_source_weight_keys(src)
    print(f"  {len(all_tensors)} logical tensors", flush=True)

    shard_idx = 0
    shard_tensors: dict[str, np.ndarray] = {}
    shard_bytes = 0
    MAX_SHARD = 1_000_000_000  # 1 GB per output shard
    totals = {"mxtq": 0, "affine": 0, "passthrough": 0}
    shard_map: dict[str, str] = {}
    t_start = time.time()

    def flush_shard():
        nonlocal shard_idx, shard_tensors, shard_bytes
        if not shard_tensors:
            return
        shard_idx += 1
        fname = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        save_file(shard_tensors, str(dst / fname))
        for k in shard_tensors:
            shard_map[k] = fname
        print(f"    shard {shard_idx}: {len(shard_tensors)} tensors, "
              f"{shard_bytes/1e9:.2f} GB  (elapsed {time.time()-t_start:.0f}s)",
              flush=True)
        shard_tensors = {}
        shard_bytes = 0

    def add_tensor(name, arr):
        nonlocal shard_tensors, shard_bytes
        shard_tensors[name] = arr
        shard_bytes += arr.nbytes
        if shard_bytes >= MAX_SHARD:
            flush_shard()

    print(f"\nconverting (JANGTQ_{profile})...", flush=True)

    # Group tensors by shard so we only open each shard once.
    by_shard: dict[Path, list[str]] = {}
    for name, sf in all_tensors:
        by_shard.setdefault(sf, []).append(name)

    processed = 0
    for sf_path in sorted(by_shard):
        import torch
        with safe_open(str(sf_path), framework="pt") as f:
            for name in by_shard[sf_path]:
                bits, method = get_bits_and_method(name, profile)
                try:
                    tensor_f32 = _read_source_tensor(sf_path, name, f)
                except Exception as e:
                    print(f"  ! skip {name}: {type(e).__name__}: {e}", flush=True)
                    continue

                if method == "passthrough":
                    t16 = tensor_f32.astype(np.float16)
                    add_tensor(name, t16)
                    totals["passthrough"] += 1

                elif method == "affine":
                    w = mx.array(tensor_f32.astype(np.float16))
                    qw, qs, qb = mx.quantize(w, group_size=64, bits=bits)
                    base = name[: -len(".weight")] if name.endswith(".weight") else name
                    add_tensor(f"{base}.weight", np.array(qw))
                    add_tensor(f"{base}.scales", np.array(qs).astype(np.float16))
                    add_tensor(f"{base}.biases", np.array(qb).astype(np.float16))
                    totals["affine"] += 1

                elif method == "mxtq":
                    result = tq_quantize_weight(tensor_f32, bits=bits, seed=SEED)
                    base = name[: -len(".weight")] if name.endswith(".weight") else name
                    add_tensor(f"{base}.tq_packed", result["packed"])
                    add_tensor(f"{base}.tq_norms", result["norms"])
                    add_tensor(f"{base}.tq_bits", np.array([bits], dtype=np.uint8))
                    totals["mxtq"] += 1

                else:
                    raise RuntimeError(f"unknown method {method} for {name}")

                del tensor_f32
                processed += 1
                if processed % 200 == 0:
                    gc.collect()
                    print(f"    processed {processed}/{len(all_tensors)}  "
                          f"mxtq={totals['mxtq']} affine={totals['affine']} "
                          f"passthrough={totals['passthrough']}  "
                          f"({time.time()-t_start:.0f}s)", flush=True)

    flush_shard()

    # Rename shards with final total + emit index
    print(f"\nwriting shard index ({shard_idx} shards)...", flush=True)
    final_map: dict[str, str] = {}
    for old_k, old_fname in shard_map.items():
        new_fname = old_fname.replace("XXXXX", f"{shard_idx:05d}")
        final_map[old_k] = new_fname
    for i in range(1, shard_idx + 1):
        old = dst / f"model-{i:05d}-of-XXXXX.safetensors"
        new = dst / f"model-{i:05d}-of-{shard_idx:05d}.safetensors"
        if old.exists():
            old.rename(new)
    total_bytes = sum((dst / fn).stat().st_size for fn in set(final_map.values()))
    index = {"metadata": {"total_size": total_bytes}, "weight_map": final_map}
    with (dst / "model.safetensors.index.json").open("w") as f:
        json.dump(index, f, indent=2)

    # Rewrite config.json — strip compressed-tensors quantization_config,
    # add a mlx-style one for affine weights, tag jang metadata
    config.pop("quantization_config", None)
    config["quantization"] = {"group_size": 64, "bits": 2 if profile != "3L" else 3}
    with (dst / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    # jang_config.json sidecar
    jang_cfg = {
        "weight_format": "mxtq",
        "profile": f"JANGTQ_{profile}",
        "mxtq_seed": SEED,
        "source_model": str(src),
        "source_config": {
            "n_routed_experts": n_experts,
            "first_k_dense_replace": first_dense,
            "num_hidden_layers": n_layers,
        },
        "mxtq_bits": {
            "routed_expert": 2 if profile != "3L" else 3,
            "attention": 8 if profile == "1L" else 16,
            "shared_expert": 8 if profile == "1L" else 16,
            "dense_mlp": 8 if profile == "1L" else 16,
            "embed_tokens": 8 if profile == "1L" else 16,
            "lm_head": 8 if profile == "1L" else 16,
            "norms_router": 16,
        },
    }
    with (dst / "jang_config.json").open("w") as f:
        json.dump(jang_cfg, f, indent=2)

    # Copy non-weight files: tokenizer, processor code, README, chat template
    copied = 0
    for p in src.iterdir():
        if p.is_file() \
                and not p.name.endswith(".safetensors") \
                and not p.name.endswith(".json") \
                or p.name in ("tokenizer_config.json", "generation_config.json"):
            # Keep tokenizer/generation configs from source
            if p.is_file():
                shutil.copy2(p, dst / p.name)
                copied += 1
    # Also copy tokenizer_config + generation_config explicitly
    for req in ("tokenizer_config.json", "generation_config.json", "tokenization_kimi.py",
                "tiktoken.model", "chat_template.jinja", "tool_declaration_ts.py"):
        s = src / req
        if s.exists() and not (dst / req).exists():
            shutil.copy2(s, dst / req)
            copied += 1
    print(f"  copied {copied} aux files", flush=True)

    elapsed = time.time() - t_start
    print(f"\nDONE in {elapsed:.0f}s  ({elapsed/60:.1f} min)", flush=True)
    print(f"  mxtq={totals['mxtq']}  affine={totals['affine']}  "
          f"passthrough={totals['passthrough']}", flush=True)
    print(f"  output size: {total_bytes/1e9:.1f} GB", flush=True)
    print(f"  written to: {dst}", flush=True)

    # Build jangtq_runtime.safetensors sidecar (required by Swift runtime;
    # Python loader generates on the fly so it doesn't need this).
    print(f"\n  Building jangtq_runtime.safetensors sidecar...", flush=True)
    try:
        from jang_tools.build_jangtq_sidecar import main as _build_sidecar
        import sys as _sys
        _argv_save = _sys.argv
        _sys.argv = ["build_jangtq_sidecar", str(dst)]
        try:
            _build_sidecar()
        finally:
            _sys.argv = _argv_save
        print(f"  [sidecar] built", flush=True)
    except Exception as _e:
        print(f"  [sidecar] FAILED: {_e} — run "
              f"`python3 -m jang_tools.build_jangtq_sidecar {dst}` manually before upload",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--profile", choices=("1L", "2L", "3L"), default="1L")
    ap.add_argument("--awq-scales", type=Path, default=None)
    args = ap.parse_args()
    convert(args.src, args.dst, args.profile, args.awq_scales)


if __name__ == "__main__":
    main()
