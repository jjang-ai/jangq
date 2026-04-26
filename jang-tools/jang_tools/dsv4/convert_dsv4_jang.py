"""DSV4-Flash → JANG_2L (standard mx.quantize path, not MXTQ codebook).

Differences vs convert_dsv4_jangtq.py:
  - Routed experts use mx.quantize(bits=2) affine (standard mlx-lm) instead
    of MXTQ codebook. Larger size vs JANGTQ2, but works with stock mlx-lm
    loader if our custom runtime isn't ready yet.
  - Attention + shared + embed + lm_head: 8-bit affine (same).
  - Norms / router / mHC: fp16 passthrough (same).

Usage:
  python -m jang_tools.dsv4.convert_dsv4_jang \\
      --src <path/to/DeepSeek-V4-Flash> \\
      --dst ~/.mlxstudio/models/JANGQ-AI/DSV4-Flash-JANG_2L \\
      --profile 2
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file as sf_save_np

from jang_tools.dsv4.weight_loader import ShardIndex


def classify(name: str, profile_bits: int) -> tuple[int, str]:
    """Same rules as convert_dsv4_jangtq.classify but all quantizable
    weights go through `affine` (mx.quantize)."""
    if ("norm" in name or name.endswith(".bias") or "attn_sink" in name
            or ".ape" in name or "tid2eid" in name or name.startswith("hc_")
            or re.search(r"^layers\.\d+\.hc_", name)
            or re.search(r"^mtp\.\d+\.hc_", name)):
        return 16, "passthrough"
    if name.endswith(".gate.weight") and "experts" not in name:
        return 16, "passthrough"
    # Routed expert → affine at profile_bits (2, 3, or 4)
    if re.search(r"ffn\.experts\.\d+\.(w1|w2|w3)\.weight$", name):
        return profile_bits, "affine"
    # Everything else → 8-bit affine
    if name.endswith(".weight"):
        return 8, "affine"
    return 16, "passthrough"


def convert(src: Path, dst: Path, profile_bits: int) -> None:
    import mlx.core as mx

    dst.mkdir(parents=True, exist_ok=True)
    idx = ShardIndex(src)
    print(f"[convert] source: {src}")
    print(f"[convert] target: {dst}")
    print(f"[convert] profile: JANG_{profile_bits}L (all-affine)")
    weight_keys = [k for k in idx.keys if not k.endswith(".scale")]
    print(f"[convert] {len(weight_keys)} logical tensors")

    MAX_SHARD_BYTES = 1_000_000_000
    shard_idx = 1
    shard_bytes = 0
    shard_buf: dict[str, np.ndarray] = {}
    shard_map: dict[str, str] = {}
    totals = {"affine": 0, "passthrough": 0}
    t_start = time.time()

    def flush_shard():
        nonlocal shard_idx, shard_bytes, shard_buf
        if not shard_buf:
            return
        shard_name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        sf_save_np(shard_buf, str(dst / shard_name))
        for k in shard_buf:
            shard_map[k] = shard_name
        print(f"    shard {shard_idx}: {len(shard_buf)} tensors, "
              f"{shard_bytes / 1e9:.2f} GB  "
              f"(elapsed {time.time() - t_start:.0f}s)", flush=True)
        shard_buf = {}
        shard_bytes = 0
        shard_idx += 1

    def add_tensor(name: str, arr: np.ndarray):
        nonlocal shard_bytes
        shard_buf[name] = arr
        shard_bytes += arr.nbytes
        if shard_bytes >= MAX_SHARD_BYTES:
            flush_shard()

    for i, name in enumerate(weight_keys):
        bits, method = classify(name, profile_bits)
        if method == "passthrough":
            t = idx.read_tensor(name, out_dtype=torch.float16)
            arr = t.numpy() if t.dtype != torch.bfloat16 else t.float().numpy().astype(np.float16)
            add_tensor(name, arr)
            totals["passthrough"] += 1
        else:  # affine
            t = idx.read_tensor(name, out_dtype=torch.float32)
            w = mx.array(t.numpy())
            qw, qs, qb = mx.quantize(w, group_size=32, bits=bits)
            base = name[:-len(".weight")] if name.endswith(".weight") else name
            add_tensor(f"{base}.weight", np.array(qw))
            add_tensor(f"{base}.scales", np.array(qs).astype(np.float16))
            add_tensor(f"{base}.biases", np.array(qb).astype(np.float16))
            totals["affine"] += 1
        if (i + 1) % 500 == 0:
            print(f"    processed {i + 1}/{len(weight_keys)}  "
                  f"affine={totals['affine']} passthrough={totals['passthrough']}  "
                  f"({time.time() - t_start:.0f}s)", flush=True)
    flush_shard()

    for k in range(1, shard_idx):
        old = dst / f"model-{k:05d}-of-XXXXX.safetensors"
        new = dst / f"model-{k:05d}-of-{shard_idx - 1:05d}.safetensors"
        if old.exists():
            old.rename(new)
    final_map = {k: v.replace("XXXXX", f"{shard_idx - 1:05d}") for k, v in shard_map.items()}
    total_bytes = sum((dst / fn).stat().st_size for fn in set(final_map.values()))
    (dst / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": total_bytes},
        "weight_map": final_map,
    }, indent=2))

    src_cfg = json.loads((src / "config.json").read_text())
    src_cfg.pop("quantization_config", None)
    # ROOT-CAUSE FIX 2026-04-24: top-level MUST stay bits=8 mode=affine to
    # match the dominant 8-bit-affine layout (attention/shared/embed/head/
    # compressor/indexer). Routed experts at profile_bits affine via
    # per-module overrides. Setting top-level bits=2 (old behavior) made
    # MLX dequantize 8-bit attention weights with the 2-bit kernel → silent
    # garbage. See research/JANGTQ-CONFIG-METADATA-BUG-2026-04-24.md.
    n_layers = src_cfg.get("num_hidden_layers", 43)
    quant_cfg: dict = {"group_size": 32, "bits": 8, "mode": "affine"}
    for L in range(n_layers):
        for proj in ("gate_proj", "down_proj", "up_proj"):
            path = f"model.layers.{L}.mlp.switch_mlp.{proj}"
            quant_cfg[path] = {"group_size": 32, "bits": profile_bits, "mode": "affine"}
    src_cfg["quantization"] = quant_cfg
    src_cfg["_name_or_path"] = f"DSV4-Flash-JANG_{profile_bits}L"

    # transformers ≥4.50 expects `rope_parameters` instead of `rope_scaling`.
    # Without it, mlx-lm load_tokenizer falls back to bare PreTrainedTokenizerFast.
    # See convert_dsv4_jangtq.py for full rationale.
    if "rope_parameters" not in src_cfg:
        rs = src_cfg.get("rope_scaling")
        if isinstance(rs, dict):
            rp = dict(rs)
            rp.setdefault("rope_type", rp.pop("type", "yarn"))
            rp.setdefault("rope_theta", src_cfg.get("rope_theta", 10000))
        else:
            rp = {
                "rope_type": "default",
                "rope_theta": src_cfg.get("rope_theta", 10000),
            }
        for k in ("factor", "beta_fast", "beta_slow", "rope_theta"):
            if k in rp and not isinstance(rp[k], float):
                rp[k] = float(rp[k])
        src_cfg["rope_parameters"] = rp
    (dst / "config.json").write_text(json.dumps(src_cfg, indent=2))

    (dst / "jang_config.json").write_text(json.dumps({
        "weight_format": "affine",
        "profile": f"JANG_{profile_bits}L",
        "source_model": str(src),
        "affine_bits": {
            "routed_expert": profile_bits,
            "attention": 8,
            "shared_expert": 8,
            "embed_tokens": 8,
            "lm_head": 8,
            "norms_router_hc": 16,
        },
    }, indent=2))

    copied = 0
    for p in src.iterdir():
        if p.is_file() and not p.name.endswith(".safetensors") \
                and p.name not in ("config.json", "model.safetensors.index.json"):
            shutil.copy2(p, dst / p.name)
            copied += 1
    enc = src / "encoding"
    if enc.is_dir():
        shutil.copytree(enc, dst / "encoding", dirs_exist_ok=True)
        copied += 1
    print(f"[convert] copied {copied} aux files/dirs")

    elapsed = time.time() - t_start
    print(f"\nDONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  affine={totals['affine']}  passthrough={totals['passthrough']}")
    print(f"  output size: {total_bytes / 1e9:.1f} GB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--profile", type=int, default=2, choices=(2, 3, 4))
    args = ap.parse_args()
    convert(args.src, args.dst, args.profile)
    return 0


if __name__ == "__main__":
    sys.exit(main())
