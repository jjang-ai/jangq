"""Build the `jangtq_runtime.safetensors` sidecar for a JANGTQ artifact.

The Python loader (`load_jangtq.py`) computes signs and codebooks on
the fly from `(in_features, seed, bits)` — no sidecar needed.

The Swift runtime (`vMLXLMCommon/JANGTQKernels.swift::JANGTQRuntimeCache`)
expects a sidecar file colocated with the model. This script reads the
JANGTQ artifact, identifies every unique `(in_features, seed)` and
`(in_features, bits)` triple referenced by its `.tq_packed` weights,
and writes the sidecar in the format Swift expects:

    signs.{in_features}.{seed}    : float32 [in_features]
    codebook.{in_features}.{bits} : float32 [2^bits]

Usage:
    python3 -m jang_tools.build_jangtq_sidecar \\
        /path/to/Qwen3.6-35B-A3B-JANGTQ_2L

After this runs, `vmlxctl` Swift will detect `jangtq_runtime.safetensors`
in the model dir and `JANGTQRuntimeCache.shared.loadSidecar` will
populate the cache before first forward — no fatalError.

Why a sidecar (not on-the-fly compute) on Swift: the Hadamard signs +
Lloyd-Max codebook are fp32 arrays that depend purely on
(in_features, seed, bits). Computing them in Python is trivial; porting
the Lloyd-Max iteration to Swift would duplicate ~200 LOC of numerically
fragile code. Writing them to disk once at convert time is the
simpler contract.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

# Reuse the Python-side generators directly so signs + codebooks are
# bit-identical to what `load_jangtq.py` computes at runtime.
from jang_tools.turboquant.rotation import generate_random_signs
from jang_tools.turboquant.codebook import compute_codebook


def _scan_tq_tensors(model_dir: Path) -> list[tuple[str, list[int]]]:
    """Return list of (tensor_key, shape) for every .tq_packed weight."""
    out = []
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        # M125 (iter 48): context-manage the read so the fd closes promptly.
        with open(index_path) as fh:
            index = json.load(fh)
        weight_map = index["weight_map"]
        # Group by shard for I/O efficiency.
        by_shard: dict[str, list[str]] = {}
        for k, fname in weight_map.items():
            by_shard.setdefault(fname, []).append(k)
        for fname, keys in by_shard.items():
            with safe_open(str(model_dir / fname), framework="numpy") as f:
                for k in keys:
                    if k.endswith(".tq_packed"):
                        out.append((k, list(f.get_slice(k).get_shape())))
    else:
        # Single safetensors file fallback
        for sf in sorted(model_dir.glob("model-*.safetensors")):
            with safe_open(str(sf), framework="numpy") as f:
                for k in f.keys():
                    if k.endswith(".tq_packed"):
                        out.append((k, list(f.get_slice(k).get_shape())))
    return out


def _infer_in_features(packed_shape: list[int], bits: int) -> int:
    """`tq_packed` is shape (..., out_features, packed_cols) where
    packed_cols = ceil(in_features * bits / 32). Recover in_features
    from packed_cols + bits.

    Each uint32 stores 32/bits codebook indices, so:
        packed_cols * 32 / bits == in_features  (assuming in_features divisible)
    """
    packed_cols = packed_shape[-1]
    return (packed_cols * 32) // bits


def _read_tq_bits(model_dir: Path, packed_key: str) -> int:
    """Read the companion `.tq_bits` tensor and return its scalar value."""
    bits_key = packed_key[: -len(".tq_packed")] + ".tq_bits"
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as fh:
            index = json.load(fh)
        fname = index["weight_map"].get(bits_key)
        if fname is None:
            return 0
        with safe_open(str(model_dir / fname), framework="numpy") as f:
            arr = f.get_tensor(bits_key)
            return int(arr.flat[0])
    for sf in sorted(model_dir.glob("model-*.safetensors")):
        with safe_open(str(sf), framework="numpy") as f:
            if bits_key in f.keys():
                return int(f.get_tensor(bits_key).flat[0])
    return 0


def _read_tq_bits_batch(model_dir: Path, packed_keys: list[str]) -> dict[str, int]:
    """Batched companion to `_read_tq_bits` — open each shard ONCE and read all
    `.tq_bits` scalars from it. ~30× faster than per-key opens for 30K+ tensors
    on TB5 external SSDs (verified during 2026-04-24 gsz=32 sidecar build:
    per-key path took ~15 min for 33,792 tensors; batched path is ~30 sec).
    """
    bits_keys = [k[: -len(".tq_packed")] + ".tq_bits" for k in packed_keys]
    out: dict[str, int] = {}
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as fh:
            index = json.load(fh)
        weight_map = index["weight_map"]
        # Group bits_keys by shard.
        by_shard: dict[str, list[str]] = {}
        missing: list[str] = []
        for bk in bits_keys:
            fname = weight_map.get(bk)
            if fname is None:
                missing.append(bk)
            else:
                by_shard.setdefault(fname, []).append(bk)
        for fname, keys in by_shard.items():
            with safe_open(str(model_dir / fname), framework="numpy") as f:
                for bk in keys:
                    arr = f.get_tensor(bk)
                    out[bk] = int(arr.flat[0])
        for bk in missing:
            out[bk] = 0
        return out
    # Fallback: scan all shards once, harvesting any bits_keys present.
    remaining = set(bits_keys)
    for sf in sorted(model_dir.glob("model-*.safetensors")):
        if not remaining:
            break
        with safe_open(str(sf), framework="numpy") as f:
            shard_keys = set(f.keys())
            hits = remaining & shard_keys
            for bk in hits:
                out[bk] = int(f.get_tensor(bk).flat[0])
            remaining -= hits
    for bk in remaining:
        out[bk] = 0
    return out


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    model_dir = Path(sys.argv[1])
    if not model_dir.is_dir():
        sys.exit(f"FATAL: {model_dir} is not a directory")

    jang_config_path = model_dir / "jang_config.json"
    if not jang_config_path.exists():
        sys.exit(f"FATAL: missing {jang_config_path} — not a JANGTQ artifact?")
    with open(jang_config_path) as fh:
        jang_cfg = json.load(fh)
    seed = int(jang_cfg.get("mxtq_seed", 42))
    print(f"  seed: {seed}")

    print("  Scanning for .tq_packed weights...")
    packed = _scan_tq_tensors(model_dir)
    if not packed:
        print("  [sidecar] No .tq_packed tensors found; skipping sidecar generation.")
        return
    print(f"  Found {len(packed)} TQ-packed tensors")

    # Collect the unique (in_features, bits) pairs and (in_features, seed)
    # pairs that the runtime cache will be queried for.
    # Batched read: open each shard once, harvest all .tq_bits at once.
    print(f"  Reading .tq_bits for {len(packed)} tensors (batched)...", flush=True)
    keys_only = [k for k, _ in packed]
    bits_map = _read_tq_bits_batch(model_dir, keys_only)
    unique_in_bits: set[tuple[int, int]] = set()
    unique_in_seed: set[tuple[int, int]] = set()
    for key, shape in packed:
        bk = key[: -len(".tq_packed")] + ".tq_bits"
        bits = bits_map.get(bk, 0)
        if bits == 0:
            print(f"  WARN: missing .tq_bits for {key}, skipping")
            continue
        in_feat = _infer_in_features(shape, bits)
        unique_in_bits.add((in_feat, bits))
        unique_in_seed.add((in_feat, seed))

    print(f"  Unique (in_features, bits) pairs: {len(unique_in_bits)}")
    print(f"  Unique (in_features, seed) pairs: {len(unique_in_seed)}")
    for ib in sorted(unique_in_bits):
        print(f"    in_features={ib[0]}  bits={ib[1]}")

    sidecar: dict[str, np.ndarray] = {}

    print("  Generating signs...")
    for in_feat, s in sorted(unique_in_seed):
        signs = np.asarray(generate_random_signs(in_feat, seed=s), dtype=np.float32)
        if signs.shape != (in_feat,):
            # generate_random_signs may return a mlx array; ensure shape.
            signs = signs.reshape(in_feat)
        key = f"signs.{in_feat}.{s}"
        sidecar[key] = signs
        print(f"    {key}  shape={list(signs.shape)}")

    print("  Computing codebooks...")
    for in_feat, bits in sorted(unique_in_bits):
        cb = np.asarray(compute_codebook(in_feat, bits), dtype=np.float32)
        # Lloyd-Max codebook should be 2^bits entries.
        if cb.shape != (1 << bits,):
            cb = cb.reshape(1 << bits)
        key = f"codebook.{in_feat}.{bits}"
        sidecar[key] = cb
        print(f"    {key}  shape={list(cb.shape)}  range=[{cb.min():.4f}, {cb.max():.4f}]")

    out_path = model_dir / "jangtq_runtime.safetensors"
    save_file(sidecar, str(out_path))
    total_bytes = sum(v.nbytes for v in sidecar.values())
    print(f"\n  Wrote {out_path} ({len(sidecar)} tensors, {total_bytes / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
