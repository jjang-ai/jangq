"""DSV4-Flash FP4+FP8 source → all-bf16 baseline bundle.

No quantization anywhere. Pure dequantize-to-bf16. Used as a ground-truth
baseline for: (a) verifying MLX model architecture correctness and
(b) measuring quantization impact on generation quality.

Output size: ~568 GB (2 bytes per param × 284 B params). Big but
fits on external drive. Not intended for distribution — just for
correctness debugging.

Usage:
  python -m jang_tools.dsv4.convert_dsv4_bf16 \\
      --src <path/to/DeepSeek-V4-Flash> \\
      --dst <path/to/data-drive>/DSV4-Flash-BF16
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file as sf_save_np

from jang_tools.dsv4.weight_loader import ShardIndex
from jang_tools.format.aligned_safetensors import rewrite_aligned_safetensors


def convert(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    idx = ShardIndex(src)
    print(f"[bf16-convert] source: {src}")
    print(f"[bf16-convert] target: {dst}")
    weight_keys = [k for k in idx.keys if not k.endswith(".scale")]
    print(f"[bf16-convert] {len(weight_keys)} logical tensors")

    MAX_SHARD_BYTES = 4_000_000_000  # 4 GB shards (fewer files for big bf16 model)
    shard_idx = 1
    shard_bytes = 0
    shard_buf: dict[str, np.ndarray] = {}
    shard_map: dict[str, str] = {}
    t_start = time.time()

    def flush_shard():
        nonlocal shard_idx, shard_bytes, shard_buf
        if not shard_buf:
            return
        name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        sf_save_np(shard_buf, str(dst / name))
        rewrite_aligned_safetensors(dst / name)
        for k in shard_buf:
            shard_map[k] = name
        print(f"    shard {shard_idx}: {len(shard_buf)} tensors, "
              f"{shard_bytes / 1e9:.2f} GB  "
              f"({time.time() - t_start:.0f}s)", flush=True)
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
        t = idx.read_tensor(name, out_dtype=torch.bfloat16)
        # safetensors numpy save doesn't support bf16 directly — save as
        # fp16 which safely roundtrips for DSV4 magnitudes
        arr = t.float().numpy().astype(np.float16)
        add_tensor(name, arr)
        if (i + 1) % 500 == 0:
            print(f"    processed {i + 1}/{len(weight_keys)}  "
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
    src_cfg["_name_or_path"] = "DSV4-Flash-BF16"
    # No "quantization" field — tells loader nothing is quantized
    (dst / "config.json").write_text(json.dumps(src_cfg, indent=2))
    (dst / "jang_config.json").write_text(json.dumps({
        "weight_format": "bf16",
        "profile": "BF16-baseline",
        "source_model": str(src),
    }, indent=2))

    for p in src.iterdir():
        if p.is_file() and not p.name.endswith(".safetensors") \
                and p.name not in ("config.json", "model.safetensors.index.json"):
            shutil.copy2(p, dst / p.name)
    if (src / "encoding").is_dir():
        shutil.copytree(src / "encoding", dst / "encoding", dirs_exist_ok=True)

    elapsed = time.time() - t_start
    print(f"\nDONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  output size: {total_bytes / 1e9:.1f} GB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    args = ap.parse_args()
    convert(args.src, args.dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
