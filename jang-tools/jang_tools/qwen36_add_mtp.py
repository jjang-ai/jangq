"""Retain the native MTP head in a Qwen3.6-27B JANG bundle.

`mlx_vlm.load()` does not instantiate `mtp.*`, so `save_weights` never writes
those tensors — the head silently disappears from the bundle. This copies the 15
`mtp.*` tensors from the source, quantizes the 2-D ones at the requested width,
appends them as an extra shard, and updates `model.safetensors.index.json`.

Retaining it is cheap (0.42 B params, ~1.5 % of the model) and the head is real:
`mtp_num_hidden_layers: 1`, shared embeddings, driven upstream as
`qwen3_next_mtp` with 2 speculative tokens. No MLX runtime decodes with it yet,
so `jang_config.mtp.runtime_available` stays False — presence of the weights is
not a claim of speculative-decoding acceleration.

    python -m jang_tools.qwen36_add_mtp <src> <bundle> [--bits 4] [--group-size 128]
"""
from __future__ import annotations

import glob
import json
import struct
import sys
from pathlib import Path

import mlx.core as mx
from safetensors.numpy import save_file
import numpy as np


def main(argv) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 1
    src, bundle = Path(argv[1]), Path(argv[2])
    bits, gs = 4, 128
    for i, a in enumerate(argv):
        if a == "--bits":
            bits = int(argv[i + 1])
        if a == "--group-size":
            gs = int(argv[i + 1])

    wm_src = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    mtp_keys = sorted(k for k in wm_src if k.startswith("mtp."))
    if not mtp_keys:
        print("  source has no mtp.* tensors — nothing to do")
        return 0
    print(f"  source MTP tensors: {len(mtp_keys)}")

    by_shard: dict[str, list[str]] = {}
    for k in mtp_keys:
        by_shard.setdefault(wm_src[k], []).append(k)

    out_tensors: dict[str, np.ndarray] = {}
    quantized, passthrough = 0, 0
    overrides: dict[str, dict] = {}

    for shard, keys in by_shard.items():
        arrs = mx.load(str(src / shard))
        for k in keys:
            w = arrs[k]
            base = k[: -len(".weight")] if k.endswith(".weight") else k
            # Only 2-D weights whose input dim fits a group are quantizable.
            if w.ndim == 2 and w.shape[-1] % gs == 0:
                q, s, b = mx.quantize(w.astype(mx.bfloat16), group_size=gs,
                                      bits=bits, mode="affine")
                mx.eval(q, s, b)
                out_tensors[f"{base}.weight"] = np.array(q)
                out_tensors[f"{base}.scales"] = np.array(s.astype(mx.float16))
                out_tensors[f"{base}.biases"] = np.array(b.astype(mx.float16))
                overrides[base] = {"group_size": gs, "bits": bits, "mode": "affine"}
                quantized += 1
            else:
                out_tensors[k] = np.array(w.astype(mx.float16))
                passthrough += 1
        del arrs

    print(f"  quantized {quantized} @ {bits}-bit gs{gs}, {passthrough} fp16 passthrough")

    # append as a new shard, renaming existing shards is unnecessary
    existing = sorted(glob.glob(str(bundle / "model-*.safetensors")))
    new_name = f"model-mtp-of-{len(existing)+1:05d}.safetensors"
    save_file(out_tensors, str(bundle / new_name))
    nbytes = (bundle / new_name).stat().st_size
    print(f"  wrote {new_name}  ({nbytes/2**20:.1f} MiB, {len(out_tensors)} tensors)")

    idx_p = bundle / "model.safetensors.index.json"
    idx = json.loads(idx_p.read_text())
    for k in out_tensors:
        idx["weight_map"][k] = new_name
    idx.setdefault("metadata", {})
    idx["metadata"]["total_size"] = idx["metadata"].get("total_size", 0) + nbytes
    idx_p.write_text(json.dumps(idx, indent=2))

    # 🚨 Write the per-module overrides under BOTH `mtp.*` and
    # `language_model.mtp.*`.
    #
    # The source tensors are named `mtp.*`, but mlx_vlm's qwen3_5 `sanitize()`
    # relocates them to `language_model.mtp.*`, which is the path the quantized
    # loader uses when it looks up a module's per-module spec. With only the
    # bare `mtp.*` keys the lookup MISSES, the loader falls back to the bundle's
    # DEFAULT bits/group_size, and `load_weights` then dies on a shape mismatch:
    #
    #   ValueError: Expected shape (5120, 1280) but received shape (5120, 1920)
    #               for parameter language_model.mtp.fc.weight
    #
    # i.e. the head is unloadable at any width that differs from the bundle
    # default. MEASURED on the shipped v1 Qwen3.8 bundles too (6-bit gs128 head
    # against a 4-bit gs128 default), so every Qwen3.8 bundle we have published
    # carries an MTP head the runtime cannot instantiate. The bare key is kept
    # as well so older readers keep working.
    aliased = {f"language_model.{k}": dict(v) for k, v in overrides.items()}
    cfg_p = bundle / "config.json"
    cfg = json.loads(cfg_p.read_text())
    for key in ("quantization", "quantization_config"):
        if key in cfg:
            cfg[key].update(overrides)
            cfg[key].update(aliased)
    cfg_p.write_text(json.dumps(cfg, indent=2))
    print(f"  quant overrides: {len(overrides)} mtp.* + "
          f"{len(aliased)} language_model.mtp.* (loader looks up the latter)")

    print(f"  index now {len(idx['weight_map'])} tensors "
          f"({sum(1 for k in idx['weight_map'] if k.startswith('mtp.'))} mtp.*)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
