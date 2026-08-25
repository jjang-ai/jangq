"""Refit low-bit modules of a Qwen3.6-27B bundle with imatrix-weighted codes.

Replaces the RTN codes of every module at or below ``--max-bits`` with an
activation-weighted affine fit (`quantize_imatrix_affine_numpy`), using the
per-channel second moments from the calibration capture. Same storage ABI —
packed uint32 codes + fp16 scales/biases — so the runtime is completely
unchanged; only the code values improve.

Measured on real tensors at 2-bit gs128 (activation-weighted rel. error):
    30.mlp.gate_proj          0.498 -> 0.285   (-42.8 %)
    10.mlp.down_proj          0.428 -> 0.330   (-23.0 %)
    20.linear_attn.in_proj_qkv 0.353 -> 0.206  (-41.8 %)

This is the imatrix half of what llama.cpp's IQ-series does. Their non-uniform
lattice codebooks have no MLX kernel, so that half is out of scope; this is the
part that transfers. Safe for the GDN hybrid because the fit is strictly
per-tensor — unlike AWQ folding it never touches an adjacent norm or projection
(this family stores norms zero-centered with the runtime applying +1, so
norm-folding is deliberately avoided).

    python -m jang_tools.qwen36_imatrix_refit <src> <calib.safetensors> <bundle> \
        [--max-bits 3] [--group-size 128]
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors.numpy import load_file

from .affine import quantize_imatrix_affine_numpy


def bundle_key(path: str) -> str:
    """calib/module path -> bundle tensor base (post-sanitize namespace)."""
    return path


def source_key(path: str) -> str:
    p = path.replace("language_model.model.", "model.language_model.")
    p = p.replace("language_model.lm_head", "lm_head")
    p = p.replace("vision_tower.", "model.visual.")
    return p + ".weight"


def main(argv) -> int:
    if len(argv) < 4:
        print(__doc__)
        return 1
    src, calib_p, bundle = Path(argv[1]), Path(argv[2]), Path(argv[3])
    max_bits, gs = 3, 128
    awq_scales_p = None
    for i, a in enumerate(argv):
        if a == "--max-bits":
            max_bits = int(argv[i + 1])
        if a == "--group-size":
            gs = int(argv[i + 1])
        # 🚨 REQUIRED whenever the bundle was built with AWQ. This refit
        # re-derives codes from the ORIGINAL SOURCE weight (see the W = ...
        # load below), so without re-applying the AWQ scales it silently
        # reverts W*s -> W while the bundle's norms stay divided by s, leaving
        # every folded layer off by a factor of s per channel. Tell-tale: the
        # rel-err printed here is byte-identical with and without AWQ.
        if a == "--awq-scales":
            awq_scales_p = argv[i + 1]

    calib = load_file(str(calib_p))
    awq_s = {}
    if awq_scales_p:
        awq_s = {k[: -len(".awq_scale")]: v
                 for k, v in load_file(awq_scales_p).items()
                 if k.endswith(".awq_scale")}
        print(f"  AWQ scales loaded for {len(awq_s)} modules "
              f"(re-applied to the source weight before fitting)")
    cfg = json.loads((bundle / "config.json").read_text())
    if cfg.get("awq") and not awq_s:
        print("  !! bundle config records an AWQ fold but no --awq-scales was "
              "given: refitting from unscaled source would BREAK the fold "
              "(norms stay /s, weights revert). Refusing.")
        return 2
    q = cfg["quantization"]
    targets = {k: v for k, v in q.items()
               if isinstance(v, dict) and v.get("bits", 99) <= max_bits}
    print(f"  modules at <= {max_bits}-bit: {len(targets)}")

    src_wm = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    bun_wm = json.loads((bundle / "model.safetensors.index.json").read_text())["weight_map"]

    # group work by bundle shard so each shard is rewritten once
    by_shard: dict[str, list[str]] = {}
    for path in targets:
        wkey = f"{path}.weight"
        if wkey in bun_wm:
            by_shard.setdefault(bun_wm[wkey], []).append(path)

    t0 = time.time()
    done, err_before_after = 0, []
    for shard, paths in sorted(by_shard.items()):
        shard_p = bundle / shard
        # mx.load is LAZY (mmap-backed). Saving to the same path while arrays
        # still reference the mapping corrupts every untouched tensor — proven:
        # a norm in a rewritten shard read back all-zeros. Materialize first,
        # then write to a temp file and atomically rename.
        tensors = dict(mx.load(str(shard_p)))
        mx.eval(list(tensors.values()))
        for path in paths:
            bits = targets[path]["bits"]
            g = targets[path]["group_size"]
            skey = source_key(path)
            if skey not in src_wm:
                continue
            imp_key = f"{path}.second_moment"
            if imp_key not in calib:
                continue
            W = np.array(mx.load(str(src / src_wm[skey]))[skey].astype(mx.float32))
            # Re-apply the AWQ column scaling that the build folded in — this
            # load came from the SOURCE, which knows nothing about it.
            s_awq = awq_s.get(path)
            if s_awq is not None and s_awq.shape[0] == W.shape[-1]:
                W = W * s_awq[None, :].astype(np.float32)
            imp = calib[imp_key].astype(np.float32)
            packed, scales, biases, werr = quantize_imatrix_affine_numpy(
                W, imp, bits=bits, group_size=g)
            tensors[f"{path}.weight"] = mx.array(packed)
            tensors[f"{path}.scales"] = mx.array(scales).astype(mx.float16)
            tensors[f"{path}.biases"] = mx.array(biases).astype(mx.float16)
            err_before_after.append(werr)
            done += 1
            if done % 40 == 0:
                print(f"    {done}/{len(targets)}  ({time.time()-t0:.0f}s)",
                      flush=True)
        tmp = shard_p.with_suffix(".tmp.safetensors")
        mx.save_safetensors(str(tmp), tensors)
        tmp.replace(shard_p)
        print(f"  rewrote {shard} ({len(paths)} modules)", flush=True)

    jang_p = bundle / "jang_config.json"
    if jang_p.exists():
        jang = json.loads(jang_p.read_text())
        qz = jang.setdefault("quantization", {})
        qz["imatrix_refit"] = {
            "modules": done, "max_bits": max_bits,
            "calibration": calib_p.name,
            "method": "activation-weighted affine fit (quantize_imatrix_affine_numpy)",
        }
        jang_p.write_text(json.dumps(jang, indent=2) + "\n")

    print(f"\n  refit {done} modules in {time.time()-t0:.0f}s")
    if err_before_after:
        print(f"  mean weighted rel-err after fit: "
              f"{float(np.mean(err_before_after)):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
