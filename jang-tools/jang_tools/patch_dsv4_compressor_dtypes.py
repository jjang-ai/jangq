"""Post-publication patcher for DSV4 JANGTQ bundles built before the
Compressor + Indexer per-tensor quant policy was correct (P2 of the
HSA+CSA+SWA plan).

Older bundles store these as 8-bit affine:
  - layers.N.attn.indexer.weights_proj.{weight,scales,biases}
  - layers.N.attn.indexer.compressor.wkv.{weight,scales,biases}
  - layers.N.attn.compressor.wkv.{weight,scales,biases}

The runtime now expects them as bf16/fp16 passthrough (`<base>.weight` only,
no scales/biases siblings). This tool dequantizes those tensors back to
fp16 IN-PLACE on a bundle directory, removes the .scales/.biases sidecars,
and updates `model.safetensors.index.json`.

Cannot recover the F32 source dtype for `hc_*_{fn,base,scale}`, `attn_sink`,
`gate.bias` from a pre-existing fp16 bundle — those need a re-convert from
the bf16 source. This patcher is for the Compressor + Indexer fix only.

Usage:
    python -m jang_tools.patch_dsv4_compressor_dtypes <bundle> [--dry-run]
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

import numpy as np
import mlx.core as mx
from safetensors import safe_open
from safetensors.numpy import save_file as st_save


PATCHED_PATTERNS = [
    re.compile(r"^model\.layers\.\d+\.self_attn\.(indexer\.)?compressor\.wkv\.weight$"),
    re.compile(r"^model\.layers\.\d+\.self_attn\.indexer\.weights_proj\.weight$"),
]
def is_patched_target(key: str) -> bool:
    return any(p.match(key) for p in PATCHED_PATTERNS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--bits", type=int, default=8, help="bits the old bundle quantized at")
    ap.add_argument("--group-size", type=int, default=32)
    args = ap.parse_args()

    src = Path(args.src)
    idx_path = src / "model.safetensors.index.json"
    idx = json.loads(idx_path.read_text())
    wm = idx["weight_map"]

    by_shard: dict = {}
    for k, sh in wm.items():
        by_shard.setdefault(sh, []).append(k)

    # Find every patched target tensor. Each has a paired .scales/.biases.
    targets = [k for k in wm if k.endswith(".weight") and is_patched_target(k)]
    print(f"  found {len(targets)} target weight tensors to dequantize")
    if not targets:
        print("  nothing to patch."); return

    affected_shards = sorted({wm[k] for k in targets} |
                             {wm.get(k.replace(".weight", ".scales"), "") for k in targets} |
                             {wm.get(k.replace(".weight", ".biases"), "") for k in targets} - {""})
    print(f"  affected shards: {len(affected_shards)}")
    if args.dry_run:
        for k in sorted(targets)[:10]:
            print(f"    {k}")
        if len(targets) > 10: print(f"    ... ({len(targets)-10} more)")
        return

    dropped: list = []
    for shard_name in affected_shards:
        shard_path = src / shard_name
        new_tensors: dict = {}
        with safe_open(str(shard_path), framework="numpy") as f:
            shard_keys = list(f.keys())
            for k in shard_keys:
                base = k.rsplit(".", 1)[0]
                if k.endswith(".weight") and is_patched_target(k):
                    qw = f.get_tensor(k)
                    sk = base + ".scales"
                    bk = base + ".biases"
                    if sk in wm and wm[sk] in (shard_name,):
                        with safe_open(str(src / wm[sk]), framework="numpy") as g:
                            sc = g.get_tensor(sk)
                    else:
                        with safe_open(str(src / wm[sk]), framework="numpy") as g:
                            sc = g.get_tensor(sk)
                    if bk in wm:
                        with safe_open(str(src / wm[bk]), framework="numpy") as g:
                            bi = g.get_tensor(bk)
                    else:
                        bi = None
                    dq = mx.dequantize(
                        mx.array(qw, dtype=mx.uint32),
                        mx.array(sc),
                        mx.array(bi) if bi is not None else None,
                        group_size=args.group_size, bits=args.bits,
                    )
                    new_tensors[k] = np.array(dq).astype(np.float16)
                elif k.endswith(".scales") or k.endswith(".biases"):
                    base = k.rsplit(".", 1)[0]
                    weight_key = base + ".weight"
                    if weight_key in wm and is_patched_target(weight_key):
                        dropped.append(k); continue
                    new_tensors[k] = f.get_tensor(k)
                else:
                    new_tensors[k] = f.get_tensor(k)
        st_save(new_tensors, str(shard_path))
        print(f"  rewrote {shard_name}: {len(new_tensors)} tensors")

    new_wm = {k: v for k, v in wm.items() if k not in dropped}
    idx["weight_map"] = new_wm
    total = sum((src / fn).stat().st_size for fn in set(new_wm.values()) if (src / fn).exists())
    idx["metadata"]["total_size"] = total
    idx_path.write_text(json.dumps(idx, indent=2))
    print(f"\n  dropped {len(dropped)} sidecars (.scales/.biases for patched modules)")
    print(f"  new total_size = {total/1e9:.2f} GB")
    print("DONE")


if __name__ == "__main__":
    main()
