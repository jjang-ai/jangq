"""Exact on-disk size estimate for a Hy3 JANG bundle, from the real policy.

Walks the source ``model.safetensors.index.json`` tensor-by-tensor, classifies
each with the *actual* converter policy (``convert_hy3_jang.classify_tensor``),
and sums the bytes each one will occupy:

  affine-N  : ceil(bits) packed weight + fp16 scales + fp16 biases per group
  passthrough: fp16
  drop      : 0

Reports totals per tensor class so the MTP layer's cost is explicit.

Usage:
  python -m jang_tools.hy3.estimate_size --src /Volumes/EricsLLMDrive/sources/Hy3 \
      [--profile JANG_2L] [--mtp-policy preserve-affine8] [--group-size 128]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from jang_tools.convert_hy3_jang import classify_tensor, profile_policy


def _shape_of(headers: dict, name: str) -> list[int]:
    return headers[name]


def _read_shapes(src: Path) -> dict[str, list[int]]:
    """Shapes for every tensor, read from the safetensors headers only."""
    import struct

    wm = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    shapes: dict[str, list[int]] = {}
    for shard in sorted(set(wm.values())):
        with open(src / shard, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for k, v in hdr.items():
            if k != "__metadata__":
                shapes[k] = v["shape"]
    return shapes


def tensor_bytes(shape: list[int], bits: int, method: str, group_size: int) -> int:
    n = 1
    for d in shape:
        n *= d
    if method == "drop":
        return 0
    if method == "passthrough":
        return n * 2  # fp16
    # affine: packed weights + fp16 scale + fp16 bias per group along last dim
    packed = n * bits / 8.0
    n_groups = n / group_size
    return int(packed + n_groups * 4)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--profile", default="JANG_2L")
    ap.add_argument("--mtp-policy", default=None)
    ap.add_argument("--group-size", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = json.loads((args.src / "config.json").read_text())
    NL = int(cfg["num_hidden_layers"])
    policy = profile_policy(args.profile, args.mtp_policy, mtp_layer_start=NL)
    gs = args.group_size or policy.group_size

    shapes = _read_shapes(args.src)

    def bucket(name: str) -> str:
        if name.startswith(f"model.layers.{NL}."):
            return "MTP layer"
        if ".mlp.experts." in name:
            return "routed experts"
        if ".shared_mlp." in name:
            return "shared experts"
        if "self_attn" in name:
            return "attention"
        if "embed_tokens" in name:
            return "embed_tokens"
        if name.startswith("lm_head"):
            return "lm_head"
        if ".mlp." in name and "router" not in name:
            return "dense FFN (L0)"
        return "router / norms"

    totals: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)
    src_bytes = 0
    for name, shape in shapes.items():
        bits, method = classify_tensor(name, policy)
        b = tensor_bytes(shape, bits, method, gs)
        totals[bucket(name)] += b
        counts[bucket(name)] += 1
        n = 1
        for d in shape:
            n *= d
        src_bytes += n * 2  # bf16 source

    print(f"  Hy3 {policy.profile}  gs={gs}  mtp={policy.mtp_policy}")
    print(f"  source (bf16): {src_bytes / 1e9:.2f} GB\n")
    print(f"  {'class':<18}{'tensors':>9}{'GB':>10}{'%':>8}")
    total = sum(totals.values())
    for k in sorted(totals, key=lambda x: -totals[x]):
        print(f"  {k:<18}{counts[k]:>9}{totals[k]/1e9:>10.2f}"
              f"{totals[k]/total*100:>7.1f}%")
    print(f"  {'-'*45}")
    print(f"  {'TOTAL':<18}{sum(counts.values()):>9}{total/1e9:>10.2f}")
    mtp = totals.get("MTP layer", 0)
    if mtp:
        print(f"\n  without MTP: {(total - mtp)/1e9:.2f} GB "
              f"(MTP costs {mtp/1e9:.2f} GB)")
    print(f"  compression vs bf16 source: {src_bytes/total:.2f}x")


if __name__ == "__main__":
    main()
