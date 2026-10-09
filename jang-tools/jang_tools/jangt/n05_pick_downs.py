"""Copy only the JANGH down_proj tensors an allocation needs into one stage dir (input to n05_assemble --pick).

  --todo   json {"have": [L...], "conv": [[L, ..., bits], ...]}: layers already in --have-dir, layers re-converted into
           --conv-dir (jang_tools.jangh.n05.convert --keep-stage writes <out>/.stage/L##.safetensors)
  --alloc  allocation json + --key ("97.55:free"); every copied down must match the plan's jh<bits> format (asserted)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--todo", required=True); ap.add_argument("--have-dir", required=True)
    ap.add_argument("--conv-dir", required=True); ap.add_argument("--alloc", required=True)
    ap.add_argument("--key", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--k-in", type=int, default=2048, help="down_proj input width (moe_intermediate_size)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t = json.loads(Path(a.todo).read_text())
    picks = {L: Path(a.have_dir) for L in t["have"]}
    picks.update({c[0]: Path(a.conv_dir) for c in t["conv"]})
    fmt = json.loads(Path(a.alloc).read_text())[a.key]["formats"]
    for L, d in sorted(picks.items()):
        sf = mx.load(str(d / f"L{L:02d}.safetensors")); m = f"model.layers.{L}.mlp.switch_mlp.down_proj"
        t2 = {m + ".tq2_packed": sf[m + ".tq2_packed"], m + ".tq2_scales": sf[m + ".tq2_scales"]}
        bits = t2[m + ".tq2_packed"].shape[-1] * 32 // a.k_in
        assert f"jh{bits}" == fmt[f"{L}:down"], (L, bits, fmt[f"{L}:down"])
        tmp = out / f".L{L:02d}.tmp.safetensors"; mx.save_safetensors(str(tmp), t2); tmp.replace(out / f"L{L:02d}.safetensors")
        print(f"L{L} down {bits}-bit from {d}", flush=True)


if __name__ == "__main__":
    main()
