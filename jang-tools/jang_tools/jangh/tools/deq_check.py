"""Are the expert tensors in the bundle the right experts, decoded the right way? Bundle files vs BF16 source.
A wrong expert order, wrong bit order, wrong rotation or wrong scale gives a relative error near or above 1.0 and cosine near 0."""
import os
import json, sys
from pathlib import Path
import numpy as np, mlx.core as mx
from jang_tools.jangh.format import unpack_bitstream, codebook, h32
B = Path(sys.argv[1]); S = Path(os.environ["JANGH_SOURCE"])
bq = json.loads((B / "config.json").read_text())["quantization"]; bwm = json.loads((B / "model.safetensors.index.json").read_text())["weight_map"]
swm = json.loads((S / "model.safetensors.index.json").read_text())["weight_map"]
files = {}
def src(k):
    f = swm[k]
    if f not in files: files.clear(); files[f] = mx.load(str(S / f))
    return files[f][k].astype(mx.float32)
for L in [int(x) for x in sys.argv[2].split(",")]:
    for p_ in ("gate", "up", "down"):
        n = f"model.layers.{L}.mlp.switch_mlp.{p_}_proj"; spec = bq[n]
        pk = mx.load(str(B / bwm[n + ".tq2_packed"]))[n + ".tq2_packed"]; sc = mx.load(str(B / bwm[n + ".tq2_scales"]))[n + ".tq2_scales"]
        K = pk.shape[-1] * 32 // spec["bits"]; cb = mx.array(codebook(spec["bits"])); rel, cos, cross = [], [], []
        for e in range(0, 288, 9):
            w = cb[unpack_bitstream(pk[e], spec["bits"], K)] * sc[e].astype(mx.float32)[..., None]
            if spec.get("rotation") == "hadamard32": w = h32(w)
            r = src(f"model.language_model.layers.{L}.mlp.experts.{e}.{p_}_proj.weight")
            o = src(f"model.language_model.layers.{L}.mlp.experts.{(e + 1) % 288}.{p_}_proj.weight")     # control: a DIFFERENT expert
            rel.append(float(mx.sqrt(mx.sum((w - r) ** 2) / mx.sum(r ** 2)))); cos.append(float(mx.sum(w * r) / mx.sqrt(mx.sum(w * w) * mx.sum(r * r))))
            cross.append(float(mx.sum(w * o) / mx.sqrt(mx.sum(w * w) * mx.sum(o * o))))
        print(f"L{L:2d} {p_:4s} {spec['bits']}-bit: relative error min {min(rel):.3f} median {np.median(rel):.3f} max {max(rel):.3f} | cosine to its own source expert "
              f"min {min(cos):.3f} median {np.median(cos):.3f} | cosine to a different expert (control) median {np.median(cross):.3f}  [{len(rel)} experts]", flush=True)
