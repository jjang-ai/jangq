"""Structural audit of a JANGTQ v2 GLM bundle: per-module config vs shard tensors, TQ shapes vs bits,
mxfp8/affine component shapes, alignment, root layout (only model-* + ONE index), no MTP tensors."""
import json, re, sys, struct
from pathlib import Path

from jang_tools.format.aligned_safetensors import verify_safetensors_alignment

def main(b, complete=True):
    d = Path(b); cfg = json.loads((d / "config.json").read_text()); q = cfg["quantization"]
    idx = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    shapes = {}
    for f in sorted(set(idx.values())):
        with open(d / f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n))
        shapes.update({k: (v["dtype"], v["shape"]) for k, v in h.items() if k != "__metadata__"})
        t, bad = verify_safetensors_alignment(d / f)
        assert bad == 0, f"{f}: {bad} unaligned"
    errs = []
    root_st = [p.name for p in d.glob("*.safetensors")]
    if any(not n.startswith("model-") for n in root_st): errs.append(f"non-model safetensors in root: {root_st}")
    if len(list(d.glob("*index.json"))) != 1: errs.append("root must contain exactly one *index.json")
    if any(re.match(r"model\.layers\.45\.", k) for k in shapes): errs.append("MTP tensors present")
    for mod, spec in q.items():
        if not isinstance(spec, dict): continue
        if spec.get("mode") == "jangtq2":
            p, s = shapes.get(mod + ".tq2_packed"), shapes.get(mod + ".tq2_scales")
            if p is None or s is None:
                if complete: errs.append(f"{mod}: missing tq tensors")
                continue
            E, N, W = p[1]; K = W * 32 // spec["bits"]
            exp_k = 4096 if ("gate_proj" in mod or "up_proj" in mod) else 2048
            if p[0] != "U32" or s[0] != "F16" or s[1] != [E, N] or K != exp_k: errs.append(f"{mod}: bad tq shapes {p} {s} bits={spec['bits']}")
        elif spec.get("mode") == "mxfp8":
            w, s = shapes.get(mod + ".weight"), shapes.get(mod + ".scales")
            if w is None or s is None or w[0] != "U32" or s[0] != "U8" or s[1][1] * 32 != w[1][1] * 4: errs.append(f"{mod}: bad mxfp8 {w} {s}")
            if mod + ".biases" in shapes: errs.append(f"{mod}: mxfp8 must not have biases")
        else:
            w, s, bi = shapes.get(mod + ".weight"), shapes.get(mod + ".scales"), shapes.get(mod + ".biases")
            if w is None or s is None or bi is None or s[0] != "BF16" or bi[0] != "BF16" or s[1][1] * 64 != w[1][1] * 32 // spec["bits"]:
                errs.append(f"{mod}: bad affine {w} {s} {bi}")
    n_tq = sum(1 for v in q.values() if isinstance(v, dict) and v.get("mode") == "jangtq2")
    total = sum(__import__("os").path.getsize(d / f) for f in set(idx.values()))
    print(f"tensors {len(shapes)}  tq modules {n_tq}  shards {len(set(idx.values()))}  bytes {total/2**30:.3f} GiB  errors {len(errs)}")
    for e in errs[:30]: print("  ", e)
    return not errs

if __name__ == "__main__":
    ok = main(sys.argv[1], complete=(len(sys.argv) < 3 or sys.argv[2] != "--partial"))
    sys.exit(0 if ok else 1)
