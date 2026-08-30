"""Per-unit quantization option measurement for glm5_next routed experts.

Unit = (layer, projection) over the 288 stacked experts. For each sampled
expert: quantize at each candidate option and measure ROUTER-WEIGHTED output
NMSE on the reservoir rows that actually routed there (pod capture).
down_proj inputs go through the bf16 gate/up with the clamped swiglu
(BRECQ-consistent).

Output: unit_scores.json keyed "layer:proj" — same schema as qwen4_exp, so
the heap-MCKP solver is reused unchanged.

  python -m jang_tools.glm5_next.calibrate_units --model <bf16 src> \
      --diag <pod diag.safetensors> --out unit_scores.json [--experts 24]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

OPTIONS = [
    ("2b_g128", dict(bits=2, group_size=128)),
    ("2b_g64", dict(bits=2, group_size=64)),
    ("3b_g64", dict(bits=3, group_size=64)),
    ("4b_g64", dict(bits=4, group_size=64)),
    ("6b_g64", dict(bits=6, group_size=64)),
]
E, D_IN, D_H = 288, 4096, 2048
LIMIT = 10.0


def eff_bytes(n_weights: int, bits: int, gs: int) -> int:
    return int(n_weights * (bits + 32 / gs) / 8)


def qdq(w: mx.array, bits: int, group_size: int) -> mx.array:
    wq, s, b = mx.quantize(w, group_size=group_size, bits=bits)
    s = s.astype(mx.float16).astype(mx.float32)
    b = b.astype(mx.float16).astype(mx.float32)
    return mx.dequantize(wq, s, b, group_size=group_size, bits=bits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--diag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--experts", type=int, default=24)
    ap.add_argument("--layers", default="all")
    args = ap.parse_args()

    model_dir = Path(args.model)
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    d = mx.load(args.diag)

    moe_mods = sorted(
        {k[: -len(".rows")] for k in d if k.endswith(".mlp.experts.rows")},
        key=lambda p: int(re.search(r"layers\.(\d+)\.", p).group(1)))
    if args.layers != "all":
        want = {int(x) for x in args.layers.split(",")}
        moe_mods = [m for m in moe_mods
                    if int(re.search(r"layers\.(\d+)\.", m).group(1)) in want]

    cache: dict = {}

    def get(name):
        f = wm[name]
        if f not in cache:
            cache.clear()
            mx.clear_cache()
            cache[f] = mx.load(str(model_dir / f))
        return cache[f][name]

    out = {}
    t0 = time.time()
    for mod in moe_mods:
        layer = int(re.search(r"layers\.(\d+)\.", mod).group(1))
        X = d[mod + ".rows"].astype(mx.float32)          # [R, 4096]
        tidx = np.asarray(d[mod + ".rows_topk_idx"])     # [R, 8]
        tw = np.asarray(d[mod + ".rows_topk_w"])         # [R, 8]

        counts = np.bincount(tidx.ravel(), minlength=E)
        by_freq = np.argsort(-counts)
        sampled = list(by_freq[: args.experts // 2])
        sampled += list(np.linspace(0, E - 1, args.experts - len(sampled)).astype(int))
        sampled = sorted({int(e) for e in sampled if counts[e] >= 8})

        errs = {name: {p: [0.0, 0.0] for p in ("gate_proj", "up_proj", "down_proj")}
                for name, _ in OPTIONS}
        for e in sampled:
            rmask, slot = np.nonzero(tidx == e)
            base = f"model.language_model.layers.{layer}.mlp.experts.{e}"
            Wg = get(base + ".gate_proj.weight").astype(mx.float32)
            Wu = get(base + ".up_proj.weight").astype(mx.float32)
            Wd = get(base + ".down_proj.weight").astype(mx.float32)
            xe = X[mx.array(rmask.astype(np.uint32))]
            we = mx.array(tw[rmask, slot].astype(np.float32))[:, None]
            g_ref = xe @ Wg.T
            u_ref = xe @ Wu.T
            gc = mx.minimum(g_ref, LIMIT)
            uc = mx.clip(u_ref, -LIMIT, LIMIT)
            a_ref = (gc * mx.sigmoid(gc)) * uc
            dn_ref = a_ref @ Wd.T
            for name, spec in OPTIONS:
                gq = xe @ qdq(Wg, **spec).T
                uq = xe @ qdq(Wu, **spec).T
                dq_ = a_ref @ qdq(Wd, **spec).T
                for proj, ref, got in (("gate_proj", g_ref, gq),
                                       ("up_proj", u_ref, uq),
                                       ("down_proj", dn_ref, dq_)):
                    num = float((we * (got - ref) ** 2).sum())
                    den = float((we * ref ** 2).sum())
                    errs[name][proj][0] += num
                    errs[name][proj][1] += max(den, 1e-12)
            mx.clear_cache()

        n_w = {"gate_proj": E * D_H * D_IN, "up_proj": E * D_H * D_IN,
               "down_proj": E * D_IN * D_H}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            unit = f"{layer}:{proj}"
            out[unit] = {"n_weights": n_w[proj], "options": {}}
            for name, spec in OPTIONS:
                num, den = errs[name][proj]
                out[unit]["options"][name] = {
                    "nmse": num / max(den, 1e-12),
                    "bytes": eff_bytes(n_w[proj], spec["bits"], spec["group_size"]),
                }
        print(f"layer {layer}: {len(sampled)} experts measured "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)

    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {len(out)} units → {args.out}")


if __name__ == "__main__":
    main()
