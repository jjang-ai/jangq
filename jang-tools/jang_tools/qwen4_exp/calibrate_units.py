"""Per-unit quantization option measurement for qwen4_exp routed experts.

Adaptation of jang_tools.dots3.calibrate_units: for each (layer, projection)
routed unit, sample experts, quantize at each candidate option, and measure
ROUTER-WEIGHTED output NMSE on the rows that actually routed to that expert
(from the capture's moe_rows reservoir). down_proj inputs are derived through
the SOURCE gate/up (BRECQ-consistent with the later GPTQ sequencing).

Output: unit_scores.json keyed "layer:proj" — consumed UNCHANGED by
jang_tools.dots3.solve_plan (convex-hull MCKP).

  python -m jang_tools.qwen4_exp.calibrate_units --model <src> \
      --capture ~/models/Logs/q38fn-calib/capture --out unit_scores.json \
      [--experts 24] [--layers all]
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

OPTIONS = [
    ("2b_g64", dict(bits=2, group_size=64)),
    ("2b_g32", dict(bits=2, group_size=32)),
    ("3b_g64", dict(bits=3, group_size=64)),
    ("4b_g64", dict(bits=4, group_size=64)),
    ("mxfp4_g32", dict(bits=4, group_size=32, mode="mxfp4")),  # equal-bytes A/B (dots3 directive)
    ("6b_g64", dict(bits=6, group_size=64)),
    ("8b_g64", dict(bits=8, group_size=64)),
]


def eff_bytes(n_weights: int, bits: int, gs: int, mode: str = "affine") -> int:
    if mode == "mxfp4":
        return int(n_weights * (bits + 8 / gs) / 8)   # e8m0 scale only
    return int(n_weights * (bits + 32 / gs) / 8)      # f16 scale + bias


def qdq(w: mx.array, bits: int, group_size: int, mode: str = "affine") -> mx.array:
    if mode == "mxfp4":
        wq, s = mx.quantize(w, group_size=group_size, bits=bits, mode="mxfp4")
        return mx.dequantize(wq, s, group_size=group_size, bits=bits, mode="mxfp4").astype(mx.float32)
    wq, s, b = mx.quantize(w, group_size=group_size, bits=bits)
    s = s.astype(mx.float16).astype(mx.float32)
    b = b.astype(mx.float16).astype(mx.float32)
    return mx.dequantize(wq, s, b, group_size=group_size, bits=bits)


def load_expert_weights(model_dir: Path, layer: int):
    """Returns (gate_up [E,1280,2560], down [E,2560,640]) lazily."""
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    gu_key = f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"
    dn_key = f"model.language_model.layers.{layer}.mlp.experts.down_proj"
    gu = mx.load(str(model_dir / wm[gu_key]))[gu_key]
    dn = mx.load(str(model_dir / wm[dn_key]))[dn_key]
    return gu, dn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--experts", type=int, default=24, help="sampled experts per layer")
    ap.add_argument("--layers", default="all")
    args = ap.parse_args()

    model_dir = Path(args.model)
    rows = mx.load(str(Path(args.capture) / "moe_rows.safetensors"))
    layer_keys = sorted(
        {k[: -len(".rows_x")] for k in rows if k.endswith(".rows_x")},
        key=lambda p: int(p.split(".layers.")[1].split(".")[0]),
    )
    if args.layers != "all":
        want = {int(x) for x in args.layers.split(",")}
        layer_keys = [k for k in layer_keys if int(k.split(".layers.")[1].split(".")[0]) in want]

    out = {}
    t0 = time.time()
    for lk in layer_keys:
        layer = int(lk.split(".layers.")[1].split(".")[0])
        X = rows[lk + ".rows_x"].astype(mx.float32)          # [R, 2560]
        inds = np.asarray(rows[lk + ".rows_inds"])            # [R, k]
        W = np.asarray(rows[lk + ".rows_w"].astype(mx.float32))  # [R, k]

        # sample experts: most-routed first, then even coverage
        counts = np.bincount(inds.ravel(), minlength=512)
        by_freq = np.argsort(-counts)
        sampled = list(by_freq[: args.experts // 2])
        sampled += list(np.linspace(0, 511, args.experts - len(sampled)).astype(int))
        sampled = sorted(set(int(e) for e in sampled if counts[e] > 0))

        gu, dn = load_expert_weights(model_dir, layer)
        h = gu.shape[1] // 2

        errs = {name: {"gate_proj": [0.0, 0.0], "up_proj": [0.0, 0.0],
                       "down_proj": [0.0, 0.0]} for name, _ in OPTIONS}
        for e in sampled:
            rmask, slot = np.nonzero(inds == e)
            if rmask.size < 8:
                continue
            xe = X[mx.array(rmask.astype(np.uint32))]                    # [r, 2560]
            we = mx.array(W[rmask, slot])[:, None]                       # router weight
            Wg = gu[e, :h, :].astype(mx.float32)
            Wu = gu[e, h:, :].astype(mx.float32)
            Wd = dn[e].astype(mx.float32)
            g_ref = xe @ Wg.T
            u_ref = xe @ Wu.T
            a_ref = mx.sigmoid(g_ref) * g_ref * u_ref                    # silu(g)*u
            d_ref = a_ref @ Wd.T
            for name, spec in OPTIONS:
                gq = xe @ qdq(Wg, **spec).T
                uq = xe @ qdq(Wu, **spec).T
                dq_ = a_ref @ qdq(Wd, **spec).T
                for proj, ref, got in (("gate_proj", g_ref, gq),
                                       ("up_proj", u_ref, uq),
                                       ("down_proj", d_ref, dq_)):
                    num = float((we * (got - ref) ** 2).sum())
                    den = float((we * ref ** 2).sum())
                    errs[name][proj][0] += num
                    errs[name][proj][1] += max(den, 1e-12)
            mx.clear_cache()

        n_w = {"gate_proj": 512 * h * 2560, "up_proj": 512 * h * 2560,
               "down_proj": 512 * 2560 * 640}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            unit = f"{layer}:{proj}"
            out[unit] = {"n_weights": n_w[proj], "options": {}}
            for name, spec in OPTIONS:
                num, den = errs[name][proj]
                out[unit]["options"][name] = {
                    "nmse": num / max(den, 1e-12),
                    "bytes": eff_bytes(n_w[proj], spec["bits"], spec["group_size"],
                                       spec.get("mode", "affine")),
                }
        del gu, dn
        mx.clear_cache()
        print(f"layer {layer}: {len(sampled)} experts measured "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)

    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {len(out)} units → {args.out}")


if __name__ == "__main__":
    main()
