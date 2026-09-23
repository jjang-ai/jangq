"""Compute AWQ per-site scales for qwen4_exp from the calibration capture.

s_c = (amax_c / geomean(amax))^alpha, clipped to [1/clip, clip]. The site's
input amax is read from a designated consumer module (its input IS the mixed
block input). Emits site-path → s [hidden] for convert.py --awq-scales.

  python -m jang_tools.qwen4_exp.awq_scales --capture .../capture \
      --out .../awq_scales.safetensors [--alpha 0.5] [--clip 8.0]
"""

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np

N_LAYERS = 48


def site_scale(amax: np.ndarray, alpha: float, clip: float) -> np.ndarray:
    a = np.maximum(amax.astype(np.float64), 1e-8)
    g = np.exp(np.log(a).mean())
    s = (a / g) ** alpha
    return np.clip(s, 0.5, clip).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--clip", type=float, default=2.0)
    args = ap.parse_args()

    diag = mx.load(str(Path(args.capture) / "diag.safetensors"))
    out = {}
    missing = []
    for l in range(N_LAYERS):
        lp = f"language_model.layers.{l}"
        # attn site: GDN layers expose in_proj_qkv; QSA layers expose q_proj
        for probe in (f"{lp}.linear_attn.in_proj_qkv.amax", f"{lp}.self_attn.q_proj.amax"):
            if probe in diag:
                out[f"{lp}.attn"] = mx.array(
                    site_scale(np.asarray(diag[probe]), args.alpha, args.clip))
                break
        else:
            missing.append(f"{lp}.attn")
        # mlp site: shared_expert.gate_proj input == mixed block input
        probe = f"{lp}.mlp.shared_expert.gate_proj.amax"
        if probe in diag:
            out[f"{lp}.mlp"] = mx.array(
                site_scale(np.asarray(diag[probe]), args.alpha, args.clip))
        else:
            missing.append(f"{lp}.mlp")
    if "lm_head.amax" in diag:
        out["final"] = mx.array(site_scale(np.asarray(diag["lm_head.amax"]), args.alpha, args.clip))

    if missing:
        print(f"WARNING: no capture stats for {len(missing)} sites (skipped): {missing[:4]}...")
    mx.save_safetensors(args.out, out)
    print(f"wrote {len(out)} site scales → {args.out}")


if __name__ == "__main__":
    main()
