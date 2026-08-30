"""Prepare converter-ready calibration artifacts from the pod capture.

Inputs:  diag.safetensors (pod, HF naming) + the bf16 source (for BRECQ
         down-proj expert diags computed from the routed reservoirs).
Outputs: glm53_imatrix.safetensors  — runtime-named .diag / .expert_diag
         glm53_awq.safetensors     — per-mlp-site input scales s[4096]

AWQ recipe (dots3/qwen38 standard): s = clip((rms / gmean(rms))^0.25, 0.5, 2)
computed on the MoE-site input rms (per channel, from sum_x2/count).

  python -m jang_tools.glm5_next.prep_calib --diag diag.safetensors \
      --model <bf16 src> --out-imatrix ... --out-awq ...
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import mlx.core as mx
import numpy as np


def runtime_name(hf_mod: str) -> str:
    m = hf_mod.replace("model.language_model.", "model.")
    m = m.replace(".self_attn.forget_gate.", ".self_attn.")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-imatrix", required=True)
    ap.add_argument("--out-awq", required=True)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--clip", type=float, nargs=2, default=(0.5, 2.0))
    args = ap.parse_args()

    d = mx.load(args.diag)
    model_dir = Path(args.model)
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]

    out_im: dict = {}
    out_awq: dict = {}

    mods = sorted({k[: k.rfind(".")] for k in d if k.endswith(".sum_x2")})
    for mod in mods:
        s2 = d[mod + ".sum_x2"].astype(mx.float32)
        cnt = float(np.asarray(d[mod + ".count"]).reshape(-1)[0])
        diag = s2 / max(cnt, 1.0)
        rn = runtime_name(mod)
        if rn.endswith(".mlp.experts"):
            site = rn[: -len(".experts")]                      # model.layers.N.mlp
            # gate/up per-expert diag (shared input, expert-bucketed stats)
            es2 = d[mod + ".expert_sum_x2"].astype(mx.float32)  # [288, 4096]
            ec = mx.maximum(d[mod + ".expert_count"].astype(mx.float32), 1.0)
            ed = es2 / ec[:, None]
            out_im[site + ".switch_mlp.gate_proj.expert_diag"] = ed
            out_im[site + ".switch_mlp.up_proj.expert_diag"] = ed
            # AWQ site scale from the site-level input rms
            rms = mx.sqrt(mx.maximum(diag, 1e-12))
            log_g = mx.mean(mx.log(rms))
            s = mx.clip((rms / mx.exp(log_g)) ** args.alpha,
                        args.clip[0], args.clip[1])
            out_awq[site] = s.astype(mx.float32)
            # BRECQ down diag from the routed reservoir through bf16 gate/up
            rows = d[mod + ".rows"].astype(mx.float32)          # [R, 4096]
            tidx = np.asarray(d[mod + ".rows_topk_idx"])        # [R, 8]
            layer = int(re.search(r"layers\.(\d+)\.", mod).group(1))
            dd = _down_expert_diag(model_dir, wm, layer, rows, tidx)
            out_im[site + ".switch_mlp.down_proj.expert_diag"] = dd
        else:
            out_im[rn + ".diag"] = diag

    mx.save_safetensors(args.out_imatrix, out_im)
    mx.save_safetensors(args.out_awq, out_awq)
    print(f"imatrix: {len(out_im)} entries → {args.out_imatrix}")
    print(f"awq: {len(out_awq)} sites → {args.out_awq}")


def _down_expert_diag(model_dir, wm, layer: int, rows: mx.array, tidx: np.ndarray,
                      limit: float = 10.0) -> mx.array:
    """Per-expert mean h² of the down_proj input, h = clamped_swiglu(gate, up),
    on the reservoir rows routed to each expert. [E, 2048]."""
    E = 288
    diag = mx.zeros((E, 2048), dtype=mx.float32)
    gate_w = {}
    cache: dict = {}

    def get(name):
        f = wm[name]
        if f not in cache:
            cache.clear()
            cache[f] = mx.load(str(model_dir / f))
        return cache[f][name]

    counts = np.zeros(E)
    for e in range(E):
        rmask = np.nonzero((tidx == e).any(axis=1))[0]
        if rmask.size < 4:
            continue
        base = f"model.language_model.layers.{layer}.mlp.experts.{e}"
        Wg = get(base + ".gate_proj.weight").astype(mx.float32)
        Wu = get(base + ".up_proj.weight").astype(mx.float32)
        xe = rows[mx.array(rmask.astype(np.uint32))]
        g = mx.minimum(xe @ Wg.T, limit)
        u = mx.clip(xe @ Wu.T, -limit, limit)
        h = (g * mx.sigmoid(g)) * u
        diag[e] = mx.mean(h * h, axis=0)
        counts[e] = rmask.size
        if e % 32 == 0:
            mx.eval(diag)
            mx.clear_cache()
    # experts with too few rows: fall back to the mean diag of covered experts
    mx.eval(diag)
    covered = counts >= 4
    if covered.any() and (~covered).any():
        mean_d = mx.mean(diag[mx.array(np.nonzero(covered)[0].astype(np.uint32))], axis=0)
        for e in np.nonzero(~covered)[0]:
            diag[int(e)] = mean_d
    return diag


if __name__ == "__main__":
    main()
