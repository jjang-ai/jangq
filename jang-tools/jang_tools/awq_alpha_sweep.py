"""Pick the AWQ alpha per bit width by measuring, not by inheriting 0.25.

alpha=0.25 is the canonical AWQ value and our measured kappa says it is a clear
win at 4-bit and above (0.827 -> 0.818). At 2-bit kappa is 0.899 -- AWQ helps
least exactly where quantization hurts most -- and the Ornith-397B run measured
alpha=0.25 producing garbage at 2.15-bit with alpha=0.05 a near-no-op. The 2.6
bpw tier is dominated by 2- and 3-bit modules, so the value is worth measuring
rather than assuming.

Metric is the ACTIVATION-WEIGHTED relative error (plain Frobenius is the
quantity AWQ deliberately trades away and scores it backwards -- see
02-DEFECTS-FOUND.md).

    err = || (q(W)-W) diag(sqrt(E[x^2])) ||_F / || W diag(sqrt(E[x^2])) ||_F

Scales are recomputed from the capture at each alpha with the same formula
ornith_awq_fold uses (s = mag^alpha, renormalised to geometric mean 1, clipped
to [1e-2, 1e2]), so this measures the real knob and not an approximation.
"""
import glob
import json
import struct

import mlx.core as mx
import numpy as np
from safetensors.numpy import load_file

CAL = load_file('/Users/eric/models/Logs/q38s-calib/calib.safetensors')
SM = {k[:-len('.second_moment')]: v
      for k, v in CAL.items() if k.endswith('.second_moment')}

# norm-fed modules only -- these are the ones AWQ can reach at all
SUF = ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
       "mlp.gate_proj", "mlp.up_proj", "self_attn.q_proj", "self_attn.v_proj")
want = []
for L in (1, 7, 15, 23, 31, 39, 47, 55, 63):
    for s in SUF:
        p = f"language_model.model.layers.{L}.{s}"
        if p in SM:
            want.append(p)


def src_key(p):
    return p.replace("language_model.model.", "model.language_model.") + ".weight"


need = {src_key(p) for p in want}
W = {}
for sh in sorted(glob.glob('/Users/eric/models/Qwen3.8-27B/model-*.safetensors')):
    with open(sh, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n))
    hit = [k for k in hdr if k in need]
    if not hit:
        continue
    a = mx.load(sh)
    for k in hit:
        W[k] = a[k].astype(mx.float32)
    del a
want = [p for p in want if src_key(p) in W]
print(f"sampling {len(want)} norm-fed modules across 9 layers\n")


def scale_for(p, alpha):
    mag = np.sqrt(np.maximum(SM[p], 0.0)) + 1e-8
    s = np.power(mag, alpha)
    s = s / np.exp(np.mean(np.log(s)))
    return np.clip(s, 1e-2, 1e2).astype(np.float32)


def werr(dq, w, d):
    return float((mx.sqrt((((dq - w) * d[None, :]) ** 2).sum())
                  / mx.sqrt(((w * d[None, :]) ** 2).sum())).item())


ALPHAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50]
BITS = [2, 3, 4, 5, 6, 8]
table = {}
print(f"{'bits':>5} " + " ".join(f"a={a:<5.2f}" for a in ALPHAS) + "   best")
for b in BITS:
    row = []
    for alpha in ALPHAS:
        errs = []
        for p in want:
            w = W[src_key(p)]
            d = mx.array(np.sqrt(np.maximum(SM[p], 0)).astype(np.float32))
            if alpha == 0.0:
                q, sc, z = mx.quantize(w, group_size=128, bits=b)
                dq = mx.dequantize(q, sc, z, group_size=128, bits=b)
            else:
                s = mx.array(scale_for(p, alpha))
                q, sc, z = mx.quantize(w * s[None, :], group_size=128, bits=b)
                dq = mx.dequantize(q, sc, z, group_size=128, bits=b) / s[None, :]
            errs.append(werr(dq, w, d))
        row.append(float(np.mean(errs)))
    best = ALPHAS[int(np.argmin(row))]
    table[b] = {"errs": dict(zip(map(str, ALPHAS), row)), "best_alpha": best,
                "gain_vs_none": row[0] / min(row)}
    cells = " ".join(f"{e:<7.5f}" for e in row)
    print(f"{b:>5} {cells}   {best:.2f}  ({row[0]/min(row):.3f}x vs none)")

out = '/Users/eric/models/Logs/q38v2/awq-alpha-sweep.json'
json.dump(table, open(out, 'w'), indent=1)
print(f"\n-> {out}")
print("\nbest alpha by width: " + ", ".join(f"{b}:{table[b]['best_alpha']}" for b in BITS))
