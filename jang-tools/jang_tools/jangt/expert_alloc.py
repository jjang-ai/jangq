"""Per-expert trellis widths inside one (layer, group) at a fixed byte budget (N0.5).

Expert e's share of the layer objective is its router-weight^2 traffic t_e (Hessian `wsum`: cal_tr rows only).
With the layer's measured error ratio r(K) = NMSE(K) / NMSE(2) (curves_jt.json for gate/up, curves.json down;
K = 2.5 interpolated geometrically — the K=2.5 trellis measured 7 % better than that on Gaussian data), the expert
error is t_e * r(K_e). Greedy on marginal error drop per byte (exact for convex per-expert curves) from all-K_min
until the group budget is spent. Held-out check (te traffic) 0.79x layer error at mean 2.5, cap 3 (01-LOG).
Writes {"L": {"gu": [K_e], "dn": [K_e]}} for moe_encode --expert-k.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from jang_tools.jangt.plan import words_per_row

E, D, I = 256, 4096, 2048
LEVELS = (2.0, 2.5, 3.0)


def row_bytes(kin: int, k: float, n_out: int = 8) -> int:
    k = int(k) if float(k).is_integer() else k
    return words_per_row_k(kin, k) * 4 + 4 + n_out * 2


def words_per_row_k(kin: int, k) -> int:
    if float(k).is_integer():
        return words_per_row(kin, 12, int(k))
    bits = 12 + 5 * (kin // 2 - 1) - (1 if (kin // 2 - 1) % 2 else 0)        # alternating 6/4-bit steps (K=2.5)
    return (bits + 31) // 32 + 1


def group_bytes(g: str, k: float) -> int:
    """bytes of ONE expert of the group at width k (gate+up together for 'gu')"""
    return (2 * I * row_bytes(D, k)) if g == "gu" else D * row_bytes(I, k)


def allocate(t: np.ndarray, r: dict, g: str, budget: int, levels=LEVELS, gamma: float = 1.0) -> list[float]:
    """gamma < 1 tempers the traffic weight (t^gamma): measured on L33/L46, gamma = 1 lowered held-out NMSE ~10 % but
    not the frozen-KL mean, and raised its p99 (tokens routed to cold 2-bit experts)."""
    t = np.maximum(t, 1e-12) ** gamma; t = t / t.sum()
    k = np.zeros(len(t), dtype=int)                    # index into levels
    spent = sum(group_bytes(g, levels[0]) for _ in t)
    assert spent <= budget, f"budget {budget} below all-{levels[0]} ({spent})"
    while True:
        best, bi = 0.0, -1
        for e in range(len(t)):
            if k[e] + 1 >= len(levels):
                continue
            db = group_bytes(g, levels[k[e] + 1]) - group_bytes(g, levels[k[e]])
            if spent + db > budget:
                continue
            gain = t[e] * (r[levels[k[e]]] - r[levels[k[e] + 1]]) / db
            if gain > best:
                best, bi = gain, e
        if bi < 0:
            break
        spent += group_bytes(g, levels[k[bi] + 1]) - group_bytes(g, levels[k[bi]]); k[bi] += 1
    return [levels[i] for i in k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", required=True); ap.add_argument("--mean-k", type=float, required=True,
                                                                help="budget = all experts at this width (2, 2.5 or 3)")
    ap.add_argument("--hess-dir", required=True)
    ap.add_argument("--curves", required=True)
    ap.add_argument("--curves-jt", required=True)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from jang_tools.jangh.n05.hess_cpu import HessStore
    C = json.loads(Path(a.curves).read_text()); J = json.loads(Path(a.curves_jt).read_text())
    out = {}
    for L in [int(x) for x in a.layers.split(",")]:
        t = np.array(HessStore(str(Path(a.hess_dir) / f"L{L:02d}.safetensors"), "gu").wsum, dtype=np.float64)
        j, c = J[str(L)], C[str(L)]
        rg = {2.0: 1.0, 3.0: j["jt_gate_up3"] / j["jt_gate_up2"]}; rg[2.5] = math.sqrt(rg[3.0])
        rd = {2.0: 1.0, 3.0: c["down3"] / c["down2"]}; rd[2.5] = math.sqrt(rd[3.0])
        out[str(L)] = {}
        for g, r in (("gu", rg), ("dn", rd)):
            budget = E * group_bytes(g, a.mean_k)
            ks = allocate(t, r, g, budget, gamma=a.gamma)
            tt = t / t.sum()
            f = float(sum(tt[e] * r[ks[e]] for e in range(E)) / r[a.mean_k])
            out[str(L)][g] = ks
            print(f"L{L} {g}: widths 2/2.5/3 = {[ks.count(x) for x in LEVELS]}  bytes {sum(group_bytes(g, x) for x in ks)} "
                  f"<= {budget}  predicted error factor vs uniform {a.mean_k}: {f:.3f}", flush=True)
    Path(a.out).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
