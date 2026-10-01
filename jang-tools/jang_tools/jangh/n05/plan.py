"""Naive-N0.5-Flash JANGH build plan: exact per-tensor formats + expert bit allocation.

Per-tensor rules (source header shapes; runtime names == source names except stacked experts):
  routed experts  model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight  -> JANGH units, bits solved here
  KEEP (source dtype): norms, router mlp.gate.weight + e_score_correction_bias (fp32), attention_sink_bias,
                       indexer.* (wq / wk / weights_proj / k_norm: they decide the DSA top-k SELECTION; 0.16 GiB)
  other 2-D weights (attention q/k/v/o, layer-0 dense MLP, embed_tokens, lm_head) -> --nonexpert affine8 | mxfp8
Allocation: multiple-choice knapsack over units with measured held-out error curves (curves.py), greedy on the convex
hull (exact for convex curves), objective = sum over layers of expert-output NMSE (additive over projections).
  --tie-gate-up : gate and up of a layer must share bits (required by the current fused gate/up kernels)
Budget: total tensor payload <= --total-gib.
"""
from __future__ import annotations

import argparse
import heapq
import json
from collections import Counter
from pathlib import Path

E, D, I = 256, 4096, 2048
DT = {"BF16": 2, "F16": 2, "F32": 4}


def classify(k: str, dtype: str, shape) -> str:
    if ".mlp.experts." in k:
        return "expert"
    if ".self_attn.indexer." in k or "norm" in k or k.endswith(("mlp.gate.weight", "e_score_correction_bias", "attention_sink_bias")):
        return "keep"
    if len(shape) == 2 and k.endswith(".weight") and shape[1] % 64 == 0:
        return "q8"
    return "keep"


def unit_bytes(unit: str, bits: int) -> int:
    n, k = (I, D) if unit in ("gate", "up") else (D, I)
    return E * n * (k * bits // 8 + 2)                      # packed + fp16 row scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headers", required=True); ap.add_argument("--curves", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--total-gib", type=float, default=96.0)
    ap.add_argument("--nonexpert", default="affine8", choices=["affine8", "mxfp8"])
    ap.add_argument("--tie-gate-up", action="store_true")
    ap.add_argument("--max-bits", type=int, default=4)
    ap.add_argument("--sens", default=None, nargs="*", help="sens2.py output(s): weight every unit's NMSE curve by the measured FROZEN-SELECTION "
                    "KL per unit NMSE of its layer/group (several files: geometric mean)")
    ap.add_argument("--debug-fill", action="store_true", help="DEBUG ONLY: fill layers missing from curves with the mean curve")
    a = ap.parse_args()
    H = json.loads(Path(a.headers).read_text())
    curves = json.loads(Path(a.curves).read_text())
    fixed = Counter(); fmt = {}
    for k, (f, dt, shp) in H.items():
        c = classify(k, dt, shp)
        n = 1
        for s in shp:
            n *= s
        if c == "q8":
            c = a.nonexpert
            fixed[c] += n + (n / 64 * 4 if c == "affine8" else n / 32)
        elif c == "keep":
            fixed[c] += n * DT[dt]
        fmt[k] = c
    fixed_total = sum(fixed.values())
    budget = a.total_gib * 2**30 - fixed_total
    layers = sorted(int(x) for x in curves)
    if a.debug_fill and layers != list(range(1, 48)):
        import numpy as _np
        keys = [k for k, v in curves[str(layers[0])].items() if isinstance(v, float)]
        mean = {k: float(_np.mean([curves[str(L)][k] for L in layers])) for k in keys}
        for L in range(1, 48):
            curves.setdefault(str(L), {**mean, "gu_m_min": 0.5, "dn_m_min": 0.5})
        layers = list(range(1, 48)); print("DEBUG FILL: plan is NOT valid for a release build")
    assert layers == list(range(1, 48)), f"curves incomplete: {len(layers)} layers"
    cw = {(L, g): 1.0 for L in layers for g in ("gu", "dn")}
    if a.sens:
        import math
        sens = [json.loads(Path(f).read_text()) for f in a.sens]
        assert all(x.get("valid_for_planning") and "sigma" in x for x in sens), "need complete sens2.py outputs (frozen selection)"
        for L in layers:
            for g in ("gu", "dn"):
                cs = [x["layers"][str(L)][g]["c"] for x in sens]
                assert all(c > 0 and c == c for c in cs), f"sens: layer {L} {g} weights {cs}"
                cw[(L, g)] = math.exp(sum(math.log(c) for c in cs) / len(cs))
    units = {}
    for L in layers:
        r = {k: v for k, v in curves[str(L)].items()}
        for k in list(r):
            for pre, g in (("gate_up", "gu"), ("gate", "gu"), ("up", "gu"), ("down", "dn")):
                if k.startswith(pre) and k[len(pre):].isdigit():
                    r[k] = r[k] * cw[(L, g)]; break
        if a.tie_gate_up:
            units[(L, "gate_up")] = [(b, unit_bytes("gate", b) + unit_bytes("up", b), r[f"gate_up{b}"]) for b in (2, 3, 4) if b <= a.max_bits]
        else:
            for p in ("gate", "up"):
                units[(L, p)] = [(b, unit_bytes(p, b), r[f"{p}{b}"]) for b in (2, 3, 4) if b <= a.max_bits]
        units[(L, "down")] = [(b, unit_bytes("down", b), r[f"down{b}"]) for b in (2, 3, 4) if b <= a.max_bits]
    # convex hull per unit (drop options that are not on the lower hull of (bytes, error))
    hull = {}
    for u, opts in units.items():
        h = [opts[0]]
        for o in opts[1:]:
            if o[2] >= h[-1][2]:
                continue                                         # more bytes, no gain
            while len(h) >= 2 and (h[-1][2] - o[2]) / (o[1] - h[-1][1]) >= (h[-2][2] - h[-1][2]) / (h[-1][1] - h[-2][1]):
                h.pop()
            h.append(o)
        hull[u] = h
    state = {u: 0 for u in hull}
    spent = sum(h[0][1] for h in hull.values())
    assert spent <= budget, f"minimum allocation {spent/2**30:.2f} GiB > expert budget {budget/2**30:.2f} GiB"
    heap = []
    for u, h in hull.items():
        if len(h) > 1:
            heapq.heappush(heap, (-(h[0][2] - h[1][2]) / (h[1][1] - h[0][1]), u, 1))
    while heap:
        _, u, i = heapq.heappop(heap)
        if state[u] != i - 1:
            continue
        h = hull[u]; delta = h[i][1] - h[i - 1][1]
        if spent + delta > budget:
            continue
        spent += delta; state[u] = i
        if i + 1 < len(h):
            heapq.heappush(heap, (-(h[i][2] - h[i + 1][2]) / (h[i + 1][1] - h[i][1]), u, i + 1))
    bits = {}
    for (L, p), i in state.items():
        b = hull[(L, p)][i][0]
        if p == "gate_up":
            bits[f"{L}:gate"] = b; bits[f"{L}:up"] = b
        else:
            bits[f"{L}:{p}"] = b
    objective = sum(hull[u][i][2] for u, i in state.items())
    floor_obj = sum(h[0][2] for h in hull.values())
    total = fixed_total + spent
    plan = {"format": "jangh (on-disk jangtq v2)", "total_gib_budget": a.total_gib, "total_bytes": total, "total_gib": total / 2**30,
            "fixed_gib_by_format": {k: v / 2**30 for k, v in fixed.items()}, "experts_gib": spent / 2**30,
            "expert_bits": bits, "tie_gate_up": a.tie_gate_up, "nonexpert": a.nonexpert,
            "objective": objective, "objective_all_2bit": floor_obj,
            "objective_unit": "predicted frozen-selection KL (sens2-weighted NMSE)" if a.sens else "sum of expert-output NMSE", "sens": a.sens,
            "sens_weights": {f"{L}:{g}": v for (L, g), v in cw.items()} if a.sens else None,
            "expert_bits_histogram": dict(Counter(f"{k.split(':')[1]}:{b}" for k, b in bits.items())),
            "recipe_m_min": {str(L): {"gu": curves[str(L)]["gu_m_min"], "dn": curves[str(L)]["dn_m_min"]} for L in layers},
            "tensor_formats": fmt}
    Path(a.out).write_text(json.dumps(plan, indent=1))
    print(f"fixed {fixed_total/2**30:.3f} GiB: " + ", ".join(f"{k} {v/2**30:.3f}" for k, v in fixed.items()))
    print(f"experts {spent/2**30:.3f} GiB -> TOTAL {total/2**30:.3f} GiB (budget {a.total_gib}); "
          f"objective [{plan['objective_unit']}] {objective:.4f} (all 2-bit {floor_obj:.4f}); avg expert bits {spent*8/(E*(2*I*D+D*I)*47):.3f}")
    print("histogram:", plan["expert_bits_histogram"])
    for L in layers:
        print(f"  L{L:2d} gate {bits[f'{L}:gate']} up {bits[f'{L}:up']} down {bits[f'{L}:down']}")


if __name__ == "__main__":
    main()
