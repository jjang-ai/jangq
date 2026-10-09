"""JANGT allocator for N0.5: fill an exact byte budget with the best per-unit formats.

Objective (same as the JANGH2 plan, jang_tools.jangh.n05.plan): predicted FROZEN-SELECTION KL = sum over units of
c[L][group] * expert-output NMSE, c from n05_sens.py (geomean over files), NMSE from held-out curves.
Units: per MoE layer, gate_up (tied: the fused gate+up kernel needs one format) and down.
Options per unit (bytes exact, jang_tools.jangt.plan):
  gate_up : jt2 / jt3 (JANGT, curves_jt.json)  and  jh2 / jh3 / jh4 (JANGH on the SAME Hessians, curves_jt.json)
  down    : jh2 / jh3 / jh4 (JANGH, curves.json)  [JANGT down loses to JANGH: hybrid]
Greedy on the per-unit lower convex hull of (bytes, error), exact for convex curves.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path

from jang_tools.jangt.plan import GIB, jangh_bytes, jangt_bytes

E, D, I = 256, 4096, 2048


def opt_bytes(fmt: str, proj: str) -> int:
    n, k = (I, D) if proj in ("gate", "up") else (D, I)
    b = int(fmt[2])
    return jangt_bytes(E, n, k, b) if fmt.startswith("jt") else jangh_bytes(E, n, k, b)


def solve(units: dict, budget: int):
    hull = {}
    for u, opts in units.items():
        opts = sorted(opts, key=lambda o: (o[1], o[2]))
        h = [opts[0]]
        for o in opts[1:]:
            if o[2] >= h[-1][2]:
                continue
            while len(h) >= 2 and (h[-1][2] - o[2]) / (o[1] - h[-1][1]) >= (h[-2][2] - h[-1][2]) / (h[-1][1] - h[-2][1]):
                h.pop()
            h.append(o)
        hull[u] = h
    state = {u: 0 for u in hull}; spent = sum(h[0][1] for h in hull.values())
    if spent > budget:
        raise SystemExit(f"minimum allocation {spent/GIB:.3f} GiB > expert budget {budget/GIB:.3f} GiB")
    heap = [(-(h[0][2] - h[1][2]) / (h[1][1] - h[0][1]), u, 1) for u, h in hull.items() if len(h) > 1]
    heapq.heapify(heap)
    while heap:
        _, u, i = heapq.heappop(heap)
        if state[u] != i - 1:
            continue
        h = hull[u]; d = h[i][1] - h[i - 1][1]
        if spent + d > budget:
            continue
        spent += d; state[u] = i
        if i + 1 < len(h):
            heapq.heappush(heap, (-(h[i][2] - h[i + 1][2]) / (h[i + 1][1] - h[i][1]), u, i + 1))
    choice = {u: hull[u][i] for u, i in state.items()}
    return choice, spent, sum(c[2] for c in choice.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", required=True)
    ap.add_argument("--curves-jt", required=True)
    ap.add_argument("--sens", required=True)
    ap.add_argument("--jangh-plan", required=True)
    ap.add_argument("--target-gib", type=float, default=93.0)
    ap.add_argument("--fixed-gib", type=float, default=6.20666)
    ap.add_argument("--drafter-bytes", type=int, default=1305601530)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    C = json.loads(Path(a.curves).read_text()); J = json.loads(Path(a.curves_jt).read_text())
    S = json.loads(Path(a.sens).read_text())["layers"]; P = json.loads(Path(a.jangh_plan).read_text())
    layers = sorted(int(L) for L in J)
    cw = {(L, g): S[str(L)][g]["c"] for L in layers for g in ("gu", "dn")}
    budget = int(a.target_gib * GIB - a.fixed_gib * GIB - a.drafter_bytes)

    # 0) reproduce the published JANGH2 objective with ITS curves (sanity of the objective)
    eb = P["expert_bits"]
    pub_sept = sum(cw[(L, "gu")] * C[str(L)][f"gate_up{eb[f'{L}:gate']}"] + cw[(L, "dn")] * C[str(L)][f"down{eb[f'{L}:down']}"] for L in layers)
    pub_same = sum(cw[(L, "gu")] * J[str(L)][f"jh_gate_up{eb[f'{L}:gate']}"] + cw[(L, "dn")] * C[str(L)][f"down{eb[f'{L}:down']}"] for L in layers)
    pub_bytes = sum(opt_bytes(f"jh{eb[f'{L}:gate']}", "gate") * 2 + opt_bytes(f"jh{eb[f'{L}:down']}", "down") for L in layers)
    print(f"layers {len(layers)}; expert budget {budget/GIB:.3f} GiB (target {a.target_gib} incl. fixed {a.fixed_gib} + drafter {a.drafter_bytes/GIB:.3f})")
    print(f"published JANGH2 ({pub_bytes/GIB:.2f} GiB experts): objective {pub_sept:.5f} with Sept curves "
          f"(plan file says {P['objective']:.5f}) | {pub_same:.5f} with same-Hessian gate_up curves")

    def units_for(gu_fmts):
        U = {}
        for L in layers:
            U[(L, "gate_up")] = [(f, opt_bytes(f, "gate") * 2, cw[(L, "gu")] * (J[str(L)][f"jt_gate_up{f[2]}"] if f.startswith("jt") else J[str(L)][f"jh_gate_up{f[2]}"]))
                                 for f in gu_fmts if (f"jt_gate_up{f[2]}" if f.startswith("jt") else f"jh_gate_up{f[2]}") in J[str(L)]]
            U[(L, "down")] = [(f"jh{b}", opt_bytes(f"jh{b}", "down"), cw[(L, "dn")] * C[str(L)][f"down{b}"]) for b in (2, 3, 4)]
        return U

    res = {"budget_gib": budget / GIB, "published": {"objective_sept": pub_sept, "objective_same": pub_same, "experts_gib": pub_bytes / GIB}}
    for name, fm in (("jangh_only", ("jh2", "jh3", "jh4")), ("hybrid", ("jt2", "jt3")), ("free", ("jt2", "jt3", "jh2", "jh3", "jh4"))):
        choice, spent, obj = solve(units_for(fm), budget)
        hist = {}
        for (L, p), c in choice.items():
            hist[f"{p}:{c[0]}"] = hist.get(f"{p}:{c[0]}", 0) + 1
        res[name] = {"objective": obj, "experts_gib": spent / GIB, "total_gib": (spent + a.fixed_gib * GIB + a.drafter_bytes) / GIB,
                     "histogram": hist, "formats": {f"{L}:{p}": c[0] for (L, p), c in choice.items()}}
        print(f"{name:10s} objective {obj:.5f} ({obj/pub_same - 1:+.1%} vs published JANGH2 at 96 GiB)  experts {spent/GIB:.2f} GiB  "
              f"total {res[name]['total_gib']:.3f} GiB  {dict(sorted(hist.items()))}")
    Path(a.out).write_text(json.dumps(res, indent=1))




# ------------------------------------------------------------------------------------------- v2 (curves_jt2.json)
def jt_bytes_v2(proj: str, k) -> int:
    """Exact JANGT v2 bytes of one projection (all experts) at width k, ragged layout: packed + f32 row scales +
    8 outlier cols (f16 weights, i32 index) + per-layer signs + offsets/kbits; down also stores f32 column scales."""
    from jang_tools.jangt.expert_alloc import words_per_row_k
    n, kin = (I, D) if proj in ("gate", "up") else (D, I)
    b = E * n * (words_per_row_k(kin, k) * 4 + 4 + 8 * 2) + E * 8 * 4 + kin * 4 + (E + 1) * 8 + E * 4
    if proj == "down":
        b += E * kin * 4
    return b


def plan_v2(J2: dict, cw: dict, budget: int, fmts_gu, fmts_dn):
    U = {}
    for L in sorted(int(x) for x in J2):
        j = J2[str(L)]
        def opt(f, grp):
            kind, k = f[:2], f[2:]
            key = f"{kind}_{'gate_up' if grp == 'gu' else 'down'}{k}"
            if key not in j:
                return None
            kk = float(k); kk = int(kk) if kk.is_integer() else kk
            by = (2 * jt_bytes_v2("gate", kk) if grp == "gu" else jt_bytes_v2("down", kk)) if kind == "jt" else \
                 (2 * opt_bytes(f"jh{int(kk)}", "gate") if grp == "gu" else opt_bytes(f"jh{int(kk)}", "down"))
            return (f, by, cw[(L, grp)] * j[key])
        U[(L, "gate_up")] = [o for o in (opt(f, "gu") for f in fmts_gu) if o]
        U[(L, "down")] = [o for o in (opt(f, "dn") for f in fmts_dn) if o]
    return solve(U, budget)


def main_v2():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves-v2", required=True)
    ap.add_argument("--sens", required=True)
    ap.add_argument("--jangh-plan", required=True)
    ap.add_argument("--targets", default="93,94,95,96")
    ap.add_argument("--fixed-gib", type=float, default=6.20666); ap.add_argument("--drafter-bytes", type=int, default=1305601530)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    J2 = json.loads(Path(a.curves_v2).read_text()); S = json.loads(Path(a.sens).read_text())["layers"]
    P = json.loads(Path(a.jangh_plan).read_text()); layers = sorted(int(x) for x in J2)
    cw = {(L, g): S[str(L)][g]["c"] for L in layers for g in ("gu", "dn")}
    eb = P["expert_bits"]
    pub = sum(cw[(L, "gu")] * J2[str(L)][f"jh_gate_up{eb[f'{L}:gate']}"] + cw[(L, "dn")] * J2[str(L)][f"jh_down{eb[f'{L}:down']}"] for L in layers)
    print(f"{len(layers)} layers; published JANGH2 objective on v2 curves: {pub:.5f}")
    res = {"published_v2": pub}
    JH = ("jh2", "jh3", "jh4"); JT = ("jt2", "jt2.5", "jt3")
    for t in [float(x) for x in a.targets.split(",")]:
        budget = int(t * GIB - a.fixed_gib * GIB - a.drafter_bytes)
        for name, fg, fd in (("jangh_only", JH, JH), ("jangt_only", JT, JT), ("free", JT + JH, JT + JH)):
            ch, spent, obj = plan_v2(J2, cw, budget, fg, fd)
            hist = {}
            for (L, p), c_ in ch.items():
                hist[f"{p}:{c_[0]}"] = hist.get(f"{p}:{c_[0]}", 0) + 1
            res[f"{t}:{name}"] = {"objective": obj, "experts_gib": spent / GIB, "formats": {f"{L}:{p}": c_[0] for (L, p), c_ in ch.items()}, "hist": hist}
            print(f"{t:5.1f} GiB {name:10s} objective {obj:.5f} ({obj/pub-1:+.1%} vs JANGH2)  experts {spent/GIB:.2f} GiB  {dict(sorted(hist.items()))}", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    import sys as _sys
    if "--curves-v2" in _sys.argv or "--v2" in _sys.argv:
        if "--v2" in _sys.argv:
            _sys.argv.remove("--v2")
        main_v2()
    else:
        main()
