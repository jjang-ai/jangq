"""Multiple-choice knapsack over measured per-unit option scores.

Objective: minimize Sum proj_weight * nmse (or absolute error energy with
--objective abs) subject to a routed-unit byte budget. Convex-hull greedy per
unit (optimal in the fractional relaxation; gap <= one upgrade).

Constraints honored:
  - every unit starts at 2b_g64 (the DSV4 coherence floor: no g128 option
    is even offered at 2-bit);
  - budget covers ROUTED units only; fixed floors (attention 8b etc.) are
    accounted by the caller when choosing --budget-bytes.

    python -m jang_tools.dots3.solve_plan unit_scores.json out_plan.json \
        --budget-bytes 88000000000 [--objective nmse]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJ_W = {"gate_proj": 1.15, "up_proj": 1.0, "down_proj": 1.35}
BASE_OPT = "2b_g64"

OPT_SPECS = {
    "2b_g64": dict(bits=2, group_size=64),
    "2b_g32": dict(bits=2, group_size=32),
    "3b_g128": dict(bits=3, group_size=128),
    "3b_g64": dict(bits=3, group_size=64),
    "4b_g64": dict(bits=4, group_size=64),
    "mxfp4_g32": dict(bits=4, group_size=32, mode="mxfp4"),
}


def convex_hull(points):
    """points: (bytes, err, name). Keep the dominance frontier: strictly
    decreasing error as bytes increase (dominated options dropped)."""
    out = []
    best = float("inf")
    for b, e, n in sorted(points):
        if e < best - 1e-15:
            out.append((b, e, n))
            best = e
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scores", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--budget-bytes", type=float, required=True)
    ap.add_argument("--objective", choices=("nmse", "abs"), default="nmse")
    ap.add_argument("--allow-mxfp4-routed", action="store_true",
                    help="off by default: routed GPTQ runs on the affine "
                         "grid only; mxfp4 stays a tower-side option")
    ap.add_argument("--max-group", type=int, default=None,
                    help="e.g. 64: drop any option with a larger group "
                         "(DSV4 v6 g64-ceiling conservatism)")
    a = ap.parse_args()

    scores = json.loads(a.scores.read_text())
    if not a.allow_mxfp4_routed:
        for rec in scores.values():
            rec["options"].pop("mxfp4_g32", None)
    if a.max_group:
        for rec in scores.values():
            for name in list(rec["options"]):
                if OPT_SPECS[name]["group_size"] > a.max_group:
                    rec["options"].pop(name)
    state: dict[str, str] = {}
    spent = 0.0
    obj = 0.0
    hulls: dict[str, list] = {}
    for unit, rec in scores.items():
        proj = unit.split(":")[1]
        w = PROJ_W[proj]
        pts = []
        for name, o in rec["options"].items():
            err = o["nmse"] if a.objective == "nmse" else o["wnum"]
            pts.append((o["bytes"], w * err, name))
        hull = convex_hull(pts)
        base = next(((b, e, n) for b, e, n in hull if n == BASE_OPT), None)
        if base is None:
            base_bytes = rec["options"][BASE_OPT]["bytes"]
            base_err = (rec["options"][BASE_OPT]["nmse"] if a.objective == "nmse"
                        else rec["options"][BASE_OPT]["wnum"]) * w
            base = (base_bytes, base_err, BASE_OPT)
            hull = [base] + [h for h in hull if h[0] > base[0] and h[1] < base[1]]
        # drop anything cheaper than base (no sub-floor options)
        hull = [h for h in hull if h[0] >= base[0] or h[2] == BASE_OPT]
        hulls[unit] = hull
        state[unit] = base[2]
        spent += base[0]
        obj += base[1]

    budget = float(a.budget_bytes)
    if spent > budget:
        raise SystemExit(f"floor already exceeds budget: {spent/1e9:.2f} GB "
                         f"> {budget/1e9:.2f} GB")

    def cur(unit):
        h = hulls[unit]
        n = state[unit]
        for i, (b, e, name) in enumerate(h):
            if name == n:
                return i, b, e
        raise KeyError(unit)

    upgrades = 0
    while True:
        best = None
        for unit in scores:
            i, b, e = cur(unit)
            if i + 1 >= len(hulls[unit]):
                continue
            nb, ne, nn = hulls[unit][i + 1]
            db, de = nb - b, e - ne
            if db <= 0 or de <= 0:
                continue
            if spent + db > budget:
                continue
            gain = de / db
            if best is None or gain > best[0]:
                best = (gain, unit, nb, ne, nn, db, de)
        if best is None:
            break
        _, unit, nb, ne, nn, db, de = best
        state[unit] = nn
        spent += db
        obj -= de
        upgrades += 1

    overrides = {}
    counts: dict[str, int] = {}
    for unit, opt in sorted(state.items()):
        counts[opt] = counts.get(opt, 0) + 1
        if opt != BASE_OPT:
            li, proj = unit.split(":")
            overrides[f"{li}:{proj}"] = dict(OPT_SPECS[opt])
    result = {
        "objective": a.objective,
        "budget_bytes": budget,
        "routed_bytes": spent,
        "residual_objective": obj,
        "upgrades": upgrades,
        "option_counts": counts,
        "routed_overrides": overrides,
    }
    a.out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items()
                      if k != "routed_overrides"}, indent=1))
    print(f"overrides: {len(overrides)} units above floor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
