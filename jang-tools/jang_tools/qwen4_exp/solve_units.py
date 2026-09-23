"""Convex-hull greedy MCKP over measured unit scores (dots3 solve_plan
pattern, extended option set). Emits a convert.py bit-map overlay.

  python -m jang_tools.qwen4_exp.solve_units unit_scores.json base_map.json \
      out_map.json --budget-bytes 36.5e9
"""

import argparse
import json
from collections import Counter

SPEC = {"2b_g64": (2, 64), "2b_g32": (2, 32), "3b_g64": (3, 64),
        "4b_g64": (4, 64), "6b_g64": (6, 64), "8b_g64": (8, 64)}
PROJ_W = {"gate_proj": 1.15, "up_proj": 1.0, "down_proj": 1.35}
BASE = "2b_g64"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores"); ap.add_argument("base_map"); ap.add_argument("out_map")
    ap.add_argument("--budget-bytes", type=float, required=True)
    a = ap.parse_args()

    scores = json.loads(open(a.scores).read())
    state, spent = {}, 0
    upgrades = []
    for unit, rec in scores.items():
        proj = unit.split(":")[1]
        w = PROJ_W[proj]
        opts = {k: v for k, v in rec["options"].items() if k in SPEC}
        # dominance frontier by bytes
        hull, best = [], float("inf")
        for name, o in sorted(opts.items(), key=lambda kv: kv[1]["bytes"]):
            if o["nmse"] < best - 1e-15:
                hull.append((name, o["bytes"], o["nmse"] * w))
                best = o["nmse"]
        state[unit] = hull[0][0]
        spent += hull[0][1]
        for i in range(1, len(hull)):
            prev, cur = hull[i - 1], hull[i]
            gain = (prev[2] - cur[2]) / max(cur[1] - prev[1], 1)
            upgrades.append((gain, unit, i, hull))

    # heap-based MCKP greedy: taking an upgrade unlocks the unit's next one
    import heapq

    hulls = {}
    heap = []
    for gain, unit, i, hull in upgrades:
        hulls[unit] = hull
        if i == 1:
            heapq.heappush(heap, (-gain, unit, i))
    level = {u: 0 for u in state}
    while heap:
        neg, unit, i = heapq.heappop(heap)
        hull = hulls[unit]
        if level[unit] != i - 1:
            continue
        delta = hull[i][1] - hull[i - 1][1]
        if spent + delta > a.budget_bytes:
            continue
        spent += delta
        level[unit] = i
        state[unit] = hull[i][0]
        if i + 1 < len(hull):
            g = (hull[i][2] - hull[i + 1][2]) / max(hull[i + 1][1] - hull[i][1], 1)
            heapq.heappush(heap, (-g, unit, i + 1))

    base = json.loads(open(a.base_map).read())
    cnt = Counter(state.values())
    for unit, opt in state.items():
        layer, proj = unit.split(":")
        bits, gs = SPEC[opt]
        base[f"language_model.layers.{layer}.mlp.switch_mlp.{proj}"] = {
            "bits": bits, "group_size": gs}
    json.dump(base, open(a.out_map, "w"), indent=1)
    print(f"routed bytes: {spent/1e9:.1f} GB / budget {a.budget_bytes/1e9:.1f} GB")
    print("choices:", dict(cnt))
    print(f"wrote {a.out_map}")


if __name__ == "__main__":
    main()
