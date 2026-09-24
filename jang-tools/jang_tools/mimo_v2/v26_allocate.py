"""Exact multiple-choice-knapsack allocation from the calibrated sweep.

Each MoE layer picks ONE measured candidate (layout + AWQ alpha); cost is the
measured end-to-end KL of that layer alone (v26_sweep), weight is the unit
bytes. Minimise sum(KL) s.t. sum(bytes) <= budget. Solved exactly by DP on a
1-MiB grid. Assumes per-layer KL costs add (first-order); the built bundle is
then scored on the held-out KL set, against a uniform control of equal size.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .v26_sweep import CANDS, FULL_NATIVE

LAYOUT = dict(CANDS)
LAYOUT[FULL_NATIVE[0]] = FULL_NATIVE[1]
PROJS = ("gate_proj", "up_proj", "down_proj")
MIB = 2 ** 20


def options(layer: dict):
    """-> list of (name, alpha, kl, bytes)."""
    out = []
    for key, v in layer["kl"].items():
        if key.startswith("alpha_"):
            name, alpha = "A_222", float(key.split("_", 1)[1])
        else:
            name, a = key.split("@")
            alpha = float(a)
        out.append((name, alpha, v["kl"], layer["bytes"][name]))
    return out


def allocate(sweep: dict, expert_budget_bytes: int):
    layers = sorted(sweep, key=int)
    opts = [options(sweep[L]) for L in layers]
    cap = expert_budget_bytes // MIB
    INF = float("inf")
    # dp[c] = min KL using exactly <= c MiB so far; keep choice tables per layer
    dp = [0.0] + [INF] * cap
    choice = []
    for o in opts:
        new = [INF] * (cap + 1)
        arg = [-1] * (cap + 1)
        w = [-(-b // MIB) for (_, _, _, b) in o]
        for c in range(cap + 1):
            if dp[c] == INF:
                continue
            for j, (n, al, kl, b) in enumerate(o):
                c2 = c + w[j]
                if c2 <= cap and dp[c] + kl < new[c2]:
                    new[c2], arg[c2] = dp[c] + kl, (j, c)
        dp = new
        choice.append(arg)
    best = min(range(cap + 1), key=lambda c: dp[c])
    if dp[best] == INF:
        raise ValueError("budget infeasible")
    picks, c = [], best
    for li in range(len(layers) - 1, -1, -1):
        j, c = choice[li][c]
        picks.append((layers[li], opts[li][j]))
    picks.reverse()
    return picks, dp[best]


def to_plan(picks, stats_path: str, profile: str, source_revision: str):
    experts, alpha = {}, {}
    for L, (name, al, kl, b) in picks:
        g, u, d = LAYOUT[name]
        experts[str(L)] = {"gate_proj": g, "up_proj": u, "down_proj": d}
        if al:
            alpha[str(L)] = al
    return {"schema": "mimo-v26-jang-plan-v1", "profile": profile, "source_revision": source_revision,
            "expert_default": {"mode": "affine", "bits": 2, "group_size": 64},
            "experts": experts, "awq_alpha": alpha, "stats": stats_path}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=Path, required=True)
    ap.add_argument("--budget-gib", type=float, required=True, help="TEXT loaded budget (experts + non-expert)")
    ap.add_argument("--nonexpert-bytes", type=int, required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--source-revision", default="5711b268169967567844e1e560e8a3966da959b1")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    sweep = json.loads(a.sweep.read_text())
    budget = int(a.budget_gib * 2 ** 30) - a.nonexpert_bytes
    picks, total_kl = allocate(sweep, budget)
    used = sum(p[1][3] for p in picks)
    plan = to_plan(picks, a.stats, a.profile, a.source_revision)
    plan["allocation"] = {"text_budget_gib": a.budget_gib, "expert_bytes": used,
                          "text_loaded_gib": round((used + a.nonexpert_bytes) / 2 ** 30, 3),
                          "sum_layer_kl": total_kl,
                          "per_layer": {str(L): {"cand": n, "alpha": al, "kl": kl} for L, (n, al, kl, b) in picks}}
    a.out.write_text(json.dumps(plan, indent=1))
    from collections import Counter
    print(f"text loaded {plan['allocation']['text_loaded_gib']} GiB, sum KL {total_kl:.4f}")
    print(Counter(p[1][0] for p in picks))
    for L, (n, al, kl, b) in picks:
        print(f"  L{L:>2} {n:6s} alpha={al:<4} kl={kl:.4f}")


if __name__ == "__main__":
    main()
