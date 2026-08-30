"""Build the full glm5_next bit-map: fixed floors + heap-MCKP-solved routed
expert units under a TOTAL bundle budget (default 95 GiB).

Floors (Eric's hard rules, 2026-08-27): ALL attention 8-bit (KDA q/k/v/o, MLA
projs, indexer), shared experts + dense mlp 8-bit, embeds/lm_head 8-bit,
vision 8-bit, MTP attention 8-bit + MTP experts fixed low-bit. Gate producers
/ norms / mHC / conv / router are keeps (converter enforces regardless).

  python -m jang_tools.glm5_next.solve_map unit_scores.json out_map.json \
      [--total-gib 95] [--mtp-experts 3b_g64]
"""

from __future__ import annotations

import argparse
import heapq
import json
from collections import Counter

SPEC = {"2b_g128": (2, 128), "2b_g64": (2, 64), "3b_g64": (3, 64),
        "4b_g64": (4, 64), "6b_g64": (6, 64), "8b_g64": (8, 64)}
PROJ_W = {"gate_proj": 1.15, "up_proj": 1.0, "down_proj": 1.35}

E, D_IN, D_H = 288, 4096, 2048
N_KDA, N_MLA_MAIN, N_LAYERS = 34, 11, 45
VOCAB, D = 154880, 4096


def eff_bytes(n: float, bits: int, gs: int) -> float:
    return n * (bits + 32 / gs) / 8


def floors_bytes(mtp_experts_spec, no_mtp=False, embed_bits=8) -> float:
    b8 = lambda n: eff_bytes(n, 8, 64)
    total = 0.0
    total += eff_bytes(2 * VOCAB * D, embed_bits, 64)             # embed + lm_head
    total += b8(N_KDA * (3 * 8192 * D + D * 8192))                # KDA qkv+o
    total += b8((N_MLA_MAIN + 1) * (1536 * D + 16384 * 1536 + 512 * D
                                    + 32768 * 512 + D * 16384))   # MLA (+MTP layer)
    total += b8(12 * (128 * D + D * 1536 + 32 * D + 128 * D))     # indexer (12 incl MTP)
    total += b8(43 * 3 * D_H * D)                                 # shared experts
    total += b8(3 * (2 * 12288 * D + D * 12288))                  # dense layers 0-2
    total += b8(0.45e9)                                           # vision tower approx
    if not no_mtp:
        bits, gs = SPEC[mtp_experts_spec]
        total += eff_bytes(E * 3 * D_H * D, bits, gs)             # MTP experts
        total += b8(D * 8192)                                     # MTP eh_proj
    # bf16 keeps: f/g low-rank, b_proj, convs, norms, hc, router fp32
    total += (N_KDA * (2 * (128 * D + 8192 * 128) + 64 * D + 3 * 8192 * 4) * 2
              + 90 * (24 * 16384) * 2 + 43 * (E * D * 4 + E * 4)
              + N_LAYERS * 4 * D * 2 + 2e7)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    ap.add_argument("out_map")
    ap.add_argument("--total-gib", type=float, default=95.0)
    ap.add_argument("--mtp-experts", default="3b_g64", choices=sorted(SPEC))
    ap.add_argument("--no-mtp", action="store_true")
    ap.add_argument("--embed-bits", type=int, default=8)
    a = ap.parse_args()

    floors = floors_bytes(a.mtp_experts, no_mtp=a.no_mtp, embed_bits=a.embed_bits)
    budget = a.total_gib * 2**30 - floors
    print(f"floors: {floors/2**30:.2f} GiB  → routed-expert budget "
          f"{budget/2**30:.2f} GiB of {a.total_gib} GiB total")
    assert budget > 0

    scores = json.loads(open(a.scores).read())
    state, spent = {}, 0.0
    hulls, heap = {}, []
    for unit, rec in scores.items():
        proj = unit.split(":")[1]
        w = PROJ_W[proj]
        opts = {k: v for k, v in rec["options"].items() if k in SPEC}
        hull, best = [], float("inf")
        for name, o in sorted(opts.items(), key=lambda kv: kv[1]["bytes"]):
            if o["nmse"] < best - 1e-15:
                hull.append((name, o["bytes"], o["nmse"] * w))
                best = o["nmse"]
        state[unit] = hull[0][0]
        spent += hull[0][1]
        hulls[unit] = hull
        if len(hull) > 1:
            gain = (hull[0][2] - hull[1][2]) / max(hull[1][1] - hull[0][1], 1)
            heapq.heappush(heap, (-gain, unit, 1))

    level = {u: 0 for u in state}
    while heap:
        neg, unit, i = heapq.heappop(heap)
        hull = hulls[unit]
        if level[unit] != i - 1:
            continue
        delta = hull[i][1] - hull[i - 1][1]
        if spent + delta > budget:
            continue
        spent += delta
        level[unit] = i
        state[unit] = hull[i][0]
        if i + 1 < len(hull):
            g = (hull[i][2] - hull[i + 1][2]) / max(hull[i + 1][1] - hull[i][1], 1)
            heapq.heappush(heap, (-g, unit, i + 1))

    bit_map = {
        # global floors — longest-prefix matching resolves specifics first
        "model.embed_tokens": {"bits": a.embed_bits, "group_size": 64},
        "lm_head": {"bits": a.embed_bits, "group_size": 64},
        "visual.": {"bits": 8, "group_size": 64},
        "model.layers.": {"bits": 8, "group_size": 64},   # attention/shared/dense default
    }
    if not a.no_mtp:
        mb, mg = SPEC[a.mtp_experts]
        for proj in ("gate_proj", "up_proj", "down_proj"):
            bit_map[f"model.layers.45.mlp.switch_mlp.{proj}"] = {"bits": mb, "group_size": mg}
    for unit, opt in state.items():
        layer, proj = unit.split(":")
        bits, gs = SPEC[opt]
        bit_map[f"model.layers.{layer}.mlp.switch_mlp.{proj}"] = {
            "bits": bits, "group_size": gs}

    json.dump(bit_map, open(a.out_map, "w"), indent=1)
    total = (floors + spent) / 2**30
    print(f"routed: {spent/2**30:.2f} GiB → TOTAL est {total:.2f} GiB "
          f"(target {a.total_gib})")
    print("choices:", dict(Counter(state.values())))
    print(f"wrote {a.out_map}")


if __name__ == "__main__":
    main()
