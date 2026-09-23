"""Hessian-driven mixed-bit allocation for Ling-3.0 routed experts.

Created by Jinho Jang (eric@jangq.ai) — 2026-08-26.

Assigns per-tensor bit widths by **measured sensitivity** rather than by
name-matching TIER_RULES. That matters here twice over: zero Ling-3.0 tensor
names match the existing rules (the trap that bit Inkling, Nemotron 3.5 and
openPangu), and the measured spread is wildly asymmetric between projections —
`down_proj` varies ~4900x across layers while `gate/up_proj` vary only ~23x.
A uniform assignment would spend equal bits on tensors that differ by three
orders of magnitude in impact.

Sensitivity per tensor group (all 128 experts of one layer/projection):

    S_t = sum_ij  W_ij^2 * a_j^2

which is the diagonal-Hessian-weighted weight energy — `a_j^2` is the captured
per-channel 2nd moment, i.e. the same statistic that feeds the imatrix and AWQ.

Allocation is **budget-neutral**: given a target average bit width, spend the
budget where it buys the most error reduction. Under the standard uniform-
quantizer model the error of tensor `t` at `b` bits is proportional to
`S_t * 2^(-2b)`, so promoting `b -> b+1` gains `0.75 * S_t * 2^(-2b)` at a cost
of `numel_t / 8` bytes. Greedily take the best gain-per-byte until the budget is
spent — the discrete analogue of water-filling.

    python -m jang_tools.ling3.allocate <model_dir> <calib.safetensors> <out.json> \
        [--profile JANG_4M|JANG_6M] [--floor N] [--ceiling N]
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
from collections import defaultdict
from pathlib import Path

import mlx.core as mx

# MLX `mx.quantize` supports exactly these widths — 7 does NOT exist.
# Assuming a contiguous ladder silently produces a bitmap that cannot be built.
ALLOWED_BITS = (2, 3, 4, 5, 6, 8)

# Floors are per-projection, not global. `down_proj` is the SwiGLU amplifier
# (no swiglu_limit clamping in this config) and historically wants the extra bit
# first; the measurement above independently agrees for the late layers.
PROFILES = {
    "JANG_4M": {"target": 4.0, "floor": {"gate_proj": 3, "up_proj": 3, "down_proj": 4}, "ceiling": 8},
    "JANG_6M": {"target": 6.0, "floor": {"gate_proj": 5, "up_proj": 5, "down_proj": 6}, "ceiling": 8},
}


def expert_sensitivity(model_dir: Path, calib: dict[str, mx.array]) -> dict[str, float]:
    """S_t = sum over experts of sum_ij W_ij^2 * a_j^2."""
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    by_shard: dict[str, list[str]] = defaultdict(list)
    for k, shard in idx.items():
        if ".mlp.experts." in k:
            by_shard[shard].append(k)

    sens: dict[str, float] = defaultdict(float)
    for shard in sorted(by_shard):
        weights = mx.load(str(model_dir / shard))
        for k in by_shard[shard]:
            head, tail = k.split(".mlp.experts.")
            _, proj = tail.split(".", 1)
            path = head + ".mlp.switch_mlp." + proj.replace(".weight", "")
            a = calib[path].astype(mx.float32)
            W = weights[k].astype(mx.float32)
            s = ((W * W) @ a).sum()
            mx.eval(s)
            sens[path] += float(s)
        del weights
    return dict(sens)


def expert_numel(model_dir: Path) -> dict[str, int]:
    """Total element count per tensor group (all experts of one layer/projection)."""
    cfg = json.loads((model_dir / "config.json").read_text())
    E = cfg["num_experts"]
    H = cfg["hidden_size"]
    I = cfg["moe_intermediate_size"]
    n_layers = cfg["num_hidden_layers"]
    first_dense = cfg["first_k_dense_replace"]

    out: dict[str, int] = {}
    for layer in range(first_dense, n_layers):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            out[f"model.layers.{layer}.mlp.switch_mlp.{proj}"] = E * H * I
    return out


def allocate(
    sens: dict[str, float],
    numel: dict[str, int],
    target_bits: float,
    floor: dict[str, int],
    ceiling: int = 8,
) -> dict[str, int]:
    """Greedy water-filling: spend the bit budget where it buys the most.

    Returns a path -> bit-width map whose weighted average equals `target_bits`
    (to within one promotion), so the result is budget-neutral against a uniform
    assignment at the same average.
    """
    paths = sorted(sens)
    bits = {p: floor[p.rsplit(".", 1)[1]] for p in paths}

    ladder = [b for b in ALLOWED_BITS if b <= ceiling]
    bad = sorted({b for b in bits.values() if b not in ALLOWED_BITS})
    if bad:
        raise ValueError(f"floor uses unsupported bit width(s) {bad}; allowed {ALLOWED_BITS}")

    def next_bits(b: int) -> int | None:
        for cand in ladder:
            if cand > b:
                return cand
        return None

    total_numel = sum(numel[p] for p in paths)
    budget_bits = target_bits * total_numel
    spent = sum(bits[p] * numel[p] for p in paths)
    if spent > budget_bits:
        raise ValueError(
            f"floors alone cost {spent / total_numel:.2f} bits/weight, "
            f"over the {target_bits} target — lower the floors or raise the target"
        )

    def push(heap, p):
        nb = next_bits(bits[p])
        if nb is None:
            return
        # error ~ S * 2^(-2b); gain is the drop from b to the NEXT ALLOWED width,
        # and the cost is the full bit delta — 6->8 costs two bits, not one.
        gain = sens[p] * (2.0 ** (-2 * bits[p]) - 2.0 ** (-2 * nb))
        delta = nb - bits[p]
        heapq.heappush(heap, (-gain / (numel[p] * delta), p))

    heap: list[tuple[float, str]] = []
    for p in paths:
        push(heap, p)

    while heap:
        _, p = heapq.heappop(heap)
        nb = next_bits(bits[p])
        if nb is None:
            continue
        cost = numel[p] * (nb - bits[p])
        if spent + cost > budget_bits:
            continue                         # cannot afford; try the next best
        bits[p] = nb
        spent += cost
        push(heap, p)

    return bits


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="jang_tools.ling3.allocate")
    ap.add_argument("model_dir")
    ap.add_argument("calib")
    ap.add_argument("out")
    ap.add_argument("--profile", default="JANG_4M", choices=sorted(PROFILES))
    args = ap.parse_args(argv)

    model_dir = Path(args.model_dir)
    prof = PROFILES[args.profile]

    calib = mx.load(args.calib)
    sens = expert_sensitivity(model_dir, calib)
    numel = expert_numel(model_dir)

    missing = sorted(set(numel) - set(sens))
    if missing:
        raise SystemExit(
            f"refusing to allocate: {len(missing)} expert groups have no captured "
            f"statistic, e.g. {missing[:4]} — the calibration did not cover the model"
        )

    bits = allocate(sens, numel, prof["target"], prof["floor"], prof["ceiling"])

    total = sum(numel[p] for p in bits)
    avg = sum(bits[p] * numel[p] for p in bits) / total
    hist: dict[int, int] = defaultdict(int)
    for p, b in bits.items():
        hist[b] += 1

    payload = {
        "profile": args.profile,
        "target_bits": prof["target"],
        "achieved_bits": avg,
        "floor": prof["floor"],
        "ceiling": prof["ceiling"],
        "bits": bits,
        "sensitivity": sens,
        "histogram": {str(k): v for k, v in sorted(hist.items())},
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True))

    print(f"[{args.profile}] target={prof['target']} achieved={avg:.4f} bits/weight")
    print(f"  histogram (bits -> #tensor groups): {dict(sorted(hist.items()))}")
    ranked = sorted(bits.items(), key=lambda kv: -sens[kv[0]])
    print("  most sensitive:")
    for p, b in ranked[:5]:
        print(f"    {b}b  S={sens[p]:.3e}  {p}")
    print("  least sensitive:")
    for p, b in ranked[-5:]:
        print(f"    {b}b  S={sens[p]:.3e}  {p}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
