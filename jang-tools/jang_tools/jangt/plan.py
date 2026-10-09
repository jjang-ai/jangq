"""JANGT size planner: exact byte accounting per tensor format and the target-GiB budget (the core of `jangt plan`).

Byte model (must equal what the encoders write; `--check` compares against stage files):
  JANGT  K bits/weight, V=2, L-bit state, per (expert, row) of K_in inputs:
           packed words  words_per_row(K_in, L) * 4      (K=2; K=3 uses words_per_row_k3)
           row scale     4  (f32 jt_scale)
           outliers      n_out * 2 (f16 jt_wout) per row + n_out * 4 per expert (i32 jt_ocol); padded to n_max
           signs         K_in * 4 once per layer and projection (jt_su)
  JANGH  b bits: ceil(K_in * b / 32) * 4 packed + 2 (fp16 row scale)      (jangtq2 bitstream, tq2_scales)
  fixed  non-expert tensors from the JANGH plan (affine8 g64 / keep) + drafter file bytes
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

GIB = 2 ** 30


def words_per_row(kin: int, L: int = 12, kbits: int = 2) -> int:
    """K=2 layout of v2l.words_per_row; K=3 = 6 state bits per 2 weights (same structure, 3 bits/weight)."""
    if kbits == 2:
        return (2 * kin + (L - 4) + 31) // 32 + 1
    return (3 * kin + (L - 6) + 31) // 32 + 1


def jangt_bytes(E: int, N: int, kin: int, kbits: int = 2, L: int = 12, n_out_max: int = 8) -> int:
    return E * N * (words_per_row(kin, L, kbits) * 4 + 4 + n_out_max * 2) + E * n_out_max * 4 + kin * 4


def jangh_bytes(E: int, N: int, kin: int, bits: int) -> int:
    return E * N * (math.ceil(kin * bits / 32) * 4 + 2)


def expert_units(cfg: dict) -> list[tuple[int, str, int, int]]:
    """(layer, proj, N, K_in) for every routed-expert projection."""
    D, I, nL = cfg["hidden_size"], cfg["moe_intermediate_size"], cfg["num_hidden_layers"]
    if "moe_layer_freq" in cfg and isinstance(cfg["moe_layer_freq"], list):
        moe = [bool(x) for x in cfg["moe_layer_freq"]]
    else:
        fk = cfg.get("first_k_dense_replace", 1); moe = [L >= fk for L in range(nL)]
    out = []
    for L in (L for L in range(nL) if moe[L]):
        out += [(L, "gate", I, D), (L, "up", I, D), (L, "down", D, I)]
    return out


def unit_bytes(fmt: str, E: int, N: int, kin: int) -> int:
    """fmt: 'jt2' | 'jt3' | 'jh2' | 'jh3' | 'jh4'."""
    if fmt.startswith("jt"):
        return jangt_bytes(E, N, kin, int(fmt[2]))
    return jangh_bytes(E, N, kin, int(fmt[2]))


def budget(cfg: dict, fixed_bytes: int, drafter_bytes: int, target_gib: float, base: dict[str, str], up: dict[str, str]):
    """Uniform-sensitivity view of the budget: start every unit at its base format (per proj), then report how many
    units can be upgraded to `up` within the target. The real allocator orders upgrades by measured
    frozen-KL gain per byte; this function is the byte bookkeeping it runs on."""
    E = cfg["n_routed_experts"]; units = expert_units(cfg)
    exp_budget = target_gib * GIB - fixed_bytes - drafter_bytes
    b0 = sum(unit_bytes(base[p], E, N, K) for _, p, N, K in units)
    deltas = sorted(((unit_bytes(up[p], E, N, K) - unit_bytes(base[p], E, N, K)), L, p) for L, p, N, K in units)
    used, n_up = b0, 0
    for d, _, _ in deltas:
        if used + d > exp_budget:
            break
        used += d; n_up += 1
    nw = sum(E * N * K for _, _, N, K in units)
    return {"expert_budget_gib": exp_budget / GIB, "base_gib": b0 / GIB, "units": len(units), "upgradable_units": n_up,
            "experts_gib": used / GIB, "experts_bpw": used * 8 / nw, "total_gib": (used + fixed_bytes + drafter_bytes) / GIB}


def check_stage(path: Path, cfg: dict) -> dict:
    """Compare the byte model with a moe_encode stage file (jt_* tensors)."""
    import struct
    raw = path.read_bytes()[:8]; n = struct.unpack("<Q", raw)[0]
    hdr = json.loads(path.read_bytes()[8:8 + n])
    E = cfg["n_routed_experts"]; D, I = cfg["hidden_size"], cfg["moe_intermediate_size"]
    out = {}
    for p, N, K in (("gate", I, D), ("up", I, D), ("down", D, I)):
        got = sum(v["data_offsets"][1] - v["data_offsets"][0] for k, v in hdr.items() if f".{p}_proj.jt_" in k and not k.endswith("jt_kbits"))
        nmax = hdr[[k for k in hdr if k.endswith(f"{p}_proj.jt_ocol")][0]]["shape"][-1]
        out[p] = {"file_bytes": got, "model_bytes": jangt_bytes(E, N, K, 2, 12, nmax), "n_out_max": nmax}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--target-gib", type=float, required=True)
    ap.add_argument("--fixed-gib", type=float, required=True, help="non-expert tensors (from the JANGH plan)")
    ap.add_argument("--drafter", default=None, help="drafter safetensors file (its bytes count)")
    ap.add_argument("--check", default=None, help="moe_encode stage file to verify the byte model against")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    dr = Path(a.drafter).stat().st_size if a.drafter else 0
    fixed = int(a.fixed_gib * GIB)
    for name, base, up in (("all-JANGT K2->K3", {"gate": "jt2", "up": "jt2", "down": "jt2"}, {"gate": "jt3", "up": "jt3", "down": "jt3"}),
                           ("hybrid (JANGT gu K2->K3, JANGH dn 2->3)", {"gate": "jt2", "up": "jt2", "down": "jh2"}, {"gate": "jt3", "up": "jt3", "down": "jh3"}),
                           ("JANGH 2->3 (reference)", {"gate": "jh2", "up": "jh2", "down": "jh2"}, {"gate": "jh3", "up": "jh3", "down": "jh3"})):
        print(f"{name:42s}", json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in budget(cfg, fixed, dr, a.target_gib, base, up).items()}))
    if a.check:
        print("byte model vs stage file:", json.dumps(check_stage(Path(a.check), cfg)))


if __name__ == "__main__":
    main()
