"""GLM-5.3-Flash JANGTQ v2 build plan: exact per-tensor formats + heap-MCKP expert bit allocation.

Per-tensor rules (source header shapes, runtime naming):
  layers.45.* (MTP)                              -> dropped
  routed experts (layers.N.mlp.experts.E.*)      -> JANGTQ v2 unit (layer, gate_up) / (layer, down), bits 2|3|4 solved
  KEEP (norms, router, KDA gate producers, conv, A_log, dt_bias, mHC, kpool ape, e_score bias) -> source dtype
  vision (model.visual.*)                        -> bf16 (VL + video kept at source precision)
  2-D quantizable on the MXFP8 grid (census)     -> mxfp8 (group 32, e8m0 scales, OUR packer)
  other 2-D quantizable (incl. embed/lm_head)    -> affine 8-bit g64, bf16 scales+biases
Budget: total tensor payload (text + vision) <= --total-gib (default 96.0 GiB).

  python -m jang_tools.jangh.glm53.plan --model SRC --census grid_census.json --calib calib_tq.json --out plan.json
"""
from __future__ import annotations

import argparse
import heapq
import json
import re
import struct
from collections import Counter
from pathlib import Path

E, D_IN, D_H = 288, 4096, 2048
KEEP_SUFFIXES = ("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight", "A_log", "dt_bias", "o_norm.weight",
                 "e_score_correction_bias", "hc_attn_fn", "hc_attn_base", "hc_attn_scale", "hc_ffn_fn", "hc_ffn_base",
                 "hc_ffn_scale", "index_kpool_compress_ape")
KEEP_CONTAINS = (".f_a_proj.", ".f_b_proj.", ".g_a_proj.", ".g_b_proj.", ".b_proj.", "layernorm", ".norm.",
                 "norm.weight", ".mlp.gate.weight", ".enorm.", ".hnorm.", ".shared_head.")
DT_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "U8": 1, "I8": 1}
UNIT_W = {"gate_up": 1.075, "down": 1.35}   # prior-campaign projection weights (gate 1.15 / up 1.0 averaged; down 1.35)


def read_headers(model_dir: Path) -> dict:
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    shapes = {}
    for f in sorted(set(idx.values())):
        with open(model_dir / f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k, v in h.items():
            if k != "__metadata__":
                shapes[k] = (v["dtype"], v["shape"])
    return shapes


def classify(k: str, dtype: str, shape, mx8_set: set) -> str:
    if re.search(r"layers\.45\.", k):
        return "drop_mtp"
    if re.search(r"\.mlp\.experts\.\d+\.", k):
        return "expert"
    if k.startswith("model.visual."):
        return "vision_bf16"
    if any(k.endswith(s) for s in KEEP_SUFFIXES) or any(c in k for c in KEEP_CONTAINS):
        return "keep"
    if len(shape) != 2 or not k.endswith(".weight"):
        return "keep"
    short = k.replace("model.language_model.", "")
    if short in mx8_set and shape[1] % 32 == 0:
        return "mxfp8"
    if shape[1] % 64 == 0:
        return "affine8"
    return "keep"


def fmt_bytes(fmt: str, dtype: str, shape) -> float:
    n = 1
    for s in shape:
        n *= s
    if fmt == "mxfp8":
        return n + n / 32
    if fmt == "affine8":
        return n + (n / 64) * 4
    if fmt in ("keep", "vision_bf16"):
        return n * DT_BYTES.get(dtype, 2)
    return 0.0


def unit_bytes(unit: str, bits: int) -> int:
    if unit == "gate_up":
        return 2 * E * D_H * D_IN * bits // 8 + 2 * E * D_H * 2
    return E * D_IN * D_H * bits // 8 + E * D_IN * 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--census", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--total-gib", type=float, default=96.0)
    ap.add_argument("--min-bits", type=int, default=2)
    ap.add_argument("--no-mxfp8", action="store_true", help="ablation: every non-expert quantizable tensor affine8")
    a = ap.parse_args()
    shapes = read_headers(Path(a.model))
    census = json.loads(Path(a.census).read_text())
    mx8_set = set() if a.no_mxfp8 else {r["k"] for r in census if r["grid"] == "mx8"}
    fixed = Counter(); per_tensor = {}
    for k, (dt, shp) in shapes.items():
        f = classify(k, dt, shp, mx8_set)
        per_tensor[k] = f
        fixed[f] += fmt_bytes(f, dt, shp)
    fixed_total = sum(v for f, v in fixed.items() if f not in ("expert", "drop_mtp"))
    budget = a.total_gib * 2**30 - fixed_total
    calib = json.loads(Path(a.calib).read_text())
    hulls, state, heap, spent = {}, {}, [], 0.0
    for layer, rec in calib.items():
        for unit in ("gate_up", "down"):
            opts = sorted(((int(o[2:]), rec["units"][o][unit]["nmse"]) for o in rec["units"] if int(o[2:]) >= a.min_bits))
            hull = [(b, unit_bytes(unit, b), n * UNIT_W[unit]) for b, n in opts]
            key = f"{layer}:{unit}"
            hulls[key] = hull; state[key] = 0; spent += hull[0][1]
            if len(hull) > 1:
                heapq.heappush(heap, (-(hull[0][2] - hull[1][2]) / (hull[1][1] - hull[0][1]), key, 1))
    assert spent <= budget, f"minimum allocation {spent/2**30:.2f} GiB exceeds expert budget {budget/2**30:.2f} GiB"
    while heap:
        _, key, i = heapq.heappop(heap)
        if state[key] != i - 1:
            continue
        h = hulls[key]
        delta = h[i][1] - h[i - 1][1]
        if spent + delta > budget:
            continue
        spent += delta; state[key] = i
        if i + 1 < len(h):
            heapq.heappush(heap, (-(h[i][2] - h[i + 1][2]) / (h[i + 1][1] - h[i][1]), key, i + 1))
    alloc = {k: hulls[k][v][0] for k, v in state.items()}
    total = fixed_total + spent
    plan = {
        "format": "jangtq2", "total_gib_budget": a.total_gib,
        "total_bytes": total, "total_gib": total / 2**30,
        "fixed_gib_by_format": {f: v / 2**30 for f, v in fixed.items() if f not in ("expert", "drop_mtp")},
        "experts_gib": spent / 2**30,
        "expert_bits": alloc,
        "expert_bits_histogram": dict(Counter(f"{k.split(':')[1]}:{b}" for k, b in alloc.items())),
        "tensor_formats": per_tensor,
        "awq": "not applied: TQ-objective alpha search gain <= 0.1% on all 42 MoE layers (calib_tq.json)",
    }
    Path(a.out).write_text(json.dumps(plan, indent=1))
    print(f"fixed (non-expert) {fixed_total/2**30:.2f} GiB: " + ", ".join(f"{f} {v/2**30:.2f}" for f, v in fixed.items() if f not in ('expert','drop_mtp')))
    print(f"experts {spent/2**30:.2f} GiB -> TOTAL {total/2**30:.3f} GiB (budget {a.total_gib})")
    print("allocation:", plan["expert_bits_histogram"])
    for L in sorted({int(k.split(':')[0]) for k in alloc}):
        print(f"  L{L:2d} gate_up tq{alloc[f'{L}:gate_up']} down tq{alloc[f'{L}:down']}")


if __name__ == "__main__":
    main()
