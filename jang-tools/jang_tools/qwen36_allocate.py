"""Turn the Qwen3.6-27B calibration capture into a measured bit allocation.

Score per quantizable module:

    s = tr(H) * ||W||_F^2

`tr(H)` comes from the capture (`sum_c E[x_c^2]`, the Hessian diagonal);
`||W||_F^2` is read from the source weights. The product approximates the output
error a unit relative perturbation of that module would cause, so ranking by it
tells us where bits actually buy coherence — rather than inferring it from the
tensor's *name*, which is what `TIER_RULES` does and what has broken on every
new architecture we've touched.

Emits a per-module bit map under a target size budget, plus the evidence
(scores, traces, norms) so the choice is auditable.

    python -m jang_tools.qwen36_allocate <model_dir> <calib.json> <out.json> \
        [--target-gib 10.0] [--base-bits 2] [--group-size 128] \
        [--min-bits N] [--allowed-bits 4,8] [--attn-min-bits N] \
        [--vision-min-bits N] [--embed-min-bits N] \
        [--target-bpw 4.0] [--gdn-state-bits 16] [--floor-bits 2]

TWO TARGETING MODES
===================
`--target-gib` (legacy) pins every module to `--base-bits` and then spends or
reclaims the difference. Reproduces the shipped JANG_*D maps exactly.

`--target-bpw` (preferred) instead targets an AVERAGE bits-per-weight over the
TEXT trunk and lets the widths fall wherever the measurement says. Vision is
budgeted separately at its floor and excluded from the average, because it is
not read during text decode.

The bpw mode allocates by MARGINAL VALUE rather than by rank. Affine
quantization error falls as 4^-b, so a module's contribution to output error is
approximately `score * 4^-b`, and promoting it from b to b' buys

    gain      = score * (4^-b - 4^-b')
    cost      = numel * (b' - b) / 8            bytes
    value     = gain / cost

Every movable module starts at `--floor-bits`; the highest-value promotion is
taken repeatedly until the budget is spent. Ranking by score alone (what the
legacy path does) ignores both how many bytes a promotion costs and the
diminishing return of each extra bit, so it overspends on large low-density
tensors.

GDN / SSM STATE — why a name-based floor is correct here
========================================================
`tr(H)*||W||_F^2` estimates the error from ONE forward pass. That is the wrong
model for the recurrent-state projections: on this family `linear_attn.in_proj_a`
produces the delta-rule decay and `in_proj_b` the write strength, and their error
is compounded through the recurrence over the whole sequence rather than added
once. Both are shape (48, 5120) — 200x smaller in Frobenius norm than the
`in_proj_qkv` that shares their input and their exact `tr(H)` — so the score
ranks them near the BOTTOM and the demotion pass eats them first.

Holding both at fp16 across all 48 layers costs 23.6 MB. `--gdn-state-bits`
sets that floor (default 16 = fp16, `0` disables it).

`conv1d`, `A_log`, `dt_bias` and the GDN head norm are not Linear layers, so
they never appear in the capture and stay at source precision regardless.

The `--*-min-bits` flags override the name-based FORCED floors below. They are
defaults, not laws: on a speed-targeted budget the floors are the first thing
worth re-measuring.
"""
from __future__ import annotations

import glob
import json
import struct
import sys
from pathlib import Path

import numpy as np

# Modules that must never be driven by the score alone.
# These are DEFAULTS, overridable per run by --{vision,attn,embed}-min-bits.
# They are name-based rules, so on a speed-targeted budget they are exactly the
# thing worth re-measuring rather than inheriting: `attn_min_bits: 8` alone
# spends ~1.7 GB on Qwen3.8-27B's 16 full-attention layers.
FORCED = {
    # vision tower: proven to collapse under aggressive quant / AWQ on this
    # family (QWEN36-A3B-JANGTQ4-COHERENCE-BUG). Floor it.
    "vision_min_bits": 4,
    # full attention is the coherence anchor and is only ~6% of params.
    "attn_min_bits": 8,
    # embeddings / untied head
    "embed_min_bits": 4,
}

# Recurrent-state projections: error compounds through the GDN recurrence
# instead of adding once, so the single-forward score under-ranks them.
# See the module docstring. 23.6 MB total at fp16 across 48 layers.
GDN_STATE = ("linear_attn.in_proj_a", "linear_attn.in_proj_b")

# Modules AWQ can fold into a producing RMSNorm (mirrors
# ornith_awq_fold._FOLD_GROUPS). Everything else — down_proj (GLU product),
# o_proj / out_proj (attention outputs), lm_head, the vision tower — reads an
# activation with no norm partner, so AWQ cannot reach it and the imatrix refit
# is its only correction. With --awq-kappa the water-fill knows this and stops
# spending its cheapest bits on the tensors that have no second line of defence.
AWQ_COVERED = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
    "linear_attn.in_proj_a", "linear_attn.in_proj_b",
    "mlp.gate_proj", "mlp.up_proj",
)
# in_features not divisible by any MLX group size (32/64/128) -> cannot quantize
UNQUANTIZABLE_IN_FEATURES = {4304}


def source_key(path: str) -> str:
    p = path.replace("language_model.model.", "model.language_model.")
    p = p.replace("language_model.lm_head", "lm_head")
    p = p.replace("vision_tower.", "model.visual.")
    return p + ".weight"


def read_norms(src: Path, keys: set[str]) -> dict[str, tuple[float, int]]:
    """||W||_F^2 and numel per tensor, streaming shard by shard."""
    import mlx.core as mx
    out = {}
    for shard in sorted(glob.glob(str(src / "model-*.safetensors"))):
        with open(shard, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        want = [k for k in hdr if k in keys]
        if not want:
            continue
        arrs = mx.load(shard)
        for k in want:
            w = arrs[k].astype(mx.float32)
            fro = float((w * w).sum().item())
            out[k] = (fro, int(w.size))
        del arrs
    return out


def classify(path: str) -> str:
    if path.startswith("vision_tower"):
        return "vision"
    if "self_attn" in path:
        return "attn"
    if "linear_attn" in path:
        return "gdn"
    if ".mlp." in path:
        return "mlp"
    if "lm_head" in path or "embed" in path:
        return "embed"
    return "other"


def main(argv) -> int:
    if len(argv) < 4:
        print(__doc__)
        return 1
    src, calib_json, out_p = Path(argv[1]), Path(argv[2]), Path(argv[3])
    target_gib, base_bits, gs = 10.0, 2, 128
    min_bits = None
    allowed = None
    target_bpw = None
    # 🚨 DEFAULT OFF — MEASURED HARMFUL. Flooring linear_attn.in_proj_a /
    # in_proj_b to fp16 is *strictly more precision* and yet produced a bundle
    # that immediately emits <|im_end|> (empty output) on every prompt.
    # Controlled A/B: two bitmaps differing in EXACTLY those 96 tensors
    # (4/5-bit vs bf16), everything else byte-identical --
    #     quantized  -> KL 0.00996, top-1 97.01%, coherent
    #     fp16 floor -> KL 0.81393, top-1 89.58%, EMPTY OUTPUT
    # Verified NOT the obvious causes: the AWQ fold is intact on disk
    # (|W_bundle - W*s| = 3.6e-4 vs |W_bundle - W| = 8.1e-2), the modules load
    # as plain bf16 Linear, input_layernorm / A_log / dt_bias / conv1d are
    # byte-identical between the two bundles, tensor sets and index are
    # complete, and the logits are finite. Mechanism UNRESOLVED. Do not enable
    # without an end-to-end generation check -- KL alone understates it and
    # every structural check passes.
    gdn_state_bits = 0
    floor_bits = 2
    awq_kappa_p = None
    floors = dict(FORCED)
    # bpw mode drops the name-based attn floor and lets the measurement fund
    # attention, UNLESS the caller pins it explicitly.
    seen_floor_flags: set[str] = set()
    for i, a in enumerate(argv):
        if a == "--target-gib":
            target_gib = float(argv[i + 1])
        if a == "--base-bits":
            base_bits = int(argv[i + 1])
        if a == "--group-size":
            gs = int(argv[i + 1])
        if a == "--min-bits":
            min_bits = int(argv[i + 1])
        if a == "--allowed-bits":
            allowed = sorted({int(x) for x in argv[i + 1].split(",")})
        if a == "--attn-min-bits":
            floors["attn_min_bits"] = int(argv[i + 1])
            seen_floor_flags.add("attn_min_bits")
        if a == "--vision-min-bits":
            floors["vision_min_bits"] = int(argv[i + 1])
            seen_floor_flags.add("vision_min_bits")
        if a == "--embed-min-bits":
            floors["embed_min_bits"] = int(argv[i + 1])
        if a == "--target-bpw":
            target_bpw = float(argv[i + 1])
        if a == "--gdn-state-bits":
            gdn_state_bits = int(argv[i + 1])
        if a == "--floor-bits":
            floor_bits = int(argv[i + 1])
        if a == "--awq-kappa":
            awq_kappa_p = argv[i + 1]
    if min_bits is None:
        min_bits = base_bits

    stats = json.loads(calib_json.read_text())["stats"]
    keys = {source_key(p) for p in stats}
    print(f"  reading ||W||_F^2 for {len(keys)} tensors ...", flush=True)
    norms = read_norms(src, keys)
    print(f"  read {len(norms)}", flush=True)

    mods = []
    for path, st in stats.items():
        k = source_key(path)
        if k not in norms:
            continue
        fro, numel = norms[k]
        tr = st["trace"]
        mods.append({
            "path": path, "group": classify(path),
            "trace": tr, "fro2": fro, "numel": numel,
            "in_features": st["in_features"],
            "score": tr * fro,
            "quantizable": st["in_features"] not in UNQUANTIZABLE_IN_FEATURES
                           and st["in_features"] % gs == 0,
        })

    # normalise score to a 0..1 rank
    qs = [m for m in mods if m["quantizable"]]
    order = sorted(qs, key=lambda m: -m["score"])
    for i, m in enumerate(order):
        m["rank"] = i
        m["pct"] = i / max(len(order) - 1, 1)

    # ── allocate ──────────────────────────────────────────────────────────
    # Floors first (safety), then spend the remaining budget top-down by score.
    for m in mods:
        g = m["group"]
        if not m["quantizable"]:
            m["bits"] = 16          # fp16 passthrough (e.g. in_features 4304)
            m["reason"] = "in_features not divisible by an MLX group size"
        elif g == "attn":
            m["bits"] = max(floors["attn_min_bits"], base_bits)
            m["reason"] = "coherence anchor (forced)"
        elif g == "vision":
            # floor, never a ceiling: on 6-bit+ profiles vision keeps base width
            m["bits"] = max(floors["vision_min_bits"], base_bits)
            m["reason"] = "vision floor (collapse precedent)"
        elif g == "embed":
            m["bits"] = max(floors["embed_min_bits"], base_bits)
            m["reason"] = "embed/head floor"
        else:
            m["bits"] = base_bits; m["reason"] = "base"

    def total_gib():
        b = 0
        for m in mods:
            if m["bits"] >= 16:
                b += m["numel"] * 2
            else:
                b += m["numel"] * (m["bits"] + 32.0 / gs) / 8
        return b / 2**30

    base_size = total_gib()
    # promote highest-score MLP/GDN modules while budget allows.
    # `--allowed-bits 4,8` restricts the ladder to widths a downstream stage can
    # handle: `gptq_mlx` packs with `vals_per_u32 = 32 // bits`, so 3/5/6-bit
    # mis-pack silently. A map that may be GPTQ'd must be built from {2,4,8}.
    ALLOWED = allowed if allowed else [2, 3, 4, 5, 6, 8]

    def next_width(b):
        ups = [w for w in ALLOWED if w > b]
        return ups[0] if ups else None

    def prev_width(b):
        downs = [w for w in ALLOWED if w < b]
        return downs[-1] if downs else None

    # ── average-bpw water-fill ────────────────────────────────────────────
    # Start every movable module at its floor, then repeatedly buy the single
    # cheapest unit of error reduction until the text budget is spent. See the
    # module docstring for the marginal-value model.
    waterfill_log = None
    if target_bpw is not None:
        MOVABLE = ("mlp", "gdn", "attn", "embed")

        # Error-energy curve. Default is the analytic affine model (error ~ 2^-b,
        # so energy ~ 4^-b). --awq-kappa replaces it with curves MEASURED on this
        # model's real weights through the real activation distribution, and
        # splits them by whether AWQ can reach the module.
        kap = None
        if awq_kappa_p:
            kap = json.loads(Path(awq_kappa_p).read_text())
            print(f"  awq-aware energy curve: {awq_kappa_p}")
            print(f"    kappa by bits: "
                  f"{ {int(k): round(v,4) for k,v in kap['kappa'].items()} }")

        def energy(m, b):
            """Relative error ENERGY of module m at width b (1.0 = fully lost)."""
            if b >= 16:
                return 0.0
            if kap is None:
                return 4.0 ** -b
            tbl = (kap["err_awq"] if m.get("awq_covered") else kap["err_plain"])
            e = tbl.get(str(b))
            if e is None:                      # width outside the measured set
                lo = min(tbl, key=lambda k: abs(int(k) - b))
                e = tbl[lo] * (2.0 ** (int(lo) - b))
            return e * e

        text = [m for m in mods if m["group"] != "vision"]
        text_numel = sum(m["numel"] for m in text)
        budget = target_bpw * text_numel / 8.0          # bytes

        for m in mods:
            m["awq_covered"] = any(m["path"].endswith(s) for s in AWQ_COVERED)
            if not m["quantizable"]:
                m["lo"] = 16
                m["reason"] = "in_features not divisible by an MLX group size"
            elif any(m["path"].endswith(s) for s in GDN_STATE):
                m["lo"] = gdn_state_bits
                m["reason"] = (f"GDN recurrent state floored at {gdn_state_bits} "
                               "(error compounds through the recurrence; the "
                               "single-forward score under-ranks it)")
            elif m["group"] == "vision":
                m["lo"] = max(floors["vision_min_bits"], min(ALLOWED))
                m["reason"] = "vision floor (collapse precedent)"
            elif m["group"] == "embed":
                m["lo"] = floors["embed_min_bits"]
                m["reason"] = "embed/head floor"
            elif m["group"] == "attn" and "attn_min_bits" in seen_floor_flags:
                m["lo"] = floors["attn_min_bits"]
                m["reason"] = "coherence anchor (explicit --attn-min-bits)"
            else:
                m["lo"] = floor_bits
                m["reason"] = "floor"
            # snap the floor onto the legal ladder
            if m["lo"] < 16:
                legal = [w for w in ALLOWED if w >= m["lo"]]
                m["lo"] = legal[0] if legal else 16
            m["bits"] = m["lo"]
            m["movable"] = (m["quantizable"] and m["group"] in MOVABLE
                            and not any(m["path"].endswith(s) for s in GDN_STATE))

        def text_bytes():
            b = 0.0
            for m in text:
                b += (m["numel"] * 2 if m["bits"] >= 16
                      else m["numel"] * (m["bits"] + 32.0 / gs) / 8)
            return b

        start_bytes = text_bytes()
        if start_bytes > budget:
            # ADVISORY only — we still emit the map. The floors are what they
            # are; the caller asked for this bpw and gets the closest layout.
            print(f"  !! floors alone are {start_bytes*8/text_numel:.3f} bpw, "
                  f"above the {target_bpw:.3f} bpw target — emitting the floor "
                  f"layout (nothing to spend)")

        def cost_of(m, b2):
            cur = (m["numel"] * 2 if m["bits"] >= 16
                   else m["numel"] * (m["bits"] + 32.0 / gs) / 8)
            nxt = (m["numel"] * 2 if b2 >= 16
                   else m["numel"] * (b2 + 32.0 / gs) / 8)
            return nxt - cur

        # ── vision tower: its own water-fill to the SAME bpw target ──────
        # The tower is not read during text decode, so it is excluded from the
        # text average — but a 6-bit-class bundle should still ship a 6-bit-class
        # vision tower. Budget it over its QUANTIZABLE modules only: the 27
        # `linear_fc2` tensors (in_features 4304) are fp16 passthrough and would
        # otherwise drag the tower's average above every target on their own,
        # leaving the other 83 modules stuck at the floor in every tier.
        vis = [m for m in mods if m["group"] == "vision" and m["quantizable"]]
        vis_steps = 0
        if vis:
            vis_numel = sum(m["numel"] for m in vis)
            vis_budget = max(target_bpw, floors["vision_min_bits"]) * vis_numel / 8.0

            def vis_bytes():
                return sum(m["numel"] * (m["bits"] + 32.0 / gs) / 8 for m in vis)

            spent_v = vis_bytes()
            while True:
                best, best_val, best_b2, best_cost = None, 0.0, None, 0.0
                for m in vis:
                    b2 = next_width(m["bits"])
                    if b2 is None or b2 >= 16:
                        continue
                    c = m["numel"] * (b2 - m["bits"]) / 8
                    if c <= 0 or spent_v + c > vis_budget:
                        continue
                    val = m["score"] * (energy(m, m["bits"]) - energy(m, b2)) / c
                    if val > best_val:
                        best, best_val, best_b2, best_cost = m, val, b2, c
                if best is None:
                    break
                best["bits"] = best_b2
                best["reason"] = f"vision water-fill -> {best_b2}"
                spent_v += best_cost
                vis_steps += 1

        steps = 0
        spent = text_bytes()
        while True:
            best, best_val, best_b2, best_cost = None, 0.0, None, 0.0
            for m in text:
                if not m["movable"]:
                    continue
                b2 = next_width(m["bits"])
                if b2 is None:
                    continue
                c = cost_of(m, b2)
                if c <= 0 or spent + c > budget:
                    continue
                gain = m["score"] * (energy(m, m["bits"]) - energy(m, b2))
                val = gain / c
                if val > best_val:
                    best, best_val, best_b2, best_cost = m, val, b2, c
            if best is None:
                break
            best["bits"] = best_b2
            best["reason"] = (f"water-fill -> {best_b2} "
                              f"(rank {best['rank']}/{len(order)})")
            spent += best_cost
            steps += 1

        waterfill_log = {
            "target_bpw": target_bpw, "text_numel": text_numel,
            "budget_bytes": budget, "floor_bytes": start_bytes,
            "final_bytes": text_bytes(),
            "final_text_bpw": text_bytes() * 8 / text_numel,
            "promotion_steps": steps,
            "vision_promotion_steps": vis_steps,
            "vision_bpw": (sum(m["numel"] * (m["bits"] + 32.0 / gs) / 8
                               for m in vis) * 8 / sum(m["numel"] for m in vis))
                          if vis else None,
            "gdn_state_bits": gdn_state_bits, "floor_bits": floor_bits,
            "awq_kappa": awq_kappa_p,
            "residual_error_energy": sum(
                m["score"] * energy(m, m["bits"]) for m in text),
        }

    # ── demote (optional) ─────────────────────────────────────────────────
    # Walk the LOWEST-scoring MLP/GDN modules down, one legal width per sweep,
    # until the projection fits the budget. Sweeping (rather than draining each
    # module to min_bits in turn) keeps the demotion spread across the tail
    # instead of gutting a handful of modules outright.
    demoted = 0
    if target_bpw is None and min_bits < base_bits:
        tail = [m for m in reversed(order) if m["group"] in ("mlp", "gdn")]
        while total_gib() > target_gib:
            moved = False
            for m in tail:
                if total_gib() <= target_gib:
                    break
                down = prev_width(m["bits"])
                if down is None or down < min_bits:
                    continue
                m["bits"] = down
                m["reason"] = (
                    f"demoted -> {down} to fund budget "
                    f"(rank {m['rank']}/{len(order)})"
                )
                demoted += 1
                moved = True
            if not moved:
                break                 # whole tail is at min_bits; budget is what it is

    promoted = 0
    if target_bpw is None:
        for m in order:
            if m["group"] not in ("mlp", "gdn") or m["bits"] != base_bits:
                continue
            up1 = next_width(base_bits)
            if up1 is None:
                break                 # base is already 8 — nothing to promote to
            cost = m["numel"] * (up1 - base_bits) / 8 / 2**30
            if total_gib() + cost > target_gib:
                continue
            m["bits"] = up1
            m["reason"] = f"promoted {base_bits}->{up1} (rank {m['rank']}/{len(order)})"
            promoted += 1
            up2 = next_width(up1)
            if up2 is not None and m["pct"] < 0.10:
                cost2 = m["numel"] * (up2 - up1) / 8 / 2**30
                if total_gib() + cost2 <= target_gib:
                    m["bits"] = up2
                    m["reason"] = f"promoted {base_bits}->{up2} (top decile, rank {m['rank']})"
    else:
        promoted = waterfill_log["promotion_steps"]

    final = total_gib()
    import collections
    dist = collections.Counter(m["bits"] for m in mods)
    print(f"\n  base @ {base_bits}-bit gs{gs}: {base_size:.2f} GiB")
    print(f"  floors                : {floors}")
    print(f"  allowed widths        : {ALLOWED}")
    if target_bpw is not None:
        w = waterfill_log
        print(f"  MODE                  : average-bpw water-fill")
        print(f"  text trunk            : {w['text_numel']/1e9:.3f} B params")
        print(f"  floors alone          : "
              f"{w['floor_bytes']*8/w['text_numel']:.3f} bpw")
        print(f"  achieved TEXT bpw     : {w['final_text_bpw']:.3f}  "
              f"(target {target_bpw:.3f})")
        print(f"  text trunk bytes      : {w['final_bytes']/2**30:.2f} GiB")
        print(f"  promotion steps       : {w['promotion_steps']}")
        if w.get("vision_bpw"):
            print(f"  vision quantizable    : {w['vision_bpw']:.3f} bpw "
                  f"({w['vision_promotion_steps']} steps, floor "
                  f"{floors['vision_min_bits']})")
        print(f"  gdn state floor       : {gdn_state_bits}-bit "
              f"(in_proj_a / in_proj_b)")
    else:
        print(f"  after alloc           : {final:.2f} GiB  (target {target_gib})")
        print(f"  demotion steps        : {demoted}  (min-bits {min_bits})")
        print(f"  promoted modules      : {promoted}")
    print(f"  whole-file projection : {final:.2f} GiB")
    print(f"  bit distribution      : {dict(sorted(dist.items()))}")
    print()
    bygrp = collections.defaultdict(collections.Counter)
    for m in mods:
        bygrp[m["group"]][m["bits"]] += 1
    for g, c in sorted(bygrp.items()):
        print(f"    {g:8s} {dict(sorted(c.items()))}")

    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps({
        "source": str(src), "target_gib": target_gib, "base_bits": base_bits,
        "group_size": gs, "projected_gib": final,
        "min_bits": min_bits, "floors": floors, "allowed_bits": ALLOWED,
        "demotion_steps": demoted, "promoted_modules": promoted,
        "waterfill": waterfill_log,
        "gdn_state_floor": {"bits": gdn_state_bits, "suffixes": list(GDN_STATE)},
        "method": ("marginal-value water-fill on tr(H)*||W||_F^2 * 4^-b"
                   if target_bpw is not None
                   else "hessian-trace x frobenius (tr(H)*||W||_F^2)"),
        "modules": sorted(mods, key=lambda m: -m["score"]),
    }, indent=1))
    print(f"\n  bit map -> {out_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
