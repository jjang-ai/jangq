"""Measured bit allocation for Ornith-1.5-35B-A3B (`qwen3_5_moe`).

Created by Jinho Jang (eric@osaurus.ai) — 2026-08-19.

`qwen36_allocate` was written for the DENSE qwen3_5 family and cannot resolve
MoE routed experts. Its `source_key()` produces
`model.language_model.layers.N.mlp.switch_mlp.gate_proj.weight`, but the source
checkpoint stores the experts **pre-stacked and gate/up FUSED**:

    model.language_model.layers.N.mlp.experts.gate_up_proj   (256, 1024, 2048)
    model.language_model.layers.N.mlp.experts.down_proj      (256,  512, 2048)

The unresolved keys hit a bare `continue`, so the allocator silently emitted a
plan covering **501 of 621 modules — 8 % of the parameters** — and reported a
"1.55 GiB" budget for a 36 B model without raising anything. Measured, not
hypothetical.

This module:
  * maps `switch_mlp.{gate,up,down}_proj` onto the fused/stacked source tensors,
    accounting the gate and up HALVES of `gate_up_proj` separately (which is
    also how `--split-fused-gate-up` quantizes them);
  * applies the same FORCED floors as the dense allocator, plus a routed-expert
    floor, because `moe_intermediate_size` here is only **512** — four times
    narrower than the openPangu experts that collapsed at 2-bit;
  * 🚨 **refuses** if capture modules fail to resolve, instead of continuing.
    Silent partial coverage is the single most repeated failure in this
    pipeline (dense hooks missing SwitchLinear, mlx_vlm-vs-mlx_lm class
    identity, the imatrix refit reverting AWQ). A loud stop is cheap; a
    confident bit map over 8 % of a model is not.

    python -m jang_tools.ornith_moe_allocate <src> <calib.json> <out.json> \
        [--target-gib 19.0] [--base-bits 4] [--group-size 64]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .qwen36_allocate import read_norms, UNQUANTIZABLE_IN_FEATURES

# Same spirit as the dense allocator's FORCED table.
FORCED = {
    "vision_min_bits": 4,
    "attn_min_bits": 8,
    "embed_min_bits": 4,
    # Always-active, tiny, and error compounds through every layer.
    "shared_expert_min_bits": 8,
    "router_min_bits": 8,
    # moe_intermediate_size is 512 on this model. openPangu collapsed at 2-bit
    # with 2048-wide experts; do not put a 512-wide expert below 3.
    "routed_expert_min_bits": 3,
}


def source_key(path: str) -> tuple[str, str | None]:
    """Capture path -> (source tensor key, half) where half is 'gate'|'up'|None."""
    p = path.replace("language_model.model.", "model.language_model.")
    p = p.replace("language_model.lm_head", "lm_head")
    p = p.replace("vision_tower.", "model.visual.")
    # NOTE: the stacked expert tensors carry NO ".weight" suffix — they are
    # 3-D parameters (num_experts, out, in), not nn.Linear modules.
    if ".mlp.switch_mlp.gate_proj" in p:
        return p.replace(".mlp.switch_mlp.gate_proj",
                         ".mlp.experts.gate_up_proj"), "gate"
    if ".mlp.switch_mlp.up_proj" in p:
        return p.replace(".mlp.switch_mlp.up_proj",
                         ".mlp.experts.gate_up_proj"), "up"
    if ".mlp.switch_mlp.down_proj" in p:
        return p.replace(".mlp.switch_mlp.down_proj",
                         ".mlp.experts.down_proj"), None
    return p + ".weight", None


def classify(path: str) -> str:
    if path.startswith("vision_tower"):
        return "vision"
    if "shared_expert" in path:
        return "shared_expert"
    if "switch_mlp" in path or ".experts." in path:
        return "routed_expert"
    if "self_attn" in path:
        return "attn"
    if "linear_attn" in path:
        return "gdn"
    if path.endswith(".mlp.gate") or ".mlp.gate." in path:
        return "router"
    if "lm_head" in path or "embed" in path:
        return "embed"
    if ".mlp." in path:
        return "mlp"
    return "other"


def main(argv) -> int:
    if len(argv) < 4:
        print(__doc__)
        return 1
    src, calib_json, out_p = Path(argv[1]), Path(argv[2]), Path(argv[3])
    target_gib, base_bits, gs = 19.0, 4, 64
    for i, a in enumerate(argv):
        if a == "--target-gib":
            target_gib = float(argv[i + 1])
        if a == "--base-bits":
            base_bits = int(argv[i + 1])
        if a == "--group-size":
            gs = int(argv[i + 1])

    stats = json.loads(calib_json.read_text())["stats"]
    keymap = {p: source_key(p) for p in stats}
    wanted = {k for k, _ in keymap.values()}
    print(f"  reading ||W||_F^2 for {len(stats)} modules "
          f"({len(wanted)} distinct source tensors) ...", flush=True)
    norms = read_norms(src, wanted)

    missing = sorted({p for p, (k, _) in keymap.items() if k not in norms})
    if missing:
        print(f"  !! {len(missing)} of {len(stats)} capture modules did not "
              f"resolve to a source tensor. REFUSING — a partial bit map is "
              f"how a 36B model gets 'allocated' at 1.55 GiB.")
        for p in missing[:8]:
            print(f"       {p}  ->  {keymap[p][0]}")
        return 2
    print(f"  resolved {len(stats)}/{len(stats)} modules", flush=True)

    mods = []
    for path, st in stats.items():
        k, half = keymap[path]
        fro, numel = norms[k]
        if half:                      # gate/up share one fused tensor
            fro, numel = fro / 2.0, numel // 2
        mods.append({
            "path": path, "group": classify(path), "trace": st["trace"],
            "fro2": fro, "numel": numel, "in_features": st["in_features"],
            "score": st["trace"] * fro,
            "source_key": k, "fused_half": half,
            "quantizable": st["in_features"] not in UNQUANTIZABLE_IN_FEATURES
                           and st["in_features"] % gs == 0,
        })

    qs = [m for m in mods if m["quantizable"]]
    order = sorted(qs, key=lambda m: -m["score"])
    for i, m in enumerate(order):
        m["rank"], m["pct"] = i, i / max(len(order) - 1, 1)

    def size_gib(ms):
        return sum(m["numel"] * (m["bits"] + 32.0 / gs) / 8 for m in ms) / 2**30

    for m in mods:
        g = m["group"]
        if not m["quantizable"]:
            m["bits"], m["reason"] = 16, "in_features not divisible by an MLX group size"
        elif g == "attn":
            m["bits"], m["reason"] = max(FORCED["attn_min_bits"], base_bits), "coherence anchor (forced)"
        elif g in ("shared_expert", "router"):
            m["bits"] = max(FORCED[f"{g}_min_bits"], base_bits)
            m["reason"] = "always-active / routing (forced)"
        elif g == "vision":
            m["bits"], m["reason"] = max(FORCED["vision_min_bits"], base_bits), "vision floor"
        elif g == "embed":
            m["bits"], m["reason"] = max(FORCED["embed_min_bits"], base_bits), "embedding floor"
        elif g == "routed_expert":
            m["bits"] = max(FORCED["routed_expert_min_bits"], base_bits)
            m["reason"] = "routed expert (moe_intermediate=512 floor)"
        else:
            m["bits"], m["reason"] = base_bits, "base"

    base = size_gib(mods)
    print(f"\n  base @ {base_bits}-bit gs{gs}: {base:.2f} GiB", flush=True)

    # spend the remaining budget top-down by measured score
    promoted = 0
    for m in order:
        if size_gib(mods) >= target_gib:
            break
        if m["bits"] >= 8 or not m["quantizable"]:
            continue
        nxt = {2: 3, 3: 4, 4: 5, 5: 6, 6: 8}.get(m["bits"])
        if nxt is None:
            continue
        prev = m["bits"]
        m["bits"] = nxt
        if size_gib(mods) > target_gib:
            m["bits"] = prev
            continue
        m["reason"] = f"promoted {prev}->{nxt} by score rank {m['rank']}"
        promoted += 1

    import collections
    dist = collections.Counter(m["bits"] for m in mods)
    print(f"  after promotion       : {size_gib(mods):.2f} GiB  (target {target_gib})")
    print(f"  promoted modules      : {promoted}")
    print(f"  bit distribution      : {dict(sorted(dist.items()))}\n")
    per = collections.defaultdict(collections.Counter)
    for m in mods:
        per[m["group"]][m["bits"]] += 1
    for g in sorted(per):
        print(f"    {g:<15}{dict(sorted(per[g].items()))}")

    out_p.write_text(json.dumps({
        "source": str(src), "group_size": gs, "base_bits": base_bits,
        "target_gib": target_gib, "estimated_gib": size_gib(mods),
        "modules": mods,
    }, indent=1))
    print(f"\n  bit map -> {out_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
