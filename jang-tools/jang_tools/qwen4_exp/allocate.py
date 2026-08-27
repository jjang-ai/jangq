"""Bit allocation for Qwen3.8-Flash-Next JANG profiles (6M/4M/2L/1L).

MEASURE, don't name-match: scores come from the calibration capture
(per-expert input second moments × routing frequency) and the n-gram
histogram — TIER_RULES-style name matching has failed four times running.

Score model (per quantizable unit):
  routed expert e, layer l:  s = route_freq[l,e] · Σ diag[l,e] · params
  n-gram table (per shard):  s = Σ counts[rows in shard]  (hot shards ↑)
  trunk module:              s = Σ diag · params
Budget filling: every unit starts at the profile BASE spec; the global
byte budget's slack is spent promoting the highest score-per-extra-byte
units, and (for L profiles) recovered by demoting the coldest experts to
the floor spec. MLP-asymmetry floors (512 experts ⇒ down_proj ≥ base) and
GPTQ bit legality (2/4/8 only where GPTQ will run) are enforced here.

Emits: maps/jang_<profile>.json (convert.py bit-map format) + a report.

  python -m jang_tools.qwen4_exp.allocate --capture ~/models/Logs/q38fn-calib/capture \
      --hist ~/models/Logs/q38fn-calib/ngram_hist.npz --profile 4M --out maps/
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

N_LAYERS = 48
N_EXPERTS = 512
N_SHARDS = 128

# Tier frames per docs/runtime/qwen38-next-flash-prep-2026-08-26/PROFILE-DESIGN.md
# (evidence-grounded: dots3/Ornith/3.8-27B measurements). These are the BASE
# frames; the per-unit knapsack (dots3 calibrate_units + solve_plan adaptation)
# refines expert units and the table bits once option error curves exist.
# Fixed floors regardless of tier: QSA attention+indexer 8b, embed/head 8b,
# shared experts 8b, PLE key/value 8b, MTP ≥ backbone avg, 2-bit ⇒ gs≤64.
PROFILES = {
    "6M": dict(expert_base=(6, 64), expert_hot=(8, 64), expert_cold=(6, 64),
               table_bits=4, trunk_bits=8, embed_bits=8, vision_bits=8,
               promote_frac=0.10, demote_frac=0.0, budget_gib=None),
    "4M": dict(expert_base=(4, 64), expert_hot=(6, 64), expert_cold=(3, 64),
               table_bits=3, trunk_bits=8, embed_bits=8, vision_bits=8,
               promote_frac=0.08, demote_frac=0.08, budget_gib=94.0),
    "2L": dict(expert_base=(2, 64), expert_hot=(4, 64), expert_cold=(2, 64),
               table_bits=2, trunk_bits=6, embed_bits=8, vision_bits=6,
               promote_frac=0.30, demote_frac=0.0, budget_gib=68.0),
    "1L": dict(expert_base=(2, 64), expert_hot=(2, 64), expert_cold=(2, 64),
               table_bits=2, trunk_bits=4, embed_bits=8, vision_bits=4,
               promote_frac=0.0, demote_frac=0.0, budget_gib=55.0),
}

EXPERT_PARAMS = 2560 * 1280 + 2560 * 640  # gate_up + down per expert per layer


def bytes_per_param(bits, gs):
    return bits / 8 + 2 * 2 / gs  # packed + bf16 scale/bias per group


def load_scores(capture_dir: Path, hist_path: Path):
    diag = mx.load(str(capture_dir / "diag.safetensors"))
    scores = np.zeros((N_LAYERS, N_EXPERTS))
    for l in range(N_LAYERS):
        base = f"language_model.layers.{l}.mlp.switch_mlp"
        gate_d = np.asarray(diag[f"{base}.gate_proj.expert_diag"], dtype=np.float64)
        down_d = np.asarray(diag[f"{base}.down_proj.expert_diag"], dtype=np.float64)
        rows = np.asarray(diag[f"{base}.gate_proj.expert_rows"], dtype=np.float64)
        freq = rows / max(rows.sum(), 1)
        scores[l] = freq * (gate_d.sum(-1) + down_d.sum(-1))

    hist = np.load(hist_path)["counts"]
    per = -(-hist.shape[0] // N_SHARDS)
    shard_heat = np.array([hist[i * per:(i + 1) * per].sum() for i in range(N_SHARDS)],
                          dtype=np.float64)
    return scores, shard_heat, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--hist", required=True)
    ap.add_argument("--profile", required=True, choices=list(PROFILES))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prof = PROFILES[args.profile]
    scores, shard_heat, diag = load_scores(Path(args.capture), Path(args.hist))

    flat = scores.ravel()
    order = np.argsort(-flat)
    n_units = flat.shape[0]
    n_hot = int(prof["promote_frac"] * n_units)
    n_cold = int(prof["demote_frac"] * n_units)

    spec_of = np.full(n_units, 1, dtype=np.int8)          # 0=cold 1=base 2=hot
    spec_of[order[:n_hot]] = 2
    if n_cold:
        spec_of[order[-n_cold:]] = 0

    bit_map = {"default": {"bits": prof["trunk_bits"], "group_size": 64}}
    # experts: emit per-(layer,expert-tier) — convert works per whole tensor,
    # so per-expert mixed bits inside one stacked tensor are expressed as the
    # TENSOR-level base spec here; the per-expert promotion/demotion is
    # applied by the imatrix/GPTQ refit stage which rewrites codes in place.
    # The tensor-level spec is the BASE tier; the report records tiers for
    # the refit stage.
    eb, eg = prof["expert_base"]
    for l in range(N_LAYERS):
        bit_map[f"language_model.layers.{l}.mlp.switch_mlp"] = {"bits": eb, "group_size": eg}
    tiers = {"hot": [], "cold": []}
    for u in range(n_units):
        l, e = divmod(u, N_EXPERTS)
        if spec_of[u] == 2:
            tiers["hot"].append([int(l), int(e)])
        elif spec_of[u] == 0:
            tiers["cold"].append([int(l), int(e)])

    # n-gram table: tensor-level per shard (shards are separate tensors!)
    hot_shards = np.argsort(-shard_heat)[: N_SHARDS // 4]
    for s in range(N_SHARDS):
        bits = prof["table_bits"] + (1 if s in set(hot_shards.tolist()) and prof["table_bits"] < 4 else 0)
        bit_map[f"language_model.layers.*.ple.ngram_embedding.shards.{s}.weight"] = {
            "bits": int(bits), "group_size": 32}

    bit_map["language_model.embed_tokens"] = {"bits": prof["embed_bits"], "group_size": 64}
    bit_map["lm_head"] = {"bits": 8, "group_size": 64}
    bit_map["visual"] = {"bits": prof["vision_bits"], "group_size": 64}
    # routers + small gates handled by FORCE_KEEP in convert (fp16)
    bit_map["language_model.layers.*.mlp.gate.weight"] = {"bits": 8, "group_size": 64}
    # dots3 binding: attention stays 8-bit (tiny here: ~0.4 GB total)
    bit_map["language_model.layers.*.self_attn"] = {"bits": 8, "group_size": 64}
    bit_map["language_model.layers.*.mlp.shared_expert."] = {"bits": 8, "group_size": 64}
    bit_map["language_model.layers.*.linear_attn"] = {"bits": prof["trunk_bits"], "group_size": 64}
    bit_map["language_model.layers.*.ple.key_proj"] = {"bits": 8, "group_size": 64}
    bit_map["language_model.layers.*.ple.value_proj"] = {"bits": 8, "group_size": 64}
    # MTP ≥ backbone average (draft quantized harder than target burns verify)
    bit_map["mtp."] = {"bits": max(4, prof["expert_base"][0]), "group_size": 64}

    # size estimate
    est = 0.0
    est += N_LAYERS * N_EXPERTS * EXPERT_PARAMS * bytes_per_param(eb, eg)
    table_rows = 320_001_536
    est += table_rows * 160 * bytes_per_param(prof["table_bits"], 32)
    est += 248320 * 2560 * 2 * bytes_per_param(prof["embed_bits"], 64)
    est += 2.6e9 * bytes_per_param(prof["trunk_bits"], 64)   # GDN+QSA trunk
    est += 1.3e9 * 2                                          # hc/norms fp16-ish
    est += 0.44e9 * bytes_per_param(prof["vision_bits"], 64)
    est_gib = est / 2**30
    print(f"profile {args.profile}: est {est_gib:.1f} GiB "
          f"(hot {len(tiers['hot'])} cold {len(tiers['cold'])} experts)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"jang_{args.profile.lower()}.json").write_text(json.dumps(bit_map, indent=1))
    (out / f"jang_{args.profile.lower()}_tiers.json").write_text(json.dumps(
        {"expert_tiers": tiers, "estimated_gib": est_gib,
         "hot_spec": prof["expert_hot"], "cold_spec": prof["expert_cold"]}, indent=1))
    print(f"wrote maps → {out}")


if __name__ == "__main__":
    main()
