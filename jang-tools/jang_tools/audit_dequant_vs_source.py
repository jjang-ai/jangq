"""Post-build quant fidelity audit — dequantize a sample of tensors from a
JANG bundle and compare to the bf16 source. Ship gate for every JANG bundle
per feedback_structural_verification_not_enough.

Fails the ship if any sampled tensor exceeds the per-bit-level rel-err
tolerance. Structural pass (all tensors present) is not enough — this catches
the class of bug where the allocator silently downgrades a critical tensor
to too few bits (e.g. openpangu-v2 `attn_mhc.phi` at 2-bit with 62% error).

Usage:
    python3 -m jang_tools.audit_dequant_vs_source <bundle_dir> <source_dir>

Tolerance table (rel err = mean|deq-src| / mean|src|) — calibrated to what
affine RTN quantization actually delivers in practice, matches GGUF Q*_K numbers.
Overly-tight tolerances flag routed-expert 2/3-bit weights that are at the
designed noise floor (they always noise this much; the model works anyway).
Loose enough to catch genuine bugs (e.g. a critical tensor assigned 2-bit when
it should be 8-bit — that's a 60%+ error, easily distinguished from 45%
design-intended noise on a routed up_proj).

    8-bit  :  2%
    6-bit  :  5%
    4-bit  : 15%
    3-bit  : 25%
    2-bit  : 50%
    fp16   : 0.1% (passthrough — should be near-exact)

Exit code 0 = clean ship. Exit code 1 = fail (any tensor over tolerance).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

TOL = {2: 0.50, 3: 0.25, 4: 0.15, 6: 0.05, 8: 0.02, 16: 0.001}


def _load_from(bundle_dir: Path, wm: dict, key: str):
    return mx.load(str(bundle_dir / wm[key]))[key]


def _rel_err(A: mx.array, B: mx.array) -> float:
    A = A.astype(mx.float32)
    B = B.astype(mx.float32)
    return float(mx.mean(mx.abs(A - B)) / (mx.mean(mx.abs(B)) + 1e-12))


def _build_test_set(config: dict, swm: dict) -> list[tuple[str, str, str]]:
    """Compose (label, source_key, output_key) triples.

    Covers: embed, lm_head, MLA/GQA attention, MoE gate/up/down at two mid
    layers (one first-half, one second-half — different tiers), a shared
    expert, dense MLP layers, MTP-specific tensors, and any family-specific
    control tensor via a name-heuristic scan.
    """
    tc = config.get("text_config", config)
    n_layers = int(tc.get("num_hidden_layers", 0) or 0)
    n_mtp = int(tc.get("num_nextn_predict_layers", 0) or tc.get("mtp_num_hidden_layers", 0) or 0)
    n_experts = int(tc.get("n_routed_experts", tc.get("num_local_experts", tc.get("num_experts", 0))) or 0)

    tests: list[tuple[str, str, str]] = []
    if "model.embed_tokens.weight" in swm:
        tests.append(("embed_tokens", "model.embed_tokens.weight", "model.embed_tokens.weight"))
    if "lm_head.weight" in swm:
        tests.append(("lm_head", "lm_head.weight", "lm_head.weight"))

    # MLA / GQA attention on 4 layers spread across depth
    if n_layers:
        spread = sorted({0, n_layers // 4, n_layers // 2, (3 * n_layers) // 4, n_layers - 1})
        for li in spread:
            for tail in (
                "self_attn.q_a_proj.weight",
                "self_attn.q_proj.weight",
                "self_attn.o_proj.weight",
            ):
                k = f"model.layers.{li}.{tail}"
                if k in swm:
                    tests.append((f"L{li} {tail.split('.')[-1].replace('.weight','')}", k, k))
                    break

    # Routed experts at two layers — sample expert 0 (source per-expert)
    if n_experts > 0:
        for li in (n_layers // 4, (3 * n_layers) // 4):
            for proj in ("gate", "up", "down"):
                src = f"model.layers.{li}.mlp.experts.0.{proj}_proj.weight"
                out = f"model.layers.{li}.mlp.switch_mlp.{proj}_proj.weight"
                if src in swm:
                    tests.append((f"L{li} routed[0].{proj}", src, out))
        # Shared expert
        for li in (n_layers // 3,):
            k = f"model.layers.{li}.mlp.shared_experts.down_proj.weight"
            if k in swm:
                tests.append((f"L{li} shared_expert.down_proj", k, k))
        # Router
        for li in (n_layers // 3,):
            k = f"model.layers.{li}.mlp.gate.weight"
            if k in swm:
                tests.append((f"L{li} mlp.gate (router)", k, k))

    # Dense MLP layers (first_k_dense_replace)
    n_dense = int(tc.get("first_k_dense_replace", 0) or 0)
    for li in range(min(n_dense, 2)):
        for tail in ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"):
            k = f"model.layers.{li}.{tail}"
            if k in swm:
                tests.append((f"L{li} dense {tail.split('.')[-2]}", k, k))

    # MTP-specific tensors (indexed layers at end)
    for j in range(n_mtp):
        li = n_layers + j
        for tail in (
            "eh_proj.weight",
            "embed_tokens.weight",
            "shared_head.head.weight",
        ):
            k = f"model.layers.{li}.{tail}"
            if k in swm:
                tests.append((f"L{li} MTP {tail.replace('.weight','')}", k, k))

    # Family-specific control tensors (heuristic scan across a mid-layer)
    mid = max(n_layers // 2, 1)
    stem = f"model.layers.{mid}."
    control_hits = 0
    for k in swm:
        if not k.startswith(stem):
            continue
        low = k.lower()
        if any(
            p in low
            for p in (
                "attn_mhc_module.phi",
                "mlp_mhc_module.phi",
                "indexer.wq_b",
                "indexer.wk",
                "indexer.weights_proj",
                "phi.weight",  # generic phi projections
            )
        ) and k.endswith(".weight"):
            tests.append((f"L{mid} ctl {k.split('.')[-2]}", k, k))
            control_hits += 1
            if control_hits >= 3:
                break

    # Top-level merge modules
    for k in list(swm)[:2000]:
        low = k.lower()
        if k.startswith("model.merge_") and k.endswith(".weight"):
            tests.append((f"top {k.split('.')[-2]}", k, k))
            break

    return tests


def _fallback_test_set(swm: dict, owm: dict, num_experts: int, want: int = 24
                       ) -> list[tuple[str, str, str]]:
    """Family-agnostic sample, used when the hardcoded branches match nothing.

    Every branch in `_build_test_set` keys on transformers-style names
    (`model.layers.{i}.self_attn.q_proj.weight`, `...mlp.experts.0.gate_proj...`).
    A family with its own naming scheme matches NONE of them and the audit then
    runs on an EMPTY sample and reports success — the Round-2 openpangu-v2
    failure mode (`shared_head.head` at 2-bit passed a 32/32 sampled audit
    because it was outside the sample) in its most extreme form.

    Verified case: Inkling names attention `model.llm.layers.N.attn.wq_du.weight`
    and fuses routed experts into `...mlp.experts.w13_weight`, so the hardcoded
    set yields zero tests.

    Strategy: normalize layer indices out of every source key, keep patterns that
    are quantizable (present in the bundle with a `.scales` sibling, or at least
    present in the bundle), and take one representative per pattern — CRITICAL
    tier first so the most dangerous tensors are always covered.
    """
    from .allocate import Tier, classify_tensor

    by_pattern: dict[str, list[str]] = {}
    for k in swm:
        by_pattern.setdefault(re.sub(r"\.\d+\.", ".N.", k), []).append(k)

    ranked: list[tuple[int, str, str]] = []
    for pattern, members in by_pattern.items():
        # Prefer a mid-stack member: extremes are the least representative.
        rep = sorted(members)[len(members) // 2]
        if _resolve_out_key(rep, owm) is None:
            continue
        tier = classify_tensor(rep, num_experts=num_experts)
        # CRITICAL first (0), then IMPORTANT (1), then COMPRESS (2).
        rank = {Tier.CRITICAL: 0, Tier.IMPORTANT: 1}.get(tier, 2)
        ranked.append((rank, pattern, rep))

    ranked.sort(key=lambda t: (t[0], t[1]))
    # Guarantee the biggest tensors are in the sample regardless of the cap.
    # Ranking CRITICAL-first is right for catching under-allocation, but it
    # pushed the routed experts (COMPRESS by design, and the ONLY 2/3-bit
    # tensors in the bundle) past `want` -- i.e. the audit covered everything
    # except the 97% of parameters whose bit width is actually in question.
    big = [r for r in ranked if "experts" in r[2] or "switch_mlp" in r[2]]
    rest = [r for r in ranked if r not in big]
    ranked = big + rest
    out: list[tuple[str, str, str]] = []
    for _, pattern, rep in ranked[:want]:
        label = pattern.replace("model.", "").replace(".weight", "")
        out.append((label[:38], rep, _resolve_out_key(rep, owm) or rep))
    return out


def _resolve_out_key(src_key: str, owm: dict) -> str | None:
    """Bundle key for a source key. Quantized modules gain a trailing '.weight'
    (manifest weight_key = f'{base}.weight'), which source names like
    `...experts.w13_weight` do not have."""
    for cand in (src_key, src_key + ".weight"):
        if cand in owm:
            return cand
    return None


def _coverage_gaps(swm: dict, tested: set[str], owm: dict) -> list[str]:
    """Source tensor patterns that are in the bundle but got NO dequant test."""
    tested_patterns = {re.sub(r"\.\d+\.", ".N.", k) for k in tested}
    gaps: dict[str, int] = {}
    for k in swm:
        if not k.endswith((".weight", ".proj")) or k not in owm:
            continue
        p = re.sub(r"\.\d+\.", ".N.", k)
        if p not in tested_patterns:
            gaps[p] = gaps.get(p, 0) + 1
    return [f"{p} ({n} tensors)" for p, n in sorted(gaps.items())]


def _census_tier_check(qcfg: dict, num_experts: int) -> list[str]:
    """Full census of the quantization map — not a sample.

    Classifies EVERY quantized module with the allocator's own classify_tensor
    and flags any CRITICAL-tier module that received < 6 bits. This is the
    check that would have caught `shared_head.head` at 2-bit (openpangu-v2
    build 3): the sampled dequant audit passed because the broken tensor was
    outside the sample set. Costs nothing — reads only config.json.

    Routed-expert names (`switch_mlp` / `.experts.`) are excluded: they are
    COMPRESS-by-design and governed by the asymmetry floors, and their stacked
    output names misclassify as dense-MLP CRITICAL in classify_tensor.
    """
    from .allocate import Tier, classify_tensor

    problems: list[str] = []
    for base, v in qcfg.items():
        if not isinstance(v, dict) or "bits" not in v:
            continue
        if "switch_mlp" in base or ".experts." in base:
            continue
        tier = classify_tensor(base + ".weight", num_experts=num_experts)
        if tier is Tier.CRITICAL and int(v["bits"]) < 6:
            problems.append(f"{base}: CRITICAL-tier module at {v['bits']}-bit")
    return problems


def audit(bundle_dir: Path, source_dir: Path) -> int:
    swm = json.loads((source_dir / "model.safetensors.index.json").read_text())["weight_map"]
    owm = json.loads((bundle_dir / "model.safetensors.index.json").read_text())["weight_map"]
    src_config = json.loads((source_dir / "config.json").read_text())
    qcfg = json.loads((bundle_dir / "config.json").read_text()).get("quantization", {})

    tc0 = src_config.get("text_config", src_config)
    _nexp = int(tc0.get("n_routed_experts", tc0.get("num_local_experts",
                tc0.get("num_experts", 0))) or 0)
    tests = _build_test_set(src_config, swm)
    _from_hardcoded = len(tests)
    if _from_hardcoded < 6:
        # The hardcoded, transformers-named branches matched (almost) nothing —
        # this family uses its own naming scheme. Fall back to a structural scan
        # so the audit can never run on an empty sample and report success.
        tests = tests + _fallback_test_set(swm, owm, _nexp)
        print(f"  note: hardcoded test set yielded {_from_hardcoded}; "
              f"added {len(tests) - _from_hardcoded} via structural fallback")

    print(f"{'label':<40} {'bits':>5}  {'rel_err':>8}  {'tol':>6}  {'result'}")
    print("-" * 76)
    fails: list[tuple[str, int, float, float]] = []
    skipped = 0

    for label, src_key, out_key in tests:
        if src_key not in swm:
            skipped += 1
            continue
        if out_key not in owm:
            skipped += 1
            continue
        src = _load_from(source_dir, swm, src_key).astype(mx.float32)

        base = out_key[:-len(".weight")] if out_key.endswith(".weight") else out_key
        scales_key = base + ".scales"
        if scales_key in owm:
            W = _load_from(bundle_dir, owm, out_key)
            Sc = _load_from(bundle_dir, owm, scales_key)
            Bi = _load_from(bundle_dir, owm, base + ".biases") if base + ".biases" in owm else None
            cfg = qcfg.get(base) or qcfg.get(out_key.replace(".weight", "")) or {}
            bits = int(cfg.get("bits", qcfg.get("bits", 0)) or 0)
            gs = int(cfg.get("group_size", qcfg.get("group_size", 128)) or 128)
            mode = cfg.get("mode", qcfg.get("mode", "affine"))
            if not bits:
                skipped += 1
                continue
            deq = mx.dequantize(W, Sc, Bi, group_size=gs, bits=bits, mode=mode).astype(mx.float32)
        else:
            deq = _load_from(bundle_dir, owm, out_key).astype(mx.float32)
            bits = 16

        _stacked = ("experts" in out_key or "switch_mlp" in out_key)
        if (_stacked and deq.ndim == 3 and src.ndim == 3
                and deq.shape[0] == src.shape[0] > 8):
            # Stacked routed experts: compare expert 0 only. All 256 would be an
            # 8.6 GB source read plus a 17 GB dequant. Gated on the NAME because
            # a shape-only test also matches grouped-conv (C,1,K) tensors.
            deq_slice, src = deq[0], src[0]
        elif deq.ndim == 3 and "routed" in label:
            deq_slice = deq[0]
        else:
            deq_slice = deq

        # Grouped-Conv1d weights are intentionally relaid out for MLX:
        # torch (C, 1, K) -> MLX (C, K, 1). Undo before comparing, or every
        # sconv/conv1d tensor reports a bogus SHAPE MISMATCH.
        # Gate on the NAME, not the shape: `shared_w13_weight` is [2, 4096, 4096],
        # square in the last two dims, so a shape-only test matches it and
        # transposes a tensor that was never relaid out -> bogus 142% error.
        if (("sconv" in out_key or "conv1d" in out_key)
                and deq_slice.ndim == 3 and src.ndim == 3
                and deq_slice.shape == (src.shape[0], src.shape[2], src.shape[1])):
            deq_slice = mx.transpose(deq_slice, (0, 2, 1))

        if deq_slice.shape != src.shape:
            print(f"  {label:<38} SHAPE MISMATCH  {deq_slice.shape} vs {src.shape}")
            fails.append((label, bits, float("inf"), 0.0))
            continue

        err = _rel_err(deq_slice, src)
        tol = TOL.get(bits, 0.5)
        ok = err <= tol
        if not ok:
            fails.append((label, bits, err, tol))
        print(f"  {label:<38} {bits:>5}  {err:>8.4f}  {tol:>6.3f}  {'✓' if ok else '✗ FAIL'}")

    print()
    print(f"tested: {len(tests) - skipped} (skipped {skipped})")
    if len(tests) - skipped == 0:
        print("❌ ZERO tensors tested — the audit proves NOTHING. Do not ship.")
        return 1
    gaps = _coverage_gaps(swm, {sk for _, sk, _ in tests}, owm)
    if gaps:
        print(f"coverage gaps ({len(gaps)} pattern(s) in the bundle with no dequant test):")
        for g in gaps[:12]:
            print(f"    - {g}")
        if len(gaps) > 12:
            print(f"    ... and {len(gaps) - 12} more")

    tc = src_config.get("text_config", src_config)
    n_experts = int(tc.get("n_routed_experts", tc.get("num_local_experts", tc.get("num_experts", 0))) or 0)
    tier_problems = _census_tier_check(qcfg, n_experts)
    if tier_problems:
        print(f"❌ tier census: {len(tier_problems)} CRITICAL module(s) under-allocated")
        for p in tier_problems:
            print(f"   {p}")
    else:
        print(f"tier census: all {sum(1 for v in qcfg.values() if isinstance(v, dict))} quantized modules consistent ✓")

    if not fails and not tier_problems:
        print("✅ ALL tested tensors within quant-noise tolerance")
        print("   Bundle is numerically correct — ship gate PASS")
        print("   (Runtime coherence probes are still REQUIRED before ship.)")
        return 0
    if fails:
        print(f"❌ {len(fails)} tensor(s) exceeded tolerance — ship gate FAIL")
        for label, bits, err, tol in fails:
            print(f"   {label}: {bits}-bit  err={err:.4f}  tol={tol:.4f}")
    return 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bundle")
    ap.add_argument("source")
    args = ap.parse_args()
    sys.exit(audit(Path(args.bundle).expanduser(), Path(args.source).expanduser()))


if __name__ == "__main__":
    main()
