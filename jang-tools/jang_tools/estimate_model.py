"""Predict converted size for a source model + profile combo.

Swift's PreflightRunner should call this instead of hardcoding bits-per-weight
estimates — keeps predictions accurate as new profiles ship.

JSON output:
  {
    "source_bytes": 12345678,
    "source_gb": 12.3,
    "predicted_output_bytes": 3456789,
    "predicted_output_gb": 3.5,
    "predicted_avg_bits": 4.23,
    "profile": "JANG_4K",
    "method": "bits_per_weight_linear"
  }
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Any

from .allocate import JANG_PROFILES, JANG_K_TARGETS
from .jangtq_matrix import profile_catalog


def _source_bytes(model_dir: Path) -> int:
    """Sum of all .safetensors files. Approximates unquantized source size."""
    return sum(p.stat().st_size for p in model_dir.glob("*.safetensors"))


def _source_bytes_per_weight(model_dir: Path) -> int:
    """M173 (iter 99): peek at the first safetensors shard header to
    determine bytes-per-weight for the source model. BF16/FP16 → 2,
    FP8 (e4m3/e5m2) → 1, FP32 → 4. Falls back to 2 (BF16 assumption)
    if detection fails (unknown dtype, missing shards, read error) —
    conservative over-estimate is safer than under-estimate for the
    preflight disk-space gate. Mirrors inspect_source._sniff_dtype's
    peek-pattern but returns bytes, not the dtype name."""
    import struct as _struct
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        return 2
    try:
        with open(shards[0], "rb") as fh:
            hdr_len = _struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(hdr_len))
        dtypes = {v.get("dtype") for k, v in hdr.items() if isinstance(v, dict) and "dtype" in v}
        # Scan in priority order — if mixed, go with the dominant dtype.
        if any(d in dtypes for d in ("F8_E4M3", "F8_E5M2")):
            return 1
        if any(d in dtypes for d in ("BF16", "F16")):
            return 2
        if "F32" in dtypes:
            return 4
        return 2   # fallback on unknown
    except Exception:
        return 2


def _predict_avg_bits(profile: str) -> float:
    """Best-effort avg-bits/weight for a profile. Used as the multiplier vs source bf16."""
    if profile in JANG_K_TARGETS:
        return JANG_K_TARGETS[profile]
    if profile in JANG_PROFILES:
        crit, imp, comp = JANG_PROFILES[profile]
        # Approximation: compress is majority of weights; critical + important are minorities
        # Typical distribution: 80% compress, 15% important, 5% critical
        return round(0.80 * comp + 0.15 * imp + 0.05 * crit, 2)
    # JANGTQ
    for p in profile_catalog():
        if p["name"] == profile:
            return float(p.get("avg_bits") or p["bits"])
    raise ValueError(f"unknown profile: {profile}")


def predict(model_dir: Path, profile: str) -> dict[str, Any]:
    src_bytes = _source_bytes(model_dir)
    if src_bytes == 0:
        # Best-effort fallback: read config and estimate from num params.
        # M133 (iter 55): mirror `recommend._estimate_params_billion`'s
        # MoE-aware formula. Pre-iter-55 this used a flat
        # `12 * h² * layers + 2 * h * vocab` that assumed dense + ignored
        # num_experts — a 256-expert Qwen3.5-MoE fell through this path
        # with ~12 GB predicted source when real bf16 source is ~700 GB
        # (off by ~55x). Wizard then told users "predicted output: 3 GB"
        # for a conversion that actually writes 180+ GB, followed by a
        # disk-full failure mid-convert.
        import json as _json
        cfg_path = model_dir / "config.json"
        if cfg_path.exists():
            cfg = _json.loads(cfg_path.read_text())
            text_cfg = cfg.get("text_config", {}) or {}
            hidden = int(cfg.get("hidden_size", 0) or text_cfg.get("hidden_size", 0) or 0)
            layers = int(cfg.get("num_hidden_layers", 0) or text_cfg.get("num_hidden_layers", 0) or 0)
            vocab = int(cfg.get("vocab_size", 0) or text_cfg.get("vocab_size", 0) or 0)
            intermediate = int(
                cfg.get("intermediate_size", 0)
                or text_cfg.get("intermediate_size", 0)
                or 4 * hidden
            )
            num_experts = int(
                cfg.get("num_experts")
                or cfg.get("n_routed_experts")
                or cfg.get("num_local_experts")
                or 0
            )
            if hidden and layers:
                # Attention weights per layer: 4 × h² (q/k/v/o projections).
                attn = 4 * hidden * hidden
                # MLP per expert: 3 × h × intermediate (gate + up + down).
                mlp_per = 3 * hidden * intermediate
                mlp = mlp_per * num_experts if num_experts > 1 else mlp_per
                per_layer = attn + mlp
                approx_params = per_layer * layers + 2 * hidden * vocab
                src_bytes = approx_params * 2   # assume bf16 source
    avg_bits = _predict_avg_bits(profile)
    # M173 (iter 99): source dtype matters for the divisor.
    # Pre-M173 the formula hardcoded /16 (BF16 assumption) — correct for
    # the common case but WRONG for FP8 sources (DeepSeek V3/V3.2 etc.)
    # where src_bytes = weights × 1. Under-predicted output by 2× → user
    # started convert thinking "plenty of disk", hit disk-full mid-way.
    # Detect dtype by peeking at the first shard header (same mechanism
    # inspect_source uses). 2 bytes/weight for BF16/FP16, 1 for FP8,
    # fallback to 2 (safer over-estimate than under-estimate).
    bytes_per_weight = _source_bytes_per_weight(model_dir)
    # output_bytes = weights × (avg_bits / 8) × 1.05
    # weights = src_bytes / bytes_per_weight
    # → output_bytes = src_bytes × avg_bits / (8 × bytes_per_weight) × 1.05
    predicted = int(src_bytes * avg_bits / (8.0 * bytes_per_weight) * 1.05)
    return {
        "source_bytes": src_bytes,
        "source_gb": round(src_bytes / 1_000_000_000, 3),
        "predicted_output_bytes": predicted,
        "predicted_output_gb": round(predicted / 1_000_000_000, 3),
        "predicted_avg_bits": avg_bits,
        "profile": profile,
        "method": "bits_per_weight_linear",
    }


def cmd_estimate_model(args) -> None:
    model_dir = Path(args.model)
    if not model_dir.exists():
        print(f"ERROR: model dir not found: {model_dir}", file=sys.stderr)
        sys.exit(2)
    try:
        result = predict(model_dir, args.profile)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(3)
    if args.json:
        print(json.dumps(result, indent=None))
    else:
        print(f"Source: {result['source_gb']} GB")
        print(f"Profile: {result['profile']} (avg {result['predicted_avg_bits']} bits/weight)")
        print(f"Predicted output: {result['predicted_output_gb']} GB")


def register(subparsers) -> None:
    p = subparsers.add_parser("estimate-model", help="Predict converted size for a source model + profile")
    p.add_argument("--model", required=True, help="Path to source HuggingFace model dir")
    p.add_argument("--profile", required=True, help="Target JANG or JANGTQ profile name")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_estimate_model)
