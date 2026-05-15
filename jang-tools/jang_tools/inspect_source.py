"""Fast, no-model-load source inspector. Used by JANG Studio's Step 1."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .jangtq_matrix import supported_model_types

_JANGTQ_V1_WHITELIST = set(supported_model_types())


def _sniff_dtype(model_path: Path) -> str:
    """Peek at the first safetensors shard header to determine source dtype."""
    shards = sorted(model_path.glob("*.safetensors"))
    if not shards:
        return "unknown"
    try:
        import struct
        with open(shards[0], "rb") as fh:
            hdr_len = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(hdr_len))
        dtypes = {v.get("dtype") for k, v in hdr.items() if isinstance(v, dict) and "dtype" in v}
        for preferred in ("BF16", "F16", "F8_E4M3", "F8_E5M2"):
            if preferred in dtypes:
                return {
                    "BF16": "bfloat16", "F16": "float16",
                    "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2",
                }[preferred]
        return "unknown"
    except Exception:
        return "unknown"


def _is_moe(cfg: dict) -> bool:
    for key in ("num_experts", "n_routed_experts", "num_local_experts"):
        if cfg.get(key, 0) and int(cfg[key]) > 1:
            return True
    return False


def _total_bytes(model_path: Path) -> int:
    return sum(f.stat().st_size for f in model_path.glob("*.safetensors"))


def _find_broken_shard(model_path: Path) -> Path | None:
    """M167 (iter 90): return the first shard that is a dangling symlink,
    or None if all shards resolve. huggingface_hub caches symlink
    `snapshots/<hash>/*.safetensors` to `blobs/<sha>`; a `git gc`-style
    prune on the cache (or a disk-failure mid-download) can leave the
    snapshot symlink pointing at a missing blob. Without this guard,
    `_total_bytes` raises a bare FileNotFoundError on the first
    `f.stat()` and the user sees a cryptic Python traceback instead of
    an actionable "shard broken, re-download" message."""
    for f in model_path.glob("*.safetensors"):
        if not f.exists():   # .exists() returns False for dangling symlinks
            return f
    return None


def cmd_inspect_source(args) -> None:
    src = Path(args.model)
    cfg_path = src / "config.json"
    if not cfg_path.exists():
        print(f"ERROR: config.json not found under {src}", file=sys.stderr)
        sys.exit(2)
    # M120: surface config.json parse errors as plain-English diagnostics, not
    # bare JSONDecodeError tracebacks. JANG Studio invokes this from
    # SourceStep.inspect() and captures exit-code only — a traceback on stderr
    # used to be silently dropped and the wizard would show "inspect-source
    # exited 1" with no clue that the user's config.json is malformed.
    try:
        raw = cfg_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        print(f"ERROR: could not read config.json at {cfg_path}: {exc}", file=sys.stderr)
        sys.exit(2)
    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(
            f"ERROR: config.json at {cfg_path} is not valid JSON "
            f"(line {exc.lineno}, col {exc.colno}): {exc.msg}",
            file=sys.stderr,
        )
        sys.exit(2)
    if not isinstance(cfg, dict):
        print(
            f"ERROR: config.json at {cfg_path} has a top-level "
            f"{type(cfg).__name__}, expected a JSON object",
            file=sys.stderr,
        )
        sys.exit(2)
    # M167 (iter 90): pre-validate shards so a broken symlink (HF cache
    # with a git-gc'd blob) surfaces as an actionable diagnostic instead
    # of the bare FileNotFoundError traceback that _total_bytes would
    # otherwise produce on its first stat() call.
    broken = _find_broken_shard(src)
    if broken is not None:
        print(
            f"ERROR: shard {broken.name} is a broken symlink "
            f"(points to {broken.resolve(strict=False)}, target does not exist). "
            f"Re-download the model via `huggingface-cli download {src.name}` "
            f"or clear the HF cache and re-snapshot. Source: {src}",
            file=sys.stderr,
        )
        sys.exit(2)
    model_type = cfg.get("model_type") or cfg.get("text_config", {}).get("model_type", "unknown")
    summary = {
        "model_type": model_type,
        "is_moe": _is_moe(cfg),
        "num_experts": int(cfg.get("num_experts") or cfg.get("n_routed_experts") or 0),
        "num_hidden_layers": int(cfg.get("num_hidden_layers", 0)),
        "hidden_size": int(cfg.get("hidden_size", 0)),
        "dtype": _sniff_dtype(src),
        "total_bytes": _total_bytes(src),
        "shard_count": len(list(src.glob("*.safetensors"))),
        "jangtq_compatible": model_type in _JANGTQ_V1_WHITELIST,
        "is_vl": bool((src / "preprocessor_config.json").exists()),
        "is_video_vl": bool((src / "video_preprocessor_config.json").exists()),
        "has_generation_config": bool((src / "generation_config.json").exists()),
    }
    if args.json:
        print(json.dumps(summary, indent=None, separators=(",", ":")))
    else:
        for k, v in summary.items():
            print(f"  {k}: {v}")


def register(subparsers) -> None:
    p = subparsers.add_parser("inspect-source", help="Fast source-model inspector")
    p.add_argument("model", help="Path to HuggingFace model directory")
    p.add_argument("--json", action="store_true", help="Emit single JSON line on stdout")
    p.set_defaults(func=cmd_inspect_source)
