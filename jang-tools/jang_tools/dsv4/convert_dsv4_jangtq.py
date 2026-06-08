"""DeepSeek-V4-Flash FP4+FP8 source → JANGTQ bundle.

No REAP prune — converts all 256 experts as-is.

Profiles (--profile sets ROUTED-EXPERT bits):
  JANGTQ2 (default): V3 runtime-candidate lane. Routed experts use 2-bit
                     MXTQ with V3 mixed-bit overrides, attention + embed +
                     lm_head + shared experts stay 8-bit affine, norms +
                     router + mHC params are passthrough, and MTP is dropped.
  JANGTQ4: routed experts 4-bit MXTQ (bigger bundle, better fidelity)

Variants (--variant tweaks NON-routed bits + drops MTP):
  std               : legacy baseline/repro mode (8-bit affine on
                      attention/shared/embed/head; MTP layers shipped). This
                      is not the current DSV4 runtime candidate.
  K (JANG_DSV4_K)   : MAX-QUALITY 70-80 GB profile. Drops MTP head
                      (~6.5B params unused at decode-time sanitize) and
                      keeps EVERY non-routed module at 8-bit affine
                      gsz=32 — attention (wq_a/wq_b/wkv/wo_a/wo_b),
                      shared experts, Compressor, Indexer, embed, head.
                      Router (gate.weight), mHC fn matrices, all
                      RMSNorms, attn_sink, ape stay fp16 passthrough.
                      Source FP4 routed → 2-bit MXTQ; source FP8
                      non-routed → 8-bit affine (≤0.5% RMS, lossless
                      vs FP8). Use --profile 2 for 70-80 GB bundle.
  V3 (default)      : runtime-candidate lane. Drops MTP, uses 2/4-bit MXTQ
                      routed experts, and accepts DSV4_V3_PLAN_PATH for the
                      exact proven/rebuilt bit plan. A production candidate
                      needs DSV4_V3_PLAN_PATH; no-plan V3 is a fallback.

Usage:
  python -m jang_tools.dsv4.convert_dsv4_jangtq \\
      --src <path/to/DeepSeek-V4-Flash> \\
      --dst ~/.mlxstudio/models/JANGQ-AI/DSV4-Flash-JANGTQ_V3 \\
      --profile 2 --variant V3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file as sf_save_np

from jang_tools.dsv4.weight_loader import ShardIndex


SEED = 42
FORMAT = "jangtq"  # set by main() from --format flag: "jang" or "jangtq"
VARIANT = "V3"     # set by main() from --variant flag: "std", "K", or "V3"
_V3_PLAN_CACHE_PATH: str | None = None
_V3_PLAN_CACHE: dict | None = None
_V3_PLAN_CONFIG_CACHE: dict | None = None

CRITICAL_F32_RE = re.compile(
    r"^(hc_head_(?:fn|base|scale)|"
    r"layers\.\d+\.hc_(?:attn|ffn)_(?:fn|base|scale)|"
    r"layers\.\d+\.attn\.attn_sink|"
    r"layers\.\d+\.ffn\.gate\.bias)$"
)


def read_passthrough(idx: ShardIndex, name: str) -> np.ndarray:
    """Read tensors that should not be quantized.

    DSV4 control tensors are small but numerically load-bearing. The old
    converter read all passthrough tensors as fp16, which destroyed the F32
    mHC/Sinkhorn/sink/router controls and produced the local broken bundle.
    Preserve true source F32 when present, and store critical controls as F32
    even if a bad source copy already rounded them.
    """
    if idx.dtype_of(name) == torch.float32 or CRITICAL_F32_RE.match(name):
        return idx.read_tensor(name, out_dtype=torch.float32).numpy().astype(np.float32)
    src_dtype = idx.dtype_of(name)
    if src_dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
        return idx.read_tensor(name).numpy()
    out_dtype = torch.float16
    tensor = idx.read_tensor(name, out_dtype=out_dtype)
    if tensor.dtype == torch.bfloat16:
        return tensor.float().numpy().astype(np.float16)
    return tensor.numpy()


def finalize_jangtq_bundle(dst: Path, *, prestack: bool, build_sidecar: bool) -> None:
    """Finalize a DSV4 JANGTQ artifact for shipping.

    The converter naturally emits per-expert TQ triplets. Shipping bundles
    should be pre-stacked into switch_mlp layout, and Swift-compatible runtimes
    need a deterministic Hadamard signs/codebook sidecar. Fail loudly on either
    finalization error; a half-finalized bundle is worse than no bundle.
    """
    if prestack:
        from jang_tools.rebundle_jangtq_stacked import rebundle

        tmp = dst.parent / f"{dst.name}.prestack_tmp"
        old = dst.parent / f"{dst.name}.perexpert_tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        if old.exists():
            shutil.rmtree(old)

        rebundle(dst, tmp)
        if not (tmp / "model.safetensors.index.json").is_file():
            raise RuntimeError(f"prestack failed: missing index in {tmp}")
        dst.rename(old)
        tmp.rename(dst)
        shutil.rmtree(old)
        print("[convert] prestack complete; routed experts are switch_mlp layout", flush=True)

    if build_sidecar:
        from jang_tools.build_jangtq_sidecar import main as build_jangtq_sidecar

        old_argv = sys.argv[:]
        try:
            sys.argv = ["build_jangtq_sidecar", str(dst)]
            build_jangtq_sidecar()
        finally:
            sys.argv = old_argv
        if not (dst / "jangtq_runtime.safetensors").is_file():
            raise RuntimeError("sidecar build completed but jangtq_runtime.safetensors is missing")
        print("[convert] jangtq_runtime.safetensors sidecar present", flush=True)


def build_routed_expert_bit_plan(profile_bits: int) -> dict:
    """Return explicit routed-expert bit metadata for config files.

    `mxtq_bits.routed_expert` is a role default. V3 is intentionally mixed:
    the hash-routed layers use 4-bit MXTQ while the smooth-routed layers use
    the requested profile width. Store that plan explicitly so loaders/audits
    never infer uniform routed bits from the scalar default.
    """
    plan = {
        "default_bits": profile_bits,
        "codec": "mxtq" if FORMAT == "jangtq" else "affine",
    }
    if FORMAT == "jangtq" and VARIANT == "V3":
        plan.update(
            {
                "hash_layer_indices": [0, 1, 2],
                "hash_layer_bits": 4,
                "smooth_layer_bits": profile_bits,
            }
        )
        explicit_meta = _v3_plan_metadata()
        plan.update(explicit_meta)
        routed_layer_bits = explicit_meta.get("routed_layer_bits") or {}
        explicit_hash_bits = {
            str(idx): int(routed_layer_bits[str(idx)])
            for idx in (0, 1, 2)
            if str(idx) in routed_layer_bits
        }
        if any(bits != 4 for bits in explicit_hash_bits.values()):
            plan.pop("hash_layer_bits", None)
            plan["hash_layer_default_bits"] = 4
            plan["hash_layer_bits_source"] = "DSV4_V3_PLAN_PATH"
    return plan


def _v3_plan_metadata() -> dict:
    plan, _cfg = _load_v3_plan()
    path = os.environ.get("DSV4_V3_PLAN_PATH", "").strip()
    if plan is None or not path:
        return {}

    routed_layer_bits: dict[str, int] = {}
    bit_counts: dict[str, int] = {}
    for logical, bits_raw in plan.items():
        bits = int(bits_raw)
        bit_counts[str(bits)] = bit_counts.get(str(bits), 0) + 1
        m = re.match(
            r"^model\.layers\.(\d+)\.mlp\.switch_mlp\.SWITCH_MLP_LAYER$",
            logical,
        )
        if m:
            routed_layer_bits[m.group(1)] = bits

    plan_path = Path(path)
    return {
        "plan_source": "DSV4_V3_PLAN_PATH",
        "plan_file": plan_path.name,
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "routed_layer_bits": dict(sorted(
            routed_layer_bits.items(), key=lambda item: int(item[0])
        )),
        "plan_bit_counts": dict(sorted(
            bit_counts.items(), key=lambda item: int(item[0])
        )),
    }


def _reset_v3_plan_cache_for_tests() -> None:
    global _V3_PLAN_CACHE_PATH, _V3_PLAN_CACHE, _V3_PLAN_CONFIG_CACHE
    _V3_PLAN_CACHE_PATH = None
    _V3_PLAN_CACHE = None
    _V3_PLAN_CONFIG_CACHE = None


def _load_v3_plan() -> tuple[dict, dict] | tuple[None, None]:
    """Load the optional JANG v3 bit plan.

    Internal docs have long described `DSV4_V3_PLAN_PATH`, but the canonical
    converter used to ignore it and only hardcoded the first three hash-routed
    layers at 4-bit. Keep the load path local and cheap: one JSON read, cached
    by path, and no imports from the heavier calibration/consolidation tools.
    """
    global _V3_PLAN_CACHE_PATH, _V3_PLAN_CACHE, _V3_PLAN_CONFIG_CACHE
    path = os.environ.get("DSV4_V3_PLAN_PATH", "").strip()
    if not path:
        return None, None
    if _V3_PLAN_CACHE is not None and _V3_PLAN_CACHE_PATH == path:
        return _V3_PLAN_CACHE, _V3_PLAN_CONFIG_CACHE or {}

    data = json.loads(Path(path).read_text())
    if "plan" in data and isinstance(data["plan"], dict):
        plan = data["plan"]
        cfg = data.get("config", {})
    elif isinstance(data, dict):
        plan = data
        cfg = {}
    else:
        raise ValueError(f"DSV4_V3_PLAN_PATH={path} is not a JSON object")

    _V3_PLAN_CACHE_PATH = path
    _V3_PLAN_CACHE = plan
    _V3_PLAN_CONFIG_CACHE = cfg if isinstance(cfg, dict) else {}
    return _V3_PLAN_CACHE, _V3_PLAN_CONFIG_CACHE


def _v3_logical_from_source(n: str) -> str | None:
    """Map DSV4 source tensor names to JANG v3 logical bit-plan names."""
    if n.endswith((".scales", ".biases", ".scale")):
        return None
    if "_norm." in n or n.endswith("norm.weight"):
        return None
    if n.endswith((".attn_sink", ".bias", ".ape", ".tid2eid")):
        return None
    if not n.endswith(".weight"):
        return None
    if "hc_" in n and n.endswith(("_base", "_fn", "_scale")):
        return None

    m = re.match(r"^mtp\.(\d+)\.(.+)$", n)
    if m:
        midx, rest = m.groups()
        prefix = f"mtp.{midx}"
    else:
        m = re.match(r"^layers\.(\d+)\.(.+)$", n)
        if m:
            lidx, rest = m.groups()
            prefix = f"model.layers.{lidx}"
        else:
            if n == "embed.weight":
                return "model.embed_tokens"
            if n == "head.weight":
                return "lm_head"
            return None

    m = re.match(r"^ffn\.experts\.\d+\.(w[123])\.weight$", rest)
    if m:
        proj = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}[m.group(1)]
        return f"{prefix}.mlp.switch_mlp.{proj}"
    m = re.match(r"^ffn\.shared_experts\.(w[123])\.weight$", rest)
    if m:
        proj = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}[m.group(1)]
        return f"{prefix}.mlp.shared_experts.{proj}"
    if rest == "ffn.gate.weight":
        return f"{prefix}.mlp.gate"
    m = re.match(r"^attn\.(wq_a|wq_b|wkv|wo_a|wo_b)\.weight$", rest)
    if m:
        sub = m.group(1)
        if sub in ("wq_a", "wkv"):
            return f"{prefix}.self_attn._wq_a_kv_fused"
        return f"{prefix}.self_attn.{sub}"
    m = re.match(r"^attn\.indexer\.(wq_b|weights_proj)\.weight$", rest)
    if m:
        return f"{prefix}.self_attn.indexer.{m.group(1)}"
    m = re.match(r"^attn\.indexer\.compressor\.(wkv|wgate)\.weight$", rest)
    if m:
        return f"{prefix}.self_attn.indexer.compressor.{m.group(1)}"
    m = re.match(r"^attn\.compressor\.(wkv|wgate)\.weight$", rest)
    if m:
        return f"{prefix}.self_attn.compressor.{m.group(1)}"
    if rest in ("e_proj.weight", "h_proj.weight"):
        return f"{prefix}.{rest.replace('.weight', '')}"
    return None


def _lookup_v3_plan(name: str) -> tuple[int, str, int] | None:
    """Return a converter classification from DSV4_V3_PLAN_PATH, if present."""
    if FORMAT != "jangtq" or VARIANT != "V3":
        return None

    plan, cfg = _load_v3_plan()
    if plan is None:
        return None

    logical = _v3_logical_from_source(name)
    if logical is None:
        return None
    if ".mlp.switch_mlp." in logical and logical.endswith(
        (".gate_proj", ".down_proj", ".up_proj")
    ):
        coalesced = logical.rsplit(".", 1)[0] + ".SWITCH_MLP_LAYER"
        if coalesced in plan:
            logical = coalesced

    bits_raw = plan.get(logical)
    if bits_raw is None:
        return None
    bits = int(bits_raw)
    default_gs = int((cfg or {}).get("default_gs", 32))

    if bits == 16:
        return 16, "passthrough", 0
    if ".mlp.switch_mlp." in logical:
        # The shipped DSV4 JANGTQ prestack path is TQ switch_mlp-first. An
        # 8-bit affine routed layer would bypass rebundle_jangtq_stacked and
        # rely on mixed affine-routed/TQ-routed MoE hydration that has not
        # passed live DSV4 gates. Fail here instead of producing another
        # bundle that "loads" but is semantically unproven.
        if bits not in (2, 4):
            raise ValueError(
                f"DSV4 V3 affine routed plan is not currently supported for "
                f"{logical}: requested {bits}-bit. Use routed 2/4-bit MXTQ "
                "or implement/prove affine-routed prestack/runtime support."
            )
        return bits, "mxtq", 0
    if bits not in (2, 3, 4, 5, 6, 8):
        raise ValueError(f"unsupported DSV4 V3 plan bit width for {logical}: {bits}")
    return bits, "affine", default_gs


def classify(name: str, profile_bits: int) -> tuple[int, str, int]:
    """Map tensor name → (bits, method, group_size).

    Uses uniform **group_size=32** across all quantized tensors — matches
    FP4's native block size, minimal extra scale overhead for FP8
    source tensors (~3% vs group=128). Keeps mlx_lm's single-config
    path simple (one group_size value).

    Routed experts (FP4 origin):
      * profile 8 → 8-bit affine g=32 (max fidelity; ~0.5% RMS)
      * profile 4 → 4-bit affine g=32 (matches FP4 source; ~9% RMS)
      * profile 2 → 2-bit MXTQ codebook (aggressive; lossy)

    Non-routed (variant="std" or "K", FP8 origin): ALL 8-bit affine g=32.
    Variant K's only delta vs std is dropping the MTP head (handled in
    convert(), not here). Non-routed bit-width stays 8 because attention
    + shared experts run on EVERY token — FP8 source → 8-bit affine is
    bit-faithful (~0.5% RMS), 4-bit would be a quality regression.

    Norms / small / mHC fn matrices / router gate / hash table /
    attn_sink / ape: fp16 passthrough (or int passthrough for tid2eid).
    """
    n = name

    # Tier 1: norms + small tensors + mHC fn + tid2eid + attn_sink + ape
    # → fp16 passthrough (int tensors stay int via weight_loader)
    if ("norm" in n or n.endswith(".bias") or "attn_sink" in n
            or ".ape" in n or "tid2eid" in n or n.startswith("hc_")
            or re.search(r"^layers\.\d+\.hc_", n)
            or re.search(r"^mtp\.\d+\.hc_", n)
            or re.search(r"^layers\.\d+\.ffn\.gate\.(weight|bias)$", n)
            or re.search(r"^mtp\.\d+\.ffn\.gate\.(weight|bias)$", n)
            ):
        return 16, "passthrough", 0

    # Router gate.weight (non-MoE-expert) → fp16 passthrough
    if n.endswith(".gate.weight") and "experts" not in n:
        return 16, "passthrough", 0

    plan_hit = _lookup_v3_plan(n)
    if plan_hit is not None:
        return plan_hit

    # Routed experts
    if re.search(r"ffn\.experts\.\d+\.(w1|w2|w3)\.weight$", n):
        # JANG format: always use standard affine (all bit-widths).
        # JANGTQ format: MXTQ codebook for 2-bit, affine for 3+.
        if FORMAT == "jang":
            return profile_bits, "affine", 32
        # V3 variant: hash-routed layers 0-2 get 4-bit MXTQ floor (no soft
        # routing → quantization noise can't average out), rest at profile_bits.
        if VARIANT == "V3":
            m = re.match(r"^layers\.(\d+)\.", n)
            if m and int(m.group(1)) < 3:
                return 4, "mxtq", 0
            return profile_bits, "mxtq", 0
        if profile_bits in (3, 4, 5, 6, 8):
            return profile_bits, "affine", 32
        return profile_bits, "mxtq", 0

    # Non-routed quantized tensors. Env DSV4_HIGH_PRECISION=1 keeps these at
    # bf16 (no quant) to eliminate compound quant error for arithmetic-
    # sensitive reasoning. Trade-off: bundle +5-8 GB, but removes all quant
    # noise from non-routed path.
    import os as _os
    hp = _os.environ.get("DSV4_HIGH_PRECISION", "0") == "1"

    # Variant K's only delta vs std is dropping MTP (see convert()).
    # Non-routed stay at 8-bit because attention/shared run every token
    # and FP8 source → 8-bit affine is bit-faithful (~0.5% RMS).
    nonrouted_bits = 8

    if "shared_experts" in n and n.endswith(".weight"):
        return (16, "passthrough", 0) if hp else (nonrouted_bits, "affine", 32)
    if re.search(r"attn\.(wq_a|wq_b|wkv|wo_a|wo_b)\.weight$", n):
        return (16, "passthrough", 0) if hp else (nonrouted_bits, "affine", 32)
    # Compressor / Indexer (long-context modules, fire only on prompts >= compress_ratio)
    if re.search(r"\.(compressor|indexer)\.", n) and n.endswith(".weight"):
        return (16, "passthrough", 0) if hp else (nonrouted_bits, "affine", 32)
    # Embed + head — Tier 1 every-token path, keep 8-bit even in K profile
    if n == "embed.weight" or n == "head.weight":
        return (16, "passthrough", 0) if hp else (8, "affine", 32)

    if n.endswith(".weight"):
        return 8, "affine", 32

    return 16, "passthrough", 0


def convert(
    src: Path,
    dst: Path,
    profile_bits: int,
    *,
    prestack: bool = True,
    build_sidecar: bool = True,
) -> None:
    import mlx.core as mx
    from jang_tools.turboquant.linear import tq_quantize_weight

    dst.mkdir(parents=True, exist_ok=True)
    idx = ShardIndex(src)
    print(f"[convert] source: {src}")
    print(f"[convert] target: {dst}")
    if FORMAT == "jang":
        profile_name = f"JANG_{profile_bits}L"
    elif VARIANT == "K":
        profile_name = "JANGTQ_K" if profile_bits == 2 else f"JANGTQ{profile_bits}_K"
    elif VARIANT == "V3":
        profile_name = "JANGTQ_V3" if profile_bits == 2 else f"JANGTQ{profile_bits}_V3"
    else:
        profile_name = f"JANGTQ{profile_bits}"
    print(f"[convert] profile: {profile_name} (format={FORMAT}, variant={VARIANT})")
    drop_mtp = (VARIANT in ("K", "V3"))
    if drop_mtp:
        print(f"[convert] MTP head: DROP (variant={VARIANT}, ~6.5B params unused at decode)")
    if VARIANT == "V3" and not os.environ.get("DSV4_V3_PLAN_PATH", "").strip():
        print(
            "[convert] WARN: V3 without DSV4_V3_PLAN_PATH uses the built-in "
            "hash-layer fallback. A production candidate needs DSV4_V3_PLAN_PATH.",
            flush=True,
        )
    print(f"[convert] scanning for .weight keys (skip sibling .scale)...")
    weight_keys = [k for k in idx.keys if not k.endswith(".scale")]
    if drop_mtp:
        before = len(weight_keys)
        weight_keys = [k for k in weight_keys if not k.startswith("mtp.")]
        print(f"[convert] dropped {before - len(weight_keys)} mtp.* tensors")
    print(f"[convert] {len(weight_keys)} logical tensors to process")

    MAX_SHARD_BYTES = 1_000_000_000
    shard_idx = 1
    shard_bytes = 0
    shard_buf: dict[str, np.ndarray] = {}
    shard_map: dict[str, str] = {}

    totals = {"mxtq": 0, "affine": 0, "passthrough": 0}
    # Per-module quantization overrides for config.json. Keyed on the storage
    # tensor stem (e.g. "layers.0.attn.wq_a"). We write an entry for every
    # affine tensor whose (bits, mode, group_size) differs from the top-level
    # default (bits=8 mode=affine group_size=32). Mirrors what the
    # patch_dsv4_quant_config tool produces — having this baked in at convert
    # time means we never ship a misconfig'd bundle (root cause of the
    # JANGTQ-CONFIG-METADATA-BUG that capped DSV4 HumanEval at 42%).
    quant_overrides: dict[str, dict] = {}
    t_start = time.time()

    def flush_shard():
        nonlocal shard_idx, shard_bytes, shard_buf
        if not shard_buf:
            return
        shard_name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        sf_save_np(shard_buf, str(dst / shard_name))
        for k in shard_buf:
            shard_map[k] = shard_name
        print(f"    shard {shard_idx}: {len(shard_buf)} tensors, "
              f"{shard_bytes / 1e9:.2f} GB  "
              f"(elapsed {time.time() - t_start:.0f}s)", flush=True)
        shard_buf = {}
        shard_bytes = 0
        shard_idx += 1

    def add_tensor(name: str, arr: np.ndarray):
        nonlocal shard_bytes
        shard_buf[name] = arr
        shard_bytes += arr.nbytes
        if shard_bytes >= MAX_SHARD_BYTES:
            flush_shard()

    for i, name in enumerate(weight_keys):
        bits, method, gsz = classify(name, profile_bits)

        if method == "passthrough":
            add_tensor(name, read_passthrough(idx, name))
            totals["passthrough"] += 1

        elif method == "affine":
            t = idx.read_tensor(name, out_dtype=torch.float32)
            w = mx.array(t.numpy())
            # FP4-origin routed experts at 4-bit: use MXFP4 mode which
            # exactly replicates source FP4 16-level log-spaced codebook.
            # All other tensors (FP8-origin attention, shared, embed, head):
            # use standard affine — they're FP8 source so linear 8-bit
            # affine represents them losslessly within 0.5% RMS.
            is_routed_expert = re.search(r"ffn\.experts\.\d+\.(w1|w2|w3)\.weight$", name) is not None
            # Direct-copy only applies if source is still in FP4 format (int8
            # packed + float8_e8m0fnu scale). BF16-dequant sources have no
            # .scale sibling — fall through to mx.quantize on the bf16 tensor.
            raw_w = idx.read_raw(name) if is_routed_expert and bits == 4 else None
            src_is_fp4 = raw_w is not None and raw_w.dtype == torch.int8
            if is_routed_expert and bits == 4 and src_is_fp4:
                # BIT-EXACT preservation: source is already MXFP4 format
                # (int8 packed FP4 + UE8M0 scale). MLX's mxfp4 uint32 layout
                # matches source int8 byte-for-byte (little-endian packing of
                # 4 source bytes per uint32, nibbles LSB→MSB). So we can
                # DIRECT-COPY without going through bf16 intermediate.
                sk = name[:-len(".weight")] + ".scale" if name.endswith(".weight") else name + ".scale"
                raw_s = idx.read_raw(sk)    # float8_e8m0fnu torch tensor
                # int8 (out, in/2) → reinterpret as uint8 bytes → pack into uint32
                w_bytes = raw_w.numpy().view(np.uint8)  # (out, in/2)
                out_dim, packed_in = w_bytes.shape
                in_dim = packed_in * 2
                assert in_dim % 8 == 0, f"in_dim {in_dim} not multiple of 8"
                # View as (out, in/8, 4 bytes) then as little-endian uint32
                w_u32 = w_bytes.reshape(out_dim, in_dim // 8, 4).copy().view(np.uint32).reshape(out_dim, in_dim // 8)
                # float8_e8m0fnu doesn't support .numpy() directly; reinterpret as uint8
                s_bytes = raw_s.view(torch.uint8).numpy()   # (out, in/32)
                assert s_bytes.shape == (out_dim, in_dim // 32), f"scale shape {s_bytes.shape}"
                base = name[:-len(".weight")] if name.endswith(".weight") else name
                add_tensor(f"{base}.weight", np.ascontiguousarray(w_u32))
                add_tensor(f"{base}.scales", np.ascontiguousarray(s_bytes))
                quant_overrides[base] = {"bits": 4, "group_size": 32, "mode": "mxfp4"}
                totals["affine"] += 1
            else:
                qw, qs, qb = mx.quantize(w, group_size=gsz or 64, bits=bits)
                base = name[:-len(".weight")] if name.endswith(".weight") else name
                add_tensor(f"{base}.weight", np.array(qw))
                add_tensor(f"{base}.scales", np.array(qs).astype(np.float16))
                add_tensor(f"{base}.biases", np.array(qb).astype(np.float16))
                # Record override iff diverges from top-level (bits=8 affine gsz=32).
                if bits != 8 or (gsz or 64) != 32:
                    quant_overrides[base] = {
                        "bits": bits, "group_size": gsz or 64, "mode": "affine",
                    }
                totals["affine"] += 1

        elif method == "mxtq":
            t = idx.read_tensor(name, out_dtype=torch.float32)
            arr = t.numpy()
            result = tq_quantize_weight(arr, bits=bits, seed=SEED)
            base = name[:-len(".weight")] if name.endswith(".weight") else name
            add_tensor(f"{base}.tq_packed", np.asarray(result["packed"]))
            add_tensor(f"{base}.tq_norms", np.asarray(result["norms"]))
            add_tensor(f"{base}.tq_bits", np.array([bits], dtype=np.int32))
            totals["mxtq"] += 1
        else:
            raise ValueError(f"unknown method {method} for {name}")

        if (i + 1) % 500 == 0:
            print(f"    processed {i + 1}/{len(weight_keys)}  "
                  f"mxtq={totals['mxtq']} affine={totals['affine']} "
                  f"passthrough={totals['passthrough']}  "
                  f"({time.time() - t_start:.0f}s)", flush=True)

    flush_shard()

    # Rename shards to final count
    for k in range(1, shard_idx):
        old = dst / f"model-{k:05d}-of-XXXXX.safetensors"
        new = dst / f"model-{k:05d}-of-{shard_idx - 1:05d}.safetensors"
        if old.exists():
            old.rename(new)
    final_map = {k: v.replace("XXXXX", f"{shard_idx - 1:05d}") for k, v in shard_map.items()}
    total_bytes = sum((dst / fn).stat().st_size for fn in set(final_map.values()))
    (dst / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": total_bytes},
        "weight_map": final_map,
    }, indent=2))

    # config.json + jang_config.json
    src_cfg = json.loads((src / "config.json").read_text())
    src_cfg.pop("quantization_config", None)
    if drop_mtp:
        # The current JANG DSV4 runtime explicitly drops mtp.* in
        # Model.sanitize() and does not instantiate an MTP decode head. Keep
        # config metadata honest so validators do not expect missing weights.
        src_cfg["num_nextn_predict_layers"] = 0

    # transformers 4.45+ renamed `rope_scaling` -> `rope_parameters` and
    # `type` -> `rope_type` inside the dict. Add the newer spelling for
    # compatibility, but never remove rope_scaling: jang_tools.dsv4.mlx_model
    # consumes it directly for DSV4 Flash compressed-context YaRN.
    if "rope_scaling" in src_cfg and "rope_parameters" not in src_cfg:
        rs = dict(src_cfg["rope_scaling"])
        rp = dict(rs)
        if "type" in rp:
            rp["rope_type"] = rp.pop("type")
        if "rope_theta" not in rp:
            rp["rope_theta"] = float(src_cfg.get("rope_theta", 10000))
        for k in ("beta_fast", "beta_slow", "factor"):
            if k in rp:
                rp[k] = float(rp[k])
        src_cfg["rope_parameters"] = rp
    if FORMAT == "jang":
        # Pure-affine bundle (JANG_2L/4L). One bit-width across all routed
        # experts; non-routed are 8-bit. Per-module overrides cover
        # divergences (and routed_expert_bits is documented at top-level).
        quant_cfg: dict = {"bits": 8, "group_size": 32, "mode": "affine"}
    else:
        # JANGTQ bundle. Top-level stays bits=8 mode=affine (the safe default
        # for non-routed; that's what _fix_quantized_bits expects too).
        # Routed-expert codec is signalled via top-level routed_expert_bits.
        # Per-module overrides for any tensor that diverges from this default
        # (currently mxfp4 routed @ profile=4, plus 4-bit attn/shared/etc on
        # variant=K).
        quant_cfg = {"bits": 8, "group_size": 32, "mode": "affine"}
    quant_cfg.update(quant_overrides)
    # Variant K and std both keep non-routed at 8-bit affine. Variant K
    # only differs by dropping MTP (see drop_mtp above).
    nonrouted_bits_meta = 8
    # Mirror the Swift-required metadata INSIDE the quantization dict so the
    # vMLX strict Codable decoder finds it whether it looks at top-level or
    # quantization sub-object (existing bundles ship both).
    routed_expert_bit_plan = build_routed_expert_bit_plan(profile_bits)
    if FORMAT == "jangtq":
        mxtq_bits_meta = {
            "routed_expert": profile_bits,
            "attention": nonrouted_bits_meta,
            "shared_expert": nonrouted_bits_meta,
            "compressor": nonrouted_bits_meta,
            "indexer": nonrouted_bits_meta,
            "embed_tokens": 8,
            "lm_head": 8,
            "norms_router_hc": 16,
        }
        quant_cfg["routed_expert_bits"] = profile_bits
        quant_cfg["routed_expert_bit_plan"] = routed_expert_bit_plan
        quant_cfg["mxtq_bits"] = mxtq_bits_meta
    src_cfg["quantization"] = quant_cfg
    # Top-level Swift-required keys (DeepseekV4JANGTQ §414 Codable decode +
    # build_jangtq_sidecar.py's seed lookup). Without these, the Swift loader
    # falls back to defaults that may mismatch the bundle's actual codec.
    if FORMAT == "jangtq":
        src_cfg["weight_format"] = "mxtq"
        src_cfg["routed_expert_bits"] = profile_bits
        src_cfg["routed_expert_bit_plan"] = routed_expert_bit_plan
        src_cfg["mxtq_bits"] = mxtq_bits_meta
        src_cfg["mxtq_seed"] = SEED
        src_cfg["group_size"] = 32
    src_cfg["_name_or_path"] = f"DSV4-Flash-{profile_name}"
    (dst / "config.json").write_text(json.dumps(src_cfg, indent=2))

    (dst / "jang_config.json").write_text(json.dumps({
        "weight_format": "mxfp4_mixed" if FORMAT == "jangtq" and profile_bits == 4 else (
            "affine" if FORMAT == "jang" else "mxtq"
        ),
        "profile": profile_name,
        "variant": VARIANT,
        "mxtq_seed": SEED,
        "drop_mtp": drop_mtp,
        "critical_f32_preserved": True,
        "dsv4_runtime_requirements": {
            "limited_swiglu_tq_patch": FORMAT == "jangtq",
            "generic_mlx_sinks": False,
            "native_cache_schema": "deepseek_v4_v7",
            "generic_turboquant_kv": False,
        },
        "quantization": {
            "method": "affine+mxtq" if FORMAT == "jangtq" else "affine",
            "routed_experts": {
                "bits": profile_bits,
                "codec": "mxtq" if FORMAT == "jangtq" else "affine",
                "bit_plan": routed_expert_bit_plan,
            },
            "non_routed": {
                "bits": nonrouted_bits_meta,
                "codec": "affine",
                "group_size": 32,
            },
            "critical_control_tensors": "source-f32",
        },
        "source_model": str(src),
        "source_config": {
            "n_routed_experts": src_cfg.get("n_routed_experts"),
            "num_hidden_layers": src_cfg.get("num_hidden_layers"),
            "n_hash_layers": src_cfg.get("num_hash_layers"),
        },
        "routed_expert_bit_plan": routed_expert_bit_plan,
        "mxtq_bits": mxtq_bits_meta if FORMAT == "jangtq" else {},
        # DSV4-Flash chat + reasoning + tool-parser metadata for runtime wiring.
        # Python loader can use this to auto-wire chat encoding; Swift port
        # reads this to know which encoder to use.
        "model_family": "deepseek_v4",
        "chat": {
            "encoder": "encoding_dsv4",  # Python module in ./encoding/
            "encoder_fn": "encode_messages",
            "chat_template_source": "builtin_encoding_module",
            "has_tokenizer_chat_template": False,  # tokenizer_config.json has no chat_template
            "bos_token": "<｜begin▁of▁sentence｜>",
            "eos_token": "<｜end▁of▁sentence｜>",
            "bos_token_id": 0,
            "eos_token_id": 1,
            "role_tokens": {
                "user": "<｜User｜>",
                "assistant": "<｜Assistant｜>",
                "latest_reminder": "<｜latest_reminder｜>",
            },
            "reasoning": {
                "supported": True,
                "modes": ["chat", "thinking"],
                "default_mode": "chat",
                "thinking_start": "<think>",
                "thinking_end": "</think>",
                # "chat" mode: prompt ends with <Assistant></think> (empty reasoning closed)
                # "thinking" mode: prompt ends with <Assistant><think> (open, model fills)
                "reasoning_effort_levels": ["max", "high", None],
                "drop_earlier_reasoning": True,  # drop_thinking in encode_messages
            },
            "tool_calling": {
                "supported": True,
                "parser": "dsml",  # DeepSeek Markup Language (｜DSML｜...)
                "dsml_token": "｜DSML｜",
                "tool_calls_block": "tool_calls",
                "invoke_block": "invoke",
                "parameter_block": "parameter",
                "tool_output_tag": "tool_result",
            },
            "sampling_defaults": {
                # DSV4-Flash chat defaults. T=0.6/top_p=0.95 match the
                # standalone JANG reference path. Repetition penalty is
                # mode-split because the model's degeneration profile is
                # different in chat vs thinking:
                #   * thinking mode: keep neutral (1.0). >1.0 makes
                #     vMLX thinking mode fail to close </think>.
                #   * chat mode: 1.05. With penalty 1.0 long chat replies
                #     drift into single-token loops on K bundles
                #     (live-observed on JANGTQ_K). 1.05 is light enough
                #     to preserve fluency, hard enough to break loops.
                # max_new_tokens 4096 because thinking traces routinely run
                # 1500–3500 chars and 300 was clipping mid-reasoning.
                "temperature": 0.6,
                "top_p": 0.95,
                "repetition_penalty": 1.0,
                "repetition_penalty_thinking": 1.0,
                "repetition_penalty_chat": 1.05,
                "max_new_tokens": 4096,
            },
        },
    }, indent=2))

    # Copy aux files (tokenizer, chat template, modeling files if any)
    copied = 0
    for p in src.iterdir():
        if p.is_file() and not p.name.endswith(".safetensors") \
                and p.name not in ("config.json", "model.safetensors.index.json"):
            shutil.copy2(p, dst / p.name)
            copied += 1
    # Also copy encoding/ directory (DSV4's Python chat-template impl)
    enc = src / "encoding"
    if enc.is_dir():
        shutil.copytree(enc, dst / "encoding", dirs_exist_ok=True)
        copied += 1
    print(f"[convert] copied {copied} aux files/dirs")

    # Force eos_token_id to LIST form across all three config files. Upstream
    # DSV4 ships single-int eos=1; without <｜User｜>=128803 also as a stop,
    # the model auto-continues past <｜end▁of▁sentence｜> into a fake user
    # turn, causing "🤖 My name is..." restart loops on multi-turn chat.
    # Resolve token ids dynamically from the actual tokenizer (don't hardcode).
    try:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(str(dst), trust_remote_code=True)
        _eos_id = _tok.convert_tokens_to_ids("<｜end▁of▁sentence｜>")
        _user_id = _tok.convert_tokens_to_ids("<｜User｜>")
        if _eos_id is not None and _user_id is not None and _eos_id != _user_id:
            _eos_list = sorted({_eos_id, _user_id})
            for _fn in ("config.json", "generation_config.json", "tokenizer_config.json"):
                _p = dst / _fn
                if not _p.exists():
                    continue
                _d = json.loads(_p.read_text())
                _d["eos_token_id"] = _eos_list
                _p.write_text(json.dumps(_d, indent=2, ensure_ascii=False))
            print(f"[convert] forced eos_token_id={_eos_list} in 3 config files")
    except Exception as _eos_err:
        print(f"[convert] WARN eos list patch failed: {_eos_err}")

    if FORMAT == "jangtq":
        finalize_jangtq_bundle(
            dst,
            prestack=prestack,
            build_sidecar=build_sidecar,
        )

    elapsed = time.time() - t_start
    print(f"\nDONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  mxtq={totals['mxtq']}  affine={totals['affine']}  passthrough={totals['passthrough']}")
    print(f"  output size: {total_bytes / 1e9:.1f} GB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--profile", type=int, default=2, choices=(2, 3, 4, 5, 6, 8),
                    help="Routed-expert bit count.")
    ap.add_argument("--format", default="jangtq", choices=("jang", "jangtq"),
                    help="jang=standard affine everywhere; jangtq=MXTQ for 2-bit routed.")
    ap.add_argument("--variant", default="V3", choices=("std", "K", "V3"),
                    help="std=legacy baseline with MTP shipped; K=MAX-QUALITY 70-80GB "
                         "profile (drops MTP head, all non-routed stay 8-bit); "
                         "V3=K + hash layers 0-2 routed lifted to 4-bit MXTQ "
                         "(target ~80%% MMLU, ~80GB); production candidate needs DSV4_V3_PLAN_PATH.")
    ap.add_argument("--no-prestack", action="store_true",
                    help="Debug only: leave per-expert TQ tensors instead of "
                         "finalizing to JANGTQ-PRESTACK layout.")
    ap.add_argument("--no-sidecar", action="store_true",
                    help="Debug only: skip jangtq_runtime.safetensors sidecar.")
    args = ap.parse_args()
    global FORMAT, VARIANT
    FORMAT = args.format
    VARIANT = args.variant
    convert(
        args.src,
        args.dst,
        args.profile,
        prestack=not args.no_prestack,
        build_sidecar=not args.no_sidecar,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
