"""Gemma 4 (gemma4_unified) omni-modal -> JANG (MLX-native affine) conversion.

Produces a JANG_4M bundle: **8-bit affine attention (q/k/v/o) + 8-bit MoE
router projections + 4-bit affine MLP/expert bulk weights**, with the tied
token embedding/output projection, Gemma 4 per-layer embedding/projector
gates, thin multimodal embedders, norms, and layer_scalar kept fp16.

This reuses the verified gemma4 plumbing from `convert_gemma4_mxfp.py`
(`sanitize_key`, multimodal passthrough fragments, single-file scan, no-+1 norm
passthrough, k_eq_v missing-v_proj tolerance) and differs only in:

  * affine quantization (`mode="affine"`) at per-tier bit widths, and
  * writing a CORRECT mixed-bit `config.json["quantization"]` block —
    top-level `{group_size, bits=8, mode="affine"}` PLUS a per-module override
    `{bits=4,...}` for every 4-bit module. Without the per-module overrides a
    loader dequantizes the 8-bit attention with the 4-bit kernel and emits
    garbage (the "config-metadata bit bug",
    research/JANGTQ-CONFIG-METADATA-BUG-2026-04-24.md).

MoE note: Gemma 4 26B-A4B has a router. The router projection stays 8-bit;
the stacked expert gate/up/down payload remains 4-bit.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import mlx.core as mx
from safetensors.numpy import save_file

from jang_tools.capabilities import build_capabilities, verify_directory
from jang_tools.convert import _remove_stale_jang_artifacts
from jang_tools.format.aligned_safetensors import rewrite_aligned_safetensors
from jang_tools.progress import ProgressEmitter
from jang_tools.convert_gemma4_mxfp import (
    MAX_SHARD,
    SIDECAR_FILES,
    _MULTIMODAL_FRAGMENTS,
    _copy_sidecars,
    _load_tensor,
    _prepare_passthrough,
    _scan_source,
    sanitize_key,
)

# JANG_4M dense tier widths.
ATTN_BITS = 8       # CRITICAL: self_attn q/k/v/o
ROUTER_BITS = 8     # CRITICAL: MoE router projection
MLP_BITS = 4        # COMPRESS: mlp/expert gate/up/down
DEFAULT_TOP_BITS = 8  # config.json top-level default (matches the 8-bit majority)
# None => fp16 passthrough (stock JANG_4M). Set via --embed-bits. The tied
# embedding is both the input lookup AND the lm_head, so quantizing it trades
# lookup accuracy against logit accuracy — that is why it is opt-in.
EMBED_BITS: int | None = None


def jang_bits(tensor_name: str) -> int | None:
    """Return affine bit width for a tensor, or None for fp16 passthrough."""
    name = tensor_name.lower()
    if tensor_name.endswith("_scale_inv"):
        return None
    if any(frag in name for frag in _MULTIMODAL_FRAGMENTS):
        return None
    if (
        tensor_name.endswith("embed_tokens_per_layer.weight")
        or tensor_name.endswith("per_layer_input_gate.weight")
        or tensor_name.endswith("per_layer_projection.weight")
    ):
        return None
    if (
        "norm" in name
        or tensor_name.endswith(".bias")
        or tensor_name.endswith("layer_scalar")
        or tensor_name.endswith("pos_embedding")
        or tensor_name.endswith("embed_scale")
    ):
        return None
    if len(tensor_name.split(".")) == 1:
        return None
    if ".self_attn." in name and any(
        name.endswith(f"{p}.weight") for p in ("q_proj", "k_proj", "v_proj", "o_proj")
    ):
        return ATTN_BITS
    if ".router." in name:
        if name.endswith(".proj.weight"):
            return ROUTER_BITS
        return None
    if tensor_name.endswith("embed_tokens.weight"):
        return EMBED_BITS
    # mlp gate/up/down + any other 2D decoder linear
    return MLP_BITS


# --- dynamic per-projection bit plan -------------------------------------
#
# The stock tier widths above are uniform: every routed-expert projection gets
# MLP_BITS. A bit plan overrides that per projection (and optionally per layer),
# so the corpus-derived expert profile can starve the projections it shows to be
# cold while `gate_proj` — the SiLU amplifier, and the dominant sensitivity —
# stays at 4-bit.
#
# This matters on Gemma 4 specifically. `allocate._apply_mlp_asymmetry_floor`
# is the guard that keeps `gate_proj` >= 4 and `down_proj` >= 3 to stop 2-bit
# repetition loops, but it only fires at `_MLP_ASYMMETRY_MIN_EXPERTS = 256`.
# Gemma 4 26B-A4B has 128 experts, so the floor never fires here and the plan
# has to carry that protection explicitly.
#
# Plan JSON:
#   {"default": {"gate_proj": 4, "up_proj": 2, "down_proj": 2},
#    "layers":  {"0": {"up_proj": 4, "down_proj": 4}, "29": {...}},
#    "group_size": {"up_proj": 32}}
#
# `group_size` is per projection because the per-module override block already
# records group_size alongside bits, so a mixed-group bundle needs no format
# change. Scales+biases are fp16 per group and do NOT shrink with bit width —
# at group 32 that is a full extra bit/weight, so group size is the difference
# between a "2-bit" tensor costing 2 bits and costing 3.
_BIT_PLAN: dict | None = None
_GPTQ_HINV: dict[int, np.ndarray] = {}
_GPTQ_DIR: Path | None = None


def _layer_of(name: str) -> int | None:
    parts = name.split(".")
    for i, tok in enumerate(parts):
        if tok == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


def _proj_of(name: str) -> str | None:
    for p in ("gate_proj", "up_proj", "down_proj"):
        if p in name:
            return p
    return None


def plan_bits(job_base: str, default_bits: int) -> int:
    """Resolve bit width for one projection, honoring the bit plan."""
    if _BIT_PLAN is None:
        return default_bits
    # Attention: optional per-layer override via an "attention" section.
    #
    # Motivation, measured on this checkpoint: relative weight error is 0.0066 for
    # 8-bit attention but 0.42 for a 2-bit expert gate_proj -- a 64x precision
    # imbalance while sub-10GB plans leave gate layers at 2 bits. Sliding-attention
    # layers only look back `sliding_window` tokens, so they are the cheapest place
    # to give bits back. Absent an "attention" section this is a no-op and every
    # existing plan behaves exactly as before.
    if ".self_attn." in job_base:
        att = _BIT_PLAN.get("attention")
        if not att:
            return default_bits
        layer = _layer_of(job_base)
        per_layer = (att.get("layers") or {})
        if str(layer) in per_layer:
            return int(per_layer[str(layer)])
        return int(att.get("default", default_bits))
    proj = _proj_of(job_base)
    if proj is None or ".experts." not in job_base:
        return default_bits
    layer = _layer_of(job_base)
    per_layer = (_BIT_PLAN.get("layers") or {}).get(str(layer), {})
    if proj in per_layer:
        return int(per_layer[proj])
    return int((_BIT_PLAN.get("default") or {}).get(proj, default_bits))


def plan_group_size(job_base: str, default_gs: int) -> int:
    if _BIT_PLAN is None:
        return default_gs
    if ".self_attn." in job_base:
        att = _BIT_PLAN.get("attention") or {}
        # Measured: 4-bit g64 and 4-bit g128 are indistinguishable in output error
        # (1.0249e5 vs 1.0225e5 on expert gate), so g64 at 4 bits is pure waste.
        return int(att.get("group_size", default_gs))
    proj = _proj_of(job_base)
    if proj is None or ".experts." not in job_base:
        return default_gs
    return int((_BIT_PLAN.get("group_size") or {}).get(proj, default_gs))


def _load_hinv(layer: int, in_features: int) -> np.ndarray | None:
    """Per-(layer, input-dim) H_inv, cached.

    A Gemma-4 MoE layer needs TWO Hessians, not one: gate_proj/up_proj consume
    the expert input (hidden_size), while down_proj consumes the GeGLU
    intermediate (moe_intermediate_size). Keying on layer alone silently drops
    GPTQ for down_proj, because the shape guard in _affine_quantize would reject
    the mismatched H and fall back to RTN.
    """
    if _GPTQ_DIR is None or layer is None:
        return None
    key = (layer, in_features)
    if key in _GPTQ_HINV:
        return _GPTQ_HINV[key]
    for stem in (
        f"H_L{layer}_d{in_features}.npy",
        f"H_L{layer}.npy",
        f"H_FP8_L{layer}.npy",
    ):
        path = _GPTQ_DIR / stem
        if path.exists():
            H = np.load(str(path)).astype(np.float64)
            if H.shape[0] != in_features:
                continue
            # Escalating damping. The DSV4 lesson: f32 inversion of a
            # rank-deficient H silently falls back to RTN, so this stays f64
            # and raises damping until the inverse is finite.
            mean_diag = float(np.mean(np.diag(H)))
            for mult in (0.01, 0.05, 0.1, 0.5, 1.0):
                try:
                    Hd = H + np.eye(H.shape[0]) * (mult * mean_diag)
                    Hinv = np.linalg.inv(Hd)
                    if np.all(np.isfinite(Hinv)):
                        _GPTQ_HINV[key] = Hinv
                        return Hinv
                except np.linalg.LinAlgError:
                    continue
            print(f"    ! layer {layer} d{in_features}: H not invertible -> RTN")
            return None
    return None


def _affine_quantize(
    tensor: np.ndarray,
    *,
    bits: int,
    group_size: int,
    h_inv: np.ndarray | None = None,
):
    if tensor.ndim >= 3:
        original_shape = tensor.shape
        tensor = tensor.reshape(-1, tensor.shape[-1])
    else:
        original_shape = None

    # GPTQ only below 4 bits. At 4-bit these are Google's QAT weights sitting on
    # the grid QAT trained them for, so RTN is already near-optimal there and
    # GPTQ buys ~nothing. Below 4 bits the weights are off that grid and the
    # Hessian is what recovers the error.
    if h_inv is not None and bits < 4 and tensor.shape[1] == h_inv.shape[0]:
        from .gptq_mlx import gptq_quantize_fast_with_hinv

        weight, scales, biases = gptq_quantize_fast_with_hinv(
            tensor.astype(np.float32), h_inv, bits=bits, group_size=group_size
        )
    else:
        q_w, q_s, q_b = [], [], []
        chunk_rows = max(1, min(tensor.shape[0], 100_000_000 // max(1, tensor.shape[1])))
        for start in range(0, tensor.shape[0], chunk_rows):
            chunk = mx.array(tensor[start : start + chunk_rows].astype(np.float16))
            qw, qs, qb = mx.quantize(chunk, group_size=group_size, bits=bits, mode="affine")
            q_w.append(np.array(qw)); q_s.append(np.array(qs)); q_b.append(np.array(qb))
            mx.eval(qw, qs, qb)
            del chunk, qw, qs, qb
        weight = np.concatenate(q_w, axis=0)
        scales = np.concatenate(q_s, axis=0)
        biases = np.concatenate(q_b, axis=0)

    if original_shape is not None:
        weight = weight.reshape(original_shape[0], original_shape[1], -1)
        scales = scales.reshape(original_shape[0], original_shape[1], -1)
        biases = biases.reshape(original_shape[0], original_shape[1], -1)
    return weight, scales, biases


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Gemma 4 (gemma4_unified) source to JANG_4M.")
    p.add_argument("src", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quiet-text", action="store_true")
    p.add_argument(
        "--bit-plan",
        type=Path,
        default=None,
        help="JSON per-projection (and per-layer) routed-expert bit plan. "
             "Keys: default{gate_proj,up_proj,down_proj}, layers{N:{...}}, group_size{proj:N}.",
    )
    p.add_argument(
        "--gptq-hessian-dir",
        type=Path,
        default=None,
        help="Directory of per-layer H_L{idx}.npy expert-input Hessians. "
             "GPTQ is applied only to tensors below 4 bits.",
    )
    p.add_argument(
        "--embed-bits",
        type=int,
        default=None,
        help="Quantize the tied embed_tokens to this width (default: fp16 passthrough). "
             "8 saves ~0.74 GB on Gemma 4's 262k vocab.",
    )
    return p.parse_args()


def main() -> None:
    global _BIT_PLAN, _GPTQ_DIR, EMBED_BITS
    args = parse_args()
    gs = args.group_size
    profile = "JANG_4M"
    if args.bit_plan is not None:
        _BIT_PLAN = json.loads(args.bit_plan.expanduser().read_text(encoding="utf-8"))
    if args.gptq_hessian_dir is not None:
        _GPTQ_DIR = args.gptq_hessian_dir.expanduser()
        if not _GPTQ_DIR.is_dir():
            raise SystemExit(f"--gptq-hessian-dir is not a directory: {_GPTQ_DIR}")
    if args.embed_bits is not None:
        EMBED_BITS = int(args.embed_bits)
    progress = ProgressEmitter(json_to_stderr=False, quiet_text=args.quiet_text)
    src = args.src.expanduser()
    out = args.out.expanduser()

    config = json.loads((src / "config.json").read_text(encoding="utf-8"))
    text_cfg = config.get("text_config", config)
    has_vision = bool(config.get("vision_config"))
    has_audio = bool(config.get("audio_config"))
    has_video = bool(config.get("video_config"))
    n_layers = int(text_cfg.get("num_hidden_layers", 0))
    layer_types = text_cfg.get("layer_types") or []
    n_full = sum(1 for t in layer_types if t == "full_attention")

    print("=" * 70)
    print(f"  Gemma 4 (gemma4_unified) -> {profile} (dense, affine)")
    print("=" * 70)
    print(f"  Source:  {src}")
    print(f"  Output:  {out}")
    print(f"  Layers:  {n_layers}  (full-attn {n_full} / sliding {n_layers - n_full})")
    print(f"  Bits:    attn={ATTN_BITS}  router={ROUTER_BITS}  mlp={MLP_BITS}  embed=fp16  (group_size {gs})")
    print("  Norm:    scale_shift=0 (NO +1) ; tied/per-layer embeddings, media gates/projectors fp16 ; MTP none")

    progress.phase(1, 3, "scan")
    tensors = _scan_source(src)
    print(f"  Found {len(tensors)} tensors")
    if args.dry_run:
        # Count at job_base granularity, i.e. after the gate_up split, so the
        # numbers reflect the bit plan that will actually be applied rather than
        # the pre-split source names.
        counts: dict[str, int] = {}
        wbytes = 0.0
        obytes = 0.0
        for name, shape, _sf in tensors:
            b = jang_bits(name)
            if b is None or len(shape) < 2:
                counts["fp16"] = counts.get("fp16", 0) + 1
                obytes += float(np.prod(shape)) * 2
                continue
            base = sanitize_key(name)
            base = base[: -len(".weight")] if base.endswith(".weight") else base
            jobs = [(base, list(shape))]
            if ".experts.gate_up_proj" in base:
                half = list(shape)
                half[1] //= 2
                jobs = [
                    (base.replace(".experts.gate_up_proj", ".experts.switch_glu.gate_proj"), half),
                    (base.replace(".experts.gate_up_proj", ".experts.switch_glu.up_proj"), list(half)),
                ]
            elif ".experts.down_proj" in base:
                jobs = [(base.replace(".experts.down_proj", ".experts.switch_glu.down_proj"), list(shape))]
            for jb, jshape in jobs:
                jb_bits = plan_bits(jb, b)
                jb_gs = plan_group_size(jb, gs)
                counts[f"affine-{jb_bits}"] = counts.get(f"affine-{jb_bits}", 0) + 1
                n = float(np.prod(jshape))
                wbytes += n * jb_bits / 8.0
                # scale + bias, both fp16, one pair per group -> fixed overhead
                obytes += (n / jb_gs) * 4.0
        print(json.dumps(counts, indent=2, sort_keys=True))
        print(f"  predicted quantized payload : {wbytes/1e9:6.2f} GB")
        print(f"  predicted scale/bias + fp16 : {obytes/1e9:6.2f} GB")
        print(f"  predicted TOTAL             : {(wbytes+obytes)/1e9:6.2f} GB")
        progress.done(ok=True, output="dry-run")
        return

    out.mkdir(parents=True, exist_ok=True)
    removed = _remove_stale_jang_artifacts(out)
    if removed:
        print(f"  Removed {len(removed)} stale output file(s)")

    shard_idx = 0
    shard_tensors: dict[str, np.ndarray] = {}
    shard_bytes = 0
    shard_map: dict[str, str] = {}
    overrides: dict[str, dict] = {}
    n_affine = n_pass = 0

    def flush_shard() -> None:
        nonlocal shard_idx, shard_tensors, shard_bytes
        if not shard_tensors:
            return
        shard_idx += 1
        name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        shard_path = out / name
        mx.save_safetensors(
            str(shard_path),
            {
                k: (mx.array(v) if isinstance(v, np.ndarray) else v)
                for k, v in shard_tensors.items()
            },
        )
        rewrite_aligned_safetensors(shard_path)
        for k in shard_tensors:
            shard_map[k] = name
        print(f"    Shard {shard_idx}: {len(shard_tensors)} tensors, {shard_bytes / 1e9:.2f} GB")
        shard_tensors = {}
        shard_bytes = 0

    def add_tensor(name: str, arr: np.ndarray) -> None:
        nonlocal shard_bytes
        shard_tensors[name] = arr
        shard_bytes += arr.nbytes
        if shard_bytes >= MAX_SHARD:
            flush_shard()

    progress.phase(2, 3, "convert")
    from tqdm import tqdm
    for tensor_name, shape, sf_path in tqdm(tensors, desc="  Processing"):
        bits = jang_bits(tensor_name)
        out_name = sanitize_key(tensor_name)
        tensor = _load_tensor(sf_path, tensor_name, shape)
        if bits is None or tensor.ndim < 2:
            tensor = _prepare_passthrough(out_name, tensor)
            if any(frag in out_name.lower() for frag in _MULTIMODAL_FRAGMENTS):
                add_tensor(out_name, mx.array(tensor.astype(np.float32), dtype=mx.bfloat16))
            else:
                add_tensor(out_name, tensor.astype(np.float16))
            n_pass += 1
        else:
            base = out_name[: -len(".weight")] if out_name.endswith(".weight") else out_name
            quant_jobs = [(base, tensor)]
            if ".experts.gate_up_proj" in base:
                if tensor.ndim < 3 or tensor.shape[1] % 2 != 0:
                    raise ValueError(
                        f"Gemma4 expert gate_up tensor has unexpected shape: "
                        f"{tensor_name} {tensor.shape}"
                    )
                gate, up = np.split(tensor, 2, axis=1)
                quant_jobs = [
                    (
                        base.replace(
                            ".experts.gate_up_proj", ".experts.switch_glu.gate_proj"
                        ),
                        gate,
                    ),
                    (
                        base.replace(
                            ".experts.gate_up_proj", ".experts.switch_glu.up_proj"
                        ),
                        up,
                    ),
                ]
            elif ".experts.down_proj" in base:
                quant_jobs = [
                    (
                        base.replace(
                            ".experts.down_proj", ".experts.switch_glu.down_proj"
                        ),
                        tensor,
                    )
                ]

            for job_base, job_tensor in quant_jobs:
                # Resolve bits per job_base, not per source tensor: gate_up_proj
                # is one fused source tensor but two output projections, and the
                # whole point of the plan is to give gate and up different widths.
                job_bits = plan_bits(job_base, bits)
                job_gs = plan_group_size(job_base, gs)
                h_inv = (
                    _load_hinv(_layer_of(job_base), int(job_tensor.shape[-1]))
                    if job_bits < 4
                    else None
                )
                qw, qs, qb = _affine_quantize(
                    job_tensor, bits=job_bits, group_size=job_gs, h_inv=h_inv
                )
                add_tensor(f"{job_base}.weight", qw)
                add_tensor(f"{job_base}.scales", qs)
                add_tensor(f"{job_base}.biases", qb)
                # Per-module override ONLY when this module differs from the 8-bit
                # top-level default (i.e. the 4-bit MLP/embed modules).
                if job_bits != DEFAULT_TOP_BITS or job_gs != gs:
                    overrides[job_base] = {
                        "group_size": job_gs,
                        "bits": job_bits,
                        "mode": "affine",
                    }
                n_affine += 1
                del qw, qs, qb
        del tensor
        if (n_affine + n_pass) % 200 == 0:
            gc.collect(); mx.clear_cache()

    flush_shard()

    progress.phase(3, 3, "write")
    for idx in range(1, shard_idx + 1):
        old = out / f"model-{idx:05d}-of-XXXXX.safetensors"
        new = out / f"model-{idx:05d}-of-{shard_idx:05d}.safetensors"
        if old.exists():
            old.rename(new)
    shard_map = {k: v.replace("XXXXX", f"{shard_idx:05d}") for k, v in shard_map.items()}
    total_size = sum((out / name).stat().st_size for name in set(shard_map.values()))
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"format": "jang_affine", "total_size": total_size}, "weight_map": shard_map}, indent=2),
        encoding="utf-8",
    )

    # config.json: mlx-loadable mixed-bit quantization block.
    config.pop("quantization_config", None)
    config["weight_format"] = "jang_affine"
    config["has_vision"] = has_vision
    config["has_audio"] = has_audio
    config["has_video"] = has_video
    config["modalities"] = {
        "text": True,
        "vision": has_vision,
        "audio": has_audio,
        "video": has_video,
    }
    quant_block: dict = {
        "group_size": gs,
        "bits": DEFAULT_TOP_BITS,
        "mode": "affine",
        "tied_embedding": "fp16_passthrough",
    }
    quant_block.update(overrides)
    config["quantization"] = quant_block
    caps = build_capabilities({}, config, out)
    if caps is not None:
        config["capabilities"] = caps
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    jang_config = {
        "version": 2,
        "weight_format": "jang_affine",
        "profile": profile,
        "source_model": {
            "name": src.name,
            "architecture": text_cfg.get("model_type", config.get("model_type", "gemma4_unified_text")),
        },
        "has_vision": has_vision,
        "has_audio": has_audio,
        "has_video": has_video,
        "modalities": {
            "text": True,
            "vision": has_vision,
            "audio": has_audio,
            "video": has_video,
        },
        "quantization": {
            "method": "jang_affine",
            "quantization_backend": "mx.quantize",
            "mode": "affine",
            "group_size": gs,
            "tier_bits": {
                "attention": ATTN_BITS,
                "router": ROUTER_BITS,
                "mlp": MLP_BITS,
                "embed": 16,
                "per_layer_media": 16,
            },
            "tied_embedding": "fp16_passthrough",
            "norm_convention": "gemma4_scale_shift_zero",
            "multimodal": "fp16_passthrough_embedders_early_fusion",
            "preserved_modal_components": [
                component
                for enabled, component in (
                    (has_vision, "vision_embedder"),
                    (has_audio, "audio_embedder"),
                )
                if enabled
            ],
            "mtp_policy": "none",
            "per_module_override_count": len(overrides),
            "passthrough_tensor_count": n_pass,
        },
        "runtime": {
            "total_weight_bytes": total_size,
            "total_weight_gb": round(total_size / (1024 ** 3), 2),
            "attention": "hybrid_swa_full",
            "sliding_window": text_cfg.get("sliding_window"),
            "attention_k_eq_v_on_full_layers": bool(text_cfg.get("attention_k_eq_v")),
            "full_attention_layers": [i for i, t in enumerate(layer_types) if t == "full_attention"],
        },
    }
    caps = build_capabilities(jang_config, config, out)
    if caps is not None:
        jang_config["capabilities"] = caps
    (out / "jang_config.json").write_text(json.dumps(jang_config, indent=2), encoding="utf-8")
    _copy_sidecars(src, out)

    ok, msg = verify_directory(out)
    print(f"  verify: {msg}")
    if not ok:
        raise SystemExit(f"capabilities verify failed: {msg}")

    print("\n  Done!")
    print(f"  Affine tensors:      {n_affine}  (per-module 4-bit overrides: {len(overrides)})")
    print(f"  Passthrough tensors: {n_pass}")
    print(f"  Output:              {out}")
    progress.done(ok=True, output=str(out))


if __name__ == "__main__":
    main()
