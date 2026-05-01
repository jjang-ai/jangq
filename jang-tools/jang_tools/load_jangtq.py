"""
JANGTQ loader — drop-in MLX inference path for `weight_format: mxtq` models.
Created by Jinho Jang (eric@jangq.ai)

JANGTQ ("Turbo Quant") is the most-compressed, highest-quality JANG quant
format. Weights stay in their codebook+Hadamard form at runtime — no
dequant to affine. The matmul path uses custom Metal kernels (P2-P18)
that read packed uint32 weights, look up centroids in a 4-entry codebook,
and accumulate dot products against a Hadamard-rotated input.

This module is fully self-contained: the only externals it needs are
`mlx`, `mlx_lm`, and `jang_tools.{loader, turboquant}`. Drop into any
inference engine by calling `load_jangtq_model(path)` and using the
returned `(model, tokenizer)` pair with `mlx_lm.generate` (or any equivalent
generate loop).

What gets applied on load (all class-level monkeypatches that survive
the function call so subsequent decodes are fast):

  * P3  — single-dispatch multi-block Hadamard (`jang_tools.turboquant.hadamard_kernel`)
  * P15 — `mx.compile`'d router math + compile-friendly `_mlp` fast path
  * P17 — thread-tiling sweet spot OPT=10 (fused_gate_up_swiglu) and
          OPT=20 (gather_tq_matmul) — re-swept per Apple GPU generation
  * P18 — QKV fusion in attention (3 quantized matmuls → 1 + slice views)

Architectures supported:
  * MiniMax M2.7 (`minimax_m2`) — standard Q/K/V attention, sigmoid+bias router,
    no shared expert; all patches apply. Measured 44.3 tok/s on M3 Ultra.
  * Qwen3.5 / Qwen3.6 / Qwen3-Next (`qwen3_5_moe`, `qwen3_next`) — hybrid
    linear_attn + full_attn, softmax+topk router, shared expert with sigmoid
    gate, pre-stacked experts with combined gate_up_proj (split at load).
    P3/P15 apply to all MoE layers; P18 QKV fusion applies only to full_attn
    layers (linear_attn layers skip silently — they have a different layout).
  * GLM-5.1 (`glm_moe_dsa`) — MLA attention; P18 QKV fusion will silently
    skip because it has q_a_proj/kv_a_proj_with_mqa instead of q/k/v_proj.
    The MoE-side P3/P15 still apply. Shared expert (if present) handled.

See `research/JANGTQ-REFERENCE.md` for full architecture, math, kernel
inventory, and known traps.
"""
import json, gc, re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from jang_tools.turboquant.tq_kernel import TurboQuantLinear, TurboQuantSwitchLinear


# ── Runtime config-metadata fix (2026-04-25) ──────────────────────────────
#
# Background: legacy JANG/JANGTQ bundles ship `config.json` with a uniform
# `quantization = {"bits": N, "group_size": M}` block, no per-module
# overrides. But the on-disk weights are MIXED bit-widths (routed experts at
# N-bit, attention/embed/lm_head at 8-bit affine). MLX's loader uses the
# top-level `bits` for every QuantizedLinear → the 8-bit-stored attention
# weights get dequantized with the N-bit kernel → silent garbage on hard
# tasks (HumanEval pass@1 42% instead of 70%).
#
# This function fixes ALL such bundles at LOAD TIME, regardless of whether
# config.json was patched on disk. Algorithm:
#   1. Walk safetensors index, find every `(.weight, .scales, .biases)` triple
#   2. Infer (bits, group_size) from shape ratio:
#        bits × group_size = 32 × (weight.shape[-1] / scales.shape[-1])
#   3. Try fixed-preference (bits, gsz) candidates, prefer 8-bit (matches
#      converter convention for non-routed weights). First valid wins.
#   4. Patch `model_config["quantization"]` IN MEMORY with per-module
#      overrides + `bits=8 mode=affine` top-level. Idempotent: if config
#      already has correct overrides, this is a no-op.
#
# Verified: produces bit-for-bit identical overrides vs the trusted local
# `patch_dsv4_quant_config.py` on 3 spot-checked bundles (DSV4-JANGTQ,
# Holo3-JANGTQ4, Nemotron-Cascade-2-JANG_4M).
def _infer_quant_overrides_from_shapes(model_path: Path) -> tuple[dict, list[str]]:
    """Return (overrides_dict, warnings) inferred from safetensors metadata.

    overrides_dict maps `<module_path>` → `{"bits": N, "group_size": M, "mode": "affine"}`.
    """
    from safetensors import safe_open

    warnings: list[str] = []
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        # Single-shard bundle — fall back to globbing safetensors files
        single = list(model_path.glob("*.safetensors"))
        if not single:
            return {}, ["no safetensors found"]
        weight_map = {}
        for sf in single:
            with safe_open(str(sf), framework="numpy") as f:
                for k in f.keys():
                    weight_map[k] = sf.name
    else:
        try:
            weight_map = json.loads(index_path.read_text())["weight_map"]
        except Exception as e:
            return {}, [f"index parse failed: {e}"]

    keys = set(weight_map.keys())
    quantized_bases: set[str] = set()
    for k in keys:
        if k.endswith(".scales"):
            base = k[:-len(".scales")]
            if (base + ".weight") in keys and (base + ".biases") in keys:
                quantized_bases.add(base)

    if not quantized_bases:
        return {}, ["no quantized triples found"]

    # Open shards once, cache shapes
    shape_cache: dict[str, tuple] = {}
    open_handles: dict[str, "safe_open"] = {}
    try:
        for base in sorted(quantized_bases):
            for suffix in (".weight", ".scales"):
                k = base + suffix
                fname = weight_map.get(k)
                if not fname:
                    continue
                if fname not in open_handles:
                    open_handles[fname] = safe_open(
                        str(model_path / fname), framework="numpy"
                    )
                    open_handles[fname].__enter__()
                f = open_handles[fname]
                try:
                    shape_cache[k] = tuple(f.get_slice(k).get_shape())
                except Exception:
                    pass
    finally:
        for h in open_handles.values():
            try:
                h.__exit__(None, None, None)
            except Exception:
                pass

    # Preference order: prefer 8-bit (converter convention for non-routed),
    # then 4-bit, 2-bit, 3-bit, 6-bit. group_size 32 first, then 64, 128.
    candidates = [
        (8, 32), (8, 64), (8, 128),
        (4, 32), (4, 64), (4, 128),
        (2, 32), (2, 64), (2, 128),
        (3, 32), (3, 64), (3, 128),
        (6, 32), (6, 64), (6, 128),
    ]

    overrides: dict[str, dict] = {}
    for base in sorted(quantized_bases):
        w_shape = shape_cache.get(base + ".weight")
        s_shape = shape_cache.get(base + ".scales")
        if not w_shape or not s_shape or len(w_shape) < 2 or len(s_shape) < 2:
            continue
        w_inner = int(w_shape[-1])
        s_inner = int(s_shape[-1])
        if s_inner == 0:
            continue
        # bits × group_size = 32 × (w_inner / s_inner)
        # Walk candidates in preference order, take first that fits.
        chosen = None
        for bits, gsz in candidates:
            in_feat = s_inner * gsz
            bits_check = w_inner * 32.0 / in_feat if in_feat else 0
            if bits_check == bits and bits in (2, 3, 4, 6, 8):
                chosen = (bits, gsz)
                break
        if not chosen:
            warnings.append(
                f"could not infer quant for {base} "
                f"(w_inner={w_inner} s_inner={s_inner})"
            )
            continue
        bits, gsz = chosen
        overrides[base] = {"bits": bits, "group_size": gsz, "mode": "affine"}

    return overrides, warnings


def _patch_quant_config_inplace(model_path: Path, model_config: dict) -> dict:
    """Patch `model_config["quantization"]` in place if it's missing per-module
    overrides for a mixed-bit bundle. Returns dict of stats for logging.

    Safe to call on already-patched configs (idempotent).
    """
    q = model_config.get("quantization") or {}
    existing_overrides = {k: v for k, v in q.items() if isinstance(v, dict)}
    top_bits = q.get("bits")

    inferred, warns = _infer_quant_overrides_from_shapes(model_path)
    if not inferred:
        return {"action": "no_quant_layers", "warnings": warns}

    # Distribution of inferred bits — used to decide top-level fallback.
    bit_dist: dict[int, int] = {}
    for ov in inferred.values():
        bit_dist[ov["bits"]] = bit_dist.get(ov["bits"], 0) + 1
    gsz_dist: dict[int, int] = {}
    for ov in inferred.values():
        gsz_dist[ov["group_size"]] = gsz_dist.get(ov["group_size"], 0) + 1

    # Mismatches: keys where existing override (or top-level fallback)
    # disagrees with shape-inferred bits.
    mismatches: list[tuple[str, int, int]] = []
    for base, ov in inferred.items():
        existing = existing_overrides.get(base)
        if existing is not None:
            if existing.get("bits") != ov["bits"]:
                mismatches.append((base, existing.get("bits"), ov["bits"]))
        elif top_bits is not None and top_bits != ov["bits"]:
            mismatches.append((base, top_bits, ov["bits"]))

    is_bug = len(mismatches) > 0 and not existing_overrides
    is_partial_bug = len(mismatches) > 0 and existing_overrides
    is_clean = len(mismatches) == 0

    # Always write back the full set of per-module overrides (idempotent).
    # Pick top-level bits/gsz: prefer to KEEP existing if it's a safe
    # fallback (matches one of the inferred bits AND one of the inferred
    # group sizes). Otherwise pick majority of inferred.
    new_top_bits = top_bits if (top_bits in bit_dist) else (
        8 if 8 in bit_dist else max(bit_dist.items(), key=lambda kv: kv[1])[0]
    )
    existing_top_gsz = q.get("group_size")
    new_top_gsz = existing_top_gsz if (existing_top_gsz in gsz_dist) else (
        max(gsz_dist.items(), key=lambda kv: kv[1])[0]
    )
    # If the existing config was already clean (no mismatches AND we have
    # full per-module override coverage), keep it byte-identical — even if
    # the top-level gsz/bits is a stale fallback. With overrides covering
    # everything, top-level is dead metadata at runtime. Idempotency.
    if (
        is_clean
        and existing_overrides
        and len(existing_overrides) == len(inferred)
    ):
        return {
            "action": "verified_clean",
            "old_top_bits": top_bits,
            "old_overrides": len(existing_overrides),
            "new_top_bits": top_bits,
            "new_overrides": len(existing_overrides),
            "mismatches": 0,
            "bit_distribution": bit_dist,
            "warnings": warns,
        }
    new_q: dict = {"group_size": new_top_gsz, "bits": new_top_bits, "mode": "affine"}
    new_q.update(inferred)
    model_config["quantization"] = new_q
    model_config.pop("quantization_config", None)

    return {
        "action": "patched_bug" if is_bug else (
            "patched_partial" if is_partial_bug else "verified_clean"
        ),
        "old_top_bits": top_bits,
        "old_overrides": len(existing_overrides),
        "new_top_bits": new_top_bits,
        "new_overrides": len(inferred),
        "mismatches": len(mismatches),
        "bit_distribution": bit_dist,
        "warnings": warns,
    }


def _ensure_rope_parameters_inplace(config: dict) -> bool:
    """Add rope_parameters if missing. Without it, transformers ≥4.50 falls
    back to bare PreTrainedTokenizerFast, dropping chat_template + special
    tokens. Returns True if a change was made.
    """
    if "rope_parameters" in config:
        return False
    rs = config.get("rope_scaling")
    rope_theta = config.get("rope_theta", 10000)
    if isinstance(rs, dict):
        rp = dict(rs)
        rp.setdefault("rope_type", rp.pop("type", "yarn"))
        rp.setdefault("rope_theta", rope_theta)
    else:
        rp = {"rope_type": "default", "rope_theta": rope_theta}
    for k in ("factor", "beta_fast", "beta_slow", "rope_theta"):
        if k in rp:
            try:
                rp[k] = float(rp[k])
            except Exception:
                pass
    config["rope_parameters"] = rp
    return True


def _apply_wired_limit_safe_default():
    """Set mx.wired_limit to ~70% of total RAM if the caller hasn't already.

    Why: for per-expert JANGTQ bundles (e.g. GLM-5.1-JANGTQ_1L, ~190 GB),
    the loader's `mx.stack(packed_list)` at inference-time creates fresh
    non-mmap-backed stacked expert tensors. If MLX's wired_limit is too
    high (e.g. 240 GB on a 256 GB machine), macOS has no pagecache
    headroom to spill the materialization spike and SIGKILLs the process
    silently (no OOM traceback — just vanishes).

    Ralph iter-14 measurement: setting wired_limit=180 GB on 256 GB
    Mac Studio let the forward complete in 80 s (cold I/O) instead of
    SIGKILL. 70% of total RAM is a safe general-case default; callers
    who have already set their own limit (higher or lower) are respected.

    Enabled only on Apple Silicon macOS where psutil is available.
    No-op elsewhere.
    """
    try:
        import os, psutil, sys, mlx.core as _mx
        if sys.platform != "darwin":
            return
        # Explicit override: JANG_WIRED_LIMIT_GB=N
        override = os.environ.get("JANG_WIRED_LIMIT_GB")
        total_gb = psutil.virtual_memory().total / 1e9
        if override:
            target_gb = int(override)
            why = f"JANG_WIRED_LIMIT_GB={override}"
        else:
            # 70% on big-RAM (≥200 GB), 55% on smaller machines (≤150 GB).
            # MacBook Pro 128 GB: 55% = 70 GB → leaves 58 GB for OS/cache —
            # avoids the 2026-04-24 OOM kill seen at 96 GB wired on 137 GB.
            pct = 0.70 if total_gb >= 200 else 0.55
            target_gb = int(total_gb * pct)
            why = f"~{int(pct*100)}% of {total_gb:.0f} GB total RAM"
        target_gb = max(32, min(target_gb, 220))
        target_bytes = target_gb * 1000 * 1000 * 1000
        _mx.set_wired_limit(target_bytes)
        print(f"  [wired_limit] set to {target_gb} GB ({why})", flush=True)
    except Exception as _e:
        # Non-fatal: older MLX, no psutil, non-Apple OS, etc.
        pass


def load_jangtq_model(model_path, skip_params_eval=False):
    """Load JANGTQ model with TurboQuantLinear (Metal kernel, no dequant).

    Automatically caps MLX's wired_limit at ~70% of total RAM to avoid the
    forward-pass OOM seen on large per-expert bundles (GLM-5.1-JANGTQ_1L)
    where non-mmap stacked expert tensors need pagecache headroom to
    materialize on first forward. See `_apply_wired_limit_safe_default`
    for the Ralph iter-14 investigation.
    """
    _apply_wired_limit_safe_default()

    model_path = Path(model_path)
    # M125 (iter 48): context-manage reads so fds close deterministically.
    with open(model_path / "config.json") as f:
        config = json.load(f)

    jang_cfg_path = model_path / "jang_config.json"
    if jang_cfg_path.exists():
        with open(jang_cfg_path) as f:
            jang_cfg = json.load(f)
    else:
        jang_cfg = {}
    mxtq_seed = jang_cfg.get("mxtq_seed", 42)
    mxtq_bits_map = jang_cfg.get("mxtq_bits", {})

    print(f"Loading JANGTQ: {model_path.name}", flush=True)
    print(f"  seed={mxtq_seed}, bits_map={mxtq_bits_map}", flush=True)

    from mlx_lm.utils import load_config, load_model as _load_skeleton, load_tokenizer
    model_config = load_config(model_path)

    # Auto-register custom model_types with mlx_lm for bundles whose
    # architecture isn't (yet) in mlx_lm main. deepseek_v4 ships in
    # jang_tools.dsv4.mlx_register; importing the package triggers it.
    _model_type = model_config.get("model_type", "")
    if _model_type == "deepseek_v4":
        try:
            import jang_tools.dsv4  # noqa: F401  (registers on import)
        except Exception as _e:
            warnings.warn(f"jang_tools.dsv4 register failed: {_e}")

    if "quantization" not in model_config:
        model_config["quantization"] = {"group_size": 64, "bits": 2}

    # Runtime config-metadata fix: ALWAYS infer per-module overrides from
    # safetensors shapes and patch in-memory config. Idempotent — clean
    # configs verify-pass, broken configs get fixed. This protects against
    # legacy bundles uploaded before the converter metadata fix landed
    # (2026-04-24). See `_infer_quant_overrides_from_shapes` for the math.
    try:
        stats = _patch_quant_config_inplace(model_path, model_config)
        action = stats.get("action", "?")
        if action == "patched_bug":
            print(f"  ⚠ config-metadata BUG detected, patched in-memory: "
                  f"top_bits {stats['old_top_bits']} → {stats['new_top_bits']}, "
                  f"+{stats['new_overrides']} per-module overrides "
                  f"({stats['mismatches']} mismatches)", flush=True)
        elif action == "patched_partial":
            print(f"  ⚠ config-metadata PARTIAL bug, fixing: "
                  f"{stats['mismatches']} per-module mismatches resolved", flush=True)
        elif action == "verified_clean":
            print(f"  ✓ config-metadata clean ({stats['new_overrides']} overrides verified)", flush=True)
    except Exception as e:
        import warnings
        warnings.warn(f"config-from-shape inference skipped: {e}")

    # Tokenizer fallback fix: transformers ≥4.50 needs `rope_parameters`,
    # not legacy `rope_scaling`. Without it `load_tokenizer` falls back to
    # bare PreTrainedTokenizerFast, dropping chat_template + special tokens.
    if _ensure_rope_parameters_inplace(model_config):
        print(f"  ⚠ added rope_parameters (was missing — tokenizer would fall back)", flush=True)

    model, model_config = _load_skeleton(
        model_path, lazy=True, strict=False, model_config=model_config
    )

    _hydrate_jangtq_model(
        model=model,
        model_path=model_path,
        mxtq_seed=mxtq_seed,
        mxtq_bits_map=mxtq_bits_map,
        model_config=model_config,
        skip_params_eval=skip_params_eval,
    )

    eos_ids = config.get("eos_token_id")
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    try:
        tokenizer = load_tokenizer(model_path, eos_token_ids=eos_ids)
    except Exception as _e:
        # Newer model_types (e.g. deepseek_v4) may trip transformers' config
        # validation. Fall back to direct PreTrainedTokenizerFast from tokenizer.json.
        # CRITICAL FIX (2026-04-25): the bare PreTrainedTokenizerFast doesn't
        # read tokenizer_config.json, so it drops chat_template + special
        # tokens — apply_chat_template then raises. Manually rehydrate from
        # tokenizer_config.json to preserve chat-mode functionality without
        # modifying the user's on-disk bundle.
        import warnings
        warnings.warn(f"load_tokenizer failed ({_e}); using PreTrainedTokenizerFast fallback with manual rehydrate")
        from transformers import PreTrainedTokenizerFast
        import os
        tok_file = os.path.join(str(model_path), "tokenizer.json")
        tok_config_path = os.path.join(str(model_path), "tokenizer_config.json")
        # Load tokenizer_config.json kwargs to restore chat_template + special tokens
        tok_kwargs = {}
        if os.path.exists(tok_config_path):
            try:
                with open(tok_config_path) as f:
                    tok_kwargs = json.load(f)
                # Filter to known PreTrainedTokenizerFast init kwargs.
                # Drop transformers-internal keys + any deeper nested config.
                _keep = {
                    "chat_template", "model_max_length", "padding_side",
                    "truncation_side", "clean_up_tokenization_spaces",
                    "split_special_tokens", "add_eos_token", "add_bos_token",
                    "bos_token", "eos_token", "unk_token", "pad_token",
                    "sep_token", "cls_token", "mask_token", "additional_special_tokens",
                }
                tok_kwargs = {k: v for k, v in tok_kwargs.items() if k in _keep}
            except Exception as _ee:
                warnings.warn(f"tokenizer_config.json parse failed: {_ee}")
                tok_kwargs = {}
        if eos_ids:
            tok_kwargs.setdefault("eos_token_id", eos_ids[0])
        # Special tokens may be dicts in tokenizer_config.json (HF AddedToken
        # serialization). PreTrainedTokenizerFast __init__ wants str/AddedToken;
        # convert dict-form by extracting `content`.
        from transformers import AddedToken
        for _k in ("bos_token", "eos_token", "unk_token", "pad_token",
                   "sep_token", "cls_token", "mask_token"):
            v = tok_kwargs.get(_k)
            if isinstance(v, dict):
                content = v.get("content")
                if content:
                    tok_kwargs[_k] = AddedToken(
                        content,
                        lstrip=v.get("lstrip", False),
                        rstrip=v.get("rstrip", False),
                        single_word=v.get("single_word", False),
                        normalized=v.get("normalized", False),
                        special=v.get("special", True),
                    )
                else:
                    tok_kwargs.pop(_k)
        # additional_special_tokens may be a list of dicts.
        ast = tok_kwargs.get("additional_special_tokens")
        if isinstance(ast, list):
            tok_kwargs["additional_special_tokens"] = [
                AddedToken(
                    item.get("content", ""),
                    lstrip=item.get("lstrip", False),
                    rstrip=item.get("rstrip", False),
                    single_word=item.get("single_word", False),
                    normalized=item.get("normalized", False),
                    special=item.get("special", True),
                ) if isinstance(item, dict) else item
                for item in ast
            ]
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_file, **tok_kwargs)

    # Pre-compile Metal kernels layer-by-layer so the FIRST call to
    # generate() doesn't pay 60+ seconds of cold JIT inside a single
    # command buffer (which trips Metal's watchdog on quantized
    # 191+ GB MoEs like Kimi K2.6). One-time ~20-30 s cost; subsequent
    # forwards hit cached shaders.
    try:
        from jang_tools.load_jangtq_kimi_vlm import _warmup_jit_per_layer
        _warmup_jit_per_layer(model)
    except Exception as _e:
        print(f"  [warmup] skipped ({_e!r})", flush=True)

    # Apply jang_config.json chat metadata to the tokenizer so downstream
    # generation works out-of-the-box (EOS stop, etc.). See
    # research/DSV-FAMILY-RUNTIME-GUIDE.md §26 for the full schema.
    chat_cfg = jang_cfg.get("chat", {}) if isinstance(jang_cfg, dict) else {}
    if chat_cfg:
        # Set EOS if tokenizer fallback didn't pick it up.
        cfg_eos_id = chat_cfg.get("eos_token_id")
        cfg_eos_tok = chat_cfg.get("eos_token")
        if cfg_eos_id is not None and getattr(tokenizer, "eos_token_id", None) is None:
            try:
                tokenizer.eos_token_id = cfg_eos_id
            except Exception:
                pass
        if cfg_eos_tok is not None and getattr(tokenizer, "eos_token", None) in (None, ""):
            try:
                tokenizer.eos_token = cfg_eos_tok
            except Exception:
                pass
        # Expose the chat metadata for caller convenience.
        try:
            tokenizer.jang_chat = chat_cfg
        except Exception:
            pass

    # Optional AWQ runtime scales: if the bundle ships
    # `awq_scales.safetensors`, wire each tensor onto its matching
    # TurboQuant{Linear,SwitchLinear} module as `awq_scale`. The runtime
    # path in `tq_kernel.py:222-224` divides x by `awq_scale` pre-matmul
    # so the model produces `(x / s) @ Q(W * s)` ≈ `x @ W` with outlier
    # input channels preserved at higher effective resolution.
    # Required precondition: the bundle was quantized AGAINST these same
    # scales (each routed-expert weight stored as `Q(W * s)`). Loose
    # bundles without pre-scaled weights MUST NOT contain the file.
    import os
    awq_path = os.path.join(str(model_path), "awq_scales.safetensors")
    if os.path.exists(awq_path):
        try:
            from safetensors.numpy import load_file as _safe_load
            scales_np = _safe_load(awq_path)
            from jang_tools.turboquant.tq_kernel import (
                TurboQuantLinear, TurboQuantSwitchLinear,
            )
            applied = 0
            missing = 0
            for name, mod in model.named_modules():
                if not isinstance(mod, (TurboQuantLinear, TurboQuantSwitchLinear)):
                    continue
                arr = scales_np.get(name)
                if arr is None:
                    missing += 1
                    continue
                mod.awq_scale = mx.array(arr)
                applied += 1
            print(f"  AWQ runtime scales: {applied} applied, "
                  f"{missing} TQ modules without scale entry", flush=True)
        except Exception as _e:
            print(f"  AWQ scale load skipped: {_e}", flush=True)

    print(f"  Done (eos_token_ids={eos_ids})", flush=True)
    return model, tokenizer


def _vlm_minimal_sanitize(model, weights):
    """VLM weight sanitize that skips the (already-applied) expert split.

    Mirrors the LLM-path mlx_lm.qwen3_5.sanitize behavior — same as what
    `load_jangtq_model` triggers via `model.sanitize(regular)` — but
    avoids mlx_vlm's qwen3_5_moe.sanitize which would also try to
    re-split routed experts (we already split them via TQ above) and
    unconditionally shift norm weights (mlx_lm only shifts conditionally
    on `has_unsanitized_conv1d`, and our artifact's conv1d shape `(out,
    1, kernel)` always trips that flag, so the shift IS needed).

    Steps applied:
      * drop mtp.* (speculative decode extras — unused at inference time)
      * drop lm_head.weight when tie_word_embeddings
      * rename `model.language_model.*` → `language_model.model.*`
      * rename `model.visual.*` → `vision_tower.*` (alternate HF layout)
      * rename `lm_head.*` → `language_model.lm_head.*`
      * conv1d.weight: moveaxis(2, 1) so shape (out, 1, k) → (out, k, 1)
      * RMSNorm.weight: `+= 1.0` (same shift mlx_lm applies for our
        artifact's pre-shift convention)
    """
    text_tied = False
    tc = getattr(model.config, "text_config", None)
    if tc is not None:
        text_tied = bool(getattr(tc, "tie_word_embeddings", False))

    norm_suffixes = (
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "model.norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    )

    out = {}
    for key, value in weights.items():
        if "mtp." in key:
            continue
        if text_tied and key == "lm_head.weight":
            continue
        if key.startswith("model.language_model"):
            key = key.replace("model.language_model", "language_model.model")
        elif key.startswith("model.visual"):
            key = key.replace("model.visual", "vision_tower")
        elif key.startswith("lm_head"):
            key = key.replace("lm_head", "language_model.lm_head")

        # GatedDeltaNet conv1d weights are stored on disk as
        # (out, 1, kernel) but mlx's `nn.Conv1d` expects (out, kernel, 1).
        # Same fix mlx_vlm + mlx_lm do in their sanitize passes.
        if "conv1d.weight" in key and value.ndim == 3 and value.shape[-1] != 1:
            value = value.moveaxis(2, 1)

        # RMSNorm `+= 1.0`: matches mlx_lm.qwen3_5.sanitize behavior when
        # `has_unsanitized_conv1d` (always true for our artifact). The
        # text-path verifies this is correct: norms saved at ~0.03 mean,
        # decoded after shift to ~1.03, produces coherent text at 96 tok/s.
        if value.ndim == 1 and any(key.endswith(sfx) for sfx in norm_suffixes):
            value = value + 1.0

        out[key] = value
    return out


def _hydrate_jangtq_model(model, model_path, mxtq_seed, mxtq_bits_map,
                          model_config, skip_params_eval=False):
    """Apply JANGTQ TQ replacement, weight load, and runtime patches in-place.

    Shared by `load_jangtq_model` (LLM-only) and `load_jangtq_vlm.load_jangtq_vlm_model`
    (VLM with vision_tower). Caller is responsible for building `model` and (for
    LLM path) the tokenizer/processor.
    """
    model_path = Path(model_path)
    weight_files = sorted(model_path.glob("model-*.safetensors"))
    print(f"  {len(weight_files)} shards", flush=True)

    # Collect TQ groups + regular weights
    tq_groups = {}  # base_path -> {packed, norms, bits}
    regular = {}
    for shard_i, sf_path in enumerate(weight_files):
        weights = mx.load(str(sf_path))
        for k, v in weights.items():
            if k.endswith(".tq_packed"):
                tq_groups.setdefault(k[:-10], {})["packed"] = v
            elif k.endswith(".tq_norms"):
                tq_groups.setdefault(k[:-9], {})["norms"] = v
            elif k.endswith(".tq_bits"):
                tq_groups.setdefault(k[:-8], {})["bits"] = int(v[0].item())
            else:
                regular[k] = v
        if (shard_i + 1) % 40 == 0:
            print(f"  shard {shard_i + 1}/{len(weight_files)}", flush=True)

    print(f"  TQ groups: {len(tq_groups)}, regular: {len(regular)}", flush=True)

    # Stack per-expert TQ tensors into 3D switch_mlp tensors.
    # Three naming conventions handled:
    #   GLM-5.1:       model.layers.L.mlp.experts.E.{gate_proj,up_proj,down_proj}
    #   MiniMax M2.7:  model.layers.L.block_sparse_moe.experts.E.{w1,w2,w3}
    #                  (w1=gate, w2=down, w3=up — Mixtral convention)
    #   Qwen3.5/3.6:   [model.language_model.]layers.L.mlp.experts.{gate_up_proj,down_proj}
    #                  ALREADY 3D-stacked ([n_experts, ...]); gate_up_proj is combined (split at mid).
    # All converge to: prefix + switch_mlp.{gate_proj,up_proj,down_proj}
    # Three prefix flavors appear across arches and sanitize passes:
    #   model.                         — standard MiniMax / GLM / Qwen3
    #   model.language_model.          — raw HF Qwen3.5/3.6 VLM
    #   language_model.model.          — post-sanitize Qwen3.5/3.6 VLM
    _VLM_PREFIX = r"(?:model\.language_model\.|language_model\.model\.|model\.)?"
    glm_pat = re.compile(rf"^({_VLM_PREFIX}layers\.\d+\.mlp\.)experts\.(\d+)\.(gate_proj|up_proj|down_proj)$")
    mm_pat  = re.compile(rf"^({_VLM_PREFIX}layers\.\d+\.block_sparse_moe\.)experts\.(\d+)\.(w[123])$")
    qw_pat  = re.compile(rf"^({_VLM_PREFIX}layers\.\d+\.mlp\.)experts\.(gate_up_proj|down_proj|gate_proj|up_proj)$")
    # DSV4: layers.N.ffn.experts.E.w[123] → map to switch_mlp.{gate_proj,down_proj,up_proj}
    dsv4_pat = re.compile(rf"^({_VLM_PREFIX}layers\.\d+\.ffn\.)experts\.(\d+)\.(w[123])$")
    # Nemotron-H (Cascade-2 + Nano-Omni): backbone.layers.N.mixer.experts.E.{up_proj|down_proj}
    # Routed-expert ReLU² activation has only up + down (no gate). mlx_lm
    # nemotron_h.sanitize stacks per-expert into switch_mlp.fc1/fc2 where
    # fc1 = up_proj, fc2 = down_proj. Per
    # research/NEMOTRON-OMNI-RUNTIME-2026-04-28.md §6 + trap #5.
    # NOTE: Nemotron uses `backbone.` prefix (NOT `model.` like other
    # families), so we use a dedicated literal-prefix regex here instead
    # of the generic _VLM_PREFIX.
    nemo_pat = re.compile(r"^(backbone\.layers\.\d+\.mixer\.)experts\.(\d+)\.(up_proj|down_proj)$")
    nemo_map = {"up_proj": "fc1", "down_proj": "fc2"}
    mm_map  = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
    grouped_experts = {}
    # Pre-stacked tensors go straight into tq_groups with switch_mlp naming (no stacking).
    prestacked = {}  # new_base -> dict(packed/norms/bits)

    for base in list(tq_groups.keys()):
        m = glm_pat.match(base)
        if m:
            layer_prefix = m.group(1)
            expert_id = int(m.group(2))
            proj_name = m.group(3)
            grouped_experts.setdefault((layer_prefix, proj_name), {})[expert_id] = tq_groups.pop(base)
            continue
        m = mm_pat.match(base)
        if m:
            layer_prefix = m.group(1)
            expert_id = int(m.group(2))
            proj_name = mm_map[m.group(3)]
            grouped_experts.setdefault((layer_prefix, proj_name), {})[expert_id] = tq_groups.pop(base)
            continue
        m = dsv4_pat.match(base)
        if m:
            # DSV4: layers.N.ffn.experts.E.w[123] — emit under layers.N.mlp.
            # prefix (matches our mlx_model sanitize convention).
            layer_prefix = m.group(1).replace(".ffn.", ".mlp.")
            expert_id = int(m.group(2))
            proj_name = mm_map[m.group(3)]
            grouped_experts.setdefault((layer_prefix, proj_name), {})[expert_id] = tq_groups.pop(base)
            continue
        m = nemo_pat.match(base)
        if m:
            # Nemotron-H: backbone.layers.N.mixer.experts.E.{up,down}_proj →
            # backbone.layers.N.mixer.switch_mlp.{fc1,fc2}.
            # CRITICAL: emit under .mixer., NOT .mlp. (NemotronH uses .mixer.).
            layer_prefix = m.group(1)
            expert_id = int(m.group(2))
            proj_name = nemo_map[m.group(3)]  # up_proj→fc1, down_proj→fc2
            grouped_experts.setdefault((layer_prefix, proj_name), {})[expert_id] = tq_groups.pop(base)
            continue
        m = qw_pat.match(base)
        if m:
            layer_prefix = m.group(1)
            proj_name = m.group(2)
            parts = tq_groups.pop(base)
            # Expect packed.ndim == 3 (pre-stacked [n_experts, out, packed_in]).
            # If converter already split gate_up → gate_proj + up_proj, proj_name
            # is gate_proj/up_proj — handle like any other pre-stacked projection.
            if proj_name == "gate_up_proj":
                # Split combined gate+up along the output-row axis.
                # packed: (n_exp, 2*inter_packed, in_packed_cols)
                # norms:  (n_exp, 2*inter_packed)
                packed = parts["packed"]
                norms = parts["norms"]
                bits = parts["bits"]
                # In MLX's qwen3_5_moe.sanitize: mid = gate_up.shape[-2] // 2
                mid = packed.shape[-2] // 2
                for half, name in ((slice(None, mid), "gate_proj"),
                                   (slice(mid, None), "up_proj")):
                    new_base = f"{layer_prefix}switch_mlp.{name}"
                    prestacked[new_base] = {
                        "packed": packed[..., half, :],
                        "norms": norms[..., half],
                        "bits": bits,
                    }
            else:
                new_base = f"{layer_prefix}switch_mlp.{proj_name}"
                prestacked[new_base] = parts
            continue

    print(f"  Expert groups to stack: {len(grouped_experts)}, pre-stacked: {len(prestacked)}", flush=True)

    for (layer_prefix, proj_name), experts in grouped_experts.items():
        n_exp = max(experts.keys()) + 1
        packed_list = [experts[e]["packed"] for e in range(n_exp)]
        norms_list = [experts[e]["norms"] for e in range(n_exp)]
        bits = experts[0]["bits"]
        stacked_packed = mx.stack(packed_list)
        stacked_norms = mx.stack(norms_list)
        new_base = f"{layer_prefix}switch_mlp.{proj_name}"
        tq_groups[new_base] = {
            "packed": stacked_packed,
            "norms": stacked_norms,
            "bits": bits,
        }
    del grouped_experts
    # Merge pre-stacked (Qwen3.5/3.6) groups straight in.
    for new_base, parts in prestacked.items():
        tq_groups[new_base] = parts
    del prestacked
    gc.collect()
    print(f"  After stacking: {len(tq_groups)} TQ groups", flush=True)

    # Replace modules with TurboQuant variants
    print("  Replacing modules with TurboQuantLinear...", flush=True)
    n_replaced = 0

    def get_module(root, dotted):
        cur = root
        for p in dotted.split("."):
            if p.isdigit():
                cur = cur[int(p)]
            else:
                cur = getattr(cur, p)
        return cur

    def set_module(root, dotted, new_mod):
        parts = dotted.split(".")
        cur = root
        for p in parts[:-1]:
            if p.isdigit():
                cur = cur[int(p)]
            else:
                cur = getattr(cur, p)
        last = parts[-1]
        if last.isdigit():
            cur[int(last)] = new_mod
        else:
            setattr(cur, last, new_mod)

    for base, parts in list(tq_groups.items()):
        packed = parts["packed"]
        norms = parts["norms"]
        bits = parts["bits"]
        vals_per_u32 = 32 // bits

        try:
            existing = get_module(model, base)
        except (AttributeError, IndexError, KeyError):
            print(f"    Skip (not in model): {base}", flush=True)
            continue

        if packed.ndim == 3:
            n_exp, out_feat, packed_cols = packed.shape
            in_features = packed_cols * vals_per_u32
            new_module = TurboQuantSwitchLinear(
                in_features=in_features, out_features=out_feat,
                num_experts=n_exp, bits=bits, bias=False, seed=mxtq_seed,
            )
        else:
            out_feat, packed_cols = packed.shape
            in_features = packed_cols * vals_per_u32
            new_module = TurboQuantLinear(
                in_features=in_features, out_features=out_feat,
                bits=bits, bias=False, seed=mxtq_seed,
            )
        new_module.packed = packed
        new_module.norms = norms

        set_module(model, base, new_module)
        n_replaced += 1

    print(f"  Replaced {n_replaced} modules", flush=True)
    del tq_groups
    gc.collect()

    # Load regular weights
    if hasattr(model, "sanitize"):
        # mlx_vlm's qwen3_5_moe.sanitize would re-split experts (already split
        # via TQ above) and unconditionally shift norm weights by +1.0 (already
        # baked into our converted artifact). Detect VLM models by the presence
        # of vision_tower and use a minimal sanitize that only handles key
        # renames + lm_head tie + mtp drops.
        is_vlm_model = hasattr(model, "vision_tower") and hasattr(model, "language_model")
        # Kimi K2.6 (kimi_k25 model_type -> routed to mlx_vlm.kimi_vl via
        # MODEL_REMAPPING in load_jangtq_kimi_vlm). Its on-disk projector is
        # named `mm_projector.{pre_norm, proj.0, proj.2}` (class PatchMergerMLP
        # in modeling_kimi_k25.py), but mlx_vlm's `KimiVLMultiModalProjector`
        # expects `multi_modal_projector.{pre_norm, linear_1, linear_2}`.
        # Architecturally identical — both are LN -> Linear(H, H) -> GELU ->
        # Linear(H, text_hidden). Just needs the Python attribute rename.
        model_type = getattr(model, "model_type", None) or (
            getattr(model.config, "model_type", None)
            if hasattr(model, "config") else None
        )
        is_kimi_vlm = is_vlm_model and model_type == "kimi_k25"
        if is_kimi_vlm:
            renamed = {}
            for k, v in regular.items():
                if k.startswith("mm_projector.pre_norm."):
                    nk = k.replace(
                        "mm_projector.pre_norm.",
                        "multi_modal_projector.pre_norm.", 1,
                    )
                elif k.startswith("mm_projector.proj.0."):
                    nk = k.replace(
                        "mm_projector.proj.0.",
                        "multi_modal_projector.linear_1.", 1,
                    )
                elif k.startswith("mm_projector.proj.2."):
                    nk = k.replace(
                        "mm_projector.proj.2.",
                        "multi_modal_projector.linear_2.", 1,
                    )
                else:
                    nk = k
                renamed[nk] = v
            regular = renamed
            # kimi_vl's Model.sanitize just strips `encoder.` from vision_tower
            # keys — safe to call directly. No expert re-split risk (Kimi's
            # on-disk experts are per-index, already handled by our TQ stack).
            regular = model.sanitize(regular)
        elif is_vlm_model:
            regular = _vlm_minimal_sanitize(model, regular)
        else:
            regular = model.sanitize(regular)
    model.load_weights(list(regular.items()), strict=False)
    del regular
    gc.collect()

    # P7 + P15: replace SwitchGLU.__call__ with a two-path version:
    #
    #   Fast path (decode, batch=1, K<64, not training):
    #     Uses mx.compile'd closure that bakes meta/grid/threadgroup as Python
    #     constants and runs: rotate → fused_gate_up_swiglu → rotate → gather_dn.
    #     Separate `gp.codebook` (keyed on in_features=hidden) and `dp.codebook`
    #     (keyed on in_features=inter) MUST be passed — codebooks differ by
    #     exactly sqrt(inter/hidden) so mixing them silently scales output.
    #
    #   Slow path (prefill, sorted routing, training):
    #     Calls the dynamic-shape-detection helpers
    #     `fused_gate_up_swiglu_matmul` + `gather_tq_matmul` directly.
    #
    # Class-level monkey-patch because Python looks up `__call__` on the type.
    try:
        from mlx_lm.models.switch_layers import SwitchGLU, _gather_sort, _scatter_unsort
        from jang_tools.turboquant.fused_gate_up_kernel import (
            fused_gate_up_matmul, fused_gate_up_swiglu_matmul,
            make_fused_gate_up_swiglu_decode,
        )
        from jang_tools.turboquant.gather_tq_kernel import (
            gather_tq_matmul, make_gather_tq_decode_per_row,
            make_fused_rot_gather_decode,
        )
        from jang_tools.turboquant.hadamard_kernel import hadamard_rotate_metal

        # P15: compiled decode helpers, cached per (in_f, out_f, bits, K).
        _DECODE_COMPILED = {}

        def _get_compiled_decode(in_f, out_f, bits, K):
            key = (in_f, out_f, bits, K)
            if key in _DECODE_COMPILED:
                return _DECODE_COMPILED[key]
            fused_gu = make_fused_gate_up_swiglu_decode(in_f, out_f, bits, K)
            # P19 (fused rot+gather) was tested but regressed 44.6 → 34.2 tok/s
            # in real decode despite +3 μs microbench win. Likely shmem/occupancy
            # interaction with other kernels. Reverted to split path.
            gather_dn = make_gather_tq_decode_per_row(out_f, in_f, bits, K)

            def _mlp(x_flat, pg, ng, pu, nu, pd, nd, cb_gate, cb_down, signs_in, signs_dn, idx):
                x_rot = hadamard_rotate_metal(x_flat, signs_in)
                x_act = fused_gu(x_rot, pg, ng, pu, nu, cb_gate, idx)  # (K, out_f)
                x_act_rot = hadamard_rotate_metal(x_act, signs_dn)
                y = gather_dn(x_act_rot, pd, nd, cb_down, idx)  # (K, in_f)
                return y

            _DECODE_COMPILED[key] = mx.compile(_mlp)
            return _DECODE_COMPILED[key]

        def _fused_switchglu_call(self, x, indices):
            # Fallback for non-TQ switch layers
            gp = self.gate_proj
            up = self.up_proj
            dp = self.down_proj
            if not isinstance(gp, TurboQuantSwitchLinear) or not isinstance(up, TurboQuantSwitchLinear):
                return _ORIG_SWITCHGLU_CALL(self, x, indices)

            # Decode fast path: batch=1, K=topk, broadcast mode.
            # Detect by: x has flattenable-to-(1, in_f) layout AND indices is 1D-equivalent.
            x_sq = x
            while x_sq.ndim > 2 and x_sq.shape[-2] == 1:
                x_sq = x_sq.squeeze(-2)
            x_flat = x_sq.reshape(-1, gp.in_features)
            batch = x_flat.shape[0]
            K = indices.shape[-1] if indices.ndim > 0 else 1
            do_sort_ok = indices.ndim >= 1 and indices.size < 64
            can_fast = (batch == 1 and K > 0 and do_sort_ok and not getattr(self, "training", False))

            if can_fast:
                idx_flat = indices.reshape(-1).astype(mx.uint32)
                compiled_mlp = _get_compiled_decode(gp.in_features, gp.out_features, gp.bits, K)
                y = compiled_mlp(
                    x_flat.astype(mx.float32),
                    gp.packed, gp.norms, up.packed, up.norms,
                    dp.packed, dp.norms,
                    gp.codebook, dp.codebook,
                    gp.signs, dp.signs, idx_flat,
                )  # (K, in_f)
                out = y.reshape(*indices.shape[:-1], K, 1, gp.in_features)
                if out.dtype != x.dtype:
                    out = out.astype(x.dtype)
                return out.squeeze(-2)

            # Slow path: original fused call with dynamic shape detection.
            x_exp = mx.expand_dims(x, (-2, -3))
            do_sort = indices.size >= 64
            idx = indices
            inv_order = None
            if do_sort:
                x_exp, idx, inv_order = _gather_sort(x_exp, indices)
            if getattr(self, "training", False):
                idx = mx.stop_gradient(idx)

            x_act = fused_gate_up_swiglu_matmul(
                x_exp,
                gp.packed, gp.norms,
                up.packed, up.norms,
                gp.codebook, gp.signs,
                idx,
                bits=gp.bits,
            )
            x_out = self.down_proj(x_act, idx, sorted_indices=do_sort)
            if do_sort:
                x_out = _scatter_unsort(x_out, inv_order, indices.shape)
            return x_out.squeeze(-2)

        _ORIG_SWITCHGLU_CALL = SwitchGLU.__call__
        SwitchGLU.__call__ = _fused_switchglu_call

        # Count how many instances will benefit
        patched = sum(
            1 for name, m in model.named_modules()
            if isinstance(m, SwitchGLU)
            and isinstance(getattr(m, "gate_proj", None), TurboQuantSwitchLinear)
            and isinstance(getattr(m, "up_proj", None), TurboQuantSwitchLinear)
        )
        print(f"  Patched SwitchGLU class for fused gate+up ({patched} TQ instances)", flush=True)
    except Exception as _e:
        print(f"  SwitchGLU fusion skipped: {_e}", flush=True)

    # P15: mx.compile ONLY pure router math (mlx_lm pattern — free function,
    # no closures over self, shapeless so prefill/decode share one graph).
    # Safer than patching __call__ which thrashes the compile cache.
    try:
        # Cache compiled variants keyed on (k, bits).
        _ROUTER_CACHE = {}
        _MLP_CACHE = {}

        # Variant A: sigmoid + e_score_correction_bias topk (MiniMax, GLM, DeepSeek V3).
        def _get_compiled_router_sigmoid_bias(k):
            key = ("sigbias", k)
            if key in _ROUTER_CACHE:
                return _ROUTER_CACHE[key]
            def _router(gates_f32, e_bias):
                scores = mx.sigmoid(gates_f32)
                orig = scores
                scores = scores + e_bias
                inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
                sel = mx.take_along_axis(orig, inds, axis=-1)
                sel = sel / (mx.sum(sel, axis=-1, keepdims=True) + 1e-20)
                return inds, sel
            _ROUTER_CACHE[key] = mx.compile(_router)
            return _ROUTER_CACHE[key]

        # Variant B: softmax → topk → optional renormalize (Qwen3-Next, Qwen3.5/3.6).
        # Matches mlx_lm.models.qwen3_next.Qwen3NextSparseMoeBlock.__call__ semantics.
        def _get_compiled_router_softmax(k, renorm):
            key = ("softmax", k, bool(renorm))
            if key in _ROUTER_CACHE:
                return _ROUTER_CACHE[key]
            if renorm:
                def _router(gates_f32):
                    scores = mx.softmax(gates_f32, axis=-1, precise=True)
                    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
                    sel = mx.take_along_axis(scores, inds, axis=-1)
                    sel = sel / (mx.sum(sel, axis=-1, keepdims=True) + 1e-20)
                    return inds, sel
            else:
                def _router(gates_f32):
                    scores = mx.softmax(gates_f32, axis=-1, precise=True)
                    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
                    sel = mx.take_along_axis(scores, inds, axis=-1)
                    return inds, sel
            _ROUTER_CACHE[key] = mx.compile(_router)
            return _ROUTER_CACHE[key]

        # Back-compat name (kept so any external references don't break).
        _get_compiled_router = _get_compiled_router_sigmoid_bias

        # Compiling the MLP path regressed 35.5 → 30 tok/s AND changed output
        # length (likely recompile thrash — metal_kernel internal mx.array alloc
        # for `meta` invalidates the trace every call). Router-only is safe.
        #
        # Unified patch handles three arch families via hasattr probes:
        #   - `e_score_correction_bias` present → sigmoid-bias router (MiniMax/GLM/DSV3)
        #   - `e_score_correction_bias` absent  → softmax router (Qwen3-Next/3.5/3.6)
        #   - `shared_expert` + `shared_expert_gate` present → add gated shared output
        _n_patched = 0
        _patched_classes = set()
        for _, _mod in model.named_modules():
            _cls = _mod.__class__
            if _cls in _patched_classes or "SparseMoeBlock" not in _cls.__name__:
                continue

            def _patched_moe(self, x):
                gates = self.gate(x.astype(mx.float32))
                # MiniMax stores K as `num_experts_per_tok`, Qwen3-Next as `top_k`,
                # some GLM variants as `num_routed_experts`. Probe in order.
                k = getattr(self, "num_experts_per_tok",
                            getattr(self, "top_k",
                                    getattr(self, "num_routed_experts", None)))
                if k is None:
                    raise AttributeError(
                        f"{type(self).__name__} missing one of "
                        "num_experts_per_tok / top_k / num_routed_experts"
                    )
                if hasattr(self, "e_score_correction_bias"):
                    inds, scores = _get_compiled_router_sigmoid_bias(k)(
                        gates, self.e_score_correction_bias
                    )
                else:
                    renorm = getattr(self, "norm_topk_prob", True)
                    inds, scores = _get_compiled_router_softmax(k, renorm)(gates)
                scores = scores.astype(x.dtype)
                y = self.switch_mlp(x, inds)
                y = (y * scores[..., None]).sum(axis=-2)
                # Shared expert (always active, gated): Qwen3-Next/3.5/3.6; some GLM configs.
                shared = getattr(self, "shared_expert", None)
                if shared is not None:
                    sh_out = shared(x)
                    gate_lin = getattr(self, "shared_expert_gate", None)
                    if gate_lin is not None:
                        sh_out = sh_out * mx.sigmoid(gate_lin(x))
                    y = y + sh_out
                return y

            _cls.__call__ = _patched_moe
            _patched_classes.add(_cls)
            _n_patched += 1
        print(f"  P15 mx.compile(router-only) applied to {_n_patched} MoE class(es)", flush=True)
    except Exception as _e:
        print(f"  P15 skipped: {_e}", flush=True)

    # Fix mixed bit widths in remaining QuantizedLinear/QuantizedEmbedding
    # (embed_tokens, lm_head, attention may be at different bits than config default)
    # Self-contained — uses jang_tools.loader's bit-width fixer. No vMLX dependency.
    from jang_tools.loader import _fix_quantized_bits
    _fix_quantized_bits(model, {})

    # MLA models (GLM-5.1 glm_moe_dsa, Mistral-4 MLA, etc.) have an extra
    # `QuantizedMultiLinear` class (mlx_lm.models.mla) for the embed_q/unembed_out
    # absorbed projections. These are NOT handled by _fix_quantized_bits (which
    # only knows QuantizedLinear/Embedding/SwitchLinear), so their `.bits` stays
    # at the skeleton default while the actual packed weights are 8-bit → shape
    # mismatch on first forward. Fix by walking the model and inferring bits
    # from weight/scales shapes, same as _fix_quantized_bits does for 2D tensors.
    try:
        from mlx_lm.models.mla import QuantizedMultiLinear
        _mla_fixed = 0
        for _name, _mod in model.named_modules():
            if not isinstance(_mod, QuantizedMultiLinear):
                continue
            if not (hasattr(_mod, "weight") and hasattr(_mod, "scales")):
                continue
            # weight: (num_heads, out, in_packed), scales: (num_heads, out, in/gs)
            w_cols = _mod.weight.shape[-1]
            s_cols = _mod.scales.shape[-1]
            for try_gs in (_mod.group_size, 64, 128):
                in_dim = s_cols * try_gs
                if in_dim <= 0:
                    continue
                if (w_cols * 32) % in_dim != 0:
                    continue
                try_bits = (w_cols * 32) // in_dim
                if try_bits in (2, 3, 4, 5, 6, 8):
                    if try_bits != _mod.bits:
                        _mod.bits = try_bits
                    if try_gs != _mod.group_size:
                        _mod.group_size = try_gs
                    _mla_fixed += 1
                    break
        if _mla_fixed:
            print(f"  MLA QuantizedMultiLinear bits fixed: {_mla_fixed} modules", flush=True)
    except ImportError:
        pass  # mlx_lm version without mla.py — MLA models unsupported anyway

    # P18: QKV fusion — must run AFTER _fix_quantized_bits so the attention
    # QuantizedLinear modules report the correct bits (8 instead of default 2).
    # Concat q/k/v weights into one matmul: 3 dispatches → 1 per layer.
    try:
        import mlx.nn as _nn
        _qkv_fused = {}  # id(attn_instance) -> (fused_linear, Hq, Hk, Hv)
        _patched_attn_classes = set()
        for _, _mod in model.named_modules():
            if not (hasattr(_mod, "q_proj") and hasattr(_mod, "k_proj") and hasattr(_mod, "v_proj")):
                continue
            # P18 skip: NemotronHAttention (Cascade-2 + Nano-Omni hybrid).
            # Its attention class uses `num_heads`/`n_kv_groups` instead of
            # `num_attention_heads`/`num_key_value_heads` AND has no RoPE
            # (position info comes from Mamba state). The fused path below
            # crashes with `AttributeError: 'NemotronHAttention' object has
            # no attribute 'num_attention_heads'`. Per
            # research/NEMOTRON-OMNI-RUNTIME-2026-04-28.md §6 trap #4.
            # Affine 8-bit attention without fusion is fine — only 6 attn
            # layers in the 52-layer hybrid pattern.
            if _mod.__class__.__name__ == "NemotronHAttention":
                continue
            q, k, v = _mod.q_proj, _mod.k_proj, _mod.v_proj
            if not all(isinstance(p, _nn.QuantizedLinear) for p in (q, k, v)):
                continue
            if not (q.bits == k.bits == v.bits and q.group_size == k.group_size == v.group_size):
                continue
            # Skip P18 fusion when q_proj is doubled for the
            # attn_output_gate split (Qwen 3.5/3.6). The original
            # __call__ knows to split q into (queries | gate); the
            # patched fused path here doesn't, and would feed an
            # over-wide query into SDPA. Detection: q.weight rows are
            # 2x the standard Hq when num_heads * head_dim * 2 matches.
            num_heads = getattr(_mod, "num_attention_heads", None) \
                or getattr(_mod, "n_heads", None) or 0
            head_dim = getattr(_mod, "head_dim", 0)
            if num_heads and head_dim and q.weight.shape[0] == num_heads * head_dim * 2:
                continue
            Hq, Hk, Hv = q.weight.shape[0], k.weight.shape[0], v.weight.shape[0]
            in_f = q.weight.shape[1] * (32 // q.bits)
            fused = _nn.QuantizedLinear(
                input_dims=in_f, output_dims=Hq + Hk + Hv,
                bias=False, group_size=q.group_size, bits=q.bits,
            )
            fused.weight = mx.concatenate([q.weight, k.weight, v.weight], axis=0)
            fused.scales = mx.concatenate([q.scales, k.scales, v.scales], axis=0)
            fused.biases = mx.concatenate([q.biases, k.biases, v.biases], axis=0)
            _qkv_fused[id(_mod)] = (fused, Hq, Hk, Hv)
            _patched_attn_classes.add(_mod.__class__)

        for _attn_cls in _patched_attn_classes:
            _orig_attn_call = _attn_cls.__call__
            def _make_patched(orig_call):
                def _patched(self, x, mask=None, cache=None):
                    info = _qkv_fused.get(id(self))
                    if info is None:
                        return orig_call(self, x, mask=mask, cache=cache)
                    fused, Hq, Hk, Hv = info
                    B, L, _ = x.shape
                    qkv = fused(x)
                    queries = qkv[..., :Hq]
                    keys = qkv[..., Hq:Hq + Hk]
                    values = qkv[..., Hq + Hk:]
                    if getattr(self, "use_qk_norm", False):
                        queries = self.q_norm(queries)
                        keys = self.k_norm(keys)
                    queries = queries.reshape(B, L, self.num_attention_heads, -1).transpose(0, 2, 1, 3)
                    keys = keys.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
                    values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
                    if cache is not None:
                        queries = self.rope(queries, offset=cache.offset)
                        keys = self.rope(keys, offset=cache.offset)
                        keys, values = cache.update_and_fetch(keys, values)
                    else:
                        queries = self.rope(queries); keys = self.rope(keys)
                    from mlx_lm.models.base import scaled_dot_product_attention
                    out = scaled_dot_product_attention(
                        queries, keys, values, cache=cache, scale=self.scale, mask=mask,
                    )
                    out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
                    return self.o_proj(out)
                return _patched
            _attn_cls.__call__ = _make_patched(_orig_attn_call)
        print(f"  P18 QKV fusion: {len(_patched_attn_classes)} class(es), {len(_qkv_fused)} instances", flush=True)
    except Exception as _e:
        print(f"  P18 QKV fusion skipped: {_e}", flush=True)

    # P19: DSV4-MLA-specific wq_a + wkv fusion. P18 expects q_proj/k_proj/v_proj
    # naming which DSV4 doesn't have (uses MLA's wq_a/wq_b/wkv low-rank LoRA
    # decomposition). DSV4 wq_a (4096→1024) and wkv (4096→512) BOTH take x as
    # input → can fuse into single quantized matmul (4096→1536). Saves 1
    # dispatch/layer × 43 layers = 43 dispatches/token. Profiler showed
    # attention is 45% of layer time (2026-04-24).
    try:
        import mlx.nn as _nn
        _wq_a_kv_fused_count = 0
        for _name, _mod in model.named_modules():
            if not (hasattr(_mod, "wq_a") and hasattr(_mod, "wkv")):
                continue
            qa, kv = _mod.wq_a, _mod.wkv
            if not (isinstance(qa, _nn.QuantizedLinear) and isinstance(kv, _nn.QuantizedLinear)):
                continue
            if not (qa.bits == kv.bits and qa.group_size == kv.group_size):
                continue
            in_f = qa.weight.shape[1] * (32 // qa.bits)
            out_f = qa.weight.shape[0] + kv.weight.shape[0]
            fused = _nn.QuantizedLinear(
                input_dims=in_f, output_dims=out_f,
                bias=False, group_size=qa.group_size, bits=qa.bits,
            )
            fused.weight = mx.concatenate([qa.weight, kv.weight], axis=0)
            fused.scales = mx.concatenate([qa.scales, kv.scales], axis=0)
            if hasattr(qa, "biases") and qa.biases is not None and \
               hasattr(kv, "biases") and kv.biases is not None:
                fused.biases = mx.concatenate([qa.biases, kv.biases], axis=0)
            _mod._wq_a_kv_fused = fused
            _wq_a_kv_fused_count += 1
        print(f"  P19 MLA wq_a+wkv fusion: {_wq_a_kv_fused_count} instances", flush=True)
    except Exception as _e:
        print(f"  P19 MLA wq_a+wkv fusion skipped: {_e}", flush=True)

    # bfloat16 for MLA models. model_config can be a dict (mlx_lm path) or
    # a dataclass (mlx_vlm path); normalize via getattr/getitem fallback.
    def _cfg_get(cfg, key, default=None):
        if hasattr(cfg, key):
            return getattr(cfg, key)
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return default
    tc = _cfg_get(model_config, "text_config", model_config)
    if _cfg_get(tc, "kv_lora_rank", 0) > 0 or _cfg_get(tc, "model_type", "") == "glm_moe_dsa":
        model.set_dtype(mx.bfloat16)
        print("  bfloat16 enabled", flush=True)

    if not skip_params_eval:
        mx.synchronize()
    print("  Hydration complete", flush=True)
