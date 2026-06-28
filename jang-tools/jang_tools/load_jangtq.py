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

See the JANGTQ section of the project docs for full architecture, math,
kernel inventory, and known traps.
"""
import json, gc, os, re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from jang_tools.ssm_layout import sanitize_grouped_conv1d_layout
from jang_tools.turboquant.tq_kernel import TurboQuantLinear, TurboQuantSwitchLinear


def _sanitize_grouped_conv1d_layout(weights: dict) -> dict:
    """Idempotently normalize grouped Conv1d weights after model sanitize."""
    return sanitize_grouped_conv1d_layout(weights, lambda v: v.moveaxis(2, 1))


def _infer_tq_input_features(existing):
    """Return the logical input width for a module replaced by TurboQuant.

    MLX affine-quantized SwitchLinear placeholders store packed uint32 columns
    in ``weight.shape[-1]``.  The logical input width lives in ``input_dims`` on
    those modules.  Reading the packed storage width directly builds a
    TurboQuant module with the wrong Hadamard/codebook cache key and corrupts
    routed-expert decode.
    """
    for attr in ("in_features", "input_dims", "input_features"):
        value = getattr(existing, attr, None)
        if value is not None:
            return int(value)

    weight = getattr(existing, "weight", None)
    if weight is None:
        return None

    packed_cols = int(weight.shape[-1])
    bits = getattr(existing, "bits", None)
    if bits is not None:
        try:
            return packed_cols * (32 // int(bits))
        except (TypeError, ValueError, ZeroDivisionError):
            # Non-int bits or zero — fall through to the unpacked column count.
            pass
    return packed_cols


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
        import psutil, sys, mlx.core as _mx
        if sys.platform != "darwin":
            return
        total_gb = psutil.virtual_memory().total / 1e9
        env_target = os.environ.get("JANGTQ_WIRED_LIMIT_GB")
        if env_target:
            target_gb = int(float(env_target))
        else:
            target_gb = int(total_gb * 0.70)
        # Clamp to reasonable range: at least 32 GB, at most 220 GB
        # (leaving enough headroom even on very-large-RAM machines).
        target_gb = max(32, min(target_gb, 220))
        target_bytes = target_gb * 1000 * 1000 * 1000
        _mx.set_wired_limit(target_bytes)
        src = "env JANGTQ_WIRED_LIMIT_GB" if env_target else "ralph iter-14 tuning"
        print(f"  [wired_limit] auto-set to {target_gb} GB "
              f"(~{target_gb / total_gb:.0%} of {total_gb:.0f} GB total RAM; {src})",
              flush=True)
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
    elif _model_type == "hy_v3":
        try:
            import jang_tools.hy3  # noqa: F401  (registers on import)
        except Exception as _e:
            warnings.warn(f"jang_tools.hy3 register failed: {_e}")
    elif _model_type == "zaya":
        try:
            import jang_tools.zaya  # noqa: F401  (registers on import)
        except Exception as _e:
            warnings.warn(f"jang_tools.zaya register failed: {_e}")

    if "quantization" not in model_config:
        model_config["quantization"] = {"group_size": 64, "bits": 2}

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
    if eos_ids is None:
        eos_ids = []
    # Merge generation_config.json eos_token_id into the list. Fixes two
    # bug classes observed live 2026-05-04:
    #   (a) config.json eos_token_id is null but generation_config.json
    #       has the real EOS — engine registered empty eos list and the
    #       model emitted the EOS token as visible content (MiniMax:
    #       200020 = "[e~[" loops endlessly after the first answer).
    #   (b) config.json eos_token_id is a single int but generation_config
    #       lists multiple — Ling-2.6 ships [156892, 156895] in
    #       generation_config but only 156895 in config; the secondary
    #       turn-boundary stop is silently dropped.
    # Reading generation_config.json as a strict superset rather than a
    # fallback handles both cases.
    try:
        from pathlib import Path as _P
        import json as _json
        _gc_path = _P(model_path) / "generation_config.json"
        if _gc_path.is_file():
            _gc = _json.loads(_gc_path.read_text())
            _gc_eos = _gc.get("eos_token_id")
            _added = []
            if isinstance(_gc_eos, int):
                if _gc_eos not in eos_ids:
                    eos_ids = list(eos_ids) + [_gc_eos]
                    _added.append(_gc_eos)
            elif isinstance(_gc_eos, list):
                for x in _gc_eos:
                    if isinstance(x, int) and x not in eos_ids:
                        eos_ids = list(eos_ids) + [int(x)]
                        _added.append(int(x))
            if _added:
                print(
                    f"  [load_jangtq] merged eos_token_id from "
                    f"generation_config.json: +{_added} → {eos_ids}",
                    flush=True,
                )
    except Exception as _gc_err:
        warnings.warn(
            f"generation_config.json eos merge failed: {_gc_err}"
        )
    # DSV4-Flash needs <｜User｜> as a turn-boundary stop in addition to
    # <｜end▁of▁sentence｜>. Upstream DSV4 + early jang bundles ship eos as
    # single int, leaving the model free to auto-continue into a fake user
    # turn after EOS (manifests as "🤖 My name is..." restart loops on
    # /v1/chat/completions). Add it dynamically from the tokenizer so this
    # works even on bundles whose configs were never patched.
    if _model_type == "deepseek_v4":
        try:
            from transformers import AutoTokenizer
            _tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
            _user_id = _tok.convert_tokens_to_ids("<｜User｜>")
            if _user_id is not None and _user_id not in eos_ids:
                eos_ids = list(eos_ids) + [_user_id]
                print(f"  [load_jangtq] DSV4: added <｜User｜>={_user_id} to eos list", flush=True)
        except Exception as _e:
            warnings.warn(f"DSV4 eos list expansion failed: {_e}")
    try:
        tokenizer = load_tokenizer(model_path, eos_token_ids=eos_ids)
    except Exception as _e:
        # Newer model_types (e.g. deepseek_v4) may trip transformers' config
        # validation. Fall back to direct PreTrainedTokenizerFast from tokenizer.json.
        import warnings
        warnings.warn(f"load_tokenizer failed ({_e}); using PreTrainedTokenizerFast fallback")
        from transformers import PreTrainedTokenizerFast
        import os
        tok_file = os.path.join(str(model_path), "tokenizer.json")
        # PreTrainedTokenizerFast accepts a single int for eos_token_id. We
        # store the FULL list on the tokenizer object so downstream consumers
        # (mlx_lm TokenizerWrapper, vmlx BatchedEngine pre-wrap) can pick up
        # all turn-boundary stops, not just <｜end▁of▁sentence｜>. Without this
        # DSV4 multi-turn loops past EOS into a fake user/assistant turn.
        _primary_eos = eos_ids[0] if eos_ids else None
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_file, eos_token_id=_primary_eos)
        if eos_ids:
            try:
                tokenizer.eos_token_ids = list(eos_ids)
            except (AttributeError, TypeError):
                tokenizer._jang_eos_token_ids = list(eos_ids)

    # Pre-compile Metal kernels + materialize TQ weights so the FIRST
    # /v1/chat/completions request doesn't pay full kernel-JIT + weight-
    # mmap-realize cost on the user's first message. Without this, the
    # session "starts" but weights remain lazy until a real forward fires
    # — visible to users as a 30+ s stall on first prompt.
    #
    # Two-tier approach:
    #
    # 1. `_warmup_jit_per_layer` (Kimi-style layer-by-layer) — fastest path
    #    when the model's DecoderLayer accepts (x, mask=None, cache=None)
    #    or (x, None, None). Caches per-layer shaders one at a time so
    #    each command buffer is small (avoids Metal watchdog on 100+ layer
    #    191 GB MoEs). Works for: Kimi K2.6, Qwen3-VL, Qwen3.5-MoE, GLM-5.1.
    #
    # 2. Full-model 1-token forward — fallback for architectures whose
    #    DecoderLayer signature doesn't match (DSV4 MLA: q_a_proj output
    #    feeds back through compressed kv_lora_rank, so a per-layer
    #    1-token forward with H-dim input mismatches the packed weight
    #    column count and matmul raises ValueError). The full-model path
    #    uses `model.make_cache()` + `model(tiny_ids, cache=cache)` which
    #    is what real inference does, so every kernel shape that prefill
    #    will need is JIT-cached on the loader thread before return.
    _warmed = False
    try:
        from jang_tools.load_jangtq_kimi_vlm import _warmup_jit_per_layer
        _warmup_jit_per_layer(model)
        _warmed = True
    except Exception as _e:
        print(f"  [warmup] per-layer skipped ({type(_e).__name__}: {_e}); "
              f"trying full-model 1-token forward", flush=True)

    if not _warmed:
        try:
            import time as _time
            _t0 = _time.time()
            # Some models expose `make_cache()`; others use `make_prompt_cache(model)`.
            _cache = None
            if hasattr(model, "make_cache"):
                _cache = model.make_cache()
            else:
                try:
                    from mlx_lm.models.cache import make_prompt_cache as _mpc
                    _cache = _mpc(model)
                except Exception:
                    _cache = None
            _tiny_ids = mx.array([[0]], dtype=mx.int32)
            try:
                if _cache is not None:
                    _ = model(_tiny_ids, cache=_cache)
                else:
                    _ = model(_tiny_ids)
            except TypeError:
                # Some VLM wrappers want `inputs=` kwarg.
                _ = model(inputs=_tiny_ids, cache=_cache) if _cache is not None else model(inputs=_tiny_ids)
            mx.synchronize()
            print(f"  [warmup] full-model 1-token forward done "
                  f"({_time.time() - _t0:.1f}s)", flush=True)
            _warmed = True
        except Exception as _fe:
            print(f"  [warmup] full-model fallback ALSO skipped "
                  f"({type(_fe).__name__}: {_fe}). First request will be slow.",
                  flush=True)

    # Apply jang_config.json chat metadata to the tokenizer so downstream
    # generation works out-of-the-box (EOS stop, etc.). See
    # See the DSV-family runtime guide for the full schema.
    chat_cfg = jang_cfg.get("chat", {}) if isinstance(jang_cfg, dict) else {}
    if chat_cfg:
        # Set EOS if tokenizer fallback didn't pick it up.
        cfg_eos_id = chat_cfg.get("eos_token_id")
        cfg_eos_tok = chat_cfg.get("eos_token")
        if cfg_eos_id is not None and getattr(tokenizer, "eos_token_id", None) is None:
            try:
                tokenizer.eos_token_id = cfg_eos_id
            except (AttributeError, TypeError):
                tokenizer._jang_eos_token_id = cfg_eos_id
        if cfg_eos_tok is not None and getattr(tokenizer, "eos_token", None) in (None, ""):
            try:
                tokenizer.eos_token = cfg_eos_tok
            except (AttributeError, TypeError):
                tokenizer._jang_eos_token = cfg_eos_tok
        # Expose the chat metadata for caller convenience.
        try:
            tokenizer.jang_chat = chat_cfg
        except (AttributeError, TypeError):
            tokenizer._jang_chat = chat_cfg

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

    model_type = (
        getattr(model, "model_type", None)
        or getattr(getattr(model, "config", None), "model_type", None)
        or getattr(tc, "model_type", None)
    )
    is_zaya1_vl = str(model_type or "").lower() == "zaya1_vl"

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
        if is_zaya1_vl:
            # ZAYA1-VL bundles keep language weights in raw HF text layout
            # (`model.layers.*`) while the local mlx-vlm adapter exposes them
            # under `language_model.model.layers.*`.  `model.load_weights` is
            # strict=False below, so leaving these raw keys in place silently
            # drops attention/router/LoRA weights and produces incoherent text
            # even before any image path is involved.
            if key.startswith("model.visual"):
                key = key.replace("model.visual", "vision_tower", 1)
            elif key.startswith("model.vision_tower"):
                key = key.replace("model.vision_tower", "vision_tower", 1)
            elif key.startswith("model."):
                key = key.replace("model.", "language_model.model.", 1)
            elif key.startswith("lm_head"):
                key = key.replace("lm_head", "language_model.lm_head", 1)

            if (
                ".layers." in key
                and ".zaya_block." in key
                and ".mlp.zaya_block." not in key
            ):
                key = key.replace(".zaya_block.", ".mlp.zaya_block.", 1)

            if (
                ".self_attn.qkv.conv_qk." in key
                and key.endswith(".weight")
                and value.ndim == 3
            ):
                value = value.swapaxes(-1, -2)
            if (
                key.endswith("patch_embed.proj.weight")
                and value.ndim == 5
                and value.shape[1] in (1, 3)
                and value.shape[-1] not in (1, 3)
            ):
                value = value.transpose(0, 2, 3, 4, 1)
        elif key.startswith("model.language_model"):
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


def _hydrate_dsv4_jangtq_streaming(
    model, model_path, mxtq_seed, skip_params_eval=False
):
    """Streaming DSV4 JANGTQ hydration.

    The generic JANGTQ loader first keeps every shard tensor in Python dicts,
    then stacks every routed expert projection while the unstacked per-expert
    tensors are still live. DSV4 has 43 layers * 256 experts * 3 projections,
    so that peak can exceed 128 GB even though the final JANGTQ2 model is only
    ~74 GB. This path scans safetensors headers first, stacks one
    layer/projection at a time, installs the TurboQuant module immediately,
    then streams regular weights per shard.
    """
    from safetensors import safe_open
    import mlx.core as mx

    weight_files = sorted(model_path.glob("model-*.safetensors"))

    # JANGTQ-PRESTACK STANDARD detection: if the bundle ships routed-expert
    # tensors pre-stacked under `{prefix}.{ffn|mlp|block_sparse_moe}.switch_mlp.{proj}`,
    # the streaming-restack hot path is unnecessary — defer to the generic
    # loader (which now has a prestack_pat branch and creates
    # TurboQuantSwitchLinear directly from 3D tensors). This avoids the
    # 65 GB pre-stacked sidecar entirely on freshly converted DSV4 bundles.
    _is_prestacked = False
    for sf_path in weight_files[:1]:
        try:
            with safe_open(str(sf_path), framework="numpy") as sf:
                for key in sf.keys():
                    if (
                        ".switch_mlp.gate_proj.tq_packed" in key
                        or ".switch_mlp.up_proj.tq_packed" in key
                        or ".switch_mlp.down_proj.tq_packed" in key
                        or ".switch_mlp.gate_up_proj.tq_packed" in key
                    ):
                        _is_prestacked = True
                        break
        except Exception as _e:
            print(f"  [prestack] probe skipped for {sf_path.name}: {_e}", flush=True)
        if _is_prestacked:
            break
    if _is_prestacked:
        print(
            "  DSV4 bundle is JANGTQ-PRESTACK format — deferring to generic "
            "loader (skipping streaming restack + sidecar write)",
            flush=True,
        )
        return _hydrate_jangtq_model(
            model=model,
            model_path=model_path,
            mxtq_seed=mxtq_seed,
            mxtq_bits_map=None,  # generic loader infers from per-tensor tq_bits
            model_config=getattr(model, "config", None),
            skip_params_eval=skip_params_eval,
        )

    dsv4_pat = re.compile(r"^(layers\.\d+\.ffn\.)experts\.(\d+)\.(w[123])$")
    mm_map = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
    grouped = {}
    regular_keys_by_shard = {}

    print("  DSV4 streaming hydrate: scanning safetensors headers", flush=True)
    for sf_path in weight_files:
        regular_keys = []
        with safe_open(str(sf_path), framework="numpy") as sf:
            keys = list(sf.keys())
        for key in keys:
            if key.endswith(".tq_packed"):
                base = key[:-10]
                part = "packed"
            elif key.endswith(".tq_norms"):
                base = key[:-9]
                part = "norms"
            elif key.endswith(".tq_bits"):
                base = key[:-8]
                part = "bits"
            else:
                regular_keys.append(key)
                continue

            m = dsv4_pat.match(base)
            if not m:
                if base.startswith("mtp."):
                    # DSV4 inference does not instantiate/use MTP layers; the
                    # model.sanitize() path drops mtp.* regular weights too.
                    continue
                raise RuntimeError(
                    "DSV4 streaming hydrate saw non-routed TQ key "
                    f"{base!r}; generic loader path must handle this bundle"
                )
            layer_prefix = m.group(1).replace(".ffn.", ".mlp.")
            expert_id = int(m.group(2))
            proj_name = mm_map[m.group(3)]
            group_key = (layer_prefix, proj_name)
            grouped.setdefault(group_key, {}).setdefault(expert_id, {})[part] = (
                sf_path,
                key,
            )
        if regular_keys:
            regular_keys_by_shard[sf_path] = set(regular_keys)

    def get_module(root, dotted):
        cur = root
        for p in dotted.split("."):
            cur = cur[int(p)] if p.isdigit() else getattr(cur, p)
        return cur

    def set_module(root, dotted, new_mod):
        parts = dotted.split(".")
        cur = root
        for p in parts[:-1]:
            cur = cur[int(p)] if p.isdigit() else getattr(cur, p)
        last = parts[-1]
        if last.isdigit():
            cur[int(last)] = new_mod
        else:
            setattr(cur, last, new_mod)

    print(f"  DSV4 streaming hydrate: stacking {len(grouped)} TQ groups", flush=True)
    n_replaced = 0
    for idx, ((layer_prefix, proj_name), experts) in enumerate(
        sorted(grouped.items()), start=1
    ):
        n_exp = max(experts.keys()) + 1
        packed_list = [None] * n_exp
        norms_list = [None] * n_exp
        bits = None
        by_path = {}
        for expert_id, parts in experts.items():
            for part_name in ("packed", "norms", "bits"):
                try:
                    sf_path, tensor_key = parts[part_name]
                except KeyError as e:
                    raise RuntimeError(
                        f"DSV4 TQ group {layer_prefix}{proj_name} expert "
                        f"{expert_id} missing {part_name}"
                    ) from e
                by_path.setdefault(sf_path, []).append((expert_id, part_name, tensor_key))

        for sf_path, reqs in by_path.items():
            with safe_open(str(sf_path), framework="mlx") as sf:
                for expert_id, part_name, tensor_key in reqs:
                    tensor = sf.get_tensor(tensor_key)
                    if part_name == "packed":
                        packed_list[expert_id] = tensor
                    elif part_name == "norms":
                        norms_list[expert_id] = tensor
                    else:
                        bits = int(tensor[0].item())

        if bits is None:
            raise RuntimeError(f"DSV4 TQ group {layer_prefix}{proj_name} missing bits")
        if any(t is None for t in packed_list) or any(t is None for t in norms_list):
            raise RuntimeError(f"DSV4 TQ group {layer_prefix}{proj_name} incomplete")

        stacked_packed = mx.stack(packed_list)
        stacked_norms = mx.stack(norms_list)
        # Do not force-evaluate here. Evaluating while the per-expert source
        # arrays are still live doubles the group's peak memory. The final
        # full-model warmup/materialization runs after every source list has
        # been dropped, so steady-state tensors are still materialized before
        # first user decode without the load-time spike.
        n_exp, out_feat, packed_cols = stacked_packed.shape
        vals_per_u32 = 32 // bits
        in_features = packed_cols * vals_per_u32
        new_module = TurboQuantSwitchLinear(
            in_features=in_features,
            out_features=out_feat,
            num_experts=n_exp,
            bits=bits,
            bias=False,
            seed=mxtq_seed,
        )
        new_module.packed = stacked_packed
        new_module.norms = stacked_norms
        new_base = f"{layer_prefix}switch_mlp.{proj_name}"
        # Validate the target exists before mutating so bad naming fails loud.
        get_module(model, new_base)
        set_module(model, new_base, new_module)
        n_replaced += 1
        del packed_list, norms_list, stacked_packed, stacked_norms
        if idx % 12 == 0 or idx == len(grouped):
            print(
                f"  DSV4 streaming hydrate: stacked {idx}/{len(grouped)} groups",
                flush=True,
            )
        gc.collect()

    print(f"  Replaced {n_replaced} DSV4 routed TQ modules", flush=True)
    del grouped
    gc.collect()

    print("  DSV4 streaming hydrate: loading regular weights shard-by-shard", flush=True)
    for shard_i, sf_path in enumerate(weight_files, start=1):
        keep = regular_keys_by_shard.get(sf_path)
        if not keep:
            continue
        shard_weights = mx.load(str(sf_path))
        shard_regular = {k: v for k, v in shard_weights.items() if k in keep}
        del shard_weights
        if shard_regular:
            if hasattr(model, "sanitize"):
                shard_regular = model.sanitize(shard_regular)
            shard_regular = _sanitize_grouped_conv1d_layout(shard_regular)
            model.load_weights(list(shard_regular.items()), strict=False)
        del shard_regular
        if shard_i % 10 == 0 or shard_i == len(weight_files):
            print(
                f"  DSV4 streaming hydrate: regular shard {shard_i}/{len(weight_files)}",
                flush=True,
            )
        gc.collect()

    # Install the DSV4-safe fused routed-expert decode path. This mirrors the
    # generic post-hydration patch but now propagates `_DSV4SwiGLU.swiglu_limit`
    # into the Metal kernel so DSV4 gets silu(min(gate, 10)) * clip(up, +/-10).
    try:
        from mlx_lm.models.switch_layers import SwitchGLU, _gather_sort, _scatter_unsort
        from jang_tools.turboquant.fused_gate_up_kernel import (
            fused_gate_up_swiglu_matmul,
            make_fused_gate_up_swiglu_decode,
        )
        from jang_tools.turboquant.gather_tq_kernel import make_gather_tq_decode_per_row
        from jang_tools.turboquant.hadamard_kernel import hadamard_rotate_metal

        _orig_switchglu_call = SwitchGLU.__call__
        _decode_compiled = {}

        def _get_compiled_decode(in_f, out_f, bits, k, swiglu_limit=0.0, dp_bits=None):
            # Mixed-bit regression report (jang-tools 2.5.23 MiniMax-M2.7-
            # JANGTQ_K end-to-end test): JANGTQ_K is a mixed-bit profile
            # (gate=2 / up=2 / down=4). Without ``dp_bits`` the gather_dn
            # kernel below builds with gate_proj's bit width, then unpacks
            # down_proj.packed at the wrong stride — invisible on uniform-bit
            # bundles (4M, CRACK), but on JANGTQ_K it produces silent garbage
            # that compounds layer-by-layer into multilingual token salad.
            # Default ``dp_bits=None`` falls back to ``bits`` so uniform-bit
            # callers get the previous behavior unchanged.
            if dp_bits is None:
                dp_bits = bits
            limit_milli = int(round(float(swiglu_limit or 0.0) * 1000.0))
            cache_key = (in_f, out_f, bits, dp_bits, k, limit_milli)
            if cache_key in _decode_compiled:
                return _decode_compiled[cache_key]
            fused_gu = make_fused_gate_up_swiglu_decode(
                in_f, out_f, bits, k, swiglu_limit=swiglu_limit
            )
            # NB: gather_dn uses dp_bits, NOT bits.
            gather_dn = make_gather_tq_decode_per_row(out_f, in_f, dp_bits, k)

            def _mlp(x_flat, pg, ng, pu, nu, pd, nd, cb_gate, cb_down, signs_in, signs_dn, idx_flat):
                x_rot = hadamard_rotate_metal(x_flat, signs_in)
                x_act = fused_gu(x_rot, pg, ng, pu, nu, cb_gate, idx_flat)
                x_act_rot = hadamard_rotate_metal(x_act, signs_dn)
                return gather_dn(x_act_rot, pd, nd, cb_down, idx_flat)

            _decode_compiled[cache_key] = mx.compile(_mlp)
            return _decode_compiled[cache_key]

        def _dsv4_fused_switchglu_call(self, x, indices):
            gp = self.gate_proj
            up = self.up_proj
            dp = self.down_proj
            if not isinstance(gp, TurboQuantSwitchLinear) or not isinstance(up, TurboQuantSwitchLinear):
                return _orig_switchglu_call(self, x, indices)
            activation = getattr(self, "activation", None)
            swiglu_limit = getattr(activation, "swiglu_limit", 0.0) or 0.0
            x_sq = x
            while x_sq.ndim > 2 and x_sq.shape[-2] == 1:
                x_sq = x_sq.squeeze(-2)
            x_flat = x_sq.reshape(-1, gp.in_features)
            batch = x_flat.shape[0]
            k = indices.shape[-1] if indices.ndim > 0 else 1
            can_fast = batch == 1 and k > 0 and indices.ndim >= 1 and indices.size < 64
            if can_fast and not getattr(self, "training", False):
                idx_flat = indices.reshape(-1).astype(mx.uint32)
                compiled_mlp = _get_compiled_decode(
                    gp.in_features, gp.out_features, gp.bits, k, swiglu_limit,
                    dp_bits=dp.bits,
                )
                y = compiled_mlp(
                    x_flat.astype(mx.float32),
                    gp.packed, gp.norms, up.packed, up.norms,
                    dp.packed, dp.norms,
                    gp.codebook, dp.codebook,
                    gp.signs, dp.signs, idx_flat,
                )
                out = y.reshape(*indices.shape[:-1], k, 1, gp.in_features)
                if out.dtype != x.dtype:
                    out = out.astype(x.dtype)
                return out.squeeze(-2)

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
                swiglu_limit=swiglu_limit,
            )
            x_out = self.down_proj(x_act, idx, sorted_indices=do_sort)
            if do_sort:
                x_out = _scatter_unsort(x_out, inv_order, indices.shape)
            return x_out.squeeze(-2)

        SwitchGLU.__call__ = _dsv4_fused_switchglu_call
        patched = sum(
            1 for _, m in model.named_modules()
            if isinstance(m, SwitchGLU)
            and isinstance(getattr(m, "gate_proj", None), TurboQuantSwitchLinear)
            and isinstance(getattr(m, "up_proj", None), TurboQuantSwitchLinear)
        )
        print(
            f"  Patched SwitchGLU class for DSV4 limited SwiGLU fused gate+up ({patched} TQ instances)",
            flush=True,
        )
    except Exception as e:
        raise RuntimeError(
            "DSV4 JANGTQ SwitchGLU fusion failed. DSV4 routed experts require "
            "the fused limited-SwiGLU path; refusing to continue with stock "
            f"SwitchGLU. Original error: {e}"
        ) from e

    from jang_tools.loader import _fix_quantized_bits
    _fix_quantized_bits(model, {})

    if not skip_params_eval:
        try:
            from mlx.utils import tree_flatten
            flat = tree_flatten(model.parameters())
            for i in range(0, len(flat), 128):
                mx.eval(*[v for _, v in flat[i:i + 128]])
            mx.synchronize()
        except Exception as e:
            print(f"  [hydrate] chunked materialization failed: {e!r}", flush=True)

    print("  Hydration complete", flush=True)


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

    _model_type = (
        model_config.get("model_type", "")
        if isinstance(model_config, dict)
        else getattr(model_config, "model_type", "")
    )
    if (
        _model_type == "deepseek_v4"
        and os.environ.get("JANGTQ_DISABLE_DSV4_STREAM_LOAD", "0") != "1"
    ):
        return _hydrate_dsv4_jangtq_streaming(
            model, model_path, mxtq_seed, skip_params_eval=skip_params_eval
        )

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
    # Nemotron-H (nemotron_h, also Nemotron-3-Nano-Omni): backbone.layers.N.mixer
    # is `NemotronHMoE` for `E` block_type with `self.switch_mlp = SwitchMLP(...)`.
    # Per-expert shards land as `backbone.layers.N.mixer.experts.E.{gate,up,down}_proj`.
    # Without this pattern, the loop at line ~316 falls through, the per-expert
    # TQ tensors stay in raw 2D form keyed under the original `mixer.experts.E.X`
    # path, and the `get_module(model, base)` lookup at the replace step raises
    # AttributeError on every entry → "Replaced 0 modules" + the model loads
    # all expert weights as full bf16 nn.Linear (Metal RSS 13 GB → 111 GB → 503).
    # Symptom: /v1/responses 503 "Metal working set too full (104% of 107.5GB)".
    # Stacking emits under `{layer_prefix}switch_mlp.{proj_name}` which matches
    # the NemotronHMoE attribute path so `get_module(model, ...)` resolves
    # correctly and the per-expert shards become a 3-D TurboQuantSwitchLinear.
    nemo_pat = re.compile(
        rf"^({_VLM_PREFIX}backbone\.layers\.\d+\.mixer\.)"
        rf"experts\.(\d+)\.(gate_proj|up_proj|down_proj)$"
    )
    # JANGTQ-PRESTACK STANDARD (effective 2026-05-04): bundles MAY ship routed
    # expert tensors pre-stacked along axis 0 directly under
    # `{prefix}.{ffn|mlp|block_sparse_moe}.switch_mlp.{gate_proj|down_proj|up_proj|gate_up_proj}.{tq_packed|tq_norms|tq_bits}`
    # with shapes [n_experts, out, packed_in] (3D) / [n_experts, out] (2D) / [1] (scalar).
    # When matched, the bundle bypasses the sidecar / streaming-restack hot path
    # entirely — TurboQuantSwitchLinear is built from the file as-is.
    prestack_pat = re.compile(
        rf"^({_VLM_PREFIX}layers\.\d+\.(?:ffn|mlp|block_sparse_moe)\.)"
        rf"switch_mlp\.(gate_up_proj|gate_proj|up_proj|down_proj)$"
    )
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
            # Nemotron-H: backbone.layers.N.mixer.experts.E.{up,down}_proj
            # → backbone.layers.N.mixer.switch_mlp.{fc1,fc2}
            #
            # MLX `mlx_lm.models.switch_layers.SwitchMLP` exposes ONLY `fc1`
            # and `fc2` (no gate_proj/up_proj/down_proj). MLX `nemotron_h.py`
            # sanitize() (lines 540-555) maps the per-expert weight keys via
            # the table `[("down_proj", "fc2"), ("up_proj", "fc1")]`. We MUST
            # mirror that mapping here, otherwise `set_module(model, base)`
            # fails on every entry → "Replaced 0 modules" → 128 experts × 26
            # MoE layers default-initialise as bf16 SwitchLinear placeholders
            # (~110 GB Metal allocation on a 30B-A3B bundle).
            #
            # Nemotron-H is the GLU activation variant (`relu2(up_proj(x))`,
            # NemotronHMLP.__call__), NOT SwiGLU — gate_proj does not exist
            # in this architecture. nemo_pat allows gate_proj for forward
            # compat, but we drop it here if encountered.
            layer_prefix = m.group(1)
            expert_id = int(m.group(2))
            proj_name = m.group(3)
            _nemo_proj_map = {"up_proj": "fc1", "down_proj": "fc2"}
            mapped = _nemo_proj_map.get(proj_name)
            if mapped is None:
                # gate_proj or unknown — drop (Nemotron-H is GLU, not SwiGLU).
                tq_groups.pop(base, None)
                continue
            grouped_experts.setdefault((layer_prefix, mapped), {})[expert_id] = tq_groups.pop(base)
            continue
        m = prestack_pat.match(base)
        if m:
            # JANGTQ-PRESTACK STANDARD: bundle ships routed-expert tensors
            # pre-stacked under `{prefix}.{ffn|mlp|block_sparse_moe}.switch_mlp.{proj}`
            # directly. No restacking, no sidecar — load and assign as-is.
            layer_prefix = m.group(1)
            proj_name = m.group(2)
            parts = tq_groups.pop(base)
            if proj_name == "gate_up_proj":
                # Split combined gate+up along the output-row axis.
                packed = parts["packed"]
                norms = parts["norms"]
                bits = parts["bits"]
                mid = packed.shape[-2] // 2
                for half, name in (
                    (slice(None, mid), "gate_proj"),
                    (slice(mid, None), "up_proj"),
                ):
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

    def _tq_target_candidates(base: str) -> list[str]:
        """Return possible runtime module paths for one TQ tensor group.

        VLM skeletons can expose the text decoder under
        ``language_model.model`` while bundles store text weights as
        ``model.layers``. Regular weights are normalized later in
        sanitize/load; TQ module replacement runs before that and needs the
        same path reconciliation.
        """
        candidates = [base]
        if base.startswith("model.layers."):
            candidates.append(
                "language_model.model.layers." + base[len("model.layers."):]
            )
        if base.startswith("model.language_model."):
            candidates.append(
                "language_model.model." + base[len("model.language_model."):]
            )
        seen = set()
        ordered = []
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                ordered.append(candidate)
        return ordered

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

    # Hydrate-skip allowlist: the legacy "Skip (not in
    # model)" branch silently dropped TQ tensors whose target module didn't
    # exist on the loaded mlx_lm class. That's correct ONLY for documented
    # non-inference paths (multi-token-prediction heads, training-only
    # auxiliaries). Any other miss means the loader is silently
    # under-replacing — a model loads but routes through fp16 fallback or
    # raises mid-decode, both worse than failing fast at load time.
    #
    # Allowlist regexes match the substring of the missing key. Anything
    # NOT matching one of these → hard-fail with VerificationError-style
    # context. Override: VMLX_JANGTQ_ALLOW_HYDRATE_SKIPS=1 keeps legacy
    # silent-skip behavior for emergency loads (NOT recommended).
    _hydrate_allowlist = (
        # Multi-Token-Prediction layer (DSV4) — used only at training time.
        re.compile(r"(^|\.)mtp\."),
        re.compile(r"(^|\.)eh_proj(\.|$)"),
        re.compile(r"(^|\.)shared_head(\.|$)"),
        # Embedding tied to lm_head — exact embed_tokens TQ aux group only.
        re.compile(r"(^|\.)embed_tokens$"),
    )
    _allow_skips = os.environ.get(
        "VMLX_JANGTQ_ALLOW_HYDRATE_SKIPS", "0"
    ).lower() in ("1", "true", "yes")
    _missed_required: list[str] = []

    for base, parts in list(tq_groups.items()):
        packed = parts["packed"]
        norms = parts["norms"]
        bits = parts["bits"]
        vals_per_u32 = 32 // bits

        resolved_base = None
        existing = None
        for candidate in _tq_target_candidates(base):
            try:
                existing = get_module(model, candidate)
                resolved_base = candidate
                break
            except (AttributeError, IndexError, KeyError):
                continue

        if resolved_base is None:
            allowlisted = any(p.search(base) for p in _hydrate_allowlist)
            if allowlisted:
                print(f"    Skip allowlisted (not in model): {base}", flush=True)
                continue
            if _allow_skips:
                print(
                    f"    Skip (env override; not in model): {base}",
                    flush=True,
                )
                continue
            # Real silent miss — defer hard-fail to end of loop so all
            # missing modules are reported at once.
            _missed_required.append(base)
            continue

        # Read the logical input width off the pre-replacement module rather
        # than recovering it from `packed_cols * vals_per_u32`. The recovery
        # math is only exact when in_features divides vals_per_u32 cleanly
        # (true for bits ∈ {1,2,4,8,16} where vals_per_u32 ∈ {32,16,8,4,2}),
        # but JANGTQ3 ships bits=3 → vals_per_u32=10 which over-rounds for any
        # in_features not a multiple of 10 (4096 → 4100, 7168 → 7170, 3072 →
        # 3080). MLX affine-quantized SwitchLinear placeholders also expose
        # packed storage in `weight.shape[-1]`; their logical width lives in
        # `input_dims`. A wrong in_features mis-keys the per-(in_features,
        # seed) signs cache and the per-(in_features, bits) codebook cache so
        # the Hadamard rotation applied to x silently mismatches what the
        # weights were quantized against → garbage decode.
        existing_in = _infer_tq_input_features(existing)

        if packed.ndim == 3:
            n_exp, out_feat, packed_cols = packed.shape
            in_features = int(existing_in) if existing_in else packed_cols * vals_per_u32
            new_module = TurboQuantSwitchLinear(
                in_features=in_features, out_features=out_feat,
                num_experts=n_exp, bits=bits, bias=False, seed=mxtq_seed,
            )
        else:
            out_feat, packed_cols = packed.shape
            in_features = int(existing_in) if existing_in else packed_cols * vals_per_u32
            new_module = TurboQuantLinear(
                in_features=in_features, out_features=out_feat,
                bits=bits, bias=False, seed=mxtq_seed,
            )
        new_module.packed = packed
        new_module.norms = norms

        set_module(model, resolved_base, new_module)
        n_replaced += 1

    print(f"  Replaced {n_replaced} modules", flush=True)

    # Hydrate-skip enforcement. Any TQ keys whose
    # target module is missing AND not on the documented allowlist
    # (mtp/eh_proj/shared_head/embed_tokens) are real silent-misses.
    # Hard-fail at load time so they don't manifest as fp16-fallback
    # garbage or mid-decode crashes.
    if _missed_required:
        head = _missed_required[:8]
        more = (
            f" (+{len(_missed_required)-8} more)"
            if len(_missed_required) > 8
            else ""
        )
        raise RuntimeError(
            f"JANGTQ load: {len(_missed_required)} TQ tensor groups have no "
            f"target module on the loaded model class — this means the model "
            f"would silently route through fp16 fallback or crash mid-decode. "
            f"First {len(head)}: {head}{more}. Override (NOT recommended): "
            f"VMLX_JANGTQ_ALLOW_HYDRATE_SKIPS=1."
        )

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
    regular = _sanitize_grouped_conv1d_layout(regular)
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
    # Class-level patch because Python looks up `__call__` on the type.
    # Production rule: fused SwitchGLU is mandatory for JANGTQ.
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

        def _get_compiled_decode(in_f, out_f, bits, K, swiglu_limit=0.0, dp_bits=None):
            # JANGTQ_K (per-projection mixed bits): gate_proj/up_proj share `bits`
            # (fused gate+up kernel requires gate==up), but down_proj may differ —
            # MiniMax-M2.7-JANGTQ_K ships gate=2, up=2, down=4. Without splitting
            # the gather kernel's bit width, the down_proj's 4-bit packed tensors
            # would be unpacked as 2-bit → wrong indices into the (correct) 4-bit
            # codebook → garbage output. dp_bits=None defaults to the legacy
            # uniform-bits behavior (bits) so existing JANGTQ2/3/4 callers are
            # unchanged byte-for-byte.
            if dp_bits is None:
                dp_bits = bits
            limit_milli = int(round(float(swiglu_limit or 0.0) * 1000.0))
            key = (in_f, out_f, bits, dp_bits, K, limit_milli)
            if key in _DECODE_COMPILED:
                return _DECODE_COMPILED[key]
            fused_gu = make_fused_gate_up_swiglu_decode(
                in_f, out_f, bits, K, swiglu_limit=swiglu_limit
            )
            # P19 (fused rot+gather) was tested but regressed 44.6 → 34.2 tok/s
            # in real decode despite +3 μs microbench win. Likely shmem/occupancy
            # interaction with other kernels. Reverted to split path.
            gather_dn = make_gather_tq_decode_per_row(out_f, in_f, dp_bits, K)

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
            _activation = getattr(self, "activation", None)
            _swiglu_limit = getattr(_activation, "swiglu_limit", 0.0) or 0.0

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
                # JANGTQ_K: dp.bits may differ from gp.bits (e.g. 4 vs 2 on
                # MiniMax-M2.7-JANGTQ_K). Pass both so the gather_dn kernel
                # built inside _get_compiled_decode unpacks down_proj weights
                # at the correct bit width. Cache key includes both bits, so
                # JANGTQ2 (uniform) and JANGTQ_K layers don't share kernels.
                compiled_mlp = _get_compiled_decode(
                    gp.in_features, gp.out_features, gp.bits, K, _swiglu_limit,
                    dp_bits=dp.bits,
                )
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
                swiglu_limit=_swiglu_limit,
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
        raise RuntimeError(
            "JANGTQ SwitchGLU fusion failed. This fast path is mandatory for "
            "correct JANGTQ routed-expert decoding; refusing to continue with "
            f"stock SwitchGLU. Original error: {_e}"
        ) from _e

    # P15: mx.compile ONLY pure router math (mlx_lm pattern — free function,
    # no closures over self, shapeless so prefill/decode share one graph).
    # Safer than patching __call__ which thrashes the compile cache.
    try:
        from jang_tools.turboquant.router_kernel import make_sigmoid_bias_topk_router

        # Cache compiled variants keyed on (k, bits).
        _ROUTER_CACHE = {}
        _MLP_CACHE = {}
        _CUSTOM_ROUTER_CACHE = {}

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
            # mlx 0.29+ removed `precise=True` from `mx.softmax`. fp32 input
            # already gives a precise computation, so just drop the kwarg.
            if renorm:
                def _router(gates_f32):
                    scores = mx.softmax(gates_f32, axis=-1)
                    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
                    sel = mx.take_along_axis(scores, inds, axis=-1)
                    sel = sel / (mx.sum(sel, axis=-1, keepdims=True) + 1e-20)
                    return inds, sel
            else:
                def _router(gates_f32):
                    scores = mx.softmax(gates_f32, axis=-1)
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
        _compile_full_moe_block = os.environ.get(
            "JANGTQ_COMPILE_FULL_MOE_BLOCK", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if _compile_full_moe_block:
            print(
                "  [experimental] JANGTQ_COMPILE_FULL_MOE_BLOCK=1: "
                "decode MoE block will be mx.compile'd per module",
                flush=True,
            )
        _custom_sigmoid_router = os.environ.get(
            "JANGTQ_CUSTOM_ROUTER_TOPK", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if _custom_sigmoid_router:
            print(
                "  [experimental] JANGTQ_CUSTOM_ROUTER_TOPK=1: "
                "using Metal sigmoid+bias top-k router for single-token decode",
                flush=True,
            )

        def _decode_batch_size(x, in_features):
            x_sq = x
            while x_sq.ndim > 2 and x_sq.shape[-2] == 1:
                x_sq = x_sq.squeeze(-2)
            try:
                return x_sq.reshape(-1, in_features).shape[0]
            except Exception:
                return -1

        _n_patched = 0
        _patched_classes = set()
        for _, _mod in model.named_modules():
            _cls = _mod.__class__
            if _cls in _patched_classes or "SparseMoeBlock" not in _cls.__name__:
                continue
            # Skip MoE classes whose `gate` is a custom routing module that
            # already returns `(topk_idx, topk_weight)` — e.g. bailing_hybrid
            # / Ling-2.6-flash. The P15 patch assumes `self.gate(x)` returns
            # raw logits; calling it on a Gate that returns a tuple yields
            # `mx.softmax(<tuple>, axis=-1)` which is a TypeError. Detect by
            # checking whether `gate` is a plain Linear-like (has `weight`
            # attribute) or a wrapper module (has `gate_proj` etc).
            _gate_mod = getattr(_mod, "gate", None)
            if _gate_mod is None or hasattr(_gate_mod, "gate_proj"):
                continue

            def _patched_moe(self, x):
                k = getattr(self, "num_experts_per_tok",
                            getattr(self, "top_k",
                                    getattr(self, "num_routed_experts", None)))
                if k is None:
                    raise AttributeError(
                        f"{type(self).__name__} missing one of "
                        "num_experts_per_tok / top_k / num_routed_experts"
                    )

                if _compile_full_moe_block:
                    gate_in_features = int(
                        getattr(getattr(self, "switch_mlp", None), "gate_proj", None).in_features
                    )
                    # Only compile the decode shape. Prefill/large-batch paths keep
                    # the long-tested current implementation because SwitchGLU's
                    # compiled fast path is specifically batch=1, small top-k.
                    if _decode_batch_size(x, gate_in_features) == 1:
                        compiled = getattr(self, "_jangtq_compiled_full_moe_block", None)
                        if compiled is None:
                            has_bias = hasattr(self, "e_score_correction_bias")
                            renorm = bool(getattr(self, "norm_topk_prob", True))

                            def _full_moe_decode(x_in):
                                gates = self.gate(x_in.astype(mx.float32))
                                if has_bias:
                                    scores = mx.sigmoid(gates)
                                    orig = scores
                                    biased = scores + self.e_score_correction_bias
                                    inds = mx.argpartition(-biased, kth=k - 1, axis=-1)[..., :k]
                                    sel = mx.take_along_axis(orig, inds, axis=-1)
                                    sel = sel / (mx.sum(sel, axis=-1, keepdims=True) + 1e-20)
                                else:
                                    scores = mx.softmax(gates, axis=-1)
                                    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
                                    sel = mx.take_along_axis(scores, inds, axis=-1)
                                    if renorm:
                                        sel = sel / (mx.sum(sel, axis=-1, keepdims=True) + 1e-20)
                                y = self.switch_mlp(x_in, inds)
                                return (y * sel.astype(x_in.dtype)[..., None]).sum(axis=-2)

                            compiled = mx.compile(_full_moe_decode)
                            setattr(self, "_jangtq_compiled_full_moe_block", compiled)
                        return compiled(x)

                gates = self.gate(x.astype(mx.float32))
                # MiniMax stores K as `num_experts_per_tok`, Qwen3-Next as `top_k`,
                # some GLM variants as `num_routed_experts`. Probe in order.
                if hasattr(self, "e_score_correction_bias"):
                    gate_in_features = int(
                        getattr(getattr(self, "switch_mlp", None), "gate_proj", None).in_features
                    )
                    if (
                        _custom_sigmoid_router
                        and _decode_batch_size(x, gate_in_features) == 1
                    ):
                        experts = int(self.e_score_correction_bias.shape[-1])
                        key = ("custom_sigbias", experts, int(k))
                        router = _CUSTOM_ROUTER_CACHE.get(key)
                        if router is None:
                            router = make_sigmoid_bias_topk_router(
                                experts=experts,
                                top_k=int(k),
                            )
                            _CUSTOM_ROUTER_CACHE[key] = router
                        inds, scores = router(
                            gates.reshape(-1),
                            self.e_score_correction_bias.reshape(-1),
                        )
                        out_shape = tuple(gates.shape[:-1]) + (int(k),)
                        inds = inds.reshape(out_shape)
                        scores = scores.reshape(out_shape)
                    else:
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
                    # Different model classes name the head-count
                    # attribute differently — Llama-style uses
                    # `num_attention_heads`, Qwen3 uses `n_heads`,
                    # NemotronHAttention / DeepSeek use `num_heads`,
                    # MLA classes use `n_q_heads` etc. The pre-patch
                    # safety-check at the gather step (line ~797)
                    # already does this fallback; mirror it here so
                    # the runtime reshape doesn't AttributeError on
                    # families that don't expose `num_attention_heads`.
                    n_heads = getattr(self, "num_attention_heads", None) \
                        or getattr(self, "num_heads", None) \
                        or getattr(self, "n_heads", None)
                    n_kv_heads = getattr(self, "num_key_value_heads", None) \
                        or getattr(self, "n_kv_heads", None) \
                        or getattr(self, "num_kv_heads", None) \
                        or n_heads
                    if n_heads is None or n_kv_heads is None:
                        # No recognised head-count attribute — bail to
                        # the original __call__ rather than reshape
                        # to a wrong layout.
                        return orig_call(self, x, mask=mask, cache=cache)
                    queries = queries.reshape(B, L, n_heads, -1).transpose(0, 2, 1, 3)
                    keys = keys.reshape(B, L, n_kv_heads, -1).transpose(0, 2, 1, 3)
                    values = values.reshape(B, L, n_kv_heads, -1).transpose(0, 2, 1, 3)
                    # Some attention families don't apply rope at all
                    # (NemotronHAttention does cache update directly,
                    # no rotary embedding). Bail to the original
                    # `__call__` rather than crash on `self.rope` —
                    # the QKV fusion savings still apply via the
                    # `qkv = fused(x)` step above; we just give up
                    # the dispatch fusion for the rope+SDPA stage.
                    has_rope = hasattr(self, "rope")
                    if cache is not None:
                        if has_rope:
                            queries = self.rope(queries, offset=cache.offset)
                            keys = self.rope(keys, offset=cache.offset)
                        keys, values = cache.update_and_fetch(keys, values)
                    else:
                        if has_rope:
                            queries = self.rope(queries)
                            keys = self.rope(keys)
                    # SDPA scale: most classes have `self.scale`. Some
                    # MLA-style use `self.softmax_scale` or compute
                    # inline. Fallback to standard `1/sqrt(head_dim)`.
                    sdpa_scale = getattr(self, "scale", None)
                    if sdpa_scale is None:
                        head_dim = queries.shape[-1]
                        sdpa_scale = head_dim ** -0.5
                    from mlx_lm.models.base import scaled_dot_product_attention
                    out = scaled_dot_product_attention(
                        queries, keys, values, cache=cache, scale=sdpa_scale, mask=mask,
                    )
                    out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
                    return self.o_proj(out)
                return _patched
            _attn_cls.__call__ = _make_patched(_orig_attn_call)
        print(f"  P18 QKV fusion: {len(_patched_attn_classes)} class(es), {len(_qkv_fused)} instances", flush=True)
    except Exception as _e:
        print(f"  P18 QKV fusion skipped: {_e}", flush=True)

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

    from jang_tools.jangrt.inference_mode import ensure_inference_mode
    _infer_report = ensure_inference_mode(model, label="JANGTQ")
    if _infer_report["training_modules_remaining"]:
        print(
            "  [hydrate] WARNING: inference mode left "
            f"{_infer_report['training_modules_remaining']} training=True modules "
            f"(examples={_infer_report['remaining_examples']})",
            flush=True,
        )
    elif _infer_report["eval_called"]:
        print("  [hydrate] inference mode enabled", flush=True)

    if not skip_params_eval:
        # mx.synchronize alone only flushes the dispatch queue; it does
        # not force MLX's lazy parameter materialization. Without an
        # explicit evaluate, TQ-hydrated weights remain deferred and the
        # FIRST /v1/chat/completions request pays the full materialization
        # cost mid-decode — producing garbage logits, hallucinated
        # thinking content, and degenerate repetition on short prompts.
        # Force materialization here so the returned handle is truly ready.
        _evaluate = getattr(mx, "eval", None)
        if _evaluate is not None:
            try:
                _evaluate(model.parameters())
            except Exception as _e:
                print(f"  [hydrate] parameter materialization failed: {_e!r}", flush=True)
        mx.synchronize()
    print("  Hydration complete", flush=True)

    # JANGTQ_TOPK_OVERRIDE env var: lower MoE router top_k at inference for
    # decode speedup at the cost of some quality. This is an explicit opt-in
    # runtime flag; invalid values must fail loudly instead of silently
    # falling back to the trained K. Top-1 families (ZAYA) no-op.
    try:
        from jang_tools.topk_override import (
            apply_topk_override,
            topk_override_from_env,
        )
        _topk_k = topk_override_from_env()
        if _topk_k is not None:
            _n = apply_topk_override(model, _topk_k)
            print(
                f"  [topk-override] JANGTQ_TOPK_OVERRIDE={_topk_k}: "
                f"patched {_n} router/MoE attribute(s)",
                flush=True,
            )
    except ValueError:
        raise
    except Exception as _e:
        print(f"  [topk-override] skipped: {_e!r}", flush=True)
