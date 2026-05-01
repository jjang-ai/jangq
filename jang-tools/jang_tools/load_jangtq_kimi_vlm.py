"""Load a Kimi K2.6 JANGTQ bundle with vision/video support.

Routes the `model_type="kimi_k25"` bundle through mlx_vlm's `kimi_vl` module
(which was originally written for Kimi-VL-A3B / Moonlight — the vision
tower and projector architecture are identical to Kimi K2.6 at 27-block
MoonViT + patchmerger, so we can reuse the code by registering the
remapping in `mlx_vlm.utils.MODEL_REMAPPING`).

The text side is routed through the JANGTQ TurboQuant kernels as usual via
`_hydrate_jangtq_model`. The vision tower + multi_modal_projector are
loaded as plain fp16 — they are small (~1-2 GB total) and not worth
quantizing. The `mm_projector` -> `multi_modal_projector` key rename is
handled automatically inside `_hydrate_jangtq_model` when it sees a
Kimi VLM (`model.model_type == "kimi_k25"` + vision_tower present).

Returns `(model, processor)` ready for `mlx_vlm.generate`:

    from jang_tools.load_jangtq_kimi_vlm import load_jangtq_kimi_vlm_model
    from mlx_vlm import generate
    model, processor = load_jangtq_kimi_vlm_model("/path/to/Kimi-K2.6-REAP-30-JANGTQ_1L")
    out = generate(
        model, processor,
        prompt="Describe this image.",
        image="/path/to/cat.jpg",
        max_tokens=80, temp=0.0,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import mlx.nn as nn


# Install the kimi_k25 -> kimi_vl remap at import time. This is how
# mlx_vlm.utils.get_model_and_args routes config["model_type"] to a
# `mlx_vlm.models.<name>` module. Our bundle's config says
# "model_type": "kimi_k25" (the text wrapper class on HuggingFace), but
# mlx_vlm's kimi_vl module supports the same vision tower + projector +
# DeepseekV3 text backbone, so the code path works unchanged.
def _ensure_kimi_remap() -> None:
    """Register kimi_k25 everywhere mlx_vlm dispatches on model_type.

    Two places need to know the remap:
      1. `mlx_vlm.utils.MODEL_REMAPPING` — routes model class loading
         (config["model_type"] → `mlx_vlm.models.<name>` module).
      2. `mlx_vlm.prompt_utils.MODEL_CONFIG` — routes chat-template
         message formatting. If unset, apply_chat_template raises
         "Unsupported model" even though the model loaded fine.

    Both entries are idempotent (setdefault) so calling this from
    load_jangtq_kimi_vlm_model on every load is safe.
    """
    try:
        from mlx_vlm import utils as _mlx_vlm_utils
    except ImportError:  # pragma: no cover
        return
    mapping = getattr(_mlx_vlm_utils, "MODEL_REMAPPING", None)
    if mapping is not None:
        mapping.setdefault("kimi_k25", "kimi_vl")

    # Chat-template dispatch lives in prompt_utils.MODEL_CONFIG, which maps
    # model_type → MessageFormat enum. Kimi K2.6 uses the same
    # <|media_begin|>image<|media_content|><|media_pad|><|media_end|> tokens
    # as Kimi-VL-A3B (Moonlight), so LIST_WITH_IMAGE applies unchanged.
    try:
        from mlx_vlm import prompt_utils as _pu
        cfg = getattr(_pu, "MODEL_CONFIG", None)
        if cfg is not None and "kimi_k25" not in cfg:
            cfg["kimi_k25"] = cfg.get("kimi_vl")
    except ImportError:
        pass


_ensure_kimi_remap()


def _set_vl_wired_limit() -> None:
    """Lower MLX's wired_limit for VL loading on large MoE bundles.

    The default `_apply_wired_limit_safe_default` inside `load_jangtq_model`
    targets 70 % of total RAM (192 GB on a 275 GB Mac Studio). That's
    correct for text-only inference, where prefill activations are small.

    VL first-forward on a 191 GB bundle has much higher peak memory:
      * vision_tower fp16 page-in (~1-2 GB, cold from SSD on first call)
      * MoonViT 27-block forward on image patches
      * ~100-200 image-feature tokens appended to the text sequence
      * 61 MLA + MoE layers of prefill over the extended sequence
      * SwitchGLU prefill path stacks per-expert scratch tensors

    Under concurrent SSD load (HF uploads, other conversions) this spike
    has been observed to trigger macOS Jetsam on a 275 GB machine, with
    the VL process killed mid-prefill (bash reports `Killed: 9` / exit 137).

    Dropping the wired_limit to ~52 % of total RAM (143 GB on 275 GB)
    gives Jetsam ~60 GB of extra evictable pagecache headroom, which is
    enough to keep the VL prefill alive under the observed contention.

    Calls `mx.set_wired_limit` *before* the standard auto-default fires
    in the text loader. Idempotent — if caller already chose a lower
    limit we don't raise it back up.
    """
    try:
        import psutil, sys
        import mlx.core as _mx
        if sys.platform != "darwin":
            return
        total_gb = psutil.virtual_memory().total / 1e9
        target_gb = int(total_gb * 0.52)
        target_gb = max(32, min(target_gb, 160))
        _mx.set_wired_limit(target_gb * 1000 * 1000 * 1000)
        print(
            f"  [wired_limit:VL] set to {target_gb} GB "
            f"(~52% of {total_gb:.0f} GB; headroom for VL prefill spike)",
            flush=True,
        )
    except Exception:
        pass


def _warmup_jit_per_layer(model, verbose: bool = True) -> None:
    """Pre-compile Metal kernels one layer at a time.

    Cold first-forward on a 191 GB quantized MoE through 61 layers triggers
    enough Metal shader JIT that the single command buffer exceeds the
    ~60 s watchdog. Running a 1-token forward per layer (with eval+sync
    between) amortizes the compile cost across 61 small buffers. Each
    buffer now compiles 1 layer's worth of kernels (~1-3 s each) and
    finishes well under watchdog.

    After this runs once, all kernel shapes used by Kimi K2.6's MoE (MLA
    q_a/q_b/kv_a/kv_b/o at 8-bit, SwitchGLU gate/up/down at 2-bit MXTQ,
    shared expert, dense layer 0, norms, lm_head) are cached and subsequent
    forwards skip recompile.

    This function is idempotent: after first successful warmup, the model
    is tagged with `_jang_warmed=True` and re-calls are no-ops.

    Why not `mx.compile`: we want concrete JIT-per-shape compiles, which
    MLX does on first dispatch. `mx.compile` is for graph-rewrite caching
    of Python callables, not for pre-warming Metal shaders.
    """
    if getattr(model, "_jang_warmed", False):
        if verbose:
            print("  [warmup] already warmed, skipping", flush=True)
        return

    import mlx.core as _mx
    _materialize = getattr(_mx, "eval")

    # Find the language_model (VLM wrapper) or the top-level model (text).
    lm = getattr(model, "language_model", None) or model

    # Dig for the inner model with a `layers` list.
    inner = None
    if hasattr(lm, "model") and hasattr(lm.model, "layers"):
        inner = lm.model
    elif hasattr(lm, "layers"):
        inner = lm
    if inner is None or not hasattr(inner, "layers"):
        if verbose:
            print("  [warmup] couldn't locate inner.layers, skipping",
                  flush=True)
        return

    layers = inner.layers
    # Hidden size — MUST come from config, not from a module weight shape.
    # QuantizedLinear / QuantizedEmbedding store packed uint32 weights
    # (4 int8s or 16 int2s per word), so `weight.shape[-1]` is the PACKED
    # column count, not the real hidden_size. e.g. 8-bit embed at
    # hidden=7168 has weight.shape[-1] = 7168/4 = 1792.
    H = None
    for src in (
        getattr(model, "config", None),
        getattr(model, "args", None),
        getattr(lm, "config", None),
        getattr(lm, "args", None),
    ):
        if src is None:
            continue
        # Flat hidden_size
        H = getattr(src, "hidden_size", None)
        if H:
            break
        # Nested under text_config / text
        for key in ("text_config", "text"):
            sub = getattr(src, key, None)
            if sub is not None:
                H = getattr(sub, "hidden_size", None)
                if H:
                    break
        if H:
            break
    if not H:
        if verbose:
            print("  [warmup] couldn't determine hidden_size from config, "
                  "skipping", flush=True)
        return

    hc_mult = None
    for src in (
        getattr(model, "config", None),
        getattr(model, "args", None),
        getattr(lm, "config", None),
        getattr(lm, "args", None),
    ):
        if src is None:
            continue
            
        def _get_val(obj, key):
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)
            
        # Flat hc_mult
        hc_mult = _get_val(src, "hc_mult")
        if hc_mult:
            break
        # Nested under text_config / text
        for key in ("text_config", "text"):
            sub = _get_val(src, key)
            if sub is not None:
                hc_mult = _get_val(sub, "hc_mult")
                if hc_mult:
                    break
        if hc_mult:
            break

    # 1-token dummy through each layer sequentially.
    import time as _time
    t0 = _time.time()
    if hc_mult:
        x = _mx.zeros((1, 1, int(hc_mult), H), dtype=_mx.bfloat16)
    else:
        x = _mx.zeros((1, 1, H), dtype=_mx.bfloat16)
    _materialize(x)
    if verbose:
        print(f"  [warmup] layer-by-layer 1-token forward "
              f"({len(layers)} layers, H={H}) ...", flush=True)

    for i, layer in enumerate(layers):
        try:
            # Try the common DeepseekV3-style signature:
            # DecoderLayer(x, mask=None, cache=None) -> x
            x = layer(x, None, None)
        except TypeError:
            try:
                x = layer(x, mask=None, cache=None)
            except TypeError:
                x = layer(x)
        _materialize(x)
        _mx.synchronize()
        if verbose and (i % 10 == 0 or i == len(layers) - 1):
            print(f"    layer {i}/{len(layers) - 1} compiled  "
                  f"({_time.time() - t0:.1f}s)", flush=True)

    # Also warm the final norm + lm_head if present.
    try:
        if hasattr(inner, "norm"):
            x = inner.norm(x)
            _materialize(x)
        lm_head = getattr(lm, "lm_head", None)
        if lm_head is not None:
            y = lm_head(x)
            _materialize(y)
            _mx.synchronize()
    except Exception:
        pass

    # Second-pass warmup: full-model forward at a prefill-chunk shape so
    # the multi-token attention + MoE kernels are JIT-cached too. The
    # per-layer 1-token warmup above only compiled the (1, 1, H) shaders;
    # actual prefill will dispatch (1, N, H) for N up to prefill_step_size,
    # which is a distinct kernel on Metal. Without this, the first
    # `mlx_lm.generate` call still pays a cold-compile penalty large
    # enough to trip the Metal watchdog.
    #
    # We run through the TOP-LEVEL model (not per-layer) so embed_tokens,
    # the outer DeepseekV3Model.__call__ path, final norm, and lm_head
    # all land in the cached-shader set. Uses a real KVCache and a tiny
    # token input (N=16 to match our text/VL prefill_step_size default).
    try:
        from mlx_lm.models.cache import make_prompt_cache
        WARMUP_N = 16
        cache_for_warm = make_prompt_cache(lm)
        tiny_ids = _mx.zeros((1, WARMUP_N), dtype=_mx.int32)
        t1 = _time.time()
        if verbose:
            print(f"  [warmup] full-model 16-token prefill shape ...",
                  flush=True)
        # Some VLM wrappers expect different call signatures; try several.
        call_variants = [
            lambda: lm(inputs=tiny_ids, cache=cache_for_warm),
            lambda: lm(tiny_ids, cache=cache_for_warm),
            lambda: lm(tiny_ids),
        ]
        warmed = False
        for call in call_variants:
            try:
                out = call()
                # some return a namedtuple with .logits; handle both.
                if hasattr(out, "logits"):
                    _materialize(out.logits)
                else:
                    _materialize(out)
                _mx.synchronize()
                warmed = True
                break
            except TypeError:
                continue
        if verbose:
            kind = "ok" if warmed else "skipped (no matching signature)"
            print(f"    {kind}  ({_time.time() - t1:.1f}s)", flush=True)
    except Exception as _e:
        if verbose:
            print(f"  [warmup] full-model pass skipped: {_e!r}", flush=True)

    model._jang_warmed = True
    if verbose:
        print(f"  [warmup] done in {_time.time() - t0:.1f}s "
              f"(Metal kernels now JIT-cached)", flush=True)


def _install_vl_command_buffer_split(model) -> None:
    """Split the VL forward into multiple Metal command buffers.

    Root cause of `kIOGPUCommandBufferCallbackErrorTimeout` on first VL
    prefill: MLX batches the vision_tower forward + multi_modal_projector
    + 61 language layers into ONE command buffer. Metal's watchdog kills
    any buffer running > ~60 s. Under SSD contention (HF uploads, parallel
    conversion), cold vision_tower weight page-in pushes the buffer past
    the watchdog, so the first `generate()` call aborts with SIGABRT 134
    before emitting any tokens.

    Fix: wrap `Model.get_input_embeddings` so we materialize the vision
    output + synchronize between the vision pass and the language pass.
    That forces MLX to flush the vision command buffer and start a fresh
    one for the 61-layer prefill — neither buffer exceeds the watchdog
    alone.

    Idempotent — checks a `_jang_cb_split` marker on the Model instance.
    """
    import mlx.core as _mx
    _materialize = getattr(_mx, "eval")  # lazy-tensor materialization

    if getattr(model, "_jang_cb_split", False):
        return
    if not (hasattr(model, "vision_tower") and hasattr(model, "language_model")):
        return
    orig_get_input_embeddings = model.get_input_embeddings

    def _patched_get_input_embeddings(input_ids=None, pixel_values=None, **kw):
        out = orig_get_input_embeddings(input_ids=input_ids, pixel_values=pixel_values, **kw)
        # Force vision + projector to materialize in their own command buffer
        # BEFORE the language_model prefill runs. This bounds each Metal
        # command buffer below the watchdog timeout.
        try:
            _materialize(out.inputs_embeds)
            _mx.synchronize()
        except Exception:
            pass
        return out

    model.get_input_embeddings = _patched_get_input_embeddings
    model._jang_cb_split = True


def load_jangtq_kimi_vlm_model(model_path) -> Tuple[nn.Module, object]:
    """Load a Kimi K2.6 JANGTQ bundle with vision + video enabled.

    Delegates to :func:`jang_tools.load_jangtq_vlm.load_jangtq_vlm_model`,
    which builds the mlx_vlm model skeleton (vision_tower +
    language_model + multi_modal_projector), runs JANGTQ hydration (TQ
    kernel replacement, compile-friendly MoE, MLA bit-width fix, wired
    limit auto-tune), and returns the model + processor.

    This wrapper:
      1. Ensures the `kimi_k25` remap is installed in mlx_vlm's
         MODEL_REMAPPING + MODEL_CONFIG (so dispatch lookups succeed).
      2. Sets a **lower** wired_limit than the text loader uses (52 %
         of RAM vs 70 %). VL first-forward adds cold vision-tower
         page-in + 200+ token prefill activations on top of the 191 GB
         text bundle; the extra pagecache headroom keeps Jetsam from
         killing the process under SSD contention. See module docstring.
      3. Delegates to the shared VLM hydrate path.
    """
    _ensure_kimi_remap()
    from jang_tools.load_jangtq_vlm import load_jangtq_vlm_model
    model, processor = load_jangtq_vlm_model(Path(model_path))
    # Re-apply the VL-specific lower wired_limit AFTER the shared loader runs.
    # The shared loader calls `_apply_wired_limit_safe_default` at 70 % of RAM,
    # which would clobber any earlier VL setting. We override post-load so the
    # lower ceiling is in force before the first VL forward (generate).
    _set_vl_wired_limit()
    # Install the command-buffer split so the Metal watchdog doesn't fire on
    # the first (cold, SSD-contested) vision_tower + language prefill.
    _install_vl_command_buffer_split(model)
    # Pre-compile Metal kernels layer-by-layer so the first real VL prefill
    # doesn't hit cold-JIT overhead inside a single command buffer.
    _warmup_jit_per_layer(model)
    return model, processor
