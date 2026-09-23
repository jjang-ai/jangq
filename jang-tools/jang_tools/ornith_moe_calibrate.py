"""Calibration capture for Ornith-1.5-35B-A3B (`qwen3_5_moe`) — MoE-aware.

Created by Jinho Jang (eric@jangq.ai) — 2026-08-19.

`qwen36_calibrate` patches **`nn.Linear` / `nn.QuantizedLinear` only**. In
mlx_vlm's `qwen3_5_moe` the routed experts are a `SwitchGLU`
(`layers.N.mlp.switch_mlp`) whose three projections are `SwitchLinear`, which
is NOT an `nn.Linear`. Running the dense capture unchanged on the 35B collects
**zero** statistics for the routed experts — **33.02 B of 35.95 B params
(91.8 %)** — while appearing to succeed, so the Hessian trace, the imatrix
refit and the AWQ scales would all silently operate on 8 % of the model.

This module reuses everything in `qwen36_calibrate` (accumulator, corpus,
emit format) and adds the `SwitchLinear` tap, so the output is byte-compatible
with `qwen36_allocate` and `qwen36_imatrix_refit` — no downstream change.

TWO activations per MoE layer, not one. Two different inputs feed the three
expert projections:

    post_attention_layernorm(h)   -> gate_proj, up_proj   (hidden_size 2048)
    activation(up_out, gate_out)  -> down_proj            (moe_inter    512)

Capturing only the first is the documented 397B AWQ mistake — it "skips
experts.down_proj automatically, its input is intermediate-dim, not in the
capture" (`capture_gemma4_hessians`). Here both are captured.

All experts in a layer share one statistic, matching `gptq_mlx`. Per-expert
capture would be 256x the memory for a second-order refinement of an already
second-order method, and at top-8 routing most experts would see too few
tokens for a stable estimate anyway.

The emitted paths deliberately mirror the SOURCE tensor names so the allocator
and refit resolve them:

    language_model.model.layers.N.mlp.switch_mlp.gate_proj
    language_model.model.layers.N.mlp.switch_mlp.up_proj
    language_model.model.layers.N.mlp.switch_mlp.down_proj

    python -m jang_tools.ornith_moe_calibrate <model_dir> <out.safetensors> \
        [--limit N] [--max-tokens N] [--images dir]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from jang_tools import qwen36_calibrate as Q

_SWITCH_PATCHED: list = []


def _tap_switch_glu(model) -> int:
    """Patch ``SwitchLinear.__call__`` so each expert projection records its
    OWN true input. Returns the number of SwitchLinear modules registered.

    Tapping `SwitchLinear` rather than `SwitchGLU` is deliberate. Reconstructing
    the down_proj input from outside would mean re-deriving SwitchGLU's
    internals, and all three details are easy to get wrong:

      * `x = mx.expand_dims(x, (-2, -3))` is applied unconditionally;
      * when `indices.size >= 64` the input is `_gather_sort`ed and passed with
        `sorted_indices=True` — calling `gather_mm` with sorted_indices but the
        unsorted layout is the documented silent-garbage trap;
      * `activation` is `SwiGLU()`, taking **two** args `(x_up, x_gate)` in that
        order, not a unary gate times up.

    Tapping the projections themselves sidesteps all three and costs no extra
    expert compute. The captured statistic is a per-input-channel second moment
    (a sum of squares over rows), which is order-invariant — so the gather-sort
    applied upstream does not affect it.
    """
    # 🚨 mlx_vlm and mlx_lm each define their OWN SwitchLinear — same name,
    # DIFFERENT class objects (`mlx_vlm.models.switch_layers.SwitchLinear is
    # not mlx_lm.models.switch_layers.SwitchLinear`). mlx_vlm's qwen3_5_moe
    # does `from ..switch_layers import SwitchGLU`, i.e. its own. Importing
    # only the mlx_lm class makes every isinstance() check fail and the capture
    # silently covers ZERO experts. Patch whichever are importable.
    classes = []
    for modname in ("mlx_vlm.models.switch_layers", "mlx_lm.models.switch_layers"):
        try:
            import importlib
            cls = getattr(importlib.import_module(modname), "SwitchLinear", None)
            if cls is not None and cls not in classes:
                classes.append(cls)
        except Exception:
            pass
    if not classes:
        return 0

    # Map id(switch_linear) -> dotted path, e.g.
    #   language_model.model.layers.3.mlp.switch_mlp.down_proj
    targets: dict[int, str] = {}
    for path, mod in model.named_modules():
        if isinstance(mod, tuple(classes)):
            targets[id(mod)] = path

    if not targets:
        return 0

    for SwitchLinear in classes:
        if getattr(SwitchLinear, "_ornith_tapped", False):
            continue
        orig = SwitchLinear.__call__

        def make(orig=orig):
            def tapped(self, x, indices, *a, **k):
                p = targets.get(id(self))
                if p is not None:
                    try:
                        Q._accumulate(p, x)
                    except Exception:  # never let capture break the forward
                        pass
                return orig(self, x, indices, *a, **k)
            return tapped

        SwitchLinear.__call__ = make()
        SwitchLinear._ornith_tapped = True
        _SWITCH_PATCHED.append((SwitchLinear, orig))

    return len(targets)


def _untap() -> None:
    for cls, orig in _SWITCH_PATCHED:
        cls.__call__ = orig
        cls._ornith_tapped = False
    _SWITCH_PATCHED.clear()


def main(argv) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 1
    src, out = Path(argv[1]), Path(argv[2])
    limit, max_tokens, image_dir = None, 96, None
    for i, a in enumerate(argv):
        if a == "--limit":
            limit = int(argv[i + 1])
        if a == "--max-tokens":
            max_tokens = int(argv[i + 1])
        if a == "--images":
            image_dir = Path(argv[i + 1])

    from mlx_vlm import load, generate

    print(f"  loading {src.name} ...", flush=True)
    t0 = time.time()
    model, proc = load(str(src))
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    n_lin = Q.install_hooks(model, include_vision=True)
    n_moe = _tap_switch_glu(model)
    print(f"  hooked {n_lin} Linear modules + {n_moe} SwitchLinear expert "
          f"projections", flush=True)
    if n_moe == 0:
        print("  !! no SwitchLinear found — this is not a MoE bundle; "
              "use qwen36_calibrate instead", flush=True)
        return 2

    corpus = Q.build_corpus(proc.tokenizer, limit)
    print(f"  corpus: {len(corpus)} prompts, {max_tokens} tok each", flush=True)

    t0 = time.time()
    for i, prompt in enumerate(corpus, 1):
        generate(model, proc, prompt, max_tokens=max_tokens,
                 temperature=1.0, verbose=False)
        if i % 3 == 0 or i == len(corpus):
            print(f"    text {i}/{len(corpus)}  ({time.time()-t0:.0f}s, "
                  f"{len(Q._SUMSQ)} modules seen)", flush=True)

    if image_dir and image_dir.is_dir():
        img_corpus = Q.build_image_corpus(proc, model, image_dir)
        print(f"  vision: {len(img_corpus)} images", flush=True)
        for i, (prompt, img) in enumerate(img_corpus, 1):
            generate(model, proc, prompt, image=[img], max_tokens=max_tokens,
                     temperature=1.0, verbose=False)
            print(f"    image {i}/{len(img_corpus)} {Path(img).name} "
                  f"({time.time()-t0:.0f}s, {len(Q._SUMSQ)} modules seen)",
                  flush=True)

    Q.remove_hooks()
    _untap()

    tensors, meta = {}, {}
    for path, ssq in Q._SUMSQ.items():
        cnt = max(Q._COUNT[path], 1)
        sm = (ssq / cnt).astype(np.float32)
        tensors[f"{path}.second_moment"] = sm
        meta[path] = {"count": cnt, "trace": float(sm.sum()),
                      "in_features": int(sm.shape[0])}

    from safetensors.numpy import save_file
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out))
    out.with_suffix(".json").write_text(json.dumps(
        {"source": str(src), "modules": len(meta), "prompts": len(corpus),
         "max_tokens": max_tokens, "moe_blocks": n_moe, "stats": meta}, indent=1))

    n_exp = sum(1 for p in meta if "switch_mlp" in p)
    print(f"\n  captured {len(meta)} modules ({n_exp} routed-expert) -> {out}")
    print(f"  sidecar  -> {out.with_suffix('.json')}")
    print(f"  total row-samples: {sum(v['count'] for v in meta.values()):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
