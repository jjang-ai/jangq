"""Capture per-layer routed-expert Hessians (H = XᵀX) for GPTQ on Ornith 1.5.

Created by Jinho Jang (eric@osaurus.ai) — 2026-08-19.

GPTQ minimizes ``||(deq(codes) - W) Xᵀ||²``, which needs the FULL ``H = XᵀX``,
not the diagonal. `ornith_moe_calibrate` captures only the per-channel second
moment ``E[x_c²]`` — enough for Hessian-trace allocation, the imatrix fit and
AWQ scales, but NOT for GPTQ's error-compensated rounding. That is why GPTQ
needs this second pass rather than a flag on the first.

Ported from `capture_gemma4_hessians` with three qwen3_5_moe changes:

  1. layers live at ``model.language_model.model.layers`` (mlx_vlm VL wrapper),
     not ``model.model.layers``;
  2. the MoE block is ``layer.mlp.switch_mlp`` (a `SwitchGLU` directly), not
     Gemma's ``layer.experts.switch_glu``;
  3. 🚨 the `SwitchGLU` class must come from **mlx_vlm**, not mlx_lm —
     ``mlx_vlm.models.switch_layers.SwitchGLU is not
     mlx_lm.models.switch_layers.SwitchGLU``. Patching the wrong one leaves
     every Hessian empty while the run reports success. Both are patched.

TWO Hessians per layer, because two different activations feed the three expert
projections:

    post_attention_layernorm(h)   -> gate_proj, up_proj   (hidden 2048)
    SwiGLU(x_up, x_gate)          -> down_proj            (moe_inter 512)

The down_proj input is taken by swapping ``glu.activation`` for a tap object —
this sits exactly where the SwiGLU does, so it observes the real input without
re-deriving SwitchGLU's internals (`expand_dims`, `_gather_sort`, and the
two-arg SwiGLU are all easy to get wrong; see `ornith_moe_calibrate`).

All experts in a layer share one H, matching `gptq_mlx`: per-expert H would be
256x the memory for a second-order refinement of an already second-order
method, and at top-8 routing most experts see too few tokens for a stable
estimate.

Accumulation is float64 throughout — the DSV4 lesson is that f32 inversion of a
rank-deficient H silently falls back to RTN, i.e. you get no GPTQ at all and no
error saying so.

Memory: 2048² f64 = 33.5 MiB per layer for the `in` Hessian, 512² = 2 MiB for
`mid`; ~1.4 GiB across 40 layers.

    python -m jang_tools.ornith_moe_hessians --model <src> --out <dir> \
        [--max-tokens 96] [--images <dir>] [--limit N]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


class _Accum:
    """Running XᵀX in float64, plus a token counter for conditioning checks."""

    def __init__(self) -> None:
        self.H: np.ndarray | None = None
        self.n: int = 0

    def add(self, x: mx.array) -> None:
        a = np.asarray(x.astype(mx.float32)).reshape(-1, x.shape[-1]).astype(np.float64)
        if a.shape[0] == 0:
            return
        if self.H is None:
            self.H = np.zeros((a.shape[1], a.shape[1]), dtype=np.float64)
        self.H += a.T @ a
        self.n += a.shape[0]


def _switch_glu_classes():
    import importlib
    out = []
    for m in ("mlx_vlm.models.switch_layers", "mlx_lm.models.switch_layers"):
        try:
            c = getattr(importlib.import_module(m), "SwitchGLU", None)
            if c is not None and c not in out:
                out.append(c)
        except Exception:
            pass
    return out


def _layers(model):
    lm = getattr(model, "language_model", model)
    inner = getattr(lm, "model", lm)
    return getattr(inner, "layers", [])


def _patch(model) -> dict[tuple[int, str], _Accum]:
    classes = _switch_glu_classes()
    if not classes:
        raise RuntimeError("no SwitchGLU class importable")

    acc: dict[tuple[int, str], _Accum] = {}
    registry: dict[int, _Accum] = {}

    n_patched = 0
    for idx, layer in enumerate(_layers(model)):
        mlp = getattr(layer, "mlp", None)
        glu = getattr(mlp, "switch_mlp", None) if mlp is not None else None
        if glu is None or not isinstance(glu, tuple(classes)):
            continue

        a_in, a_mid = _Accum(), _Accum()
        acc[(idx, "in")] = a_in
        acc[(idx, "mid")] = a_mid
        registry[id(glu)] = a_in

        class _TapAct:
            """Sits exactly where the SwiGLU does, so it sees down_proj's input."""

            def __init__(self, inner, sink):
                self._inner = inner
                self._sink = sink

            def __call__(self, x_up, x_gate):
                out = self._inner(x_up, x_gate)
                self._sink.add(out)
                return out

        glu.activation = _TapAct(glu.activation, a_mid)
        n_patched += 1

    for cls in classes:
        if getattr(cls, "_ornith_hessian_tapped", False):
            continue
        orig = cls.__call__

        def make(_orig=orig):
            def tapped(self, x, indices, *a, **k):
                sink = registry.get(id(self))
                if sink is not None:
                    sink.add(x)
                return _orig(self, x, indices, *a, **k)
            return tapped

        cls.__call__ = make()
        cls._ornith_hessian_tapped = True

    print(f"  patched {n_patched} MoE layers "
          f"({len(classes)} SwitchGLU class(es))", flush=True)
    if n_patched == 0:
        raise RuntimeError(
            "no MoE layers patched — layer.mlp.switch_mlp not found. Refusing "
            "rather than writing empty Hessians.")
    return acc


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--images", type=Path, default=None)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()

    from mlx_vlm import load, generate
    from . import qwen36_calibrate as Q

    print(f"  loading {a.model.name} ...", flush=True)
    t0 = time.time()
    model, proc = load(str(a.model))
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    acc = _patch(model)
    corpus = Q.build_corpus(proc.tokenizer, a.limit or None)
    print(f"  corpus: {len(corpus)} prompts x {a.max_tokens} tok", flush=True)

    t0 = time.time()
    for i, prompt in enumerate(corpus, 1):
        generate(model, proc, prompt, max_tokens=a.max_tokens,
                 temperature=1.0, verbose=False)
        if i % 3 == 0 or i == len(corpus):
            seen = sum(1 for v in acc.values() if v.H is not None)
            print(f"    text {i}/{len(corpus)}  ({time.time()-t0:.0f}s, "
                  f"{seen}/{len(acc)} Hessians live)", flush=True)

    if a.images and a.images.is_dir():
        for i, (prompt, img) in enumerate(Q.build_image_corpus(proc, model, a.images), 1):
            generate(model, proc, prompt, image=[img], max_tokens=a.max_tokens,
                     temperature=1.0, verbose=False)
            print(f"    image {i} {Path(img).name} ({time.time()-t0:.0f}s)", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    empty = [k for k, v in acc.items() if v.H is None]
    if empty:
        print(f"  !! {len(empty)} Hessians never received a sample: {empty[:6]} "
              f"— REFUSING to write a partial capture")
        return 2

    for (idx, which), v in sorted(acc.items()):
        f = a.out / f"H_L{idx:02d}_{which}_d{v.H.shape[0]}.npy"
        np.save(f, v.H)
    tot = sum(v.H.nbytes for v in acc.values())
    print(f"\n  wrote {len(acc)} Hessians ({tot/2**20:.0f} MiB) -> {a.out}")
    print(f"  rows accumulated: {sum(v.n for v in acc.values()):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
