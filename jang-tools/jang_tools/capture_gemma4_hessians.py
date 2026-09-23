"""Capture per-layer routed-expert Hessians for GPTQ on Gemma 4 MoE.

GPTQ minimizes ``||(deq(codes) - W) X^T||^2``, which needs ``H = X^T X`` over the
activations that actually feed each weight. A Gemma 4 MoE layer needs **two**
Hessians, because two different activations feed the three expert projections:

    pre_feedforward_layernorm_2(h)  --> gate_proj, up_proj   (hidden_size)
    GeGLU(up_out, gate_out)         --> down_proj            (moe_intermediate)

Capturing only the first is the mistake the 397B AWQ work documented
("Skips experts.down_proj automatically -- its input is intermediate-dim, not in
the capture"). Here both are captured, so GPTQ covers all three projections.

All experts within a layer share one H, matching ``gptq_mlx`` -- computing H per
expert would be 128x the memory for a second-order refinement of an
already-second-order method.

Accumulation is float64 throughout. The DSV4 lesson recorded in the QAT-AWQ
pipeline doc is that f32 inversion of a rank-deficient H silently falls back to
RTN, i.e. you get no GPTQ at all and no error telling you so.

Output: ``H_L{layer}_d{in_features}.npy`` -- the naming the converter's
``_load_hinv`` resolves against.

Usage:
    python -m jang_tools.capture_gemma4_hessians \
        --model  <JANG bundle or bf16 source> \
        --prompts calib_prompts.jsonl \
        --out    hessians/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


class _Accum:
    """Running X^T X in float64, plus a token counter for conditioning checks."""

    def __init__(self) -> None:
        self.H: np.ndarray | None = None
        self.n: int = 0

    def add(self, x: mx.array) -> None:
        # x arrives with the SwitchGLU broadcast dims still attached; flatten
        # everything except the feature axis.
        a = np.asarray(x.astype(mx.float32)).reshape(-1, x.shape[-1]).astype(np.float64)
        if a.shape[0] == 0:
            return
        if self.H is None:
            self.H = np.zeros((a.shape[1], a.shape[1]), dtype=np.float64)
        self.H += a.T @ a
        self.n += a.shape[0]


def _patch(model) -> dict[tuple[int, str], _Accum]:
    """Wrap each layer's SwitchGLU to record both expert-input activations.

    The gate_proj/up_proj input is taken by patching ``SwitchGLU.__call__`` on the
    CLASS, not the instance: ``glu(x, idx)`` resolves through ``type(glu).__call__``,
    so an instance attribute would be silently ignored and every Hessian would come
    back empty. Instances are told apart by an id-keyed registry.

    The down_proj input is taken by swapping ``activation`` for a tap object. That
    one works per-instance because the tap is a distinct object whose own class
    defines ``__call__``.
    """
    from mlx_lm.models.switch_layers import SwitchGLU

    acc: dict[tuple[int, str], _Accum] = {}
    registry: dict[int, _Accum] = {}
    layers = model.model.layers if hasattr(model, "model") else model.layers

    n_patched = 0
    for idx, layer in enumerate(layers):
        experts = getattr(layer, "experts", None)
        glu = getattr(experts, "switch_glu", None) if experts is not None else None
        if glu is None:
            continue

        a_in, a_mid = _Accum(), _Accum()
        acc[(idx, "in")] = a_in
        acc[(idx, "mid")] = a_mid
        registry[id(glu)] = a_in

        class _TapAct:
            """Sits exactly where the GeGLU does, so it sees down_proj's input."""

            def __init__(self, inner, sink):
                self._inner = inner
                self._sink = sink

            def __call__(self, x_up, x_gate):
                out = self._inner(x_up, x_gate)
                self._sink.add(out)
                return out

        glu.activation = _TapAct(glu.activation, a_mid)
        n_patched += 1

    if n_patched and not getattr(SwitchGLU, "_hessian_tapped", False):
        orig_call = SwitchGLU.__call__

        def tapped(self, x, indices, _orig=orig_call, _reg=registry):
            sink = _reg.get(id(self))
            if sink is not None:
                sink.add(x)
            return _orig(self, x, indices)

        SwitchGLU.__call__ = tapped
        SwitchGLU._hessian_tapped = True

    print(f"  patched {n_patched} MoE layers")
    return acc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=1024,
                   help="Truncate each calibration sequence to this many tokens.")
    p.add_argument("--limit", type=int, default=0, help="Only use the first N prompts.")
    args = p.parse_args()

    from mlx_lm import load

    print(f"  loading {args.model}")
    t0 = time.time()
    model, tokenizer = load(str(args.model))
    print(f"  loaded in {time.time() - t0:.1f}s")

    rows = [json.loads(l) for l in args.prompts.read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"  {len(rows)} calibration rows")

    acc = _patch(model)
    if not acc:
        raise SystemExit("no MoE layers found -- wrong model or changed structure")

    tok_total = 0
    t0 = time.time()
    for i, row in enumerate(rows):
        text = row.get("text") or row.get("prompt") or ""
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)[: args.max_tokens]
        if len(ids) < 8:
            continue
        # Forward only. No sampling: the Hessian wants the teacher-forced
        # distribution over real text, not the model's own rollout.
        model(mx.array([ids]))
        mx.eval(mx.zeros(1))
        tok_total += len(ids)
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{len(rows)}  {tok_total} tokens  {time.time() - t0:.0f}s")

    args.out.mkdir(parents=True, exist_ok=True)
    written, warned = 0, 0
    for (layer, kind), a in sorted(acc.items()):
        if a.H is None:
            continue
        dim = a.H.shape[0]
        # A Hessian needs more rows than columns to be full-rank. Fewer tokens
        # than features means the damping term is doing all the work and GPTQ
        # degenerates toward RTN -- worth surfacing rather than discovering later.
        if a.n < dim:
            print(f"    ! L{layer} {kind}: {a.n} tokens < {dim} dims (rank-deficient)")
            warned += 1
        np.save(str(args.out / f"H_L{layer}_d{dim}.npy"), a.H)
        written += 1

    print(f"\n  wrote {written} Hessians to {args.out}")
    print(f"  total tokens: {tok_total}")
    if warned:
        print(f"  WARNING: {warned} rank-deficient -- add more/longer prompts")


if __name__ == "__main__":
    main()
