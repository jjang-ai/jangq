"""Full-covariance Hessian capture for GPTQ on Ling-3.0 routed experts.

Created by Jinho Jang (eric@jangq.ai) — 2026-08-26.

The main calibration pass stored only per-channel DIAGONAL second moments —
right for imatrix / trace allocation / AWQ, insufficient for GPTQ, whose error
feedback needs the full input covariance `H = sum x x^T` per fold group.

Scope is experts-only and per-layer-shared, deliberately:

  * Only `switch_mlp.{gate,up,down}_proj` inputs are captured (2 distinct
    activations per MoE layer — gate/up share one, down has its own), i.e.
    46 matrices of 1536^2 + 23 of 512^2, fp64 ≈ 460 MB. Attention/shared-expert
    tensors sit at 8-bit where GPTQ buys nothing measurable.
  * All 128 experts of a layer share one H (same input distribution; per-expert
    H at top-8 routing would be token-starved — the standing rank rule).

    python -m jang_tools.ling3.gptq_capture <model_dir> <out.npz> \
        [--corpus PATH] [--target-tokens N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from jang_tools.ling3.calibrate import build_corpus
from jang_tools.ling3.load import load_model

_H: dict[str, np.ndarray] = {}
_N: dict[str, int] = {}


def _accum(path: str, x: mx.array) -> None:
    xf = x.reshape(-1, x.shape[-1]).astype(mx.float32)
    h = xf.T @ xf                      # [d, d] on GPU
    mx.eval(h)
    hv = np.array(h, dtype=np.float64)
    if path in _H:
        _H[path] += hv
    else:
        _H[path] = hv
    _N[path] = _N.get(path, 0) + int(xf.shape[0])


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="jang_tools.ling3.gptq_capture")
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument(
        "--corpus",
        default=str(Path.home() / ".cache" / "jang" / "corpus_v3.jsonl"),
    )
    ap.add_argument("--target-tokens", type=int, default=400_000,
                    help="H is d_in^2-parameterized; 400k tokens >> the 2x rank floor")
    ap.add_argument("--max-prompt-tokens", type=int, default=2048)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer
    from mlx_lm.models.switch_layers import SwitchLinear

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = load_model(args.model_dir)

    # tap SwitchLinear only — experts are the entire GPTQ scope
    targets: dict[int, str] = {}
    for path, mod in model.named_modules():
        if isinstance(mod, SwitchLinear):
            targets[id(mod)] = path
    if not targets:
        raise SystemExit("refusing to run: no SwitchLinear found")

    orig = SwitchLinear.__call__

    def patched(self, x, *a, **k):
        p = targets.get(id(self))
        if p is not None:
            try:
                _accum(p, x)
            except Exception:
                pass
        return orig(self, x, *a, **k)

    SwitchLinear.__call__ = patched

    items, total, spent = build_corpus(
        tok, Path(args.corpus), args.target_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    print(f"[corpus] {len(items)} prompts ~{total} tokens", flush=True)

    t0 = time.time()
    seen = 0
    for i, text in enumerate(items):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        out = model(mx.array([ids]))
        mx.eval(out)
        seen += len(ids)
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"[{i+1}/{len(items)}] {seen} tok {el:.0f}s {seen/max(el,1):.0f} tok/s",
                  flush=True)

    SwitchLinear.__call__ = orig

    if not _H:
        raise SystemExit("refusing to write: captured nothing")

    # sanity: H must be far from singular for every group
    worst = None
    for p, h in _H.items():
        d = h.shape[0]
        rank_ratio = _N[p] / d
        if worst is None or rank_ratio < worst[1]:
            worst = (p, rank_ratio)
    print(f"[rank] worst rows/d_in = {worst[1]:.1f}x at {worst[0]}")
    if worst[1] < 2.0:
        raise SystemExit("refusing to write: shared-H rank margin < 2x — raise tokens")

    np.savez_compressed(
        args.out,
        **{p: h for p, h in _H.items()},
        __counts__=np.array(list(_N.values())),
        __paths__=np.array(list(_N.keys())),
    )
    print(f"[done] {len(_H)} H matrices, {seen} tokens -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
