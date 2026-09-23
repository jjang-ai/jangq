"""Calibration capture for Ling-3.0 (`bailing_hybrid` / BailingMoeV3).

Created by Jinho Jang (eric@jangq.ai) — 2026-08-26.

One forward sweep produces the statistic that feeds all three consumers
(`feedback_calibrated_trio_mandatory`):

  * **imatrix**        — per-channel mean square of each linear's input
  * **Hessian trace**  — `tr(H)` = the sum of that same per-channel 2nd moment
  * **AWQ scales**     — derived from the same moments

Emitted paths mirror the SOURCE tensor names so the allocator and refit resolve
them directly.

Two Ling-3.0-specific hazards this handles:

1. **The routed experts are a `SwitchGLU`, not `nn.Linear`.** Patching only
   `nn.Linear` collects *zero* statistics for 128 experts x 23 layers — the bulk
   of the model — while appearing to succeed. This is the documented Ornith-1.5
   trap ("FOUR tools silently skipped the MoE experts"), so `SwitchLinear` is
   tapped explicitly and the run **fails closed** if the expert tap sees nothing.

2. **Two distinct activations per MoE layer.** `gate_proj`/`up_proj` consume
   `post_attention_layernorm(h)` (width 1536); `down_proj` consumes the SwiGLU
   product (width 512). Capturing only the first silently skips `down_proj` —
   the documented 397B AWQ mistake. Both are captured because the tap sits on the
   projections themselves rather than on the block input.

The KDA gate/conv params and the router are **not** capture targets: they are
fp16/fp32 keeps (see the plan doc), so a statistic for them would be unused.

    python -m jang_tools.ling3.calibrate <model_dir> <out.safetensors> \
        [--corpus PATH] [--target-tokens N] [--max-prompt-tokens N] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.switch_layers import SwitchLinear

from jang_tools.ling3.load import load_model

_TARGETS: dict[int, str] = {}
_SUMSQ: dict[str, np.ndarray] = {}
_COUNT: dict[str, int] = {}
_PATCHED: list[tuple[type, object]] = []


def _accumulate(path: str, x: mx.array) -> None:
    xf = x.reshape(-1, x.shape[-1])
    s = (xf.astype(mx.float32) ** 2).sum(axis=0)
    mx.eval(s)
    v = np.array(s, dtype=np.float64)
    if path in _SUMSQ:
        _SUMSQ[path] += v
    else:
        _SUMSQ[path] = v
    _COUNT[path] = _COUNT.get(path, 0) + int(xf.shape[0])


def install_hooks(model) -> tuple[int, int]:
    """Patch `Linear` and `SwitchLinear` at class level.

    Returns ``(n_dense, n_switch)`` so the caller can assert the expert tap is live.
    """
    classes: list[type] = [nn.Linear, SwitchLinear]
    q = getattr(nn, "QuantizedLinear", None)
    if q is not None:
        classes.append(q)

    for cls in classes:
        orig = cls.__call__

        def make(orig=orig):
            def patched(self, x, *a, **k):
                p = _TARGETS.get(id(self))
                if p is not None:
                    try:
                        _accumulate(p, x)
                    except Exception:      # never let capture break the forward
                        pass
                return orig(self, x, *a, **k)
            return patched

        _PATCHED.append((cls, orig))
        cls.__call__ = make()

    n_dense = n_switch = 0
    for path, mod in model.named_modules():
        if isinstance(mod, SwitchLinear):
            _TARGETS[id(mod)] = path
            n_switch += 1
        elif isinstance(mod, tuple(c for c in classes if c is not SwitchLinear)):
            _TARGETS[id(mod)] = path
            n_dense += 1
    return n_dense, n_switch


def remove_hooks() -> None:
    for cls, orig in _PATCHED:
        cls.__call__ = orig
    _PATCHED.clear()
    _TARGETS.clear()


# The v4 mix (docs/runtime/CALIBRATION-MIX-v4-2026-08-22.md): coding-heaviest,
# matching what this model is actually for.
CORPUS_MIX = {
    "coding": 0.35, "agentic": 0.20, "academic_mc": 0.15, "general": 0.12,
    "chinese": 0.10, "longctx": 0.04, "science": 0.02, "cybersec": 0.02,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Execute a read-only SQL query.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]


def build_corpus(
    tokenizer,
    corpus_path: Path,
    target_tokens: int,
    mix: dict[str, float] | None = None,
    max_prompt_tokens: int = 2048,
) -> tuple[list[str], int, dict[str, int]]:
    """Draw ~`target_tokens` of domain-weighted text, rendered through the REAL template.

    Rendering matters as much as volume: the model ships always-thinking with an
    XML tool frame, so a corpus of bare completions would calibrate for a
    distribution the model is never used in.

    A **tool-framed slice** and a **non-thinking slice** are mixed in so both
    presets are represented — the tool frame is what the XML-arg parser will meet
    in production.

    Records are interleaved round-robin across domains, so a truncated run still
    carries the whole mix rather than only the first domain.
    """
    mix = mix or CORPUS_MIX
    by_domain: dict[str, list[str]] = {}
    with corpus_path.open() as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (rec.get("text") or "").strip()
            if text:
                by_domain.setdefault(rec.get("domain", "general"), []).append(text)

    missing = [d for d in mix if d not in by_domain]
    if missing:
        print(f"[warn] corpus has no records for domains: {missing}", file=sys.stderr)

    budgets = {d: int(target_tokens * w) for d, w in mix.items() if d in by_domain}
    queues = {d: iter(by_domain[d]) for d in budgets}
    spent = {d: 0 for d in budgets}
    items: list[str] = []
    total = 0
    n = 0
    while queues:
        for d in list(queues):
            if spent[d] >= budgets[d]:
                queues.pop(d)
                continue
            try:
                text = next(queues[d])
            except StopIteration:
                queues.pop(d)
                continue
            ids = tokenizer.encode(text)
            if len(ids) > max_prompt_tokens:
                text = tokenizer.decode(ids[:max_prompt_tokens])
                ntok = max_prompt_tokens
            else:
                ntok = len(ids)

            # rotate the rendering preset so every slice is represented
            kw: dict = {"add_generation_prompt": True, "tokenize": False}
            if n % 10 == 0:
                kw["tools"] = TOOLS
            if n % 7 == 0:
                kw["enable_thinking"] = False
            items.append(
                tokenizer.apply_chat_template([{"role": "user", "content": text}], **kw)
            )
            spent[d] += ntok
            total += ntok
            n += 1
    return items, total, spent


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="jang_tools.ling3.calibrate")
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument(
        "--corpus",
        default=str(Path.home() / ".cache" / "jang" / "corpus_v3.jsonl"),
    )
    ap.add_argument(
        "--target-tokens",
        type=int,
        default=1_200_000,
        help="shared-H rank is tokens/24576 here; 1.2M keeps it off the singular floor",
    )
    ap.add_argument("--max-prompt-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = load_model(args.model_dir)

    n_dense, n_switch = install_hooks(model)
    print(f"[hooks] dense={n_dense} switch={n_switch}", flush=True)
    if n_switch == 0:
        remove_hooks()
        raise SystemExit(
            "refusing to run: no SwitchLinear tapped — the routed experts would be "
            "silently skipped (the Ornith-1.5 trap)"
        )

    items, total, spent = build_corpus(
        tok, Path(args.corpus), args.target_tokens, max_prompt_tokens=args.max_prompt_tokens
    )
    if args.limit:
        items = items[: args.limit]
    print(f"[corpus] {len(items)} prompts, ~{total} tokens, per-domain={spent}", flush=True)

    t0 = time.time()
    seen = 0
    for i, text in enumerate(items):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        out = model(mx.array([ids]))
        mx.eval(out)
        seen += len(ids)
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(
                f"[{i+1}/{len(items)}] {seen} tok  {el:.0f}s  {seen/max(el,1e-9):.0f} tok/s",
                flush=True,
            )

    remove_hooks()

    if not _SUMSQ:
        raise SystemExit("refusing to write: captured nothing")

    # mean square per input channel = tr(H) contribution, normalized by token count
    out_arrays: dict[str, mx.array] = {}
    for path, sumsq in _SUMSQ.items():
        cnt = max(_COUNT.get(path, 0), 1)
        out_arrays[path] = mx.array((sumsq / cnt).astype(np.float32))

    n_expert_paths = sum(1 for p in out_arrays if "switch_mlp" in p)
    if n_expert_paths == 0:
        raise SystemExit("refusing to write: no expert statistics captured")

    meta = {
        "model": str(args.model_dir),
        "model_type": "bailing_hybrid",
        "tokens": str(seen),
        "prompts": str(len(items)),
        "corpus": str(args.corpus),
        "mix": json.dumps(CORPUS_MIX),
        "expert_paths": str(n_expert_paths),
        "created_by": "jang_tools.ling3.calibrate",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.out), out_arrays, metadata=meta)
    print(
        f"[done] {len(out_arrays)} paths ({n_expert_paths} expert), {seen} tokens "
        f"-> {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
