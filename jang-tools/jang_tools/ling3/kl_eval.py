"""KL / top-1 evaluation of a Ling-3.0 JANG bundle against the bf16 source.

Created by Jinho Jang (eric@jangq.ai) — 2026-08-26.

Reports, on **held-out** prompts:

  * mean KL(P_bf16 || Q_quant) per token
  * top-1 agreement
  * the **margin-conditioned flip curve** — flips binned by the SOURCE's
    `top1 - top2` margin.

The flip curve is the part that distinguishes "lossy" from "broken", and it is
the reason a bare KL number is not enough. A healthy quant flips mostly where the
source was itself undecided, so the flip rate must fall **monotonically** as the
source margin grows. Flat means noise; **rising means a structural defect**.
Do not approximate it with `margin < mean_KL` — that shortcut previously cried
structural on a perfectly healthy bundle.

Held-out discipline: the eval corpus must come from a different file than the
calibration corpus. `qwen36_kl_eval` was 33 % contaminated with calibration
prompts, which flatters every number it produced.

    python -m jang_tools.ling3.kl_eval <src_bf16> <bundle> \
        [--corpus PATH] [--n 64] [--max-tokens 512]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from jang_tools.ling3.load import load_config, load_weights
from jang_tools.ling3.model import Model

MARGIN_BINS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, np.inf]


def load_quantized(bundle: Path) -> Model:
    """Rebuild the module tree, re-quantize it to the bundle's per-tensor plan, load."""
    cfg = json.loads((bundle / "config.json").read_text())
    args = load_config(bundle)
    model = Model(args)

    plan = cfg["quantization"].get("per_tensor", {})
    gs = cfg["quantization"]["group_size"]

    def predicate(path: str, module) -> bool | dict:
        spec = plan.get(path)
        if spec is None:
            return False
        out = {"group_size": spec["group_size"], "bits": spec["bits"]}
        if spec.get("mode", "affine") != "affine":
            out["mode"] = spec["mode"]
        return out

    nn.quantize(
        model,
        group_size=gs,
        bits=cfg["quantization"].get("bits", 8),
        mode=cfg["quantization"].get("mode", "affine"),
        class_predicate=predicate,
    )

    weights = load_weights(bundle)
    expected = dict(tree_flatten(model.parameters()))
    missing = sorted(set(expected) - set(weights))
    unexpected = sorted(set(weights) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"bundle/module mismatch\n  missing({len(missing)}): {missing[:6]}\n"
            f"  unexpected({len(unexpected)}): {unexpected[:6]}"
        )
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    model.eval()
    return model


def build_holdout(tokenizer, corpus: Path, n: int, max_tokens: int) -> list[list[int]]:
    """Round-robin across domains so the eval set carries the whole distribution."""
    by_domain: dict[str, list[str]] = {}
    with corpus.open() as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = (rec.get("text") or "").strip()
            if t:
                by_domain.setdefault(rec.get("domain", "general"), []).append(t)

    out: list[list[int]] = []
    queues = {d: iter(v) for d, v in by_domain.items()}
    while queues and len(out) < n:
        for d in list(queues):
            if len(out) >= n:
                break
            try:
                text = next(queues[d])
            except StopIteration:
                queues.pop(d)
                continue
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True, tokenize=False,
            )
            ids = tokenizer(rendered, add_special_tokens=False)["input_ids"][:max_tokens]
            if len(ids) >= 16:
                out.append(ids)
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="jang_tools.ling3.kl_eval")
    ap.add_argument("src")
    ap.add_argument("bundle")
    ap.add_argument(
        "--corpus",
        default=str(Path.home() / ".cache" / "jang" / "corpus_v2.jsonl"),
        help="MUST differ from the calibration corpus",
    )
    ap.add_argument(
        "--calib-corpus",
        default=str(Path.home() / ".cache" / "jang" / "corpus_v3.jsonl"),
    )
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args(argv)

    if Path(args.corpus).resolve() == Path(args.calib_corpus).resolve():
        raise SystemExit("refusing to run: eval corpus is the calibration corpus")

    from transformers import AutoTokenizer
    from jang_tools.ling3.load import load_model

    tok = AutoTokenizer.from_pretrained(args.src, trust_remote_code=True)
    prompts = build_holdout(tok, Path(args.corpus), args.n, args.max_tokens)
    print(f"[holdout] {len(prompts)} prompts from {args.corpus}", flush=True)

    src_model = load_model(args.src)
    tot_kl = 0.0
    tot_tok = 0
    agree = 0
    bin_tot = np.zeros(len(MARGIN_BINS) - 1)
    bin_flip = np.zeros(len(MARGIN_BINS) - 1)

    ref_logits: list[np.ndarray] = []
    for ids in prompts:
        lg = src_model(mx.array([ids]))[0].astype(mx.float32)
        mx.eval(lg)
        ref_logits.append(np.array(lg))
    del src_model

    q_model = load_quantized(Path(args.bundle))
    for ids, ref in zip(prompts, ref_logits):
        lg = q_model(mx.array([ids]))[0].astype(mx.float32)
        mx.eval(lg)
        got = np.array(lg)

        p = ref - ref.max(-1, keepdims=True)
        p = np.exp(p); p /= p.sum(-1, keepdims=True)
        lq = got - got.max(-1, keepdims=True)
        lq = lq - np.log(np.exp(lq).sum(-1, keepdims=True))
        lp = np.log(np.maximum(p, 1e-12))
        kl = (p * (lp - lq)).sum(-1)

        tot_kl += float(kl.sum())
        tot_tok += kl.shape[0]

        srt = np.sort(ref, axis=-1)
        margin = srt[:, -1] - srt[:, -2]
        flip = ref.argmax(-1) != got.argmax(-1)
        agree += int((~flip).sum())
        idx = np.digitize(margin, MARGIN_BINS) - 1
        idx = np.clip(idx, 0, len(bin_tot) - 1)
        for b in range(len(bin_tot)):
            m = idx == b
            bin_tot[b] += int(m.sum())
            bin_flip[b] += int(flip[m].sum())

    print(f"\nmean KL      : {tot_kl / tot_tok:.6f}")
    print(f"top-1 agree  : {100.0 * agree / tot_tok:.2f}%  ({tot_tok} tokens)")
    print("\nmargin-conditioned flip curve (source top1-top2):")
    rates = []
    for b in range(len(bin_tot)):
        lo, hi = MARGIN_BINS[b], MARGIN_BINS[b + 1]
        if bin_tot[b] == 0:
            continue
        r = 100.0 * bin_flip[b] / bin_tot[b]
        rates.append(r)
        print(f"  [{lo:4.1f},{hi:5.1f})  n={int(bin_tot[b]):7d}  flips={r:6.2f}%")

    mono = all(rates[i] >= rates[i + 1] - 1e-9 for i in range(len(rates) - 1))
    verdict = "HEALTHY (monotone decreasing)" if mono else "NOT MONOTONE — inspect"
    print(f"\nflip curve: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
