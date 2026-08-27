"""KL / top-1 for Ornith 1.5 bundles against an MXFP8 reference — with a floor check.

Created by Jinho Jang (eric@osaurus.ai) — 2026-08-23.

Replaces the untrustworthy result in
`docs/runtime/ornith-1.5/00-SOURCE-AND-PLAN.md` §17, where 9B 6D measured the
SAME as 4D (0.0526 vs 0.0514) despite being a whole bit-tier apart with 3.4x
better imatrix rel-err. Two causes were identified there: **under-sampling**
(12 prompts x 64 tokens ~ 770 positions) and a harness so slow by construction
that 64 tokens was the practical ceiling.

Two changes fix both at once:

1. **Teacher-force on fixed held-out text instead of a generated rollout.**
   `qwen36_kl_eval` first builds each rollout with a FULL forward per token and
   no KV cache — 12 x 256 growing-sequence forwards, which is the entire reason
   64 was the default. Scoring was never the bottleneck: it is already ONE
   forward per sequence. Dropping the rollout makes 300+ tokens per prompt
   essentially free and both models score the identical token sequence, which
   is what a KL comparison actually needs.

2. **A floor check is mandatory, not optional.** Scoring the reference against
   its OWN cached logprobs must return KL ~= 0. Without that, a small KL is
   unfalsifiable — you cannot separate "these bundles are close" from "this
   harness cannot resolve anything". The prior run never got a floor number
   (the attempt died to GPU contention), which is precisely why its numbers
   could not be defended.

Reference logprobs are written to disk per sequence, so exactly ONE model is
resident at a time — the 35B MXFP8 is 35 GiB and the 6M is 28 GiB, and
`feedback_no_concurrent_mlx` on this family is not "2x slower" but a hard
`kIOGPUCommandBufferCallbackErrorTimeout` that also poisons whatever else is on
the device.

    # once — reference + held-out sequences
    python -m jang_tools.ornith_kl_eval ref  --model <MXFP8> --out <dir>
    # floor check FIRST, then each candidate
    python -m jang_tools.ornith_kl_eval test --model <MXFP8> --cache <dir>
    python -m jang_tools.ornith_kl_eval test --model <JANG_4M> --cache <dir>
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


def _forward(model, toks: list[int]):
    """One full-sequence forward with the mRoPE position cache reset.

    mlx_vlm's qwen3_5 caches `_position_ids` / `_rope_deltas` on the language
    model and only clears them inside the top-level text-only branch of
    `get_input_embeddings()`. Calling `model.language_model(...)` directly skips
    that reset, so the FIRST call's position ids stick and every later call at a
    different length dies in `apply_multimodal_rotary_pos_emb` with cos/sin
    sized to the first call. Clearing per call is what the text path does.
    """
    lm = model.language_model
    lm._position_ids = None
    lm._rope_deltas = None
    out = lm(mx.array([toks]))
    return out.logits if hasattr(out, "logits") else out


def _logprobs(model, toks: list[int], start: int) -> np.ndarray:
    logits = _forward(model, toks)
    lp = logits[0, start - 1:-1].astype(mx.float32)   # predicts toks[start:]
    lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
    mx.eval(lp)
    out = np.array(lp, copy=True).astype(np.float16)
    del logits, lp
    mx.clear_cache()
    return out


def held_out_sequences(tokenizer, corpus: Path, n: int, n_tok: int,
                       calib_tokens: int = 300_000):
    """Draw from the BACK of each domain queue and assert disjoint from calibration."""
    from .qwen36_calibrate import build_text_corpus

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

    used, _, _ = build_text_corpus(tokenizer, corpus, calib_tokens,
                                   max_prompt_tokens=2048)
    used_set = set(used)

    out, domains, i = [], sorted(by_domain), 0
    while len(out) < n and domains:
        d = domains[i % len(domains)]
        i += 1
        if not by_domain[d]:
            domains.remove(d)
            continue
        text = by_domain[d].pop()               # from the BACK
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True, tokenize=False)
        if rendered in used_set:
            continue
        ids = tokenizer.encode(rendered)
        body = tokenizer.encode(text)
        if len(body) < n_tok:                    # need enough continuation
            continue
        # prompt = rendered chat frame; continuation = the next n_tok real tokens
        toks = ids + body[:n_tok]
        out.append((d, len(ids), toks))
    if len(out) < n:
        raise RuntimeError(
            f"only {len(out)}/{n} held-out sequences met the {n_tok}-token "
            "continuation requirement — refusing to score an under-sampled set, "
            "which is exactly what made the previous result untrustworthy")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ref", "test"])
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--cache", type=Path)
    ap.add_argument(
        "--corpus",
        type=Path,
        default=Path.home() / ".cache" / "jang" / "corpus_v3.jsonl",
    )
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=320)
    a = ap.parse_args()

    from mlx_vlm import load

    if a.mode == "ref":
        a.out.mkdir(parents=True, exist_ok=True)
        model, proc = load(str(a.model))
        tok = proc.tokenizer
        seqs = held_out_sequences(tok, a.corpus, a.prompts, a.tokens)
        print(f"  {len(seqs)} held-out sequences x {a.tokens} scored tokens "
              f"= {len(seqs)*a.tokens} positions", flush=True)
        meta = {"reference": str(a.model), "tokens": a.tokens,
                "n": len(seqs), "positions": len(seqs) * a.tokens,
                "domains": [d for d, _, _ in seqs]}
        t0 = time.time()
        for i, (_d, start, toks) in enumerate(seqs):
            lp = _logprobs(model, toks, start)
            np.save(a.out / f"ref_{i}.npy", lp)
            np.save(a.out / f"toks_{i}.npy", np.array(toks, dtype=np.int32))
            np.save(a.out / f"start_{i}.npy", np.array([start], dtype=np.int32))
            print(f"    ref {i+1}/{len(seqs)} ({time.time()-t0:.0f}s)", flush=True)
        (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"  wrote reference cache -> {a.out}")
        return 0

    meta = json.loads((a.cache / "meta.json").read_text())
    n = meta["n"]
    model, proc = load(str(a.model))
    kls, tops, t0 = [], [], time.time()
    for i in range(n):
        toks = list(np.load(a.cache / f"toks_{i}.npy"))
        start = int(np.load(a.cache / f"start_{i}.npy")[0])
        r = np.load(a.cache / f"ref_{i}.npy").astype(np.float32)
        t = _logprobs(model, [int(x) for x in toks], start).astype(np.float32)
        pr = np.exp(r)
        kls.append(np.sum(pr * (r - t), axis=-1))          # KL(ref || test)
        tops.append((r.argmax(-1) == t.argmax(-1)).astype(np.float32))
        del r, t, pr
        gc.collect()
        print(f"    test {i+1}/{n} ({time.time()-t0:.0f}s)", flush=True)
    kl = np.concatenate(kls)
    top = np.concatenate(tops)
    print(f"\n  reference : {Path(meta['reference']).name}")
    print(f"  test      : {a.model.name}")
    print(f"  positions : {kl.size}")
    print(f"  mean KL   : {kl.mean():.6f} nats   median {np.median(kl):.6f}")
    print(f"  top-1     : {100*top.mean():.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
