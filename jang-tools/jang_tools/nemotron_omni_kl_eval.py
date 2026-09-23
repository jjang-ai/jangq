"""KL / top-1 agreement for Nemotron-3-Nano-Omni bundles, on HELD-OUT prompts.

Created by Jinho Jang (eric@osaurus.ai) — 2026-08-22.

The judge for every quality claim about these bundles. Two passes, because the
BF16 reference (~59 GiB) and a quantized bundle (~20 GiB) do not comfortably
co-reside with the Hessians on a 128 GiB box, and because one cached reference
should grade *every* candidate rather than being recomputed per bundle:

    # once
    python -m jang_tools.nemotron_omni_kl_eval ref  --src <bf16_omni> --cache <f.npz>
    # per candidate
    python -m jang_tools.nemotron_omni_kl_eval test --bundle <jang_dir> --cache <f.npz>

🚨 Held-out by construction. `nemotron_omni_calibrate` draws its corpus from the
FRONT of each domain queue in `corpus_v3.jsonl`; this draws from the BACK and
asserts zero overlap with the calibration draw. AWQ, the imatrix fit and GPTQ
are all fit on the calibration text, so scoring them in-sample flatters them —
the existing `qwen36_kl_eval` built-in prompt set is 33 % contaminated for
exactly this reason, and an in-sample A/B is how GPTQ came to look like a win on
a 35 B MoE that it actually made worse.

Teacher-forced on the reference's own greedy continuation, so both models score
the same token sequence and KL is not confounded by divergent sampling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def held_out_prompts(tokenizer, corpus: Path, n: int = 32,
                     calib_tokens: int = 300_000, max_tok: int = 512):
    """Draw from the BACK of each domain queue; assert disjoint from calibration."""
    from .qwen36_calibrate import build_text_corpus, CORPUS_MIX

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

    out, domains = [], sorted(by_domain)
    i = 0
    while len(out) < n and domains:
        d = domains[i % len(domains)]
        i += 1
        if not by_domain[d]:
            domains.remove(d)
            continue
        text = by_domain[d].pop()          # from the BACK
        ids = tokenizer.encode(text)
        if len(ids) > max_tok:
            text = tokenizer.decode(ids[:max_tok])
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True, tokenize=False)
        if rendered in used_set:
            continue                        # contaminated — skip
        out.append(rendered)
    overlap = len(set(out) & used_set)
    if overlap:
        raise RuntimeError(f"{overlap} held-out prompts appear in calibration set")
    return out


def _greedy_and_logits(model, ids: list[int], n_tok: int):
    """Greedy-continue n_tok from the prompt, returning the continuation and the
    full next-token log-probs at each step."""
    cur = mx.array([ids])
    out_tokens, out_logp = [], []
    cache = None
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    logits = model(cur, cache=cache)[:, -1, :]
    for _ in range(n_tok):
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        out_logp.append(np.asarray(lp.astype(mx.float16))[0])
        nxt = int(mx.argmax(logits, axis=-1).item())
        out_tokens.append(nxt)
        logits = model(mx.array([[nxt]]), cache=cache)[:, -1, :]
    return out_tokens, np.stack(out_logp)


def _teacher_forced_logits(model, ids: list[int], cont: list[int]):
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    logits = model(mx.array([ids]), cache=cache)[:, -1, :]
    out = []
    for t in cont:
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        out.append(np.asarray(lp.astype(mx.float16))[0])
        logits = model(mx.array([[t]]), cache=cache)[:, -1, :]
    return np.stack(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ref", "test"])
    ap.add_argument("--src", type=Path, help="BF16 omni source (ref mode)")
    ap.add_argument("--bundle", type=Path, help="JANG bundle (test mode)")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument(
        "--corpus",
        type=Path,
        default=Path.home() / ".cache" / "jang" / "corpus_v3.jsonl",
    )
    ap.add_argument("--prompts", type=int, default=32)
    ap.add_argument("--tokens", type=int, default=48)
    a = ap.parse_args()

    from transformers import AutoTokenizer

    if a.mode == "ref":
        from .nemotron_omni_calibrate import load_llm
        tok = AutoTokenizer.from_pretrained(str(a.src), trust_remote_code=True)
        prompts = held_out_prompts(tok, a.corpus, a.prompts)
        print(f"  {len(prompts)} held-out prompts (disjoint from calibration)")
        model, _, _ = load_llm(a.src)
        # Prompts are ragged, so store them under indexed keys rather than as
        # object arrays: that keeps the cache loadable WITHOUT allow_pickle,
        # which would otherwise make reading it equivalent to executing it.
        store: dict[str, np.ndarray] = {"n": np.array([len(prompts)])}
        for i, p in enumerate(prompts):
            ids = tok.encode(p)
            cont, lp = _greedy_and_logits(model, ids, a.tokens)
            store[f"ids_{i}"] = np.array(ids, dtype=np.int32)
            store[f"cont_{i}"] = np.array(cont, dtype=np.int32)
            store[f"logp_{i}"] = lp.astype(np.float16)
            print(f"    ref {i+1}/{len(prompts)}", flush=True)
        np.savez(a.cache, **store)
        (a.cache.with_suffix(".prompts.json")).write_text(json.dumps(prompts, indent=1))
        print(f"  wrote reference cache -> {a.cache}")
        return 0

    # ---- test ----
    d = np.load(a.cache)          # no allow_pickle: plain arrays only
    n_prompts = int(d["n"][0])
    from mlx_lm import load
    model, _ = load(str(a.bundle))
    kls, tops, n = [], [], 0
    for i in range(n_prompts):
        ids, cont = list(d[f"ids_{i}"]), list(d[f"cont_{i}"])
        tlp = _teacher_forced_logits(model, ids, cont)
        r = d[f"logp_{i}"].astype(np.float32)
        t = tlp.astype(np.float32)
        pr = np.exp(r)
        kl = np.sum(pr * (r - t), axis=-1)          # KL(ref || test) per token
        kls.append(kl)
        tops.append((r.argmax(-1) == t.argmax(-1)).astype(np.float32))
        n += len(kl)
        print(f"    test {i+1}/{len(ids_all)}  KL={kl.mean():.4f}", flush=True)
    kl = np.concatenate(kls)
    top = np.concatenate(tops)
    print(f"\n  bundle: {a.bundle}")
    print(f"  tokens scored: {n}")
    print(f"  mean KL(ref||test): {kl.mean():.4f} nats   median {np.median(kl):.4f}")
    print(f"  top-1 agreement:    {100*top.mean():.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
