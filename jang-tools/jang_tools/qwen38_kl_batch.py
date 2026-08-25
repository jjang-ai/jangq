"""Held-out KL / top-1 for MANY bundles against ONE reference, in a single load.

`qwen36_kl_eval` loads a reference and exactly one test bundle. Scoring a whole
lineup that way reloads the reference once per tier — and when the reference is
the 52 GB bf16 source that is most of the wall clock and all of the memory risk.

This builds the teacher-forced rollouts and the reference logprobs ONCE, drops
the reference, then streams each test bundle through the same fixed sequences.

    python -m jang_tools.qwen38_kl_batch <ref> <out.json> \
        --prompts held_out.json [--tokens 32] <bundle> [<bundle> ...]

🚨 REFERENCE CHOICE CHANGES THE NUMBER. The v1 (2026-08-14) Qwen3.8 runs used
`Qwen3.8-27B-MXFP8` as the reference, so their published KLs (4D = 0.02775) are
distance-to-MXFP8, not distance-to-truth — MXFP8 has its own error against bf16,
and scoring against it understates every tier. v2 uses the bf16 source. The two
sets of numbers are NOT comparable; re-score v1 bundles here before any A/B.

🚨 HELD-OUT PROMPTS ARE NOT OPTIONAL. AWQ, the imatrix refit and the Hessian bit
map are all FIT on the calibration corpus. `qwen36_kl_eval`'s built-in prompt
list is 5/6 verbatim calibration prompts, so it scores those methods in-sample
and flatters them.

MEMORY. The rollout loop re-runs the full sequence per generated token with no
KV cache, so MLX's pool grows across prefills; this clears it every forward and
moves stored logprobs to host numpy. That combination is what a previous run
lacked when it exhausted system RAM.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


def fwd(model, toks):
    """One full-sequence forward with the mRoPE position cache reset.

    mlx_vlm's qwen3_5 caches `_position_ids` / `_rope_deltas` on the language
    model and only clears them in the top-level text-only branch. Calling
    `model.language_model(...)` directly skips that reset, so the first call's
    position ids stick and every later call at a different length dies in
    `apply_multimodal_rotary_pos_emb` with cos/sin sized to the FIRST call.
    """
    model.language_model._position_ids = None
    model.language_model._rope_deltas = None
    out = model.language_model(mx.array([toks]))
    return out.logits if hasattr(out, "logits") else out


def logprobs(model, seqs):
    out = []
    for start, toks in seqs:
        logits = fwd(model, toks)
        lp = logits[0, start - 1:-1].astype(mx.float32)   # predicts toks[start:]
        lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        mx.eval(lp)
        out.append(np.array(lp, copy=True))               # -> host, frees the pool
        del logits, lp
        mx.clear_cache()
    return out


def main(argv) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 1
    ref_p, out_p = Path(argv[1]), Path(argv[2])
    n_tok, prompts_p = 32, None
    tests = []
    skip = {0, 1, 2}
    for i, a in enumerate(argv):
        if i in skip:
            continue
        if a == "--tokens":
            n_tok = int(argv[i + 1]); skip |= {i, i + 1}
        elif a == "--prompts":
            prompts_p = argv[i + 1]; skip |= {i, i + 1}
        elif not a.startswith("--") and i not in skip:
            tests.append(Path(a))
    if prompts_p is None:
        print("refusing to run in-sample: pass --prompts <held_out.json>")
        return 2
    prompts = json.loads(Path(prompts_p).read_text())["prompts"]
    tests = [t for t in tests if t != ref_p]

    from mlx_vlm import load

    print(f"  reference : {ref_p.name}")
    print(f"  prompts   : {prompts_p} ({len(prompts)} held-out)")
    print(f"  tokens    : {n_tok}")
    print(f"  bundles   : {[t.name for t in tests]}\n", flush=True)

    ref, proc = load(str(ref_p))

    seqs, t0 = [], time.time()
    for j, p in enumerate(prompts):
        text = proc.tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True,
            tokenize=False, enable_thinking=True)
        ids = proc.tokenizer.encode(text)
        toks = list(ids)
        for _ in range(n_tok):
            logits = fwd(ref, toks)
            toks.append(int(mx.argmax(logits[0, -1]).item()))
            del logits
            mx.clear_cache()
        seqs.append((len(ids), toks))
        print(f"    rollout {j+1}/{len(prompts)}  "
              f"({len(ids)}+{n_tok} tok, {time.time()-t0:.0f}s)", flush=True)
    print(f"  rollouts built in {time.time()-t0:.0f}s", flush=True)

    ref_lp = logprobs(ref, seqs)
    del ref
    mx.clear_cache()

    # FAIL CLOSED: a degenerate reference makes every downstream KL meaningless
    # but still prints a healthy-looking small number.
    tot_pos = sum(r.shape[0] for r in ref_lp)
    if tot_pos == 0:
        print("INVALID: reference produced zero scored positions")
        return 3
    ref_ent = float(np.mean([-(np.exp(r) * r).sum(-1).mean() for r in ref_lp]))
    print(f"  scored positions : {tot_pos}")
    print(f"  reference entropy: {ref_ent:.4f} nats", flush=True)
    if not np.isfinite(ref_ent):
        print("INVALID: reference logprobs are not finite")
        return 3

    results = {}
    for tp in tests:
        print(f"\n  ---- {tp.name} ----", flush=True)
        try:
            test, _ = load(str(tp))
        except Exception as e:                      # noqa: BLE001
            print(f"    LOAD FAILED: {e}")
            results[tp.name] = {"error": str(e)}
            continue
        tl = logprobs(test, seqs)
        del test
        mx.clear_cache()

        kl, agree, pos, per_prompt = 0.0, 0, 0, []
        for r, t in zip(ref_lp, tl):
            p = np.exp(r)
            k = float((p * (r - t)).sum())
            a = int((r.argmax(-1) == t.argmax(-1)).sum())
            per_prompt.append({"kl": k / r.shape[0],
                               "top1": 100.0 * a / r.shape[0]})
            kl += k; agree += a; pos += r.shape[0]
        rec = {"positions": pos, "mean_kl": kl / pos,
               "top1_pct": 100.0 * agree / pos,
               "worst_prompt_kl": max(x["kl"] for x in per_prompt),
               "per_prompt": per_prompt}
        results[tp.name] = rec
        print(f"    mean KL (nats) : {rec['mean_kl']:.5f}")
        print(f"    top-1 agreement: {rec['top1_pct']:.2f} %")
        print(f"    worst prompt KL: {rec['worst_prompt_kl']:.5f}", flush=True)
        del tl

    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps({
        "reference": str(ref_p), "prompts": prompts_p, "tokens": n_tok,
        "scored_positions": tot_pos, "reference_entropy": ref_ent,
        "results": results,
    }, indent=1))
    print(f"\n  -> {out_p}")

    print(f"\n  {'bundle':40s} {'KL':>9} {'top-1':>8}")
    for k, v in sorted(results.items(),
                       key=lambda kv: kv[1].get("mean_kl", 9e9)):
        if "error" in v:
            print(f"  {k:40s} {'LOAD FAIL':>9}")
        else:
            print(f"  {k:40s} {v['mean_kl']:>9.5f} {v['top1_pct']:>7.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
