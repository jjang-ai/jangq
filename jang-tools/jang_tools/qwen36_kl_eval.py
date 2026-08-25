"""KL / top-1 agreement of a Qwen3.6-27B bundle against a reference bundle.

The judge for every quality claim in this pipeline. Runs a fixed prompt set
through both models (teacher-forced on the reference's greedy continuation) and
reports mean KL(ref || quant) per token and top-1 agreement. Text-only prompts
so results are comparable across profiles; rendered through the real chat
template with thinking ON (the default) plus a coding slice.

    python -m jang_tools.qwen36_kl_eval <ref_bundle> <test_bundle> [--tokens 64] \
        [--prompts held_out.json]

🚨 The built-in PROMPTS are CONTAMINATED: five of them are verbatim or
near-verbatim entries of `qwen36_calibrate`'s built-in prompt set. AWQ, the
imatrix refit and GPTQ are all FIT ON that data, so scoring them in-sample
flatters them. Any A/B that decides whether a calibrated method helped MUST be
run with `--prompts` pointing at a held-out set.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

PROMPTS = [
    "Prove that the square root of 2 is irrational.",
    "Write a Python function that merges overlapping intervals and state its complexity.",
    "A bag has 4 red, 6 blue and 5 green marbles. Two are drawn without replacement. What is P(same colour)?",
    "Explain the difference between a B-tree and an LSM tree for storage engines.",
    "Implement binary search and explain the classic off-by-one pitfall.",
    "Summarise the causes of the 1873 financial panic in two paragraphs.",
]


def main(argv) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 1
    ref_p, test_p = Path(argv[1]), Path(argv[2])
    n_tok = 64
    prompts = PROMPTS
    prompts_src = "built-in (CONTAMINATED — in-sample for AWQ/imatrix/GPTQ)"
    for i, a in enumerate(argv):
        if a == "--tokens":
            n_tok = int(argv[i + 1])
        if a == "--prompts":
            import json as _json
            prompts = _json.loads(Path(argv[i + 1]).read_text())["prompts"]
            prompts_src = f"{argv[i + 1]} ({len(prompts)} held-out)"

    from mlx_vlm import load

    def fwd(model, toks):
        """One full-sequence forward, with the mRoPE position cache reset.

        mlx_vlm's qwen3_5 caches `_position_ids` / `_rope_deltas` on the
        language model and only clears them inside the top-level
        `get_input_embeddings()` text-only branch (qwen3_5.py). This harness
        calls `model.language_model(...)` DIRECTLY, so that reset never runs:
        the first call's position ids stick, and every later call at a
        different length dies in `apply_multimodal_rotary_pos_emb` with
        cos/sin sized to the FIRST call (measured: fine at 64, then broadcast
        errors at 128/256/373/512 on the same instance, while a FRESH instance
        handles every one of those lengths). Clearing per call is exactly what
        the text-only path does.
        """
        model.language_model._position_ids = None
        model.language_model._rope_deltas = None
        out = model.language_model(mx.array([toks]))
        return out.logits if hasattr(out, "logits") else out

    print(f"  reference: {ref_p.name}")
    print(f"  test     : {test_p.name}")
    print(f"  prompts  : {prompts_src}")

    ref, proc = load(str(ref_p))

    # 1) build teacher-forced sequences with the REFERENCE (greedy)
    seqs = []
    t0 = time.time()
    for p in prompts:
        prompt = proc.tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True,
            tokenize=False, enable_thinking=True)
        ids = proc.tokenizer.encode(prompt)
        toks = list(ids)
        for _ in range(n_tok):
            logits = fwd(ref, toks)
            nxt = int(mx.argmax(logits[0, -1]).item())
            toks.append(nxt)
            del logits
            mx.clear_cache()
        seqs.append((len(ids), toks))
    print(f"  rollouts built in {time.time()-t0:.0f}s")

    # 2) reference logprobs on those sequences
    def logprobs(model, seqs):
        out = []
        for start, toks in seqs:
            logits = fwd(model, toks)
            lp = logits[0, start - 1:-1].astype(mx.float32)   # predicts toks[start:]
            lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
            mx.eval(lp)
            out.append(np.array(lp, copy=True))    # -> host, frees the pool
            del logits, lp
            mx.clear_cache()
        return out

    ref_lp = logprobs(ref, seqs)
    del ref
    mx.clear_cache()

    test, _ = load(str(test_p))
    test_lp = logprobs(test, seqs)
    del test
    mx.clear_cache()

    tot_kl, tot_pos, agree = 0.0, 0, 0
    for (start, toks), rl, tl in zip(seqs, ref_lp, test_lp):
        p = np.exp(rl)
        tot_kl += float((p * (rl - tl)).sum())
        tot_pos += rl.shape[0]
        agree += int((rl.argmax(-1) == tl.argmax(-1)).sum())

    print(f"\n  positions      : {tot_pos}")
    print(f"  mean KL (nats) : {tot_kl / tot_pos:.5f}")
    print(f"  top-1 agreement: {100.0 * agree / tot_pos:.2f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
