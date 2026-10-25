"""Teacher-forced held-out NLL / perplexity for a DSV4.1-1017 JANG bundle.

Scores a fixed set of held-out conversations under one bundle and writes the
per-row and aggregate negative log-likelihood (mean over scored next-token
positions) plus perplexity. Run it once per bundle (RTN, then the imatrix
build) over the SAME rows and compare: lower NLL = the better-calibrated
weights. The default hold-out is the corpus rows whose domain was NOT used for
trace generation / stats (so the imatrix experts never saw them), rendered from
their recorded content.

  python -m jang_tools.dsv41.calib_eval --bundle <bundle> \
      --rows calib/dsv41-1017/corpus_v41.jsonl --out eval/rtn.json
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import mlx.core as mx

# domains used for calibration (must match calib_generate.DEFAULT_DOMAINS) -> excluded from hold-out
from jang_tools.dsv41.calib_generate import DEFAULT_DOMAINS


def render_full(enc, row):
    msgs = copy.deepcopy(row["messages"])
    for m in msgs:
        if m["role"] == "assistant":
            m.setdefault("reasoning_content", "")
    return enc.encode_messages(msgs, thinking_mode="thinking", reasoning_effort=row["effort"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--engram-dir")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=2560)
    ap.add_argument("--n", type=int, default=32, help="held-out rows to score")
    ap.add_argument("--held-out", action="store_true", default=True,
                    help="score only domains NOT in DEFAULT_DOMAINS (genuine hold-out)")
    ap.add_argument("--domains", help="comma list to score instead of the default hold-out")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from jang_tools.dsv41.model import DSV41
    from jang_tools.dsv41.generate import load_encoding

    bundle = Path(a.bundle).expanduser()
    enc = load_encoding(bundle)
    m = DSV41(bundle, a.engram_dir)

    rows = [json.loads(l) for l in open(a.rows)]
    if a.domains:
        want = set(a.domains.split(","))
        rows = [r for r in rows if r["domain"] in want]
    else:
        rows = [r for r in rows if r["domain"] not in DEFAULT_DOMAINS]
    # deterministic pick, stable across bundles
    import random

    random.Random(a.seed).shuffle(rows)
    rows = rows[: a.n]

    t0 = time.time()
    per_row = []
    tot_nll = 0.0
    tot_tok = 0
    for i, r in enumerate(rows):
        text = render_full(enc, r)
        ids = m.tokenizer.encode(text, add_special_tokens=False).ids[: a.max_tokens]
        if len(ids) < 2:
            continue
        sess = m.new_session()
        logits = m.forward(ids, sess, all_logits=True)  # [T, vocab]
        lp = logits[:-1].astype(mx.float32)  # predict positions 1..T-1
        tgt = mx.array(ids[1:], mx.int32)
        logz = mx.logsumexp(lp, axis=-1)
        tok_lp = mx.take_along_axis(lp, tgt[:, None], axis=-1)[:, 0] - logz
        nll = float(-mx.sum(tok_lp))
        n = len(ids) - 1
        mx.eval(logits)
        del logits, lp, sess
        mx.clear_cache()
        tot_nll += nll
        tot_tok += n
        per_row.append({"domain": r["domain"], "effort": r["effort"], "n_tok": n, "nll_mean": nll / n})
        if (i + 1) % 8 == 0:
            print(f"  {i + 1}/{len(rows)} rows, {tot_tok} tok, nll/tok={tot_nll / tot_tok:.4f} ({time.time() - t0:.0f}s)", flush=True)

    mean_nll = tot_nll / max(tot_tok, 1)
    out = {"bundle": str(bundle), "rows": len(per_row), "tokens": tot_tok,
           "nll_per_tok": mean_nll, "ppl": float(mx.exp(mx.array(mean_nll))),
           "per_row": per_row}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"{bundle.name}: nll/tok={mean_nll:.4f}  ppl={out['ppl']:.3f}  "
          f"over {tot_tok} tok / {len(per_row)} rows in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
