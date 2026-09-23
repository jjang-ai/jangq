"""Offline n-gram row-frequency histogram for the Qwen4-Exp PLE table.

Pure tokenizer + integer hashing — no model forward needed. Each corpus
record is hashed as its own EOS-fresh segment (matching runtime semantics
where history resets at EOS). Output: uint32 counts over the padded
~320M-row table + summary stats. This drives per-row/bit allocation and
KL-prompt selection for the 51B table.

Usage:
  python -m jang_tools.qwen4_exp.ngram_histogram \
      --corpus ~/jang/kimi_v3_calib/corpus_v3.jsonl \
      --tokenizer ~/models/Qwen3.8-Flash-Next \
      --out ~/models/Logs/q38fn-calib/ngram_hist.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np


def iter_texts(corpus_path: str):
    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if isinstance(rec, str):
                yield rec
                continue
            if "text" in rec:
                yield rec["text"]
            elif "messages" in rec:
                yield "\n".join(
                    m.get("content", "") if isinstance(m.get("content"), str) else ""
                    for m in rec["messages"]
                )
            elif "prompt" in rec:
                yield str(rec.get("prompt", "")) + str(rec.get("response", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-records", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)

    from jang_tools.qwen4_exp.ngram import NGramHasher

    hasher = NGramHasher(vocab_size=248320, eos_token_id=248044)
    counts = np.zeros(hasher.padded_vocab_size, dtype=np.uint32)

    n_rec = 0
    n_tok = 0
    batch: list = []

    def flush(batch):
        nonlocal n_tok
        for ids in batch:
            if len(ids) < 2:
                continue
            rows = hasher.hash_tokens(np.array(ids, dtype=np.int64)[None, :], None)
            np.add.at(counts, rows.ravel(), 1)
            n_tok += len(ids)

    for text in iter_texts(args.corpus):
        if not text:
            continue
        batch.append(tok.encode(text, add_special_tokens=False))
        n_rec += 1
        if len(batch) >= 256:
            flush(batch)
            batch = []
            print(f"  {n_rec} records, {n_tok/1e6:.2f}M tokens", flush=True)
        if args.max_records and n_rec >= args.max_records:
            break
    flush(batch)

    nz = int((counts > 0).sum())
    total = int(counts.sum())
    order = np.argsort(counts)[::-1]
    top = counts[order[:1000]].astype(np.int64)
    print(f"records={n_rec} tokens={n_tok/1e6:.2f}M lookups={total/1e6:.2f}M")
    print(f"nonzero rows: {nz/1e6:.2f}M / {hasher.padded_vocab_size/1e6:.1f}M "
          f"({100*nz/hasher.padded_vocab_size:.2f}%)")
    print(f"top-1000 rows cover {100*top.sum()/max(total,1):.2f}% of lookups")
    for frac in (0.5, 0.9, 0.99):
        csum = np.cumsum(counts[order].astype(np.int64))
        k = int(np.searchsorted(csum, frac * total)) + 1
        print(f"  {int(frac*100)}% of lookups hit the hottest {k/1e6:.3f}M rows")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, counts=counts, n_tokens=n_tok, n_records=n_rec)
    print(f"saved → {args.out}")


if __name__ == "__main__":
    main()
