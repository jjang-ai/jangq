"""DSV41.commit_prefix == feeding exactly the kept tokens.

A capture_verify forward of W+1 tokens followed by commit_prefix(keep) must leave the cache a
forward of only those `keep` tokens would leave: same window rows, compressed / index pool rows,
compressor pending tail, Engram history and position. Two checks on a small random DSV4.1 from
parity_reference (ratio-2 and ratio-1 compressors, 16-token window, index owners), CPU:
  1. exact: commit_prefix(W+1) after a capture forward == that forward's own state, bit for bit;
  2. against a forward of only the kept tokens (a different batch shape, so bf16 numerics differ in
     the last bits and the small model's top-8 indexer picks can flip): same row counts and
     pending lengths, values within cos 0.995. Logit drift after a flipped pick is reported only.

  PYTHONPATH=jang-tools python jang_tools/dsv41/tests/test_commit_prefix.py --src ~/models/DeepSeek-V4.1-Flash-1017
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from jang_tools.dsv41.tests import parity_reference as P


def cos(a, b):
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def arr(x):
    return None if x is None else np.array(x.astype(mx.float32))


def pool(buf):
    v = buf.view()
    return None if v is None else arr(v)


def close(a, b, tol=0.995):
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and (a.size == 0 or cos(a, b) > tol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    a = ap.parse_args()
    src = Path(a.src).expanduser()
    sys.path[:0] = [str(Path(P.__file__).parent / "ref_shim"), str(src / "inference")]
    import model as ref
    from transformers import AutoTokenizer

    from jang_tools.dsv41.model import DSV41

    mx.set_default_device(mx.cpu)
    torch.set_default_dtype(torch.bfloat16)
    text = P.build_config()
    text.update(index_head_dim=64)  # 4-bit pools quantize in groups of 64
    tok = AutoTokenizer.from_pretrained(str(src))
    rm = ref.Transformer(P.ref_args(ref, text), tok)
    torch.set_default_dtype(torch.float32)
    P.randomize(rm)
    ids = tok.encode((src / "encoding" / "README.md").read_text(), add_special_tokens=False)[:160]
    ok = True
    with tempfile.TemporaryDirectory() as td:
        P.write_bundle(rm, text, src, Path(td) / "b")
        m = DSV41(Path(td) / "b", wired=False)
        W1 = 6  # verify width W+1
        for start in (31, 32, 33, 47, 63):  # pending tails of every length at the ratio-2 layers
            for keep in range(1, W1 + 1):
                s_ref = m.new_session()
                m.forward(ids[:start], s_ref)
                m.forward(ids[start:start + keep], s_ref)
                s = m.new_session()
                m.forward(ids[:start], s)
                ck = s.checkpoint()
                m.forward(ids[start:start + W1], s, all_logits=True, capture_verify=True)
                m.commit_prefix(s, ck, keep)
                why = []
                if not (s.pos == s_ref.pos and np.array_equal(s.history, s_ref.history)):
                    why.append("pos/history")
                for li, (c, r) in enumerate(zip(s.caches, s_ref.caches)):
                    if not close(arr(c.window), arr(r.window)):
                        why.append(f"L{li} window")
                    if not (c.compress_kv.n == r.compress_kv.n and close(pool(c.compress_kv), pool(r.compress_kv))):
                        why.append(f"L{li} pool {c.compress_kv.n}/{r.compress_kv.n}")
                    if not (c.index_k.n == r.index_k.n and close(pool(c.index_k), pool(r.index_k))):
                        why.append(f"L{li} index {c.index_k.n}/{r.index_k.n}")
                    if not (close(arr(c.pend_kv), arr(r.pend_kv)) and close(arr(c.pend_score), arr(r.pend_score))):
                        why.append(f"L{li} pend")
                good = not why
                nxt = ids[start + keep:start + keep + 12]
                la = np.array(m.forward(nxt, s, all_logits=True))
                lb = np.array(m.forward(nxt, s_ref, all_logits=True))
                c_ = min(cos(la[i], lb[i]) for i in range(len(nxt)))
                good &= c_ > 0.999
                if why:
                    print(f"FAIL start={start} keep={keep}: state {why}")
                elif c_ <= 0.999:
                    print(f"note start={start} keep={keep}: state matches; later logits cos {c_:.5f} (top-k pick flipped)")
                ok &= not why
        for start in (31, 32, 47):
            s1, s2 = m.new_session(), m.new_session()
            m.forward(ids[:start], s1); m.forward(ids[:start], s2)
            m.forward(ids[start:start + W1], s1, all_logits=True)
            ck = s2.checkpoint()
            m.forward(ids[start:start + W1], s2, all_logits=True, capture_verify=True)
            m.commit_prefix(s2, ck, W1)
            for li, (c, r) in enumerate(zip(s1.caches, s2.caches)):
                for name, x, y in (("window", arr(c.window), arr(r.window)), ("pool", pool(c.compress_kv), pool(r.compress_kv)),
                                   ("index", pool(c.index_k), pool(r.index_k)), ("pend", arr(c.pend_kv), arr(r.pend_kv))):
                    same = (x is None and y is None) or (x is not None and y is not None and np.array_equal(x, y))
                    if not same:
                        print(f"FAIL exact replay start={start} L{li} {name}")
                    ok &= same
        print("commit_prefix == kept-token forward:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
