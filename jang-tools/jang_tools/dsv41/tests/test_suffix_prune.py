"""Prompt-suffix pruning (Model.__call__ prompt_end) == a full prefill.

A small random DSV4.1 from parity_reference (8 layers, 16-token window, KV sources 1 and 3, Engram 1
and 4, so layers 5-7 are pruned, one of them an index source), CPU. For several prompt lengths and
chunk sizes the body is prefilled with and without prompt_end, then the last prompt token and a few
continuation tokens are fed one by one. Checks:
  1. layers before prune_from: window / pools / index keys / pending tails bit-identical;
  2. pruned layers: the window cache at the prompt boundary within cos 0.9999 (fewer rows per matmul,
     so bf16 numerics differ in the last bits);
  3. the last-token and continuation logits within cos 0.999 and the same argmax.

  PYTHONPATH=jang-tools python jang_tools/dsv41/tests/test_suffix_prune.py --src ~/models/DeepSeek-V4.1-Flash-1017
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
from jang_tools.dsv41.tests.test_commit_prefix import arr, cos, pool


def prefill(m, ids, step, prune):
    s = m.new_session()
    body, P_end = ids[:-1], len(ids) - 1
    for a in range(0, len(body), step):
        m.forward(body[a:a + step], s, prompt_end=P_end if prune else None)
    return s


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
    text.update(index_head_dim=64, num_hidden_layers=8, compress_ratios=[0, 2, 2, 1, 1, 1, 1, 1],
                index_source_layer_ids=[1, 3, 5, 7])
    tok = AutoTokenizer.from_pretrained(str(src))
    rm = ref.Transformer(P.ref_args(ref, text), tok)
    torch.set_default_dtype(torch.float32)
    P.randomize(rm)
    ids_all = tok.encode((src / "encoding" / "README.md").read_text(), add_special_tokens=False)
    ok = True
    with tempfile.TemporaryDirectory() as td:
        P.write_bundle(rm, text, src, Path(td) / "b")
        m = DSV41(Path(td) / "b", wired=False)
        pf = m.model.prune_from
        assert pf == 5, pf
        # keep per pruned layer: 1 + (8 - i) * 15 -> layer 5: 46, 6: 31, 7: 16 rows
        for n, step in ((40, 64), (47, 64), (100, 64), (161, 64), (161, 32), (161, 17), (250, 256)):
            ids = ids_all[:n]
            s0, s1 = prefill(m, ids, step, False), prefill(m, ids, step, True)
            why = []
            for li, (c, r) in enumerate(zip(s0.caches, s1.caches)):
                w0, w1 = arr(c.window), arr(r.window)
                if li < pf:
                    same = all((x is None and y is None) or (x is not None and y is not None and np.array_equal(x, y))
                               for x, y in ((w0, w1), (pool(c.compress_kv), pool(r.compress_kv)),
                                            (pool(c.index_k), pool(r.index_k)), (arr(c.pend_kv), arr(r.pend_kv))))
                    if not same:
                        why.append(f"L{li} state differs")
                elif w0.shape != w1.shape or cos(w0, w1) < 0.9999:
                    why.append(f"L{li} window {w0.shape}/{w1.shape} cos {cos(w0, w1) if w0.shape == w1.shape else 0:.6f}")
            tail = ids_all[n - 1:n + 8]  # the last prompt token, then continuation tokens
            worst, flips = 1.0, 0
            for t in tail:
                l0 = np.array(m.forward([t], s0)); l1 = np.array(m.forward([t], s1))
                worst = min(worst, cos(l0, l1)); flips += int(l0.argmax() != l1.argmax())
            if worst < 0.999 or flips:
                why.append(f"logits cos {worst:.5f}, argmax flips {flips}")
            print(f"n={n} step={step}: {'ok' if not why else 'FAIL ' + '; '.join(why)} (logits cos {worst:.6f})")
            ok &= not why
        print("suffix pruning == full prefill:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
