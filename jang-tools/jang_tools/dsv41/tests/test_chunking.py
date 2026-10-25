"""Chunked continuation == single-shot prefill, and batched decode == per-session decode.

Uses the small random DSV4.1 from parity_reference (no reference forward needed).

  PYTHONPATH=jang-tools python jang_tools/dsv41/tests/test_chunking.py --src ~/models/DeepSeek-V4.1-Flash-1017
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from jang_tools.dsv41.tests import parity_reference as P


def cos(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    a = ap.parse_args()
    src = Path(a.src).expanduser()
    sys.path[:0] = [str(Path(P.__file__).parent / "ref_shim"), str(src / "inference")]
    import model as ref
    from transformers import AutoTokenizer

    from jang_tools.dsv41.model import DSV41

    torch.set_default_dtype(torch.bfloat16)
    text = P.build_config()
    # no discrete choices left to flip on bf16 noise: full index selection, every expert routed
    text.update(index_topk=512, candidate_topk_blocks=512, n_routed_experts=2, num_experts_per_tok=2)
    tok = AutoTokenizer.from_pretrained(str(src))
    rm = ref.Transformer(P.ref_args(ref, text), tok)
    torch.set_default_dtype(torch.float32)
    P.randomize(rm)
    ids = tok.encode((src / "encoding" / "README.md").read_text(), add_special_tokens=False)[:150]
    ok = True
    with tempfile.TemporaryDirectory() as td:
        P.write_bundle(rm, text, src, Path(td) / "b")
        m = DSV41(Path(td) / "b", wired=False)

        full = np.array(m.forward(ids, m.new_session(), all_logits=True))
        for plan in ([37, 1, 1, 50, 61], [1] * 20 + [130], [7, 9, 11, 13, 110], [128, 22]):
            sess, outs = m.new_session(), []
            o = 0
            for c in plan:
                outs.append(np.array(m.forward(ids[o:o + c], sess, all_logits=True)))
                o += c
            cc = np.concatenate(outs)
            per = [cos(full[i], cc[i]) for i in range(len(ids))]
            c_ = min(per)
            if c_ < 0.999:
                bad = [i for i, v in enumerate(per) if v < 0.999]
                print(f"   first bad token {bad[0]}, n bad {len(bad)}, boundaries {np.cumsum(plan)[:-1].tolist()[:8]}")
            print(f"chunk plan {plan[:6]}{'...' if len(plan) > 6 else ''}: worst per-token logit cos {c_:.5f}")
            ok &= c_ > 0.999

        # batched decode: three sessions at different positions vs each alone
        starts = [40, 77, 101]
        solo, batch = [], []
        for st in starts:
            s = m.new_session()
            m.forward(ids[:st], s)
            solo.append([np.array(m.forward([ids[st + k]], s)) for k in range(5)])
        sess = [m.new_session() for _ in starts]
        for s, st in zip(sess, starts):
            m.forward(ids[:st], s)
        for k in range(5):
            out = np.array(m.decode_batch(sess, [ids[st + k] for st in starts]))
            batch.append(out)
        c_ = min(cos(solo[b][k], batch[k][b]) for b in range(3) for k in range(5))
        print(f"batched decode vs solo: worst logit cos {c_:.5f}")
        ok &= c_ > 0.999
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
