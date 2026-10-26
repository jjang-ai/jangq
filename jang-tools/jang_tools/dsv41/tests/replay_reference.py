"""Strict port check: replay the release reference's per-block inputs through the MLX blocks.

End-to-end logits on a random MoE drift on routing near-ties (bf16 noise flips a
top-k expert and the error compounds), so parity_reference.py is a loose check.
Here every MLX attention / block / Engram call receives the reference's exact
input for that layer and step, over a single-shot prefill plus teacher-forced
decode, so cache, compressor-state, KV-sharing, indexer and hc logic are checked
without compounding. Selection covers all positions (no near-tie top-k flips).
Block outputs whose MoE router has a near-tie (top-k margin < 0.02) are reported
but not failed.

  PYTHONPATH=jang-tools python jang_tools/dsv41/tests/replay_reference.py --src ~/models/DeepSeek-V4.1-Flash-1017
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from jang_tools.dsv41.tests import parity_reference as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--prompt-len", type=int, default=70)
    ap.add_argument("--decode", type=int, default=40)
    ap.add_argument("--tol", type=float, default=0.02)
    a = ap.parse_args()
    src = Path(a.src).expanduser()
    sys.path[:0] = [str(Path(P.__file__).parent / "ref_shim"), str(src / "inference")]
    import model as ref
    from transformers import AutoTokenizer

    import mlx.core as mx
    from jang_tools.dsv41 import model as M

    torch.set_default_dtype(torch.bfloat16)
    text = P.build_config()
    text.update(index_topk=512, candidate_topk_blocks=512)
    tok = AutoTokenizer.from_pretrained(str(src))
    rm = ref.Transformer(P.ref_args(ref, text), tok)
    torch.set_default_dtype(torch.float32)
    P.randomize(rm)
    n = a.prompt_len
    ids = tok.encode((src / "encoding" / "README.md").read_text(), add_special_tokens=False)[: n + a.decode]

    rec = {"blk": [], "attn": [], "eng": []}

    def wrap(cls, key, fn):
        orig = cls.forward

        def f(self, *args, **kw):
            out = orig(self, *args, **kw)
            rec[key].append(fn(args, out))
            return out

        cls.forward = f

    cpy = lambda t: t.float().numpy().copy()  # noqa: E731
    wrap(ref.Block, "blk", lambda ar, o: (cpy(ar[0]), cpy(ar[2]), cpy(o[0]), cpy(o[1])))
    wrap(ref.Attention, "attn", lambda ar, o: (cpy(ar[0]), cpy(o)))
    wrap(ref.Engram, "eng", lambda ar, o: (cpy(ar[0]), cpy(o)))

    with tempfile.TemporaryDirectory() as td:
        bundle = Path(td) / "bundle"
        P.write_bundle(rm, text, src, bundle)
        with torch.inference_mode():
            x = torch.tensor([ids])
            rm.forward(x[:, :n], 0)
            for i in range(n, len(ids) - 1):
                rm.forward(x[:, i : i + 1], i)
        m = M.DSV41(bundle, wired=False)

    layers = m.model.layers
    nl = len(layers)
    steps = len(rec["blk"]) // nl

    def rel(o, r, base=None):
        return float(np.linalg.norm(o - r) / np.linalg.norm(r if base is None else base))

    # attention replay (own caches)
    caches = [M.LayerCache() for _ in layers]
    worst_attn = 0.0
    for t in range(steps):
        sp = 0 if t == 0 else n - 1 + t
        seg = M.Seg(0, n if t == 0 else 1, sp, caches)
        for i, layer in enumerate(layers):
            xin, out = rec["attn"][t * nl + i]
            o = layer.attn(mx.array(xin).astype(mx.bfloat16), [seg])
            worst_attn = max(worst_attn, rel(np.array(o.astype(mx.float32)), out))

    # block + engram replay (fresh caches)
    sess = m.new_session()
    caches = sess.caches
    worst_blk, worst_eng, ties, k = 0.0, 0.0, 0, 0
    for t in range(steps):
        sp = 0 if t == 0 else n - 1 + t
        tk = ids[:n] if t == 0 else [ids[n - 1 + t]]
        seg = M.Seg(0, len(tk), sp, caches)
        rows = m._engram_rows(np.array(tk), sp, sess)
        for i, layer in enumerate(layers):
            if i in rows:
                ein, eout = rec["eng"][k]
                k += 1
                o = layer.engram(mx.array(ein).astype(mx.bfloat16), rows[i])
                worst_eng = max(worst_eng, rel(np.array(o.astype(mx.float32)), eout, eout - ein))
            xin, pm, out, opm = rec["blk"][t * nl + i]
            h, _ = layer(mx.array(xin).astype(mx.bfloat16), [seg], mx.array(pm))
            e = rel(np.array(h.astype(mx.float32)), out)
            if e > a.tol:
                ties += 1
                print(f"  block {i} step {t}: rel {e:.3f} (router near-tie suspected)")
            else:
                worst_blk = max(worst_blk, e)

    print(f"attention worst rel {worst_attn:.4f} | block worst rel {worst_blk:.4f} "
          f"({ties} near-tie outliers) | engram worst rel-of-delta {worst_eng:.4f}")
    ok = worst_attn < a.tol and worst_blk < a.tol and worst_eng < a.tol and ties <= 2
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
