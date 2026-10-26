"""Collect per-expert activation statistics for DSV4.1-1017 expert requantization.

Runs the local bundle forward (prefill only) over the traced calibration
conversations and, at every routed MoE layer, accumulates for each expert e:
  gate/up input (dim 5120):  sum x, sum x^2, sum |x|, token count
  down input    (dim 2304):  sum h, sum h^2, sum |h|, token count
(the down input is the post-SwiGLU, post-routing-weight intermediate actually
fed to down_proj). These give both AWQ per-channel scales (mean |x|) and the
imatrix diagonal importance (mean x^2) per expert, which the requantizer uses.

Streaming: stats live in ~0.9 GB of fp32 accumulators; activations are freed per
row. Output: one .npz of {layer}.{proj}.{sq,abs,cnt} plus meta.

  python -m jang_tools.dsv41.calib_stats --bundle <bundle> \
      --rows calib/dsv41-1017/traced_technical.jsonl --out calib/dsv41-1017/stats.npz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


class Accumulators:
    def __init__(self, n_layers, n_experts, d_in, d_inter):
        z = lambda d: mx.zeros((n_layers, n_experts, d), mx.float32)  # noqa: E731
        self.gu_sq, self.gu_abs = z(d_in), z(d_in)
        self.dn_sq, self.dn_abs = z(d_inter), z(d_inter)
        self.gu_cnt = mx.zeros((n_layers, n_experts), mx.float32)

    def capture(self, layer, e_idx, gu_in, dn_in):
        gu = gu_in.astype(mx.float32)
        dn = dn_in.astype(mx.float32)
        self.gu_sq[layer] = self.gu_sq[layer].at[e_idx].add(gu * gu)
        self.gu_abs[layer] = self.gu_abs[layer].at[e_idx].add(mx.abs(gu))
        self.dn_sq[layer] = self.dn_sq[layer].at[e_idx].add(dn * dn)
        self.dn_abs[layer] = self.dn_abs[layer].at[e_idx].add(mx.abs(dn))
        self.gu_cnt[layer] = self.gu_cnt[layer].at[e_idx].add(mx.ones(e_idx.shape, mx.float32))

    def save(self, path, meta):
        mx.eval(self.gu_sq, self.gu_abs, self.dn_sq, self.dn_abs, self.gu_cnt)
        np.savez(path, gu_sq=np.array(self.gu_sq), gu_abs=np.array(self.gu_abs),
                 dn_sq=np.array(self.dn_sq), dn_abs=np.array(self.dn_abs),
                 gu_cnt=np.array(self.gu_cnt), meta=json.dumps(meta))


def render_full(enc, row):
    """The whole conversation as one prefill string (thinking kept on every assistant turn)."""
    import copy

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
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    from jang_tools.dsv41.model import DSV41
    from jang_tools.dsv41.generate import load_encoding

    bundle = Path(a.bundle).expanduser()
    enc = load_encoding(bundle)
    m = DSV41(bundle, a.engram_dir)
    acc = Accumulators(m.args.num_hidden_layers, m.args.n_routed_experts, m.args.hidden_size, m.args.moe_intermediate_size)
    for layer in m.model.layers:
        layer.capture = acc.capture

    rows = [json.loads(l) for l in open(a.rows)]
    if a.limit:
        rows = rows[: a.limit]
    t0 = time.time()
    tot_tok = 0
    for i, r in enumerate(rows):
        text = render_full(enc, r)
        ids = m.tokenizer.encode(text, add_special_tokens=False).ids[: a.max_tokens]
        if len(ids) < 2:
            continue
        sess = m.new_session()
        m.forward(ids, sess)  # prefill; capture fires inside every MoE layer
        # Force ALL accumulators to materialize now. Evaluating only gu_cnt leaves the
        # other four scatter-add graphs lazy, pinning every row's activations and OOMing.
        mx.eval(acc.gu_sq, acc.gu_abs, acc.dn_sq, acc.dn_abs, acc.gu_cnt)
        del sess
        mx.clear_cache()
        tot_tok += len(ids)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(rows)} rows, {tot_tok} tok ({time.time() - t0:.0f}s)", flush=True)
    meta = {"rows": len(rows), "tokens": tot_tok, "n_layers": m.args.num_hidden_layers,
            "n_experts": m.args.n_routed_experts, "d_in": m.args.hidden_size,
            "d_inter": m.args.moe_intermediate_size, "source_bundle": str(bundle)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    acc.save(a.out, meta)
    print(f"saved {a.out}: {meta['tokens']} tokens over {meta['rows']} rows in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
