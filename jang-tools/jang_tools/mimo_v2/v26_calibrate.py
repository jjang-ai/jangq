"""MiMo-V2.6 calibration capture on the EXACT source (layer-streamed).

One pass over the calibration corpus produces everything the calibrated trio
needs (feedback_calibrated_trio_mandatory):
  * imatrix / Hessian diagonal  E[x_c^2] for every quantized linear input
      {L}.attn_in [H]        -> qkv_proj
      {L}.o_in    [nH*vD]    -> o_proj
      {L}.moe_in  [E, H]     -> per-expert gate/up (router-selected tokens only)
      {L}.down_in [E, I]     -> per-expert down_proj
      0.dense_in [H], 0.dense_down_in [16384], final_norm [H] (lm_head)
  * AWQ statistics for the post_attention_layernorm fold
      {L}.moe_absmax [H], {L}.moe_absmean [H], {L}.moe_in_all [H]
  * routing coverage {L}.moe_count [E]
Hessian trace tr(H) per unit = sum_c E[x_c^2] is written to calib_meta.json.

Statistics are means over tokens (per expert: over the tokens routed to it).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .v26_source import SourceStream, layer_masks


class _Acc:
    def __init__(self):
        self.d = {}

    def add(self, key, sumsq, count):
        if key in self.d:
            s, c = self.d[key]
            self.d[key] = (s + sumsq, c + count)
        else:
            self.d[key] = (sumsq, count)


class RecLinear(nn.Module):
    """Wraps a dense Linear; records E[x^2] of its input."""

    def __init__(self, inner, acc, key):
        super().__init__()
        self.inner, self._acc, self._key = inner, acc, key

    def __call__(self, x):
        x2 = x.astype(mx.float32).reshape(-1, x.shape[-1])
        self._acc.add(self._key, (x2 * x2).sum(0), x2.shape[0])
        return self.inner(x)


class RecSwitch(nn.Module):
    """Wraps a (Quantized)SwitchLinear; records per-expert E[x^2] of its input."""

    def __init__(self, inner, acc, key, n_exp):
        super().__init__()
        self.inner, self._acc, self._key, self._E = inner, acc, key, n_exp

    def __call__(self, x, indices, sorted_indices=False):
        D = x.shape[-1]
        xf = x.astype(mx.float32).reshape(-1, D)
        idx = indices.reshape(-1)
        if xf.shape[0] != idx.shape[0]:
            # unsorted path: x is broadcast over the K selected experts
            k = idx.shape[0] // xf.shape[0]
            xf = mx.repeat(xf, k, axis=0)
        sq = mx.zeros((self._E, D), dtype=mx.float32).at[idx].add(xf * xf)
        cnt = mx.zeros((self._E,), dtype=mx.float32).at[idx].add(1.0)
        self._acc.add(self._key, sq, cnt)
        return self.inner(x, indices, sorted_indices=sorted_indices)


def run(src: Path, tokens: np.ndarray, out: Path, batch: int = 8):
    out.mkdir(parents=True, exist_ok=True)
    ss = SourceStream(src)
    a = ss.args
    N, T = tokens.shape
    # Trailing <|endoftext|> padding would pollute per-expert stats. Causal
    # attention means trailing pads never affect earlier tokens, so sort
    # sequences by real length and trim every batch to its longest member.
    pad = 151643
    real = np.array([T - int(np.argmax(r[::-1] != pad)) if (r != pad).any() else 0 for r in tokens])
    order = np.argsort(-real)
    batches = []
    for i in range(0, N, batch):
        sel = order[i:i + batch]
        tl = int(real[sel].max())
        batches.append(tokens[sel, :tl])
    kept = sum(b.size for b in batches)
    print(f"[calib] {N} seqs, {int(real.sum())} real tokens, {kept} processed (pad kept {kept-int(real.sum())})", flush=True)
    masks = {}
    acc = _Acc()
    hs = [ss.embed(mx.array(b)) for b in batches]
    mx.eval(hs)
    t0 = time.time()
    meta = {"tokens": int(real.sum()), "processed_tokens": int(kept), "seq_len": int(T), "layers": {}}
    for L in range(a.num_hidden_layers):
        lay = ss.build_layer(L)
        at = lay.self_attn
        at.qkv_proj = RecLinear(at.qkv_proj, acc, f"{L}.attn_in")
        at.o_proj = RecLinear(at.o_proj, acc, f"{L}.o_in")
        moe = bool(a.moe_layer_freq[L])
        if moe:
            sw = lay.mlp.switch_mlp
            sw.gate_proj = RecSwitch(sw.gate_proj, acc, f"{L}.moe_in", a.n_routed_experts)
            sw.down_proj = RecSwitch(sw.down_proj, acc, f"{L}.down_in", a.n_routed_experts)
            absmax = mx.zeros((a.hidden_size,), dtype=mx.float32)
            abssum = mx.zeros((a.hidden_size,), dtype=mx.float32)
        else:
            lay.mlp.gate_proj = RecLinear(lay.mlp.gate_proj, acc, "0.dense_in")
            lay.mlp.down_proj = RecLinear(lay.mlp.down_proj, acc, "0.dense_down_in")
        for i, h in enumerate(hs):
            tl = h.shape[1]
            if tl not in masks:
                masks[tl] = layer_masks(a, tl)
            full_m, swa_m = masks[tl]
            mask = swa_m if lay.is_swa else full_m
            h = h + at(lay.input_layernorm(h), mask)
            xin = lay.post_attention_layernorm(h)
            if moe:
                xf = xin.astype(mx.float32).reshape(-1, a.hidden_size)
                absmax = mx.maximum(absmax, mx.abs(xf).max(0))
                abssum = abssum + mx.abs(xf).sum(0)
                acc.add(f"{L}.moe_in_all", (xf * xf).sum(0), xf.shape[0])
            hs[i] = h + lay.mlp(xin)
            mx.eval(hs[i], *[v[0] for v in acc.d.values()])
        if moe:
            ntok = N * T
            acc.d[f"{L}.moe_absmax"] = (absmax, 1)
            acc.d[f"{L}.moe_absmean"] = (abssum, ntok)
            cnt = acc.d[f"{L}.moe_in"][1]
            meta["layers"][L] = {
                "experts_seen_ge1": int((cnt >= 1).sum()),
                "experts_seen_ge64": int((cnt >= 64).sum()),
                "min_count": int(cnt.min()), "max_count": int(cnt.max()),
            }
        del lay
        mx.clear_cache()
        print(f"[calib] layer {L} {time.time()-t0:.0f}s {meta['layers'].get(L, '')}", flush=True)
    for h in hs:
        acc.add("final_norm", *(lambda z: ((z * z).sum(0), z.shape[0]))(
            mx.fast.rms_norm(h, ss.dense("model.norm.weight"), a.layernorm_epsilon).astype(mx.float32).reshape(-1, a.hidden_size)))
    # finalize means
    tensors = {}
    for k, (s, c) in acc.d.items():
        if isinstance(c, mx.array):
            mean = s / mx.maximum(c, 1.0)[:, None]
            tensors[k] = mean.astype(mx.float32)
            tensors[k.replace("_in", "_count") if k.endswith("moe_in") else k + "_count"] = c
        else:
            tensors[k] = (s / c).astype(mx.float32)
    mx.save_safetensors(str(out / "imatrix.safetensors"), tensors)
    # Hessian trace per unit (for the record / allocation diagnostics)
    trace = {k: float(v.sum()) if v.ndim == 1 else [float(x) for x in v.sum(-1).tolist()]
             for k, v in tensors.items() if k.endswith(("attn_in", "o_in", "moe_in_all", "dense_in", "dense_down_in", "final_norm"))}
    meta["hessian_trace"] = trace
    (out / "calib_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[calib] DONE {time.time()-t0:.0f}s -> {out}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--tokens", type=Path, required=True, help="int32 .npy [N, T]")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N sequences (smoke test)")
    a = ap.parse_args(argv)
    toks = np.load(a.tokens)
    if a.limit:
        toks = toks[: a.limit]
    run(a.src, toks, a.out, a.batch)


if __name__ == "__main__":
    main()
