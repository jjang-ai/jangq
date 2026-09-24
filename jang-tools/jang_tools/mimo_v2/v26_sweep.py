"""Calibrated per-layer candidate sweep on the EXACT source (drives allocation).

For each MoE layer L and each candidate expert layout c, a stream is identical
to the source up to L, uses the CALIBRATED quantization of c at L (imatrix fit
+ AWQ fold exactly as convert_v26 will do it), and runs source afterwards.
Cost of (L, c) = KL(source || stream) of the final logits (mean over tokens).

Step 1 per layer picks the AWQ alpha from {0 (control), 0.15, 0.25} on the
cheapest layout (gate 2/g64, up 2/g64, down 2/g64). Layouts with a native
MXFP4 gate cannot carry an AWQ fold (shared norm) and always run alpha=0.

Output JSON: per layer {alpha_kl: {...}, alpha, cands: {name: {kl, top1, bytes}}}.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.switch_layers import SwitchLinear

from .convert_v26 import expert_unit
from .v26_source import SourceStream, layer_masks

A = lambda b, g: {"mode": "affine", "bits": b, "group_size": g}
N4 = {"mode": "mxfp4"}
CANDS = {
    "A_222": (A(2, 64), A(2, 64), A(2, 64)),
    "B_2u2": (A(2, 64), A(2, 128), A(2, 64)),
    "C_322": (A(3, 64), A(2, 64), A(2, 64)),
    "D_3u2": (A(3, 64), A(2, 128), A(2, 64)),
    "E_N22": (N4, A(2, 64), A(2, 64)),
    "F_Nu2": (N4, A(2, 128), A(2, 64)),
    "G_323": (A(3, 64), A(2, 64), A(3, 64)),
    "H_N23": (N4, A(2, 64), A(3, 64)),
}
FULL_NATIVE = ("J_NNN", (N4, N4, N4))
ALPHAS = (0.0, 0.15, 0.25)
PROJS = ("gate_proj", "up_proj", "down_proj")


def unit_bytes(spec, out_dim, in_dim, n_exp):
    if spec["mode"] == "mxfp4":
        return n_exp * (out_dim * in_dim // 2 + out_dim * in_dim // 32)
    b, g = spec["bits"], spec["group_size"]
    return n_exp * (out_dim * in_dim * b // 8 + out_dim * (in_dim // g) * 2 * 2)


def awq_scale(stats, L, alpha):
    if alpha == 0:
        return None
    a = stats[f"{L}.moe_absmax"].astype(mx.float32)
    s = mx.power(mx.maximum(a, 1e-6), alpha)
    s = s / mx.exp(mx.mean(mx.log(s)))
    return mx.clip(s, 0.5, 2.0)


def expert_importance(stats, L, proj, s, min_count=16):
    key = f"{L}.down_in" if proj == "down_proj" else f"{L}.moe_in"
    imp = stats[key].astype(mx.float32)
    cnt = stats[f"{L}.down_in_count" if proj == "down_proj" else f"{L}.moe_count"]
    fallback = imp.mean(0) if proj == "down_proj" else stats[f"{L}.moe_in_all"].astype(mx.float32)
    imp = mx.where((cnt >= min_count)[:, None], imp, fallback[None, :])
    if s is not None and proj != "down_proj":
        imp = imp / (s * s)[None, :]
    return imp


def build_switch(t, spec, in_dim, out_dim, n_exp):
    q = SwitchLinear(in_dim, out_dim, n_exp, bias=False)
    if spec["mode"] == "mxfp4":
        q = q.to_quantized(group_size=32, bits=4, mode="mxfp4")
        q.weight, q.scales = t["weight"], t["scales"]
    else:
        q = q.to_quantized(group_size=spec["group_size"], bits=spec["bits"], mode="affine")
        q.weight, q.scales, q.biases = t["weight"], t["scales"], t["biases"]
    return q


def kl_top1(ref_lp, ref_top, logits):
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    kl = (mx.exp(ref_lp) * (ref_lp - lp)).sum(-1)
    return float(kl.mean()), float((mx.argmax(logits, -1) == ref_top).astype(mx.float32).mean())


def run(src, tokens, stats_path, out, full_native_layers):
    ss = SourceStream(src)
    a = ss.args
    stats = mx.load(str(stats_path))
    ids = mx.array(tokens)
    B, T = ids.shape
    full_m, swa_m = layer_masks(a, T)
    H, I, E = a.hidden_size, a.moe_intermediate_size, a.n_routed_experts
    dims = {"gate_proj": (I, H), "up_proj": (I, H), "down_proj": (H, I)}
    h_src = ss.embed(ids)
    streams = {}
    result = {}
    t0 = time.time()
    for L in range(a.num_hidden_layers):
        lay = ss.build_layer(L)
        mask = swa_m if lay.is_swa else full_m
        for k in list(streams):
            streams[k] = lay(streams[k], mask)
            # Materialize before moving on: otherwise every candidate retains
            # the graphs and source weights of all subsequent layers until
            # final scoring, defeating layer-streamed memory bounds.
            mx.eval(streams[k])
        if a.moe_layer_freq[L]:
            sw = lay.mlp.switch_mlp
            src_mods = {p: getattr(sw, p) for p in PROJS}
            norm_w = lay.post_attention_layernorm.weight
            router_w = lay.mlp.gate.weight
            cache = {}

            def module(p, spec, alpha):
                key = (p, json.dumps(spec, sort_keys=True), alpha if p != "down_proj" else 0.0)
                if key not in cache:
                    s = awq_scale(stats, L, alpha) if spec["mode"] == "affine" else None
                    imp = expert_importance(stats, L, p, s) if spec["mode"] == "affine" else None
                    t = expert_unit(ss.idx, L, p, spec, E, s if p != "down_proj" else None, imp)
                    cache[key] = build_switch(t, spec, dims[p][1], dims[p][0], E)
                    mx.eval(cache[key].parameters())
                return cache[key]

            def run_layout(name, layout, alpha):
                s = awq_scale(stats, L, alpha)
                if s is not None:
                    lay.post_attention_layernorm.weight = (norm_w.astype(mx.float32) / s).astype(norm_w.dtype)
                    lay.mlp.gate.weight = router_w * s
                for p, spec in zip(PROJS, layout):
                    setattr(sw, p, module(p, spec, alpha))
                streams[(L, name)] = lay(h_src, mask)
                mx.eval(streams[(L, name)])
                lay.post_attention_layernorm.weight, lay.mlp.gate.weight = norm_w, router_w
                for p in PROJS:
                    setattr(sw, p, src_mods[p])

            for al in ALPHAS:
                run_layout(f"alpha_{al}", CANDS["A_222"], al)
            result[L] = {"pending_alpha": True}
            cands = dict(CANDS)
            if L in full_native_layers:
                cands[FULL_NATIVE[0]] = FULL_NATIVE[1]
            # alpha is chosen after the pass (needs final logits); run every
            # affine-gate candidate under every alpha is too costly, so we run
            # the non-A candidates under alpha 0.15 (the 2-bit rule of thumb)
            # AND record the A_222 alpha curve; the allocator only uses a
            # candidate's measured (alpha, kl) pair.
            for name, layout in cands.items():
                if name == "A_222":
                    continue
                al = 0.0 if layout[0]["mode"] == "mxfp4" else 0.15
                run_layout(f"{name}@{al}", layout, al)
            result[L]["bytes"] = {name: sum(unit_bytes(sp, *dims[p], E) for p, sp in zip(PROJS, lay_))
                                  for name, lay_ in cands.items()}
            del cache
        h_src = lay(h_src, mask)
        mx.eval(h_src)
        del lay
        mx.clear_cache()
        print(f"[sweep] layer {L} {time.time()-t0:.0f}s streams={len(streams)} "
              f"active_gib={mx.get_active_memory()/2**30:.2f} "
              f"peak_gib={mx.get_peak_memory()/2**30:.2f}", flush=True)
    ref = ss.head(h_src).astype(mx.float32).reshape(-1, a.vocab_size)
    ref_lp = ref - mx.logsumexp(ref, axis=-1, keepdims=True)
    ref_top = mx.argmax(ref, -1)
    mx.eval(ref_lp, ref_top)
    for (L, name), h in streams.items():
        kl, t1 = kl_top1(ref_lp, ref_top, ss.head(h).astype(mx.float32).reshape(-1, a.vocab_size))
        result[L].setdefault("kl", {})[name] = {"kl": kl, "top1": t1}
    for L, r in result.items():
        r.pop("pending_alpha", None)
    Path(out).write_text(json.dumps({str(k): v for k, v in result.items()}, indent=1))
    print(f"[sweep] DONE {time.time()-t0:.0f}s -> {out}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--tokens", type=Path, required=True, help="int32 npy [B, T]")
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--full-native-layers", default="1,4,5,6,7,47")
    a = ap.parse_args(argv)
    run(a.src, np.load(a.tokens), a.stats, a.out, {int(x) for x in a.full_native_layers.split(",") if x})


if __name__ == "__main__":
    main()
