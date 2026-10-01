"""Streamed-KL harness for Naive-N0.5-Flash.

Layer-streams the BF16 source over the held-out klref sequences (and optionally agentic_ref), applying a fake-quant
transform to each layer's weights on the fly, then scores KL / top-1 / top-5 against the BF16 reference logits
(n05_ref.safetensors from stream_capture). Lets us A/B format choices WITHOUT building a 96 GiB bundle.

  --nonexpert bf16 | affine8 | mxfp8        attention q/k/v/o, layer-0 dense MLP, embed, lm_head (indexer kept, as in the plan)
  --experts   bf16 | rtn:<rot>:<gu_bits>:<down_bits>    (rot = none|hadamard32; codebook per-row scale, imatrix-weighted
                                                          LS scale when rot=none)
                     | noise:<sigma>                    (CHAOS FLOOR: BF16 everywhere, relative Gaussian noise sigma on
                                                          layer 1's down_proj only; what the model does to itself)
                     | rtnplan:<plan.json>              (hadamard32 codebook RTN with the plan's per-layer gate/up/down
                                                          bits: A/B of two ALLOCATIONS without building a bundle)
Metrics use the same math as the GLM harness: top-128-renormalized KL(ref || q) per position, median/mean/p90/p99,
top-1 agreement, ref-top1-in-q-top5.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np



from jang_tools.jangh.n05.model import Args, DecoderLayer, sanitize_layer  # noqa: E402
from jang_tools.jangh.n05.stream_capture import load_layer_raw  # noqa: E402
from jang_tools.jangh.n05.metrics import (call_line, compact, score_compact, next_ids, summarize, by_domain, line,  # noqa: E402
                         decisions_new, decisions_update, decisions_summary)


def fq_affine8(w):
    wb = w.astype(mx.bfloat16)
    q, s, b = mx.quantize(wb, group_size=64, bits=8)
    return mx.dequantize(q, s, b, group_size=64, bits=8).astype(mx.bfloat16)


def fq_mxfp8(w):
    W = w.astype(mx.float32); o, i = W.shape
    G = W.reshape(o, i // 32, 32)
    a = mx.maximum(mx.abs(G).max(-1, keepdims=True), 1e-30)
    e = mx.ceil(mx.log2(a / 448.0))
    best, berr = None, None
    for de in (0, -1):
        sc = 2.0 ** (e + de)
        rt = mx.from_fp8(mx.to_fp8(mx.clip(G / sc, -448, 448)), dtype=mx.float32) * sc
        err = ((G - rt) ** 2).sum(-1, keepdims=True)
        best = rt if best is None else mx.where(err < berr, rt, best)
        berr = err if berr is None else mx.minimum(err, berr)
    return best.reshape(o, i).astype(mx.bfloat16)


NONEXP = {"affine8": fq_affine8, "mxfp8": fq_mxfp8, "bf16": lambda w: w}


def is_nonexpert_linear(k, v):
    # same rule as plan.classify: the DSA indexer, norms and routers are KEPT (never quantized)
    return (v.ndim == 2 and k.endswith(".weight") and "switch_mlp" not in k and "norm" not in k and ".indexer." not in k
            and "mlp.gate.weight" not in k and v.shape[1] % 64 == 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--headers", required=True)
    ap.add_argument("--seqs", required=True); ap.add_argument("--ref", default=None)
    ap.add_argument("--make-ref", default=None, help="BF16 only: WRITE a reference (top-128 ids/log-probs + exact NLL) for the selected sequences instead of scoring")
    ap.add_argument("--stats", default=None)
    ap.add_argument("--nonexpert", default="affine8", choices=list(NONEXP))
    ap.add_argument("--experts", default="bf16")
    ap.add_argument("--sets", default="klref")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-nll", default=None, help="write the evaluated model's NLL of the actual next token per sequence (run on the BF16 source = exact reference NLL)")
    ap.add_argument("--ref-nll", default=None, help="exact reference NLL file (from --save-nll on the BF16 source)")
    a = ap.parse_args()
    src = Path(a.model)
    A = Args.from_config(json.loads((src / "config.json").read_text()))
    H = json.loads(Path(a.headers).read_text())
    sets = set(a.sets.split(","))
    rows = [json.loads(l) for l in open(a.seqs)]
    sel = [(j, r) for j, r in enumerate(rows) if r.get("set") in sets]
    if a.make_ref:
        assert a.nonexpert == "bf16" and a.experts == "bf16" and a.ref is None, "--make-ref is the unmodified BF16 forward"
        ref = None
    else:
        ref = mx.load(a.ref)
        sel = [(j, r) for j, r in sel if f"p{j}.top_ids" in ref]
    if not sel:
        print("INVALID: no sequences selected"); sys.exit(2)
    print(f"{len(sel)} sequences, {sum(len(r['ids']) for _, r in sel)} tokens; nonexpert={a.nonexpert} experts={a.experts}", flush=True)
    fq = NONEXP[a.nonexpert]
    st = mx.load(a.stats) if a.stats else None
    ex_spec = a.experts.split(":", 1) if a.experts.startswith("rtnplan:") else a.experts.split(":")
    pbits = json.loads(Path(ex_spec[1]).read_text())["expert_bits"] if ex_spec[0] == "rtnplan" else None
    t0 = time.time()
    emb = mx.load(str(src / H["model.embed_tokens.weight"][0]))["model.embed_tokens.weight"]
    emb = fq(emb)
    hs = []
    for _, r in sel:
        h = emb[mx.array(r["ids"])][None].astype(mx.bfloat16); mx.eval(h); hs.append(h)
    del emb; mx.clear_cache()
    cache = {}
    for i in range(A.num_hidden_layers):
        layer = DecoderLayer(A, i)
        w = sanitize_layer(load_layer_raw(src, H, i, cache), i, A.n_routed_experts)
        w = {k: (fq(v) if is_nonexpert_linear(k, v) else v) for k, v in w.items()}
        if ex_spec[0] == "noise" and i == 1:
            k = "mlp.switch_mlp.down_proj.weight"; mx.random.seed(4242); outs = []
            for e0 in range(0, A.n_routed_experts, 32):
                Wc = w[k][e0:e0 + 32].astype(mx.float32)
                outs.append((Wc + float(ex_spec[1]) * mx.sqrt(mx.mean(Wc * Wc, axis=-1, keepdims=True)) * mx.random.normal(Wc.shape)).astype(mx.bfloat16))
                mx.eval(outs[-1])
            w[k] = mx.concatenate(outs, axis=0)
        if ex_spec[0] in ("rtn", "rtnplan") and A.moe_layer_freq[i]:
            from jang_tools.jangh.encode import encode, dequant
            from jang_tools.jangh.format import h32
            if pbits is not None:
                rot = "hadamard32"; bset = {p_: int(pbits[f"{i}:{p_}"]) for p_ in ("gate", "up", "down")}
            else:
                rot = ex_spec[1]; bset = {"gate": int(ex_spec[2]), "up": int(ex_spec[2]), "down": int(ex_spec[3])}
            R = h32 if rot == "hadamard32" else (lambda M: M)
            b = f"model.layers.{i}.mlp.experts"
            cnt = st[b + ".expert_count"].astype(mx.float32)
            imx = st[b + ".expert_sum_x2"].astype(mx.float32) / mx.maximum(cnt, 1.0)[:, None]
            imxd = st[f"model.layers.{i}.mlp.switch_mlp.down_proj.expert_sum_a2"].astype(mx.float32) / mx.maximum(cnt, 1.0)[:, None]
            for p, bits, D in (("gate", bset["gate"], imx), ("up", bset["up"], imx), ("down", bset["down"], imxd)):
                k = f"mlp.switch_mlp.{p}_proj.weight"
                Wf = w[k]; outs = []
                for e0 in range(0, A.n_routed_experts, 32):
                    Wc = R(Wf[e0:e0 + 32].astype(mx.float32))
                    q, s = encode(Wc, bits, None if rot == "hadamard32" else D[e0:e0 + 32][:, None, :])
                    outs.append(R(dequant(q, s, bits)).astype(mx.bfloat16)); mx.eval(outs[-1])
                w[k] = mx.concatenate(outs, axis=0)
        layer.load_weights(list(w.items()), strict=True)
        mx.eval(layer.parameters())
        for j in range(len(hs)):
            hs[j] = layer(hs[j]); mx.eval(hs[j])
        del layer, w; mx.clear_cache()
        if i % 8 == 0 or i == A.num_hidden_layers - 1:
            print(f"layer {i} done {(time.time()-t0)/60:.1f} min", flush=True)
    normw = mx.load(str(src / H["model.norm.weight"][0]))["model.norm.weight"].astype(mx.bfloat16)
    head = fq(mx.load(str(src / H["lm_head.weight"][0]))["lm_head.weight"]).astype(mx.bfloat16)
    if a.make_ref:
        out = {}
        for (j, r), h in zip(sel, hs):
            lg = (mx.fast.rms_norm(h, normw, A.layernorm_epsilon) @ head.T)[0].astype(mx.float32)
            lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
            top = mx.argsort(-lp, axis=-1)[:, :128]
            ids_ = np.array(r["ids"], np.int32)
            out[f"p{j}.input_ids"] = mx.array(ids_); out[f"p{j}.top_ids"] = top.astype(mx.int32)
            out[f"p{j}.top_logprobs"] = mx.take_along_axis(lp, top, axis=-1)
            out[f"p{j}.nll_actual"] = -mx.take_along_axis(lp[:-1], mx.array(ids_[1:])[:, None], axis=-1)[:, 0]
            mx.eval([out[k] for k in list(out)[-4:]]); mx.clear_cache()
        mx.save_safetensors(a.make_ref, out, metadata={"reference_precision": "BF16 (layer-streamed source)"})
        print(f"REFERENCE WRITTEN: {len(sel)} sequences -> {a.make_ref}", flush=True); return
    parts, doms, nll_out = [], [], {}
    rn = mx.load(a.ref_nll) if a.ref_nll else None
    from transformers import AutoTokenizer
    TC = AutoTokenizer.from_pretrained(str(Path(a.headers).resolve().parent)).convert_tokens_to_ids("<tool_call>")
    assert isinstance(TC, int) and TC > 0
    TCE = AutoTokenizer.from_pretrained(str(Path(a.headers).resolve().parent)).convert_tokens_to_ids("</tool_call>")
    dec = decisions_new()
    for (j, r), h in zip(sel, hs):
        lg = (mx.fast.rms_norm(h, normw, A.layernorm_epsilon) @ head.T)[0]
        rt, rl = ref[f"p{j}.top_ids"], ref[f"p{j}.top_logprobs"]
        cp = compact(lg, rt, next_ids(r["ids"]), extra_ids=(TC,))
        if not cp["finite"]:
            print(f"INVALID: non-finite logits in sequence {j}"); sys.exit(2)
        parts.append(score_compact(cp, rt, rl, r["ids"], rn[f"p{j}.nll_actual"] if rn is not None else None, TC, TCE))
        decisions_update(dec, r.get("decisions", []), cp, rt, rl, TC)
        doms.append(r["domain"]); nll_out[f"p{j}.nll_actual"] = mx.array(parts[-1]["nll_q"])
        del lg; mx.clear_cache()
    tot = summarize(parts)
    if tot is None:
        print("INVALID: zero reference positions or non-finite values"); sys.exit(2)
    if a.save_nll:
        if a.nonexpert != "bf16" or a.experts != "bf16" or tot["mean_kl"] > 1e-6:
            print("INVALID: --save-nll is the REFERENCE NLL: it needs the unmodified BF16 forward reproducing the reference (KL 0)"); sys.exit(2)
        mx.save_safetensors(a.save_nll, nll_out)
    res = {"nonexpert": a.nonexpert, "experts": a.experts, "sets": a.sets, **tot, "per_domain": by_domain(parts, doms),
           "decisions": decisions_summary(dec),
           "minutes": (time.time() - t0) / 60}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(line("ALL", tot), flush=True)
    if call_line(tot):
        print(call_line(tot), flush=True)
    for d, v in res["per_domain"].items():
        print("  " + line(d, v), flush=True)
    if res["decisions"]:
        print("decisions", json.dumps(res["decisions"]), flush=True)
    print("STREAM-EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
