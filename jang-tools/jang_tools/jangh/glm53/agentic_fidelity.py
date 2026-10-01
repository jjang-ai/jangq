"""Agentic fidelity vs the layer-streamed bf16 reference.

For each held-out ("ref") sequence of agentic_corpus.jsonl: teacher-forced logits from the bundle under test vs the
bf16 reference top-128 (agentic_ref.safetensors from stream_capture.py). Reports
  * KL (top-128 renormalized, same math as jang_tools.glm5_next.kl_eval) over all positions: median/mean/p90/p99, top-1
  * DECISION positions (next token after "<|assistant|><think></think>"): argmax agreement with bf16, and for "call"
    decisions the model's P(<tool_call>) vs bf16's, and the count of decisions whose argmax flips call<->answer.
Loader: --kind tq (JANGTQ v2, jang_tools.jangh.glm53.load_tq) | affine (jang_tools.glm5_next.load.load_bundle).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np



TOOL_CALL = 154843


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True); ap.add_argument("--kind", choices=["tq", "affine"], required=True)
    ap.add_argument("--corpus", required=True); ap.add_argument("--ref", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.kind == "tq":
        from jang_tools.jangh.glm53.load_tq import load_tq_bundle
        model = load_tq_bundle(a.bundle)
    else:
        from jang_tools.glm5_next.load import load_bundle
        model = load_bundle(a.bundle)
    ref = mx.load(a.ref)
    rows = [json.loads(l) for l in open(a.corpus)]
    kls, top1 = [], []
    dec = {"call": {"agree": 0, "n": 0, "p_ref": [], "p_q": [], "flips": 0}, "answer": {"agree": 0, "n": 0, "flips": 0},
           "end_think": {"agree": 0, "n": 0, "flips": 0}}
    per_seq = []
    for j, r in enumerate(rows):
        if not r.get("ref") or f"p{j}.top_ids" not in ref:
            continue
        ids = np.array(r["ids"])
        logits = model(mx.array(ids[None]))[0].astype(mx.float32)
        lp = np.asarray(logits - mx.logsumexp(logits, axis=-1, keepdims=True), dtype=np.float64)
        rt = np.asarray(ref[f"p{j}.top_ids"]); rl = np.asarray(ref[f"p{j}.top_logprobs"]).astype(np.float64)
        for t in range(len(ids) - 1):
            rp = np.exp(rl[t]); rp /= rp.sum()
            qp = np.exp(lp[t, rt[t]]); qp /= max(qp.sum(), 1e-12)
            kls.append(float(np.sum(rp * (np.log(rp + 1e-12) - np.log(qp + 1e-12)))))
            top1.append(int(rt[t][np.argmax(rl[t])]) == int(np.argmax(lp[t])))
        for p, kind in r["decisions"]:
            ref_arg = int(rt[p][np.argmax(rl[p])]); q_arg = int(np.argmax(lp[p]))
            d = dec[kind]; d["n"] += 1; d["agree"] += int(ref_arg == q_arg)
            if (ref_arg == TOOL_CALL) != (q_arg == TOOL_CALL):
                d["flips"] += 1
            if kind == "call":
                pos = np.nonzero(rt[p] == TOOL_CALL)[0]
                d["p_ref"].append(float(np.exp(rl[p][pos[0]])) if len(pos) else 0.0)
                d["p_q"].append(float(np.exp(lp[p, TOOL_CALL])))
                per_seq.append({"seq": j, "domain": r.get("domain"), "pos": p, "p_toolcall_ref": d["p_ref"][-1],
                                "p_toolcall_q": d["p_q"][-1], "ref_argmax": ref_arg, "q_argmax": q_arg})
        mx.clear_cache()
    k = np.array(kls)
    out = {"bundle": a.bundle, "positions": len(k), "median_kl": float(np.median(k)), "mean_kl": float(k.mean()),
           "p90": float(np.percentile(k, 90)), "p99": float(np.percentile(k, 99)), "top1_pct": 100 * float(np.mean(top1)),
           "decisions": {kk: {"n": v["n"], "argmax_agree_pct": 100 * v["agree"] / max(v["n"], 1), "call_answer_flips": v["flips"],
                              **({"median_p_toolcall_ref": float(np.median(v["p_ref"])), "median_p_toolcall_q": float(np.median(v["p_q"])),
                                  "min_p_toolcall_q": float(np.min(v["p_q"]))} if kk == "call" and v["p_q"] else {})}
                         for kk, v in dec.items()}, "per_decision": per_seq if len(per_seq) <= 16 else []}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
