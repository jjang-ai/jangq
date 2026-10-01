"""Shared scoring for stream_eval.py and eval_bundle.py.

Per position t (the prediction of token t+1; the last position of a sequence is dropped):
  kl        top-128-renormalized KL(ref || q)                      (same math as the GLM-5.3 harness)
  a1/a5/a10 ref top-1 == q top-1 / ref top-1 in q top-5 / top-10    (AGREEMENT with the reference)
  nll_q     -log q(actual next token)                               (QUALITY on real text, independent of the reference's
  nll_ref   -log ref(actual next token): EXACT when the reference NLL   own routing noise: see 03-CALIBRATION section 6)
            file is given (stream_eval --save-nll on the BF16 source); otherwise from the stored top-128, NaN if the
            token is not among them. 🚨 The top-128 fallback is BIASED against the evaluated model (positions where
            the reference failed are dropped, positions where the model failed are kept): use it only as a lower bound.
  hit_q / hit_ref   top-1 == actual next token
WHY both: this model's expert selection is chaotic (thin top-8 margins); agreement metrics have a FLOOR that the
unquantized model itself cannot beat when its arithmetic changes.
🚨 text_* on DOCUMENTS IS NOT A QUALITY METRIC (measured 2026-09-28): the held-out documents are rendered inside USER
turns, where a chat-tuned model is not trained to predict. The RTN affine control (KL median 1.55, 27 of 112 tool
decisions flipped) scores text PPL 0.15x the BF16 reference there. Use text_* only as |deviation from the reference|.
QUALITY without a floor = the tokens the model IS trained to emit: `call_*` = NLL / top-1 accuracy on the tokens of
correct tool calls (from the token after "</think>\n\n" through "</tool_call>") in held-out tool conversations,
reference vs model, plus the served task evals.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np


def compact(lg: mx.array, rt: mx.array, nxt: mx.array, extra_ids=(), block: int = 256) -> dict:
    """Logits (T,V) of the evaluated model -> the few numbers the metrics need, computed in row blocks so that a loaded
    96 GiB model never has to hold a second full-vocabulary tensor. rt (T,128) reference top ids, nxt (T,) the actual
    next token of every position (any id for a position without one; the caller drops it)."""
    T = lg.shape[0]
    out = {"q_ref": [], "top10": [], "nll": [], "finite": True, **{f"lp_{i}": [] for i in extra_ids}}
    for s in range(0, T, block):
        l = lg[s:s + block].astype(mx.float32)
        lp = l - mx.logsumexp(l, axis=-1, keepdims=True)
        top = mx.argpartition(-lp, kth=10, axis=-1)[:, :10]
        tv = mx.take_along_axis(lp, top, axis=-1)
        top = mx.take_along_axis(top, mx.argsort(-tv, axis=-1), axis=-1)
        q = mx.take_along_axis(lp, rt[s:s + block], axis=-1)
        n = -mx.take_along_axis(lp, nxt[s:s + block][:, None], axis=-1)[:, 0]
        ex = [lp[:, int(i)] for i in extra_ids]
        fin = mx.all(mx.isfinite(lp))
        mx.eval(top, q, n, fin, *ex)
        out["q_ref"].append(q); out["top10"].append(top); out["nll"].append(n); out["finite"] = out["finite"] and bool(fin.item())
        for i, e in zip(extra_ids, ex):
            out[f"lp_{i}"].append(e)
        del l, lp, tv
    return {k: (mx.concatenate(v, axis=0) if isinstance(v, list) else v) for k, v in out.items()}


def merge(parts: list) -> dict:
    return {k: (mx.concatenate([p[k] for p in parts], axis=0) if not isinstance(parts[0][k], bool) else all(p[k] for p in parts))
            for k in parts[0]}


def call_span_mask(ids, tc_open: int, tc_close: int) -> np.ndarray:
    """(T-1,) bool: position t predicts a token of a tool call (the opening tag, the body, the closing tag)."""
    a = np.asarray(ids); inside = np.zeros(len(a), bool); on = False
    for i, t in enumerate(a):
        if t == tc_open:
            on = True
        inside[i] = on
        if t == tc_close:
            on = False
    return inside[1:]


def score_compact(c: dict, rt: mx.array, rl: mx.array, ids, nll_ref_exact=None, tc_open=None, tc_close=None) -> dict:
    T = rt.shape[0]
    rl = rl.astype(mx.float32)
    rp = mx.softmax(rl, axis=-1)
    qp = mx.softmax(c["q_ref"], axis=-1)
    kl = mx.sum(rp * (mx.log(rp + 1e-12) - mx.log(qp + 1e-12)), axis=-1)
    order = c["top10"]
    nxt = mx.array(np.asarray(ids, np.int32)[1:])[:, None]                      # (T-1, 1)
    nll_q = c["nll"][:-1]
    hit = rt[:-1] == nxt
    cov = mx.any(hit, axis=-1)
    nll_ref = mx.where(cov, -mx.sum(mx.where(hit, rl[:-1], 0.0), axis=-1), mx.array(float("nan")))
    if nll_ref_exact is not None:
        ex = nll_ref_exact.astype(mx.float32)
        assert ex.shape == nll_ref.shape, (ex.shape, nll_ref.shape)
        # the exact file must agree with the stored top-128 wherever both exist (same reference forward)
        d = mx.where(cov, mx.abs(ex - nll_ref), 0.0)
        assert float(mx.max(d)) < 5e-3, f"reference NLL file disagrees with the stored reference log-probs (max diff {float(mx.max(d)):.4f})"
        nll_ref = ex
    out = {"kl": kl[:-1], "a1": (order[:, 0] == rt[:, 0])[:-1], "a5": mx.any(order[:, :5] == rt[:, :1], axis=-1)[:-1],
           "a10": mx.any(order == rt[:, :1], axis=-1)[:-1], "nll_q": nll_q, "nll_ref": nll_ref,
           "hit_q": order[:-1, 0] == nxt[:, 0], "hit_ref": rt[:-1, 0] == nxt[:, 0]}
    mx.eval(list(out.values()))
    out = {k: np.asarray(v) for k, v in out.items()}
    assert all(v.shape == (T - 1,) for v in out.values())
    out["in_call"] = call_span_mask(ids, tc_open, tc_close) if tc_open is not None else np.zeros(T - 1, bool)
    return out


def next_ids(ids) -> mx.array:
    a = np.asarray(ids, np.int32)
    return mx.array(np.concatenate([a[1:], a[-1:]]))          # the last position has no next token (dropped by the scorer)


def score(lp: mx.array, rt: mx.array, rl: mx.array, ids, nll_ref_exact=None) -> dict:
    """lp (T,V) logits or log-probs of the evaluated model; rt (T,128) reference top ids; rl (T,128) reference log-probs."""
    return score_compact(compact(lp, rt, next_ids(ids)), rt, rl, ids, nll_ref_exact)


def summarize(parts: list) -> dict | None:
    """parts: list of score() dicts. None if there is nothing to summarize (the caller must fail closed)."""
    if not parts:
        return None
    c = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    k = c["kl"]
    if k.size == 0 or not np.isfinite(k).all() or not np.isfinite(c["nll_q"]).all():
        return None
    cov = np.isfinite(c["nll_ref"])
    d = c["nll_q"][cov] - c["nll_ref"][cov]
    return {"sequences": len(parts), "positions": int(k.size),
            "median_kl": float(np.median(k)), "mean_kl": float(k.mean()), "p90": float(np.percentile(k, 90)),
            "p95": float(np.percentile(k, 95)), "p99": float(np.percentile(k, 99)), "kl_max": float(k.max()),
            "top1_pct": 100 * float(c["a1"].mean()), "top5_pct": 100 * float(c["a5"].mean()), "top10_pct": 100 * float(c["a10"].mean()),
            "text_ppl_q": float(np.exp(c["nll_q"][cov].mean())) if cov.any() else None,
            "text_ppl_ref": float(np.exp(c["nll_ref"][cov].mean())) if cov.any() else None,
            "text_ppl_ratio": float(np.exp(d.mean())) if cov.any() else None,
            "text_nll_delta_mean": float(d.mean()) if cov.any() else None,
            "text_nll_delta_sem": float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else None,
            "text_positions_compared_pct": 100 * float(cov.mean()),
            "text_ppl_unbiased": bool(cov.all()),
            "text_top1_acc_q_pct": 100 * float(c["hit_q"].mean()), "text_top1_acc_ref_pct": 100 * float(c["hit_ref"].mean()),
            **call_stats(c)}


def call_stats(c: dict) -> dict:
    m = c["in_call"] & np.isfinite(c["nll_ref"])
    if not c["in_call"].any():
        return {}
    if not m.all() == c["in_call"].all() or m.sum() != c["in_call"].sum():
        return {"call_positions": int(c["in_call"].sum()), "call_note": "reference NLL not exact on every call position"}
    d = c["nll_q"][m] - c["nll_ref"][m]
    return {"call_positions": int(m.sum()), "call_nll_q": float(c["nll_q"][m].mean()), "call_nll_ref": float(c["nll_ref"][m].mean()),
            "call_nll_delta": float(d.mean()), "call_nll_delta_sem": float(d.std(ddof=1) / np.sqrt(d.size)),
            "call_top1_acc_q_pct": 100 * float(c["hit_q"][m].mean()), "call_top1_acc_ref_pct": 100 * float(c["hit_ref"][m].mean()),
            "call_p_correct_q_geomean": float(np.exp(-c["nll_q"][m].mean())), "call_p_correct_ref_geomean": float(np.exp(-c["nll_ref"][m].mean()))}


def call_line(s: dict) -> str:
    if "call_nll_q" not in s:
        return ""
    return (f"tool-call tokens n={s['call_positions']}: NLL {s['call_nll_q']:.4f} vs ref {s['call_nll_ref']:.4f} (delta {s['call_nll_delta']:+.4f} +- "
            f"{s['call_nll_delta_sem']:.4f}), top-1 accuracy {s['call_top1_acc_q_pct']:.2f}% vs ref {s['call_top1_acc_ref_pct']:.2f}%")


def by_domain(parts: list, domains: list) -> dict:
    out = {}
    for d in sorted(set(domains)):
        s = summarize([p for p, dd in zip(parts, domains) if dd == d])
        if s is None:
            raise SystemExit(f"INVALID: domain {d} has no scorable positions")
        out[d] = s
    return out


def line(name: str, s: dict) -> str:
    return (f"{name:14s} n={s['positions']:6d}  KL median {s['median_kl']:.4f} mean {s['mean_kl']:.4f} p99 {s['p99']:.3f}  "
            f"top-1 agree {s['top1_pct']:.2f}% top-5 {s['top5_pct']:.2f}%  |  text PPL {s['text_ppl_q']:.3f} vs ref {s['text_ppl_ref']:.3f} "
            f"({'exact' if s['text_ppl_unbiased'] else 'BIASED top-128 fallback, ' + format(s['text_positions_compared_pct'], '.1f') + '% of positions'}; ratio {s['text_ppl_ratio']:.4f}, dNLL {s['text_nll_delta_mean']:+.4f} +- {s['text_nll_delta_sem'] or 0:.4f})  "
            f"text top-1 acc {s['text_top1_acc_q_pct']:.2f}% vs ref {s['text_top1_acc_ref_pct']:.2f}%")


def decisions_new() -> dict:
    return {"call": {"n": 0, "agree": 0, "flips": 0, "ref_is_call": 0, "q_is_call": 0, "p_ref": [], "p_q": []},
            "answer": {"n": 0, "agree": 0, "flips": 0, "ref_is_call": 0, "q_is_call": 0, "p_ref": [], "p_q": []}}


def decisions_update(dec: dict, marks, cp: dict, rt, rl, tc: int) -> None:
    """marks: [[position, "call"|"answer"], ...] (what the CORPUS text does next); the comparison is reference vs model:
    agree = same argmax token; flip = one of them starts a tool call and the other does not."""
    if not marks:
        return
    rtn, rln = np.asarray(rt), np.asarray(rl.astype(mx.float32)); t10 = np.asarray(cp["top10"]); ltc = np.asarray(cp[f"lp_{tc}"])
    for p, kind in marks:
        ra, qa = int(rtn[p, 0]), int(t10[p, 0])
        d = dec[kind]; d["n"] += 1; d["agree"] += int(ra == qa); d["flips"] += int((ra == tc) != (qa == tc))
        d["ref_is_call"] += int(ra == tc); d["q_is_call"] += int(qa == tc)
        pos = np.nonzero(rtn[p] == tc)[0]
        d["p_ref"].append(float(np.exp(rln[p][pos[0]])) if len(pos) else 0.0); d["p_q"].append(float(np.exp(ltc[p])))


def decisions_summary(dec: dict) -> dict:
    out = {}
    for k, v in dec.items():
        if not v["n"]:
            continue
        out[k] = {"n": v["n"], "argmax_agree_pct": 100 * v["agree"] / v["n"], "call_answer_flips": v["flips"],
                  "reference_starts_tool_call": v["ref_is_call"], "model_starts_tool_call": v["q_is_call"],
                  "median_p_toolcall_ref": float(np.median(v["p_ref"])), "median_p_toolcall_q": float(np.median(v["p_q"])),
                  "min_p_toolcall_q": float(np.min(v["p_q"])), "max_p_toolcall_q": float(np.max(v["p_q"]))}
    return out

