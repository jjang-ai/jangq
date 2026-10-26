"""Quant-quality metrics for a DSV4.1 JANG bundle: teacher-forced over held-out rows, per next-token
position capture NLL, top-1 agreement (argmax == true token), top-1 probability, and entropy.
Report mean/median/p95/p99 percentiles. Self-contained (no full-precision reference needed; true
KL-vs-FP requires the 286GB source on a larger host — flagged in handoff)."""
import argparse, json, sys, time
from pathlib import Path
import mlx.core as mx
sys.path.insert(0, str(Path.home() / "jang/jang-tools"))
from jang_tools.dsv41.model import DSV41
from jang_tools.dsv41.generate import load_encoding
from jang_tools.dsv41.calib_generate import DEFAULT_DOMAINS
import copy


def render_full(enc, row):
    msgs = copy.deepcopy(row["messages"])
    for m in msgs:
        if m["role"] == "assistant":
            m.setdefault("reasoning_content", "")
    return enc.encode_messages(msgs, thinking_mode="thinking", reasoning_effort=row["effort"])


def pct(a, p):
    return float(mx.sort(a)[int(max(0, min(len(a) - 1, round(p / 100 * (len(a) - 1)))))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True); ap.add_argument("--engram-dir")
    ap.add_argument("--rows", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=2560); ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    enc = load_encoding(Path(a.bundle).expanduser())
    m = DSV41(Path(a.bundle).expanduser(), a.engram_dir)
    rows = [json.loads(l) for l in open(a.rows)]
    rows = [r for r in rows if r["domain"] not in DEFAULT_DOMAINS]
    import random; random.Random(a.seed).shuffle(rows); rows = rows[: a.n]

    all_nll, all_top1, all_p1, all_ent = [], [], [], []
    t0 = time.time(); tot = 0
    for i, r in enumerate(rows):
        ids = m.tokenizer.encode(render_full(enc, r), add_special_tokens=False).ids[: a.max_tokens]
        if len(ids) < 2:
            continue
        sess = m.new_session()
        logits = m.forward(ids, sess, all_logits=True)
        lp = logits[:-1].astype(mx.float32)
        tgt = mx.array(ids[1:], mx.int32)
        logz = mx.logsumexp(lp, axis=-1)
        logp = lp - logz[:, None]                            # log-softmax [T-1, V]
        tok_lp = mx.take_along_axis(logp, tgt[:, None], axis=-1)[:, 0]
        nll = -tok_lp                                        # per-token NLL
        am = mx.argmax(lp, axis=-1)                          # quant's top-1 token
        top1 = (am == tgt).astype(mx.float32)                # agreement with teacher-forced true token
        p = mx.exp(logp)
        p1 = mx.max(p, axis=-1)                              # top-1 probability
        ent = -mx.sum(p * logp, axis=-1)                     # predictive entropy (nats)
        mx.eval(nll, top1, p1, ent)
        all_nll += nll.tolist(); all_top1 += top1.tolist(); all_p1 += p1.tolist(); all_ent += ent.tolist()
        tot += len(ids) - 1
        del logits, lp, logp, p, sess; mx.clear_cache()
        if (i + 1) % 6 == 0:
            print(f"  {i+1}/{len(rows)} rows {tot} tok top1={sum(all_top1)/len(all_top1):.3f} ({time.time()-t0:.0f}s)", flush=True)

    nll = mx.array(all_nll); p1 = mx.array(all_p1); ent = mx.array(all_ent)
    res = {
        "bundle": Path(a.bundle).name, "tokens": tot, "rows": len(rows),
        "top1_accuracy": sum(all_top1) / len(all_top1),
        "nll": {"mean": float(mx.mean(nll)), "median": pct(nll, 50), "p95": pct(nll, 95), "p99": pct(nll, 99)},
        "ppl_from_mean_nll": float(mx.exp(mx.mean(nll))),
        "top1_prob": {"mean": float(mx.mean(p1)), "median": pct(p1, 50), "p5": pct(p1, 5), "p1": pct(p1, 1)},
        "entropy_nats": {"mean": float(mx.mean(ent)), "median": pct(ent, 50), "p95": pct(ent, 95), "p99": pct(ent, 99)},
        "note": "Self-referential quant metrics (teacher-forced). top1_accuracy = argmax==true-token. "
                "True KL vs full-precision reference requires the 286GB source on a >256GB host.",
    }
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
