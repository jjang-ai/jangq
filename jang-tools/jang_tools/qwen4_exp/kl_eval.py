"""KL evaluation: JANG bundle vs bf16 source (both streamed, run SEQUENTIALLY
— never two MLX models concurrently).

Protocol (per standing rules):
  - held-out prompts, ZERO calibration contamination
  - three probe lengths incl. one long (GDN-state stress) and one that does
    not divide the QSA compress ratio / GDN conv kernel
  - margin-conditioned flip curve: bin top-1 flips by the SOURCE's top1−top2
    margin — monotone decreasing = healthy lossiness, flat/rising = structural
  - hot-n-gram prompts included so the 51B table is actually exercised

Two-pass design: pass 1 runs the REFERENCE model over all prompts and saves
top-K logprobs + margins to disk; pass 2 runs the bundle and compares. This
keeps peak memory at one model and makes A/B across bundles cheap (reference
pass runs once).

  python -m jang_tools.qwen4_exp.kl_eval ref   --model <bf16dir> --prompts p.jsonl --out ref.npz
  python -m jang_tools.qwen4_exp.kl_eval eval  --bundle <dir>    --prompts p.jsonl --ref ref.npz
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

TOPK = 128


def run_model(model, tok, prompts, max_len):
    """Returns per-prompt (logprobs_topk, topk_ids, argmax, margin) at each
    position (teacher-forced over the prompt tokens themselves)."""
    out = []
    for p in prompts:
        ids = tok.encode(p, add_special_tokens=False)[:max_len]
        if len(ids) < 8:
            out.append(None)
            continue
        logits = model(mx.array([ids]), eval_layers=True)
        logits = logits[0].astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        top_idx = mx.argpartition(-logprobs, kth=TOPK - 1, axis=-1)[:, :TOPK]
        top_lp = mx.take_along_axis(logprobs, top_idx, axis=-1)
        mx.eval(top_idx, top_lp)
        lp = np.asarray(top_lp)
        ti = np.asarray(top_idx)
        order = np.argsort(-lp, axis=-1)
        lp = np.take_along_axis(lp, order, axis=-1)
        ti = np.take_along_axis(ti, order, axis=-1)
        margin = lp[:, 0] - lp[:, 1]
        out.append({"ids": np.array(ids), "top_lp": lp.astype(np.float16),
                    "top_ids": ti.astype(np.int32), "margin": margin.astype(np.float32)})
        mx.clear_cache()
    return out


def kl_from_topk(ref, got):
    """KL(ref || got) on the ref top-K support + agreement/containment metrics."""
    kls, flips, margins = [], [], []
    in5, in10 = [], []
    for r, g in zip(ref, got):
        if r is None or g is None:
            continue
        n = min(r["top_lp"].shape[0], g["top_lp"].shape[0])
        for t in range(n):
            r_ids, r_lp = r["top_ids"][t], r["top_lp"][t].astype(np.float64)
            g_ids, g_lp = g["top_ids"][t], g["top_lp"][t].astype(np.float64)
            gmap = {int(i): float(l) for i, l in zip(g_ids, g_lp)}
            g_on_r = np.array([gmap.get(int(i), -20.0) for i in r_ids])
            p = np.exp(r_lp)
            p = p / p.sum()
            q = np.exp(g_on_r)
            q = q / max(q.sum(), 1e-9)
            kls.append(float((p * (np.log(p + 1e-12) - np.log(q + 1e-12))).sum()))
            flips.append(int(r_ids[0]) != int(g_ids[0]))
            in5.append(int(r_ids[0]) in {int(i) for i in g_ids[:5]})
            in10.append(int(r_ids[0]) in {int(i) for i in g_ids[:10]})
            margins.append(float(r["margin"][t]))
    return (np.array(kls), np.array(flips, dtype=bool), np.array(margins),
            np.array(in5, dtype=bool), np.array(in10, dtype=bool))


def flip_curve(flips, margins, bins=8):
    qs = np.quantile(margins, np.linspace(0, 1, bins + 1))
    rows = []
    for i in range(bins):
        m = (margins >= qs[i]) & (margins <= qs[i + 1] if i == bins - 1 else margins < qs[i + 1])
        if m.sum():
            rows.append((float(qs[i]), float(qs[i + 1]), float(flips[m].mean()), int(m.sum())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ref", "eval", "floor"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-len", type=int, default=333)
    ap.add_argument("--metrics-out", default=None)  # non-aligned on purpose
    args = ap.parse_args()

    from transformers import AutoTokenizer

    prompts = [json.loads(l)["text"] if l.strip().startswith("{") else l.rstrip("\n")
               for l in open(args.prompts) if l.strip()]

    if args.mode == "ref":
        from .load import load_model

        tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
        model = load_model(args.model, lazy=True)
        res = run_model(model, tok, prompts, args.max_len)
        np.savez_compressed(args.out, **{
            f"p{i}_{k}": v for i, r in enumerate(res) if r
            for k, v in r.items()
        }, n_prompts=len(res))
        print(f"reference saved → {args.out}")
        return

    if args.mode == "floor":
        # THE gate that makes every other number falsifiable (Ornith lesson):
        # the reference model scored against its own saved reference must be
        # EXACTLY zero KL / 100% top-1, else the harness resolves nothing.
        from .load import load_model

        tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
        model = load_model(args.model, lazy=True)
        got = run_model(model, tok, prompts, args.max_len)
    else:
        from .load import load_bundle

        tok = AutoTokenizer.from_pretrained(args.tokenizer or args.bundle)
        model = load_bundle(args.bundle)
        got = run_model(model, tok, prompts, args.max_len)

    raw = np.load(args.ref)
    n = int(raw["n_prompts"])
    ref = []
    for i in range(n):
        if f"p{i}_top_lp" in raw:
            ref.append({k: raw[f"p{i}_{k}"] for k in ("ids", "top_lp", "top_ids", "margin")})
        else:
            ref.append(None)

    kls, flips, margins, in5, in10 = kl_from_topk(ref, got)
    if len(kls) < 5000:
        print(f"REFUSING to report: only {len(kls)} positions (<5000) — "
              f"under-sampled sets produced indefensible numbers before "
              f"(Ornith 9B lesson). Add prompts or raise --max-len.")
        raise SystemExit(2)
    print(f"positions: {len(kls)}")
    print(f"median KL: {np.median(kls):.6f}   mean: {kls.mean():.6f} "
          f"(median is the tier separator; mean is tail-dominated)")
    print(f"top-1 agreement: {100 * (1 - flips.mean()):.2f}% "
          f"(do NOT rank close tiers by top-1 — binomial noise)")
    if args.mode == "floor":
        ok = kls.max() == 0.0 and not flips.any()
        print(f"FLOOR CHECK: {'PASS (exactly zero)' if ok else 'FAIL — harness resolves nothing, DO NOT report bundle KLs'}")
        raise SystemExit(0 if ok else 1)
    print(f"ref-top1 within bundle top-5: {100 * in5.mean():.2f}%   "
          f"top-10: {100 * in10.mean():.2f}%")
    print("margin-conditioned flip curve (low margin → high should DECREASE):")
    curve = flip_curve(flips, margins)
    for lo, hi, fr, cnt in curve:
        print(f"  margin [{lo:6.3f},{hi:6.3f}): flip {100 * fr:5.2f}%  (n={cnt})")
    if args.metrics_out:
        json.dump({
            "positions": int(len(kls)),
            "median_kl": float(np.median(kls)), "mean_kl": float(kls.mean()),
            "top1_agreement_pct": float(100 * (1 - flips.mean())),
            "ref_top1_in_top5_pct": float(100 * in5.mean()),
            "ref_top1_in_top10_pct": float(100 * in10.mean()),
            "flip_curve": [[lo, hi, fr, cnt] for lo, hi, fr, cnt in curve],
        }, open(args.metrics_out, "w"), indent=1)
        print(f"metrics → {args.metrics_out}")


if __name__ == "__main__":
    main()
