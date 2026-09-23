"""KL evaluation for glm5_next bundles against the pod FP8 reference.

Reference: klref.safetensors from the RunPod capture — per prompt i:
  p{i}.input_ids [S], p{i}.top_ids [S,128], p{i}.top_logprobs [S,128]
(teacher-forced, positions predict token t+1, reference_precision=FP8).

Metrics (qwen38 protocol): median KL (primary), mean KL, top-1 agreement,
ref-top1-in-top5/top10 containment, margin-conditioned flip curve.

  python -m jang_tools.glm5_next.kl_eval --bundle <dir> --klref klref.safetensors \
      [--bf16 <dir>]   (--bf16 loads unquantized via load_model instead)
"""

from __future__ import annotations

import argparse
import json
import re
import time

import mlx.core as mx
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--bf16", default=None)
    ap.add_argument("--foreign", default=None,
                    help="third-party mlx bundle dir (HF naming)")
    ap.add_argument("--klref", required=True)
    ap.add_argument("--metrics-out", default=None)
    ap.add_argument("--limit", type=int, default=0, help="max prompts (0=all)")
    args = ap.parse_args()

    ref = mx.load(args.klref)
    prompts = sorted({int(m.group(1)) for k in ref
                      if (m := re.match(r"p(\d+)\.input_ids", k))})
    if args.limit:
        prompts = prompts[: args.limit]

    if args.foreign:
        from .load_foreign import load_foreign
        model = load_foreign(args.foreign)
    elif args.bundle:
        from .load import load_bundle
        model = load_bundle(args.bundle)
    else:
        from .load import load_model
        model = load_model(args.bf16)

    kls, agree, margins, flips = [], [], [], []
    in5 = in10 = 0
    t0 = time.time()
    total_pos = 0
    for pi in prompts:
        ids = np.asarray(ref[f"p{pi}.input_ids"]).astype(np.int64).reshape(-1)
        top_ids = np.asarray(ref[f"p{pi}.top_ids"])          # [S,128]
        top_lp = np.asarray(ref[f"p{pi}.top_logprobs"]).astype(np.float64)
        logits = model(mx.array(ids[None]))
        logp = np.asarray(
            (logits[0].astype(mx.float32)
             - mx.logsumexp(logits[0].astype(mx.float32), axis=-1, keepdims=True)),
            dtype=np.float64)
        S = min(len(ids) - 1, top_ids.shape[0])
        for t in range(S):
            r_ids = top_ids[t]
            r_lp = top_lp[t]
            r_p = np.exp(r_lp)
            mass = r_p.sum()
            if mass <= 0:
                continue
            r_p = r_p / mass                                  # renormalized top-128
            q_lp = logp[t, r_ids]
            q_p = np.exp(q_lp)
            q_p = q_p / max(q_p.sum(), 1e-12)
            kl = float(np.sum(r_p * (np.log(r_p + 1e-12) - np.log(q_p + 1e-12))))
            kls.append(kl)
            ref_top1 = r_ids[np.argmax(r_lp)]
            our_top1 = int(np.argmax(logp[t]))
            agree.append(ref_top1 == our_top1)
            srt = np.argsort(-logp[t, :])
            in5 += int(ref_top1 in srt[:5])
            in10 += int(ref_top1 in srt[:10])
            two = np.sort(r_lp)[-2:]
            margins.append(float(two[1] - two[0]))
            flips.append(ref_top1 != our_top1)
            total_pos += 1
        mx.clear_cache()
        if pi % 10 == 0:
            print(f"prompt {pi}/{len(prompts)}  positions={total_pos}  "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)

    kls = np.array(kls)
    agree = np.array(agree)
    margins = np.array(margins)
    flips = np.array(flips, dtype=bool)
    pct = {f"p{p}": float(np.percentile(kls, p)) for p in (50, 75, 90, 95, 99)}
    print(f"positions: {len(kls)}")
    print(f"median KL: {np.median(kls):.6f}   mean: {kls.mean():.6f}")
    print("KL percentiles: " + "  ".join(f"{k}={v:.4f}" for k, v in pct.items())
          + f"  max={kls.max():.4f}")
    print(f"top-1 agreement: {100*agree.mean():.2f}%")
    print(f"ref top-1 in our top-5: {100*in5/len(kls):.2f}%  top-10: {100*in10/len(kls):.2f}%")
    qs = np.quantile(margins, np.linspace(0, 1, 9))
    curve = []
    print("margin-conditioned flip curve (should DECREASE):")
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (margins >= lo) & (margins < hi if hi < qs[-1] else margins <= hi)
        if m.sum() == 0:
            continue
        fr = float(flips[m].mean())
        curve.append([float(lo), float(hi), fr, int(m.sum())])
        print(f"  margin [{lo:7.3f},{hi:7.3f}): flip {100*fr:5.2f}%  (n={m.sum()})")

    if args.metrics_out:
        json.dump({
            "positions": int(len(kls)),
            "median_kl": float(np.median(kls)),
            "mean_kl": float(kls.mean()),
            "kl_percentiles": pct,
            "kl_max": float(kls.max()),
            "top1_agreement_pct": float(100 * agree.mean()),
            "ref_top1_in_top5_pct": float(100 * in5 / len(kls)),
            "ref_top1_in_top10_pct": float(100 * in10 / len(kls)),
            "flip_curve": curve,
            "reference_precision": "FP8",
        }, open(args.metrics_out, "w"), indent=1)
        print("metrics →", args.metrics_out)


if __name__ == "__main__":
    main()
