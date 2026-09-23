"""Margin-conditioned flip diagnosis: is a weak top-1 a DEFECT or the bit budget?

`kl_eval_dots3` reports mean KL and top-1 agreement. Neither distinguishes
"lossy but healthy" from "structurally broken" — a bundle can post a mediocre
top-1 for either reason, and the acceptance gate cannot tell them apart.

This does. A healthy lossy quant flips tokens the source was **unsure** about,
so the flip rate must fall **monotonically** with the source's top1-top2 margin.

    monotone decreasing   -> lossy but healthy; buy quality with bits, not fixes
    flat across margins   -> noise injected somewhere; re-diagnose
    rising at high margin -> structural damage; DO NOT SHIP

🚨 Do NOT approximate this by comparing the flip rate against the fraction of
positions with `margin < mean_KL`. Per-position KL has a long tail, so mean KL
is not a per-position flip threshold. That shortcut reports "structural" on a
perfectly healthy bundle — it did exactly that on dots3-note (2026-08-21).

    python -m jang_tools.dots3.diag_flips <bundle> <capture_dir> [--npz out.npz]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .kl_eval_dots3 import full_logprobs
from .model import BundleModel

MARGIN_EDGES = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, np.inf]
CONFIDENT_NATS = 2.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", type=Path)
    ap.add_argument("capture", type=Path)
    ap.add_argument("--npz", type=Path, default=None,
                    help="also save raw per-position arrays")
    a = ap.parse_args()

    z = np.load(a.capture / "source_logprobs.npz")
    n_eval = len([k for k in z.files if k.startswith("ids_")])
    bm = BundleModel(a.bundle)

    margin, flip, klp, seq, pos = [], [], [], [], []
    for k in range(n_eval):
        ids = z[f"ids_{k}"].tolist()
        src_i, src_lp = z[f"top_i_{k}"], z[f"top_lp_{k}"]
        q_lp = np.asarray(full_logprobs(bm, ids))
        S = min(len(ids), src_i.shape[0])
        p = np.exp(src_lp[:S].astype(np.float64))
        q_at = np.take_along_axis(q_lp[:S].astype(np.float64), src_i[:S], 1)
        klp.append((p * (src_lp[:S] - q_at)).sum(-1))
        margin.append(src_lp[:S, 0] - src_lp[:S, 1])
        flip.append(q_lp[:S].argmax(-1) != src_i[:S, 0])
        seq.append(np.full(S, k))
        pos.append(np.arange(S))
        print(f"  seq {k:2d}/{n_eval}  S={S:4d}  flip={flip[-1].mean():.1%}",
              flush=True)

    m, f = np.concatenate(margin), np.concatenate(flip)
    kl, sq, po = (np.concatenate(x) for x in (klp, seq, pos))
    if a.npz:
        np.savez(a.npz, margin=m, flip=f, kl=kl, seq=sq, pos=po)

    print(f"\nmean KL {kl.mean():.4f} nats   top-1 {1 - f.mean():.4%}   "
          f"positions {len(f):,}")

    print(f"\n{'source margin':>16}{'positions':>11}{'flip rate':>11}"
          f"{'% of flips':>12}")
    rates = []
    for lo, hi in zip(MARGIN_EDGES[:-1], MARGIN_EDGES[1:]):
        s = (m >= lo) & (m < hi)
        if not s.any():
            continue
        rates.append(f[s].mean())
        print(f"{f'{lo:g}-{hi:g}':>16}{s.sum():>11,}{100 * f[s].mean():>10.1f}%"
              f"{100 * f[s].sum() / max(f.sum(), 1):>11.1f}%")

    conf = m >= CONFIDENT_NATS
    print(f"\nconfident (>= {CONFIDENT_NATS:g} nats): {100 * conf.mean():.1f}% of "
          f"positions, flip rate {100 * f[conf].mean():.2f}%")

    mono = all(x >= y - 1e-12 for x, y in zip(rates, rates[1:]))
    print(f"\nmonotone decreasing across margin bins: {mono}")
    print("  -> LOSSY BUT HEALTHY; buy quality with bits, not bug-hunting"
          if mono else
          "  -> NOT monotone: flips are not margin-explained. RE-DIAGNOSE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
