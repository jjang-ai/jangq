"""Select kept experts per MoE layer from REAP saliency -> keep_map.json.

Keeps the top-K highest-saliency experts in EACH MoE layer (uniform K across
layers, since the converter rewrites num_local_experts to a single value).
Drops the lowest-saliency (incl. never-selected) experts.

  python -m jang_tools.minimax_m3.reap_select \
      --saliency /Users/eric/m3_reap_saliency.npz \
      --out /Users/eric/m3_keep_map.json --prune-pct 0.22

Created by Jinho Jang (eric@jangq.ai).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saliency", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prune-pct", type=float, default=0.22)
    args = ap.parse_args()

    d = np.load(args.saliency)
    sal = d["saliency"]                 # (L, E)
    cnt = d["count"]                    # (L, E)
    layer_ids = d["layer_ids"]          # (L,)
    L, E = sal.shape
    keep_k = round((1.0 - args.prune_pct) * E)
    print(f"  layers={L} experts={E}  prune {args.prune_pct*100:.0f}% -> keep {keep_k}/{E} per layer")

    # normalize saliency per layer to its own max so the ranking is per-layer.
    keep_map = {}
    dead_total = 0
    min_sal_kept = 1e30
    for i, li in enumerate(layer_ids.tolist()):
        s = sal[i].copy()
        dead_total += int((cnt[i] == 0).sum())
        order = np.argsort(-s)                  # high saliency first
        kept = sorted(order[:keep_k].tolist())
        keep_map[str(int(li))] = kept
        min_sal_kept = min(min_sal_kept, float(s[order[keep_k - 1]]))

    Path(args.out).write_text(json.dumps(keep_map, indent=1))
    # report drop concentration
    avg_dead = dead_total / L
    print(f"  wrote {args.out}: {len(keep_map)} layers x {keep_k} kept")
    print(f"  avg never-selected experts/layer: {avg_dead:.1f}  (these dropped first)")
    print(f"  weakest kept-expert saliency (min over layers): {min_sal_kept:.3g}")


if __name__ == "__main__":
    main()
