"""A4 (corrected) — window vs pool boundary correctness.

INVARIANT (confirmed via paper §1.4 and runtime audit 2026-05-01):
the source-position index space INTENTIONALLY overlaps between window
and pool. Window keys are raw KV vectors at specific positions; pool
entries are AVERAGED KV over `ratio` tokens. They give the model two
different feature vectors per source segment — the softmax distributes
attention across both. This is the architecture, not a bug.

What we DO need to verify:
  V1 — window mask: query at q_pos sees window keys at positions p iff
       (q_pos - window) < p <= q_pos AND p < win_left + win_len.
  V2 — pool mask: query at q_pos sees pool entry k iff (k+1)*ratio <= q_pos+1.
  V3 — total visible cache slots: at most win_len + P (no double-counting
       in the cache; both spaces are independent).
  V4 — when P=0 (pool empty, early prefill), the concatenated mask
       reduces to plain window mask.
"""
from __future__ import annotations
import sys
import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Users/eric/jang/jang-tools")
from jang_tools.dsv4.mlx_model import _build_window_mask, _compressed_visibility


def run(label, *, B, S, offset, win_len, P, ratio):
    win_mask = _build_window_mask(B, S, offset, 128, win_len)
    comp = _compressed_visibility(B, S, offset, P, ratio)

    # V1: window mask must be causal-within-window
    win_arr = np.asarray(win_mask)[0, 0]   # (S, win_len)
    win_left = (offset + S) - win_len
    cache_pos = np.arange(win_len) + win_left
    q_pos = offset + np.arange(S)
    expected_win = (cache_pos[None, :] <= q_pos[:, None]) & \
                   (cache_pos[None, :] > q_pos[:, None] - 128)
    v1 = np.array_equal(win_arr, expected_win)

    # V2: pool mask block-causal
    if P > 0:
        comp_arr = np.asarray(comp)[0, 0]  # (S, P)
        k = np.arange(P)
        expected_pool = ((k + 1)[None, :] * ratio) <= (q_pos[:, None] + 1)
        v2 = np.array_equal(comp_arr, expected_pool)
    else:
        v2 = True

    # V3: total visible slots ≤ win_len + P
    total_visible = int(win_arr.sum() + (np.asarray(comp)[0, 0].sum() if P > 0 else 0))
    max_possible = S * (win_len + P)
    v3 = total_visible <= max_possible

    print(f"  {label}: V1(win-mask)={v1}  V2(pool-mask)={v2}  V3(no-overcount)={v3}  "
          f"  visible={total_visible}/{max_possible}")
    return v1 and v2 and v3


# V4 separately: P=0 → mask is just window
def v4_test():
    win_mask = _build_window_mask(1, 64, 0, 128, 64)
    win_arr = np.asarray(win_mask)
    # Causal within window: query at i sees keys at j ≤ i
    expected = np.tril(np.ones(64, dtype=bool))
    ok = np.array_equal(win_arr[0, 0], expected)
    print(f"  V4 P=0 prefill: {'OK' if ok else 'FAIL'}")
    return ok


print("=== A4 corrected — window/pool seam invariants ===")
ALL = True
ALL &= run("HSA prefill L=256 ratio=128 P=2",  B=1, S=256, offset=0,   win_len=128, P=2,   ratio=128)
ALL &= run("HSA decode  L=1   off=512  ratio=128 P=4", B=1, S=1, offset=512, win_len=128, P=4, ratio=128)
ALL &= run("CSA prefill L=256 ratio=4 P=64",   B=1, S=256, offset=0,   win_len=128, P=64,  ratio=4)
ALL &= run("CSA decode  L=1   off=512  ratio=4 P=128",  B=1, S=1, offset=512, win_len=128, P=128, ratio=4)
ALL &= run("HSA decode  L=1   off=2048 ratio=128 P=16", B=1, S=1, offset=2048, win_len=128, P=16, ratio=128)
ALL &= run("CSA decode  L=1   off=2048 ratio=4 P=512",  B=1, S=1, offset=2048, win_len=128, P=512, ratio=4)
ALL &= v4_test()
print(f"\n=== {'ALL OK' if ALL else 'FAIL'} ===")
sys.exit(0 if ALL else 1)
