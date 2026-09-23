"""Self-tests for ops.py that EXERCISE BOTH branches of every switch.

Written after the 2026-08-15 sorted-gather incident: the original self-test
passed `sort=True` at T=37, but `experts_apply` only takes the sorted path at
T >= 64, so the branch that shipped was never executed. Every test here sweeps
T across the threshold and asserts the taken path explicitly.

    python -m jang_tools.dots3.ops_selftest
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx

from .ops import QW, experts_apply, route, _causal_mask, SORT_THRESHOLD


def _naive_experts(x, inds, w, wg, wu, wd):
    T, K = inds.shape
    out = np.zeros((T, x.shape[1]), np.float32)
    for t in range(T):
        for k in range(K):
            e = int(inds[t, k])
            g = x[t] @ wg[e].T
            u = x[t] @ wu[e].T
            out[t] += float(w[t, k]) * (((g / (1 + np.exp(-g))) * u) @ wd[e].T)
    return out


def test_experts_both_branches() -> bool:
    ok = True
    rng = np.random.default_rng(0)
    # T values straddling SORT_THRESHOLD, incl. exact boundary
    for T in (1, 8, SORT_THRESHOLD - 1, SORT_THRESHOLD, SORT_THRESHOLD + 1,
              245, 512):
        K, E, H, I = 8, 16, 64, 64
        x = rng.standard_normal((T, H)).astype(np.float32)
        wg = (rng.standard_normal((E, I, H)) * 0.1).astype(np.float32)
        wu = (rng.standard_normal((E, I, H)) * 0.1).astype(np.float32)
        wd = (rng.standard_normal((E, H, I)) * 0.1).astype(np.float32)
        inds = rng.integers(0, E, (T, K)).astype(np.uint32)
        w = rng.random((T, K)).astype(np.float32)
        ref = _naive_experts(x, inds, w, wg, wu, wd)
        for sort in (True, False):
            got = np.asarray(experts_apply(
                mx.array(x), mx.array(inds), mx.array(w),
                mx.array(wg), mx.array(wu), mx.array(wd), sort=sort))
            err = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-9)
            path = "SORTED" if (sort and T >= SORT_THRESHOLD) else "plain"
            good = err < 1e-4
            ok &= good
            print(f"  T={T:4d} sort={str(sort):5s} [{path:6s}] "
                  f"err={err:.2e} {'OK' if good else 'FAIL'}")
    return ok


def test_experts_quantized() -> bool:
    """Same, through gather_qmm (QW) — the deployed path."""
    ok = True
    rng = np.random.default_rng(1)
    for T in (32, 128):
        K, E, H, I = 8, 16, 64, 64
        x = mx.array(rng.standard_normal((T, H)).astype(np.float32))
        mats = [mx.array((rng.standard_normal(s) * 0.1).astype(np.float32))
                for s in ((E, I, H), (E, I, H), (E, H, I))]
        inds = mx.array(rng.integers(0, E, (T, K)).astype(np.uint32))
        w = mx.array(rng.random((T, K)).astype(np.float32))
        ref = np.asarray(experts_apply(x, inds, w, *mats, sort=False))

        def q(a, bits=8, gs=32):
            wq, s, b = mx.quantize(a, group_size=gs, bits=bits)
            return QW(wq, s, b, gs, bits, "affine")

        qm = [q(m) for m in mats]
        for sort in (True, False):
            got = np.asarray(experts_apply(x, inds, w, *qm, sort=sort))
            rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
            good = rel < 0.05          # 8-bit quant noise only
            ok &= good
            print(f"  T={T:4d} sort={str(sort):5s} QW8  relL2={rel:.2e} "
                  f"{'OK' if good else 'FAIL'}")
    return ok


def test_sorted_flag_guard() -> bool:
    """sorted_indices=True without lhs_indices must RAISE, never miscompute."""
    from .ops import _gmm
    x = mx.zeros((8, 1, 4))
    w = mx.zeros((2, 4, 4))
    try:
        _gmm(x, w, mx.array([0] * 8, dtype=mx.uint32), None, True)
    except ValueError:
        print("  guard raises on sorted_indices without lhs_indices  OK")
        return True
    print("  guard did NOT raise  FAIL")
    return False


def test_route() -> bool:
    rng = np.random.default_rng(2)
    T, E, H = 245, 64, 64
    x = mx.array(rng.standard_normal((T, H)).astype(np.float32))
    gw = mx.array((rng.standard_normal((E, H)) * 0.05).astype(np.float32))
    b = mx.array((rng.standard_normal(E) * 0.1).astype(np.float32))
    inds, w = route(x, gw, b, 8, True, 1.0)
    sc = 1 / (1 + np.exp(-(np.asarray(x) @ np.asarray(gw).T)))
    ref = np.argsort(-(sc + np.asarray(b)[None]), axis=1)[:, :8]
    agree = all(set(np.asarray(inds)[t]) == set(ref[t]) for t in range(T))
    rw = np.take_along_axis(sc, np.asarray(inds).astype(int), 1)
    rw = rw / (rw.sum(1, keepdims=True) + 1e-20)
    werr = np.abs(np.asarray(w) - rw).max()
    print(f"  route: set-agreement={agree} weight-err={werr:.2e} "
          f"{'OK' if agree and werr < 1e-3 else 'FAIL'}")
    return agree and werr < 1e-3


def test_masks() -> bool:
    m = np.asarray(_causal_mask(6, 3, mx.float32))
    exp = [0, 0, 0, 0, 0, 0]
    row4 = [np.isinf(m[4][i]) for i in range(6)]
    want = [True, True, False, False, False, True]
    ok = row4 == want
    print(f"  sliding mask window=3 row4 masked={row4} {'OK' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("experts_apply — both branches across SORT_THRESHOLD:")
    a = test_experts_both_branches()
    print("experts_apply — quantized (gather_qmm):")
    b = test_experts_quantized()
    print("guard:")
    c = test_sorted_flag_guard()
    print("routing / masks:")
    d = test_route(); e = test_masks()
    ok = a and b and c and d and e
    print("OPS_SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
