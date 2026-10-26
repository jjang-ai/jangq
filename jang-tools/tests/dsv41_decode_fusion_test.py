"""DSV4.1 decode-path speedups == the stock path (small synthetic model, small GPU kernels; no bundle).

Each technique is A/B-gated by a module flag (env var at import); every test runs the same forward
with the flag off and on and compares logits / outputs.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import jang_tools.dsv41.model as M
from jang_tools.dsv41.tests import synth as S

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal kernels")

PROMPT = list(range(5, 45))


@pytest.fixture(scope="module")
def tiny():
    return S.build(S.tiny_args())


def _bits(a):
    a = mx.array(a)
    if a.dtype in (mx.bfloat16, mx.float16):
        a = a.view(mx.uint16)
    return np.array(a)


def _run_greedy(model, steps=12):
    out, lg = S.greedy(model, PROMPT, steps, return_logits=True)
    return out, np.stack(lg)


def _run_verify(model, rows=5):
    """Prefill, then one multi-row verify-shaped forward (all logits)."""
    caches = S.new_caches(model.args)
    mx.eval(S.forward(model, PROMPT, caches, 0))
    lg = S.forward(model, [7, 300, 11, 900, 3, 77, 512, 64][:rows], caches, len(PROMPT), all_logits=True)
    return np.array(lg.astype(mx.float32))


def _ab(monkeypatch, model, attr, run):
    monkeypatch.setattr(M, attr, False)
    a = run(model)
    monkeypatch.setattr(M, attr, True)
    b = run(model)
    return a, b


def test_layer_async_is_bit_exact(tiny, monkeypatch):
    (ta, la), (tb, lb) = _ab(monkeypatch, tiny, "LAYER_ASYNC", _run_greedy)
    assert ta == tb and len(set(ta)) > 3
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    va, vb = _ab(monkeypatch, tiny, "LAYER_ASYNC", _run_verify)
    assert np.array_equal(va.view(np.uint32), vb.view(np.uint32))


@pytest.mark.parametrize("E,K", [(384, 5120), (128, 1024), (512, 5120)])
def test_router_gemv_bit_exact_at_one_row(E, K):
    from jang_tools.dsv41.metal_decode import router_logits
    w = (mx.random.normal((E, K), key=mx.random.key(1)) / 70).astype(mx.bfloat16)
    for t in range(24):
        x = (mx.random.normal((1, K), key=mx.random.key(t)) * (1 + t % 5)).astype(mx.bfloat16)
        assert np.array_equal(_bits(router_logits(x, w)), _bits(x.astype(mx.float32) @ w.astype(mx.float32).T))


def test_router_gemv_rows_equal_their_one_row_decode():
    from jang_tools.dsv41.metal_decode import router_logits
    w = (mx.random.normal((384, 5120), key=mx.random.key(1)) / 70).astype(mx.bfloat16)
    for R in (1, 2, 5, 8):
        x = mx.random.normal((R, 5120), key=mx.random.key(R)).astype(mx.bfloat16)
        out = router_logits(x, w)
        per_row = mx.concatenate([x[i:i + 1].astype(mx.float32) @ w.astype(mx.float32).T for i in range(R)])
        assert np.array_equal(_bits(out), _bits(per_row))  # row r == its own 1-row decode
        stock = x.astype(mx.float32) @ w.astype(mx.float32).T
        assert float(mx.abs(out - stock).max()) < 1e-5
    assert router_logits(mx.zeros((9, 5120), mx.bfloat16), w) is None
    # outside MLX's matching GEMV tiling (split-K when K >= 16 E): stock path
    assert router_logits(mx.zeros((1, 5120), mx.bfloat16), w[:256]) is None


def test_router_gemv_decode_is_bit_exact(tiny, monkeypatch):
    (ta, la), (tb, lb) = _ab(monkeypatch, tiny, "ROUTER_GEMV", _run_greedy)
    assert ta == tb
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    va, vb = _ab(monkeypatch, tiny, "ROUTER_GEMV", _run_verify)
    assert np.abs(va - vb).max() < 0.05 * np.abs(va).max()
