"""Tests for jang_tools.turboquant.router_kernel (P15 sigmoid-bias top-k router)."""

import mlx.core as mx
import numpy as np

from jang_tools.turboquant.router_kernel import make_sigmoid_bias_topk_router


def test_router_kernel_importable():
    # Regression guard: load_jangtq P15 imports this at module load; if the file
    # is re-gitignored/removed, P15 silently disables. This must stay importable.
    r = make_sigmoid_bias_topk_router(experts=8, top_k=2)
    assert callable(r)


def test_router_matches_laguna_reference():
    from jang_tools.laguna.model import _compiled_sigmoid_bias_topk

    E, k = 16, 4
    rng = np.random.default_rng(0)
    gates = mx.array(rng.standard_normal(E).astype(np.float32))
    bias = mx.array(rng.standard_normal(E).astype(np.float32))

    inds, scores = make_sigmoid_bias_topk_router(E, k)(gates, bias)
    ref_inds, ref_scores = _compiled_sigmoid_bias_topk(k, gates, bias)
    mx.eval(inds, scores, ref_inds, ref_scores)

    got = {int(i): float(s) for i, s in zip(np.array(inds), np.array(scores))}
    ref = {int(i): float(s) for i, s in zip(np.array(ref_inds), np.array(ref_scores))}
    assert got.keys() == ref.keys()  # same top-k expert set
    for key in got:
        assert abs(got[key] - ref[key]) < 1e-5  # same renormalised weight
