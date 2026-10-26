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


_KERNEL = {"ROUTER_GEMV": "router_logits", "FUSED_MOE": "moe_decode", "FUSED_HC": "hc_mixes_decode"}


def _ab(monkeypatch, model, attr, run):
    """run(model) with M.<attr> off, then on; with a kernel behind the flag, assert it actually ran."""
    from jang_tools.dsv41 import metal_decode as MD
    calls = []
    name = _KERNEL.get(attr)
    if name:
        orig = getattr(MD, name)

        def counted(*a, **k):
            out = orig(*a, **k)
            calls.append(out is not None)
            return out

        monkeypatch.setattr(MD, name, counted)
    monkeypatch.setattr(M, attr, False)
    a = run(model)
    assert not calls
    monkeypatch.setattr(M, attr, True)
    b = run(model)
    if name:
        assert calls and all(calls), f"{name} not taken"
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


# ----------------------------------------------------------------------------- fused MoE

_QMV_SRC = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint out_row = threadgroup_position_in_grid.x * 8u + simdgroup_index_in_threadgroup * 4u;
    const int in_w = K * dsv_bytes_per_pack<BITS>() / dsv_pack_factor<BITS>();
    const device uint8_t* w[1] = {(const device uint8_t*)wq + out_row * in_w};
    const device T* s[1] = {sc + out_row * (K / 64)};
    const device T* b[1] = {bi + out_row * (K / 64)};
    float acc[4] = {0, 0, 0, 0};
    dsv_qmv4<T, 64, BITS, FAST, 1>(w, s, b, x, K, lane, acc);
    for (int r = 0; r < 4; r++) { float v = simd_sum(acc[r]); if (lane == 0) y[out_row + r] = static_cast<T>(v); }
"""


@pytest.mark.parametrize("bits", [2, 3, 8])
@pytest.mark.parametrize("K", [768, 1024, 2304, 5120])
def test_qmv_walk_matches_mlx_in_f32(bits, K):
    """The kernels' projection walk (qmv_fast / qmv chosen by metal_decode._fast) reproduces MLX's
    1-row quantized_matmul and gather_qmm bit for bit -- checked on f32 outputs, where any change
    of accumulation order shows (bf16 outputs would hide most of it)."""
    from jang_tools.dsv41 import metal_decode as MD
    from jang_tools.dsv41.tests.synth import _rand_quant
    k = mx.fast.metal_kernel(name="dsv41_test_qmv", header=MD._QHEADER, source=_QMV_SRC,
                             input_names=["x", "wq", "sc", "bi"], output_names=["y"])
    N = 512
    w, s, b = _rand_quant(mx.random.key(K + bits), (N, K * bits // 32), bits, K)
    s, b = s.astype(mx.float32), b.astype(mx.float32)
    for t in range(3):
        x = (mx.random.normal((1, K), key=mx.random.key(t)) * 3).astype(mx.bfloat16).astype(mx.float32)
        out = k(inputs=[x[0], w, s, b], template=[("T", mx.float32), ("K", K), ("BITS", bits), ("FAST", MD._fast(bits, K))],
                grid=(N // 8 * 64, 1, 1), threadgroup=(64, 1, 1), output_shapes=[(N,)], output_dtypes=[mx.float32])[0]
        ref = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=bits)[0]
        gref = mx.gather_qmm(x[None], w[None], s[None], b[None], rhs_indices=mx.array([[0]], mx.uint32),
                             transpose=True, group_size=64, bits=bits).reshape(-1)
        assert np.array_equal(np.array(out), np.array(ref))
        assert np.array_equal(np.array(out), np.array(gref))


def _route(blk, xf):
    import mlx.nn as nn
    g = blk.ffn.gate
    scores = mx.sqrt(nn.softplus(xf.astype(mx.float32) @ g.weight.astype(mx.float32).T))
    idx = mx.argpartition(-(scores + g.bias), kth=blk.topk - 1, axis=-1)[:, :blk.topk]
    w = mx.take_along_axis(scores, idx, axis=-1)
    return idx, w / (mx.sum(w, -1, keepdims=True) + 1e-20) * blk.route_scale


@pytest.mark.parametrize("dims,gate_bits", [("tiny", 2), ("tiny", 3), ("real", 2), ("real", 3)])
def test_moe_decode_matches_stock(dims, gate_bits, monkeypatch):
    from jang_tools.dsv41.metal_decode import moe_decode
    args = S.tiny_args(1) if dims == "tiny" else S.real_args(1, 16)
    blk = S.build(args, expert_bits=lambda i: (gate_bits, 3)).layers[0]
    D = args.hidden_size
    for t in range(6):  # 1 row: bit-identical to Block.moe (stock), scales up into the SwiGLU clamp
        x = (mx.random.normal((1, 1, D), key=mx.random.key(100 + t)) * (0.5 + t)).astype(mx.bfloat16)
        monkeypatch.setattr(M, "FUSED_MOE", False)
        ref = blk.moe(x)
        monkeypatch.setattr(M, "FUSED_MOE", True)
        assert np.array_equal(_bits(blk.moe(x)), _bits(ref))
    for R in (2, 5, 8):  # R rows: each row == its own 1-row computation, and close to the stock batch
        x = (mx.random.normal((R, D), key=mx.random.key(R)) * 2).astype(mx.bfloat16)
        rr = [_route(blk, x[i:i + 1]) for i in range(R)]
        idx, w = mx.concatenate([a for a, _ in rr]), mx.concatenate([b for _, b in rr])
        out = moe_decode(x, idx, w, blk.ffn.experts, blk.ffn.shared_experts, blk.swiglu_limit)
        per = mx.concatenate([moe_decode(x[i:i + 1], idx[i:i + 1], w[i:i + 1], blk.ffn.experts,
                                         blk.ffn.shared_experts, blk.swiglu_limit) for i in range(R)])
        assert np.array_equal(_bits(out), _bits(per))
        monkeypatch.setattr(M, "FUSED_MOE", False)
        stock = blk.moe(x[None])[0].astype(mx.float32)
        monkeypatch.setattr(M, "FUSED_MOE", True)
        assert float(mx.abs(blk.moe(x[None])[0].astype(mx.float32) - stock).max()) <= 0.01 * float(mx.abs(stock).max())


def test_fused_moe_decode_is_bit_exact(tiny, monkeypatch):
    (ta, la), (tb, lb) = _ab(monkeypatch, tiny, "FUSED_MOE", _run_greedy)
    assert ta == tb
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    va, vb = _ab(monkeypatch, tiny, "FUSED_MOE", _run_verify)
    assert np.abs(va - vb).max() < 0.05 * np.abs(va).max()
