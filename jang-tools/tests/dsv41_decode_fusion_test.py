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


_KERNEL = {"ROUTER_GEMV": ("router_logits",), "FUSED_MOE": ("moe_decode",), "FUSED_HC": ("hc_stage_decode",)}


def _run_sequential(model, rows=5):
    """The verify tokens decoded one forward each (what a verify row should reproduce)."""
    caches = S.new_caches(model.args)
    mx.eval(S.forward(model, PROMPT, caches, 0))
    out = [np.array(S.forward(model, [t], caches, len(PROMPT) + i).astype(mx.float32))[0]
           for i, t in enumerate([7, 300, 11, 900, 3, 77, 512, 64][:rows])]
    return np.stack(out)


def _ab(monkeypatch, model, attr, run):
    """run(model) with M.<attr> off, then on; with a kernel behind the flag, assert it actually ran."""
    from jang_tools.dsv41 import metal_decode as MD
    calls = []
    names = _KERNEL.get(attr, ())
    for name in names:
        def counted(*a, _orig=getattr(MD, name), **k):
            out = _orig(*a, **k)
            calls.append(out is not None)
            return out

        monkeypatch.setattr(MD, name, counted)
    monkeypatch.setattr(M, attr, False)
    a = run(model)
    assert not calls
    monkeypatch.setattr(M, attr, True)
    b = run(model)
    if names:
        assert calls and any(calls), f"{names} not taken"
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


# ----------------------------------------------------------------------------- fused hyper-connections

@pytest.mark.parametrize("dims", ["tiny", "real"])
def test_hc_stage_and_post_match_stock(dims):
    from jang_tools.dsv41 import metal_decode as MD
    args = S.tiny_args(1) if dims == "tiny" else S.real_args(1, 16)
    blk = S.build(args).layers[0]
    D = args.hidden_size
    for t in range(6):
        for R in (1, 3):
            x = (mx.random.normal((1, R, 4, D), key=mx.random.key(t)) * (0.3 + 2 * t)).astype(mx.bfloat16)
            pre = mx.softmax(mx.random.normal((1, R, 4), key=mx.random.key(50 + t)), -1)
            h = (mx.random.normal((1, R, D), key=mx.random.key(70 + t)) * 2).astype(mx.bfloat16)
            out = MD.hc_stage_decode(x[0], pre[0], blk.hc_ffn_fn, blk.hc_ffn_scale, blk.hc_ffn_base,
                                     blk.ffn_norm.weight, blk.ffn_norm.eps, blk.eps, blk.hc_eps, blk.iters)
            for i in range(R):  # each row == the stock ops on that row alone (= stock decode)
                xi = x[:, i:i + 1]
                ref = (blk.ffn_norm(M._hc_pre(xi, pre[:, i:i + 1])),
                       *blk.hc_mixes(xi, blk.hc_ffn_fn, blk.hc_ffn_scale, blk.hc_ffn_base))
                for a, b in zip(out, ref):
                    assert np.array_equal(_bits(a[i]), _bits(b.reshape(b.shape[2:])))
            if MD._tf32():  # the expand reproduces the M5 matrix units' TF32 GEMM; elsewhere the
                # runtime self-check keeps the stock ops (test_fused_hc_decode_is_bit_exact)
                post = MD.hc_post_decode(h[0], x[0], out[2], out[3])
                assert np.array_equal(_bits(post), _bits(M._hc_post(h, x, out[2][None], out[3][None])[0]))


@pytest.mark.parametrize("dims", ["tiny", "real"])
def test_hc_boundary_matches_stage(dims):
    """hc_mix + hc_boundary: hc_stage's normed input and hc_post's expand bit for bit; pre / post / comb (the
    reordered mix sums) within f32 rounding; every row its own."""
    from jang_tools.dsv41 import metal_decode as MD
    args = S.tiny_args(1) if dims == "tiny" else S.real_args(1, 16)
    blk = S.build(args).layers[0]
    D = args.hidden_size
    if D % 256:
        assert MD.hc_mix_decode(mx.zeros((1, 4, D), mx.bfloat16), blk.hc_ffn_fn) is None
        return
    common = (blk.hc_ffn_scale, blk.hc_ffn_base, blk.ffn_norm.weight, blk.ffn_norm.eps, blk.eps, blk.hc_eps,
              blk.iters)
    worst = 0.0
    for t in range(6):
        for R in (1, 3):
            x = (mx.random.normal((R, 4, D), key=mx.random.key(t)) * (0.3 + 2 * t)).astype(mx.bfloat16)
            pre = mx.softmax(mx.random.normal((R, 4), key=mx.random.key(50 + t)), -1)
            h = (mx.random.normal((R, D), key=mx.random.key(70 + t)) * 2).astype(mx.bfloat16)
            ref = MD.hc_stage_decode(x, pre, blk.hc_ffn_fn, *common)
            xs, part = MD.hc_mix_decode(x, blk.hc_ffn_fn)
            assert xs is x and part.shape == (R, 4 * D // 1024, 25)
            got = MD.hc_boundary_decode(xs, part, pre, *common)
            assert np.array_equal(_bits(got[0]), _bits(ref[0]))
            for a, b in zip(got[1:], ref[1:]):
                worst = max(worst, float(mx.abs(a - b).max().item()))
                assert mx.allclose(a, b, rtol=1e-5, atol=1e-6).item()
            for i in range(R):  # a row alone gives the same
                one = MD.hc_boundary_decode(x[i:i + 1], MD.hc_mix_decode(x[i:i + 1], blk.hc_ffn_fn)[1],
                                            pre[i:i + 1], *common)
                assert all(np.array_equal(_bits(a[0]), _bits(b[i])) for a, b in zip(one, got))
            if MD._tf32():
                xo, _ = MD.hc_mix_decode(x, blk.hc_ffn_fn, h, ref[2], ref[3])
                assert np.array_equal(_bits(xo), _bits(MD.hc_post_decode(h, x, ref[2], ref[3])))
    assert worst > 0.0 or dims == "tiny"  # the mixes really are summed in another order (not vacuous)


def _run_forced(model, tokens):
    """Prefill PROMPT, then feed `tokens` one forward each: the last-row logits of every forward (teacher-forced)."""
    caches = S.new_caches(model.args)
    out = [np.array(S.forward(model, PROMPT, caches, 0).astype(mx.float32)).reshape(-1, model.args.vocab_size)[-1]]
    for i, t in enumerate(tokens):
        out.append(np.array(S.forward(model, [t], caches, len(PROMPT) + i).astype(mx.float32)).reshape(-1))
    return np.stack(out)


def _assert_close_forced(a, b, max_flips=2):
    """Teacher-forced logits of two decode paths agree: at every step but at most `max_flips` the cosine is above
    0.9998 and the argmax is the same. A path that only reorders sums moves a random synthetic model's logits by
    TF32-rounding-sized amounts, and the model's near-tied routing occasionally turns that into one step that
    differs more (measured: none on an M5 Max, one of 13 on an M5 Ultra); a real defect moves most steps."""
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    off = int((cos <= 0.9998).sum())
    assert np.median(cos) > 0.9998 and off <= max_flips and cos.min() > 0.8, cos
    assert int((a.argmax(-1) != b.argmax(-1)).sum()) <= max_flips


FORCED_TOKENS = [587, 573, 358, 967, 685, 461, 787, 89, 12, 400, 33, 777]


def test_hc_boundary_decode_close_to_stage(monkeypatch):
    """Whole decode, teacher-forced, with the 2-dispatch boundary vs the bit-identical stage: logits within the
    reorder noise at every step (real hidden size: the boundary needs D % 256 == 0); the boundary really ran."""
    from jang_tools.dsv41 import metal_decode as MD
    if not MD._tf32():
        pytest.skip("the fused hc path (and so the boundary) is off without TF32 GEMMs")
    model = S.build(S.real_args(2, 16))
    calls = []
    orig = MD.hc_boundary_decode

    def counted(*a, **k):
        out = orig(*a, **k)
        calls.append(out is not None)
        return out

    monkeypatch.setattr(MD, "hc_boundary_decode", counted)
    M._HC_BOUNDARY_OK.clear()
    monkeypatch.setattr(M, "HC_BOUNDARY", False)
    a = _run_forced(model, FORCED_TOKENS)
    assert not calls
    monkeypatch.setattr(M, "HC_BOUNDARY", True)
    b = _run_forced(model, FORCED_TOKENS)
    assert calls and all(calls) and all(M._HC_BOUNDARY_OK.values())
    _assert_close_forced(a, b)


# ----------------------------------------------------------------------------- split sparse attention

@pytest.mark.parametrize("S", [1, 3])
def test_attend_decode_matches_sparse_attention(S):
    """attend_decode (window rows + 4-bit pool picks read by index) vs the gather + sparse_attention path, with
    masked window and pick slots: equal to the reordered sums' rounding; window-only rows too."""
    from jang_tools.dsv41 import metal_decode as MD
    k = mx.random.key
    H, D, L, T, KC = 64, 512, 128 + S, 1500, 512
    q = (mx.random.normal((1, S, H, D), key=k(11)) * 0.5).astype(mx.bfloat16)
    raw = (mx.random.normal((1, L, D), key=k(12)) * 0.5).astype(mx.bfloat16)
    pool = M.QuantGrowBuffer()
    pool.append((mx.random.normal((1, T, D), key=k(13)) * 0.5).astype(mx.bfloat16))
    widx = (mx.arange(S)[:, None] + mx.arange(128)[None] + (L - 128 - S + 1))[None].astype(mx.int32)
    widx = mx.where(mx.arange(128)[None, None] < 2, -1, widx)  # the window's first slots empty
    cidx = mx.random.randint(0, T, (1, S, KC), key=k(14)).astype(mx.int32)
    cidx = mx.where(mx.arange(KC)[None, None] % 97 == 0, -1, cidx)
    sink = mx.random.normal((H,), key=k(15))
    for c in (cidx, None):
        got = MD.attend_decode(q[0], raw[0], widx[0], None if c is None else c[0],
                               None if c is None else (pool.wq[0], pool.scales[0], pool.biases[0]), sink,
                               D ** -0.5, M.GROUP_SIZE)
        kv, idxs = (raw, widx) if c is None else M.append_compressed_rows(raw, widx, c, pool, T, None, gather=True)
        ref = M.sparse_attention(q, kv, sink, idxs, D ** -0.5)[0]
        rel = float((mx.abs(got - ref).max() / mx.abs(ref).max()).item())
        assert got.shape == (S, H, D) and got.dtype == mx.float32
        assert 0.0 < rel < 2e-3, rel  # not vacuous: the sums do run in another order


def test_attn_split_decode_close_to_gather(monkeypatch):
    """Whole decode (real attention width, compressed and indexed layers), teacher-forced: the split attention vs
    the gather + core with the core's matmuls in full fp32 (on the CPU stream). On M5 GPUs MLX's fp32 matmuls
    round to TF32 (MLX_ENABLE_TF32=1) and the split kernel does not, so against the GPU core the difference is the
    core's TF32 rounding, which a random synthetic model's near-tied routing then amplifies (measured: cos 0.87-0.97
    at three of 12 steps with TF32, >= 0.99987 at every step without). Close at every step but an isolated routing
    flip (_assert_close_forced); the kernel really ran."""
    from jang_tools.dsv41 import metal_decode as MD
    model = S.build(S.real_args(6, 16))
    tokens = FORCED_TOKENS
    calls = []
    orig = MD.attend_decode

    def counted(*a, **kw):
        out = orig(*a, **kw)
        calls.append(out is not None)
        return out

    monkeypatch.setattr(MD, "attend_decode", counted)
    monkeypatch.setattr(M, "_attn_core", lambda *a: _core_cpu(*a))
    monkeypatch.setattr(M, "ATTN_SPLIT", False)
    a = _run_forced(model, tokens)
    assert not calls
    monkeypatch.setattr(M, "ATTN_SPLIT", True)
    b = _run_forced(model, tokens)
    assert calls and all(calls) and M._ATTN_SPLIT_OK == [True]
    _assert_close_forced(a, b)


def _core_cpu(q, kv, sink, ix, scale):
    """sparse_attention's core (model._attn_core's arithmetic) on the CPU stream: fp32 matmuls without TF32."""
    with mx.stream(mx.cpu):
        valid = ix >= 0
        g = mx.take(kv, mx.where(valid, ix, 0), axis=0)
        s = (q.astype(mx.float32) @ g.transpose(0, 2, 1)) * scale
        s = mx.where(valid[:, None, :], s, -mx.inf)
        m = mx.maximum(mx.max(s, axis=-1, keepdims=True), sink[None, :, None])
        p = mx.exp(s - m)
        denom = mx.sum(p, axis=-1, keepdims=True) + mx.exp(sink[None, :, None] - m)
        return (p @ g) / denom


def test_fused_hc_decode_is_bit_exact(tiny, monkeypatch):
    from jang_tools.dsv41 import metal_decode as MD
    M._HC_FUSED_OK.clear()
    monkeypatch.setattr(M, "HC_BOUNDARY", False)  # the 3-dispatch stage (the 2-dispatch boundary reorders the mixes)
    monkeypatch.setattr(M, "ATTN_SPLIT", False)
    if not MD._tf32():
        # This device's f32 GEMMs are not TF32 (or MLX_ENABLE_TF32=0): the fused hc path must stay off,
        # so decode with the flag on is the stock decode.
        monkeypatch.setattr(M, "FUSED_HC", False)
        ta, la = _run_greedy(tiny)
        monkeypatch.setattr(M, "FUSED_HC", True)
        tb, lb = _run_greedy(tiny)
        assert ta == tb and np.array_equal(la.view(np.uint32), lb.view(np.uint32))
        assert M._HC_FUSED_OK and not any(M._HC_FUSED_OK.values())
        return
    (ta, la), (tb, lb) = _ab(monkeypatch, tiny, "FUSED_HC", _run_greedy)
    assert ta == tb
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    assert all(M._HC_FUSED_OK.values()) and M._HC_FUSED_OK
    # Verify rows: stock runs the mixes / einsum as TF32 GEMMs over the rows, the fused stage computes
    # each row as its 1-row decode, so the fused verify is closer to sequential decode.
    va, vb = _ab(monkeypatch, tiny, "FUSED_HC", _run_verify)
    monkeypatch.setattr(M, "FUSED_HC", False)
    seq = _run_sequential(tiny)
    assert np.abs(vb - seq).max() <= np.abs(va - seq).max()


def test_all_fusions_together_decode_bit_exact(tiny, monkeypatch):
    monkeypatch.setattr(M, "HC_BOUNDARY", False)  # not bit-identical by design (test_hc_boundary_*)
    monkeypatch.setattr(M, "ATTN_SPLIT", False)
    flags = ("ROUTER_GEMV", "FUSED_MOE", "FUSED_HC", "LAYER_ASYNC")
    for f in flags:
        monkeypatch.setattr(M, f, False)
    a, la = _run_greedy(tiny, 16)
    for f in flags:
        monkeypatch.setattr(M, f, True)
    b, lb = _run_greedy(tiny, 16)
    assert a == b and np.array_equal(la.view(np.uint32), lb.view(np.uint32))
