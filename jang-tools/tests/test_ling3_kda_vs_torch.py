"""Pin the MLX KDA implementation against the torch reference from `fla`.

Created by Jinho Jang (eric@jangq.ai)

The reference bodies below are transcribed from
`fla/ops/kda/naive.py::naive_recurrent_kda` and the `ShortConvolution` /
`FusedRMSNormGated` semantics that `modeling_bailing_moe_v3.py` relies on.

Shapes are deliberately chosen NOT to divide evenly by the conv kernel (4) or
the fla chunk size (64): T=37. An aligned length would make a padding/offset
bug invisible by construction.
"""

import mlx.core as mx
import numpy as np
import torch

from jang_tools.ling3.kda import kda_gate, kda_recurrent, kda_step, l2norm, short_conv


def _torch_naive_recurrent_kda(q, k, v, g, beta, scale=None, initial_state=None):
    """Verbatim port of fla.ops.kda.naive.naive_recurrent_kda (H == HV case)."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    q = q * scale
    S = q.new_zeros(B, H, K, V)
    if initial_state is not None:
        S = S + initial_state
    o = torch.zeros_like(v)
    for i in range(T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum(
            "b h k, b h v -> b h k v", b_i[..., None] * k_i, v_i - (k_i[..., None] * S).sum(-2)
        )
        o[:, i] = torch.einsum("b h k, b h k v -> b h v", q_i, S)
    return o, S


def _rand(*shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))


B, T, H, K, V = 2, 37, 4, 16, 16


def test_recurrence_matches_torch():
    q = _rand(B, T, H, K, seed=0)
    k = _rand(B, T, H, K, seed=1)
    v = _rand(B, T, H, V, seed=2)
    g = -torch.nn.functional.softplus(_rand(B, T, H, K, seed=3))
    beta = torch.sigmoid(_rand(B, T, H, seed=4))

    o_ref, s_ref = _torch_naive_recurrent_kda(q, k, v, g, beta)
    o_mlx, s_mlx = kda_recurrent(
        *(mx.array(x.numpy()) for x in (q, k, v, g, beta))
    )

    assert _rel_err(np.array(o_mlx), o_ref.numpy()) < 2e-5
    assert _rel_err(np.array(s_mlx), s_ref.numpy()) < 2e-5


def test_gate_matches_torch():
    """-exp(A_log) * softplus(f + dt_bias), clamped at lower_bound."""
    f = _rand(B, T, H, K, seed=5) * 4.0            # wide enough to exercise the clamp
    A_log = torch.log(torch.linspace(1, 16, H))
    dt_bias = _rand(H * K, seed=6)

    ref = -A_log.exp().view(1, 1, H, 1) * torch.nn.functional.softplus(
        f + dt_bias.view(H, K)
    )
    ref = ref.clamp(min=-5.0)

    got = kda_gate(mx.array(f.numpy()), mx.array(A_log.numpy()), mx.array(dt_bias.numpy()), -5.0)
    assert _rel_err(np.array(got), ref.numpy()) < 1e-5
    # the clamp must actually be exercised, or this test proves nothing
    assert (ref <= -5.0 + 1e-6).any()


def test_l2norm_matches_torch():
    x = _rand(B, T, H, K, seed=7)
    ref = torch.nn.functional.normalize(x.float(), dim=-1, eps=1e-6)
    got = l2norm(mx.array(x.numpy()))
    assert _rel_err(np.array(got), ref.numpy()) < 1e-4


def test_short_conv_matches_torch():
    C, W = 8, 4
    x = _rand(B, T, C, seed=8)
    w = _rand(C, 1, W, seed=9)

    ref = torch.nn.functional.conv1d(
        torch.nn.functional.pad(x.transpose(1, 2), (W - 1, 0)), w, groups=C
    ).transpose(1, 2)
    ref = torch.nn.functional.silu(ref)

    got, state = short_conv(mx.array(x.numpy()), mx.array(w.numpy()))
    assert _rel_err(np.array(got), ref.numpy()) < 2e-5
    assert state.shape == (B, W - 1, C)


def test_short_conv_state_carry_is_seamless():
    """Split at a boundary that is NOT a multiple of the kernel width."""
    C, W = 8, 4
    x = _rand(B, T, C, seed=10)
    w = _rand(C, 1, W, seed=11)
    full, _ = short_conv(mx.array(x.numpy()), mx.array(w.numpy()))

    split = 13  # 13 % 4 != 0
    a, st = short_conv(mx.array(x[:, :split].numpy()), mx.array(w.numpy()))
    b, _ = short_conv(mx.array(x[:, split:].numpy()), mx.array(w.numpy()), state=st)
    joined = mx.concatenate([a, b], axis=1)

    assert _rel_err(np.array(joined), np.array(full)) < 1e-6


def test_step_matches_recurrent_from_partial_state():
    """Prefill a non-aligned prefix, then decode token-by-token."""
    q = _rand(B, T, H, K, seed=12)
    k = _rand(B, T, H, K, seed=13)
    v = _rand(B, T, H, V, seed=14)
    g = -torch.nn.functional.softplus(_rand(B, T, H, K, seed=15))
    beta = torch.sigmoid(_rand(B, T, H, seed=16))
    mq, mk, mv, mg, mb = (mx.array(x.numpy()) for x in (q, k, v, g, beta))

    o_full, _ = kda_recurrent(mq, mk, mv, mg, mb)

    split = 23  # deliberately not a multiple of any block size
    o_pre, S = kda_recurrent(mq[:, :split], mk[:, :split], mv[:, :split], mg[:, :split], mb[:, :split])
    outs = [o_pre]
    for t in range(split, T):
        o_t, S = kda_step(mq[:, t], mk[:, t], mv[:, t], mg[:, t], mb[:, t], S)
        outs.append(o_t[:, None])
    joined = mx.concatenate(outs, axis=1)

    assert _rel_err(np.array(joined), np.array(o_full)) < 2e-5


def test_chunked_matches_recurrent_nonaligned():
    """kda_chunked must equal kda_recurrent at T NOT a multiple of 64,
    with and without an initial state."""
    from jang_tools.ling3.kda import kda_chunked
    Tn = 150                                     # 150 % 64 != 0
    q = _rand(B, Tn, H, K, seed=20)
    k = _rand(B, Tn, H, K, seed=21)
    v = _rand(B, Tn, H, V, seed=22)
    g = -torch.nn.functional.softplus(_rand(B, Tn, H, K, seed=23))
    beta = torch.sigmoid(_rand(B, Tn, H, seed=24))
    mq, mk, mv, mg, mb = (mx.array(x.numpy()) for x in (q, k, v, g, beta))

    # Tolerance is MEASURED, not aspirational: the port's A matrix matches the
    # torch reference to 2e-6 pre-substitution, but the 64-step WY recursion
    # amplifies fp32 noise ~1000x (realistic inputs: ~1.5e-3). Training-only.
    o_ref, s_ref = kda_recurrent(mq, mk, mv, mg, mb)
    o_chk, s_chk = kda_chunked(mq, mk, mv, mg, mb)
    assert _rel_err(np.array(o_chk), np.array(o_ref)) < 1e-2
    assert _rel_err(np.array(s_chk), np.array(s_ref)) < 1e-2

    st = mx.array(np.random.default_rng(3).standard_normal((B, H, K, V)).astype("float32")) * 0.1
    o_ref2, s_ref2 = kda_recurrent(mq, mk, mv, mg, mb, state=st)
    o_chk2, s_chk2 = kda_chunked(mq, mk, mv, mg, mb, state=st)
    assert _rel_err(np.array(o_chk2), np.array(o_ref2)) < 1e-2
    assert _rel_err(np.array(s_chk2), np.array(s_ref2)) < 1e-2
