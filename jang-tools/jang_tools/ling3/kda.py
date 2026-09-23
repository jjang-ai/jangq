"""Kimi Delta Attention (KDA) in MLX — the Ling-3.0 linear-attention block.

Created by Jinho Jang (eric@jangq.ai)

Reference: `fla.ops.kda` (chunk_kda / fused_recurrent_kda) as called by
`modeling_bailing_moe_v3.BailingMoeV3KimiDeltaAttention`.

The gated delta rule, per head, with state ``S`` of shape ``[K, V]``:

    D_t = diag(exp(g_t))                     # per-KEY-channel decay
    S_t = (I - beta_t k_t k_t^T) D_t S_{t-1} + beta_t k_t v_t^T
    o_t = S_t^T (q_t * scale)

where the gate is fused in the kernel as

    g_t = clamp(-exp(A_log) * softplus(f_proj(x) + dt_bias), min=lower_bound)

and ``q``/``k`` are L2-normalized along the head dim before entering the
recurrence (``use_qk_l2norm_in_kernel=True``), ``beta = sigmoid(b_proj(x))``,
``scale = K ** -0.5``.

This module implements the *recurrent* form. It is deliberately the simple,
auditable version: correctness of the state recurrence is what every downstream
imatrix / Hessian / AWQ / KL number depends on, and a hand-rolled chunked WY
transform is exactly the kind of subtly-wrong optimization that reads healthy.
`tests/test_kda_vs_torch.py` pins this against the torch reference.

Throughput comes from batching sequences (the per-step work is tiny and
launch-bound), not from chunking.
"""

from __future__ import annotations

import mlx.core as mx


def kda_gate(
    f: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    lower_bound: float | None = -5.0,
) -> mx.array:
    """Fused KDA decay gate: ``-exp(A_log) * softplus(f + dt_bias)``.

    Args:
        f: raw gate projection, shape ``[B, T, H, D]``.
        A_log: shape ``[H]``, fp32 in the checkpoint.
        dt_bias: shape ``[H * D]``, fp32 in the checkpoint.
        lower_bound: clamp floor (``kda_lower_bound``); ``None`` disables.

    Returns:
        Log-space per-channel decay, shape ``[B, T, H, D]``, fp32.
    """
    B, T, H, D = f.shape
    f = f.astype(mx.float32) + dt_bias.astype(mx.float32).reshape(H, D)
    # softplus, computed stably: log1p(exp(-|x|)) + max(x, 0)
    sp = mx.log1p(mx.exp(-mx.abs(f))) + mx.maximum(f, 0.0)
    g = -mx.exp(A_log.astype(mx.float32)).reshape(1, 1, H, 1) * sp
    if lower_bound is not None:
        g = mx.maximum(g, lower_bound)
    return g


def l2norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    """L2-normalize along the last axis, in fp32 (matches the fla kernel)."""
    x = x.astype(mx.float32)
    return x * mx.rsqrt(mx.sum(x * x, axis=-1, keepdims=True) + eps)


def kda_recurrent(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array | None = None,
    scale: float | None = None,
) -> tuple[mx.array, mx.array]:
    """Gated delta-rule recurrence.

    Args:
        q, k: ``[B, T, H, K]`` — already L2-normalized.
        v: ``[B, T, H, V]``.
        g: ``[B, T, H, K]`` — log-space per-key-channel decay (fp32).
        beta: ``[B, T, H]`` — already sigmoid'd (fp32).
        state: optional initial state ``[B, H, K, V]``.
        scale: query scale; defaults to ``K ** -0.5``.

    Returns:
        ``(o, final_state)`` with ``o`` of shape ``[B, T, H, V]``.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5

    q = q.astype(mx.float32) * scale
    k = k.astype(mx.float32)
    v = v.astype(mx.float32)
    g = g.astype(mx.float32)
    beta = beta.astype(mx.float32)

    S = mx.zeros((B, H, K, V), dtype=mx.float32) if state is None else state.astype(mx.float32)

    decay = mx.exp(g)  # [B, T, H, K]
    out = []
    for t in range(T):
        k_t = k[:, t]                      # [B, H, K]
        v_t = v[:, t]                      # [B, H, V]
        q_t = q[:, t]                      # [B, H, K]
        b_t = beta[:, t]                   # [B, H]

        S = S * decay[:, t][..., None]     # decay along the key axis
        # delta correction against the *decayed* state (order matters)
        kS = mx.sum(k_t[..., None] * S, axis=-2)          # [B, H, V]
        S = S + (b_t[..., None] * k_t)[..., None] * (v_t - kS)[..., None, :]
        out.append(mx.sum(q_t[..., None] * S, axis=-2))   # [B, H, V]

    return mx.stack(out, axis=1), S


def kda_step(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    S: mx.array,
    scale: float | None = None,
) -> tuple[mx.array, mx.array]:
    """Single-token KDA update — the decode path.

    Shapes are the ``T``-less versions of :func:`kda_recurrent`:
    ``q,k,g: [B,H,K]``, ``v: [B,H,V]``, ``beta: [B,H]``, ``S: [B,H,K,V]``.
    """
    K = q.shape[-1]
    if scale is None:
        scale = K ** -0.5
    q = q.astype(mx.float32) * scale
    k = k.astype(mx.float32)
    v = v.astype(mx.float32)
    S = S * mx.exp(g.astype(mx.float32))[..., None]
    kS = mx.sum(k[..., None] * S, axis=-2)
    S = S + (beta.astype(mx.float32)[..., None] * k)[..., None] * (v - kS)[..., None, :]
    return mx.sum(q[..., None] * S, axis=-2), S


def short_conv(
    x: mx.array,
    weight: mx.array,
    state: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Causal depthwise short convolution with silu, as ``ShortConvolution``.

    Args:
        x: ``[B, T, C]``.
        weight: ``[C, W]``. (The HF checkpoint stores ``[C, 1, W]``; the model's
            ``sanitize`` squeezes it to ``[C, W]`` so there is exactly one layout
            in play below this line.)
        state: optional left context ``[B, W-1, C]`` carried from a previous call.

    Returns:
        ``(y, new_state)`` — ``y`` is ``[B, T, C]``; ``new_state`` is the last
        ``W-1`` inputs, ready to prime the next call.
    """
    if weight.ndim == 3:
        weight = weight.reshape(weight.shape[0], -1)    # [C, 1, W] -> [C, W]
    C, W = weight.shape
    B, T, _ = x.shape

    if state is None:
        state = mx.zeros((B, W - 1, C), dtype=x.dtype)
    padded = mx.concatenate([state.astype(x.dtype), x], axis=1)   # [B, W-1+T, C]

    # depthwise causal conv: y[:, t] = sum_w padded[:, t + w] * weight[:, w]
    y = mx.zeros((B, T, C), dtype=mx.float32)
    for w in range(W):
        y = y + padded[:, w : w + T].astype(mx.float32) * weight[:, w].astype(mx.float32)

    new_state = padded[:, padded.shape[1] - (W - 1) :]
    y = y * mx.sigmoid(y)                    # silu
    return y.astype(x.dtype), new_state


def kda_chunked(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array | None = None,
    scale: float | None = None,
    chunk_size: int = 64,
) -> tuple[mx.array, mx.array]:
    """Chunked (WY-form) gated delta rule — for TRAINING, where the per-token
    recurrent loop exceeds Metal's per-graph op limit (~500k ops) once autodiff
    doubles it: 4.4k tokens x 18 layers x ~6 ops/token ~ 475k. This form runs
    ~T/64 chunk iterations of matmuls instead.

    Port of `fla.ops.kda.naive.naive_chunk_kda` (H == HV), with the reference's
    two per-token inner loops (Akk / Aqk construction) vectorized via
    broadcasting — a [BT, BT, K] tensor per chunk instead of BT scalar steps.
    Pinned against `kda_recurrent` in tests at non-aligned lengths.

    Same shapes as :func:`kda_recurrent`. `T` is padded internally to a
    multiple of `chunk_size` with beta=0 / k=0 rows (state-neutral by
    construction); output is sliced back to `T`.
    """
    B, T0, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5

    BT = chunk_size
    pad = (-T0) % BT
    if pad:
        zq = mx.zeros((B, pad, H, K), q.dtype)
        q = mx.concatenate([q, zq], axis=1)
        k = mx.concatenate([k, mx.zeros((B, pad, H, K), k.dtype)], axis=1)
        v = mx.concatenate([v, mx.zeros((B, pad, H, V), v.dtype)], axis=1)
        g = mx.concatenate([g, mx.zeros((B, pad, H, K), g.dtype)], axis=1)
        beta = mx.concatenate([beta, mx.zeros((B, pad, H), beta.dtype)], axis=1)
    T = T0 + pad
    NT = T // BT

    def chunked(x, d):
        # [B, T, H, d] -> [B, H, NT, BT, d]
        return x.reshape(B, NT, BT, H, d).transpose(0, 3, 1, 2, 4).astype(mx.float32)

    qc = chunked(q, K) * scale
    kc = chunked(k, K)
    vc = chunked(v, V)
    gc = mx.cumsum(chunked(g, K), axis=-2)                 # within-chunk cumsum
    bc = beta.reshape(B, NT, BT, H).transpose(0, 3, 1, 2).astype(mx.float32)

    # Akk (vectorized): A[..., c, j] = sum_d k[c,d] * exp(g[c,d] - g[j,d]) * k[j,d]
    # for c > j (strictly lower triangular).
    # Clamp BEFORE exp: on the used (lower-triangular) region gdiff <= 0
    # exactly (g is a cumsum of non-positives), so the clamp is lossless there;
    # the upper region would otherwise produce exp(+big)=inf, which is masked
    # in the FORWARD but turns into 0*inf = NaN in the BACKWARD.
    #
    # Memory: build Akk PER CHUNK. The all-chunks broadcast is
    # [B,H,NT,BT,BT,K] ~ 19 GB fp32 per layer at a 4.5k-token render (and its
    # exp is saved for backward, doubling it) — measured blowing a 128 GB
    # machine into 25 GB of swap. Per-chunk slices peak at ~270 MB instead.
    lower = mx.tril(mx.ones((BT, BT), dtype=mx.bool_), k=-1)
    akk_chunks = []
    for ci in range(NT):
        g_c = gc[:, :, ci]                                  # [B,H,BT,K]
        k_c = kc[:, :, ci]
        gd_c = mx.minimum(g_c[..., :, None, :] - g_c[..., None, :, :], 0.0)
        akk_chunks.append(mx.sum(k_c[..., :, None, :] * mx.exp(gd_c)
                                 * k_c[..., None, :, :], axis=-1))
    Akk = mx.stack(akk_chunks, axis=2)                      # [B,H,NT,BT,BT]
    A = mx.where(lower, -(Akk * bc[..., None]), mx.zeros_like(Akk))

    # forward substitution: (I - lower(A))^{-1}-style accumulation
    for i in range(1, BT):
        # A[..., i, :i] += sum_s A[..., i, s] * A[..., s, :i]
        upd = mx.sum(A[..., i, :, None] * A[..., :, :i], axis=-2)
        A[..., i, :i] = A[..., i, :i] + upd
    A = (A + mx.eye(BT, dtype=mx.float32)) * bc[..., None, :]

    w = A @ (mx.exp(gc) * kc)                               # [B,H,NT,BT,K]
    u = A @ vc                                              # [B,H,NT,BT,V]

    S = mx.zeros((B, H, K, V), dtype=mx.float32) if state is None else state.astype(mx.float32)
    strict_upper = mx.triu(mx.ones((BT, BT), dtype=mx.bool_), k=1)
    outs = []
    for i in range(NT):
        q_i, k_i = qc[:, :, i], kc[:, :, i]
        u_i, g_i, w_i = u[:, :, i], gc[:, :, i], w[:, :, i]
        # Aqk (vectorized): [..., c, j] = sum_d q[c,d] * exp(g[c,d]-g[j,d]) * k[j,d], c >= j
        gd = mx.minimum(g_i[..., :, None, :] - g_i[..., None, :, :], 0.0)   # same NaN guard
        Aqk = mx.sum(q_i[..., :, None, :] * mx.exp(gd) * k_i[..., None, :, :], axis=-1)
        Aqk = mx.where(strict_upper, mx.zeros_like(Aqk), Aqk)
        v_i = u_i - w_i @ S
        outs.append((q_i * mx.exp(g_i)) @ S + Aqk @ v_i)
        g_last = g_i[:, :, -1]                              # [B,H,K]
        S = S * mx.exp(g_last)[..., None]
        S = S + ((mx.exp(g_last[:, :, None, :] - g_i) * k_i).transpose(0, 1, 3, 2) @ v_i)

    o = mx.stack(outs, axis=2)                              # [B,H,NT,BT,V]
    o = o.transpose(0, 2, 3, 1, 4).reshape(B, T, H, V)
    return o[:, :T0], S
