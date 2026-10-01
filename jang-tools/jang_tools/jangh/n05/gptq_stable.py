"""Numerically stable GPTQ for JANGH experts with a dominant-constant input distribution.

Pieces
  HessStore        : reads capture-v3 files; per-expert CENTERED covariance in float64 (+ mean), optional empirical-
                     Bayes shrinkage of poorly sampled experts toward the layer's pooled covariance.
  prepare()        : float64 CPU: prior/shrinkage -> Hadamard-32 rotation -> diagonal damping -> act-order
                     permutation -> upper factor U of H^-1 by ONE Cholesky of the order-reversed matrix
                     (U = J L^-1 J, L = chol(J H J)). No explicit H^-1, no second Cholesky.
  quantize_group() : GPU (MLX, run with MLX_ENABLE_TF32=0): GPTQ column loop with the JANGH codebook, exact
                     H-weighted row-scale refit, codes returned in the (rotated) storage column order, plus the
                     BIAS CORRECTION b = (W - W_hat) mu  (per expert, per output row).
Why float64 + centering: see stream_capture.py (96-99% of the MoE input energy is a constant vector; TF32 matmul).
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np



from jang_tools.jangh.encode import dequant  # noqa: E402
from jang_tools.jangh.format import h32  # noqa: E402
from jang_tools.jangh.gptq import gptq_encode  # noqa: E402

from jang_tools.jangh.n05.hess_cpu import HessStore, had32, rot_both, rot_vec, upper_factor, prepare, st_memmap  # noqa: E402,F401

LAST_EXTRA_DAMP: list = []      # per expert of the most recent quantize_group call (0.0 = base damping sufficed)


def quantize_group(Ws, Hs, mus, bits_list, rotate: bool, **prep):
    """Ws: list of mx (B,N,K) float32 weights sharing the input (e.g. [gate, up]). Hs: list of B float64 covariances.
    mus: (B,K) float64 input means. prep: kwargs of prepare() that are per-call constants, or per-expert lists
    under keys 'rows' / 'shrink_to'.
    Returns {bits: [ (Q uint8 (B,N,K) storage order, S f32 (B,N), bias f32 (B,N), W_hat f32 (B,N,K) rotated basis) ]}."""
    B = len(Hs)
    rows = prep.pop("rows", [0.0] * B)
    Us, Hps, perms, extras = [], [], [], []
    for i in range(B):
        U, Hp, perm, extra = prepare(Hs[i], rotate, rows=float(rows[i]), **prep)
        Us.append(U); Hps.append(Hp); perms.append(perm); extras.append(extra)
    LAST_EXTRA_DAMP[:] = extras
    U = mx.array(np.stack(Us)); Hp = mx.array(np.stack(Hps)); perm = mx.array(np.stack(perms))
    inv = mx.argsort(perm, axis=-1)
    mu_r = mx.array((rot_vec(mus) if rotate else mus).astype(np.float32))            # (B,K)
    out = {b: [] for b in bits_list}
    for W in Ws:
        Wr = h32(W) if rotate else W.astype(mx.float32)
        Wp = mx.take_along_axis(Wr, mx.broadcast_to(perm[:, None, :], Wr.shape), axis=2)
        for bits in bits_list:
            Q, S = gptq_encode(Wp, Hp, bits, U=U)
            Q = mx.take_along_axis(Q, mx.broadcast_to(inv[:, None, :], Q.shape), axis=2)
            Wh = dequant(Q, S, bits)                                                 # fp16-rounded scale, as the runtime
            bias = mx.sum((Wr - Wh) * mu_r[:, None, :], axis=-1)
            mx.eval(Q, S, bias, Wh)
            out[bits].append((Q, S, bias, Wh))
    del U, Hp
    return out
