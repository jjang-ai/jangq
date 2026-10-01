"""GPTQ with the JANGTQ v2 codebook quantizer.

Batched over experts: W (B, N, K), H (B, K, K) -> codes q (B, N, K) uint8, per-row scale (B, N).
Per-row scale is first set by RTN least squares, GPTQ then chooses codes column by column with error feedback
through the upper Cholesky factor of H^-1, and finally the scale is refit to the exact H-weighted optimum for the
chosen codes: s_r = (w_r^T H c_r) / (c_r^T H c_r).  All float32; Cholesky on the CPU stream (MLX linalg)."""
from __future__ import annotations

import mlx.core as mx

from .encode import encode
from .format import codebook, nearest_level


def _hinv_upper(H: mx.array, damp: float, damp_mode: str = "mean") -> mx.array:
    """Upper Cholesky factor U of H^-1 (H^-1 = U^T U), with relative damping.
    damp_mode "mean": damp * mean(diag) * I (GPTQ default; GLM-5.3 builds).
    damp_mode "diag": damp * diag(H) (+ 1e-6 * mean(diag) * I floor). Required when inputs have MASSIVE channels:
    mean(diag) is then dominated by a handful of channels and mean-damping swamps every normal channel."""
    B, K, _ = H.shape
    dg = mx.diagonal(H, axis1=-2, axis2=-1)
    d = mx.mean(dg, axis=-1, keepdims=True)
    if damp_mode == "diag":
        Hd = H + (damp * dg + 1e-6 * d)[:, :, None] * mx.eye(K)[None]
    else:
        Hd = H + (damp * d)[..., None] * mx.eye(K)[None]
    L = mx.linalg.cholesky(Hd, stream=mx.cpu)                 # Hd = L L^T
    Linv = mx.linalg.inv(L, stream=mx.cpu)                     # triangular inverse
    Hinv = mx.matmul(mx.swapaxes(Linv, -1, -2), Linv)          # (L L^T)^-1 = L^-T L^-1
    U = mx.swapaxes(mx.linalg.cholesky(Hinv, stream=mx.cpu), -1, -2)   # Hinv = C C^T -> U = C^T
    return U


def gptq_encode(W: mx.array, H: mx.array, bits: int, damp: float = 0.01, block: int = 128, U: mx.array | None = None,
                damp_mode: str = "mean"):
    B, N, K = W.shape
    W = W.astype(mx.float32)
    cb = mx.array(codebook(bits))
    _, s = encode(W, bits)                                     # (B, N) RTN LS scale
    s = s[..., None]
    if U is None:
        U = _hinv_upper(H.astype(mx.float32), damp, damp_mode) # (B, K, K); pass U to share across gate/up
    Q = mx.zeros((B, N, K), dtype=mx.uint8)
    Wc = W
    for b0 in range(0, K, block):
        b1 = min(b0 + block, K)
        Wb = Wc[..., b0:b1]
        Ub = U[:, b0:b1, b0:b1]
        Err = []
        qs = []
        for j in range(b1 - b0):
            w = Wb[..., j]                                     # (B, N)
            q = nearest_level(w / s[..., 0], cb)
            wq = cb[q] * s[..., 0]
            e = (w - wq) / Ub[:, j, j][:, None]
            Wb = Wb - e[..., None] * Ub[:, j, :][:, None, :] * (mx.arange(b1 - b0) > j)[None, None, :]
            Err.append(e); qs.append(q)
        E_ = mx.stack(Err, axis=-1)                            # (B, N, blk)
        Q[..., b0:b1] = mx.stack(qs, axis=-1)
        if b1 < K:
            Wc = mx.concatenate([Wc[..., :b1], Wc[..., b1:] - mx.matmul(E_, U[:, b0:b1, b1:])], axis=-1)
        mx.eval(Q, Wc)
    # exact H-weighted scale refit for the chosen codes
    C = cb[Q]                                                  # (B, N, K)
    WH = mx.matmul(W, H)                                       # (B, N, K)
    CH = mx.matmul(C, H)
    num = mx.sum(WH * C, axis=-1); den = mx.maximum(mx.sum(CH * C, axis=-1), 1e-20)
    s_new = mx.maximum(num / den, 1e-12)
    return Q, s_new
