"""JANGTQ v2 encoding: per-row scale solved by (optionally importance-weighted) least squares,
alternating with nearest-level assignment on the v2 odd-cubic codebook."""
from __future__ import annotations

import mlx.core as mx

from .format import codebook, nearest_level, pack_bitstream


def encode(W: mx.array, bits: int, col_weight: mx.array | None = None, iters: int = 6):
    """W (..., N, K) float -> (q uint8 (..., N, K), scale float32 (..., N)).
    col_weight: (K,) or (..., 1, K) nonnegative input importance (E[x^2]); weights the LS scale objective
    sum_k w_k (W_rk - s_r c_rk)^2. Assignment is per element nearest level (independent of w for a fixed scale)."""
    W = W.astype(mx.float32)
    cb = mx.array(codebook(bits))
    wgt = 1.0 if col_weight is None else col_weight.astype(mx.float32)
    s = mx.maximum(mx.sqrt(mx.mean(W * W, axis=-1, keepdims=True)), 1e-12)
    for _ in range(iters):
        c = cb[nearest_level(W / s, cb)]
        num = mx.sum(wgt * W * c, axis=-1, keepdims=True)
        den = mx.maximum(mx.sum(wgt * c * c, axis=-1, keepdims=True), 1e-20)
        s = mx.maximum(num / den, 1e-12)
    q = nearest_level(W / s, cb)
    return q, s.squeeze(-1)


def dequant(q: mx.array, scale: mx.array, bits: int, scale_dtype=mx.float16) -> mx.array:
    """Exactly what the runtime computes: fp16-stored scale times codebook level."""
    cb = mx.array(codebook(bits))
    return cb[q] * scale.astype(scale_dtype).astype(mx.float32)[..., None]


def encode_packed(W, bits, col_weight=None):
    q, s = encode(W, bits, col_weight)
    return pack_bitstream(q, bits), s.astype(mx.float16), q
