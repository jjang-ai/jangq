"""Quantization primitives for the MiMo-V2.6 JANG build (MLX, GPU).

One implementation shared by the calibration/allocation pass and the
converter, so the error the allocator measured is the error that ships.

* ``pack_codes`` — LSB bitstream packing identical to MLX affine storage
  (verified by ``_selftest``: mx.dequantize(pack(codes), 1, 0) == codes for
  bits 2/3/4/5/6/8).
* ``fit_affine`` — imatrix-weighted per-group affine fit (alternating least
  squares, same math as jang_tools.qwen4_exp.affine_mx) but with the scale /
  bias STORAGE dtype as a parameter (bf16 here: avoids the fp16-scale x bf16
  activation promotion to fp32 recorded in the e8m0/matmul2d analysis).
* ``quantize_affine`` — RTN (mx.quantize on bf16) when no importance is given,
  imatrix fit otherwise.
"""

from __future__ import annotations

import mlx.core as mx


def pack_codes(codes: mx.array, bits: int) -> mx.array:
    """codes: integer array [..., n] with n % 32 == 0 -> uint32 [..., n*bits/32]."""
    n = codes.shape[-1]
    if n % 32:
        raise ValueError(f"last dim {n} not divisible by 32")
    c = codes.astype(mx.uint32).reshape(*codes.shape[:-1], n // 32, 32)
    words = [mx.zeros(c.shape[:-1], dtype=mx.uint32) for _ in range(bits)]
    for j in range(32):
        off = j * bits
        w, sh = divmod(off, 32)
        v = c[..., j]
        words[w] = words[w] | (v << sh)
        if sh + bits > 32:
            words[w + 1] = words[w + 1] | (v >> (32 - sh))
    return mx.stack(words, axis=-1).reshape(*codes.shape[:-1], n * bits // 32)


def fit_affine(w: mx.array, importance: mx.array, *, bits: int, group_size: int,
               iterations: int = 6, storage=mx.bfloat16):
    """w [..., n] fp32; importance [n] or [E, n] (leading dim of w) -> (packed, scales, biases)."""
    shape = w.shape
    g = w.reshape(*shape[:-1], shape[-1] // group_size, group_size).astype(mx.float32)
    imp = importance.astype(mx.float32)
    if imp.ndim == 1:
        h = mx.broadcast_to(imp.reshape(shape[-1] // group_size, group_size), g.shape)
    else:
        h = mx.broadcast_to(imp.reshape(imp.shape[0], *([1] * (g.ndim - 3)), shape[-1] // group_size, group_size), g.shape)
    floor = mx.maximum(mx.mean(imp) * 1e-4, 1e-12)
    h = mx.maximum(h, floor)
    bins = float((1 << bits) - 1)
    lo = g.min(axis=-1, keepdims=True)
    hi = g.max(axis=-1, keepdims=True)
    scale = mx.maximum((hi - lo) / bins, 1e-8)
    bias = lo
    codes = mx.clip(mx.round((g - bias) / scale), 0, bins)
    sh = h.sum(-1, keepdims=True)
    shw = (h * g).sum(-1, keepdims=True)
    for _ in range(iterations):
        shq = (h * codes).sum(-1, keepdims=True)
        shq2 = (h * codes * codes).sum(-1, keepdims=True)
        shqw = (h * codes * g).sum(-1, keepdims=True)
        det = sh * shq2 - shq * shq
        ok = mx.abs(det) > 1e-20
        fs = (sh * shqw - shq * shw) / mx.where(ok, det, 1.0)
        fb = (shw - fs * shq) / sh
        ok = mx.logical_and(ok, mx.abs(fs) >= 1e-8)
        scale = mx.where(ok, fs, scale)
        bias = mx.where(ok, fb, bias)
        codes = mx.clip(mx.round((g - bias) / scale), 0, bins)
    # Round-trip through storage dtype, then re-derive codes against the stored grid.
    s_st = scale.astype(storage)
    s_st = mx.where(s_st == 0, mx.array(1e-8, dtype=storage), s_st)
    b_st = bias.astype(storage)
    codes = mx.clip(mx.round((g - b_st.astype(mx.float32)) / s_st.astype(mx.float32)), 0, bins)
    packed = pack_codes(codes.reshape(shape).astype(mx.uint32), bits)
    return packed, s_st.squeeze(-1), b_st.squeeze(-1)


def quantize_affine(w: mx.array, *, bits: int, group_size: int, importance: mx.array | None = None):
    if importance is None:
        return mx.quantize(w.astype(mx.bfloat16), group_size=group_size, bits=bits, mode="affine")
    return fit_affine(w, importance, bits=bits, group_size=group_size)


def dequant_affine(packed, scales, biases, *, bits, group_size):
    return mx.dequantize(packed, scales, biases, group_size=group_size, bits=bits, mode="affine")


def _selftest():
    import numpy as np
    rng = np.random.default_rng(0)
    for bits in (2, 3, 4, 5, 6, 8):
        codes = mx.array(rng.integers(0, 1 << bits, (3, 256)).astype(np.uint32))
        p = pack_codes(codes, bits)
        ones = mx.ones((3, 256 // 64), dtype=mx.float32)
        zeros = mx.zeros((3, 256 // 64), dtype=mx.float32)
        d = mx.dequantize(p, ones, zeros, group_size=64, bits=bits, mode="affine")
        assert mx.array_equal(d.astype(mx.uint32), codes).item(), f"pack mismatch bits={bits}"
        w = mx.array(rng.normal(0, 1, (8, 512)).astype(np.float32))
        imp = mx.array((np.abs(rng.normal(1, 0.5, 512)) ** 2).astype(np.float32))
        rq, rs, rb = mx.quantize(w.astype(mx.bfloat16), group_size=64, bits=bits, mode="affine")
        fq, fs, fb = fit_affine(w, imp, bits=bits, group_size=64)
        e_r = float((((dequant_affine(rq, rs, rb, bits=bits, group_size=64).astype(mx.float32) - w) ** 2) * imp).sum())
        e_f = float((((dequant_affine(fq, fs, fb, bits=bits, group_size=64).astype(mx.float32) - w) ** 2) * imp).sum())
        print(f"bits {bits}: pack OK; weighted err RTN {e_r:.3f} -> imatrix fit {e_f:.3f}")
        assert e_f <= e_r * 1.001, "imatrix fit must not be worse than RTN on its own objective"
    print("v26_quant selftest OK")


if __name__ == "__main__":
    _selftest()
