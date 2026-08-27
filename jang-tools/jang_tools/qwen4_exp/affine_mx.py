"""GPU (MLX) port of the imatrix-weighted affine fit.

Same math as jang_tools.affine.quantize_imatrix_affine_numpy — per-group
alternating least squares under per-channel importance — but fully batched
on the M5 Max GPU: all rows/groups fit in one tensor program, ~50-100x the
numpy path on 3D expert tensors. Packing to the MLX ABI reuses the proven
affine.pack_bits on the resulting codes.

Verified against the numpy reference (see test in __main__).
"""

import mlx.core as mx
import numpy as np


def fit_affine_mx(w: mx.array, importance: mx.array, bits: int, group_size: int,
                  iterations: int = 6):
    """w: [..., in] fp32; importance: [in] (or [E, in] matching leading dim).
    Returns (codes uint8 [..., in], scales f16 groups, biases f16 groups)."""
    shape = w.shape
    gpr = shape[-1] // group_size
    g = w.reshape(*shape[:-1], gpr, group_size).astype(mx.float32)
    if importance.ndim == 1:
        h = importance.reshape(1, gpr, group_size)
        h = mx.broadcast_to(h, g.shape[:-2] + (gpr, group_size)) if g.ndim > 3 else \
            mx.broadcast_to(h, g.shape)
    else:  # [E, in] per-expert
        h = importance.reshape(shape[0], 1, gpr, group_size)
        h = mx.broadcast_to(h, g.shape)
    h = mx.maximum(h.astype(mx.float32), mx.array(1e-12, dtype=mx.float32))
    # median-based floor like the reference
    med = mx.array(np.median(np.asarray(importance).astype(np.float64))
                   * 1e-4, dtype=mx.float32)
    h = mx.maximum(h, mx.maximum(med, mx.array(1e-12, dtype=mx.float32)))

    bins = float((1 << bits) - 1)
    lo = g.min(axis=-1, keepdims=True)
    hi = g.max(axis=-1, keepdims=True)
    scale = mx.maximum((hi - lo) / bins, 1e-8)
    bias = lo
    codes = mx.clip(mx.round((g - bias) / scale), 0, bins)

    sum_h = (h).sum(-1, keepdims=True)
    sum_hw = (h * g).sum(-1, keepdims=True)
    for _ in range(iterations):
        sum_hq = (h * codes).sum(-1, keepdims=True)
        sum_hq2 = (h * codes * codes).sum(-1, keepdims=True)
        sum_hqw = (h * codes * g).sum(-1, keepdims=True)
        det = sum_h * sum_hq2 - sum_hq * sum_hq
        fs = (sum_h * sum_hqw - sum_hq * sum_hw) / mx.where(
            mx.abs(det) > 1e-20, det, mx.array(1.0, dtype=mx.float32))
        fb = (sum_hw - fs * sum_hq) / sum_h
        valid = mx.logical_and(mx.abs(det) > 1e-20, mx.abs(fs) >= 1e-8)
        scale = mx.where(valid, fs, scale)
        bias = mx.where(valid, fb, bias)
        codes = mx.clip(mx.round((g - bias) / scale), 0, bins)

    # f16 storage round-trip exactly like the ABI
    scale16 = scale.astype(mx.float16)
    scale16 = mx.where(scale16 == 0,
                       mx.array(np.float16(1e-8)), scale16)
    bias16 = bias.astype(mx.float16)
    codes = mx.clip(mx.round((g - bias16.astype(mx.float32)) /
                             scale16.astype(mx.float32)), 0, bins)
    mx.eval(codes, scale16, bias16)
    return (np.asarray(codes).astype(np.uint8).reshape(*shape[:-1], shape[-1]),
            np.asarray(scale16).squeeze(-1),
            np.asarray(bias16).squeeze(-1))


def refit_quantize_mx(w: mx.array, importance: mx.array, group_size: int, bits: int):
    """Drop-in for convert.refit_quantize's fitted path: returns packed
    uint32 codes + f16 scales/biases in the native MLX ABI."""
    from jang_tools.affine import pack_bits

    codes, scales, biases, = fit_affine_mx(w.astype(mx.float32),
                                           importance.astype(mx.float32),
                                           bits, group_size)
    shape = w.shape
    flat = codes.reshape(-1)
    packed = pack_bits(flat, bits).view(np.uint32).reshape(
        *shape[:-1], shape[-1] * bits // 32)
    return (mx.array(packed), mx.array(scales.astype(np.float16)),
            mx.array(biases.astype(np.float16)))


if __name__ == "__main__":
    from jang_tools.affine import quantize_imatrix_affine_numpy
    import mlx.core as mx

    rng = np.random.default_rng(0)
    for bits in (2, 3, 4, 6, 8):
        w = rng.normal(0, 1, (64, 256)).astype(np.float32)
        imp = (np.abs(rng.normal(1, 0.5, 256)) ** 2).astype(np.float32)
        pk_ref, sc_ref, bi_ref, err = quantize_imatrix_affine_numpy(
            w, imp, bits=bits, group_size=64)
        pk_mx, sc_mx, bi_mx = refit_quantize_mx(mx.array(w), mx.array(imp), 64, bits)
        deq_ref = mx.dequantize(mx.array(pk_ref), mx.array(sc_ref), mx.array(bi_ref),
                                group_size=64, bits=bits)
        deq_mx = mx.dequantize(pk_mx, sc_mx, bi_mx, group_size=64, bits=bits)
        e_ref = float(((np.asarray(deq_ref) - w) ** 2 * imp).sum())
        e_mx = float(((np.asarray(deq_mx) - w) ** 2 * imp).sum())
        print(f"bits {bits}: weighted err numpy {e_ref:.3f} vs mx {e_mx:.3f} "
              f"({'OK' if e_mx <= e_ref * 1.02 else 'WORSE'})")
        assert e_mx <= e_ref * 1.02
    print("MX AFFINE FIT MATCHES REFERENCE QUALITY")
