import os

import mlx.core as mx
import numpy as np
import pytest

from jang_tools.turboquant.codebook import compute_codebook
from jang_tools.turboquant.hadamard_kernel import hadamard_rotate_metal
from jang_tools.turboquant.mpp_dense_kernel import (
    mpp_tensorops_available,
    tq_matmul_mpp_dense,
)
from jang_tools.turboquant.pipeline import pack_bits
from jang_tools.turboquant.rotation import generate_random_signs
from jang_tools.turboquant.tq_kernel import tq_matmul


pytestmark = pytest.mark.skipif(
    not mpp_tensorops_available(),
    reason="Metal Performance Primitives tensor_ops not available",
)


def _quantize_rows(weight: np.ndarray, bits: int) -> tuple[mx.array, mx.array, mx.array]:
    out_features, in_features = weight.shape
    codebook = mx.array(compute_codebook(in_features, bits), dtype=mx.float32)
    w = mx.array(weight.astype(np.float32))
    norms = mx.sqrt(mx.sum(w * w, axis=1, keepdims=True))
    norms_safe = mx.maximum(norms, mx.array(1e-8))
    w_normed = w / norms_safe
    boundaries = (codebook[:-1] + codebook[1:]) / 2.0
    indices = mx.zeros(w_normed.shape, dtype=mx.uint8)
    for boundary in boundaries:
        indices = indices + (w_normed > boundary).astype(mx.uint8)
    vals_per_u32 = 32 // bits
    pad = (-in_features) % vals_per_u32
    if pad:
        indices = mx.pad(indices, [(0, 0), (0, pad)])
    packed = pack_bits(indices.reshape(-1), bits).reshape(out_features, -1)
    mx.eval(packed, norms)
    return packed, norms.squeeze(-1).astype(mx.float16), codebook


def test_mpp_dense_tq_matmul_matches_existing_kernel_small_shape():
    rng = np.random.default_rng(7)
    in_features = 64
    out_features = 32
    bits = 4
    # MPP tensor_ops requires M to be 1, 2, 4, or a multiple of 8.
    x = mx.array(rng.standard_normal((4, in_features)).astype(np.float32))
    signs = mx.array(generate_random_signs(in_features, seed=123), dtype=mx.float32)
    w_rot = rng.standard_normal((out_features, in_features)).astype(np.float32)
    packed, norms, codebook = _quantize_rows(w_rot, bits)

    current = tq_matmul(x, packed, norms, codebook, signs, in_features, bits)
    mpp = tq_matmul_mpp_dense(x, packed, norms, codebook, signs, in_features, bits)

    mx.eval(current, mpp)
    np.testing.assert_allclose(np.array(mpp), np.array(current), rtol=2e-2, atol=2e-2)


def test_mpp_dense_tq_matmul_matches_rotated_reference():
    rng = np.random.default_rng(11)
    in_features = 64
    out_features = 16
    bits = 2
    x = mx.array(rng.standard_normal((2, in_features)).astype(np.float32))
    signs = mx.array(generate_random_signs(in_features, seed=321), dtype=mx.float32)
    w_rot = rng.standard_normal((out_features, in_features)).astype(np.float32)
    packed, norms, codebook = _quantize_rows(w_rot, bits)

    mpp = tq_matmul_mpp_dense(x, packed, norms, codebook, signs, in_features, bits)

    x_rot = hadamard_rotate_metal(x.astype(mx.float32), signs)
    idx = []
    vals_per_u32 = 32 // bits
    mask = (1 << bits) - 1
    packed_np = np.array(packed)
    for row in range(out_features):
        row_idx = []
        for i in range(in_features):
            pv = packed_np[row, i // vals_per_u32]
            row_idx.append((pv >> ((i % vals_per_u32) * bits)) & mask)
        idx.append(row_idx)
    idx = mx.array(np.array(idx, dtype=np.uint32))
    dense = mx.take(codebook, idx) * norms[:, None].astype(mx.float32)
    ref = x_rot @ dense.T

    mx.eval(mpp, ref)
    np.testing.assert_allclose(np.array(mpp), np.array(ref), rtol=2e-2, atol=2e-2)


def test_tq_matmul_opt_in_uses_mpp_dense_path(monkeypatch):
    from jang_tools.turboquant import tq_kernel

    rng = np.random.default_rng(17)
    in_features = 64
    out_features = 16
    bits = 4
    x = mx.array(rng.standard_normal((1, in_features)).astype(np.float32))
    signs = mx.array(generate_random_signs(in_features, seed=42), dtype=mx.float32)
    w_rot = rng.standard_normal((out_features, in_features)).astype(np.float32)
    packed, norms, codebook = _quantize_rows(w_rot, bits)

    called = {"mpp": 0}
    real = tq_kernel._tq_matmul_mpp_dense

    def wrapped(*args, **kwargs):
        called["mpp"] += 1
        return real(*args, **kwargs)

    monkeypatch.setenv("JANGTQ_MPP_DENSE", "1")
    monkeypatch.setattr(tq_kernel, "_tq_matmul_mpp_dense", wrapped)

    y = tq_kernel.tq_matmul(x, packed, norms, codebook, signs, in_features, bits)

    mx.eval(y)
    assert called["mpp"] == 1


def test_tq_matmul_default_does_not_use_mpp_dense_path(monkeypatch):
    from jang_tools.turboquant import tq_kernel

    rng = np.random.default_rng(19)
    in_features = 64
    out_features = 16
    bits = 4
    x = mx.array(rng.standard_normal((1, in_features)).astype(np.float32))
    signs = mx.array(generate_random_signs(in_features, seed=43), dtype=mx.float32)
    w_rot = rng.standard_normal((out_features, in_features)).astype(np.float32)
    packed, norms, codebook = _quantize_rows(w_rot, bits)

    def boom(*args, **kwargs):
        raise AssertionError("MPP path must not run without JANGTQ_MPP_DENSE")

    monkeypatch.delenv("JANGTQ_MPP_DENSE", raising=False)
    monkeypatch.setattr(tq_kernel, "_tq_matmul_mpp_dense", boom)

    y = tq_kernel.tq_matmul(x, packed, norms, codebook, signs, in_features, bits)

    mx.eval(y)


def test_tq_matmul_mpp_opt_in_falls_back_unless_strict(monkeypatch):
    from jang_tools.turboquant import tq_kernel

    rng = np.random.default_rng(23)
    in_features = 64
    out_features = 16
    bits = 4
    x = mx.array(rng.standard_normal((1, in_features)).astype(np.float32))
    signs = mx.array(generate_random_signs(in_features, seed=44), dtype=mx.float32)
    w_rot = rng.standard_normal((out_features, in_features)).astype(np.float32)
    packed, norms, codebook = _quantize_rows(w_rot, bits)

    def boom(*args, **kwargs):
        raise RuntimeError("simulated MPP compile failure")

    monkeypatch.setenv("JANGTQ_MPP_DENSE", "1")
    monkeypatch.delenv("JANGTQ_MPP_DENSE_STRICT", raising=False)
    monkeypatch.setattr(tq_kernel, "_tq_matmul_mpp_dense", boom)

    fallback = tq_kernel.tq_matmul(x, packed, norms, codebook, signs, in_features, bits)
    mx.eval(fallback)

    monkeypatch.setenv("JANGTQ_MPP_DENSE_STRICT", "1")
    with pytest.raises(RuntimeError, match="simulated MPP compile failure"):
        tq_kernel.tq_matmul(x, packed, norms, codebook, signs, in_features, bits)
