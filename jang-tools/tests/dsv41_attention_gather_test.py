"""Decode-path compressed-row gather is bit-identical to the full-pool path (CPU only)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from jang_tools.dsv41.model import QuantGrowBuffer, Shared, append_compressed_rows, sparse_attention


@pytest.fixture(autouse=True)
def _cpu_only():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


@pytest.mark.parametrize("S", [1, 3, 6])
def test_gather_equals_full_pool(S):
    rng = np.random.default_rng(S)
    D, H, T, W, K = 128, 4, 300, 16, 24
    pool = QuantGrowBuffer()
    pool.append(mx.array(rng.standard_normal((1, T, D)).astype(np.float32)).astype(mx.bfloat16))
    raw = mx.array(rng.standard_normal((1, W, D)).astype(np.float32)).astype(mx.bfloat16)
    widx = mx.array(np.tile(np.arange(W, dtype=np.int32), (1, S, 1)))
    ci = rng.integers(0, T, (1, S, K)).astype(np.int32)
    ci[0, :, -3:] = -1  # unused slots
    cidx = mx.array(ci)
    q = mx.array(rng.standard_normal((1, S, H, D)).astype(np.float32)).astype(mx.bfloat16)
    sink = mx.array(rng.standard_normal((H,)).astype(np.float32))
    kv_full, ix_full = append_compressed_rows(raw, widx, cidx, pool, T, Shared(), gather=False)
    kv_g, ix_g = append_compressed_rows(raw, widx, cidx, pool, T, Shared(), gather=True)
    assert kv_g.shape[1] == W + S * K < kv_full.shape[1]
    a = sparse_attention(q, kv_full, sink, ix_full, D ** -0.5)
    b = sparse_attention(q, kv_g, sink, ix_g, D ** -0.5)
    assert np.array_equal(np.array(a), np.array(b))


def test_shared_view_memoizes_per_forward_and_tracks_appends():
    pool = QuantGrowBuffer()
    pool.append(mx.ones((1, 10, 64), mx.bfloat16))
    sh = Shared()
    v1 = sh.view(pool, 10)
    assert sh.view(pool, 10) is v1
    pool.append(mx.ones((1, 2, 64), mx.bfloat16))
    assert sh.view(pool, 12) is not v1 and sh.view(pool, 12).shape[1] == 12
