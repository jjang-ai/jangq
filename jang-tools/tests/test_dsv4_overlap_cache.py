import numpy as np
import pytest
import inspect
from types import SimpleNamespace


mx = pytest.importorskip("mlx.core")


def _tokens(start: int, length: int, width: int = 4):
    data = np.arange(start * width, (start + length) * width, dtype=np.float32)
    return mx.array(data.reshape(1, length, width))


def _np(x):
    return np.array(x, copy=False)


def test_dsv4_long_prefill_materializes_each_decoder_layer(monkeypatch):
    from jang_tools.dsv4 import mlx_model

    monkeypatch.delenv("DSV4_LAYERWISE_PREFILL", raising=False)
    monkeypatch.delenv("DSV4_LAYERWISE_PREFILL_MIN_TOKENS", raising=False)
    monkeypatch.setenv("DSV4_LAYERWISE_PREFILL_AUTO_TOKENS", "24576")
    monkeypatch.setenv("DSV4_ATTN_SUBCHUNK", "512")

    assert not mlx_model._layerwise_prefill_materialization_enabled(
        SimpleNamespace(shape=(1, 255))
    )
    assert not mlx_model._layerwise_prefill_materialization_enabled(
        SimpleNamespace(shape=(1, 256))
    )
    assert not mlx_model._layerwise_prefill_materialization_enabled(
        SimpleNamespace(shape=(1, 256)), final_context_tokens=24576
    )
    assert mlx_model._layerwise_prefill_materialization_enabled(
        SimpleNamespace(shape=(1, 256)), final_context_tokens=24577
    )
    monkeypatch.setenv("DSV4_LAYERWISE_PREFILL", "1")
    assert mlx_model._layerwise_prefill_materialization_enabled(
        SimpleNamespace(shape=(1, 256))
    )

    monkeypatch.setenv("DSV4_LAYERWISE_PREFILL", "0")
    assert not mlx_model._layerwise_prefill_materialization_enabled(
        SimpleNamespace(shape=(1, 2048))
    )

    source = inspect.getsource(mlx_model.DeepseekV4Model.__call__)
    loop = source[source.index("for _li, (layer, c) in enumerate"):source.index("h = self._hc_head_reduce")]
    assert "if layerwise_prefill:" in loop
    assert "mx.eval(h)" in loop


def test_dsv4_overlap_cache_prefill_keeps_last_full_window():
    from jang_tools.dsv4.mlx_model import DeepseekV4Cache

    cache = DeepseekV4Cache(sliding_window=128, compress_ratio=4)
    kv = _tokens(0, 8)
    gate = kv + 1000

    rows, gate_rows, pool_base = cache.accumulate_overlap_windows(
        kv, gate, "compressor_state", ratio=4, start_pos=0, head_dim=2
    )

    assert pool_base == 0
    assert rows.shape == (1, 2, 8, 2)
    assert gate_rows.shape == (1, 2, 8, 2)
    np.testing.assert_array_equal(_np(rows[:, 0, :4]), np.zeros((1, 4, 2)))
    np.testing.assert_array_equal(_np(rows[:, 0, 4:]), _np(kv[:, :4, 2:]))
    np.testing.assert_array_equal(_np(rows[:, 1, :4]), _np(kv[:, :4, :2]))
    np.testing.assert_array_equal(_np(rows[:, 1, 4:]), _np(kv[:, 4:8, 2:]))
    np.testing.assert_array_equal(_np(cache.compressor_state["buffer_kv"]), _np(kv[:, 4:8]))
    np.testing.assert_array_equal(_np(cache.compressor_state["buffer_gate"]), _np(gate[:, 4:8]))


def test_dsv4_overlap_cache_decode_uses_previous_window_for_next_pool_row():
    from jang_tools.dsv4.mlx_model import DeepseekV4Cache

    cache = DeepseekV4Cache(sliding_window=128, compress_ratio=4)
    prefill_kv = _tokens(0, 8)
    prefill_gate = prefill_kv + 1000
    cache.accumulate_overlap_windows(
        prefill_kv, prefill_gate, "compressor_state", ratio=4, start_pos=0, head_dim=2
    )

    for pos in (8, 9, 10):
        rows, _, _ = cache.accumulate_overlap_windows(
            _tokens(pos, 1),
            _tokens(pos, 1) + 1000,
            "compressor_state",
            ratio=4,
            start_pos=pos,
            head_dim=2,
        )
        assert rows.shape == (1, 0, 8, 2)

    rows, gate_rows, pool_base = cache.accumulate_overlap_windows(
        _tokens(11, 1),
        _tokens(11, 1) + 1000,
        "compressor_state",
        ratio=4,
        start_pos=11,
        head_dim=2,
    )

    current = _tokens(8, 4)
    assert pool_base == 8
    assert rows.shape == (1, 1, 8, 2)
    assert gate_rows.shape == (1, 1, 8, 2)
    np.testing.assert_array_equal(_np(rows[:, 0, :4]), _np(prefill_kv[:, 4:8, :2]))
    np.testing.assert_array_equal(_np(rows[:, 0, 4:]), _np(current[:, :, 2:]))
    np.testing.assert_array_equal(_np(cache.compressor_state["buffer_kv"]), _np(current))


def test_dsv4_compressor_decode_appends_overlap_pool_row():
    from jang_tools.dsv4.mlx_model import Compressor, DeepseekV4Cache

    class IdentityRope:
        dims = 2
        inv_freq = mx.zeros((1,), dtype=mx.float32)

        def __call__(self, x, offset=0, inverse=False, positions=None):
            return x

    cfg = SimpleNamespace(hidden_size=4, qk_rope_head_dim=2, rms_norm_eps=1e-6)
    comp = Compressor(cfg, compress_ratio=4, head_dim=2)
    cache = DeepseekV4Cache(sliding_window=128, compress_ratio=4)
    rope = IdentityRope()

    pooled = comp(mx.ones((1, 8, 4)), rope, cache, start_pos=0)
    mx.eval(pooled)
    assert pooled.shape[1] == 2

    for pos in (8, 9, 10):
        pooled = comp(mx.ones((1, 1, 4)), rope, cache, start_pos=pos)
        mx.eval(pooled)
        assert pooled.shape[1] == 2

    pooled = comp(mx.ones((1, 1, 4)), rope, cache, start_pos=11)
    mx.eval(pooled)
    assert pooled.shape[1] == 3
    assert cache.compressor_state["buffer_kv"].shape[1] == 4
