import mlx.core as mx
import pytest


def _pool_values(rows, dim, *, phase=0.0):
    values = mx.arange(rows * dim).reshape(1, rows, dim).astype(mx.float32)
    return mx.sin(values / 13.0 + phase).astype(mx.bfloat16)


def _tiny_ratio4_attention(*, index_topk=512):
    from jang_tools.dsv4.mlx_model import DeepseekV4Attention, ModelArgs

    args = ModelArgs(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        qk_rope_head_dim=4,
        q_lora_rank=8,
        o_lora_rank=4,
        o_groups=2,
        max_position_embeddings=2048,
        sliding_window=128,
        compress_ratios=[4],
        index_n_heads=2,
        index_head_dim=8,
        index_topk=index_topk,
    )
    return DeepseekV4Attention(args, layer_id=0)


def _full_attention_reference(
    q,
    local,
    pool,
    *,
    offset,
    window,
    ratio,
    scale,
    sinks,
    topk=None,
):
    from jang_tools.dsv4.mlx_model import (
        _dsv4_compressed_visibility,
        _dsv4_window_visibility,
    )

    batch, _heads, seq_len, _dim = q.shape
    full = mx.concatenate([local, pool[:, None]], axis=2)
    local_mask = _dsv4_window_visibility(
        batch, seq_len, offset, window, local.shape[2]
    )
    pool_mask = _dsv4_compressed_visibility(
        batch, seq_len, offset, pool.shape[1], ratio
    )
    if topk is not None:
        pool_rows = mx.arange(pool.shape[1])
        selected = (topk[..., None] == pool_rows[None, None, None]).any(axis=-2)
        pool_mask = pool_mask & selected[:, None]
    mask = mx.concatenate([local_mask, pool_mask], axis=-1)
    return mx.fast.scaled_dot_product_attention(
        q,
        full,
        full,
        scale=scale,
        mask=mask,
        sinks=sinks,
    )


def test_pool_quant_cache_appends_new_rows_without_requantizing_old_pool(monkeypatch):
    """Pool quant must encode only newly appended DSV4 pool rows."""
    import jang_tools.dsv4.pool_quant_cache as pq
    from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache

    monkeypatch.setattr(pq, "_POOL_BF16_MAX_BYTES", 0)
    quant_shapes = []
    original_quant = pq._quant_pool

    def recording_quant(pool, *args, **kwargs):
        quant_shapes.append(tuple(pool.shape))
        return original_quant(pool, *args, **kwargs)

    monkeypatch.setattr(pq, "_quant_pool", recording_quant)

    cache = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)
    first = mx.ones((1, 3, 16), dtype=mx.bfloat16)
    second = mx.ones((1, 1, 16), dtype=mx.bfloat16) * 2

    pool_a = cache.update_pool_view(first, "compressor_state")
    pool_b = cache.update_pool_view(second, "compressor_state")

    assert tuple(pool_a.shape) == (1, 4, 16)
    assert tuple(pool_b.shape) == (1, 4, 16)
    assert quant_shapes == [(1, 3, 16), (1, 1, 16)]


def test_pool_quant_cache_tail_compaction_is_lossless_and_never_dequantizes(monkeypatch):
    import jang_tools.dsv4.pool_quant_cache as pq
    from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache

    monkeypatch.setattr(pq, "_POOL_BF16_MAX_BYTES", 0)
    encoded = []
    original_quant = pq._quant_pool
    original_dequant = pq._dequant_pool
    dequant_count = 0

    def recording_quant(pool, *args, **kwargs):
        result = original_quant(pool, *args, **kwargs)
        encoded.append(result)
        return result

    def recording_dequant(qpool):
        nonlocal dequant_count
        dequant_count += 1
        return original_dequant(qpool)

    monkeypatch.setattr(pq, "_quant_pool", recording_quant)
    monkeypatch.setattr(pq, "_dequant_pool", recording_dequant)

    cache = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)
    for row in range(64):
        cache.update_pool_view(_pool_values(1, 16, phase=row / 17), "compressor_state")

    assert dequant_count == 0
    assert len(cache.compressor_state._pooled_q_segments) == 1
    compacted = cache.compressor_state._pooled_q_segments[0]
    expected_q = mx.concatenate([item[0] for item in encoded], axis=1)
    expected_scale = mx.concatenate([item[1] for item in encoded], axis=1)
    expected_min = mx.concatenate([item[2] for item in encoded], axis=1)
    mx.eval(compacted[0], compacted[1], compacted[2])
    assert mx.array_equal(compacted[0], expected_q).item()
    assert mx.array_equal(compacted[1], expected_scale).item()
    assert mx.array_equal(compacted[2], expected_min).item()


def test_pool_quant_live_view_is_tiled_and_owns_no_full_bf16_copy(monkeypatch):
    import jang_tools.dsv4.pool_quant_cache as pq
    from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache

    monkeypatch.setattr(pq, "_POOL_BF16_MAX_BYTES", 0)
    decoded_rows = []
    original_dequant = pq._dequant_pool

    def recording_dequant(qpool):
        decoded_rows.append(int(qpool[3][1]))
        return original_dequant(qpool)

    monkeypatch.setattr(pq, "_dequant_pool", recording_dequant)
    cache = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)
    original = _pool_values(150, 64)
    view = cache.update_pool_view(original, "compressor_state")
    encoded_bytes = cache.nbytes

    for _ in range(2):
        tiles = list(view.iter_dequantized_tiles(max_rows=32))
        mx.eval([tile for _start, tile in tiles])
        assert [start for start, _tile in tiles] == [0, 32, 64, 96, 128]
        assert all(int(tile.shape[1]) <= 32 for _start, tile in tiles)
        assert cache.nbytes == encoded_bytes

    assert max(decoded_rows) <= 32
    assert "_pooled_attention_view" not in vars(cache.compressor_state)
    assert cache.compressor_state._pooled_bf16 is None
    assert cache.nbytes < original.nbytes


def test_pool_quant_tiled_index_topk_matches_full_materialized_q8(monkeypatch):
    import jang_tools.dsv4.pool_quant_cache as pq
    from jang_tools.dsv4.mlx_model import _dsv4_tiled_index_topk
    from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache

    monkeypatch.setattr(pq, "_POOL_BF16_MAX_BYTES", 0)
    cache = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)
    view = cache.update_pool_view(_pool_values(137, 16), "indexer_state")
    q = mx.cos(
        mx.arange(1 * 3 * 4 * 16).reshape(1, 3, 4, 16).astype(mx.float32) / 11
    ).astype(mx.bfloat16)
    head_weights = (
        mx.arange(1 * 4 * 3).reshape(1, 4, 3).astype(mx.float32) + 1
    ) / 9

    actual = _dsv4_tiled_index_topk(
        q,
        head_weights,
        view,
        scale=16**-0.5,
        top_k=11,
        offset=4 * 137,
        ratio=4,
    )
    materialized = view.materialize()
    scores = q.astype(mx.float32) @ materialized[:, None].swapaxes(-1, -2).astype(
        mx.float32
    )
    scores = mx.maximum(scores, 0) * (16**-0.5)
    scores = (scores * head_weights.swapaxes(-1, -2)[..., None]).sum(axis=1)
    expected = mx.argpartition(-scores, kth=10, axis=-1)[..., :11]
    mx.eval(actual, expected)

    for row in range(actual.shape[1]):
        assert sorted(map(int, actual[0, row].tolist())) == sorted(
            map(int, expected[0, row].tolist())
        )


@pytest.mark.parametrize(
    ("seq_len", "offset", "use_topk"),
    [(1, 64, True), (5, 16, True), (5, 16, False)],
)
def test_pool_quant_tiled_attention_matches_q8_and_bf16_references(
    monkeypatch,
    seq_len,
    offset,
    use_topk,
):
    import jang_tools.dsv4.pool_quant_cache as pq
    from jang_tools.dsv4.mlx_model import _dsv4_tiled_pool_attention
    from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache

    monkeypatch.setattr(pq, "_POOL_BF16_MAX_BYTES", 0)
    batch, heads, dim, pool_rows, local_rows = 1, 3, 32, 41, 7
    original_pool = _pool_values(pool_rows, dim)
    cache = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)
    view = cache.update_pool_view(original_pool, "compressor_state")
    q = mx.cos(
        mx.arange(batch * heads * seq_len * dim)
        .reshape(batch, heads, seq_len, dim)
        .astype(mx.float32)
        / 17
    ).astype(mx.bfloat16)
    local = _pool_values(local_rows, dim, phase=0.37)[:, None]
    sinks = mx.array([0.1, -0.2, 0.3], dtype=mx.bfloat16)
    topk = None
    if use_topk:
        indices = [
            [(3 * query + step * 5) % pool_rows for step in range(9)]
            for query in range(seq_len)
        ]
        topk = mx.array([indices], dtype=mx.int32)

    actual = _dsv4_tiled_pool_attention(
        q,
        local,
        view,
        offset=offset,
        window=128,
        ratio=4,
        scale=dim**-0.5,
        sinks=sinks,
        topk=topk,
    )
    q8_reference = _full_attention_reference(
        q,
        local,
        view.materialize(),
        offset=offset,
        window=128,
        ratio=4,
        scale=dim**-0.5,
        sinks=sinks,
        topk=topk,
    )
    bf16_reference = _full_attention_reference(
        q,
        local,
        original_pool,
        offset=offset,
        window=128,
        ratio=4,
        scale=dim**-0.5,
        sinks=sinks,
        topk=topk,
    )
    mx.eval(actual, q8_reference, bf16_reference)

    assert mx.max(mx.abs(actual - q8_reference)).item() <= 0.0078125
    assert mx.max(mx.abs(actual - bf16_reference)).item() <= 0.03125


def test_pool_quant_csa_attention_dequantizes_only_selected_row_occurrences(
    monkeypatch,
):
    import jang_tools.dsv4.pool_quant_cache as pq
    from jang_tools.dsv4.mlx_model import _dsv4_tiled_pool_attention
    from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache

    monkeypatch.setattr(pq, "_POOL_BF16_MAX_BYTES", 0)
    cache = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)
    original_pool = _pool_values(193, 32)
    view = cache.update_pool_view(original_pool, "compressor_state")
    retained_before = cache.nbytes
    selected = mx.array(
        [[[0, 63, 64, 128, 192], [1, 62, 65, 127, 191]]],
        dtype=mx.int32,
    )
    q = _pool_values(3 * 2, 32).reshape(1, 3, 2, 32)
    local = _pool_values(7, 32)[:, None]
    sinks = mx.zeros((3,), dtype=mx.bfloat16)
    q8_reference = _full_attention_reference(
        q,
        local,
        view.materialize(),
        offset=1024,
        window=128,
        ratio=4,
        scale=32**-0.5,
        sinks=sinks,
        topk=selected,
    )
    mx.eval(q8_reference)
    selected_decode_rows = []
    original_selected_decode = pq._dequant_qpool_selected

    def recording_selected_decode(qpool, batch_indices, row_indices):
        selected_decode_rows.append(int(row_indices.size))
        return original_selected_decode(qpool, batch_indices, row_indices)

    def unexpected_full_decode(_qpool):
        raise AssertionError("CSA selected attention must not decode a full q8 segment")

    monkeypatch.setattr(pq, "_dequant_qpool_selected", recording_selected_decode)
    monkeypatch.setattr(pq, "_dequant_pool", unexpected_full_decode)
    output = _dsv4_tiled_pool_attention(
        q,
        local,
        view,
        offset=1024,
        window=128,
        ratio=4,
        scale=32**-0.5,
        sinks=sinks,
        topk=selected,
    )
    mx.eval(output)

    assert output.shape == q.shape
    assert sum(selected_decode_rows) == int(selected.size)
    assert mx.max(mx.abs(output - q8_reference)).item() <= 0.0078125
    assert cache.nbytes == retained_before
    assert "_pooled_attention_view" not in vars(cache.compressor_state)


def test_pool_quant_selected_attention_preserves_multi_query_tile_boundaries(
    monkeypatch,
):
    import jang_tools.dsv4.mlx_model as dsv4_model
    import jang_tools.dsv4.pool_quant_cache as pq
    from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache

    monkeypatch.setattr(pq, "_POOL_BF16_MAX_BYTES", 0)
    # Force one query per selected-attention tile so concatenation, absolute
    # query positions, local visibility, and compressed-row visibility cross
    # the boundary in this small deterministic regression.
    monkeypatch.setattr(dsv4_model, "_DSV4_POOL_TILE_TARGET_BYTES", 1_500)

    pool = _pool_values(257, 32, phase=0.19)
    cache = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)
    view = cache.update_pool_view(pool, "compressor_state")
    q = _pool_values(3 * 5, 32, phase=0.41).reshape(1, 3, 5, 32)
    local = _pool_values(11, 32, phase=0.73)[:, None]
    topk = mx.array(
        [[
            [0, 64, 128, 256, 31],
            [1, 63, 65, 256, 191],
            [2, 62, 66, 256, 190],
            [3, 61, 67, 256, 189],
            [4, 60, 68, 256, 188],
        ]],
        dtype=mx.int32,
    )
    sinks = mx.array([0.1, -0.2, 0.3], dtype=mx.bfloat16)

    assert dsv4_model._dsv4_selected_query_rows(q, topk) == 1
    actual = dsv4_model._dsv4_tiled_pool_attention(
        q,
        local,
        view,
        offset=1024,
        window=128,
        ratio=4,
        scale=32**-0.5,
        sinks=sinks,
        topk=topk,
    )
    expected = _full_attention_reference(
        q,
        local,
        view.materialize(),
        offset=1024,
        window=128,
        ratio=4,
        scale=32**-0.5,
        sinks=sinks,
        topk=topk,
    )
    mx.eval(actual, expected)

    assert actual.shape == expected.shape
    assert mx.max(mx.abs(actual - expected)).item() <= 0.0078125


def test_pool_quant_append_trim_export_import_and_repeated_reads(monkeypatch):
    import jang_tools.dsv4.pool_quant_cache as pq
    from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache

    monkeypatch.setattr(pq, "_POOL_BF16_MAX_BYTES", 0)
    cache = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)
    cache.update_pool_view(_pool_values(70, 32), "compressor_state")
    cache.update_pool_view(_pool_values(5, 32, phase=0.7), "compressor_state")
    cache.compressor_state.trim_pooled(3)
    view = cache.compressor_state.pool_view()
    assert view.shape == (1, 72, 32)

    before_reads = cache.nbytes
    for _ in range(4):
        tiles = list(view.iter_dequantized_tiles(max_rows=17))
        mx.eval([tile for _start, tile in tiles])
        assert cache.nbytes == before_reads

    storage = cache.storage_state
    restored = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)
    restored.storage_state = storage
    restored_view = restored.compressor_state.pool_view()
    assert restored_view.shape == view.shape
    assert restored.nbytes == cache.nbytes
    mx.eval(view.materialize(), restored_view.materialize())
    assert mx.array_equal(view.materialize(), restored_view.materialize()).item()

    delta = cache.compressor_state.export_pool_delta(11, 67)
    prefix = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)
    prefix.compressor_state.append_pool_delta(
        cache.compressor_state.export_pool_delta(0, 11)
    )
    prefix.compressor_state.append_pool_delta(delta)
    assert prefix.compressor_state.pool_view().shape == (1, 67, 32)
    expected = view.materialize()[:, :67]
    actual = prefix.compressor_state.materialize_pooled()
    mx.eval(expected, actual)
    assert mx.array_equal(expected, actual).item()


def test_pool_quant_compatibility_update_pool_materializes_without_retaining(monkeypatch):
    import jang_tools.dsv4.pool_quant_cache as pq
    from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache

    monkeypatch.setattr(pq, "_POOL_BF16_MAX_BYTES", 0)
    cache = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)
    materialized = cache.update_pool(_pool_values(9, 16), "compressor_state")
    mx.eval(materialized)
    assert tuple(materialized.shape) == (1, 9, 16)
    assert cache.nbytes < materialized.nbytes
    assert "_pooled_attention_view" not in vars(cache.compressor_state)


def test_ratio4_cold_512_prefill_advances_both_q8_pools_and_exports_delta(
    monkeypatch,
):
    """A production-sized cache block must populate both native DSV4 pools.

    The sparse selector is not needed below index_topk, but its compressor state
    is still part of every ratio-4 block delta and must stay row-aligned with
    the main pool from the first cold prefill.
    """
    import jang_tools.dsv4.pool_quant_cache as pq
    from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache

    monkeypatch.setattr(pq, "_POOL_BF16_MAX_BYTES", 0)
    attention = _tiny_ratio4_attention(index_topk=512)
    cache = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)
    values = mx.sin(
        mx.arange(512 * 16).reshape(1, 512, 16).astype(mx.float32) / 29
    ).astype(mx.bfloat16)

    output = attention(values, cache=cache)
    mx.eval(output)

    assert tuple(output.shape) == (1, 512, 16)
    assert tuple(cache.compressor_state.pool_view().shape) == (1, 128, 8)
    assert tuple(cache.indexer_state.pool_view().shape) == (1, 128, 8)
    assert cache.compressor_state._pooled_bf16 is None
    assert cache.indexer_state._pooled_bf16 is None

    delta = cache.export_block_delta(
        0,
        512,
        block_size=512,
        anchor_interval_blocks=1,
        force_anchor=True,
    )
    for branch in ("compressor_pool", "indexer_pool"):
        assert delta[branch]["storage"] == "q8"
        assert delta[branch]["start_row"] == 0
        assert delta[branch]["end_row"] == 128
        assert sum(segment[3][1] for segment in delta[branch]["segments"]) == 128


def test_ratio4_indexer_state_advances_before_topk_threshold(monkeypatch):
    """Below-threshold passes update state without paying sparse scoring cost."""
    import jang_tools.dsv4.mlx_model as dsv4_model
    import jang_tools.dsv4.pool_quant_cache as pq
    from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache

    monkeypatch.setattr(pq, "_POOL_BF16_MAX_BYTES", 0)
    selected_pool_rows = []
    original_select = dsv4_model.Indexer.select

    def recording_select(indexer, x, q_residual, position_rope, pooled, start_pos):
        selected_pool_rows.append(int(pooled.shape[1]))
        return original_select(
            indexer,
            x,
            q_residual,
            position_rope,
            pooled,
            start_pos,
        )

    monkeypatch.setattr(dsv4_model.Indexer, "select", recording_select)
    attention = _tiny_ratio4_attention(index_topk=2)
    cache = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)

    first = mx.sin(
        mx.arange(8 * 16).reshape(1, 8, 16).astype(mx.float32) / 11
    ).astype(mx.bfloat16)
    first_output = attention(first, cache=cache)
    mx.eval(first_output)

    assert selected_pool_rows == []
    assert tuple(cache.compressor_state.pool_view().shape) == (1, 2, 8)
    assert tuple(cache.indexer_state.pool_view().shape) == (1, 2, 8)

    second = mx.cos(
        mx.arange(4 * 16).reshape(1, 4, 16).astype(mx.float32) / 7
    ).astype(mx.bfloat16)
    second_output = attention(second, cache=cache)
    mx.eval(second_output)

    assert tuple(second_output.shape) == (1, 4, 16)
    assert selected_pool_rows == [3]
    assert tuple(cache.compressor_state.pool_view().shape) == (1, 3, 8)
    assert tuple(cache.indexer_state.pool_view().shape) == (1, 3, 8)


def test_ratio4_attention_fails_closed_on_native_pool_misalignment(monkeypatch):
    """A stale or malformed native cache must fail before selection/export."""
    import jang_tools.dsv4.pool_quant_cache as pq
    from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache

    monkeypatch.setattr(pq, "_POOL_BF16_MAX_BYTES", 0)
    attention = _tiny_ratio4_attention(index_topk=2)
    cache = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)
    cache.update_pool_view(_pool_values(1, 8), "compressor_state")
    values = mx.ones((1, 4, 16), dtype=mx.bfloat16)

    with pytest.raises(
        RuntimeError,
        match=r"compressor/indexer pool row misalignment: compressor=2 indexer=1",
    ):
        attention(values, cache=cache)


def test_pool_quant_deferred_delta_appends_compact_once_and_stay_lossless(
    monkeypatch,
):
    """Restore chains defer per-append compaction to one finalize per branch.

    Compacting per append re-concatenates the trailing slab for every record;
    a near-1M chain schedules that whole burst without a blocking eval and can
    OOM Metal beside ~95GB of weights (box stage8 crash, 2026-08-07).
    """
    import jang_tools.dsv4.pool_quant_cache as pq
    from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache, _qpool_rows

    monkeypatch.setattr(pq, "_POOL_BF16_MAX_BYTES", 0)
    source = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)
    total_rows = 64 * 40
    source.update_pool_view(_pool_values(total_rows, 32), "compressor_state")

    records = [
        source.compressor_state.export_pool_delta(start, start + 64)
        for start in range(0, total_rows, 64)
    ]

    compactions = []
    original = pq._StateProxy._compact_segments_to_slabs

    def _counting(self):
        compactions.append(True)
        return original(self)

    monkeypatch.setattr(pq._StateProxy, "_compact_segments_to_slabs", _counting)

    restored = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)
    proxy = restored.compressor_state
    for record in records:
        proxy.append_pool_delta(record, defer_compaction=True)
    assert not compactions, "deferred appends must not compact mid-chain"
    assert len(proxy._pooled_q_segments) >= len(records)

    proxy.finalize_deferred_pool_appends()
    assert len(compactions) == 1
    assert proxy._q_rows_deferred is None
    assert sum(_qpool_rows(s) for s in proxy._pooled_q_segments) == total_rows
    assert len(proxy._pooled_q_segments) <= max(
        1, -(-total_rows // pq._POOL_SLAB_MAX_ROWS) + 1
    )

    expected = source.compressor_state.materialize_pooled()
    actual = proxy.materialize_pooled()
    mx.eval(expected, actual)
    assert mx.array_equal(expected, actual).item()

    stale = dict(records[0])
    stale["start_row"] = int(stale["start_row"]) + 1
    stale["end_row"] = int(stale["end_row"]) + 1
    with pytest.raises(ValueError, match="not contiguous"):
        proxy.append_pool_delta(stale, defer_compaction=True)


def test_restore_anchor_from_deltas_finalizes_both_pool_branches(monkeypatch):
    """restore_anchor_from_deltas must leave no deferred state behind."""
    import jang_tools.dsv4.pool_quant_cache as pq

    monkeypatch.setattr(pq, "_POOL_BF16_MAX_BYTES", 0)
    finalized = []
    original = pq._StateProxy.finalize_deferred_pool_appends

    def _counting(self):
        finalized.append(self)
        return original(self)

    monkeypatch.setattr(
        pq._StateProxy, "finalize_deferred_pool_appends", _counting
    )

    from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache

    tokens = 8 * 256
    cache = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)
    rows = tokens // 4
    cache.update_pool_view(_pool_values(rows, 32), "compressor_state")
    cache.update_pool_view(_pool_values(rows, 16, phase=0.3), "indexer_state")
    # Advance the actual window to the checkpoint; meta_state is not writable
    # on every supported mlx-lm version.
    keys = _pool_values(tokens, 16)[:, None]
    cache.local.update_and_fetch(keys, keys)
    assert cache.local.offset == tokens
    cache._seen_tokens = tokens

    records = [
        cache.export_block_delta(
            start,
            start + 256,
            block_size=256,
            anchor_interval_blocks=8,
            force_anchor=(start + 256 == tokens),
        )
        for start in range(0, tokens, 256)
    ]

    restored = PoolQuantizedV4Cache.restore_anchor_from_deltas(
        records,
        target_tokens=tokens,
        block_size=256,
        anchor_interval_blocks=8,
    )
    assert restored.checkpoint_tokens == tokens
    assert restored.replayed_tokens == 0
    assert restored.cache.local.offset == tokens
    assert tuple(map(int, restored.cache.meta_state)) == (0, 128, tokens, 128)
    assert len(restored.cache.state[0]) == 2
    assert mx.array_equal(restored.cache.local.keys, keys[..., -128:, :]).item()
    proxies = {id(restored.cache.compressor_state), id(restored.cache.indexer_state)}
    assert {id(p) for p in finalized} >= proxies
    assert restored.cache.compressor_state._q_rows_deferred is None
    assert restored.cache.indexer_state._q_rows_deferred is None
