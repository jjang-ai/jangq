"""Block top-k (long index rows) selects exactly the values the direct argpartition selects."""
import mlx.core as mx
import pytest

import jang_tools.dsv41.model as M


@pytest.mark.parametrize("rows,length", [(7, 70000), (3, 140001), (2, 65536)])
def test_block_topk_matches_direct(rows, length):
    with mx.stream(mx.cpu):
        sc = mx.random.normal((max(rows, 9), length), key=mx.random.key(length))[:max(rows, 9)]
        rows = sc.shape[0]
        sc = mx.where(mx.arange(length)[None] >= (length - 5 + mx.arange(rows))[:, None], -mx.inf, sc)
        block = M.topk_indices(sc, 512)
        direct = mx.argpartition(-sc, kth=511, axis=-1)[:, :512]
        vb = mx.sort(mx.take_along_axis(sc, block, axis=-1), axis=-1)
        vd = mx.sort(mx.take_along_axis(sc, direct, axis=-1), axis=-1)
        assert block.dtype == mx.int32 and block.shape == (rows, 512)
        assert mx.array_equal(vb, vd).item()


def test_short_rows_keep_direct_path():
    with mx.stream(mx.cpu):
        sc = mx.random.normal((4, 1000), key=mx.random.key(1))
        assert M.topk_indices(sc, 1000).shape == (4, 1000)
        assert M.topk_indices(sc, 16).shape == (4, 16)


def test_decode_rows_keep_direct_path():
    """<= GATHER_MAX_QUERIES rows (decode / verify) never take the block path: identical indices."""
    with mx.stream(mx.cpu):
        sc = mx.random.normal((3, 70000), key=mx.random.key(3))
        direct = mx.argpartition(-sc, kth=511, axis=-1)[:, :512].astype(mx.int32)
        assert mx.array_equal(M.topk_indices(sc, 512), direct).item()
