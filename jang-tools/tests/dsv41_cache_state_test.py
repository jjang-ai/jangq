"""DeepseekV41Cache serialization (format v2) and prompt-snapshot clone. CPU only, no bundle.

Contracts:
- state/meta_state carry the 4-bit pools as stored (wq/scales/biases), so a restore is bit-exact and
  an entry is ~4-bit size (~0.9 KB/token at the real geometry -> ~0.9 GB per 1M-token entry);
- bf16 arrays travel as uint16 bit patterns, so a numpy round-trip (the block store) is bit-exact;
- pre-v2 entries are rejected, never re-quantized;
- clone_cache_list is bit-exact, compact (n rows, not the grow capacity), shares one _SeqState and
  is independent of later appends to the source.
"""

from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
import pytest


from jang_tools.dsv4_1.mlx_model import DeepseekV41Cache, _SeqState, clone_cache_list  # noqa: E402



@pytest.fixture(autouse=True)
def _cpu_only():
    """CPU only (another process owns the GPU); restore the previous default afterwards."""
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)

def _bits(a):
    a = mx.array(a)
    if a.dtype in (mx.bfloat16, mx.float16):
        a = a.view(mx.uint16)
    return np.array(a)


def _make(n_tokens=600, ratios=(0, 2, 1), seed=0):
    rng = np.random.default_rng(seed)
    seq = _SeqState()
    caches = [DeepseekV41Cache(seq, i, r) for i, r in enumerate(ratios)]
    seq.caches = caches
    for c in caches:
        c.window = mx.array(rng.standard_normal((1, 128, 512)).astype(np.float32)).astype(mx.bfloat16)
        if c._ratio:
            rows = n_tokens // c._ratio
            c.compress_kv.append(mx.array(rng.standard_normal((1, rows, 512)).astype(np.float32)).astype(mx.bfloat16))
            c.index_k.append(mx.array(rng.standard_normal((1, rows, 128)).astype(np.float32)).astype(mx.bfloat16))
        if c._ratio == 2 and n_tokens % 2:
            c.pend_kv = mx.array(rng.standard_normal((1, 1, 512)).astype(np.float32))
            c.pend_score = mx.array(rng.standard_normal((1, 1, 512)).astype(np.float32))
    seq.pos = n_tokens
    seq.history = rng.integers(0, 50000, n_tokens).astype(np.int64)
    mx.eval([c.state for c in caches])
    return caches


def _assert_same(a, b):
    assert a._idx == b._idx and a._ratio == b._ratio
    assert int(a.offset) == int(b.offset)
    assert np.array_equal(_bits(a.window), _bits(b.window))
    for qa, qb in ((a.compress_kv, b.compress_kv), (a.index_k, b.index_k)):
        assert qa.n == qb.n
        if qa.n:
            for x, y in ((qa.wq, qb.wq), (qa.scales, qb.scales), (qa.biases, qb.biases)):
                assert np.array_equal(_bits(x[:, : qa.n]), _bits(y[:, : qb.n]))
            assert np.array_equal(_bits(qa.view()), _bits(qb.view()))
    for x, y in ((a.pend_kv, b.pend_kv), (a.pend_score, b.pend_score)):
        assert (x is None) == (y is None)
        if x is not None:
            assert np.array_equal(np.array(x), np.array(y))


@pytest.mark.parametrize("n_tokens", [601, 1024])
@pytest.mark.parametrize("via_numpy", [False, True])
def test_state_round_trip_is_bit_exact(n_tokens, via_numpy):
    src = _make(n_tokens)
    restored = []
    for c in src:
        state, meta = c.state, c.meta_state
        assert all(a.dtype not in (mx.bfloat16, mx.float16) for a in state)  # bits only
        if via_numpy:  # what the block store / SSD path does (np.array, later mx.array)
            state = tuple(mx.array(np.array(a)) for a in state)
            meta = json.loads(json.dumps(meta))
        restored.append(DeepseekV41Cache.from_state(state, meta))
    for a, b in zip(src, restored):
        _assert_same(a, b)
    assert np.array_equal(restored[0]._seq.history, src[0]._seq.history)


def test_setter_order_independent_and_ratio_restored():
    src = _make(512)[2]
    c = DeepseekV41Cache(sliding_window=128, compress_ratio=None)  # engine reconstruct ctor
    c.state = src.state
    c.meta_state = src.meta_state
    _assert_same(src, c)
    assert c.compress_ratio == 1
    assert DeepseekV41Cache(ratio=0).compress_ratio is None


def test_pre_v2_entry_is_rejected_not_requantized():
    src = _make(512)[1]
    old_meta = json.dumps({"order": ["window", "ck", "ik"], "idx": 1, "ck_n": 256, "ik_n": 256, "pos": 512})
    old_state = (src.window, src.compress_kv.view(), src.index_k.view())
    c = DeepseekV41Cache()
    c.state = old_state
    with pytest.raises(ValueError, match="stale DeepseekV41Cache"):
        c.meta_state = old_meta
    with pytest.raises(ValueError):
        DeepseekV41Cache.from_state(old_state, old_meta)


def test_clone_is_exact_compact_shared_and_independent():
    src = _make(700)
    cl = clone_cache_list(src)
    assert len({id(c._seq) for c in cl}) == 1 and cl[0]._seq.caches == cl
    assert cl[0]._seq is not src[0]._seq
    for a, b in zip(src, cl):
        _assert_same(a, b)
        for q in (b.compress_kv, b.index_k):
            if q.n:
                assert q.wq.shape[1] == q.n  # compact: no grow-capacity rows
    before = [np.array(c.compress_kv.wq[:, : c.compress_kv.n]) for c in cl if c.compress_kv.n]
    # keep decoding on the source: appends, window replacement, pos/history growth
    for c in src:
        if c._ratio:
            c.compress_kv.append(mx.ones((1, 300, 512), mx.bfloat16))
        c.window = mx.zeros((1, 128, 512), mx.bfloat16)
    src[0]._seq.pos += 600
    src[0]._seq.history = np.concatenate([src[0]._seq.history, np.zeros(600, np.int64)])
    after = [np.array(c.compress_kv.wq[:, : c.compress_kv.n]) for c in cl if c.compress_kv.n]
    assert all(np.array_equal(x, y) for x, y in zip(before, after))
    assert int(cl[0].offset) == 700 and cl[0]._seq.history.size == 700
    assert not np.array_equal(_bits(cl[0].window), np.zeros((1, 128, 512), np.uint16))


def _state_bytes(c):
    return sum(int(a.nbytes) for a in c.state)


def test_serialized_entry_is_four_bit_size_per_token():
    """Real geometry: pool rows of 512 (kv) + 128 (index) dims at 4 bit, group 64, bf16 scale+bias.
    Per pool row: kv 256 + 32 B, index 64 + 8 B. The model's KV-source layers are 2, 8, 14 (ratio 2)
    and 20 (ratio 1): 5 ratio-2 layer equivalents -> 5 * 360 / 2 = 900 B/token."""
    n = 4096
    one = _make(n, ratios=(2,))[0]
    win = 128 * 512 * 2
    hist = n * 4
    pool_bytes = _state_bytes(one) - win - hist
    rows = n // 2
    assert pool_bytes == rows * (256 + 32) + rows * (64 + 8)
    per_token_model = pool_bytes / n * 5  # 3 ratio-2 layers + 1 ratio-1 layer (= 2 ratio-2 layers)
    assert per_token_model == pytest.approx(900.0)
    one_m_entry_gb = per_token_model * (1 << 20) / 1e9
    assert one_m_entry_gb < 1.0  # ~0.94 GB incl. nothing else; + ~5.2 MB windows + 4 MB history
    # the old format serialized the dequantized bf16 view: 3.6x the bytes
    old = rows * 512 * 2 + rows * 128 * 2
    assert old / pool_bytes == pytest.approx(3.56, rel=0.01)


def test_trim_is_refused():
    c = _make(512)[1]
    before = _state_bytes(c)
    assert c.is_trimmable() is False
    assert c.trim(10) == 0
    assert _state_bytes(c) == before and int(c.offset) == 512
