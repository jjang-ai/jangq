"""DSV4.1 suffix pruning, KV-source trim (model.KV_SOURCE_TRIM) == the full trim layer, bit for bit.

Small synthetic model (synth.tiny_args: 6 layers, window 16, KV sources 1 and 3, candidates from 3, index layers 1 / 3 /
5): prune_from = 4, trim layer = 3, layer 4 keeps the last 31 rows. Prompts are prefilled the way the served lane does
(body in chunks with prompt_end = N-1, the last token alone); a chunk that ends before the rows layer 4 keeps runs
layer 3's cache side only and stops there. Every lane of every layer at the N-1 boundary, the last-token logits,
main_hidden and a greedy continuation must be bit-identical with the trim on and off. Controls: a one-token prompt
change is seen; a trim that also cuts a chunk the kept rows reach is caught.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import jang_tools.dsv41.model as M
from jang_tools.dsv41.tests import synth as S

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal kernels")


@pytest.fixture(scope="module")
def tiny():
    model = S.build(S.tiny_args())
    model.target_layer_ids = (2, 3, 4, 5)  # main_hidden at the inputs of the layers around the trim
    assert model.prune_from == 4 and model.trim_layer == 3 and model.keep_rows(4) == 31
    return model


def _bits(a):
    if a is None:
        return None
    a = mx.array(a)
    if a.dtype in (mx.bfloat16, mx.float16):
        a = a.view(mx.uint16)
    return np.array(a)


def _lanes(caches):
    out = []
    for c in caches:
        d = {"window": _bits(c.window), "pend_kv": _bits(c.pend_kv), "pend_score": _bits(c.pend_score)}
        for name in ("compress_kv", "index_k"):
            q = getattr(c, name)
            d[name] = None if q.wq is None else tuple(_bits(a[:, :q.n]) for a in (q.wq, q.scales, q.biases))
        out.append(d)
    return out


def _same(a, b):
    if isinstance(a, tuple):
        return isinstance(b, tuple) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and np.array_equal(a, b)


def _run(model, ids, step, start_ids=(), image_rows=None, steps=6):
    """Served-lane prefill of ids (after an optional prefix start_ids prefilled the same way), then greedy decode.
    Returns lanes at the N-1 boundary, last-token logits, main_hidden of every body chunk, greedy tokens + logits."""
    caches = S.new_caches(model.args)
    pos = 0
    hiddens = []
    for part in ([list(start_ids)] if start_ids else []) + [list(ids)]:
        end = pos + len(part)
        body = part[:-1]
        for a in range(0, len(body), step):
            piece = body[a:a + step]
            mask = None
            if image_rows is not None:
                m = np.zeros((1, len(piece)), bool)
                for r in image_rows:
                    if pos + a <= r < pos + a + len(piece):
                        m[0, r - pos - a] = True
                mask = mx.array(m)
            seg = M.Seg(0, len(piece), pos + a, caches)
            lg, mh = model(mx.array([piece], mx.int32), [seg], {}, image_mask=mask, return_hidden=True,
                           prompt_end=end - 1)
            mx.eval(lg, mh)
            hiddens.append(_bits(mh[:, -min(len(piece), 16):]))  # the rows main_hidden is exact for near the end
        pos = end - 1
        lanes = _lanes(caches)
        logits = S.forward(model, part[-1:], caches, pos)
        pos = end
    lg_last = _bits(logits.astype(mx.float32))
    toks, lgs = [], []
    for _ in range(steps):
        t = int(mx.argmax(logits[0]))
        toks.append(t)
        logits = S.forward(model, [t], caches, pos)
        lgs.append(_bits(logits.astype(mx.float32)))
        pos += 1
    return {"lanes": lanes, "logits": lg_last, "hidden_last": hiddens[-1], "toks": toks, "decode": lgs}


def _compare(r0, r1):
    why = []
    for li, (a, b) in enumerate(zip(r0["lanes"], r1["lanes"])):
        for k in a:
            if not _same(a[k], b[k]):
                why.append(f"L{li}.{k}")
    if not _same(r0["logits"], r1["logits"]):
        why.append("logits")
    if not _same(r0["hidden_last"], r1["hidden_last"]):
        why.append("main_hidden")
    if r0["toks"] != r1["toks"] or not all(_same(x, y) for x, y in zip(r0["decode"], r1["decode"])):
        why.append("decode")
    return why


def _count_trims(monkeypatch):
    n = [0]
    orig = M.Block.source_rows

    def counted(self, *a, **k):
        n[0] += 1
        return orig(self, *a, **k)

    monkeypatch.setattr(M.Block, "source_rows", counted)
    return n


IDS = [int(x) for x in np.random.default_rng(3).integers(5, 1000, 400)]

CASES = [  # (prompt length, chunk) -- none a multiple of the window / chunk
    (97, 37), (161, 17), (200, 64), (301, 128), (133, 9), (75, 61), (47, 40),
]


@pytest.mark.parametrize("n,step", CASES)
def test_trim_is_bit_identical(tiny, monkeypatch, n, step):
    ids = IDS[:n]
    monkeypatch.setattr(M, "KV_SOURCE_TRIM", False)
    r0 = _run(tiny, ids, step)
    monkeypatch.setattr(M, "KV_SOURCE_TRIM", True)
    trims = _count_trims(monkeypatch)
    r1 = _run(tiny, ids, step)
    # chunks wholly before the rows layer 4 keeps (and > 8 rows) take the trim
    body = n - 1
    expect = sum(1 for a in range(0, body, step) if min(a + step, body) <= body - 31 and min(step, body - a) > 8)
    assert trims[0] == expect
    assert _compare(r0, r1) == []


def test_partial_prefix_hit_is_bit_identical(tiny, monkeypatch):
    """A restored prefix (same first 123 tokens, then a different suffix) continued with the trim on == off, and ==
    the same tokens prefilled cold."""
    prefix, suffix = IDS[:123], IDS[200:331]
    monkeypatch.setattr(M, "KV_SOURCE_TRIM", False)
    r0 = _run(tiny, suffix, 29, start_ids=prefix)
    monkeypatch.setattr(M, "KV_SOURCE_TRIM", True)
    trims = _count_trims(monkeypatch)
    r1 = _run(tiny, suffix, 29, start_ids=prefix)
    assert trims[0] > 0
    assert _compare(r0, r1) == []


def test_image_rows_in_a_trimmed_chunk(tiny, monkeypatch):
    """image_mask (bias_vl routing) on rows of trimmed and untrimmed chunks: same result."""
    ids = IDS[:180]
    rows = [3, 4, 5, 40, 41, 170]
    monkeypatch.setattr(M, "KV_SOURCE_TRIM", False)
    r0 = _run(tiny, ids, 50, image_rows=rows)
    monkeypatch.setattr(M, "KV_SOURCE_TRIM", True)
    trims = _count_trims(monkeypatch)
    r1 = _run(tiny, ids, 50, image_rows=rows)
    assert trims[0] == 2
    assert _compare(r0, r1) == []


def test_control_one_token_change_is_seen(tiny, monkeypatch):
    """The comparison is not vacuous: one prompt token changed inside a trimmed chunk changes the result."""
    monkeypatch.setattr(M, "KV_SOURCE_TRIM", True)
    ids = IDS[:200]
    alt = list(ids)
    alt[20] = (alt[20] + 1) % 1000
    r0, r1 = _run(tiny, ids, 64), _run(tiny, alt, 64)
    assert "logits" in _compare(r0, r1)
    assert r0["logits"] is not None and r0["logits"].size > 0


def test_control_trim_reaching_kept_rows_is_caught(tiny, monkeypatch):
    """A trim that also cut a chunk whose rows layer 4 reads (keep_rows made too small) must fail the comparison."""
    ids = IDS[:200]
    monkeypatch.setattr(M, "KV_SOURCE_TRIM", False)
    r0 = _run(tiny, ids, 64)
    monkeypatch.setattr(M, "KV_SOURCE_TRIM", True)
    orig = M.Model.keep_rows
    monkeypatch.setattr(M.Model, "keep_rows", lambda self, i: 1 if i == self.prune_from else orig(self, i))
    r1 = _run(tiny, ids, 64)
    assert _compare(r0, r1) != []


def test_disabled_without_prompt_end(tiny, monkeypatch):
    """No prompt_end (plain forward / verify / capture): the trim never runs."""
    monkeypatch.setattr(M, "KV_SOURCE_TRIM", True)
    trims = _count_trims(monkeypatch)
    caches = S.new_caches(tiny.args)
    mx.eval(S.forward(tiny, IDS[:150], caches, 0))
    assert trims[0] == 0
