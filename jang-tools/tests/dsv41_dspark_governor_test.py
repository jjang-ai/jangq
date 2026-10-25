"""DSparkRunner adaptive governor under a synthetic timing model (CPU only, fake backbone/draft).

The fake backbone's greedy next token is a deterministic function of (token, position); a forward of
L tokens advances a fake clock by base + per_token * L, a draft by draft_cost. The draft's accuracy
is scripted per phase. Contracts:
- output == plain greedy, always;
- when speculation is slower than AR (low acceptance) the runner spends ~all tokens in AR steps and
  its total time is within a few percent of pure AR;
- when speculation is faster it is used and beats AR;
- no permanent latch: speculative probes keep happening in AR mode, and acceptance recovering mid-
  stream returns to speculation.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from jang_tools.dsv41.dspark import BoundedWindow, DSparkRunner, _Governor  # noqa: E402

V = 101


def nxt(tok, pos):
    return (tok * 13 + pos * 5 + 7) % V


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class Sess:
    def __init__(self):
        self.pos = 0
        self.caches = []

    def checkpoint(self):
        return self.pos

    def restore(self, ck):
        self.pos = ck


class FakeBackbone:
    def __init__(self, clock, base=0.020, per_token=0.001):
        self.clock, self.base, self.per_token = clock, base, per_token

    def new_session(self):
        return Sess()

    def forward(self, ids, sess, all_logits=False, return_hidden=False):
        self.clock.t += self.base + self.per_token * len(ids)
        rows = [nxt(int(t), sess.pos + i) for i, t in enumerate(ids)]
        sess.pos += len(ids)
        logits = mx.stack([mx.eye(V)[r] for r in rows])
        out = logits if all_logits else logits[-1]
        return (out, mx.zeros((len(ids), 2))) if return_hidden else out


class FakeDraft:
    """accuracy(position) -> fraction of the block's leading tokens drafted correctly there."""

    def __init__(self, clock, accuracy, bs=5, cost=0.006):
        self.clock, self.accuracy, self.bs, self.cost = clock, accuracy, bs, cost
        self.stages = [SimpleNamespace(attn=SimpleNamespace(win=8, window=BoundedWindow(8)))]
        self.calls = 0

    def reset_windows(self):
        pass

    def seed_windows(self, main_hidden, positions):
        pass

    def draft(self, seed_tok, start_pos):
        self.clock.t += self.cost
        good = int(round(self.accuracy(start_pos) * self.bs))
        self.calls += 1
        out, t, p = [int(seed_tok)], int(seed_tok), start_pos + 1
        for i in range(self.bs):
            n = nxt(t, p) if i < good else (nxt(t, p) + 1) % V
            out.append(n)
            t, p = n, p + 1
        return out, mx.ones((self.bs,))


def greedy(prompt, n):
    pos, t, out = len(prompt), nxt(prompt[-1], len(prompt) - 1), []
    for _ in range(n):
        out.append(t)
        t = nxt(t, pos)
        pos += 1
    return out


def run(accuracy, n=400, draft_cost=0.006, **kw):
    clock = Clock()
    m = FakeBackbone(clock)
    ds = FakeDraft(clock, accuracy, cost=draft_cost)
    r = DSparkRunner(m, ds, adaptive=True, clock=clock, **kw)
    prompt = [3, 1, 4, 1, 5]
    sess, pending, c = r.prefill(prompt)
    t0 = clock.t
    stats = {}
    out = list(r.decode_stream(sess, pending, c, n, stats=stats))
    assert out == greedy(prompt, n)
    ar_time = n * (m.base + m.per_token)
    return (clock.t - t0) / ar_time, stats, ds


def test_low_acceptance_stays_at_ar_speed():
    ratio, stats, ds = run(lambda i: 0.0)
    assert ratio < 1.06, ratio
    assert stats["ar_steps"] > 0.85 * stats["emitted"]
    assert stats["spec_steps"] >= 400 // 17 - 2  # probes keep happening: no permanent latch


def test_partial_acceptance_that_loses_to_ar_falls_back():
    # one accepted token per block: k>=1 every step (the old zero-streak latch never fired), but with
    # a slow draft the rejecting step (draft + verify + commit forward) costs more per token than AR
    ratio, stats, _ = run(lambda pos: 0.2, n=400, draft_cost=0.04)
    assert ratio < 1.08, ratio
    assert stats["ar_steps"] > 0.8 * stats["emitted"]


def test_high_acceptance_speculates_and_beats_ar():
    ratio, stats, _ = run(lambda i: 1.0)
    assert ratio < 0.45, ratio
    assert stats["spec_steps"] > 3 * stats["ar_steps"]
    assert stats["ar_steps"] >= 1  # AR keeps being re-measured


def test_recovers_when_acceptance_returns():
    # hopeless until position 200, perfect afterwards: back to speculation within one probe period
    ratio, stats, ds = run(lambda pos: 0.0 if pos < 200 else 1.0, n=600)
    assert ratio < 0.7, ratio
    assert stats["spec_steps"] > 60


def test_governor_has_no_terminal_state():
    g = _Governor(probe_every=4, ar_probe_every=8)
    seq = []
    for step in range(200):
        mode = g.choose()
        seq.append(mode)
        g.record(mode, 0.05 if mode == "spec" else 0.02, 1)  # spec always slower
    assert "spec" in seq[-20:]  # still probing at the end
    g2 = _Governor(probe_every=4, ar_probe_every=8)
    seq2 = []
    for step in range(200):
        mode = g2.choose()
        seq2.append(mode)
        g2.record(mode, 0.01 if mode == "spec" else 0.02, 1)  # spec always faster
    assert "ar" in seq2[-20:]  # AR still re-measured at the end
    assert seq2[-20:].count("spec") > 15


def test_markov_chain_matches_reference_loop():
    import numpy as np

    from jang_tools.dsv41.dspark import markov_chain

    rng = np.random.default_rng(3)
    vocab, rank, bs = 97, 16, 5
    emb = mx.array(rng.standard_normal((vocab, rank)).astype(np.float32)).astype(mx.bfloat16)
    head = mx.array(rng.standard_normal((vocab, rank)).astype(np.float32)).astype(mx.bfloat16)
    logits = mx.array(rng.standard_normal((bs, vocab)).astype(np.float32))
    # reference: the pre-hoist loop (head re-cast every step, host argmax)
    out, mes = [11], []
    for i in range(bs):
        me = emb[out[i]].astype(mx.float32)
        out.append(int(mx.argmax(logits[i] + me @ head.astype(mx.float32).T)))
        mes.append(me)
    got, me = markov_chain(logits, emb, head.astype(mx.float32).T, 11)
    assert [11] + got.tolist() == out
    assert np.array_equal(np.array(me), np.array(mx.stack(mes)))
