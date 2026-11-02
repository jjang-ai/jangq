"""DSparkRunner.prefill with an image: spans stay whole, each chunk gets its own slice, plain prompts are unchanged."""
from dataclasses import dataclass

import mlx.core as mx

from jang_tools.dsv41.dspark import DSparkRunner


@dataclass
class Img:
    start: int
    types: list


class Attn:
    win = 4


class Stage:
    attn = type("A", (), {"win": 4, "window": type("W", (), {"buf": mx.zeros((1,))})()})()


class Draft:
    stages = [Stage()]
    bs = 3

    def __init__(self):
        self.seeded = []

    def reset_windows(self):
        self.seeded.clear()

    def seed_windows(self, mh, positions):
        self.seeded.extend(positions)


class Sess:
    def __init__(self, pos=0):
        self.pos = pos
        self.caches = []


class Target:
    """Records what each forward received; its signature matches DSV41.forward."""

    def __init__(self):
        self.calls = []

    def forward(self, ids, sess=None, all_logits=False, images=None, token_types=None, return_hidden=False,
                engram_lookahead=None, capture_verify=False, prompt_end=None):
        self.calls.append({"pos": sess.pos, "n": len(ids), "images": images, "types": token_types,
                           "lookahead": engram_lookahead})
        sess.pos += len(ids)
        return (mx.zeros((4,)), mx.zeros((len(ids), 2))) if return_hidden else mx.zeros((4,))


def _prefill(n, step, start=0, images=None, types=None):
    target = Target()
    sess, _, c = DSparkRunner(target, Draft()).prefill(list(range(n)), session=Sess(start), chunk_size=step,
                                                       images=images, token_types=types)
    return target.calls, sess, c


def _types(n, span):
    return [0 if span[0] <= i < span[1] else -1 for i in range(n)]


def test_plain_prompt_chunks_are_unchanged():
    calls, sess, c = _prefill(25, 10)
    assert [x["n"] for x in calls] == [10, 10, 4, 1]
    assert all(x["images"] is None and x["types"] is None for x in calls)
    assert calls[0]["lookahead"] is not None
    assert sess.pos == 25 and c == 24


def test_span_that_crosses_a_chunk_stops_before_the_image():
    calls, _, _ = _prefill(30, 10, images=[Img(8, [0] * 6)], types=_types(30, (8, 14)))
    assert [(x["pos"], x["n"]) for x in calls][:3] == [(0, 8), (8, 10), (18, 10)]
    assert calls[0]["images"] is None
    assert calls[1]["images"][0].start == 0 and len(calls[1]["images"]) == 1
    assert calls[1]["types"] == [0] * 6 + [-1] * 4


def test_span_longer_than_the_chunk_is_taken_whole():
    calls, _, _ = _prefill(40, 10, images=[Img(0, [0] * 25)], types=_types(40, (0, 25)))
    assert (calls[0]["pos"], calls[0]["n"]) == (0, 25)
    assert calls[0]["images"][0].start == 0


def test_every_token_type_is_delivered_once_and_in_order():
    types = _types(50, (20, 33))
    calls, _, _ = _prefill(50, 7, images=[Img(20, [0] * 13)], types=types)
    delivered = [t for x in calls[:-1] for t in x["types"]]
    assert delivered == types[:49]
    assert calls[-1]["n"] == 1 and calls[-1]["types"] is None


def test_restored_prefix_rebases_the_image_to_the_chunk():
    types = _types(60, (45, 50))
    target = Target()
    DSparkRunner(target, Draft()).prefill(list(range(20)), session=Sess(40), chunk_size=100,
                                          images=[Img(45, [0] * 5)], token_types=types)
    assert target.calls[0]["pos"] == 40 and target.calls[0]["images"][0].start == 5
    assert target.calls[0]["types"] == types[40:59]


def test_image_requests_skip_engram_lookahead():
    calls, _, _ = _prefill(30, 10, images=[Img(8, [0] * 6)], types=_types(30, (8, 14)))
    assert all(x["lookahead"] is None for x in calls)


def test_image_with_cuts_is_refused():
    target = Target()
    try:
        DSparkRunner(target, Draft()).prefill(list(range(30)), session=Sess(), chunk_size=10, cuts=[12],
                                              images=[Img(2, [0] * 3)], token_types=_types(30, (2, 5)))
    except ValueError as e:
        assert "cuts" in str(e)
    else:
        raise AssertionError("expected ValueError")
