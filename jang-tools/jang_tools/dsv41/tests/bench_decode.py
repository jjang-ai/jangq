"""Decode-path microbenchmarks at the bundle's per-layer dimensions (random weights, no bundle load).

  python -m jang_tools.dsv41.tests.bench_decode stack --layers 10 --experts 16   # full decode step
  python -m jang_tools.dsv41.tests.bench_decode moe   [--rows 1]                  # Block.moe per layer
  python -m jang_tools.dsv41.tests.bench_decode router [--rows 1]
  python -m jang_tools.dsv41.tests.bench_decode hc    [--rows 1]

Each A/B toggles one module flag in-process and interleaves the two sides in rounds, so both see the
same GPU clock / contention. Reports the median per round.
"""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx
import numpy as np

from jang_tools.dsv41 import model as M
from jang_tools.dsv41.tests import synth as S


def _ab(run, flags: dict, rounds: int, reps: int):
    """run() -> sync'ed work; flags {label: setter(bool)}. Interleaves off/on rounds and times every call.
    Returns {label: ((p10, median) off, (p10, median) on)} in ms. Another process shares this GPU, so the
    low percentile is the better estimate of the uncontended cost."""
    res = {}
    for label, setter in flags.items():
        t = {False: [], True: []}
        for r in range(rounds):
            for on in ((False, True) if r % 2 == 0 else (True, False)):
                setter(on)
                run()  # warm (kernel compile / cache)
                for _ in range(reps):
                    t0 = time.perf_counter()
                    run()
                    t[on].append((time.perf_counter() - t0) * 1e3)
        res[label] = tuple((float(np.percentile(t[k], 10)), statistics.median(t[k])) for k in (False, True))
        setter(True)
    return res


def _fmt(off, on):
    return (f"off p10 {off[0]:7.3f} med {off[1]:7.3f} | on p10 {on[0]:7.3f} med {on[1]:7.3f} ms "
            f"| p10 {off[0] / on[0]:.3f}x med {off[1] / on[1]:.3f}x")


def bench_stack(a):
    args = S.real_args(a.layers, a.experts)
    model = S.build(args)
    caches = S.new_caches(args)
    rng = np.random.default_rng(0)
    prompt = rng.integers(0, args.vocab_size, a.ctx).tolist()
    mx.eval(S.forward(model, prompt, caches, 0))
    state = {"pos": a.ctx, "tok": 1}

    def step():
        lg = S.forward(model, [state["tok"]], caches, state["pos"])
        state["tok"] = int(mx.argmax(lg[0]))
        state["pos"] += 1

    def set_async(on):
        M.LAYER_ASYNC = on

    flags = {"layer_async": set_async}
    for k in a.flags.split(","):
        if k and k in FLAG_SETTERS:
            flags[k] = FLAG_SETTERS[k]
    out = _ab(step, flags, a.rounds, a.reps)
    for k, (off, on) in out.items():
        print(f"stack L={a.layers} E={a.experts} ctx={a.ctx} {k:12s} {_fmt(off, on)}")


FLAG_SETTERS = {}


def _flag(name):
    def set_(on):
        setattr(M, name, on)
    return set_


FLAG_SETTERS.update(router_gemv=_flag("ROUTER_GEMV"), fused_moe=_flag("FUSED_MOE"), fused_hc=_flag("FUSED_HC"))


def _chain(a, op):
    """op(block, x [1,R,D] bf16) -> [1,R,D]; runs it over `--layers` distinct real-dimension blocks,
    `--cycles` times, as one serial graph (each layer's input depends on the previous output), so the
    weights stream from DRAM as in decode. Returns run() and the number of op calls per run."""
    args = S.real_args(a.layers, a.experts)
    model = S.build(args, expert_bits=lambda i: (a.gate_bits, 3))
    x0 = (mx.random.normal((1, a.rows, args.hidden_size)) * 0.8).astype(mx.bfloat16)
    mx.eval(x0)
    n = a.layers * a.cycles

    def run():
        x = x0
        for c in range(a.cycles):
            for blk in model.layers:
                x = mx.fast.rms_norm(op(blk, x), None, 1e-6)
        mx.eval(x)

    return run, n


def bench_moe(a):
    run, n = _chain(a, lambda blk, x: blk.moe(x))
    flags = {k: FLAG_SETTERS[k] for k in (a.flags or "fused_moe").split(",") if k in FLAG_SETTERS}
    out = _ab(run, flags, a.rounds, a.reps)
    for k, (off, on) in out.items():
        print(f"moe/layer rows={a.rows} gate_bits={a.gate_bits} E={a.experts} {k:12s} "
              f"{_fmt(tuple(v / n for v in off), tuple(v / n for v in on))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["stack", "moe"])
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--gate-bits", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--flags", default="")
    a = ap.parse_args()
    {"stack": bench_stack, "moe": bench_moe}[a.what](a)


if __name__ == "__main__":
    main()
