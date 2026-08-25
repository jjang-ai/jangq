"""Wall-clock cost of an MTP draft/verify cycle at depth 1 vs depth 3.

Acceptance alone does not decide best_depth — it gives tokens per cycle, but a
cycle at depth d costs d head forwards plus one target forward that must verify
d+1 positions. This times those pieces directly on the real bundle, then
combines them with the MEASURED acceptance curve to produce tok/s.

    cycle_time(d) = d * t_head + t_target(d+1 tokens)
    tok/s(d)      = tokens_per_cycle(d) / cycle_time(d)

Timing rules (docs/internal/_method): warmup discarded, >=5 repeats, median
reported, and the target is timed WITH a KV cache so it is a real decode step
rather than a prefill.

    python -m jang_tools.mtp_depth_speed <bundle> [--depths 1,2,3]
"""
import json
import statistics
import sys
import time
import types
from pathlib import Path

import mlx.core as mx


def _stub():
    pkg = types.ModuleType("vmlx_engine")
    pat = types.ModuleType("vmlx_engine.patches")
    mlm = types.ModuleType("vmlx_engine.patches.mlx_lm_mtp")
    mlm.is_mtp_active = lambda: True
    pkg.patches = pat
    pat.mlx_lm_mtp = mlm
    sys.modules.update({"vmlx_engine": pkg, "vmlx_engine.patches": pat,
                        "vmlx_engine.patches.mlx_lm_mtp": mlm})


def _median_time(fn, repeats=7):
    fn()                       # warmup — discard (kernel compile + page-in)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    bundle = Path(sys.argv[1])
    depths = [1, 2, 3]
    for i, a in enumerate(sys.argv):
        if a == "--depths":
            depths = [int(x) for x in sys.argv[i + 1].split(",")]

    _stub()
    sys.path.insert(0, str(Path(__file__).parent))
    from vmlx_mtp import qwen35_vl
    qwen35_vl.apply()
    from mlx_vlm import load
    from mlx_vlm.models.cache import KVCache

    model, proc = load(str(bundle))
    lm = model.language_model

    # realistic context so the target forward is a genuine decode step
    ids = proc.tokenizer.encode(proc.tokenizer.apply_chat_template(
        [{"role": "user", "content": "Explain how a B-tree index works."}],
        add_generation_prompt=True, tokenize=False, enable_thinking=True))
    lm._position_ids = None
    lm._rope_deltas = None
    out = lm(mx.array([ids]), return_hidden=True)
    logits, hidden = out
    mx.eval(logits, hidden)
    h_last = hidden[:, -1:, :]
    tok_last = mx.array([[int(mx.argmax(logits[0, -1]).item())]])

    def head_step():
        c = lm.make_mtp_cache()
        o = lm.mtp_forward(h_last, tok_last, c, return_hidden=True)
        mx.eval(o[0], o[1])

    t_head = _median_time(head_step)

    # Target forward verifying n tokens at once (n = depth + 1).
    #
    # 🚨 MUST run against a warm KV cache. Calling lm(full_sequence) with no
    # cache re-prefills the whole context every iteration, which measures
    # PREFILL, not a decode step — it overstated the per-cycle cost by ~4x here
    # and produced tok/s numbers a quarter of what the model actually does.
    from mlx_vlm.models.cache import KVCache, make_prompt_cache

    def target_n(n):
        def f():
            cache = make_prompt_cache(lm)
            lm._position_ids = None
            lm._rope_deltas = None
            lm(mx.array([ids]), cache=cache)          # prefill (not timed below)
            mx.eval([c.state for c in cache if hasattr(c, "state")])
            t0 = time.perf_counter()
            lm._position_ids = None
            lm._rope_deltas = None
            o = lm(mx.array([[tok_last.item()] * n]), cache=cache)
            mx.eval(o.logits if hasattr(o, "logits") else o)
            return time.perf_counter() - t0
        return f

    def _median_step(f, repeats=7):
        f()
        return statistics.median([f() for _ in range(repeats)])

    t_target = {n: _median_step(target_n(n)) for n in sorted({d + 1 for d in depths})}

    curve_p = Path(f"/Users/eric/models/Logs/q38v2/mtp-depth-{bundle.name}.json")
    tokens = {}
    if curve_p.is_file():
        for r in json.loads(curve_p.read_text())["rows"]:
            tokens[r["depth"]] = r["expected_tokens_per_cycle"]

    print(f"  bundle          : {bundle.name}")
    print(f"  head forward    : {t_head*1000:7.2f} ms  (median of 7, warmup dropped)")
    for n, t in sorted(t_target.items()):
        print(f"  target verify {n} : {t*1000:7.2f} ms")
    print()
    print(f"  {'depth':>5} {'tok/cycle':>10} {'cycle ms':>9} {'tok/s':>8} {'vs d1':>7}")
    base = None
    rows = []
    for d in depths:
        tc = tokens.get(d)
        if tc is None:
            print(f"  {d:>5}   (no acceptance curve — run mtp_depth_sweep first)")
            continue
        cyc = d * t_head + t_target[d + 1]
        tps = tc / cyc
        base = base or tps
        rows.append({"depth": d, "tokens_per_cycle": tc, "cycle_ms": cyc * 1000,
                     "tok_s": tps, "speedup_vs_depth1": tps / base})
        print(f"  {d:>5} {tc:>10.3f} {cyc*1000:>9.2f} {tps:>8.2f} {tps/base:>6.2f}x")

    if rows:
        best = max(rows, key=lambda r: r["tok_s"])
        print(f"\n  fastest depth = {best['depth']}  ({best['tok_s']:.2f} tok/s, "
              f"{best['speedup_vs_depth1']:.2f}x vs depth 1)")
        Path(f"/Users/eric/models/Logs/q38v2/mtp-speed-{bundle.name}.json").write_text(
            json.dumps({"bundle": bundle.name, "head_ms": t_head * 1000,
                        "target_ms": {str(k): v * 1000 for k, v in t_target.items()},
                        "rows": rows, "fastest_depth": best["depth"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
