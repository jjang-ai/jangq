"""HumanEval+ pass@5 retry on pass@1 failures.

For each pass@1 FAIL, generate 5 samples at T=0.8 with different seeds.
Problem passes if any of the 5 succeeds.

Final pass@5 = (pass@1 successes + pass@5 recoveries) / total
"""
import json, sys, os, time, subprocess, re
sys.stdout.reconfigure(line_buffering=True)

import mlx.core as mx
mx.set_memory_limit(int(os.environ.get("JANG_MEMORY_LIMIT_GB", "200")) * 1024**3)

from jang_tools.load_jangtq import load_jangtq_model
from jang_tools.dsv4.runtime import generate, GenerateOptions

# Reuse the harness's truncate_fim + run_test from the stable in-package
# location; fall back to /tmp for compatibility with older session layouts.
try:
    from jang_tools.dsv4.bench_humaneval import truncate_fim, run_test
except ImportError:
    sys.path.insert(0, "/tmp")
    from he_proper_bench import truncate_fim, run_test


def main():
    bundle = sys.argv[1]
    pass1_jsonl = sys.argv[2]
    out_jsonl = sys.argv[3] if len(sys.argv) > 3 else "/tmp/he_pass5.jsonl"
    n_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    T = float(sys.argv[5]) if len(sys.argv) > 5 else 0.8

    print(f"[bench] bundle={bundle}", flush=True)
    print(f"[bench] pass1_input={pass1_jsonl}", flush=True)
    print(f"[bench] n_samples={n_samples} T={T}", flush=True)

    # Load pass@1 results
    with open(pass1_jsonl) as f:
        pass1 = [json.loads(l) for l in f if "task_id" in l]
    pass1_pass = {r["task_id"] for r in pass1 if r.get("ok")}
    pass1_fail = [r["task_id"] for r in pass1 if not r.get("ok")]
    print(f"[bench] pass@1: {len(pass1_pass)}/{len(pass1)}; need pass@5 retry on {len(pass1_fail)}", flush=True)

    # Load problems
    from evalplus.data import get_human_eval_plus
    plus = get_human_eval_plus()

    t0 = time.time()
    model, tok = load_jangtq_model(bundle)
    print(f"[bench] load: {time.time()-t0:.1f}s", flush=True)

    # Run pass@5 on each fail
    n_recovered = 0
    fout = open(out_jsonl, "w")
    for i, tid in enumerate(pass1_fail):
        prob = plus[tid]
        msgs = [{"content": prob["prompt"]}]
        recovered = False
        per_sample = []
        t1 = time.time()
        for s in range(n_samples):
            opts = GenerateOptions(mode="fim", seed=42 + s, temperature=T)
            res = generate(model, tok, bundle, messages=msgs, opts=opts)
            completion = truncate_fim(res.raw)
            ok, err = run_test(prob, completion)
            per_sample.append({"seed": 42+s, "ok": ok, "err": err[:80] if not ok else ""})
            if ok:
                recovered = True
                break  # pass@5 = any-of-N
        dt = time.time() - t1
        if recovered: n_recovered += 1
        tag = "RECOVERED" if recovered else "STILL-FAIL"
        used = len(per_sample)
        print(f"  [{i+1:2d}/{len(pass1_fail)}] {tag:9s} {tid:<14s} {dt:6.1f}s "
              f"used_samples={used}/{n_samples} recovered_total={n_recovered}", flush=True)
        fout.write(json.dumps({"task_id": tid, "recovered": recovered,
                                "samples": per_sample, "dt": dt}) + "\n")
        fout.flush()
    fout.close()

    total = len(pass1)
    p1 = len(pass1_pass)
    p5 = p1 + n_recovered
    print(f"\n========= FINAL =========")
    print(f"  pass@1: {p1}/{total} = {100*p1/total:.1f}%")
    print(f"  pass@5: {p5}/{total} = {100*p5/total:.1f}%  (+{n_recovered} recovered)")


if __name__ == "__main__":
    main()
