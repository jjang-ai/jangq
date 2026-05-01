"""Codesign-knob impact smoke for DSV4-Flash JANGTQ.

Runs the same 30-question MMLU smoke under each of these flag combinations
to measure the per-knob delta:

    baseline           DSV4_LONG_CTX=1 (everything else default)
    high_precision     DSV4_HIGH_PRECISION=1     (attn/shared bf16)
    low_bits           DSV4_LOW_BITS=1            (attn/shared 4-bit)
    no_long_ctx        DSV4_LONG_CTX=0           (regression check)

Bundle is reused across runs; only the env vars change. Note: most of
these flags affect CONVERT-time policy, not runtime — so for pre-built
bundles only DSV4_LONG_CTX swap matters at the runtime layer.
We log the env, prefill latency, and pass rate for diff comparison.

Usage:
    python -m jang_tools.dsv4.experiments.codesign_smoke \\
        --src /Volumes/EricsLLMDrive/jangq-ai/DeepSeek-V4-Flash-JANGTQ \\
        --qps 3
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path


CASES = [
    ("baseline_long_ctx_on",  {"DSV4_LONG_CTX": "1"}),
    ("baseline_long_ctx_off", {"DSV4_LONG_CTX": "0"}),
    # The high_precision / low_bits flags don't affect a pre-built bundle.
    # Listed for documentation; uncomment to use after a re-convert run.
    # ("high_precision",       {"DSV4_LONG_CTX": "1", "DSV4_HIGH_PRECISION": "1"}),
    # ("low_bits",             {"DSV4_LONG_CTX": "1", "DSV4_LOW_BITS": "1"}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--qps", type=int, default=3)
    ap.add_argument("--out", default="codesign_smoke_results.json")
    args = ap.parse_args()

    results = {}
    for label, env_overrides in CASES:
        print(f"\n=== {label}  env={env_overrides} ===", flush=True)
        env = os.environ.copy()
        env.update(env_overrides)
        cmd = [sys.executable, "-m", "jang_tools.eval.mmlu",
               "--src", args.src, "--mode", "no-reasoning",
               "--qps", str(args.qps), "--out", f"/tmp/{label}.json"]
        t0 = time.time()
        rc = subprocess.run(cmd, env=env).returncode
        dt = time.time() - t0
        if rc == 0 and Path(f"/tmp/{label}.json").exists():
            r = json.loads(Path(f"/tmp/{label}.json").read_text())
            score = r["passes"][0]["correct"] / max(r["passes"][0]["total"], 1) * 100
            results[label] = {"score_pct": score, "elapsed_s": dt, "env": env_overrides}
            print(f"  → {score:.1f}% in {dt:.1f}s", flush=True)
        else:
            results[label] = {"error": "harness failed", "rc": rc, "env": env_overrides}

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    print("\n=== SUMMARY ===")
    for label, r in results.items():
        if "score_pct" in r:
            print(f"  {label:<24s}  {r['score_pct']:5.1f}%  ({r['elapsed_s']:.0f}s)")
        else:
            print(f"  {label:<24s}  ERROR")


if __name__ == "__main__":
    main()
