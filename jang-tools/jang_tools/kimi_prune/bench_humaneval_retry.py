"""Retry HumanEval fails at multiple seeds -> compute pass@K."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from jang_tools.kimi_prune.bench_humaneval import (
    _load_humaneval,
    _extract_code,
    _build_source,
    _run_source,
    _solve,
)


def run(*, model_path: Path, source_report: Path, seeds: list[int],
        max_tokens: int, prefill_step_size: int, thinking: bool,
        timeout: float, temp: float, top_p: float, top_k: int,
        dataset: str = "plus"):
    from jang_tools.kimi_prune.runtime_patch import apply as _apply_patch
    _apply_patch(dry_run=False)
    from jang_tools.load_jangtq import load_jangtq_model

    src = json.loads(source_report.read_text())
    failed = [r for r in src["results"] if not r["passed"]]
    failed_ids = {r["task_id"] for r in failed}
    print(f"[retry] source report: {source_report}")
    print(f"[retry] original: {src['passes']}/{src['num']} = "
          f"{src['pass_at_1']*100:.2f}%")
    print(f"[retry] failures to retry: {len(failed)} Qs x {len(seeds)} seeds "
          f"= {len(failed) * len(seeds)} inferences")
    print(f"[retry] sampling: temp={temp} top_p={top_p} top_k={top_k} "
          f"max_tokens={max_tokens}")

    problems = _load_humaneval(164, dataset=dataset)
    fail_problems = [p for p in problems if p["task_id"] in failed_ids]

    print(f"[retry] loading model...", flush=True)
    model, tokenizer = load_jangtq_model(model_path)

    ts = int(time.time())
    txt_path = Path(f"/tmp/kimi_humaneval_{dataset}_retry_{ts}.txt")
    txt_fh = txt_path.open("w", encoding="utf-8")
    print(f"[retry] live log: {txt_path}")

    import random as _pyrand
    import numpy as _np
    try:
        import mlx.core as mx_
    except ImportError:
        mx_ = None

    retry_results: dict[str, list[dict]] = {t: [] for t in failed_ids}

    t_start = time.time()
    for seed in seeds:
        print(f"\n=== RETRY PASS seed={seed} ===", flush=True)
        _pyrand.seed(seed)
        _np.random.seed(seed)
        if mx_ is not None:
            try:
                mx_.random.seed(seed)
            except Exception as _seed_exc:
                print(f"  warning: mlx seed skipped: {_seed_exc}", flush=True)
        for i, p in enumerate(fail_problems):
            if any(r["passed"] for r in retry_results[p["task_id"]]):
                print(f"  [{i+1}/{len(fail_problems)}] skip {p['task_id']} "
                      f"(already passed in earlier seed)", flush=True)
                continue
            raw, elapsed = _solve(
                model, tokenizer, p,
                max_tokens=max_tokens,
                prefill_step_size=prefill_step_size,
                thinking=thinking,
                temp=temp, top_p=top_p, top_k=top_k,
            )
            code, full_def = _extract_code(raw, p["entry_point"])
            if code == "__NO_CODE_BLOCK__":
                source = "# NO_CODE_BLOCK\n"
                passed, reason, full_stderr = (
                    False, "no_code_block", "NO_CODE_BLOCK"
                )
            else:
                source = _build_source(
                    p["prompt"], code, full_def, p["test"], p["entry_point"]
                )
                passed, reason, full_stderr = _run_source(source, timeout=timeout)
            mark = "OK " if passed else "no "
            print(f"  [{i+1}/{len(fail_problems)}] {mark} {p['task_id']:<18} "
                  f"{elapsed:>5.1f}s  seed={seed}  {reason[:70]}",
                  flush=True)
            retry_results[p["task_id"]].append({
                "seed": seed, "passed": passed, "reason": reason,
                "elapsed": elapsed, "raw_model_output": raw,
                "extracted_code": code, "executed_source": source,
                "full_stderr": full_stderr,
            })
            txt_fh.write(
                f"\n=== {p['task_id']} seed={seed} "
                f"({'PASS' if passed else 'FAIL'}) ===\n"
                f"elapsed: {elapsed:.1f}s   reason: {reason}\n"
                f"--- extracted code (full_def={full_def}) ---\n{code}\n"
                f"--- full stderr ---\n{full_stderr}\n"
                f"--- end {p['task_id']} seed={seed} ---\n"
            )
            txt_fh.flush()

    k = 1 + len(seeds)
    orig_pass_count = src["passes"]
    retry_pass_count = sum(
        1 for t, attempts in retry_results.items()
        if any(a["passed"] for a in attempts)
    )
    pass_at_k = (orig_pass_count + retry_pass_count) / src["num"]
    total = time.time() - t_start
    print(f"\n=== RETRY SUMMARY ===")
    print(f"  seeds tried: {seeds}")
    print(f"  originally passed: {orig_pass_count}/{src['num']}")
    print(f"  additional passes from retry: {retry_pass_count}/{len(failed)}")
    print(f"  pass@{k}: {orig_pass_count + retry_pass_count}/{src['num']} "
          f"= {pass_at_k*100:.2f}%")
    print(f"  retry time: {total/60:.1f} min")

    report = {
        "model": str(model_path),
        "source_report": str(source_report),
        "seeds_retried": seeds,
        "k": k,
        "pass_at_k": pass_at_k,
        "passes_at_k": orig_pass_count + retry_pass_count,
        "num": src["num"],
        "original_pass_at_1": src["pass_at_1"],
        "original_passes": orig_pass_count,
        "retry_rescued": retry_pass_count,
        "max_tokens": max_tokens,
        "temp": temp, "top_p": top_p, "top_k": top_k,
        "retry_results": retry_results,
    }
    out_path = Path(f"/tmp/kimi_humaneval_{dataset}_retry_{ts}.json")
    out_path.write_text(json.dumps(report, indent=2))
    txt_fh.write(
        f"\n=== SUMMARY ===\n"
        f"pass@{k}: {orig_pass_count + retry_pass_count}/{src['num']} "
        f"({pass_at_k*100:.2f}%)\n"
    )
    txt_fh.close()
    print(f"\njson report: {out_path}")
    print(f"txt log:     {txt_path}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--source-report", required=True, type=Path,
                    help="existing bench_humaneval JSON report")
    ap.add_argument("--seeds", default="43,44,45,46",
                    help="comma-separated seeds for retry attempts")
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--prefill-step-size", type=int, default=16)
    ap.add_argument("--thinking", action="store_true", default=False)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--dataset", default="plus", choices=("base", "plus"))
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    return run(
        model_path=args.model,
        source_report=args.source_report,
        seeds=seeds,
        max_tokens=args.max_tokens,
        prefill_step_size=args.prefill_step_size,
        thinking=args.thinking,
        timeout=args.timeout,
        temp=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        dataset=args.dataset,
    )


if __name__ == "__main__":
    sys.exit(main())
