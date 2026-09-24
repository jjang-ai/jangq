"""Pass@k retry for HumanEval: re-run prior failures with k samples each.

Reads a prior pass@1 json report, picks the failed problems, and
re-runs each with k independent samples at the model's advertised
generation config (temperature, top_p, top_k from its own
`generation_config.json`). A problem is counted as pass@k-recovered
if ANY of its k samples passes the test suite.

The final pass@k is: original pass@1 successes + retried recoveries.

Why retry only failures:
  - pass@1 greedy already showed the model's greedy answer.
  - Recoverable failures are those where the model has a correct
    answer in its distribution but greedy picked a bad one. Sampling
    with k=5 explores those. Running k=5 on the problems it already
    solved is wasted compute.

Run:
  python -m jang_tools.kimi_prune.bench_humaneval_passk \\
      --prior /tmp/kimi_humaneval_plus_1776993658.json \\
      --k 5 --max-tokens 1200 --dataset plus
"""

from __future__ import annotations

import argparse
import json
import random as _pyrand
import sys
import time
from pathlib import Path

from jang_tools.kimi_prune.bench_humaneval import (
    _build_source,
    _extract_code,
    _load_humaneval as _load_problems,
    _run_source,
    _solve,
)


def _read_gen_config(model_path: Path) -> dict:
    p = Path(model_path) / "generation_config.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError, UnicodeError):
        return {}


def run(prior_path: Path, *, k: int, max_tokens: int, prefill_step_size: int,
        thinking: bool, timeout: float, dataset: str, model_path: Path | None,
        temp: float | None, top_p: float | None, top_k: int | None,
        base_seed: int):
    prior = json.loads(Path(prior_path).read_text())
    model_path = Path(model_path or prior["model"])
    fails = [r for r in prior["results"] if not r["passed"]]
    passed_ids = {r["task_id"] for r in prior["results"] if r["passed"]}
    if not fails:
        print("no failures in prior report — pass@1 was 100%.")
        return 0
    print(f"[passk] prior: {prior_path}")
    print(f"[passk] model: {model_path}")
    print(f"[passk] prior pass@1: {len(passed_ids)}/{prior['num']}  "
          f"failures to retry: {len(fails)}")

    gen_cfg = _read_gen_config(model_path)
    if temp is None:
        temp = float(gen_cfg.get("temperature", 1.0))
    if top_p is None:
        top_p = float(gen_cfg.get("top_p", 1.0))
    if top_k is None:
        top_k = int(gen_cfg.get("top_k", 0))
    print(f"[passk] sampling: k={k}  temp={temp}  top_p={top_p}  "
          f"top_k={top_k}  max_tokens={max_tokens}  base_seed={base_seed}")

    from jang_tools.kimi_prune.runtime_patch import apply as _apply_patch
    _apply_patch(dry_run=False)
    from jang_tools.load_jangtq import load_jangtq_model
    model, tokenizer = load_jangtq_model(str(model_path))

    # Map task_id → full problem (we need prompt, test, entry_point)
    problem_list = _load_problems(prior["num"], dataset=dataset)
    problems = {p["task_id"]: p for p in problem_list}

    ts = int(time.time())
    tag = dataset
    txt_path = Path(f"/tmp/mixmax_passk_{tag}_{ts}.txt")
    json_path = Path(f"/tmp/mixmax_passk_{tag}_{ts}.json")
    txt_fh = txt_path.open("w", encoding="utf-8")
    print(f"[passk] live log: {txt_path}")

    t_start = time.time()
    retry_results = []
    recovered = 0
    for i, fr in enumerate(fails):
        tid = fr["task_id"]
        p = problems.get(tid)
        if p is None:
            print(f"  [{i+1:>3}/{len(fails)}] SKIP {tid}: not in dataset", flush=True)
            continue
        samples = []
        passed_any = False
        first_pass_idx = None
        for s in range(k):
            seed_s = base_seed + 1000 * (i + 1) + s
            # Seed PRNGs per-sample for reproducibility AND diversity
            _pyrand.seed(seed_s)
            try:
                import numpy as _np
                _np.random.seed(seed_s)
            except Exception as _np_seed_exc:
                print(f"  warning: numpy seed skipped: {_np_seed_exc}", flush=True)
            try:
                import mlx.core as mx_
                mx_.random.seed(seed_s)
            except Exception as _mx_seed_exc:
                print(f"  warning: mlx seed skipped: {_mx_seed_exc}", flush=True)
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
                passed, reason, full_stderr = (False, "no_code_block", "NO_CODE_BLOCK")
            else:
                source = _build_source(p["prompt"], code, full_def, p["test"], p["entry_point"])
                passed, reason, full_stderr = _run_source(source, timeout=timeout)
            samples.append({
                "sample_idx": s, "seed": seed_s, "elapsed": elapsed,
                "passed": passed, "reason": reason,
                "raw": raw, "code": code, "full_def": full_def,
            })
            if passed and not passed_any:
                passed_any = True
                first_pass_idx = s
            # Live tick per sample
            mark = "OK " if passed else "no "
            print(f"    sample {s+1}/{k} [{tid}] {mark} {elapsed:>5.1f}s  "
                  f"{reason[:60]}", flush=True)
            if passed_any:
                # Early stop — we only need one to count pass@k
                break
        if passed_any:
            recovered += 1
        running_passes = len(passed_ids) + recovered
        mark = "OK " if passed_any else "no "
        print(f"  [{i+1:>3}/{len(fails)}] {mark} {tid:<18}  "
              f"k={len(samples)}/{k}  "
              f"running pass@{k}={running_passes}/{prior['num']}="
              f"{running_passes / prior['num'] * 100:.1f}%  "
              f"(first_pass={first_pass_idx})", flush=True)
        retry_results.append({
            "task_id": tid,
            "entry_point": p["entry_point"],
            "recovered_pass_at_k": passed_any,
            "first_pass_sample_idx": first_pass_idx,
            "num_samples_tried": len(samples),
            "k": k,
            "samples": samples,
        })
        txt_fh.write(
            f"\n=== {tid} (recovered={passed_any}) ===\n"
            f"first_pass_sample_idx: {first_pass_idx}\n"
            f"samples_tried: {len(samples)}/{k}\n"
            f"reasons: {[s['reason'][:60] for s in samples]}\n"
            f"--- end {tid} ---\n"
        )
        txt_fh.flush()

    total = time.time() - t_start
    final_pass = len(passed_ids) + recovered
    final_pct = final_pass / prior["num"] * 100
    print()
    print(f"=== HumanEval pass@{k}: {final_pass}/{prior['num']} = {final_pct:.2f}% ===")
    print(f"    (orig pass@1: {len(passed_ids)} + recovered via pass@{k}: {recovered})")
    print(f"    retry runtime: {total/60:.1f} min over {len(fails)} failed problems")

    report = {
        "model": str(model_path),
        "prior_report": str(prior_path),
        "dataset": dataset,
        "num_total": prior["num"],
        "orig_pass_at_1": len(passed_ids) / prior["num"],
        "orig_passes": len(passed_ids),
        "k": k,
        "temp": temp, "top_p": top_p, "top_k": top_k,
        "base_seed": base_seed,
        "max_tokens": max_tokens,
        "recovered": recovered,
        f"pass_at_{k}": final_pass / prior["num"],
        f"pass_at_{k}_count": final_pass,
        "retry_elapsed_seconds": total,
        "retry_results": retry_results,
    }
    json_path.write_text(json.dumps(report, indent=2))
    txt_fh.write(
        f"\n=== SUMMARY ===\n"
        f"orig pass@1: {len(passed_ids)}/{prior['num']}  "
        f"({len(passed_ids)/prior['num']*100:.2f}%)\n"
        f"recovered  : {recovered}/{len(fails)}\n"
        f"pass@{k}    : {final_pass}/{prior['num']}  ({final_pct:.2f}%)\n"
    )
    txt_fh.close()
    print(f"json report: {json_path}")
    print(f"txt log:     {txt_path}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prior", required=True, type=Path,
                    help="path to prior pass@1 .json report")
    ap.add_argument("--model", type=Path, default=None,
                    help="override model path (default: from prior report)")
    ap.add_argument("--k", type=int, default=5, help="samples per problem")
    ap.add_argument("--max-tokens", type=int, default=1200,
                    help="per-sample max tokens (bumped from 800 since many "
                         "failures were no_code_block / ran out of tokens)")
    ap.add_argument("--prefill-step-size", type=int, default=16)
    ap.add_argument("--thinking", action="store_true", default=False)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--dataset", default="plus", choices=("base", "plus"))
    ap.add_argument("--temp", type=float, default=None,
                    help="sampling temperature (default: from model's generation_config.json)")
    ap.add_argument("--top-p", type=float, default=None,
                    help="(default: from model's generation_config.json)")
    ap.add_argument("--top-k", type=int, default=None,
                    help="(default: from model's generation_config.json)")
    ap.add_argument("--base-seed", type=int, default=42)
    args = ap.parse_args()
    return run(
        args.prior, k=args.k, max_tokens=args.max_tokens,
        prefill_step_size=args.prefill_step_size, thinking=args.thinking,
        timeout=args.timeout, dataset=args.dataset, model_path=args.model,
        temp=args.temp, top_p=args.top_p, top_k=args.top_k,
        base_seed=args.base_seed,
    )


if __name__ == "__main__":
    sys.exit(main())
