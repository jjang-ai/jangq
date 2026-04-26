"""Pass@k retry against the cloud API for prior pass@1 failures.

Mirrors `bench_humaneval_passk.py` (the local one) exactly — same k, same
sampling, same early-stop, same JSON output format — so the cloud
pass@5 number is apples-to-apples comparable against the local
pass@5 number.

Run:
  export MINIMAX_API_KEY=sk-cp-...
  python -m jang_tools.kimi_prune.bench_humaneval_api_passk \\
      --prior /tmp/minimax_cloud_MiniMax-M2_7-highspeed_plus_*.json \\
      --model MiniMax-M2.7-highspeed --k 5 --max-tokens 1200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from jang_tools.kimi_prune.bench_humaneval import (
    _build_source,
    _extract_code,
    _load_humaneval as _load_problems,
    _run_source,
)
from jang_tools.kimi_prune.bench_humaneval_api import _api_solve


def run(prior_path: Path, *, k: int, max_tokens: int, timeout: float,
        dataset: str, model: str, temp: float, top_p: float, top_k: int,
        api_key: str):
    prior = json.loads(Path(prior_path).read_text())
    fails = [r for r in prior["results"] if not r["passed"]]
    passed_ids = {r["task_id"] for r in prior["results"] if r["passed"]}
    if not fails:
        print("no failures; pass@1 was 100%.")
        return 0
    print(f"[passk-api] prior: {prior_path}")
    print(f"[passk-api] model: {model}")
    print(f"[passk-api] prior pass@1: {len(passed_ids)}/{prior['num']}  "
          f"failures to retry: {len(fails)}")
    print(f"[passk-api] sampling: k={k}  temp={temp}  top_p={top_p}  "
          f"top_k={top_k}  max_tokens={max_tokens}")

    problem_list = _load_problems(prior["num"], dataset=dataset)
    problems = {p["task_id"]: p for p in problem_list}

    ts = int(time.time())
    safe_model = model.replace(".", "_").replace("/", "_")
    txt_path = Path(f"/tmp/minimax_cloud_passk_{safe_model}_{dataset}_{ts}.txt")
    json_path = Path(f"/tmp/minimax_cloud_passk_{safe_model}_{dataset}_{ts}.json")
    txt_fh = txt_path.open("w", encoding="utf-8")
    print(f"[passk-api] log: {txt_path}")

    t_start = time.time()
    retry_results = []
    recovered = 0
    for i, fr in enumerate(fails):
        tid = fr["task_id"]
        p = problems[tid]
        samples = []
        passed_any = False
        first_idx = None
        for s in range(k):
            raw, elapsed = _api_solve(
                p, model=model, api_key=api_key,
                max_tokens=max_tokens, temp=temp, top_p=top_p, top_k=top_k,
                timeout_s=max(120.0, timeout * 6),
            )
            if raw.startswith("__API_ERROR__") or raw.startswith("__NETWORK_ERROR__") or raw.startswith("__EXHAUSTED_RETRIES__"):
                source = f"# {raw}\n"
                passed, reason, _ = False, f"api_error ({raw[:60]})", raw
                code, full_def = "__NO_CODE_BLOCK__", False
            else:
                code, full_def = _extract_code(raw, p["entry_point"])
                if code == "__NO_CODE_BLOCK__":
                    source = "# NO_CODE_BLOCK\n"
                    passed, reason, _ = False, "no_code_block", "NO_CODE_BLOCK"
                else:
                    source = _build_source(p["prompt"], code, full_def, p["test"], p["entry_point"])
                    passed, reason, _ = _run_source(source, timeout=timeout)
            samples.append({"idx": s, "elapsed": elapsed, "passed": passed,
                            "reason": reason, "raw_len": len(raw)})
            mark = "OK " if passed else "no "
            print(f"    sample {s+1}/{k} [{tid}] {mark} {elapsed:>5.1f}s  {reason[:60]}", flush=True)
            if passed and not passed_any:
                passed_any = True; first_idx = s; break
        if passed_any: recovered += 1
        running = len(passed_ids) + recovered
        mark = "OK " if passed_any else "no "
        print(f"  [{i+1:>2}/{len(fails)}] {mark} {tid:<18}  k={len(samples)}/{k}  "
              f"running pass@{k}={running}/{prior['num']}={running/prior['num']*100:.1f}%  "
              f"(first_pass={first_idx})", flush=True)
        retry_results.append({"task_id": tid, "recovered": passed_any,
                              "first_pass_sample_idx": first_idx,
                              "num_samples_tried": len(samples), "samples": samples})
        txt_fh.write(f"\n=== {tid} (recovered={passed_any}) ===\n"
                     f"reasons: {[s['reason'][:60] for s in samples]}\n"); txt_fh.flush()

    total = time.time() - t_start
    final_pass = len(passed_ids) + recovered
    final_pct = final_pass / prior["num"] * 100
    print()
    print(f"=== HumanEval (cloud {model}) pass@{k}: {final_pass}/{prior['num']} = {final_pct:.2f}% ===")
    print(f"    (orig pass@1: {len(passed_ids)} + recovered: {recovered})")
    print(f"    retry runtime: {total/60:.1f} min")

    report = {
        "model": model, "prior_report": str(prior_path), "dataset": dataset,
        "num_total": prior["num"],
        "orig_pass_at_1": len(passed_ids) / prior["num"], "orig_passes": len(passed_ids),
        "k": k, "temp": temp, "top_p": top_p, "top_k": top_k,
        "max_tokens": max_tokens, "recovered": recovered,
        f"pass_at_{k}": final_pass / prior["num"], f"pass_at_{k}_count": final_pass,
        "retry_elapsed_seconds": total, "retry_results": retry_results,
    }
    json_path.write_text(json.dumps(report, indent=2))
    txt_fh.write(f"\n=== SUMMARY ===\n"
                 f"orig pass@1: {len(passed_ids)}/{prior['num']}  "
                 f"({len(passed_ids)/prior['num']*100:.2f}%)\n"
                 f"recovered  : {recovered}/{len(fails)}\n"
                 f"pass@{k}    : {final_pass}/{prior['num']}  ({final_pct:.2f}%)\n")
    txt_fh.close()
    print(f"json: {json_path}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", required=True, type=Path)
    ap.add_argument("--model", default="MiniMax-M2.7-highspeed")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=1200,
                    help="match local pass@5 retry default")
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--dataset", default="plus", choices=("base", "plus"))
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--api-key", default=os.environ.get("MINIMAX_API_KEY", ""))
    a = ap.parse_args()
    if not a.api_key:
        print("error: MINIMAX_API_KEY not set", file=sys.stderr); return 2
    return run(a.prior, k=a.k, max_tokens=a.max_tokens, timeout=a.timeout,
               dataset=a.dataset, model=a.model, temp=a.temp, top_p=a.top_p,
               top_k=a.top_k, api_key=a.api_key)


if __name__ == "__main__":
    sys.exit(main())
