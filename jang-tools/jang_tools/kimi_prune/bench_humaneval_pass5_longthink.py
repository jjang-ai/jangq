"""Pass@k retry with LONG thinking budget — targeted at `no_code_block` failures.

Re-runs the 16 pass@5 failures where the quantized model rambled in
<think> past max_tokens=1200 and never emitted code. Uses
max_tokens=16000 to give the model room to finish reasoning.

Protocol identical to bench_humaneval_passk except:
  - Selects only problems where prior pass@5 samples showed ≥1 no_code_block
  - max_tokens defaults to 16000 instead of 1200
  - base_seed offset so new samples don't duplicate earlier ones
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


def _filter_ncb_unrecovered(prior_passk_json: Path) -> list[str]:
    p = json.loads(Path(prior_passk_json).read_text())
    tids = []
    for r in p.get("retry_results", []):
        if r.get("recovered_pass_at_k"):
            continue
        ncbs = sum(1 for s in r["samples"] if "no_code_block" in s.get("reason", ""))
        if ncbs > 0:
            tids.append(r["task_id"])
    return tids


def run(prior_passk: Path, *, k: int, max_tokens: int, prefill_step_size: int,
        thinking: bool, timeout: float, dataset: str, model_path: Path,
        temp: float, top_p: float, top_k: int, base_seed: int):
    ncb_tids = _filter_ncb_unrecovered(prior_passk)
    print(f"[passk-longthink] retrying {len(ncb_tids)} no_code_block-afflicted problems")
    for t in ncb_tids: print(f"    {t}")
    print(f"[passk-longthink] k={k}  max_tokens={max_tokens}  "
          f"temp={temp}  top_p={top_p}  top_k={top_k}  base_seed={base_seed}")

    from jang_tools.kimi_prune.runtime_patch import apply as _apply_patch
    _apply_patch(dry_run=False)
    from jang_tools.load_jangtq import load_jangtq_model
    model, tokenizer = load_jangtq_model(str(model_path))

    prior = json.loads(Path(prior_passk).read_text())
    num_total = prior["num_total"]
    already_passed = prior["orig_passes"] + prior["recovered"]

    all_problems = {p["task_id"]: p for p in _load_problems(num_total, dataset=dataset)}

    ts = int(time.time())
    out_json = Path(f"/tmp/mixmax_longthink_{dataset}_{ts}.json")
    out_txt = Path(f"/tmp/mixmax_longthink_{dataset}_{ts}.txt")
    txt_fh = out_txt.open("w", encoding="utf-8")
    print(f"[passk-longthink] log: {out_txt}")

    t_start = time.time()
    new_recovered = 0
    retry_results = []
    for i, tid in enumerate(ncb_tids):
        p = all_problems[tid]
        samples = []
        passed_any = False
        first_idx = None
        for s in range(k):
            seed_s = base_seed + 9999 + 1000 * (i + 1) + s
            _pyrand.seed(seed_s)
            try:
                import numpy as _np; _np.random.seed(seed_s)
            except Exception: pass
            try:
                import mlx.core as mx_; mx_.random.seed(seed_s)
            except Exception: pass
            raw, elapsed = _solve(
                model, tokenizer, p,
                max_tokens=max_tokens, prefill_step_size=prefill_step_size,
                thinking=thinking, temp=temp, top_p=top_p, top_k=top_k,
            )
            code, full_def = _extract_code(raw, p["entry_point"])
            if code == "__NO_CODE_BLOCK__":
                source = "# NO_CODE_BLOCK\n"
                passed, reason, full_err = False, "no_code_block", "NO_CODE_BLOCK"
            else:
                source = _build_source(p["prompt"], code, full_def, p["test"], p["entry_point"])
                passed, reason, full_err = _run_source(source, timeout=timeout)
            samples.append({"idx": s, "seed": seed_s, "elapsed": elapsed,
                            "passed": passed, "reason": reason, "raw_len": len(raw)})
            mark = "OK " if passed else "no "
            print(f"    sample {s+1}/{k} [{tid}] {mark} {elapsed:>6.1f}s  raw={len(raw)} {reason[:60]}",
                  flush=True)
            if passed and not passed_any:
                passed_any = True; first_idx = s; break
        if passed_any: new_recovered += 1
        running_total = already_passed + new_recovered
        mark = "OK " if passed_any else "no "
        print(f"  [{i+1:>2}/{len(ncb_tids)}] {mark} {tid:<18}  "
              f"running pass@{k}(longthink)={running_total}/{num_total}="
              f"{running_total / num_total * 100:.1f}%", flush=True)
        retry_results.append({"task_id": tid, "recovered": passed_any,
                              "first_idx": first_idx, "samples": samples})
        txt_fh.write(f"\n=== {tid} (recovered={passed_any}) samples: "
                     f"{[s['reason'][:30] for s in samples]}\n"); txt_fh.flush()

    total = time.time() - t_start
    final = already_passed + new_recovered
    print()
    print(f"=== HumanEval long-think pass@{k} ==="
          f"\n    recovered this round: {new_recovered}/{len(ncb_tids)}"
          f"\n    total pass: {final}/{num_total} = {final/num_total*100:.2f}%"
          f"\n    elapsed: {total/60:.1f} min")
    report = {
        "prior_passk": str(prior_passk), "model": str(model_path),
        "dataset": dataset, "num_total": num_total,
        "orig_recovered_before": already_passed,
        "new_recovered": new_recovered,
        "total_pass": final, "total_pct": final / num_total,
        "k": k, "max_tokens": max_tokens, "temp": temp, "top_p": top_p, "top_k": top_k,
        "retry_results": retry_results, "elapsed_seconds": total,
    }
    out_json.write_text(json.dumps(report, indent=2))
    txt_fh.write(f"\n=== SUMMARY ===\nnew_recovered: {new_recovered}/{len(ncb_tids)}\n"
                 f"total pass: {final}/{num_total}\n")
    txt_fh.close()
    print(f"json: {out_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior-passk", required=True, type=Path,
                    help="path to prior pass@5 .json report (mixmax_passk_plus_*.json)")
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=16000,
                    help="BIG budget — these problems need long thinking")
    ap.add_argument("--prefill-step-size", type=int, default=16)
    ap.add_argument("--thinking", action="store_true", default=False)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--dataset", default="plus", choices=("base", "plus"))
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--base-seed", type=int, default=42)
    a = ap.parse_args()
    return run(a.prior_passk, k=a.k, max_tokens=a.max_tokens,
               prefill_step_size=a.prefill_step_size, thinking=a.thinking,
               timeout=a.timeout, dataset=a.dataset, model_path=a.model,
               temp=a.temp, top_p=a.top_p, top_k=a.top_k, base_seed=a.base_seed)


if __name__ == "__main__":
    sys.exit(main() or 0)
