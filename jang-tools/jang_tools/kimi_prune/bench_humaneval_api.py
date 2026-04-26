"""HumanEval pass@1 against MiniMax's cloud API (Anthropic-compatible).

Mirrors bench_humaneval.py exactly — same prompt template, same code
extractor, same EvalPlus grading — so the official M2.7 cloud number is
apples-to-apples comparable against our JANGTQ2 local number.

Only difference: generation goes through
`https://api.minimax.io/anthropic/v1/messages` instead of local MLX.

Run:
  export MINIMAX_API_KEY=sk-cp-...
  python -m jang_tools.kimi_prune.bench_humaneval_api \\
      --model MiniMax-M2.7-highspeed --num 164 --max-tokens 800 \\
      --dataset plus
"""

from __future__ import annotations

import argparse
import httpx
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


API_BASE = "https://api.minimax.io/anthropic/v1/messages"


def _api_solve(problem, *, model, api_key, max_tokens, temp, top_p, top_k, timeout_s):
    """One call to the cloud API. Returns (raw_text, wall_seconds)."""
    user = (
        "Complete the following Python function. Output ONLY the completed "
        "function inside a single ```python fenced code block. No "
        "explanation before or after.\n\n"
        f"```python\n{problem['prompt']}```"
    )
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temp,
        "top_p": top_p,
        "messages": [{"role": "user", "content": user}],
    }
    if top_k and top_k > 0:
        body["top_k"] = top_k
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    t0 = time.time()
    for attempt in range(3):
        try:
            r = httpx.post(API_BASE, json=body, headers=headers, timeout=timeout_s)
            if r.status_code != 200:
                # rate limit / 5xx → back off and retry
                if r.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(2 ** attempt * 2)
                    continue
                return f"__API_ERROR__ {r.status_code}: {r.text[:300]}", time.time() - t0
            data = r.json()
            # Anthropic content is a list of blocks; we want only the 'text' block content.
            # Thinking blocks are discarded for code extraction (matches local bench which
            # strips <think>…</think> in _extract_code).
            blocks = data.get("content", [])
            text_parts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
            thinking_parts = [b.get("thinking", "") for b in blocks if b.get("type") == "thinking"]
            raw = "\n".join(text_parts)
            # If there was thinking, prepend as <think>…</think> so downstream
            # extractor's think-strip path is exercised (keeps parity with local).
            if thinking_parts:
                raw = "<think>" + "\n".join(thinking_parts) + "</think>\n" + raw
            return raw, time.time() - t0
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            if attempt < 2:
                time.sleep(2 ** attempt * 2)
                continue
            return f"__NETWORK_ERROR__ {e}", time.time() - t0
    return "__EXHAUSTED_RETRIES__", time.time() - t0


def run(model, *, num, max_tokens, timeout, dataset, temp, top_p, top_k,
        api_key: str):
    print(f"[humaneval-api] model: {model}")
    print(f"[humaneval-api] dataset: {dataset}")
    print(f"[humaneval-api] sampling: temp={temp} top_p={top_p} "
          f"top_k={top_k} max_tokens={max_tokens}")

    problems = _load_problems(num, dataset=dataset)
    print(f"[humaneval-api] {len(problems)} problems")

    ts = int(time.time())
    tag = "plus" if dataset == "plus" else "base"
    safe_model = model.replace(".", "_").replace("/", "_").replace(":", "_")
    txt_path = Path(f"/tmp/minimax_cloud_{safe_model}_{tag}_{ts}.txt")
    json_path = Path(f"/tmp/minimax_cloud_{safe_model}_{tag}_{ts}.json")
    txt_fh = txt_path.open("w", encoding="utf-8")
    print(f"[humaneval-api] live log: {txt_path}")

    t_start = time.time()
    results = []
    passes = 0
    for i, p in enumerate(problems):
        raw, elapsed = _api_solve(
            p, model=model, api_key=api_key,
            max_tokens=max_tokens, temp=temp, top_p=top_p, top_k=top_k,
            timeout_s=max(60.0, timeout * 2),
        )
        if raw.startswith("__API_ERROR__") or raw.startswith("__NETWORK_ERROR__") or raw.startswith("__EXHAUSTED_RETRIES__"):
            source = f"# {raw}\n"
            passed, reason, full_stderr = False, f"api_error ({raw[:60]})", raw
            code, full_def = "__NO_CODE_BLOCK__", False
        else:
            code, full_def = _extract_code(raw, p["entry_point"])
            if code == "__NO_CODE_BLOCK__":
                source = "# NO_CODE_BLOCK\n"
                passed, reason, full_stderr = False, "no_code_block", "NO_CODE_BLOCK"
            else:
                source = _build_source(p["prompt"], code, full_def, p["test"], p["entry_point"])
                passed, reason, full_stderr = _run_source(source, timeout=timeout)
        passes += int(passed)
        running = passes / (i + 1) * 100
        mark = "OK " if passed else "no "
        print(f"  [{i + 1:>3}/{len(problems)}] {mark} {p['task_id']:<18} "
              f"{elapsed:>5.1f}s   pass@1={running:.1f}%   {reason[:80]}",
              flush=True)
        results.append({
            "task_id": p["task_id"],
            "entry_point": p["entry_point"],
            "passed": passed,
            "reason": reason,
            "elapsed": elapsed,
            "raw_model_output": raw,
            "extracted_code": code,
            "extractor_full_def": full_def,
            "executed_source": source,
            "full_stderr": full_stderr,
        })
        txt_fh.write(
            f"\n=== {p['task_id']} ({'PASS' if passed else 'FAIL'}) ===\n"
            f"entry_point: {p['entry_point']}\n"
            f"elapsed: {elapsed:.1f}s   reason: {reason}\n"
            f"--- raw ---\n{raw[:4000]}\n"
            f"--- extracted ---\n{code[:2000]}\n"
            f"--- end {p['task_id']} ---\n"
        )
        txt_fh.flush()

    total = time.time() - t_start
    print()
    print(f"=== HumanEval (cloud {model}) pass@1: {passes}/{len(problems)} = "
          f"{passes/len(problems)*100:.2f}% ({total/60:.1f} min) ===")

    report = {
        "model": model,
        "api_base": API_BASE,
        "num": len(problems),
        "pass_at_1": passes / len(problems),
        "passes": passes,
        "max_tokens": max_tokens,
        "temp": temp, "top_p": top_p, "top_k": top_k,
        "elapsed_seconds": total,
        "results": results,
    }
    json_path.write_text(json.dumps(report, indent=2))
    txt_fh.write(
        f"\n=== SUMMARY ===\n"
        f"pass@1: {passes}/{len(problems)}  "
        f"({passes/len(problems)*100:.2f}%)\n"
    )
    txt_fh.close()
    print(f"json report: {json_path}")
    print(f"txt log:     {txt_path}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="MiniMax-M2.7-highspeed")
    ap.add_argument("--num", type=int, default=164)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--timeout", type=float, default=20.0,
                    help="subprocess grading timeout (matches local bench)")
    ap.add_argument("--dataset", default="plus", choices=("base", "plus"))
    ap.add_argument("--temp", type=float, default=1.0,
                    help="temperature — matches JANGTQ local bench default")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--api-key", default=os.environ.get("MINIMAX_API_KEY", ""))
    args = ap.parse_args()
    if not args.api_key:
        print("error: set MINIMAX_API_KEY env var or --api-key", file=sys.stderr)
        return 2
    return run(
        args.model, num=args.num, max_tokens=args.max_tokens,
        timeout=args.timeout, dataset=args.dataset,
        temp=args.temp, top_p=args.top_p, top_k=args.top_k,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    sys.exit(main())
