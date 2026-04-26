"""HumanEval pass@1 bench for Kimi K2.6 JANGTQ bundles.

For each of 164 problems: prompt the model, extract the function body
from the response, combine with the test suite, run in a subprocess
with a wall-clock timeout, report pass/fail.

Uses thinking=False by default (direct code, no CoT preamble).

Live log: per-problem raw response + pass/fail in
/tmp/kimi_humaneval_<ts>.txt.

Run:
  python -m jang_tools.kimi_prune.bench_humaneval \\
      --model /path/to/Kimi-K2.6-REAP-30-JANGTQ_1L \\
      --num 164 --max-tokens 800
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _load_humaneval(num, dataset: str = "base"):
    """Load HumanEval problems.

    dataset:
      - "base"  — `openai_humaneval` (original test suite, 164 problems)
      - "plus"  — `evalplus/humanevalplus` (same 164 prompts, MUCH harder
                   test cases from EvalPlus — catches "passes original
                   tests but misses edge cases")
    """
    from datasets import load_dataset
    if dataset == "plus":
        ds = load_dataset("evalplus/humanevalplus", split="test")
    elif dataset == "base":
        ds = load_dataset("openai_humaneval", split="test")
    else:
        raise ValueError(f"unknown dataset: {dataset!r}")
    return [ds[i] for i in range(min(num, len(ds)))]


def _extract_code(raw: str, entry_point: str):
    """Extract runnable Python from model output.

    Returns (code, is_full_def).
      - is_full_def=True  → `code` is a complete self-contained Python
        module (includes any helper function defs above/around the entry
        point). Caller uses it verbatim; original prompt is discarded.
      - is_full_def=False → `code` is body-only for the entry point.
        Caller indents it and appends after the prompt.

    Bug fixed 2026-04-22: earlier version sliced from `def {entry_point}`
    onward, dropping helper functions the prompt defined above the entry
    point (HumanEval/10, /32, /38, /50 failed with NameError because
    `is_palindrome`, `poly`, `encode_cyclic`, `encode_shift` got cut).
    """
    # Bug fix 2026-04-24: chat templates that prefill "<think>\n" at the
    # generation prompt boundary (MiniMax-M2, GLM-4.7, …) cause the model's
    # raw output to start MID-think with no opening "<think>" tag. The
    # original "<think>…</think>" pair regex never matched, so the broken
    # in-think attempts (with token-boundary glitches like
    # "```python一致:" or "```pythonfr") leaked into the extractor and
    # confused fence pairing — costing ~9 pts of pass@1 on MiniMax-Small.
    # Fix: if "</think>" appears with no opening "<think>" before it,
    # strip everything up to and including the first "</think>".
    if "</think>" in raw:
        idx_close = raw.index("</think>")
        idx_open = raw.find("<think>")
        if idx_open < 0 or idx_open > idx_close:
            raw = raw[idx_close + len("</think>"):]
    # Now the regular pair-strip handles models that emit complete
    # <think>…</think> blocks themselves.
    raw_stripped = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    if "<think>" in raw_stripped and "</think>" not in raw_stripped:
        raw_stripped = raw_stripped.split("<think>", 1)[0]

    # Proper markdown fence matching: opener must start a line AND be followed
    # by a newline (so inline mentions like "...inside a single ```python
    # fenced code block." don't get treated as openers and desync pairing).
    # Closer must be ``` on its own line (optionally with trailing whitespace).
    # Bug fix 2026-04-24: opener language-tag is now "anything until \n"
    # (`[^\n]*`) instead of `(?:python|py)?` — at 2-bit quant the model
    # sometimes emits token-boundary glitches like "```python一致:\n" or
    # "```pythonfr\n" where the \n is delayed by 1-3 BPE tokens. The
    # stricter regex rejected these as openers but accepted bare "```"
    # downstream, causing pairing to skip past the real answer.
    blocks = re.findall(
        r"(?:^|\n)```[^\n]*\n(.*?)\n[ \t]*```(?:\n|$)",
        raw_stripped, re.DOTALL,
    )
    # Fallback: if the stricter regex finds nothing (e.g. model forgot the
    # trailing newline before closing fence), accept a looser form.
    if not blocks:
        blocks = re.findall(
            r"(?:^|\n)```[^\n]*\n?(.*?)```",
            raw_stripped, re.DOTALL,
        )
    sig_pat = re.compile(
        rf"^\s*def\s+{re.escape(entry_point)}\s*\([^)]*\)\s*"
        rf"(?:->\s*[^:]+)?\s*:\s*$",
        re.MULTILINE,
    )
    code = None
    # Prefer a block that contains `def {entry_point}` (it's a self-
    # contained snippet). Scan from the LAST block backward so the final
    # "here's my answer" snippet wins over intermediate scratch blocks.
    for b in reversed(blocks):
        if sig_pat.search(b):
            return b.rstrip(), True
    # No block has the entry-point def — pick the last block that
    # *looks* like Python (starts with def/class/import/from).
    for b in reversed(blocks):
        s = b.lstrip()
        if s.startswith(("def ", "class ", "import ", "from ", "async def ")):
            code = b.rstrip()
            break
    if code is None:
        # No fenced block at all = model didn't emit code (e.g. ran out of
        # tokens still thinking). Return a clearly-invalid stub so callers
        # can flag it as NO_CODE_BLOCK instead of trying to parse prose.
        if not blocks:
            return "__NO_CODE_BLOCK__", False
        code = blocks[-1].rstrip()
    if sig_pat.search(code):
        return code, True
    # Body-only path: ensure top-level lines get one indent so they
    # attach correctly under the prompt's existing def + docstring.
    lines = code.splitlines()
    if lines and not lines[0].startswith((" ", "\t")):
        lines = ["    " + line if line.strip() else line for line in lines]
    return "\n".join(lines), False


def _build_source(prompt, code, is_full_def, test, entry_point):
    if is_full_def:
        src = code + "\n"
    else:
        src = prompt + code + "\n"
    return src + "\n" + test + "\n\n" + f"check({entry_point})\n"


def _run_source(source: str, timeout: float):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(source)
        tmp_path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        )
        full_err = (proc.stderr or "") + (("\n" + proc.stdout) if proc.stdout else "")
        if proc.returncode == 0:
            return True, "", full_err.strip()
        err_lines = full_err.strip().splitlines()
        short = err_lines[-1] if err_lines else f"exit {proc.returncode}"
        return False, short, full_err.strip()
    except subprocess.TimeoutExpired:
        return False, f"timeout > {timeout}s", f"TIMEOUT after {timeout}s"
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _solve(model, tokenizer, problem, *, max_tokens, prefill_step_size,
           thinking, temp, top_p, top_k):
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    user = (
        "Complete the following Python function. Output ONLY the completed "
        "function inside a single ```python fenced code block. No "
        "explanation before or after.\n\n"
        f"```python\n{problem['prompt']}```"
    )
    msgs = [{"role": "user", "content": user}]
    try:
        templated = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False,
            thinking=thinking,
        )
    except TypeError:
        templated = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False,
        )
    # temp=0 → pure greedy (ignores top_p/top_k per mlx_lm). Any temp>0
    # uses the provided top_p / top_k.
    sampler_kwargs = {"temp": temp}
    if temp > 0:
        if top_p is not None and top_p < 1.0:
            sampler_kwargs["top_p"] = top_p
        if top_k is not None and top_k > 0:
            sampler_kwargs["top_k"] = top_k
    t0 = time.time()
    raw = generate(
        model, tokenizer,
        prompt=templated,
        max_tokens=max_tokens,
        sampler=make_sampler(**sampler_kwargs),
        verbose=False,
        prefill_step_size=prefill_step_size,
    )
    return raw, time.time() - t0


def run(model_path, *, num, max_tokens, prefill_step_size, thinking,
        timeout, dataset: str = "base",
        temp: float = 0.0, top_p: float = 1.0, top_k: int = 0, seed: int = 42):
    from jang_tools.kimi_prune.runtime_patch import apply as _apply_patch
    _apply_patch(dry_run=False)
    from jang_tools.load_jangtq import load_jangtq_model
    print(f"[humaneval] model: {model_path}")
    print(f"[humaneval] dataset: {dataset}")
    model, tokenizer = load_jangtq_model(model_path)

    ts = int(time.time())
    tag = "plus" if dataset == "plus" else "base"
    txt_path = Path(f"/tmp/kimi_humaneval_{tag}_{ts}.txt")
    txt_fh = txt_path.open("w", encoding="utf-8")
    print(f"[humaneval] live log: {txt_path}")

    problems = _load_humaneval(num, dataset=dataset)
    print(f"[humaneval] {len(problems)} problems")
    print(f"[humaneval] sampling: temp={temp} top_p={top_p} top_k={top_k} "
          f"max_tokens={max_tokens} seed={seed}")
    # Reproducibility when temp>0: seed mlx, numpy, python random.
    import random as _pyrand
    import numpy as _np
    _pyrand.seed(seed)
    _np.random.seed(seed)
    try:
        import mlx.core as mx_  # local import; seed Metal PRNG
        mx_.random.seed(seed)
    except Exception:
        pass

    t_start = time.time()
    results = []
    passes = 0
    for i, p in enumerate(problems):
        raw, elapsed = _solve(
            model, tokenizer, p,
            max_tokens=max_tokens,
            prefill_step_size=prefill_step_size,
            thinking=thinking,
            temp=temp, top_p=top_p, top_k=top_k,
        )
        code, full_def = _extract_code(raw, p["entry_point"])
        if code == "__NO_CODE_BLOCK__":
            # Model didn't emit a code block (usually ran out of tokens
            # mid-reasoning). Report cleanly instead of executing prose.
            source = "# NO_CODE_BLOCK: model produced no ```python ... ``` fence\n"
            passed, reason, full_stderr = (
                False,
                "no_code_block (model ran out of tokens or formatted poorly)",
                "NO_CODE_BLOCK: extractor found zero fenced python blocks in raw output",
            )
        else:
            source = _build_source(
                p["prompt"], code, full_def, p["test"], p["entry_point"]
            )
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
            f"--- prompt ---\n{p['prompt']}\n"
            f"--- raw model output ---\n{raw}\n"
            f"--- extracted code (full_def={full_def}) ---\n{code}\n"
            f"--- executed source ---\n{source}\n"
            f"--- full stderr ---\n{full_stderr}\n"
            f"--- end {p['task_id']} ---\n"
        )
        txt_fh.flush()

    total = time.time() - t_start
    print()
    print(f"=== HumanEval pass@1: {passes}/{len(problems)} = "
          f"{passes/len(problems)*100:.2f}% ({total/60:.1f} min) ===")

    report = {
        "model": str(model_path),
        "num": len(problems),
        "pass_at_1": passes / len(problems),
        "passes": passes,
        "max_tokens": max_tokens,
        "thinking": thinking,
        "elapsed_seconds": total,
        "results": results,
    }
    out_path = Path(f"/tmp/kimi_humaneval_{tag}_{ts}.json")
    out_path.write_text(json.dumps(report, indent=2))
    txt_fh.write(f"\n=== SUMMARY ===\npass@1: {passes}/{len(problems)}  "
                 f"({passes/len(problems)*100:.2f}%)\n")
    txt_fh.close()
    print(f"json report: {out_path}")
    print(f"txt log:     {txt_path}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--num", type=int, default=164)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--prefill-step-size", type=int, default=16)
    ap.add_argument("--thinking", action="store_true", default=False)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--dataset", default="base", choices=("base", "plus"),
                    help="`base` = openai_humaneval (164q), "
                         "`plus` = evalplus/humanevalplus (same 164, harder tests)")
    ap.add_argument("--temp", type=float, default=0.0,
                    help="sampling temperature. 0.0 = greedy (default). "
                         "For proper sampling: MiniMax=1.0, Kimi=0.6.")
    ap.add_argument("--top-p", type=float, default=1.0,
                    help="nucleus top_p (only used when temp>0)")
    ap.add_argument("--top-k", type=int, default=0,
                    help="top_k (only used when temp>0, 0 = off)")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed (for reproducibility when temp>0)")
    args = ap.parse_args()
    return run(
        args.model,
        num=args.num,
        max_tokens=args.max_tokens,
        prefill_step_size=args.prefill_step_size,
        thinking=args.thinking,
        temp=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        timeout=args.timeout,
        dataset=args.dataset,
    )


if __name__ == "__main__":
    sys.exit(main())
