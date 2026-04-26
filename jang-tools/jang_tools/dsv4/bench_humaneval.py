#!/usr/bin/env python3
"""HumanEval+ pass@1 done PROPERLY per DSV4 paper protocol.

Per paper page 37:
- Temperature = 1.0 (not 0.6)
- Context window: 8K (chat) / 128K (think) / 384K (max)
  Compromised to 8K/32K/64K for Mac Studio RAM
- Math template: '{question}\\nPlease reason step by step, and put your final answer within \\boxed{}.'
- For code: prompt is just the function stub in markdown code block

Output extraction: find the LAST ```python ``` block in the response,
return body of `def {entry_point}` from inside it.

Tracks per-problem:
  - mode (chat / think / think_max)
  - PASS / FAIL / FAIL-TRUNCATED
  - n_tokens generated
  - finish_reason (eos / length / stop)
  - saw_think_close (only relevant for think modes)
  - extracted code
"""
import os, re, sys, time, json, subprocess
sys.stdout.reconfigure(line_buffering=True)

import mlx.core as mx
mx.set_memory_limit(int(os.environ.get("JANG_MEMORY_LIMIT_GB", "200")) * 1024**3)

from jang_tools.load_jangtq import load_jangtq_model
from jang_tools.dsv4.runtime import generate, GenerateOptions, MODE_DEFAULTS

CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.DOTALL)


def truncate_fim(text):
    """FIM mode: model continues function body. Stop at:
    1. First top-level non-helper line (def/class/etc) — function ends
    2. First repeated non-blank line (decode-loop tail like `return True\nreturn False\nreturn True...`)
    3. Common end markers: `# Example`, `# Test`, `if __name__`, `# Usage`
    """
    # Markers that DROP the entire line (next-section delimiters)
    SECTION_END = ("# Example", "# Test", "# example", "# test", "# Usage",
                   "# usage", "if __name__")
    # Markers that just TRUNCATE the line (EOS tokens emitted as text)
    EOS_TOKENS = ("<｜end▁of▁sentence｜>", "<｜end◁of◁sentence｜>",
                  "<|end|>", "<|e|>", "</s>")
    lines = text.split("\n")
    out = []
    last_nonblank_stripped = None
    repeat_count = 0
    for ln in lines:
        # Section-end markers — drop this line and stop
        if any(m in ln for m in SECTION_END):
            break
        # EOS token — truncate the line at the marker, append, then stop
        eos_pos = -1
        for tok in EOS_TOKENS:
            p = ln.find(tok)
            if p >= 0 and (eos_pos < 0 or p < eos_pos):
                eos_pos = p
        if eos_pos >= 0:
            out.append(ln[:eos_pos])
            break
        # Top-level non-indented line (not a helper) → function body ends
        if ln and not ln.startswith((" ", "\t")):
            s = ln.lstrip()
            if s.startswith(("import ", "from ")):
                out.append(ln); continue
            # Anything else at column 0 = function ended
            break
        # Detect decode-loop via period detection. When the line we're about to
        # add already appeared in `out`, find the gap (period) between its last
        # two occurrences. If the most recent occurrence is within 8 lines AND
        # this is the THIRD time we're about to emit `s`, truncate at the start
        # of the cycle (positions[-1], the first repeat).
        s = ln.strip()
        if s:
            positions = [i for i, o in enumerate(out) if o.strip() == s]
            # Heuristic: 2+ prior occurrences within the last 12 lines means we're
            # entering the third instance — confirms it's a loop, not legit usage.
            if len(positions) >= 2 and positions[-1] >= len(out) - 12:
                # Loop confirmed. Truncate to before the FIRST repeat (positions[-1]).
                # That keeps the legitimate first occurrence at positions[0]
                # and any function body before the cycle started.
                out = out[:positions[-1]]
                break
        out.append(ln)
    while out and not out[-1].strip(): out.pop()
    text_out = "\n".join(out)
    # Strip any trailing EOS-like tokens that escaped the END_MARKERS check
    for marker in ("<｜end▁of▁sentence｜>", "<｜end◁of◁sentence｜>", "<|end|>", "<|e|>", "</s>"):
        if marker in text_out:
            text_out = text_out.split(marker)[0]
    return text_out


def make_messages(problem, mode):
    """FIM = raw stub (no chat template). Chat/think = stub in code block."""
    if mode == "fim":
        return [{"content": problem["prompt"]}]
    return [{"role": "user",
             "content": f"```python\n{problem['prompt']}```"}]


def extract_code(text, entry_point):
    """Robust: strip <think>, prefer FIRST non-empty fence containing entry_point,
    fall back to first fence, unclosed open, then raw text."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    fences = [f.strip() for f in CODE_FENCE_RE.findall(text) if f.strip()]
    keyed = [f for f in fences if entry_point in f]
    if keyed:
        code = keyed[0]
    elif fences:
        code = fences[0]
    else:
        m = re.search(r"```(?:python)?\s*\n", text)
        if m:
            code = text[m.end():].strip()
        else:
            code = text.strip()
    # Find entry_point def in the code
    pat = re.compile(rf"^def\s+{re.escape(entry_point)}\s*\(", re.MULTILINE)
    m = pat.search(code)
    if not m:
        # Maybe function is nested or named differently; just return the whole code block
        return code
    rest = code[m.start():]
    # Take from def to next top-level def/class (or end)
    nxt = re.search(r"\n(def|class)\s", rest[m.end()-m.start():])
    if nxt:
        rest = rest[:m.end()-m.start() + nxt.start()]
    # Drop the def header line; HumanEval test prepends prob.prompt which already has the def
    lines = rest.split("\n")
    body_start = 0
    for i, ln in enumerate(lines):
        if ln.rstrip().endswith(":"):
            body_start = i + 1
            break
    body = lines[body_start:]
    while body and not body[-1].strip(): body.pop()
    return "\n".join(body)


def run_test(prob, completion, timeout=15):
    submission = prob["prompt"] + completion + "\n" + prob["test"] + "\n" + f"check({prob['entry_point']})\n"
    try:
        r = subprocess.run([sys.executable, "-c", submission], timeout=timeout, capture_output=True, text=True)
        return (r.returncode == 0, (r.stderr or "")[:200])
    except subprocess.TimeoutExpired:
        return (False, "TIMEOUT")


def main():
    bundle = sys.argv[1]
    out_jsonl = sys.argv[2] if len(sys.argv) > 2 else "/tmp/he_proper.jsonl"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    modes = sys.argv[4].split(",") if len(sys.argv) > 4 else ["chat"]
    print(f"[bench] bundle={bundle}", flush=True)
    print(f"[bench] modes={modes} limit={limit}", flush=True)
    for m in modes:
        d = MODE_DEFAULTS[m]
        print(f"  {m}: max_tokens={d['max_tokens']} T={d['temperature']} "
              f"enable_thinking={d['enable_thinking']} reasoning_effort={d['reasoning_effort']}", flush=True)

    from evalplus.data import get_human_eval_plus
    plus = get_human_eval_plus()
    items = list(plus.values())[:limit]

    t0 = time.time()
    model, tok = load_jangtq_model(bundle)
    print(f"[bench] load: {time.time()-t0:.1f}s", flush=True)

    results = {}  # (mode, task_id) -> dict
    fout = open(out_jsonl, "w")
    fout.write(json.dumps({"meta": {"modes": modes, "limit": limit, "defaults": {m: MODE_DEFAULTS[m] for m in modes}}}) + "\n")
    fout.flush()

    for mode in modes:
        print(f"\n========== MODE: {mode} ==========", flush=True)
        n_pass = 0
        for i, prob in enumerate(items):
            tid = prob["task_id"]
            entry = prob["entry_point"]
            msgs = make_messages(prob, mode)
            t1 = time.time()
            opts = GenerateOptions(mode=mode)
            res = generate(model, tok, bundle, messages=msgs, opts=opts)
            dt = time.time() - t1

            if res.error:
                ok, err, completion = False, res.error[:140], ""
            else:
                if mode == "fim":
                    completion = truncate_fim(res.raw)
                else:
                    completion = extract_code(res.content, entry)
                ok, err = run_test(prob, completion)

            if ok: n_pass += 1
            results[(mode, tid)] = {
                "ok": ok, "err": err, "dt": dt,
                "n_tokens": res.n_tokens,
                "finish_reason": res.finish_reason,
                "saw_think_close": res.saw_think_close,
                "truncated": res.truncated,
                "completion": completion,
                "raw_len": len(res.raw),
                "raw": res.raw if not ok else "",
                "content_len": len(res.content),
            }
            tag = "PASS" if ok else ("FAIL-TR" if res.truncated else "FAIL")
            pct = 100.0 * n_pass / (i + 1)
            print(f"  [{mode:9s}] [{i+1:2d}/{limit}] {tag:7s} {tid:<14s} {dt:6.1f}s "
                  f"tok={res.n_tokens:5d} {res.finish_reason:6s} pass@1={n_pass}/{i+1} ({pct:.1f}%)" +
                  (f" err={err[:50]}" if not ok else ""), flush=True)
            fout.write(json.dumps({"mode": mode, "task_id": tid, **results[(mode, tid)]}) + "\n")
            fout.flush()
        print(f"  [{mode}] FINAL pass@1 = {n_pass}/{limit} = {100.0*n_pass/limit:.1f}%", flush=True)

    fout.close()

    # Comparison matrix
    if len(modes) > 1:
        print("\n========== COMPARISON MATRIX ==========")
        header = f"{'task_id':<14s}  " + "  ".join(f"{m:<9s}" for m in modes)
        print(header)
        for prob in items:
            tid = prob["task_id"]
            row = f"{tid:<14s}  "
            for m in modes:
                r = results.get((m, tid), {})
                if r.get("ok"): cell = "PASS"
                elif r.get("truncated"): cell = "FAIL-TR"
                else: cell = "FAIL"
                row += f"{cell:<9s}  "
            print(row)


if __name__ == "__main__":
    main()
