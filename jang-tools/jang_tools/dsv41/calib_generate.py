"""Locally generate DSV4.1-1017 reasoning traces for selected calibration rows.

Runs the local JANG bundle (no network). By default only the non-security technical
domains are generated (coding, agentic/terminal tool use, reasoning, systems,
long-context, ml-inference); pass --domains to override. Each row is rendered
through the release encoder (thinking mode, its own effort) and completed with
batched decode.

  single   generate reasoning + answer for the final turn
  agentic  regenerate each assistant turn's reasoning in place, keeping the
           recorded tool calls and tool results (turn-by-turn agent logic)

Output rows carry the generated text back in `messages` (reasoning_content filled,
assistant content replaced for single rows) plus `gen` provenance, ready for
calib_audit.py and the calibration stats pass.

  python -m jang_tools.dsv41.calib_generate --bundle <bundle> \
      --rows calib/dsv41-1017/corpus_v41.jsonl --out calib/dsv41-1017/traced.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

DEFAULT_DOMAINS = {
    "reason_long", "reason_53flash", "terminal_agent", "toolcall_53flash",
    "v3_coding", "v3_agentic", "v3_systems", "v3_longctx", "ml_inference",
}
EFFORT = {"low": 50, "high": 75, "max": 100}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--engram-dir")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--domains", help="comma list; default = technical non-security domains")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--max-new", type=int, default=1536)
    ap.add_argument("--max-prompt", type=int, default=7000)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    from jang_tools.dsv41.model import DSV41
    from jang_tools.dsv41.generate import load_encoding

    bundle = Path(a.bundle).expanduser()
    enc = load_encoding(bundle)
    domains = set(a.domains.split(",")) if a.domains else DEFAULT_DOMAINS
    mx.random.seed(a.seed)

    rows = [json.loads(l) for l in open(a.rows)]
    rows = [r for r in rows if r["domain"] in domains and r.get("n_tok_prompt", 0) <= a.max_prompt]
    if a.limit:
        rows = rows[: a.limit]
    for i, r in enumerate(rows):
        r["_ord"] = i  # stable key for checkpoint/resume (deterministic given same inputs)

    # Resume: completed rows are appended to <out>.partial as they finish, so a kill
    # only loses the rows still in flight. On restart we skip rows already in partial.
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(a.out) + ".partial")
    done_ords = set()
    if partial.exists():
        for l in open(partial):
            l = l.strip()
            if not l:
                continue
            try:
                done_ords.add(json.loads(l)["_ord"])
            except (ValueError, KeyError):  # a torn last line from an interrupted run is skipped
                continue
    todo = [r for r in rows if r["_ord"] not in done_ords]
    if done_ords:
        print(f"resume: {len(done_ords)} rows already in {partial.name}; {len(todo)} remaining", flush=True)

    m = DSV41(bundle, a.engram_dir)
    eos = {m.tokenizer.token_to_id("<｜end▁of▁sentence｜>"), m.eos_id}
    think_end = m.tokenizer.token_to_id("</think>")

    # one generation job per assistant turn we must produce
    Job = lambda **k: dict(k)  # noqa: E731
    jobs = []
    remaining = {}  # id(row) -> number of its jobs not yet applied
    for r in todo:
        if r["kind"] == "agentic":
            turns = [i for i, msg in enumerate(r["messages"]) if msg["role"] == "assistant"]
            rjobs = [Job(row=r, upto=ti, want="reason") for ti in turns]  # keep recorded calls/content, regen reasoning
        else:
            rjobs = [Job(row=r, upto=len(r["messages"]), want="full")]
        jobs.extend(rjobs)
        remaining[id(r)] = len(rjobs)

    print(f"{len(todo)} rows -> {len(jobs)} generation turns; batch {a.batch}", flush=True)
    done = 0
    t0 = time.time()
    results = {id(r): copy.deepcopy(r) for r in todo}
    for r in todo:
        results[id(r)]["gen"] = {"domain": r["domain"], "effort": r["effort"], "kind": r["kind"], "turns": []}
    pf = open(partial, "a")  # append; resume preserves earlier rows
    for b in range(0, len(jobs), a.batch):
        chunk = jobs[b : b + a.batch]
        sess, texts, active = [], [], []
        for job in chunk:
            msgs = copy.deepcopy(job["row"]["messages"][: job["upto"]])
            prompt = enc.encode_messages(msgs, thinking_mode="thinking", reasoning_effort=job["row"]["effort"])
            ids = m.tokenizer.encode(prompt, add_special_tokens=False).ids
            s = m.new_session()
            lg = m.forward(ids, s)
            sess.append(s)
            texts.append([])
            active.append(True)
            job["_ids"] = ids
            job["_first"] = lg
        # seed first sampled token for each
        toks = [_sample(job["_first"], a.temperature, a.top_p) for job in chunk]
        for i, (job, tk) in enumerate(zip(chunk, toks)):
            texts[i].append(tk)
            if tk in eos:
                active[i] = False
        for _ in range(a.max_new - 1):
            idx = [i for i, on in enumerate(active) if on]
            if not idx:
                break
            lg = m.decode_batch([sess[i] for i in idx], [toks[i] for i in idx])
            for j, i in enumerate(idx):
                tk = _sample(lg[j], a.temperature, a.top_p)
                toks[i] = tk
                texts[i].append(tk)
                if tk in eos:
                    active[i] = False
        for job, out in zip(chunk, texts):
            _apply(enc, m, job, out, eos, results[id(job["row"])])
            remaining[id(job["row"])] -= 1
            if remaining[id(job["row"])] == 0:  # this row's turns are all generated -> checkpoint it
                pf.write(json.dumps(results[id(job["row"])], ensure_ascii=False) + "\n")
                pf.flush()
        done += len(chunk)
        print(f"  {done}/{len(jobs)} turns  ({time.time() - t0:.0f}s)", flush=True)
    pf.close()

    # Assemble the final file from the partial, in stable row order.
    by_ord = {}
    for l in open(partial):
        l = l.strip()
        if l:
            r = json.loads(l)
            by_ord[r["_ord"]] = r
    out_rows = [by_ord[r["_ord"]] for r in rows if r["_ord"] in by_ord]
    with open(a.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    empty = sum(not any(t["reason_chars"] for t in r["gen"]["turns"]) for r in out_rows)
    print(f"wrote {len(out_rows)} rows to {a.out} in {time.time() - t0:.0f}s; {empty} rows had no reasoning")


def _sample(logits, temperature, top_p):
    if temperature <= 0:
        return int(mx.argmax(logits))
    p = mx.softmax(logits / temperature, axis=-1)
    if top_p < 1.0:
        order = mx.argsort(-p)
        sp = p[order]
        keep = mx.cumsum(sp) - sp < top_p
        sp = mx.where(keep, sp, 0)
        return int(order[mx.random.categorical(mx.log(sp))])
    return int(mx.random.categorical(mx.log(p)))


def _apply(enc, m, job, out_ids, eos, result):
    """Write the generated turn back into `result` (the row's output copy)."""
    text = m.tokenizer.decode([t for t in out_ids if t not in eos], skip_special_tokens=False)
    try:
        parsed = enc.parse_message_from_completion_text(
            text + "<｜end▁of▁sentence｜>", thinking_mode="thinking")
    except (ValueError, AssertionError, IndexError, KeyError):  # unparseable turn: split at </think> instead
        head, _, tail = text.partition("</think>")
        parsed = {"reasoning_content": head, "content": tail}
    reason = (parsed.get("reasoning_content") or "").strip()
    content = (parsed.get("content") or "").strip()
    msg = result["messages"][job["upto"]] if job["want"] == "reason" else None
    if job["want"] == "reason":
        # agentic: keep recorded content + tool_calls, insert the model's reasoning
        if reason:
            msg["reasoning_content"] = reason
    else:
        # single: the model writes the whole final turn
        result["messages"].append({"role": "assistant", "content": content, "reasoning_content": reason})
    result["gen"]["turns"].append({"upto": job["upto"], "want": job["want"],
                                   "reason_chars": len(reason), "content_chars": len(content),
                                   "n_gen": len(out_ids), "eos": out_ids[-1] in eos})


if __name__ == "__main__":
    main()
