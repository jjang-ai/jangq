"""Audit normalized DSV4.1 calibration rows against the release encoder/parser and tokenizer.

Checks, per row (failures are listed; exit 1 if any):
  effort     thinking-mode render starts with exactly one
             "<｜System｜>Reasoning Effort: {75|100} (range 1-100, ...)" matching the row effort;
             a chat-mode render of the same messages carries no effort prefix
  prefix     generation prefixes end with "<｜Assistant｜><think>"
  tools      every tool message links to a call id issued by the preceding assistant turn;
             tool results render as <tool_result> inside <｜User｜>
  roundtrip  every assistant turn of agentic rows, rendered as a completion, parses back with
             parse_message_from_completion_text to the same reasoning, content, tool names and
             JSON-equal arguments
  tokens     template special tokens are single tokenizer ids, and every render survives
             tokenizer encode -> decode unchanged

  python -m jang_tools.dsv41.calib_audit --bundle <bundle> --rows calib/dsv41-1017/corpus_v41.jsonl
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import re
import sys
from pathlib import Path

from jang_tools.dsv41.generate import load_encoding

EFFORT = {"high": 75, "max": 100, "low": 50}
# vocab special tokens used by the 1017 template. DSML tags are "<" + "｜DSML｜" + " calls>" etc. and
# <tool_result> is plain text, both by design of the release template.
SPECIALS = ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>", "<｜System｜>",
            "<think>", "</think>", "｜DSML｜"]


def completion_text(enc, m: dict) -> str:
    """What the model would emit for this assistant turn after '<think>' (thinking mode)."""
    msgs = [{"role": "user", "content": "x"}, copy.deepcopy(m)]
    if msgs[1].get("tool_calls"):
        msgs[0] = {"role": "system", "content": "", "tools": [{"type": "function", "function": {
            "name": tc["function"]["name"], "description": "", "parameters": {"type": "object", "properties": {}}}}
            for tc in m["tool_calls"]]}
        msgs.insert(1, {"role": "user", "content": "x"})
    full = enc.encode_messages(msgs, thinking_mode="thinking", drop_thinking=False)
    head = enc.encode_messages(msgs[:-1], thinking_mode="thinking", drop_thinking=False)
    assert full.startswith(head), "assistant render does not extend the generation prefix"
    return full[len(head):]


def same_args(a: str, b) -> bool:
    try:
        return json.loads(a) == (json.loads(b) if isinstance(b, str) else b)
    except (ValueError, TypeError):  # not JSON on either side
        return False


def audit_row(enc, tok, row) -> list[str]:
    errs = []
    eff = row["effort"]
    msgs = copy.deepcopy(row["messages"])
    if row["kind"] == "agentic":
        for m in msgs:
            if m["role"] == "assistant":
                m.setdefault("reasoning_content", "")
    text = enc.encode_messages(msgs, thinking_mode="thinking", reasoning_effort=eff)
    want = (f"<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: {EFFORT[eff]} (range 1-100, the higher the value, "
            "the more thorough the reasoning)\n\n")
    if not text.startswith(want):
        errs.append(f"effort: prefix mismatch: {text[:120]!r}")
    if text.count("Reasoning Effort:") != 1:
        errs.append(f"effort: {text.count('Reasoning Effort:')} effort prefixes")
    chat = enc.encode_messages(msgs, thinking_mode="chat")
    if "Reasoning Effort:" in chat:
        errs.append("effort: present in chat mode")

    # generation prefixes: final turn for single/chat, every assistant turn for agentic
    cut_points = [i for i, m in enumerate(msgs) if m["role"] == "assistant"] if row["kind"] == "agentic" else [len(msgs)]
    for i in cut_points:
        p = enc.encode_messages(msgs[:i], thinking_mode="thinking", reasoning_effort=eff)
        if not p.endswith("<｜Assistant｜><think>"):
            errs.append(f"prefix: turn {i} ends {p[-40:]!r}")

    issued, linked = set(), True
    for m in msgs:
        if m["role"] == "assistant":
            issued = {tc["id"] for tc in m.get("tool_calls", [])}
        elif m["role"] == "tool" and m.get("tool_call_id") not in issued:
            linked = False
    if not linked:
        errs.append("tools: tool result not linked to a preceding call")
    if any(m["role"] == "tool" for m in msgs) and "<tool_result>" not in text:
        errs.append("tools: tool results not rendered as <tool_result>")

    if row["kind"] == "agentic":
        for j, m in enumerate(m for m in msgs if m["role"] == "assistant"):
            comp = completion_text(enc, m)
            try:
                parsed = enc.parse_message_from_completion_text(comp, thinking_mode="thinking")
            except (ValueError, AssertionError, IndexError, KeyError) as e:
                errs.append(f"roundtrip: turn {j} parse error {type(e).__name__}: {str(e)[:80]}")
                continue
            if (parsed.get("reasoning_content") or "").strip() != (m.get("reasoning_content") or "").strip():
                errs.append(f"roundtrip: turn {j} reasoning differs")
            if (parsed.get("content") or "").strip() != (m.get("content") or "").strip():
                errs.append(f"roundtrip: turn {j} content differs")
            want_calls = m.get("tool_calls", [])
            got_calls = parsed.get("tool_calls", [])
            if len(want_calls) != len(got_calls):
                errs.append(f"roundtrip: turn {j} {len(got_calls)} calls parsed, {len(want_calls)} expected")
            for w, g in zip(want_calls, got_calls):
                gf = g.get("function", g)
                if gf.get("name") != w["function"]["name"] or not same_args(w["function"]["arguments"], gf.get("arguments")):
                    errs.append(f"roundtrip: turn {j} call {w['function']['name']} differs")

    for t in (text,):
        ids = tok.encode(t, add_special_tokens=False).ids
        if tok.decode(ids, skip_special_tokens=False) != t:
            errs.append("tokens: encode/decode round-trip changed the text")
        if "<｜DSML｜ invoke" in t and t.count("<｜DSML｜ invoke") > ids.count(tok.token_to_id("｜DSML｜")):
            errs.append("tokens: DSML tag not tokenized through the ｜DSML｜ special token")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--rows", required=True)
    a = ap.parse_args()
    from tokenizers import Tokenizer

    bundle = Path(a.bundle).expanduser()
    enc = load_encoding(bundle)
    tok = Tokenizer.from_file(str(bundle / "tokenizer.json"))
    bad_tok = [s for s in SPECIALS if len(tok.encode(s, add_special_tokens=False).ids) != 1]
    print("special tokens as single ids:", "OK" if not bad_tok else f"NOT single: {bad_tok}")
    rows = [json.loads(l) for l in open(a.rows)]
    fails = collections.Counter()
    nbad = 0
    for r in rows:
        errs = audit_row(enc, tok, r)
        if errs:
            nbad += 1
            for e in errs:
                fails[(r["domain"], re.sub(r"turn \d+", "turn N", e)[:110])] += 1
    print(f"rows audited {len(rows)}, rows with issues {nbad}")
    for (d, e), v in fails.most_common(30):
        print(f"  x{v:4d} [{d}] {e}")
    sys.exit(1 if nbad or bad_tok else 0)


if __name__ == "__main__":
    main()
