"""Normalize a JANG calibration corpus into DeepSeek-V4.1-1017 native conversations.

Input rows (e.g. calib/glm53-flash-nvfp4-2026-10-11/corpus_v2_noreason.jsonl):
  {messages, tools, prompt_only, effort: high|max, domain}
with assistant turns in either structured form (tool_calls field) or inline
GLM/Qwen form (`<think>...</think>` and `<tool_call>{"name","arguments"}</tool_call>`).

Output rows are V4.1 encoder messages:
  - assistant: content / reasoning_content / tool_calls (OpenAI format, JSON-string
    arguments, ids call_<n>); inline think/tool_call text is parsed out
  - tool: role "tool" with tool_call_id matched to the preceding calls in order
  - tools: attached to messages[0] (a system message is inserted when needed)
plus a generation plan:
  kind = "single"   -> generate reasoning + answer for the final turn (any recorded
                       final assistant reply is dropped)
  kind = "agentic"  -> keep recorded actions; regenerate reasoning turn by turn
                       (tools present => the encoder keeps every turn's thinking)
  kind = "chat"     -> multi-turn chat: keep prior turns, generate the last reply

Every row is rendered through the release encoding/encoding.py at its effort in
thinking mode to validate it; failures are reported and skipped.

  python -m jang_tools.dsv41.calib_corpus --bundle <bundle> --corpus <in.jsonl> --out <out.jsonl>
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import re
from pathlib import Path

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def _args_str(args) -> str:
    return args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)


def parse_assistant(msg: dict, next_id) -> dict:
    out = {"role": "assistant"}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    thinks = THINK_RE.findall(content)
    if thinks:
        reasoning = "\n\n".join(t.strip() for t in [reasoning, *thinks] if t.strip())
        content = THINK_RE.sub("", content)
    calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", tc)
        calls.append({"id": next_id(), "type": "function",
                      "function": {"name": fn["name"], "arguments": _args_str(fn.get("arguments", {}))}})
    for raw in CALL_RE.findall(content):
        obj = json.loads(raw)
        calls.append({"id": next_id(), "type": "function",
                      "function": {"name": obj["name"], "arguments": _args_str(obj.get("arguments", {}))}})
    content = CALL_RE.sub("", content).strip()
    out["content"] = content
    if reasoning:
        out["reasoning_content"] = reasoning.strip()
    if calls:
        out["tool_calls"] = calls
    return out


def normalize(row: dict) -> dict:
    counter = iter(range(10**6))
    next_id = lambda: f"call_{next(counter)}"  # noqa: E731
    msgs: list[dict] = []
    pending: list[str] = []
    for m in row["messages"]:
        role = m["role"]
        if role == "assistant":
            a = parse_assistant(m, next_id)
            pending = [tc["id"] for tc in a.get("tool_calls", [])]
            msgs.append(a)
        elif role == "tool":
            tid = pending.pop(0) if pending else f"call_orphan_{len(msgs)}"
            msgs.append({"role": "tool", "tool_call_id": tid, "content": m.get("content") or ""})
        elif role in ("system", "user"):
            msgs.append({"role": role, "content": m.get("content") or ""})
        else:
            raise ValueError(f"unknown role {role}")
    tools = row.get("tools") or None
    if tools:
        if msgs[0]["role"] != "system":
            msgs.insert(0, {"role": "system", "content": ""})
        msgs[0]["tools"] = tools
    agentic = bool(tools) and any(m.get("tool_calls") for m in msgs)
    if agentic:
        kind = "agentic"
    else:
        # generate the final reply ourselves: drop a trailing recorded assistant turn
        while msgs and msgs[-1]["role"] == "assistant":
            msgs.pop()
        n_user = sum(m["role"] == "user" for m in msgs)
        kind = "chat" if n_user > 1 else "single"
    for m in msgs:
        if m["role"] == "assistant" and not agentic:
            m.pop("reasoning_content", None)  # earlier chat turns: encoder drops their thinking anyway
    return {"messages": msgs, "effort": row["effort"], "domain": row["domain"], "kind": kind}


def load_encoding(bundle: Path):
    from jang_tools.dsv41.generate import load_encoding as _le

    return _le(bundle)


def render_prefix(enc, msgs, effort):
    """Prompt text that ends with the assistant header + <think> for the next turn."""
    return enc.encode_messages(msgs, thinking_mode="thinking", reasoning_effort=effort)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="bundle dir (for encoding/ and tokenizer.json)")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from tokenizers import Tokenizer

    bundle = Path(a.bundle).expanduser()
    enc = load_encoding(bundle)
    tok = Tokenizer.from_file(str(bundle / "tokenizer.json"))
    rows = [json.loads(l) for l in open(a.corpus)]
    out, fails = [], collections.Counter()
    stats = collections.Counter()
    for i, r in enumerate(rows):
        try:
            n = normalize(r)
            msgs = copy.deepcopy(n["messages"])
            if n["kind"] == "agentic":
                # full trajectory must render (thinking kept on every turn because tools are present)
                for m in msgs:
                    if m["role"] == "assistant":
                        m.setdefault("reasoning_content", "")
                text = enc.encode_messages(msgs, thinking_mode="thinking", reasoning_effort=n["effort"])
            else:
                text = render_prefix(enc, msgs, n["effort"])
            n["n_tok_prompt"] = len(tok.encode(text, add_special_tokens=False).ids)
            n["n_assistant_turns"] = sum(m["role"] == "assistant" for m in n["messages"])
            n["id"] = i
            out.append(n)
            stats[(n["kind"], n["effort"])] += 1
        except (ValueError, AssertionError, IndexError, KeyError, TypeError) as e:  # a malformed source row is counted, not fatal
            fails[f"{r['domain']}: {type(e).__name__}: {str(e)[:80]}"] += 1
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for n in out:
            f.write(json.dumps(n, ensure_ascii=False) + "\n")
    print(f"normalized {len(out)}/{len(rows)}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    for k, v in fails.most_common():
        print(f"  FAIL x{v} {k}")


if __name__ == "__main__":
    main()
