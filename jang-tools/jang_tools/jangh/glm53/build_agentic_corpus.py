"""Build GLM-5.3-rendered agentic calibration + held-out reference sequences.

Output jsonl rows: {"ids": [...], "ref": bool, "domain": str, "decisions": [[pos, kind], ...]}
  decisions: teacher-forced positions p whose NEXT token (ids[p+1]) is the assistant's first content decision after
  "<|assistant|><think></think>": kind "call" (next token is <tool_call>) or "answer" (tools present, no call needed).
Calibration and reference use DISJOINT tool subsets and disjoint phrasings (held-out behavior, not memorized prompts).
Documents (agentic / coding / general domains of corpus_v3.jsonl) are rendered as single user turns (<= 2048 tokens).
"""
from __future__ import annotations

import argparse
import json
import random

from transformers import AutoTokenizer


def schema(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": {k: {"type": t, "description": d} for k, (t, d) in props.items()}, "required": req}}}


# name -> (schema, request phrasings, args(rng), result(args), answer(args,result))
CITIES = ["Lisbon", "Osaka", "Denver", "Nairobi", "Oslo", "Lima", "Hanoi", "Perth", "Quebec", "Tbilisi"]
TOOLS = {
    "get_weather": (schema("get_weather", "Current weather for a city.", {"city": ("string", "city name")}, ["city"]),
                    ["What's the weather in {city} right now?", "Is it raining in {city} today?", "How warm is it in {city}?"],
                    lambda r: {"city": r.choice(CITIES)}, lambda a: {"temp_c": 18, "sky": "clear"},
                    lambda a, res: f"It's {res['temp_c']}°C and {res['sky']} in {a['city']}."),
    "calendar_add": (schema("calendar_add", "Add a calendar event.", {"title": ("string", "title"), "date": ("string", "YYYY-MM-DD")}, ["title", "date"]),
                     ["Put '{title}' on my calendar for {date}.", "Schedule {title} on {date}, please."],
                     lambda r: {"title": r.choice(["Dentist", "Team sync", "Piano lesson", "Car service"]), "date": f"2026-{r.randint(1,12):02d}-{r.randint(1,28):02d}"},
                     lambda a: {"ok": True}, lambda a, res: f"Added '{a['title']}' on {a['date']}."),
    "read_file": (schema("read_file", "Read a text file from the workspace.", {"path": ("string", "file path")}, ["path"]),
                  ["Open {path} and summarize it.", "What does {path} say?"],
                  lambda r: {"path": r.choice(["notes/todo.md", "docs/plan.txt", "README.md", "src/config.yaml"])},
                  lambda a: {"content": "1. finish report\n2. email Sam"}, lambda a, res: "It lists two items: finish the report and email Sam."),
    "run_sql": (schema("run_sql", "Run a read-only SQL query on the analytics database.", {"query": ("string", "SQL")}, ["query"]),
                ["How many {table} rows are there? Check the database.", "Count the entries in the {table} table."],
                lambda r: {"table": r.choice(["orders", "users", "invoices"])}, lambda a: {"rows": [[1204]]},
                lambda a, res: "There are 1,204 rows."),
    "web_search": (schema("web_search", "Search the web and return snippets.", {"query": ("string", "search query")}, ["query"]),
                   ["Find the latest news about {topic}.", "Look up {topic} online."],
                   lambda r: {"topic": r.choice(["solar panel efficiency", "the Rust 2026 edition", "Mars sample return"])},
                   lambda a: {"results": ["snippet one", "snippet two"]}, lambda a, res: "Here is what I found: snippet one; snippet two."),
    "calculator": (schema("calculator", "Evaluate an arithmetic expression exactly.", {"expression": ("string", "expression")}, ["expression"]),
                   ["Use the calculator for {a} * {b}.", "Compute {a} times {b} exactly with the calculator tool."],
                   lambda r: {"a": r.randint(100, 9999), "b": r.randint(100, 9999)}, lambda a: {"value": a["a"] * a["b"]},
                   lambda a, res: f"{a['a']} × {a['b']} = {res['value']}."),
    "send_email": (schema("send_email", "Send an email.", {"to": ("string", "recipient"), "subject": ("string", "subject"), "body": ("string", "body")}, ["to", "subject", "body"]),
                   ["Email {to} that the build passed.", "Send {to} a note saying the meeting moved to 3pm."],
                   lambda r: {"to": r.choice(["ana@example.com", "sam@example.org", "lee@example.net"])}, lambda a: {"sent": True},
                   lambda a, res: f"Done — the email to {a['to']} was sent."),
    "translate": (schema("translate", "Translate text into a target language.", {"text": ("string", "text"), "target": ("string", "language")}, ["text", "target"]),
                  ["Translate '{text}' into {lang} using the translator.", "Use translate to put '{text}' in {lang}."],
                  lambda r: {"text": r.choice(["good morning", "where is the station", "thank you"]), "lang": r.choice(["French", "Japanese", "German"])},
                  lambda a: {"translation": "…"}, lambda a, res: "Here is the translation."),
    "convert_units": (schema("convert_units", "Convert a quantity between units.", {"value": ("number", "amount"), "from_unit": ("string", "from"), "to_unit": ("string", "to")}, ["value", "from_unit", "to_unit"]),
                      ["Convert {v} miles to kilometers.", "How many pounds is {v} kg? Use the converter."],
                      lambda r: {"v": r.choice([3, 12.5, 26.2, 100])}, lambda a: {"result": 42.0}, lambda a, res: "That converts to 42.0."),
    "set_reminder": (schema("set_reminder", "Create a reminder.", {"text": ("string", "reminder"), "time": ("string", "HH:MM")}, ["text", "time"]),
                     ["Remind me to {what} at {time}.", "Set a reminder at {time}: {what}."],
                     lambda r: {"what": r.choice(["call mom", "water plants", "take a break"]), "time": f"{r.randint(6,22):02d}:{r.choice(['00','15','30','45'])}"},
                     lambda a: {"ok": True}, lambda a, res: "Reminder set."),
}
NO_TOOL = ["What is the capital of Australia?", "Explain recursion in one sentence.", "Write a haiku about autumn.",
           "What is 12 plus 30?", "Give me a synonym for 'quick'.", "Summarize: the cat sat on the mat."]


def build_args(name, a):
    if name == "get_weather": return {"city": a["city"]}
    if name == "calendar_add": return {"title": a["title"], "date": a["date"]}
    if name == "read_file": return {"path": a["path"]}
    if name == "run_sql": return {"query": f"SELECT COUNT(*) FROM {a['table']}"}
    if name == "web_search": return {"query": a["topic"]}
    if name == "calculator": return {"expression": f"{a['a']}*{a['b']}"}
    if name == "send_email": return {"to": a["to"], "subject": "Update", "body": "See subject."}
    if name == "translate": return {"text": a["text"], "target": a["lang"]}
    if name == "convert_units": return {"value": a["v"], "from_unit": "mi", "to_unit": "km"}
    if name == "set_reminder": return {"text": a["what"], "time": a["time"]}
    raise KeyError(name)


def fill(tpl, a):
    m = {"city": a.get("city"), "title": a.get("title"), "date": a.get("date"), "path": a.get("path"), "table": a.get("table"),
         "topic": a.get("topic"), "a": a.get("a"), "b": a.get("b"), "to": a.get("to"), "text": a.get("text"), "lang": a.get("lang"),
         "v": a.get("v"), "what": a.get("what"), "time": a.get("time")}
    return tpl.format(**{k: v for k, v in m.items() if v is not None})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--corpus", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--calib-convs", type=int, default=320); ap.add_argument("--ref-convs", type=int, default=64)
    ap.add_argument("--docs", type=int, default=64); ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    r = random.Random(a.seed)
    names = sorted(TOOLS)
    r.shuffle(names)
    ref_tools, cal_tools = set(names[:3]), set(names[3:])           # disjoint tool subsets
    TC, DEC = tok.convert_tokens_to_ids("<tool_call>"), tok.convert_tokens_to_ids("</think>")
    rows = []

    def ids_of(text):
        return list(tok(text, add_special_tokens=False)["input_ids"])

    def render(msgs, tools, think):
        return ids_of(tok.apply_chat_template(msgs, tools=tools, tokenize=False, enable_thinking=think))

    def decisions(ids, n_assist_turns_expected):
        out = []
        for p in range(len(ids) - 1):
            if ids[p] == DEC:                                       # "</think>" -> next token = decision
                out.append([p, "call" if ids[p + 1] == TC else "answer"])
        return out

    def conv(tool_pool, is_ref, n):
        for _ in range(n):
            think = r.random() < 0.3
            k = r.randint(2, 5)
            present = r.sample(sorted(tool_pool), min(k, len(tool_pool)))
            tools = [TOOLS[t][0] for t in present]
            if r.random() < 0.2:                                     # negative: tools present, no call needed
                q = r.choice(NO_TOOL)
                msgs = [{"role": "user", "content": q}, {"role": "assistant", "content": "Sure — " + q.lower()}]
            else:
                name = r.choice(present)
                sch, phr, argf, resf, ansf = TOOLS[name]
                phr = phr[0::2] if not is_ref else phr[1::2] or phr   # disjoint phrasings
                av = argf(r); call_args = build_args(name, av); res = resf(av)
                msgs = [{"role": "user", "content": fill(r.choice(phr), av)},
                        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": name, "arguments": call_args}}]},
                        {"role": "tool", "tool_call_id": "c1", "content": json.dumps(res)},
                        {"role": "assistant", "content": ansf(av, res)}]
            ids = render(msgs, tools, think)
            if len(ids) <= 2048:
                rows.append({"ids": ids, "ref": is_ref, "domain": "tool_conv", "decisions": decisions(ids, 2)})

    conv(cal_tools, False, a.calib_convs)
    conv(ref_tools, True, a.ref_convs)
    docs = [json.loads(l) for l in open(a.corpus)]
    docs = [d for d in docs if d.get("domain") in ("agentic", "coding", "general")]
    r.shuffle(docs)
    for i, d in enumerate(docs[: a.docs]):
        ids = ids_of(tok.apply_chat_template([{"role": "user", "content": d["text"][:12000]}], tokenize=False, add_generation_prompt=False))
        rows.append({"ids": ids[:2048], "ref": i % 8 == 0, "domain": "doc_" + d["domain"], "decisions": []})
    with open(a.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    n_tok = sum(len(x["ids"]) for x in rows)
    n_dec = sum(len(x["decisions"]) for x in rows if x["ref"])
    print(f"{len(rows)} sequences, {n_tok} tokens; ref sequences {sum(x['ref'] for x in rows)} with {n_dec} decision positions; "
          f"ref tools {sorted(ref_tools)} | calib tools {sorted(cal_tools)}")


if __name__ == "__main__":
    main()
