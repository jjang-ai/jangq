"""Naive-N0.5-Flash calibration + held-out reference corpus v2 — weighted to AGENTIC, CODING and CYBERSEC use,
rendered with the model's OWN chat template.

Output jsonl rows: {"ids", "ref", "set": "calib"|"klref"|"agentic_ref", "domain", "decisions": [[pos, kind], ...]}
  calib (~610k tokens): coding 28 / tool conversations 18 / cybersec 15 / agentic documents 14 / general 10 /
                        chinese 5 / long 4-8k 5 / science 3 / academic 2
  klref               : 28 held-out prompts: coding 5, agentic 4, cybersec 4, general 3, science 2, academic 2, chinese 2
                        (1.5-2k tokens each) + 6 long ones (4.1-6.1k tokens, DSA top-2048 selection active)
  agentic_ref         : 96 held-out tool conversations, decision positions marked
Tool conversations use agent-style tools (shell, files, search, tests, HTTP, DNS, ports, hashes, SQL) next to everyday
ones, single and CHAINED calls, 30% with reasoning in the assistant turns, 20% "tools present, no call needed".
Held-out tools and phrasings are DISJOINT from the calibration ones; documents are disjoint at the record level.
decisions: position p whose next token is the first content token after "</think>": "call" (<tool_call>) or "answer".
"""
from __future__ import annotations

import argparse
import json
import random

from transformers import AutoTokenizer

QUOTA = {"coding": 0.28, "cybersec": 0.15, "agentic": 0.14, "general": 0.10, "chinese": 0.05, "science": 0.03, "academic_mc": 0.02}
TOOL_FRAC, LONG_FRAC = 0.18, 0.05
KLREF = [("coding", 5), ("agentic", 4), ("cybersec", 4), ("general", 3), ("science", 2), ("academic_mc", 2), ("chinese", 2)]


def fn(tool_name, desc, **props):
    return {"type": "function", "function": {"name": tool_name, "description": desc, "parameters": {
        "type": "object", "properties": {k: {"type": t, "description": d} for k, (t, d) in props.items()}, "required": list(props)}}}


PATHS = ["src/server.py", "lib/parser.rs", "app/models/user.rb", "cmd/agent/main.go", "pkg/auth/token.ts", "tests/test_cache.py"]
HOSTS = ["10.0.4.17", "staging-db.internal", "192.168.20.5", "build-runner-3", "edge-gw.corp.example"]
DOMAINS = ["example.org", "files.example.net", "login.example.com", "cdn.example.io"]
SYMS = ["parse_header", "TokenCache", "retry_with_backoff", "load_config", "verify_signature"]
CITIES = ["Lisbon", "Osaka", "Denver", "Nairobi", "Oslo", "Lima", "Hanoi", "Perth"]

# name -> (schema, generator(r, fam) -> (user text, arguments, result, final answer)); fam 0 = calibration phrasing, 1 = held-out
T = {
    "bash": (fn("bash", "Run a shell command in the workspace and return stdout/stderr.", command=("string", "command line")),
             lambda r, f: (lambda c: ([f"Run `{c}` and tell me what it prints.", f"What is the output of `{c}` here?"][f], {"command": c},
                                      {"exit_code": 0, "stdout": "ok\n3 passed\n", "stderr": ""}, "The command exited with 0 and printed `ok` and `3 passed`."))(
                 r.choice(["git status --short", "ls -la build/", "cargo check", "python -m pytest -q tests/test_cache.py", "df -h /", "uname -a"]))),
    "read_file": (fn("read_file", "Read a text file from the workspace.", path=("string", "file path")),
                  lambda r, f: (lambda p: ([f"Open {p} and summarize what it does.", f"What is in {p}?"][f], {"path": p},
                                           {"content": "def handle(req):\n    return route(req.path)\n"}, f"{p} defines `handle`, which routes a request by its path."))(r.choice(PATHS))),
    "write_file": (fn("write_file", "Create or overwrite a file.", path=("string", "file path"), content=("string", "full file content")),
                   lambda r, f: (lambda p: ([f"Create {p} with a function that returns the string 'ready'.", f"Write a stub into {p} that returns 'ready'."][f],
                                            {"path": p, "content": "def status():\n    return 'ready'\n"}, {"ok": True, "bytes": 35}, f"Wrote {p} (35 bytes) with a `status()` function."))(r.choice(PATHS))),
    "grep_search": (fn("grep_search", "Search the workspace for a regular expression.", pattern=("string", "regex"), path=("string", "directory")),
                    lambda r, f: (lambda s: ([f"Find every place where {s} is used under src/.", f"Where is {s} referenced in the source tree?"][f], {"pattern": s, "path": "src/"},
                                             {"matches": [f"src/server.py:41: {s}(", f"src/util.py:9: def {s}("]}, f"`{s}` is defined in src/util.py:9 and called in src/server.py:41."))(r.choice(SYMS))),
    "list_dir": (fn("list_dir", "List the entries of a directory.", path=("string", "directory")),
                 lambda r, f: (lambda p: ([f"What files are in {p}?", f"Show me the contents of the {p} directory."][f], {"path": p},
                                          {"entries": ["main.py", "util.py", "README.md"]}, f"{p} contains main.py, util.py and README.md."))(r.choice(["src/", "tests/", "docs/", "scripts/"]))),
    "run_tests": (fn("run_tests", "Run the project's test suite, optionally one file.", target=("string", "test file or 'all'")),
                  lambda r, f: (lambda p: ([f"Run the tests in {p}.", f"Do the tests in {p} pass?"][f], {"target": p},
                                           {"passed": 11, "failed": 1, "failures": ["test_expiry: AssertionError"]}, "11 tests pass and 1 fails: `test_expiry` raises an AssertionError."))(r.choice(PATHS[-1:] + ["tests/test_api.py", "all"]))),
    "http_get": (fn("http_get", "Fetch a URL and return status, headers and the first bytes of the body.", url=("string", "URL")),
                 lambda r, f: (lambda d: ([f"Fetch https://{d}/health and report the status.", f"Is https://{d}/health responding?"][f], {"url": f"https://{d}/health"},
                                          {"status": 200, "headers": {"server": "nginx", "x-frame-options": "DENY"}, "body": "{\"ok\":true}"}, f"https://{d}/health returns 200 with body {{\"ok\":true}}."))(r.choice(DOMAINS))),
    "port_scan": (fn("port_scan", "Scan TCP ports of a host you are authorized to test.", host=("string", "hostname or IP"), ports=("string", "port list or range")),
                  lambda r, f: (lambda h: ([f"Check which of ports 22, 80, 443 and 5432 are open on {h} (in scope for this assessment).", f"Which common service ports are reachable on {h}? It is part of the authorized test range."][f],
                                           {"host": h, "ports": "22,80,443,5432"}, {"open": [22, 443], "closed": [80, 5432]}, f"On {h}, ports 22 and 443 are open; 80 and 5432 are closed."))(r.choice(HOSTS))),
    "dns_lookup": (fn("dns_lookup", "Resolve DNS records of a name.", name=("string", "domain name"), record_type=("string", "A, AAAA, MX, TXT, ...")),
                   lambda r, f: (lambda d: ([f"What are the MX records of {d}?", f"Look up the mail servers for {d}."][f], {"name": d, "record_type": "MX"},
                                            {"records": ["10 mx1." + d, "20 mx2." + d]}, f"{d} has two MX records: mx1.{d} (priority 10) and mx2.{d} (priority 20)."))(r.choice(DOMAINS))),
    "hash_file": (fn("hash_file", "Compute a cryptographic hash of a file.", path=("string", "file path"), algorithm=("string", "sha256, sha1, md5")),
                  lambda r, f: (lambda p: ([f"Give me the SHA-256 of {p}.", f"Compute the sha256 digest of {p} so I can compare it with the advisory."][f], {"path": p, "algorithm": "sha256"},
                                           {"digest": "9f2c1a7be0d44c51a3f0b6d2e8c97715a0e4b3d6c1f28a9974d05be36c1f7a20"}, f"SHA-256 of {p}: 9f2c1a7be0d44c51a3f0b6d2e8c97715a0e4b3d6c1f28a9974d05be36c1f7a20."))(r.choice(["dist/agent.bin", "downloads/update.pkg", "samples/dropper.dll"]))),
    "log_query": (fn("log_query", "Query the security event log.", query=("string", "filter expression"), hours=("integer", "look-back window in hours")),
                  lambda r, f: (lambda h: ([f"How many failed SSH logins did {h} see in the last 24 hours?", f"Count failed ssh authentication events on {h} over the past day."][f],
                                           {"query": f"host={h} event=ssh_auth_failed", "hours": 24}, {"count": 412, "top_sources": ["203.0.113.9", "198.51.100.77"]}, f"{h} logged 412 failed SSH logins in 24 hours, mostly from 203.0.113.9 and 198.51.100.77."))(r.choice(HOSTS))),
    "run_sql": (fn("run_sql", "Run a read-only SQL query on the analytics database.", query=("string", "SQL")),
                lambda r, f: (lambda t: ([f"How many rows does the {t} table have?", f"Count the entries in {t}."][f], {"query": f"SELECT COUNT(*) FROM {t}"},
                                         {"rows": [[1204]]}, f"{t} has 1,204 rows."))(r.choice(["orders", "users", "invoices", "sessions"]))),
    "web_search": (fn("web_search", "Search the web and return snippets.", query=("string", "search query")),
                   lambda r, f: (lambda q: ([f"Find recent information about {q}.", f"Look up {q} online."][f], {"query": q},
                                            {"results": ["snippet one", "snippet two"]}, "Here is what I found: snippet one; snippet two."))(r.choice(["the Rust 2026 edition", "CVE advisories for OpenSSH", "MLX release notes"]))),
    "get_weather": (fn("get_weather", "Current weather for a city.", city=("string", "city name")),
                    lambda r, f: (lambda c: ([f"What's the weather in {c} right now?", f"Is it warm in {c} today?"][f], {"city": c}, {"temp_c": 18, "sky": "clear"}, f"It's 18 °C and clear in {c}."))(r.choice(CITIES))),
    "calendar_add": (fn("calendar_add", "Add a calendar event.", title=("string", "title"), date=("string", "YYYY-MM-DD")),
                     lambda r, f: (lambda t, d: ([f"Put '{t}' on my calendar for {d}.", f"Schedule {t} on {d}, please."][f], {"title": t, "date": d}, {"ok": True}, f"Added '{t}' on {d}."))(
                         r.choice(["Dentist", "Team sync", "Code review"]), f"2026-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}")),
    "send_email": (fn("send_email", "Send an email.", to=("string", "recipient"), subject=("string", "subject"), body=("string", "body")),
                   lambda r, f: (lambda a: ([f"Email {a} that the build passed.", f"Send {a} a note saying the deploy is done."][f], {"to": a, "subject": "Update", "body": "See subject."},
                                            {"sent": True}, f"The email to {a} was sent."))(r.choice(["ana@example.com", "sam@example.org"]))),
    "calculator": (fn("calculator", "Evaluate an arithmetic expression exactly.", expression=("string", "expression")),
                   lambda r, f: (lambda a, b: ([f"Use the calculator for {a} * {b}.", f"Compute {a} times {b} exactly."][f], {"expression": f"{a}*{b}"}, {"value": a * b}, f"{a} × {b} = {a * b}."))(
                       r.randint(100, 9999), r.randint(100, 9999))),
    "translate": (fn("translate", "Translate text into a target language.", text=("string", "text"), target=("string", "language")),
                  lambda r, f: (lambda t, l: ([f"Translate '{t}' into {l}.", f"Put '{t}' in {l} using the translator."][f], {"text": t, "target": l}, {"translation": "…"}, "Here is the translation."))(
                      r.choice(["good morning", "where is the station"]), r.choice(["French", "Japanese", "German"]))),
}
REF_TOOLS = {"run_tests", "dns_lookup", "log_query", "get_weather", "translate"}           # never seen in calibration
CAL_TOOLS = set(T) - REF_TOOLS
NO_TOOL = [["What is the capital of Australia?", "Explain recursion in one sentence.", "What does HTTP status 403 mean?", "What is the difference between a process and a thread?"],
           ["Write a haiku about autumn.", "What is 12 plus 30?", "What does the acronym CSRF stand for?", "In one sentence, what is a race condition?"]]
NO_TOOL_ANS = "Here is the answer, no tool needed: "
REASON = ["The user asks for something I cannot know without the {n} tool, so I will call it with the right arguments.",
          "I should not guess. The {n} tool gives the actual result; I will call it and then report.",
          "This needs {n}. Arguments are clear from the request."]
REASON_ANS = ["The tool returned what I need. I will report it plainly.", "I have the result now; a short answer is enough."]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--corpus", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=610_000); ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    r = random.Random(a.seed)
    TC, ENDT = tok.convert_tokens_to_ids("<tool_call>"), tok.convert_tokens_to_ids("</think>")
    assert isinstance(TC, int) and isinstance(ENDT, int) and TC != tok.unk_token_id, (TC, ENDT)
    ids_of = lambda text: list(tok(text, add_special_tokens=False)["input_ids"])
    chat = lambda text: ids_of(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=False))
    docs = {}
    for l in open(a.corpus):
        d = json.loads(l); docs.setdefault(d.get("domain"), []).append(d["text"])
    for v in docs.values():
        r.shuffle(v)
    rows = []

    def doc_block(dom, lo):
        text = ""
        while len(ids_of(text)) < lo:
            text += docs[dom].pop(0)[:14000] + "\n\n"
        return text
    # ---- held-out KL reference FIRST (records never reused)
    for dom, n in KLREF:
        for _ in range(n):
            rows.append({"ids": chat(doc_block(dom, 1500))[:2048], "ref": True, "set": "klref", "domain": dom, "decisions": []})
    for i in range(6):
        dom = ("coding", "agentic", "longctx")[i % 3]
        pool = dom if docs.get(dom) else "coding"
        rows.append({"ids": chat(doc_block(pool, 4200 + 400 * i))[: 4096 + 400 * i], "ref": True, "set": "klref", "domain": "long_" + dom, "decisions": []})
    # ---- calibration documents
    total = 0
    for dom, frac in QUOTA.items():
        got = 0
        while got < int(a.tokens * frac) and docs.get(dom):
            ids = chat(docs[dom].pop(0)[:14000])[:2048]
            if len(ids) >= 64:
                rows.append({"ids": ids, "ref": False, "set": "calib", "domain": dom, "decisions": []}); got += len(ids)
        total += got
    got = 0
    while got < int(a.tokens * LONG_FRAC):
        dom = r.choice(["coding", "agentic", "cybersec", "longctx"])
        pool = dom if len(docs.get(dom, [])) > 40 else "coding"
        ids = chat(doc_block(pool, 6000))[: r.randint(4096, 8192)]
        rows.append({"ids": ids, "ref": False, "set": "calib", "domain": "long_" + dom, "decisions": []}); got += len(ids)
    total += got

    # ---- tool conversations
    IMS = tok.convert_tokens_to_ids("<|im_start|>"); ASSIST = ids_of("assistant")
    assert isinstance(IMS, int) and len(ASSIST) == 1, (IMS, ASSIST)

    def decisions(ids):
        # only "</think>" inside an assistant turn counts: the tool instructions in the system turn contain the literal too
        out, in_asst = [], False
        for p in range(len(ids) - 1):
            if ids[p] == IMS:
                in_asst = ids[p + 1] == ASSIST[0]
            elif ids[p] == ENDT and in_asst:
                out.append([p, "call" if ids[p + 1] == TC else "answer"])
        assert out and out[-1][1] == "answer", "every conversation must end with an answer decision"
        return out

    def conv(pool, fam):
        present = r.sample(sorted(pool), min(r.randint(2, 6), len(pool)))
        tools = [T[t][0] for t in present]
        think = r.random() < 0.3
        kind = r.random()
        if kind < 0.2:
            q = r.choice(NO_TOOL[fam])
            msgs = [{"role": "user", "content": q}, {"role": "assistant", "content": NO_TOOL_ANS + q.lower()}]
        else:
            steps = [T[n][1](r, fam) + (n,) for n in (r.sample(present, 2) if kind > 0.65 and len(present) > 1 else [r.choice(present)])]
            user = steps[0][0] if len(steps) == 1 else steps[0][0] + " Then: " + steps[1][0][0].lower() + steps[1][0][1:]
            msgs = [{"role": "user", "content": user}]
            for i, (_, args, res, _, n) in enumerate(steps):
                m = {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": args}}]}
                if think:
                    m["reasoning_content"] = r.choice(REASON).format(n=n)
                msgs += [m, {"role": "tool", "tool_call_id": f"c{i}", "content": json.dumps(res)}]
            m = {"role": "assistant", "content": " ".join(s[3] for s in steps)}
            if think:
                m["reasoning_content"] = r.choice(REASON_ANS)
            msgs.append(m)
        ids = ids_of(tok.apply_chat_template(msgs, tools=tools, tokenize=False, reasoning_effort=r.choice(["low", "high", "max"])))
        return ids if len(ids) <= 2048 else None
    got = 0
    while got < int(a.tokens * TOOL_FRAC):
        ids = conv(CAL_TOOLS, 0)
        if ids:
            rows.append({"ids": ids, "ref": False, "set": "calib", "domain": "tool_conv", "decisions": decisions(ids)}); got += len(ids)
    total += got
    n_ref = 0
    while n_ref < 96:
        ids = conv(REF_TOOLS, 1)
        if ids:
            rows.append({"ids": ids, "ref": True, "set": "agentic_ref", "domain": "tool_conv", "decisions": decisions(ids)}); n_ref += 1
    with open(a.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    by = {}
    for x in rows:
        k = (x["set"], x["domain"]); by[k] = by.get(k, 0) + len(x["ids"])
    for k in sorted(by):
        print(f"{k[0]:12s} {k[1]:18s} {by[k]:8d}")
    dec = [d for x in rows if x["set"] == "agentic_ref" for d in x["decisions"]]
    kl = [x for x in rows if x["set"] == "klref"]
    print(f"TOTAL calib tokens {total} in {sum(1 for x in rows if x['set']=='calib')} sequences; klref {len(kl)} prompts {sum(len(x['ids']) for x in kl)} positions; "
          f"agentic_ref {n_ref} conversations, decisions call {sum(d[1]=='call' for d in dec)} answer {sum(d[1]=='answer' for d in dec)}; "
          f"held-out tools {sorted(REF_TOOLS)}")


if __name__ == "__main__":
    main()
