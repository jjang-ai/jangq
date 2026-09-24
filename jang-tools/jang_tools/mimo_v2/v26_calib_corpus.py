"""MiMo-V2.6-Flash-RL calibration corpus + disjoint held-out KL-eval corpus.

Every chat-shaped document is rendered through the model's REAL
``chat_template.jinja`` (``tokenizer.apply_chat_template``, trust_remote_code
off) so the calibration activations see the exact formats the model was
RL-trained on:

* reasoning turns   ``<|im_start|>assistant\\n<think>COT</think>ANSWER<|im_end|>``
* plain chat turns  ``<|im_start|>assistant\\n<think></think>ANSWER<|im_end|>``
* tool calls        ``<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>``
  with the ``tools=[...]`` system block and ``tool`` role results
* multi-turn rows, including prior assistant turns carrying reasoning_content

Raw text (source files, books, arXiv) is tokenized without a template.

Outputs (``--out``):
  calib_tokens.npy   int32 [calib_seqs, seq_len]
  klref_tokens.npy   int32 [klref_seqs, seq_len]
  corpus_meta.json   per-domain/per-source token + row accounting, dataset
                     revisions, sha256 of both npy files, skipped sources
  samples.txt        decoded samples per domain for inspection

Sequences are domain-homogeneous. Chat documents are packed whole (each
starts at a turn boundary, each keeps its assistant turn); a document that is
longer than one sequence is cut back to the longest message prefix that ends
on a complete assistant turn. A residual gap is filled with the head of an
unused document only when that head contains >= MIN_FRAG_ASSISTANT tokens of
an assistant turn (code domain: head of a raw source file); otherwise it is
padded with ``<|endoftext|>`` and counted as pad in the meta.

Held-out rows are disjoint from calibration rows: different sources wherever
possible, disjoint row ranges where a source is shared (asserted), and a
global content-hash exclusion.

No benchmark/eval splits are used; benchmark-derived subsets inside mixed
datasets (GSM8K / MATH / AMC-AIME / APPS rows) are filtered out.

Usage (on the machine with the tokenizer):
  cd ~/jang/jang-tools && HF_HUB_DISABLE_XET=1 uv run python -m \\
      jang_tools.mimo_v2.v26_calib_corpus \\
      --src /path/to/MiMo-V2.6-Flash-RL \\
      --out ~/models/mimo26-build/calib
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

SEED = 42
SEQ_LEN = 2048
MIN_FRAG_ASSISTANT = 64
OVERFILL = 2.0           # pool size vs. bin capacity, per source quota

DOMAIN_MIX = {           # share of tokens (== share of sequences)
    "coding": 0.30,
    "agentic": 0.20,
    "reasoning": 0.15,
    "chat": 0.12,
    "chinese": 0.10,
    "prose": 0.08,
    "cybersec": 0.05,
}

# Benchmark-derived subsets to drop from mixed datasets.
BENCH_SOURCES = {"gsm8k", "math", "amc_aime", "apps", "aime", "humaneval",
                 "mbpp", "mmlu", "gpqa", "livecodebench", "swe-bench"}

_CJK_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿]")


# --------------------------------------------------------------------------
# document model
# --------------------------------------------------------------------------

@dataclass
class Doc:
    messages: Optional[list] = None
    tools: Optional[list] = None
    raw: Optional[str] = None
    tags: set = field(default_factory=set)


def _frac_cjk(s: str) -> float:
    total = sum(1 for c in s if not c.isspace())
    return len(_CJK_RE.findall(s)) / total if total else 0.0


def _msg_text(doc: Doc) -> str:
    if doc.raw is not None:
        return doc.raw
    return "\n".join(str(m.get("content") or "") + str(m.get("reasoning_content") or "")
                     for m in doc.messages)


def _asst(content: str, reasoning: str = "", tool_calls=None) -> dict:
    m = {"role": "assistant", "content": content, "reasoning_content": reasoning}
    if tool_calls:
        m["tool_calls"] = tool_calls
    return m


_THINK_RE = re.compile(r"^\s*<think>(.*?)</think>(.*)$", re.S)


def _split_think(text: str) -> tuple[str, str] | None:
    m = _THINK_RE.match(text or "")
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip()


def _loads_loose(s: str):
    s = s.strip()
    try:
        return json.loads(s)
    except ValueError:
        return ast.literal_eval(s)


def _norm_call(obj) -> dict | None:
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
        return None
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = _loads_loose(args) if args.strip() else {}
        except (ValueError, SyntaxError, TypeError):
            return None
    if not isinstance(args, dict):
        return None
    return {"type": "function", "function": {"name": obj["name"], "arguments": args}}


def _norm_tool_schema(t) -> dict | None:
    if not isinstance(t, dict):
        return None
    if t.get("type") == "function" and isinstance(t.get("function"), dict):
        return t
    if "name" not in t:
        return None
    params = t.get("parameters") or {}
    if isinstance(params, dict) and params.get("type") != "object":
        # xLAM style {arg: {type, description}} -> JSON schema object
        params = {"type": "object", "properties": params}
    return {"type": "function",
            "function": {"name": t["name"], "description": t.get("description", ""),
                         "parameters": params}}


# --------------------------------------------------------------------------
# row converters  (row, rng) -> Doc | None
# --------------------------------------------------------------------------

def conv_magicoder(r, rng):
    p, s = r.get("problem") or "", r.get("solution") or ""
    if len(p) < 40 or len(s) < 40:
        return None
    return Doc([{"role": "user", "content": p.strip()}, _asst(s.strip())])


def conv_ocr(r, rng):
    if (r.get("split") or "train") != "train":
        return None
    if str(r.get("dataset") or "").lower() in BENCH_SOURCES:
        return None
    sp = _split_think(r.get("output") or "")
    if not sp or len(sp[0]) < 50 or len(sp[1]) < 20:
        return None
    return Doc([{"role": "user", "content": r["input"].strip()}, _asst(sp[1], sp[0])],
               tags={"thinking"})


def conv_codefeedback(r, rng):
    q, a = r.get("query") or "", r.get("answer") or ""
    if len(q) < 30 or len(a) < 40:
        return None
    return Doc([{"role": "user", "content": q.strip()}, _asst(a.strip())])


def conv_code_file(r, rng):
    if str(r.get("autogenerated")).lower() == "true":
        return None
    c = r.get("content") or ""
    try:
        if float(r.get("alpha_frac") or 0) < 0.3 or float(r.get("line_max") or 0) > 200:
            return None
    except (TypeError, ValueError, OverflowError):
        # Malformed quality metadata cannot establish corpus eligibility.
        return None
    if len(c) < 400:
        return None
    return Doc(raw=c)


_TC_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_TR_RE = re.compile(r"^\s*<tool_response>\s*(.*?)\s*</tool_response>\s*$", re.S)


def _hermes_conv(r, rng):
    """NousResearch hermes-function-calling / interstellarninja reasoning tool use.

    The dataset's own Hermes-format system prompt is dropped; the schemas go
    through ``tools=`` so the MiMo template renders its own tools block.
    """
    try:
        tools_raw = r.get("tools")
        tools = _loads_loose(tools_raw) if isinstance(tools_raw, str) else tools_raw
    except (ValueError, SyntaxError, TypeError, AttributeError):
        return None
    if not isinstance(tools, list) or not tools:
        return None
    tools = [_norm_tool_schema(t) for t in tools]
    if any(t is None for t in tools):
        return None
    msgs, n_calls, has_think = [], 0, False
    for turn in r.get("conversations") or []:
        role, val = turn.get("from"), turn.get("value") or ""
        if role == "system":
            continue
        if role == "human":
            msgs.append({"role": "user", "content": val.strip()})
        elif role == "gpt":
            reasoning = ""
            sp = _split_think(val)
            if sp:
                reasoning, val = sp
                has_think = True
            calls = []
            for blob in _TC_RE.findall(val):
                try:
                    c = _norm_call(_loads_loose(blob))
                except (ValueError, SyntaxError, TypeError):
                    return None
                if c is None:
                    return None
                calls.append(c)
            content = _TC_RE.sub("", val).strip()
            if "<tool_call>" in content or "</tool_call>" in content:
                return None
            n_calls += len(calls)
            msgs.append(_asst(content, reasoning, calls))
        elif role == "tool":
            m = _TR_RE.match(val)
            msgs.append({"role": "tool", "content": (m.group(1) if m else val).strip()})
        else:
            return None
    if n_calls == 0 or not any(m["role"] == "assistant" for m in msgs):
        return None
    tags = {"tool_call"} | ({"thinking"} if has_think else set())
    return Doc(msgs, tools=tools, tags=tags)


_GLAIVE_SPLIT = re.compile(r"(USER:|ASSISTANT:|FUNCTION RESPONSE:)")


def conv_glaive(r, rng):
    sysmsg, chat = r.get("system") or "", r.get("chat") or ""
    if "<functioncall>" not in chat:
        return None
    # tool schemas: concatenated JSON objects after the preamble
    i = sysmsg.find("{")
    if i < 0:
        return None
    dec, body, tools = json.JSONDecoder(), sysmsg[i:], []
    while body.strip():
        body = body.strip()
        try:
            obj, end = dec.raw_decode(body)
        except ValueError:
            return None
        t = _norm_tool_schema(obj)
        if t is None:
            return None
        tools.append(t)
        body = body[end:]
    parts = _GLAIVE_SPLIT.split(chat)
    msgs = []
    for tag, text in zip(parts[1::2], parts[2::2]):
        text = text.replace("<|endoftext|>", "").strip()
        if tag == "USER:":
            msgs.append({"role": "user", "content": text})
        elif tag == "FUNCTION RESPONSE:":
            msgs.append({"role": "tool", "content": text})
        else:
            if text.startswith("<functioncall>"):
                blob = text[len("<functioncall>"):].strip()
                m = re.match(r'^\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*\'(.*)\'\s*\}$', blob, re.S)
                try:
                    if m:
                        c = _norm_call({"name": m.group(1), "arguments": m.group(2)})
                    else:
                        c = _norm_call(_loads_loose(blob))
                except (ValueError, SyntaxError, TypeError):
                    return None
                if c is None:
                    return None
                msgs.append(_asst("", "", [c]))
            else:
                msgs.append(_asst(text))
    if not msgs or msgs[0]["role"] != "user":
        return None
    return Doc(msgs, tools=tools, tags={"tool_call"})


def _last_boxed(s: str) -> str | None:
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    k = j
    while k < len(s) and depth:
        depth += {"{": 1, "}": -1}.get(s[k], 0)
        k += 1
    return s[j:k - 1] if depth == 0 else None


def conv_numina(r, rng):
    src = str(r.get("source") or "").lower()
    if src in BENCH_SOURCES:
        return None
    p, s = (r.get("problem") or "").strip(), (r.get("solution") or "").strip()
    if len(p) < 20 or len(s) < 80:
        return None
    ans = _last_boxed(s)
    if ans is None:
        return None
    content = f"The final answer is $\\boxed{{{ans}}}$."
    return Doc([{"role": "user", "content": p}, _asst(content, s)], tags={"thinking"})


def conv_openthoughts(r, rng):
    if str(r.get("source") or "").lower() in BENCH_SOURCES:
        return None
    p = (r.get("problem") or "").strip()
    rs, sol = (r.get("deepseek_reasoning") or "").strip(), (r.get("deepseek_solution") or "").strip()
    if len(p) < 20 or len(rs) < 100 or len(sol) < 20:
        return None
    return Doc([{"role": "user", "content": p}, _asst(sol, rs)], tags={"thinking"})


def conv_openr1(r, rng):
    if str(r.get("source") or "").lower() in BENCH_SOURCES:
        return None
    gens = r.get("generations") or []
    ok = r.get("correctness_math_verify") or []
    for g, v in zip(gens, ok):
        if v:
            sp = _split_think(g)
            if sp and len(sp[0]) > 100 and len(sp[1]) > 20:
                return Doc([{"role": "user", "content": r["problem"].strip()},
                            _asst(sp[1], sp[0])], tags={"thinking"})
    return None


def _sharegpt(turns, keep_system=True):
    msgs = []
    for t in turns or []:
        if isinstance(t, str):          # some revisions store each turn as a JSON string
            t = json.loads(t)
        role = t.get("from") or t.get("role")
        val = (t.get("value") if "value" in t else t.get("content")) or ""
        role = {"human": "user", "gpt": "assistant", "user": "user",
                "assistant": "assistant", "system": "system"}.get(role)
        if role is None:
            return None
        if role == "system":
            if keep_system and val.strip():
                msgs.append({"role": "system", "content": val.strip()})
            continue
        msgs.append(_asst(val.strip()) if role == "assistant"
                    else {"role": "user", "content": val.strip()})
    if not any(m["role"] == "assistant" for m in msgs):
        return None
    return msgs


def conv_ultrachat(r, rng):
    msgs = _sharegpt(r.get("messages"))
    if not msgs or len(msgs) < 2:
        return None
    return Doc(msgs, tags={"multiturn"} if len(msgs) >= 3 else set())


def conv_openhermes(r, rng):
    msgs = _sharegpt(r.get("conversations"))
    if not msgs:
        return None
    return Doc(msgs, tags={"multiturn"} if len(msgs) >= 3 else set())


def _zh_doc(user: str, answer: str) -> Doc | None:
    user, answer = user.strip(), answer.strip()
    if len(user) < 4 or len(answer) < 30:
        return None
    if _frac_cjk(user + answer) <= 0.5:
        return None
    return Doc([{"role": "user", "content": user}, _asst(answer)], tags={"cjk"})


def conv_firefly(r, rng):
    return _zh_doc(r.get("input") or "", r.get("target") or "")


def conv_silkroad(r, rng):
    ins, inp = r.get("instruction_zh") or "", r.get("input_zh") or ""
    user = ins + ("\n\n" + inp if inp.strip() else "")
    return _zh_doc(user, r.get("output_zh") or "")


def conv_zhihu(r, rng):
    return _zh_doc(r.get("INSTRUCTION") or "", r.get("RESPONSE") or "")


def conv_pg19(r, rng):
    t = r.get("text") or ""
    if len(t) < 60_000:
        return None
    start = rng.randint(len(t) // 10, len(t) - 30_000)
    nl = t.find("\n\n", start)
    start = nl + 2 if 0 <= nl < start + 5_000 else start
    return Doc(raw=t[start:start + 24_000])


def conv_arxiv(r, rng):
    a = (r.get("article") or "").strip()
    if len(a) < 6_000:
        return None
    return Doc(raw=a[:24_000])


def conv_trendyol(r, rng):
    u, a = r.get("user") or "", r.get("assistant") or ""
    if len(u) < 20 or len(a) < 80:
        return None
    return Doc([{"role": "user", "content": u.strip()}, _asst(a.strip())])


def conv_cybernative(r, rng):
    q, a = r.get("question") or "", r.get("chosen") or ""
    if len(q) < 20 or len(a) < 40:
        return None
    return Doc([{"role": "user", "content": q.strip()}, _asst(a.strip())])


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------

@dataclass
class Src:
    key: str
    role: str                 # "calib" | "klref"
    domain: str
    weight: float             # share of the domain's tokens within the role
    dataset: str
    config: Optional[str]
    split: str
    conv: Callable
    skip: int = 0             # rows skipped before use (disjoint ranges)
    scan_cap: int = 20_000
    chain_p: float = 0.0      # prob. of chaining into a multi-turn doc


SOURCES: list[Src] = [
    # ---------------- calibration ----------------
    Src("magicoder", "calib", "coding", 0.45, "ise-uiuc/Magicoder-OSS-Instruct-75K", None, "train", conv_magicoder),
    Src("ocr", "calib", "coding", 0.30, "nvidia/OpenCodeReasoning", "split_0", "split_0", conv_ocr, scan_cap=6_000),
    Src("codeparrot", "calib", "coding", 0.25, "codeparrot/codeparrot-clean-valid", None, "train", conv_code_file, scan_cap=4_000),
    Src("hermes_fc", "calib", "agentic", 0.55, "NousResearch/hermes-function-calling-v1", "func_calling", "train", _hermes_conv),
    Src("hermes_reason_tools", "calib", "agentic", 0.45, "interstellarninja/hermes_reasoning_tool_use", None, "train", _hermes_conv),
    Src("openthoughts", "calib", "reasoning", 0.50, "open-thoughts/OpenThoughts-114k", "metadata", "train", conv_openthoughts, scan_cap=8_000),
    Src("numina", "calib", "reasoning", 0.50, "AI-MO/NuminaMath-CoT", None, "train", conv_numina, chain_p=0.4),
    Src("ultrachat", "calib", "chat", 1.00, "HuggingFaceH4/ultrachat_200k", None, "train_sft", conv_ultrachat),
    Src("firefly", "calib", "chinese", 0.50, "YeungNLP/firefly-train-1.1M", None, "train", conv_firefly),
    Src("silkroad_zh", "calib", "chinese", 0.50, "silk-road/alpaca-data-gpt4-chinese", None, "train", conv_silkroad),
    Src("pg19", "calib", "prose", 0.50, "emozilla/pg19", None, "train", conv_pg19, scan_cap=200),
    Src("arxiv", "calib", "prose", 0.50, "ccdv/arxiv-summarization", None, "train", conv_arxiv, scan_cap=500),
    Src("trendyol_cyber", "calib", "cybersec", 0.50, "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset", None, "train", conv_trendyol),
    Src("cybernative", "calib", "cybersec", 0.50, "CyberNative/Code_Vulnerability_Security_DPO", None, "train", conv_cybernative, scan_cap=3_000),
    # ---------------- held-out KL reference ----------------
    Src("codefeedback", "klref", "coding", 0.60, "m-a-p/CodeFeedback-Filtered-Instruction", None, "train", conv_codefeedback),
    Src("codeparrot_ho", "klref", "coding", 0.40, "codeparrot/codeparrot-clean-valid", None, "train", conv_code_file, skip=20_000, scan_cap=2_000),
    Src("glaive_fc", "klref", "agentic", 1.00, "glaiveai/glaive-function-calling-v2", None, "train", conv_glaive),
    Src("openr1_math", "klref", "reasoning", 1.00, "open-r1/OpenR1-Math-220k", "default", "train", conv_openr1, scan_cap=8_000, chain_p=0.3),
    Src("openhermes", "klref", "chat", 1.00, "teknium/OpenHermes-2.5", None, "train", conv_openhermes),
    Src("zhihu", "klref", "chinese", 1.00, "wangrui6/Zhihu-KOL", None, "train", conv_zhihu),
    Src("arxiv_ho", "klref", "prose", 1.00, "ccdv/arxiv-summarization", None, "train", conv_arxiv, skip=20_000, scan_cap=200),
    Src("cybernative_ho", "klref", "cybersec", 1.00, "CyberNative/Code_Vulnerability_Security_DPO", None, "train", conv_cybernative, skip=3_000, scan_cap=1_500),
]


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

class Renderer:
    def __init__(self, src_dir: Path):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(str(src_dir), trust_remote_code=False)
        self.template = (src_dir / "chat_template.jinja").read_text()
        self.template_sha = hashlib.sha256(self.template.encode()).hexdigest()
        self.pad_id = self.tok.convert_tokens_to_ids("<|endoftext|>")
        self.asst_ids = self.tok.encode("<|im_start|>assistant\n", add_special_tokens=False)

    def text(self, messages, tools=None) -> str:
        return self.tok.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=False,
            enable_thinking=True, chat_template=self.template)

    def ids(self, s: str) -> list[int]:
        return self.tok.encode(s, add_special_tokens=False)

    def render(self, doc: Doc, seq_len: int) -> list[int] | None:
        """Chat doc -> ids, cut back to the longest prefix ending on a complete
        assistant turn when longer than seq_len. Raw doc -> ids (untruncated)."""
        if doc.raw is not None:
            return self.ids(doc.raw)
        ids = self.ids(self.text(doc.messages, doc.tools))
        if len(ids) <= seq_len:
            return ids
        ends = [i for i, m in enumerate(doc.messages) if m["role"] == "assistant"]
        for e in reversed(ends[:-1]):
            ids = self.ids(self.text(doc.messages[:e + 1], doc.tools))
            if len(ids) <= seq_len:
                doc.messages = doc.messages[:e + 1]
                doc.tags.add("turn_truncated")
                return ids
        return None

    def assistant_offset(self, ids: list[int]) -> int:
        n = len(self.asst_ids)
        for i in range(len(ids) - n + 1):
            if ids[i:i + n] == self.asst_ids:
                return i + n
        return -1


def _chain(docs: list[Doc]) -> Doc:
    msgs = []
    for d in docs:
        msgs.extend(d.messages)
    tags = set().union(*(d.tags for d in docs)) | {"multiturn", "multiturn_prior_reasoning"}
    return Doc(msgs, tags=tags)


# --------------------------------------------------------------------------
# collection
# --------------------------------------------------------------------------

@dataclass
class Item:
    ids: list
    raw: bool
    src: str
    row: int
    tags: set


def _dataset_sha(repo: str) -> str | None:
    try:
        from huggingface_hub import HfApi
        return HfApi().dataset_info(repo).sha
    except Exception:
        return None


def collect(src: Src, quota: int, rend: Renderer, seq_len: int, rng: random.Random,
            seen: set, log) -> tuple[list[Item], dict]:
    from datasets import load_dataset
    sha = _dataset_sha(src.dataset)
    kw = dict(split=src.split, streaming=True)
    if sha:
        kw["revision"] = sha
    ds = load_dataset(src.dataset, src.config, **kw) if src.config else load_dataset(src.dataset, **kw)
    if src.skip:
        ds = ds.skip(src.skip)
    items, toks, scanned, rows_used, rej = [], 0, 0, [], 0
    pending: list[tuple[Doc, int]] = []
    t0 = time.time()
    for i, row in enumerate(ds):
        scanned += 1
        if scanned > src.scan_cap or toks >= quota:
            break
        try:
            doc = src.conv(row, rng)
        except Exception:
            doc = None
        if doc is None:
            rej += 1
            continue
        rowi = src.skip + i
        if src.chain_p and doc.messages and (pending or rng.random() < src.chain_p):
            pending.append((doc, rowi))
            if len(pending) < rng.choice((2, 3)):
                continue
            doc = _chain([d for d, _ in pending])
            rowi = pending[0][1]
            pending = []
        ids = rend.render(doc, seq_len)
        if ids is None or len(ids) < 32:
            rej += 1
            continue
        h = hashlib.blake2b(np.asarray(ids[:4096], dtype=np.int32).tobytes(), digest_size=12).hexdigest()
        if h in seen:
            rej += 1
            continue
        seen.add(h)
        items.append(Item(ids, doc.raw is not None, src.key, rowi, set(doc.tags)))
        rows_used.append(rowi)
        toks += min(len(ids), seq_len) if doc.raw is not None else len(ids)
    info = {
        "dataset": src.dataset, "config": src.config, "split": src.split,
        "revision": sha, "role": src.role, "domain": src.domain,
        "rows_scanned": scanned, "rows_rejected": rej, "docs": len(items),
        "row_index_min": min(rows_used) if rows_used else None,
        "row_index_max": max(rows_used) if rows_used else None,
        "pool_tokens": toks, "quota": quota, "seconds": round(time.time() - t0, 1),
    }
    log(f"  [{src.role}:{src.domain}] {src.key}: {len(items)} docs, {toks} tok "
        f"(quota {quota}), scanned {scanned}, rejected {rej}, {info['seconds']}s")
    return items, info


# --------------------------------------------------------------------------
# packing
# --------------------------------------------------------------------------

def pack(items: list[Item], n_bins: int, seq_len: int, rend: Renderer,
         rng: random.Random) -> tuple[list[list[Item]], dict]:
    """Best-fit pack whole docs into n_bins sequences; returns bins of
    (Item, used_len) fragments plus accounting."""
    bins: list[list[tuple[Item, int]]] = [[] for _ in range(n_bins)]
    free = [seq_len] * n_bins
    order = items[:]
    rng.shuffle(order)
    raw_items = [it for it in order if it.raw]
    chat_items = [it for it in order if not it.raw]
    used = set()

    # prose-style raw domains (no chat items): one window per bin
    if not chat_items:
        for b in range(n_bins):
            for k, it in enumerate(raw_items):
                if id(it) in used:
                    continue
                take = min(free[b], len(it.ids))
                bins[b].append((it, take))
                free[b] -= take
                used.add(id(it))
                if free[b] == 0:
                    break
    else:
        # raw items in a mixed domain are capped to a random window <= seq_len
        mix = chat_items + raw_items
        rng.shuffle(mix)
        for it in mix:
            L = min(len(it.ids), seq_len) if it.raw else len(it.ids)
            if it.raw and L > seq_len // 2:
                L = rng.randint(seq_len // 4, seq_len // 2)   # keep raw files from hogging bins
            best, best_free = -1, seq_len + 1
            for b in range(n_bins):
                if L <= free[b] < best_free:
                    best, best_free = b, free[b]
            if best < 0:
                continue
            bins[best].append((it, L))
            free[best] -= L
            used.add(id(it))
        # gap fill: raw head (any length) or chat head with >= MIN_FRAG_ASSISTANT asst tokens
        for b in range(n_bins):
            if free[b] == 0:
                continue
            for it in order:
                if id(it) in used:
                    continue
                g = free[b]
                if it.raw:
                    take = min(g, len(it.ids))
                else:
                    off = rend.assistant_offset(it.ids)
                    if off < 0 or g - off < MIN_FRAG_ASSISTANT or len(it.ids) <= g:
                        continue
                    take = g
                bins[b].append((it, take))
                free[b] -= take
                used.add(id(it))
                if free[b] == 0:
                    break
    for b in bins:
        rng.shuffle(b)
    stats = {"pad_tokens": int(sum(free)),
             "fragments": int(sum(1 for b in bins for it, L in b if L < len(it.ids)))}
    return bins, stats


# --------------------------------------------------------------------------
# main build
# --------------------------------------------------------------------------

def _alloc(total: int, mix: dict[str, float]) -> dict[str, int]:
    raw = {d: total * w for d, w in mix.items()}
    out = {d: int(v) for d, v in raw.items()}
    rest = total - sum(out.values())
    for d in sorted(raw, key=lambda d: (-round(raw[d] - int(raw[d]), 6), -mix[d]))[:rest]:
        out[d] += 1
    return out


def build_role(role: str, n_seq: int, seq_len: int, rend: Renderer, seen: set,
               rng: random.Random, log, meta_sources: dict, skipped: list):
    alloc = _alloc(n_seq, DOMAIN_MIX)
    log(f"[{role}] sequence allocation: {alloc}")
    domain_items: dict[str, list[Item]] = {}
    for dom in DOMAIN_MIX:
        need = alloc[dom] * seq_len
        srcs = [s for s in SOURCES if s.role == role and s.domain == dom]
        items: list[Item] = []
        ok_srcs = []
        for s in srcs:
            quota = int(need * s.weight * OVERFILL)
            try:
                its, info = collect(s, quota, rend, seq_len, random.Random(f"{SEED}:{s.key}"), seen, log)
                items += its
                meta_sources[s.key] = info
                ok_srcs.append(s)
            except Exception as e:
                msg = f"{type(e).__name__}: {str(e)[:300]}"
                log(f"  !! SKIP {s.key} ({s.dataset}): {msg}")
                skipped.append({"key": s.key, "dataset": s.dataset, "role": role, "error": msg})
        have = sum(min(len(i.ids), seq_len) for i in items)
        if have < need and ok_srcs:
            # rebalance: top up from the surviving sources of this domain
            deficit = int((need - have) * OVERFILL) + seq_len
            log(f"  [{role}:{dom}] short {need - have} tok, rebalancing over {[s.key for s in ok_srcs]}")
            for s in ok_srcs:
                s2 = Src(**{**s.__dict__, "scan_cap": s.scan_cap * 3})
                its, info = collect(s2, meta_sources[s.key]["pool_tokens"] + deficit // len(ok_srcs),
                                    rend, seq_len, random.Random(f"{SEED}:{s.key}:topup"), seen, log)
                items += its
                meta_sources[s.key]["topup"] = info
        domain_items[dom] = items
    # domains with nothing at all: hand their sequences to the largest domains
    dead = [d for d, its in domain_items.items() if not its]
    for d in dead:
        log(f"  !! domain {d} has no data in {role}; redistributing {alloc[d]} seqs")
        live = [x for x in DOMAIN_MIX if domain_items[x]]
        for k in range(alloc[d]):
            alloc[live[k % len(live)]] += 1
        alloc[d] = 0

    seqs, seq_domain, dom_meta = [], [], {}
    for dom in DOMAIN_MIX:
        if not alloc[dom]:
            continue
        bins, st = pack(domain_items[dom], alloc[dom], seq_len, rend,
                        random.Random(f"{SEED}:{role}:{dom}:pack"))
        per_src: dict[str, int] = {}
        tags: dict[str, int] = {}
        for b in bins:
            arr: list[int] = []
            for it, L in b:
                arr.extend(it.ids[:L])
                per_src[it.src] = per_src.get(it.src, 0) + L
                for t in it.tags:
                    tags[t] = tags.get(t, 0) + 1
            arr.extend([rend.pad_id] * (seq_len - len(arr)))
            assert len(arr) == seq_len
            seqs.append(arr)
            seq_domain.append(dom)
        dom_meta[dom] = {"sequences": alloc[dom],
                         "tokens_total": alloc[dom] * seq_len,
                         "tokens_real": alloc[dom] * seq_len - st["pad_tokens"],
                         "pad_tokens": st["pad_tokens"], "fragments": st["fragments"],
                         "docs": sum(len(b) for b in bins),
                         "tokens_by_source": per_src, "doc_tags": tags}
        log(f"  [{role}:{dom}] {alloc[dom]} seqs, real {dom_meta[dom]['tokens_real']}, "
            f"pad {st['pad_tokens']}, frags {st['fragments']}, by source {per_src}, tags {tags}")
    return np.asarray(seqs, dtype=np.int32), seq_domain, dom_meta


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, required=True,
                    help="Local source model directory containing the tokenizer")
    ap.add_argument("--out", type=Path, default=Path.home() / "models/mimo26-build/calib")
    ap.add_argument("--calib-seqs", type=int, default=128)
    ap.add_argument("--klref-seqs", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    logf = (args.out / "build.log").open("w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    random.seed(SEED)
    np.random.seed(SEED)
    rend = Renderer(args.src)
    log(f"tokenizer {type(rend.tok).__name__} vocab={len(rend.tok)} pad={rend.pad_id} "
        f"template_sha256={rend.template_sha}")
    seen: set = set()
    meta_sources: dict = {}
    skipped: list = []

    calib, calib_dom, calib_meta = build_role("calib", args.calib_seqs, args.seq_len, rend, seen,
                                              random.Random(SEED), log, meta_sources, skipped)
    klref, klref_dom, klref_meta = build_role("klref", args.klref_seqs, args.seq_len, rend, seen,
                                              random.Random(SEED + 1), log, meta_sources, skipped)

    # disjointness: shared datasets must use non-overlapping row ranges
    def _rng(info):
        xs = [info, info.get("topup") or {}]
        lo = [x["row_index_min"] for x in xs if x.get("row_index_min") is not None]
        hi = [x["row_index_max"] for x in xs if x.get("row_index_max") is not None]
        return (min(lo), max(hi)) if lo else None
    for ka, a in meta_sources.items():
        for kb, b in meta_sources.items():
            if a["role"] != "calib" or b["role"] != "klref":
                continue
            if (a["dataset"], a["config"], a["split"]) != (b["dataset"], b["config"], b["split"]):
                continue
            ra, rb = _rng(a), _rng(b)
            assert not (ra and rb) or ra[1] < rb[0] or rb[1] < ra[0], f"row overlap {ka} {ra} vs {kb} {rb}"
            log(f"disjoint rows OK: {ka} {ra} vs {kb} {rb}")

    # tool-call format check (fail closed)
    agentic_rows = [i for i, d in enumerate(calib_dom) if d == "agentic"]
    tc_hits = sum(rend.tok.decode(calib[i]).count("<tool_call><function=") for i in agentic_rows)
    k_rows = [i for i, d in enumerate(klref_dom) if d == "agentic"]
    tc_hits_k = sum(rend.tok.decode(klref[i]).count("<tool_call><function=") for i in k_rows)
    think_hits = sum(1 for i, d in enumerate(calib_dom) if d == "reasoning"
                     and re.search(r"<think>(?!</think>)", rend.tok.decode(calib[i])))
    cjk = [_frac_cjk(rend.tok.decode([t for t in calib[i] if t < 151643]))
           for i, d in enumerate(calib_dom) if d == "chinese"]
    log(f"checks: calib tool_call literals={tc_hits}, klref tool_call literals={tc_hits_k}, "
        f"reasoning seqs with non-empty think={think_hits}, chinese CJK frac per seq={[round(c, 2) for c in cjk]}")
    assert tc_hits > 0 and tc_hits_k > 0, "no <tool_call><function= literal in agentic sequences"
    assert all(c > 0.5 for c in cjk), "chinese sequence below 50% CJK"

    cp, kp = args.out / "calib_tokens.npy", args.out / "klref_tokens.npy"
    np.save(cp, calib)
    np.save(kp, klref)
    (args.out / "calib_domains.json").write_text(json.dumps(calib_dom))
    (args.out / "klref_domains.json").write_text(json.dumps(klref_dom))

    # samples
    with (args.out / "samples.txt").open("w") as f:
        for name, arr, doms in (("calib", calib, calib_dom), ("klref", klref, klref_dom)):
            for dom in DOMAIN_MIX:
                idx = [i for i, d in enumerate(doms) if d == dom][:2]
                for i in idx:
                    txt = rend.tok.decode(arr[i])
                    f.write(f"\n{'=' * 30} {name} seq {i} [{dom}] {'=' * 30}\n")
                    f.write(txt[:2500] + ("\n...[truncated]...\n" + txt[-800:] if len(txt) > 3300 else txt[2500:]))
                    f.write("\n")

    meta = {
        "model": "XiaomiMiMo/MiMo-V2.6-Flash-RL",
        "tokenizer_dir": str(args.src),
        "chat_template_sha256": rend.template_sha,
        "render": {"apply_chat_template": True, "enable_thinking": True,
                   "add_generation_prompt": False, "trust_remote_code": False},
        "seed": SEED, "seq_len": args.seq_len,
        "pad_token": "<|endoftext|>", "pad_id": rend.pad_id,
        "domain_mix_target": DOMAIN_MIX,
        "calib": {"file": cp.name, "shape": list(calib.shape), "sha256": _sha256(cp),
                  "domains": calib_meta, "seq_domains_file": "calib_domains.json"},
        "klref": {"file": kp.name, "shape": list(klref.shape), "sha256": _sha256(kp),
                  "domains": klref_meta, "seq_domains_file": "klref_domains.json"},
        "sources": meta_sources,
        "skipped_sources": skipped,
        "checks": {"calib_tool_call_literals": tc_hits, "klref_tool_call_literals": tc_hits_k,
                   "calib_reasoning_seqs_with_think": think_hits,
                   "calib_chinese_cjk_frac": cjk},
        "excluded_benchmark_subsets": sorted(BENCH_SOURCES),
    }
    (args.out / "corpus_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log(f"wrote {cp} {calib.shape} sha256={meta['calib']['sha256']}")
    log(f"wrote {kp} {klref.shape} sha256={meta['klref']['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
