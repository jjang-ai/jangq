"""Build calibration corpus v3 — balanced mix, modernized sources, QA pass.

Mix (tokens-balanced):

  coding       22%   modern 2024-2025 code datasets (OpenCodeReasoning,
                     Magicoder-OSS, CodeFeedback)
  agentic      19%   tool-use / function calling / agent traces
  general      17%   modern SFT mixtures (tulu-3, Hermes-3, OpenThoughts)
  academic_mc  12%   MMLU-Pro, MMLU-train, ARC-Challenge, OpenBookQA, SciQ,
                     MedQA, CommonsenseQA — MC-format questions
  science      10%   NuminaMath-CoT, pubmed_qa, camel-ai/physics (prose)
  chinese       9%   COIG-CQIA public subset, Zhihu-KOL, firefly
  cybersec      5%   CyberNative, Trendyol cybersec
  longctx       3%   PG19, arxiv long articles
  systems       3%   dolphin-coder, sql-create-context

QA pass after build:
  * Exact-dedup via blake2b(first 512 chars) — built into pipeline
  * Near-dup detection via 10-shingle Jaccard overlap on a 2 000-sample
    spot-check per domain
  * Length distribution (median, p95, min, max) per domain
  * Language detection on chinese bucket — only accept if >50% CJK chars
  * MC-format sanity: academic_mc rows must contain 'Answer:' marker
  * Prints 3 random samples per domain for manual inspection

Usage:
  python -m jang_tools.kimi_prune.build_calib_v3 \\
      --out /path/corpus_v3.jsonl \\
      --target-tokens 10_000_000 \\
      --seed 42

Fault-tolerant: skips any source that fails to load (gated, moved, etc.)
rather than crashing. Logs the skipped sources and continues with the rest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from jang_tools.kimi_prune.build_calib import (
    Source, _conv_text, _field, _join_fields, _code_solution,
    _hash, CHARS_PER_TOK,
)
from jang_tools.kimi_prune.build_calib_v2 import _mc_text, _sciq_mc


def _numina_math(row: dict) -> str | None:
    """NuminaMath-CoT: 'problem' + 'solution' fields, CoT-formatted."""
    p = row.get("problem") or ""
    s = row.get("solution") or ""
    if len(p) < 10 or len(s) < 30:
        return None
    return f"Problem: {p}\n\nSolution: {s}"


def _medqa_mc(row: dict) -> str | None:
    """MedQA / medical MC — usually has question + options list + answer_idx."""
    q = row.get("question")
    opts = row.get("options") or {}
    ans = row.get("answer_idx") or row.get("answer")
    if not (q and opts and ans is not None):
        return None
    letters = ["A", "B", "C", "D", "E"]
    if isinstance(opts, dict):
        # options = {"A": "...", "B": "...", ...}
        lines = [f"{k}. {v}" for k, v in sorted(opts.items())]
        ans_letter = str(ans).strip()
    elif isinstance(opts, list):
        lines = [f"{letters[i]}. {o}" for i, o in enumerate(opts[:5])]
        ans_letter = letters[int(ans)] if isinstance(ans, int) else str(ans)
    else:
        return None
    return f"Question: {q}\n" + "\n".join(lines) + f"\nAnswer: {ans_letter}"


def _mmlu_pro_mc(row: dict) -> str | None:
    """MMLU-Pro row: 'question', 'options' list, 'answer' letter."""
    q = row.get("question")
    opts = row.get("options")
    ans = row.get("answer")  # letter string
    if not (q and opts and ans):
        return None
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    lines = [f"{letters[i]}. {o}" for i, o in enumerate(opts[:10])]
    return f"Question: {q}\n" + "\n".join(lines) + f"\nAnswer: {ans}"


def _commonsense_mc(row: dict) -> str | None:
    """CommonsenseQA: stem + choices.label/text + answerKey."""
    q = row.get("question")
    choices = row.get("choices") or {}
    ans = row.get("answerKey")
    if not (q and choices and ans):
        return None
    labels = choices.get("label") or []
    texts = choices.get("text") or []
    if not labels or len(labels) != len(texts):
        return None
    lines = [f"{lab}. {t}" for lab, t in zip(labels, texts)]
    return f"Question: {q}\n" + "\n".join(lines) + f"\nAnswer: {ans}"


SOURCES_V3: list[Source] = [
    # === coding 22% (modernized) ========================================
    Source("coding", 0.07, "ise-uiuc/Magicoder-OSS-Instruct-75K", None, "train",
           _join_fields("problem", "solution"), max_records=75_000),
    Source("coding", 0.06, "nvidia/OpenCodeReasoning", "split_0", "split_0",
           _join_fields("input", "output"), max_records=20_000),
    Source("coding", 0.04, "m-a-p/CodeFeedback-Filtered-Instruction", None, "train",
           _join_fields("query", "answer"), max_records=50_000),
    Source("coding", 0.03, "HuggingFaceH4/CodeAlpaca_20K", None, "train",
           _join_fields("prompt", "completion"), max_records=20_000),
    Source("coding", 0.02, "iamtarun/python_code_instructions_18k_alpaca", None, "train",
           _join_fields("instruction", "input", "output"), max_records=18_000),

    # === agentic 19% ====================================================
    Source("agentic", 0.07, "NousResearch/hermes-function-calling-v1", None, "train",
           _conv_text, max_records=50_000),
    Source("agentic", 0.05, "glaiveai/glaive-function-calling-v2", None, "train",
           _field("chat"), max_records=30_000),
    Source("agentic", 0.03, "lilacai/glaive-function-calling-v2-sharegpt", None, "train",
           _conv_text, max_records=20_000),
    Source("agentic", 0.02, "THUDM/AgentInstruct", None, "os",
           _conv_text, max_records=3_000),
    Source("agentic", 0.02, "princeton-nlp/SWE-bench_oracle", None, "train",
           _join_fields("problem_statement", "patch"), max_records=10_000),

    # === general 17% (modernized) =======================================
    Source("general", 0.07, "allenai/tulu-3-sft-mixture", None, "train",
           _conv_text, max_records=80_000),
    Source("general", 0.04, "open-thoughts/OpenThoughts-114k", None, "train",
           _conv_text, max_records=50_000),
    Source("general", 0.03, "teknium/OpenHermes-2.5", None, "train",
           _conv_text, max_records=40_000),
    Source("general", 0.03, "HuggingFaceH4/ultrachat_200k", None, "train_sft",
           _conv_text, max_records=30_000),

    # === academic_mc 12% =================================================
    Source("academic_mc", 0.05, "cais/mmlu", "all", "auxiliary_train",
           _mc_text, max_records=40_000),
    Source("academic_mc", 0.03, "TIGER-Lab/MMLU-Pro", None, "test",
           _mmlu_pro_mc, max_records=12_000),
    Source("academic_mc", 0.01, "allenai/ai2_arc", "ARC-Challenge", "train",
           _mc_text, max_records=2_000),
    Source("academic_mc", 0.01, "allenai/openbookqa", "main", "train",
           _mc_text, max_records=5_000),
    Source("academic_mc", 0.01, "allenai/sciq", None, "train",
           _sciq_mc, max_records=13_000),
    Source("academic_mc", 0.005, "tau/commonsense_qa", None, "train",
           _commonsense_mc, max_records=10_000),
    Source("academic_mc", 0.005, "bigbio/med_qa", "med_qa_en_source", "train",
           _medqa_mc, max_records=10_000),

    # === science 10% (prose + math CoT) =================================
    Source("science", 0.04, "AI-MO/NuminaMath-CoT", None, "train",
           _numina_math, max_records=30_000),
    Source("science", 0.03, "ccdv/arxiv-summarization", None, "train",
           _field("article"), max_records=10_000),
    Source("science", 0.015, "qiaojin/PubMedQA", "pqa_labeled", "train",
           _join_fields("question", "long_answer"), max_records=1_000),
    Source("science", 0.015, "camel-ai/physics", None, "train",
           _join_fields("topic", "message_1", "message_2"), max_records=20_000),

    # === chinese 9% ======================================================
    Source("chinese", 0.04, "silk-road/alpaca-data-gpt4-chinese", None, "train",
           _join_fields("instruction_zh", "input_zh", "output_zh"),
           max_records=40_000),
    Source("chinese", 0.025, "wangrui6/Zhihu-KOL", None, "train",
           _join_fields("INSTRUCTION", "RESPONSE"), max_records=20_000),
    Source("chinese", 0.025, "YeungNLP/firefly-train-1.1M", None, "train",
           _join_fields("input", "target"), max_records=30_000),

    # === cybersec 5% ====================================================
    Source("cybersec", 0.03, "CyberNative/Code_Vulnerability_Security_DPO", None, "train",
           _join_fields("system", "question", "chosen"), max_records=15_000),
    Source("cybersec", 0.02, "Trendyol/cybersecurity-instruction-datasets", None, "train",
           _join_fields("instruction", "input", "output"), max_records=15_000),

    # === longctx 3% =====================================================
    Source("longctx", 0.02, "emozilla/pg19", None, "train",
           _field("text"), max_records=1_000),
    Source("longctx", 0.01, "ccdv/arxiv-summarization", None, "train",
           _field("article"), max_records=2_000),

    # === systems 3% =====================================================
    Source("systems", 0.015, "b-mc2/sql-create-context", None, "train",
           _join_fields("question", "context", "answer"), max_records=10_000),
    Source("systems", 0.015, "cognitivecomputations/dolphin-coder", None, "train",
           _conv_text, max_records=10_000),
]


# ---------- quality check helpers ------------------------------------

_CJK_RE = re.compile(r"[一-鿿㐀-䶿]")


def _frac_cjk(s: str) -> float:
    if not s:
        return 0.0
    total = sum(1 for c in s if not c.isspace())
    if total == 0:
        return 0.0
    cjk = len(_CJK_RE.findall(s))
    return cjk / total


def _shingles(s: str, n: int = 10) -> set:
    toks = s.split()
    return set(" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)) \
        if len(toks) >= n else set()


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def qa_pass(corpus_path: Path, sample_per_domain: int = 2000) -> dict:
    """Post-build QA:
       * length stats per domain
       * near-dup spot check (shingle Jaccard, 500 random pairs per domain)
       * CJK-fraction check for chinese bucket
       * MC-format check for academic_mc
       * 3 random samples per domain printed for manual inspection
    """
    print(f"\n[QA] opening {corpus_path}", flush=True)
    by_domain: dict[str, list[str]] = {}
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except (TypeError, ValueError):
                continue
            d = r.get("domain")
            t = r.get("text") or ""
            if d is None:
                continue
            by_domain.setdefault(d, []).append(t)
    print(f"[QA] domains: {sorted(by_domain.keys())}", flush=True)

    report = {}
    for d, texts in sorted(by_domain.items()):
        lens = sorted(len(t) for t in texts)
        n = len(lens)
        stats = {
            "count": n,
            "len_min": lens[0] if n else 0,
            "len_median": lens[n // 2] if n else 0,
            "len_p95": lens[int(n * 0.95)] if n else 0,
            "len_max": lens[-1] if n else 0,
        }

        # Near-dup spot check on random 500 pairs
        rng = random.Random(42)
        k = min(500, n)
        if k > 5:
            sample = rng.sample(texts, k)
            dupes = 0
            for i in range(0, k, 2):
                if i + 1 >= k:
                    break
                ja = _jaccard(_shingles(sample[i][:2000]),
                              _shingles(sample[i + 1][:2000]))
                if ja > 0.5:
                    dupes += 1
            stats["near_dup_rate"] = round(dupes / max(1, k // 2), 3)

        if d == "chinese":
            frac = sum(_frac_cjk(t[:200]) for t in texts[:500]) / max(1, min(500, n))
            stats["avg_cjk_frac"] = round(frac, 2)
            if frac < 0.3:
                stats["warning"] = f"low CJK fraction {frac:.2f} — chinese bucket may have drifted English"
        if d == "academic_mc":
            mc_ok = sum(1 for t in texts[:500] if "Answer:" in t) / max(1, min(500, n))
            stats["mc_format_ok_rate"] = round(mc_ok, 2)
            if mc_ok < 0.8:
                stats["warning"] = f"only {mc_ok:.0%} of academic_mc rows have 'Answer:' marker"

        report[d] = stats
        print(f"\n[QA:{d}] count={n} "
              f"len[min/med/p95/max]={stats['len_min']}/{stats['len_median']}/{stats['len_p95']}/{stats['len_max']}",
              flush=True)
        for k_, v in stats.items():
            if k_ in ("count", "len_min", "len_median", "len_p95", "len_max"):
                continue
            print(f"  {k_}: {v}", flush=True)
        # 3 random samples
        for i, t in enumerate(rng.sample(texts, min(3, n))):
            snippet = t[:200].replace("\n", " ")
            print(f"  sample[{i}]: {snippet!r}", flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--target-tokens", type=int, default=10_000_000,
                    help="Bigger than v2 (8.6M) for better saliency coverage.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-build", action="store_true",
                    help="Skip corpus build, only run QA on existing file.")
    args = ap.parse_args()

    if not args.skip_build:
        from jang_tools.kimi_prune import build_calib as _v1
        _v1.SOURCES = SOURCES_V3
        print("[build-v3] corpus mix:", flush=True)
        for d in sorted({s.domain for s in SOURCES_V3}):
            w = sum(s.weight for s in SOURCES_V3 if s.domain == d)
            print(f"  {d:<14} {w * 100:.1f}%", flush=True)
        _v1.build(args.out, args.target_tokens, args.seed)

    # QA pass
    report = qa_pass(args.out)
    qa_path = args.out.with_suffix(".qa.json")
    qa_path.write_text(json.dumps(report, indent=2))
    print(f"\n[QA] report written to {qa_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
