"""DSV4-Flash MMLU runner v2 — 40 subjects × 5q with FIXED extractor.

Bug in v1: `next((c for c in out if c in 'ABCD'))` picked 'C' from
"CORRECT" in "The correct answer is X" — wrong.

Fix: split on "ANSWER" first, then walk the tail.
"""
import os, sys, time, json, re
sys.path.insert(0, "/Users/eric/jang/jang-tools")
sys.stdout.reconfigure(line_buffering=True)
import mlx.core as mx
mx.set_memory_limit(110 * 1024**3); mx.set_cache_limit(8 * 1024**3)

import pandas as pd
from huggingface_hub import snapshot_download
from pathlib import Path
from mlx_lm.sample_utils import make_sampler
from mlx_lm.generate import generate_step
from jang_tools.load_jangtq import load_jangtq_model

SUBJECTS_40 = [
    "high_school_government_and_politics","public_relations","computer_security",
    "philosophy","high_school_us_history","marketing","high_school_macroeconomics",
    "high_school_psychology","prehistory","high_school_microeconomics",
    "conceptual_physics","nutrition","high_school_computer_science","human_sexuality",
    "college_medicine","miscellaneous","clinical_knowledge","high_school_geography",
    "professional_medicine","high_school_biology","world_religions","logical_fallacies",
    "security_studies","virology","high_school_chemistry","jurisprudence",
    "college_physics","management","moral_disputes","professional_psychology",
    "econometrics","high_school_european_history","professional_law",
    "high_school_statistics","human_aging","formal_logic","high_school_world_history",
    "business_ethics","abstract_algebra","high_school_mathematics",
]
QPS = 5
ANS = {0:"A",1:"B",2:"C",3:"D"}
SRC = "/Users/eric/.mlxstudio/models/_bundles/DeepSeek-V4-Flash-JANGTQ"


def extract_answer(text: str) -> str:
    """Match eval/mmlu.py — split on ANSWER, then take first letter in tail.
    Also handle the model saying 'The correct answer is **B**' or 'B.' or '**B**'."""
    body = text.strip().upper()
    # Pattern 1: explicit "ANSWER IS **X**" or "ANSWER: X" or "ANSWER X"
    m = re.search(r"ANSWER[\s:*]*(?:IS[\s*]*)?\*{0,2}([ABCD])\b", body)
    if m: return m.group(1)
    # Pattern 2: ** B ** wrap
    m = re.search(r"\*\*\s*([ABCD])\b", body)
    if m: return m.group(1)
    # Pattern 3: split on ANSWER, walk tail
    if "ANSWER" in body:
        tail = body.split("ANSWER", 1)[1]
        for ch in tail:
            if ch in "ABCD": return ch
    # Last resort
    for ch in body:
        if ch in "ABCD": return ch
    return "?"


print(f"[v2] DSV4_LONG_CTX={os.environ.get('DSV4_LONG_CTX','1')}, FIXED extractor", flush=True)
t0 = time.time(); model, tok = load_jangtq_model(SRC)
print(f"  loaded {time.time()-t0:.1f}s", flush=True)

snap = snapshot_download("cais/mmlu", repo_type="dataset",
                          allow_patterns=["all/test-*.parquet"])
df = pd.read_parquet(next(Path(snap).rglob("test-*.parquet")))
sampler = make_sampler(temp=0.0)

correct = total = 0
results = {}
samples = []  # (subject, q_idx, gold, pred, raw_first_60)
t0 = time.time()
for s in SUBJECTS_40:
    sub = df[df["subject"] == s].head(QPS)
    sc = 0
    for qi, (_, row) in enumerate(sub.iterrows()):
        q, ch, gold = row["question"], row["choices"], int(row["answer"])
        msg = ("Answer the following multiple choice question. Reply with just the "
               f"letter (A, B, C, or D).\n\n{q}\nA. {ch[0]}\nB. {ch[1]}\nC. {ch[2]}\nD. {ch[3]}\n\nAnswer:")
        try:
            text = tok.apply_chat_template([{"role":"user","content":msg}],
                                           add_generation_prompt=True,
                                           tokenize=False, enable_thinking=False)
        except Exception: text = msg
        ids = tok.encode(text); gen=[]
        for tid,_ in generate_step(prompt=mx.array(ids), model=model,
                                    max_tokens=80, sampler=sampler):
            gen.append(int(tid))
            if int(tid)==tok.eos_token_id: break
        out = tok.decode(gen)
        pred = extract_answer(out)
        ok = pred == ANS[gold]
        sc += int(ok); correct += int(ok); total += 1
        # Save first occurrence of each subject for spot check
        if qi == 0:
            samples.append((s, qi, ANS[gold], pred, out[:60].replace("\n"," ")))
    results[s] = f"{sc}/{QPS}"
    print(f"  {s:<48s} {sc}/{QPS} ({sc/QPS*100:.0f}%)", flush=True)

dt = time.time() - t0
pct = 100 * correct / total
print(f"\n  TOTAL: {correct}/{total} ({pct:.2f}%) in {dt/60:.1f}min", flush=True)
print("\n--- spot check first q per subject ---", flush=True)
for s, qi, gold, pred, raw in samples:
    flag = "OK" if gold == pred else "WRONG"
    print(f"  {flag} {s:<35s} gold={gold} pred={pred}  raw={raw!r}", flush=True)

out_path = Path(f"/tmp/dsv4_mmlu40_v2_lc{os.environ.get('DSV4_LONG_CTX','1')}.json")
out_path.write_text(json.dumps({
    "src": SRC, "DSV4_LONG_CTX": os.environ.get("DSV4_LONG_CTX","1"),
    "extractor": "v2_with_answer_split_and_regex",
    "max_tokens": 80,
    "total": total, "correct": correct, "pct": pct,
    "elapsed_min": dt/60, "by_subject": results,
}, indent=2))
print(f"\nwrote {out_path}", flush=True)
