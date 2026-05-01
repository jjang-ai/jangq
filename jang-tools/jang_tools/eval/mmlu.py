"""MMLU eval — two-pass (no-reasoning + reasoning) for any JANG bundle.

Mirrors the proven pattern from research/experiments/minimax-m2.7/benchmark_mmlu_minimax.py.

Usage:
  python -m jang_tools.eval.mmlu --src <bundle> [--mode no-reasoning|reasoning|both]
                                  [--questions-per-subject 20]
                                  [--mmlu-parquet <path-or-hf-snapshot>]

Two passes:
  - no-reasoning: greedy temp=0, max_tokens=20, blocks <think>/</think>
  - reasoning:    temp=1.0 top_p=0.95, max_tokens=2048, lets model think then extracts A/B/C/D

Generation config rules (per project memory):
  - DSV4-style: enable_thinking flag toggles reasoning mode in chat_template
  - GLM-5.1: greedy + no rep penalty + enable_thinking=False for direct-answer pass
  - MiniMax: always-reasoning model — no enable_thinking flag, but think-tag block helps
"""
from __future__ import annotations

import argparse, gc, json, sys, time
from pathlib import Path

import mlx.core as mx

ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}
SUBJECTS_DEFAULT = [
    "abstract_algebra", "anatomy", "astronomy", "college_computer_science",
    "college_physics", "high_school_biology", "high_school_chemistry",
    "high_school_mathematics", "logical_fallacies", "world_religions",
]


def _load_mmlu(path_or_None: str | None):
    """Locate MMLU test parquet. Default: HF cache snapshot of cais/mmlu."""
    import pandas as pd
    if path_or_None and Path(path_or_None).exists():
        return pd.read_parquet(path_or_None)
    from huggingface_hub import snapshot_download
    snap = snapshot_download("cais/mmlu", repo_type="dataset",
                             allow_patterns=["all/test-*.parquet"])
    p = next(Path(snap).rglob("test-*.parquet"))
    return pd.read_parquet(p)


def _resolve_loader(src: str):
    """Return (model, tokenizer, fmt) for a JANG bundle.

    Tries jang_tools.{laguna,mistral3,...}.runtime.load() first, then falls
    back to mlx_lm.load() for vanilla affine bundles.
    """
    cfg = json.loads((Path(src) / "config.json").read_text())
    mt = cfg.get("text_config", cfg).get("model_type") or cfg.get("model_type")

    if mt == "laguna":
        from ..laguna.runtime import load
        m, c, fmt = load(src)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
        return m, tok, fmt
    if mt in ("mistral3", "ministral3"):
        from ..mistral3.runtime import load
        m, c, fmt = load(src)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
        return m, tok, fmt
    if mt and mt.startswith(("deepseek_v4", "dsv4")):
        # DSV4 has its own loader (load_jangtq) + chat-template injection
        # path. Use it instead of generic mlx_lm.load so the
        # `enable_thinking` flag actually fires and the </think> sentinel
        # gets tracked. Returns model + tok already-rehydrated.
        from ..load_jangtq import load_jangtq_model
        from ..dsv4.runtime import _inject_chat_template
        m, tok = load_jangtq_model(src)
        _inject_chat_template(tok, src)
        cfg_dict = json.loads((Path(src) / "config.json").read_text())
        return m, tok, cfg_dict.get("weight_format", "jangtq")
    # Generic path: mlx_lm
    from mlx_lm import load as mlxlm_load
    m, tok = mlxlm_load(src)
    fmt = cfg.get("weight_format") or "bf16"
    return m, tok, fmt


def _think_token_ids(tok):
    """Return (open_id, close_id) for <think>/</think>; None if unknown."""
    candidates_open = ["<think>", "<|think|>"]
    candidates_close = ["</think>", "<|/think|>"]
    o = c = None
    for s in candidates_open:
        try:
            ids = tok.encode(s, add_special_tokens=False)
            if len(ids) == 1: o = ids[0]; break
        except Exception:
            pass
    for s in candidates_close:
        try:
            ids = tok.encode(s, add_special_tokens=False)
            if len(ids) == 1: c = ids[0]; break
        except Exception:
            pass
    return o, c


def _make_block_think_processor(open_id, close_id):
    if open_id is None and close_id is None:
        return None
    def proc(tokens, logits):
        if open_id is not None: logits[0, open_id] = -float("inf")
        if close_id is not None: logits[0, close_id] = -float("inf")
        return logits
    return proc


def _strip_thinking(text: str, model_type: str | None) -> str:
    """Strip <think>/[THINK] reasoning blocks using the matching parser."""
    try:
        if model_type in ("mistral3", "ministral3", "mistral4"):
            from ..reasoning.mistral_parser import MistralReasoningParser
            return MistralReasoningParser().extract_content(text) or text
        if model_type and (model_type.startswith("qwen") or model_type.startswith("laguna")):
            from ..reasoning.qwen3_parser import Qwen3ReasoningParser
            return Qwen3ReasoningParser().extract_content(text) or text
        if model_type and (model_type.startswith("deepseek") or model_type.startswith("dsv")):
            from ..reasoning.deepseek_r1_parser import DeepseekR1ReasoningParser
            return DeepseekR1ReasoningParser().extract_content(text) or text
        if model_type and model_type.startswith("gemma"):
            from ..reasoning.gemma4_parser import Gemma4ReasoningParser
            return Gemma4ReasoningParser().extract_content(text) or text
    except Exception:
        pass
    # Fallback: drop everything up to the last </think> | [/THINK]
    for tag in ("</think>", "[/THINK]", "</thinking>"):
        if tag in text:
            text = text.rsplit(tag, 1)[1]
            break
    return text


def _extract_answer(text: str, model_type: str | None = None) -> str | None:
    body = _strip_thinking(text, model_type)
    body = body.strip().upper()
    if "ANSWER" in body:
        tail = body.split("ANSWER", 1)[1]
        for ch in tail:
            if ch in "ABCD":
                return ch
    for ch in body:
        if ch in "ABCD":
            return ch
    return None


def run_pass(model, tok, df, *, subjects, qps, mode, max_new, model_type=None):
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.generate import generate_step

    if mode == "reasoning":
        sampler = make_sampler(temp=1.0, top_p=0.95)
        block = None
        enable_thinking = True
    else:
        sampler = make_sampler(temp=0.0)
        open_id, close_id = _think_token_ids(tok)
        block = _make_block_think_processor(open_id, close_id)
        enable_thinking = False

    print(f"\n=== MMLU pass: {mode}  (max_new={max_new}, "
          f"enable_thinking={enable_thinking}) ===", flush=True)

    by_subject: dict = {}
    correct = total = 0
    t0 = time.time()
    for subject in subjects:
        sub = df[df["subject"] == subject].head(qps)
        sc = 0
        for _, row in sub.iterrows():
            q = row["question"]; ch = row["choices"]; gold = int(row["answer"])
            user = (
                "Answer the following multiple choice question. "
                "Reply with just the letter (A, B, C, or D).\n\n"
                f"{q}\nA. {ch[0]}\nB. {ch[1]}\nC. {ch[2]}\nD. {ch[3]}\n\nAnswer:"
            )
            try:
                prompt = tok.apply_chat_template(
                    [{"role": "user", "content": user}],
                    add_generation_prompt=True, tokenize=False,
                    enable_thinking=enable_thinking,
                )
            except Exception:
                prompt = user
            ids = tok.encode(prompt)

            gen: list = []
            kwargs = dict(prompt=mx.array(ids), model=model,
                          max_tokens=max_new, sampler=sampler)
            if block is not None:
                kwargs["logits_processors"] = [block]
            for tok_id, _ in generate_step(**kwargs):
                gen.append(int(tok_id))
                if int(tok_id) == tok.eos_token_id:
                    break
            text = tok.decode(gen)
            pred = _extract_answer(text, model_type)
            ok = pred == ANSWER_MAP[gold]
            sc += int(ok); correct += int(ok); total += 1
        by_subject[subject] = sc / qps
        print(f"  {subject:<32s} {sc}/{qps} ({sc/qps*100:.1f}%)", flush=True)
    dt = time.time() - t0
    print(f"\n  TOTAL: {correct}/{total} ({correct/total*100:.2f}%) in {dt:.1f}s", flush=True)
    return {"mode": mode, "correct": correct, "total": total,
            "by_subject": by_subject, "elapsed_s": dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to JANG bundle")
    ap.add_argument("--mode", choices=("no-reasoning", "reasoning", "both"),
                    default="both")
    ap.add_argument("--qps", "--questions-per-subject", type=int, default=20)
    ap.add_argument("--subjects", nargs="*", default=SUBJECTS_DEFAULT)
    ap.add_argument("--max-new-direct", type=int, default=20)
    ap.add_argument("--max-new-reasoning", type=int, default=2048)
    ap.add_argument("--mmlu-parquet", default=None)
    ap.add_argument("--out", default=None, help="JSON output path")
    args = ap.parse_args()

    print(f"[mmlu] loading bundle {args.src}", flush=True)
    model, tok, fmt = _resolve_loader(args.src)
    cfg_dict = json.loads((Path(args.src) / "config.json").read_text())
    mt = cfg_dict.get("text_config", cfg_dict).get("model_type") or cfg_dict.get("model_type")
    print(f"[mmlu] loaded ({fmt}, model_type={mt}). Loading MMLU…", flush=True)
    df = _load_mmlu(args.mmlu_parquet)

    results = {"src": args.src, "weight_format": fmt,
               "subjects": args.subjects, "qps": args.qps, "passes": []}

    if args.mode in ("no-reasoning", "both"):
        results["passes"].append(run_pass(model, tok, df,
            subjects=args.subjects, qps=args.qps,
            mode="no-reasoning", max_new=args.max_new_direct, model_type=mt))
        gc.collect()
    if args.mode in ("reasoning", "both"):
        results["passes"].append(run_pass(model, tok, df,
            subjects=args.subjects, qps=args.qps,
            mode="reasoning", max_new=args.max_new_reasoning, model_type=mt))

    print("\n=== SUMMARY ===")
    for p in results["passes"]:
        print(f"  {p['mode']:<14s}  {p['correct']}/{p['total']}  "
              f"({p['correct']/p['total']*100:.2f}%)  {p['elapsed_s']:.1f}s")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
