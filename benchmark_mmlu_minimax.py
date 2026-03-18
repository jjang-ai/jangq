#!/usr/bin/env python3
"""
MMLU benchmark for MiniMax JANG models.

MiniMax's chat template ALWAYS injects <think> — there's no enable_thinking flag.
This script strips <think> from the prompt to get direct answers (20 tokens max).

Usage:
  python3 benchmark_mmlu_minimax.py /path/to/model jang
  python3 benchmark_mmlu_minimax.py /path/to/model jang --base /path/to/base  # compare
"""

import sys, time, argparse
sys.path.insert(0, '/Users/eric/jang/jang-tools')
sys.stdout.reconfigure(line_buffering=True)

import mlx.core as mx
mx.set_memory_limit(200 * 1024**3)
mx.set_cache_limit(8 * 1024**3)

import gc, pandas as pd
from mlx_lm.sample_utils import make_sampler
from mlx_lm.generate import generate_step
from jang_tools.loader import load_jang_model

SUBJECTS = ["abstract_algebra","anatomy","astronomy","college_computer_science",
    "college_physics","high_school_biology","high_school_chemistry",
    "high_school_mathematics","logical_fallacies","world_religions"]
QPS = 20
ANSWER_MAP = {0:"A",1:"B",2:"C",3:"D"}

def run_mmlu(model, tokenizer, name):
    df = pd.read_parquet(
        "/Users/eric/.cache/huggingface/hub/datasets--cais--mmlu/"
        "snapshots/c30699e8356da336a370243923dbaf21066bb9fe/all/test-00000-of-00001.parquet"
    )
    sampler = make_sampler(temp=0.0)
    correct = total = 0
    print(f"\n{'='*60}", flush=True)
    print(f"  {name}", flush=True)
    print(f"  GPU: {mx.get_active_memory()/1024**3:.1f} GB", flush=True)
    print(f"{'='*60}", flush=True)

    results = {}
    for subject in SUBJECTS:
        sub_df = df[df["subject"]==subject].head(QPS)
        sc = 0
        for _, row in sub_df.iterrows():
            q, choices, answer = row["question"], row["choices"], int(row["answer"])
            msg = f"Answer the following multiple choice question. Reply with just the letter (A, B, C, or D).\n\n{q}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n\nAnswer:"
            messages = [{"role": "user", "content": msg}]
            text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False)

            # Strip <think> from end — MiniMax template always adds it
            # This forces direct answers without thinking chain
            if text.rstrip().endswith("<think>"):
                text = text.rstrip()
                text = text[:text.rfind("<think>")]

            tokens = tokenizer.encode(text)
            gen = []
            for tok, _ in generate_step(
                    prompt=mx.array(tokens), model=model,
                    max_tokens=20, sampler=sampler):
                gen.append(int(tok))
                if int(tok) == tokenizer.eos_token_id:
                    break

            response = tokenizer.decode(gen).strip().upper()
            predicted = None
            for c in response:
                if c in "ABCD":
                    predicted = c
                    break

            if predicted == ANSWER_MAP[answer]:
                correct += 1
                sc += 1
            total += 1
            mx.clear_cache()

        results[subject] = sc
        print(f"  {subject:<30s}  {sc}/{QPS}", flush=True)

    acc = correct / total * 100
    print(f"\n  TOTAL: {correct}/{total} = {acc:.1f}%", flush=True)
    print(f"  GPU: {mx.get_active_memory()/1024**3:.1f} GB", flush=True)
    return results, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("type", choices=["jang", "mlx"])
    parser.add_argument("--base", help="Optional base model for comparison")
    args = parser.parse_args()

    if args.type == "jang":
        model, tokenizer = load_jang_model(args.model)
    else:
        from mlx_lm import load
        model, tokenizer = load(args.model)

    name = args.model.split("/")[-1]
    r1, a1 = run_mmlu(model, tokenizer, name)

    if args.base:
        del model, tokenizer
        gc.collect()
        mx.clear_cache()
        if args.type == "jang":
            model, tokenizer = load_jang_model(args.base)
        else:
            from mlx_lm import load
            model, tokenizer = load(args.base)
        r2, a2 = run_mmlu(model, tokenizer, args.base.split("/")[-1])

        print(f"\n{'='*60}", flush=True)
        print(f"  {'Subject':<30s} {'Model':>6s} {'Base':>6s}", flush=True)
        for s in SUBJECTS:
            print(f"  {s:<30s} {r1[s]:>5}/20 {r2[s]:>5}/20", flush=True)
        print(f"  {'TOTAL':<30s} {a1:>5.1f}% {a2:>5.1f}%", flush=True)
