"""
JANG vs MLX — MMLU Benchmark
5 questions per subject, 10 subjects = 50 questions per model
Uses chat template with enable_thinking=False for fair comparison.
"""

import sys
import time
import pandas as pd
import mlx.core as mx
from mlx_lm.sample_utils import make_sampler
from mlx_lm.generate import generate_step

SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "college_computer_science",
    "college_physics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_mathematics",
    "logical_fallacies",
    "world_religions",
]
QUESTIONS_PER_SUBJECT = 20
ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}

df = pd.read_parquet(
    "/Users/eric/.cache/huggingface/hub/datasets--cais--mmlu/"
    "snapshots/c30699e8356da336a370243923dbaf21066bb9fe/all/test-00000-of-00001.parquet"
)

def run_mmlu(model, tokenizer, model_name):
    sampler = make_sampler(temp=0.0)
    correct = 0
    total = 0

    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"  GPU: {mx.get_active_memory()/1024**3:.1f} GB")
    print(f"{'='*60}")

    t0 = time.time()

    for subject in SUBJECTS:
        sub_df = df[df["subject"] == subject].head(QUESTIONS_PER_SUBJECT)
        sub_correct = 0

        for _, row in sub_df.iterrows():
            q = row["question"]
            choices = row["choices"]
            answer = int(row["answer"])

            user_msg = f"Answer the following multiple choice question. Reply with just the letter (A, B, C, or D).\n\n{q}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n\nAnswer:"

            # Use chat template with thinking disabled for fair comparison
            try:
                messages = [{"role": "user", "content": user_msg}]
                prompt_str = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True,
                    enable_thinking=False, tokenize=False
                )
                tokens = tokenizer.encode(prompt_str)
            except Exception:
                # Fallback for tokenizers without chat template
                tokens = tokenizer.encode(user_msg)

            gen = []
            for tok, _ in generate_step(prompt=mx.array(tokens), model=model, max_tokens=20, sampler=sampler):
                gen.append(int(tok))
                if int(tok) == tokenizer.eos_token_id:
                    break

            response = tokenizer.decode(gen).strip().upper()
            predicted = None
            for c in response:
                if c in "ABCD":
                    predicted = c
                    break

            correct_letter = ANSWER_MAP[answer]
            is_correct = predicted == correct_letter
            if is_correct:
                correct += 1
                sub_correct += 1
            total += 1

        print(f"  {subject:<30s}  {sub_correct}/{QUESTIONS_PER_SUBJECT}")

    dt = time.time() - t0
    acc = correct / total * 100 if total > 0 else 0
    # Count total generated tokens for tok/s
    total_gen_tokens = total * 10  # ~10 tokens avg per answer (conservative estimate)
    tps = total_gen_tokens / dt if dt > 0 else 0
    print(f"\n  TOTAL: {correct}/{total} = {acc:.1f}%")
    print(f"  Time: {dt:.1f}s | Speed: ~{tps:.0f} tok/s | GPU: {mx.get_active_memory()/1024**3:.1f} GB")
    return correct, total, acc


# Model 1
model_path = sys.argv[1]
model_type = sys.argv[2]  # "jang" or "mlx"

if model_type == "jang":
    from jang_tools.loader import load_jang_model
    model, tokenizer = load_jang_model(model_path)
else:
    from pathlib import Path
    from mlx_lm import load
    model, tokenizer = load(str(Path(model_path)))

correct, total, acc = run_mmlu(model, tokenizer, f"{model_type.upper()}: {model_path.split('/')[-1]}")
