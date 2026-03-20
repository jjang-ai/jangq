"""
JANG MMLU Benchmark — With thinking mode, forced answers, full logging, and checkpointing.
Created by Jinho Jang (eric@jangq.ai)

Features:
  - Thinking ON: model reasons in <think>...</think> then answers
  - Forced answer: if model uses all thinking tokens without answering,
    appends "</think>\n\nThe answer is:" to force a response
  - Full output logging: every question's response saved to JSONL
  - Checkpointing: resume from where you left off after crash
  - Answer verification: logs predicted vs correct, checks think tag closure

Usage:
    python -m jang_tools.benchmark /path/to/model
    python -m jang_tools.benchmark /path/to/model --max-thinking 4096
    python -m jang_tools.benchmark /path/to/model --no-thinking
    python -m jang_tools.benchmark /path/to/model --resume
"""

import json
import os
import sys
import time
from pathlib import Path

SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy",
    "college_computer_science", "college_physics",
    "high_school_biology", "high_school_chemistry",
    "high_school_mathematics", "logical_fallacies", "world_religions",
]
QUESTIONS_PER_SUBJECT = 20


def _extract_answer(response: str, thinking: bool) -> tuple[str | None, bool, str]:
    """
    Extract the answer letter from a model response.

    Returns: (predicted_letter, think_closed, clean_response)
    - predicted_letter: 'A', 'B', 'C', 'D', or None
    - think_closed: whether </think> tag was properly closed
    - clean_response: the response after thinking (for logging)
    """
    think_closed = True
    clean = response

    if thinking:
        if "<think>" in response:
            think_closed = "</think>" in response
            # Get everything after the last </think>
            if think_closed:
                clean = response.split("</think>")[-1].strip()
            else:
                # Thinking wasn't closed — extract any letter from full response
                clean = response

    # Remove common tags
    clean = clean.replace("<|im_end|>", "").strip()

    # Extract answer letter — search smart, not just first occurrence.
    # The response may contain math notation with A/B/C/D letters.
    # Priority: explicit answer patterns > last standalone letter > first letter
    import re
    pred = None

    # 1. Look for explicit answer patterns (highest confidence)
    patterns = [
        r"(?:answer|option|correct)\s*(?:is|:)\s*\(?([A-D])\)?",  # "answer is B", "option: C"
        r"\b([A-D])\.\s*$",                                        # ends with "B." on last line
        r"^\s*([A-D])\s*$",                                        # standalone letter on a line
        r"\*\*([A-D])\*\*",                                        # **B** (bold)
        r"\\boxed\{([A-D])\}",                                     # \boxed{B}
    ]
    for pattern in patterns:
        m = re.search(pattern, clean, re.IGNORECASE | re.MULTILINE)
        if m:
            pred = m.group(1).upper()
            break

    # 2. If no pattern matched, check last 50 chars for a standalone letter
    if pred is None:
        tail = clean[-50:] if len(clean) > 50 else clean
        for ch in reversed(tail.upper()):
            if ch in "ABCD":
                pred = ch
                break

    # 3. Last resort: first letter in response (least reliable)
    if pred is None:
        for ch in clean.upper():
            if ch in "ABCD":
                pred = ch
                break

    return pred, think_closed, clean


def run_mmlu(
    model_path: str,
    max_thinking: int = 4096,
    max_answer: int = 30,
    thinking: bool = True,
    resume: bool = False,
):
    import mlx.core as mx
    from jang_tools.loader import load_jang_model
    from mlx_lm import generate

    model_path = Path(model_path)
    model_name = model_path.name

    # Output files
    log_dir = model_path / "benchmark"
    log_dir.mkdir(exist_ok=True)
    mode_str = "thinking" if thinking else "no_thinking"
    output_file = log_dir / f"mmlu_outputs_{mode_str}.jsonl"
    summary_file = log_dir / f"mmlu_summary_{mode_str}.json"
    checkpoint_file = log_dir / f"mmlu_checkpoint_{mode_str}.json"

    print(f"{'='*60}")
    print(f"MMLU Benchmark: {model_name}")
    print(f"  Thinking: {'ON (max {max_thinking} tokens)' if thinking else 'OFF'}")
    print(f"  Answer tokens: {max_answer}")
    print(f"  Output log: {output_file}")
    print(f"{'='*60}")

    # Load checkpoint if resuming
    completed = {}
    if resume and checkpoint_file.exists():
        completed = json.loads(checkpoint_file.read_text())
        done_count = sum(len(v) for v in completed.values())
        print(f"  Resuming: {done_count} questions already done")

    # Load model
    print(f"\nLoading model...")
    t0 = time.time()
    model, tok = load_jang_model(str(model_path))
    gpu_gb = mx.get_active_memory() / 1024**3
    print(f"  Loaded in {time.time()-t0:.0f}s, GPU: {gpu_gb:.1f} GB")

    # Load dataset
    try:
        from datasets import load_dataset
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "datasets"], capture_output=True)
        from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")

    total_correct = 0
    total_asked = 0
    total_think_unclosed = 0
    total_forced = 0
    results_by_subject = {}

    # Open output log
    mode = "a" if resume and output_file.exists() else "w"
    fout = open(output_file, mode)

    for subject in SUBJECTS:
        subject_qs = [r for r in ds if r["subject"] == subject][:QUESTIONS_PER_SUBJECT]
        correct = 0
        subject_results = []

        for qi, q in enumerate(subject_qs):
            q_id = f"{subject}_{qi}"

            # Skip if already done (resume)
            if subject in completed and qi < len(completed[subject]):
                prev = completed[subject][qi]
                if prev.get("is_correct"):
                    correct += 1
                if not prev.get("think_closed", True):
                    total_think_unclosed += 1
                if prev.get("forced_answer"):
                    total_forced += 1
                total_asked += 1
                continue

            question = q["question"]
            choices = q["choices"]
            answer_idx = q["answer"]
            correct_letter = chr(65 + answer_idx)

            prompt_text = f"{question}\n"
            for i, c in enumerate(choices):
                prompt_text += f"{chr(65+i)}. {c}\n"
            prompt_text += "Answer with just the letter (A, B, C, or D)."

            messages = [{"role": "user", "content": prompt_text}]
            try:
                templated = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=thinking,
                )
            except TypeError:
                templated = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )

            forced_answer = False
            try:
                if thinking:
                    # Generate with thinking budget
                    response = generate(
                        model, tok, prompt=templated,
                        max_tokens=max_thinking + max_answer,
                        verbose=False,
                    )

                    # Check if thinking was closed and answer given
                    pred, think_closed, clean = _extract_answer(response, thinking=True)

                    if pred is None and not think_closed:
                        # Thinking ran out of tokens without closing — force answer
                        forced_prompt = response + "</think>\n\nThe answer is:"
                        forced_response = generate(
                            model, tok, prompt=templated + forced_prompt,
                            max_tokens=max_answer,
                            verbose=False,
                        )
                        response = response + "</think>\n\n" + forced_response
                        pred, _, clean = _extract_answer(forced_response, thinking=False)
                        forced_answer = True
                        total_forced += 1

                    if not think_closed:
                        total_think_unclosed += 1
                else:
                    response = generate(
                        model, tok, prompt=templated,
                        max_tokens=max_answer,
                        verbose=False,
                    )
                    pred, think_closed, clean = _extract_answer(response, thinking=False)

                is_correct = pred == correct_letter
                if is_correct:
                    correct += 1

            except Exception as e:
                response = f"ERROR: {e}"
                pred = None
                is_correct = False
                think_closed = True
                clean = response
                forced_answer = False

            total_asked += 1

            entry = {
                "subject": subject,
                "question_idx": qi,
                "question": question[:300],
                "choices": choices,
                "correct_answer": correct_letter,
                "predicted": pred,
                "is_correct": is_correct,
                "think_closed": think_closed,
                "forced_answer": forced_answer,
                "clean_response": clean[:300],
                "full_response": response[:2000],
                "thinking": thinking,
                "max_tokens": max_thinking if thinking else max_answer,
            }
            fout.write(json.dumps(entry) + "\n")
            fout.flush()
            subject_results.append(entry)

        results_by_subject[subject] = {
            "correct": correct,
            "total": len(subject_qs),
            "pct": round(correct / len(subject_qs) * 100, 1) if subject_qs else 0,
        }
        total_correct += correct
        print(f"  {subject}: {correct}/{len(subject_qs)} ({results_by_subject[subject]['pct']}%)")

        # Checkpoint after each subject
        if subject not in completed:
            completed[subject] = subject_results
        checkpoint_file.write_text(json.dumps(completed, indent=2))

    fout.close()

    overall = round(total_correct / total_asked * 100, 1) if total_asked else 0

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_correct}/{total_asked} ({overall}%)")
    print(f"Think unclosed: {total_think_unclosed}")
    print(f"Forced answers: {total_forced}")
    print(f"GPU: {mx.get_active_memory()/1024**3:.1f} GB")
    print(f"{'='*60}")

    summary = {
        "model": model_name,
        "model_path": str(model_path),
        "total_correct": total_correct,
        "total_questions": total_asked,
        "overall_pct": overall,
        "thinking": thinking,
        "max_thinking_tokens": max_thinking if thinking else 0,
        "max_answer_tokens": max_answer,
        "think_unclosed": total_think_unclosed,
        "forced_answers": total_forced,
        "gpu_gb": round(mx.get_active_memory() / 1024**3, 1),
        "subjects": results_by_subject,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_file.write_text(json.dumps(summary, indent=2))
    print(f"Summary: {summary_file}")
    print(f"Outputs: {output_file}")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="JANG MMLU Benchmark")
    parser.add_argument("model_path", help="Path to JANG model")
    parser.add_argument("--max-thinking", type=int, default=4096, help="Max thinking tokens")
    parser.add_argument("--max-answer", type=int, default=30, help="Max answer tokens after thinking")
    parser.add_argument("--no-thinking", action="store_true", help="Disable thinking mode")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    run_mmlu(
        args.model_path,
        max_thinking=args.max_thinking,
        max_answer=args.max_answer,
        thinking=not args.no_thinking,
        resume=args.resume,
    )
