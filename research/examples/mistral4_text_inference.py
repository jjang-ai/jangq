"""
Mistral Small 4 (119B) — Text Inference Example
JANG quantized model on Apple Silicon via MLX

Requirements:
  pip install mlx mlx-lm transformers safetensors
  pip install jang  # or add jang-tools to path

Patches required on inference machine:
  - deepseek_v2.py: 7 patches (rope, scale, norm_topk_prob, llama4 scaling)
  - mistral3.py: mistral4 routing to deepseek_v2
  See research/MISTRAL4-INFERENCE-GUIDE.md for details.

Created by Jinho Jang (eric@jangq.ai)
"""
import sys
sys.path.insert(0, "/Users/eric/jang/jang-tools")  # Adjust path

from jang_tools.loader import load_jang_model
from mlx_lm import generate

MODEL_PATH = "/Users/eric/models/Mistral-Small-4-119B-JANG_2L"

# Load model (auto-detects bfloat16 for MLA models)
print("Loading model...")
model, tokenizer = load_jang_model(MODEL_PATH)
print(f"Loaded. Use model.set_dtype() was auto-applied for bfloat16.")


def chat(message, max_tokens=200, reasoning=False):
    """Generate a response with optional reasoning mode."""
    messages = [{"role": "user", "content": message}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if reasoning:
        prompt = prompt.replace(
            'reasoning_effort": "none', 'reasoning_effort": "high'
        )
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=True)


if __name__ == "__main__":
    # Basic Q&A
    print("\n=== Knowledge ===")
    print(chat("What is the capital of France?"))

    # Math
    print("\n=== Math ===")
    print(chat("What is 15 * 23?"))

    # Code
    print("\n=== Code ===")
    print(chat("Write a Python function to check if a number is prime.", max_tokens=300))

    # Reasoning mode with [THINK] tags
    print("\n=== Reasoning (with [THINK] tags) ===")
    print(chat("What is 17 * 24? Think step by step.", max_tokens=500, reasoning=True))

    # Long explanation
    print("\n=== Explanation ===")
    print(chat("Explain how neural networks learn through backpropagation.", max_tokens=400))
