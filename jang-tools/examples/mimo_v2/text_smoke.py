"""Short MiMo-V2 JANG text smoke.

This is intentionally a runtime diagnostic, not a benchmark or release gate.
It uses the bundle's tokenizer template and exposes the sink-ablation path via
JANG_MIMO_DISABLE_SINK=1.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import stream_generate, wired_limit
from mlx_lm.utils import load
from mlx.utils import tree_reduce

from jang_tools.mimo_v2 import mlx_register  # noqa: F401


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--prompt", default="What is 2 + 2? Answer in one short sentence.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--thinking", choices=("default", "on", "off"), default="default",
                        help="Pass enable_thinking to the chat template when supported.")
    parser.add_argument("--no-cache-greedy", action="store_true",
                        help="Recompute the full sequence each token to isolate cache bugs.")
    args = parser.parse_args()

    print(f"bundle={args.bundle}")
    print(f"disable_sink={os.environ.get('JANG_MIMO_DISABLE_SINK') in {'1', 'true', 'yes'}}")
    t0 = time.time()
    model, tokenizer = load(str(args.bundle), lazy=True, tokenizer_config={"trust_remote_code": True})
    print(f"load_sec={time.time() - t0:.2f}")
    model_bytes = tree_reduce(
        lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc,
        model,
        0,
    )
    max_rec = mx.device_info().get("max_recommended_working_set_size", 0)
    if model_bytes and max_rec:
        print(f"model_mb={model_bytes // 2**20}")
        print(f"max_recommended_mb={max_rec // 2**20}")

    messages = [{"role": "user", "content": args.prompt}]
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if args.thinking == "on":
        template_kwargs["enable_thinking"] = True
    elif args.thinking == "off":
        template_kwargs["enable_thinking"] = False
    prompt = tokenizer.apply_chat_template(messages, **template_kwargs)

    t1 = time.time()
    if args.no_cache_greedy:
        with wired_limit(model):
            ids = tokenizer.encode(prompt)
            for _ in range(args.max_tokens):
                logits = model(mx.array([ids], dtype=mx.int32), cache=None)
                mx.eval(logits)
                ids.append(int(mx.argmax(logits[0, -1, :]).item()))
        output = tokenizer.decode(ids)
    else:
        pieces = []
        last_response = None
        for response in stream_generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens):
            pieces.append(response.text)
            last_response = response
        output = "".join(pieces)
        if last_response is not None:
            print(f"prompt_tokens={last_response.prompt_tokens}")
            print(f"prompt_tps={last_response.prompt_tps:.3f}")
            print(f"generation_tokens={last_response.generation_tokens}")
            print(f"generation_tps={last_response.generation_tps:.3f}")
            print(f"peak_memory_gb={last_response.peak_memory:.3f}")
    elapsed = time.time() - t1
    print(f"decode_sec={elapsed:.2f}")
    print(f"tok_per_sec={args.max_tokens / elapsed:.3f}")
    print("output:")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
