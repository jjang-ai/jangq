"""Speed benchmark for DSV4 Python runtime.

Loads a JANGTQ / JANG_2L / JANGTQ4 bundle, runs prefill + decode on a
fixed prompt, reports prefill tok/s and steady-state decode tok/s.
The end-goal speed-parity comparison: run twice, once on JANG_2L and
once on JANGTQ, and compare decode tok/s. They share the same backbone
architecture so should converge to within 5%.

Usage:
    python3 -m jang_tools.dsv4.bench_speed <bundle> [n_tokens=128] [prompt]
"""
import os, sys, time
sys.stdout.reconfigure(line_buffering=True)

import mlx.core as mx
mx.set_memory_limit(int(os.environ.get("JANG_MEMORY_LIMIT_GB", "200")) * 1024**3)

from jang_tools.load_jangtq import load_jangtq_model
from jang_tools.dsv4.runtime import generate, GenerateOptions

DEFAULT_PROMPT = (
    "def fibonacci(n):\n"
    "    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"
)
DEFAULT_N = 128


def main():
    bundle = sys.argv[1]
    n_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_N
    prompt = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_PROMPT

    print(f"[bench_speed] bundle={bundle}", flush=True)
    print(f"[bench_speed] n_tokens={n_tokens}  prompt_chars={len(prompt)}", flush=True)

    t0 = time.time()
    model, tok = load_jangtq_model(bundle)
    load_dt = time.time() - t0
    print(f"[bench_speed] load: {load_dt:.1f}s", flush=True)

    # Warmup: 1 short generation to JIT-compile router/MoE/rope graphs.
    t1 = time.time()
    res = generate(
        model, tok, bundle, messages=[{"content": "Hello"}],
        opts=GenerateOptions(mode="fim", max_tokens=8, temperature=0.0),
    )
    warmup_dt = time.time() - t1
    print(f"[bench_speed] warmup ({res.n_tokens} tok): {warmup_dt:.2f}s", flush=True)

    # Real bench
    t2 = time.time()
    res = generate(
        model, tok, bundle, messages=[{"content": prompt}],
        opts=GenerateOptions(mode="fim", max_tokens=n_tokens, temperature=0.0),
    )
    bench_dt = time.time() - t2

    n_prompt = res.n_prompt_tokens if hasattr(res, "n_prompt_tokens") else None
    n_gen = res.n_tokens
    decode_tps = n_gen / max(bench_dt, 1e-6)

    print(f"[bench_speed] generated {n_gen} tokens in {bench_dt:.2f}s "
          f"= {decode_tps:.2f} tok/s (load+prefill+decode end-to-end)", flush=True)
    if n_prompt:
        print(f"[bench_speed] prompt tokens: {n_prompt}", flush=True)
    print(f"[bench_speed] finish_reason: {res.finish_reason}", flush=True)


if __name__ == "__main__":
    main()
