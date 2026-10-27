"""Example runtime: chat / generate with a DSV4.1-Flash JANG bundle on Apple Silicon.

  python -m jang_tools.dsv41.generate --bundle ~/models/DeepSeek-V4.1-Flash-1017-JANG \
      --prompt "Write a Rust function that parses an IPv4 header." --effort high

  python -m jang_tools.dsv41.generate --bundle ... --chat           # interactive, multi-turn

Prompts are rendered with the release's own encoding/encoding.py (shipped in the
bundle). Thinking mode is the default; --effort takes low/high/max or 1-100.
Engram tables are read from <bundle>/engram unless --engram-dir points elsewhere
(e.g. the original download directory).

Sampling defaults match vMLX serving this bundle: temperature / top_p / top_k from the bundle's
generation_config.json when it declares them (do_sample false: greedy), else temperature 1.0 and
top_p 1.0 (DeepSeek's recommendation). --temperature / --top-p / --top-k override them. Tokens are
drawn by mlx_lm's sampler: the top_p / top_k filters on log-probabilities, then temperature.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

from jang_tools.dsv41.model import DSV41


def load_encoding(bundle: Path):
    path = bundle / "encoding" / "encoding.py"
    spec = importlib.util.spec_from_file_location("dsv41_encoding", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sampling_defaults(bundle: Path) -> dict:
    """{"temperature", "top_p", "top_k"} as vMLX serves the bundle: generation_config.json values when
    declared (do_sample false: greedy), else temperature 1.0 / top_p 1.0 / top_k 0 (off)."""
    out = {"temperature": 1.0, "top_p": 1.0, "top_k": 0}
    path = bundle / "generation_config.json"
    if not path.is_file():
        return out
    with open(path, encoding="utf-8") as f:
        gen = json.load(f)
    if not isinstance(gen, dict):
        return out
    if gen.get("do_sample") is False:
        return {"temperature": 0.0, "top_p": 1.0, "top_k": 0}
    for key, cast in (("temperature", float), ("top_p", float), ("top_k", int)):
        v = gen.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = cast(v)
    return out


def make_sampler(temperature: float, top_p: float, top_k: int):
    """mlx_lm's sampler (temperature 0: argmax; else top_p, then top_k on log-probabilities, then
    temperature), the same function vMLX builds for AR decode."""
    from mlx_lm.sample_utils import make_sampler as mlx_lm_sampler

    return mlx_lm_sampler(temp=temperature, top_p=top_p, top_k=top_k)


def sample(logits: mx.array, sampler) -> int:
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return int(sampler(logprobs[None])[0])


def generate(m: DSV41, prompt_ids: list[int], max_tokens: int, sampler, stop_ids: set[int], stream=True):
    t0 = time.perf_counter()
    logits = m.forward(prompt_ids)
    mx.eval(logits)
    t_prefill = time.perf_counter() - t0
    out: list[int] = []
    printed = ""
    t1 = time.perf_counter()
    for _ in range(max_tokens):
        tok = sample(logits, sampler)
        if tok in stop_ids:
            out.append(tok)
            break
        out.append(tok)
        if stream:
            text = m.tokenizer.decode(out, skip_special_tokens=False)
            if not text.endswith("�"):
                sys.stdout.write(text[len(printed):])
                sys.stdout.flush()
                printed = text
        logits = m.forward([tok])
    t_decode = time.perf_counter() - t1
    if stream:
        print()
    stats = {
        "prompt_tokens": len(prompt_ids),
        "prefill_s": round(t_prefill, 3),
        "prefill_tok_s": round(len(prompt_ids) / t_prefill, 1),
        "gen_tokens": len(out),
        "decode_tok_s": round(len(out) / max(t_decode, 1e-9), 2),
        "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 2),
    }
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--engram-dir")
    ap.add_argument("--prompt")
    ap.add_argument("--system")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--mode", choices=["thinking", "chat"], default="thinking")
    ap.add_argument("--effort", default="high", help="low | high | max | 1-100 (thinking mode)")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, help="default: bundle generation_config.json, else 1.0")
    ap.add_argument("--top-p", type=float, help="default: bundle generation_config.json, else 1.0 (off)")
    ap.add_argument("--top-k", type=int, help="default: bundle generation_config.json, else 0 (off)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    bundle = Path(a.bundle).expanduser()
    enc = load_encoding(bundle)
    effort = int(a.effort) if a.effort.isdigit() else a.effort
    mx.random.seed(a.seed)
    samp = sampling_defaults(bundle)
    for key in samp:
        if getattr(a, key) is not None:
            samp[key] = getattr(a, key)
    sampler = make_sampler(samp["temperature"], samp["top_p"], samp["top_k"])
    print(f"[sampling {samp}]", file=sys.stderr)

    t = time.perf_counter()
    m = DSV41(bundle, a.engram_dir)
    print(f"[loaded in {time.perf_counter() - t:.1f}s, active memory {mx.get_active_memory() / 1e9:.1f} GB]",
          file=sys.stderr)
    eos = m.tokenizer.token_to_id("<｜end▁of▁sentence｜>")
    stop = {eos, m.eos_id}

    messages = []
    if a.system:
        messages.append({"role": "system", "content": a.system})

    def turn(user_text: str):
        messages.append({"role": "user", "content": user_text})
        prompt = enc.encode_messages(messages, thinking_mode=a.mode, reasoning_effort=effort)
        ids = m.tokenizer.encode(prompt, add_special_tokens=False).ids
        m.reset()  # the release runtime prefills from position 0; re-prefill the whole conversation
        out, stats = generate(m, ids, a.max_tokens, sampler, stop)
        completion = m.tokenizer.decode(out, skip_special_tokens=False)
        try:
            msg = enc.parse_message_from_completion_text(completion, thinking_mode=a.mode)
        except (ValueError, AssertionError, IndexError, KeyError):  # not a parseable assistant turn: keep the raw text
            msg = {"role": "assistant", "content": completion}
        messages.append(msg)
        print(f"[{stats}]", file=sys.stderr)

    if a.prompt:
        turn(a.prompt)
    if a.chat:
        while True:
            try:
                user = input("\n>>> ")
            except EOFError:
                break
            if user.strip() in ("/exit", "/quit"):
                break
            if user.strip():
                turn(user)


if __name__ == "__main__":
    main()
