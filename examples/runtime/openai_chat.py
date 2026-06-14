#!/usr/bin/env python3
"""Chat with a locally served JANG/JANGTQ bundle.

Start the server first:
    JANG_MODEL=/path/to/model ./examples/runtime/serve_vmlx.sh
"""

from __future__ import annotations

import argparse
import os
import sys

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "not-needed"))
    parser.add_argument("--model", default=os.getenv("JANG_MODEL_ALIAS", "default"))
    parser.add_argument("--prompt", default="Explain JANG in one paragraph.")
    parser.add_argument("--system", default="You are a precise technical assistant.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--stream", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]

    if args.stream:
        stream = client.chat.completions.create(
            model=args.model,
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
            if reasoning:
                print(reasoning, end="", file=sys.stderr, flush=True)
            if delta.content:
                print(delta.content, end="", flush=True)
        print()
        return 0

    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    msg = response.choices[0].message
    reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
    if reasoning:
        print("Reasoning:")
        print(reasoning)
        print()
    print(msg.content or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
