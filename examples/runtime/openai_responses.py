#!/usr/bin/env python3
"""Use the OpenAI-compatible Responses API against vmlx-engine."""

from __future__ import annotations

import argparse
import json
import os

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "not-needed"))
    parser.add_argument("--model", default=os.getenv("JANG_MODEL_ALIAS", "default"))
    parser.add_argument("--prompt", default="What is 19 * 23?")
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--stream", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    headers = {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}
    payload = {
        "model": args.model,
        "input": args.prompt,
        "max_output_tokens": args.max_output_tokens,
        "stream": args.stream,
    }
    url = args.base_url.rstrip("/") + "/responses"

    if args.stream:
        with httpx.stream("POST", url, headers=headers, json=payload, timeout=None) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    print(line)
        return 0

    response = httpx.post(url, headers=headers, json=payload, timeout=None)
    response.raise_for_status()
    data = response.json()
    if "output_text" in data:
        print(data["output_text"])
    else:
        print(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
