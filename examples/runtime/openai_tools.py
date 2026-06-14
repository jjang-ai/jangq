#!/usr/bin/env python3
"""Tool-calling client for a local JANG/JANGTQ OpenAI-compatible server."""

from __future__ import annotations

import argparse
import json
import os

from openai import OpenAI


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City and country or region."},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}


def fake_weather(city: str, unit: str = "celsius") -> str:
    return json.dumps({"city": city, "unit": unit, "summary": "clear", "temperature": 21})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "not-needed"))
    parser.add_argument("--model", default=os.getenv("JANG_MODEL_ALIAS", "default"))
    parser.add_argument("--prompt", default="What is the weather in Paris in celsius?")
    parser.add_argument("--max-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    messages = [{"role": "user", "content": args.prompt}]
    first = client.chat.completions.create(
        model=args.model,
        messages=messages,
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        max_tokens=args.max_tokens,
        temperature=0.0,
    )
    message = first.choices[0].message
    tool_calls = message.tool_calls or []

    if not tool_calls:
        print(message.content or "")
        return 0

    messages.append(message.model_dump(exclude_none=True))
    for call in tool_calls:
        args_obj = json.loads(call.function.arguments or "{}")
        if call.function.name != "get_weather":
            result = json.dumps({"error": f"unknown tool {call.function.name}"})
        else:
            result = fake_weather(**args_obj)
        print(f"tool_call {call.function.name}: {call.function.arguments}")
        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    final = client.chat.completions.create(
        model=args.model,
        messages=messages,
        tools=[WEATHER_TOOL],
        max_tokens=args.max_tokens,
        temperature=0.0,
    )
    print(final.choices[0].message.content or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
