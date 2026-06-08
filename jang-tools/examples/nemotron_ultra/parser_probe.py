"""Parse saved Nemotron Ultra generation rows for reasoning/tool readiness.

This probe does not load model weights. It consumes the JSON logs written by
the heavyweight coherence probes and records how the runtime parser contract
would split visible content, reasoning content, and XML function tool calls.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from jang_tools.reasoning.deepseek_r1_parser import DeepSeekR1ReasoningParser


THINK_LEAK_TAGS = ("<think>", "</think>")
TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
PARAM_RE = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)


def parse_xml_function_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    for match in TOOL_CALL_RE.finditer(text):
        args = {
            name: value.strip()
            for name, value in PARAM_RE.findall(match.group(2))
        }
        calls.append(
            {
                "type": "function",
                "function": {
                    "name": match.group(1),
                    "arguments": json.dumps(args, sort_keys=True),
                },
            }
        )
    return TOOL_CALL_RE.sub("", text).strip(), calls


def strip_visible_think_markers(text: str | None) -> tuple[str | None, list[str]]:
    if text is None:
        return None, []
    leaks = [tag for tag in THINK_LEAK_TAGS if tag in text]
    clean = text
    for tag in THINK_LEAK_TAGS:
        clean = clean.replace(tag, "")
    clean = clean.strip()
    return clean or None, leaks


def parse_row(row: dict[str, Any]) -> dict[str, Any]:
    raw = row["text"]
    if row.get("enable_thinking"):
        parser = DeepSeekR1ReasoningParser()
        parser.reset_state(think_in_prompt=True)
        reasoning, content = parser.extract_reasoning(raw)
    else:
        reasoning, content = None, raw
    content, calls = parse_xml_function_calls(content or "")
    content, leaks = strip_visible_think_markers(content)
    return {
        "id": row["id"],
        "enable_thinking": row.get("enable_thinking"),
        "generated_tokens": row.get("generated_tokens"),
        "ttft_s": row.get("ttft_s"),
        "decode_tps_excluding_first": row.get("decode_tps_excluding_first"),
        "raw_text": raw,
        "reasoning_content": reasoning,
        "content": content,
        "tool_calls": calls,
        "visible_think_marker_leaks": leaks,
        "truncated_reasoning": bool(reasoning and not content and not calls),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--log",
        type=Path,
        default=Path("docs/runtime/logs/2026-06-04-nemotron-ultra-jangtq1l-coherence-speed-probe.json"),
    )
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    data = json.loads(args.log.read_text())
    parsed = {
        "source_log": str(args.log),
        "bundle": data.get("bundle"),
        "parser": "deepseek_r1 compatible <think> parser + Ultra XML function calls",
        "rows": [parse_row(row) for row in data.get("rows", [])],
    }
    text = json.dumps(parsed, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
