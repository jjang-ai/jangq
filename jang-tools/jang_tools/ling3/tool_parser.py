"""Reference parser for Ling-3.0's XML-arg tool-call dialect.

Created by Jinho Jang (eric@jangq.ai) — 2026-08-26.

The model emits (per its own chat template's instruction):

    <tool_call>{function-name}
    <arg_key>{k1}</arg_key>
    <arg_value>{v1}</arg_value>
    <arg_key>{k2}</arg_key>
    <arg_value>{v2}</arg_value>
    </tool_call>

Properties a parser MUST handle, each demonstrated in tests:

* The function name is BARE TEXT between `<tool_call>` and the first newline —
  there is no name tag.
* Values are emitted RAW when the argument is a string and `tojson` otherwise
  (the template's own emission rule), so decoding must try JSON first and fall
  back to the raw string. `"true"`-the-string vs `true`-the-bool is decided by
  the tool's schema when available.
* Multiple `<tool_call>` blocks may appear in one assistant turn.
* Values may span lines (multiline strings are legal raw values).
* Reasoning (`<think>…</think>`) precedes calls and must not confuse parsing.

This is the Python reference for the vmlx-swift port; the test vectors in
`tests/test_ling3_tool_parser.py` are the contract.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_ARG_RE = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


def _decode_value(raw: str, schema_type: str | None) -> Any:
    """Template rule inverted: strings were emitted raw, everything else tojson.

    With a schema, the declared type decides. Without one, try JSON and keep the
    raw string when it does not parse — the safe default, because a raw string
    that happens to look like JSON (e.g. a user message "true") must stay a
    string unless the schema says otherwise.
    """
    # NO stripping: the template renders string values VERBATIM between the
    # tags (confirmed against chat_template.jinja lines 93-103), so a leading/
    # trailing newline is part of the value. Stripping here silently corrupts
    # e.g. file_write content ending in "\n" — caught by the datagen gate.
    if schema_type == "string":
        return raw
    if schema_type in ("number", "integer", "boolean", "object", "array", None):
        try:
            v = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return raw
        if schema_type is None:
            # unschema'd: only accept JSON decoding for non-string results;
            # a quoted string was tojson-emitted and should decode, but a bare
            # word stays a bare word (json.loads would reject it anyway).
            return v
        return v
    return raw


def parse_tool_calls(
    text: str,
    tools: list[dict] | None = None,
) -> tuple[list[ToolCall], str]:
    """Extract tool calls; return (calls, text with the call blocks removed).

    `tools` is the OpenAI-style list the request carried; used to type argument
    values. Unknown functions still parse (the harness decides what to do).
    """
    schemas: dict[str, dict] = {}
    for t in tools or []:
        fn = t.get("function", t)
        props = (fn.get("parameters") or {}).get("properties") or {}
        # Real-world schemas (e.g. the osaurus catalog) are not uniformly
        # {name: {type: ...}} — values can be strings, $refs, or nested unions.
        # Anything we can't read a plain type from degrades to None (= JSON-try-
        # then-raw decoding), never to a crash.
        out: dict[str, str | None] = {}
        for k, v in props.items():
            ty = v.get("type") if isinstance(v, dict) else None
            out[k] = ty if isinstance(ty, str) else None
        schemas[str(fn.get("name"))] = out

    calls: list[ToolCall] = []
    for m in _CALL_RE.finditer(text):
        body = m.group(1)
        # bare function name = everything before the first <arg_key> (or the
        # whole body when there are no args), trimmed
        first_arg = body.find("<arg_key>")
        name = (body if first_arg < 0 else body[:first_arg]).strip()
        types = schemas.get(name, {})
        args: dict[str, Any] = {}
        for k, v in _ARG_RE.findall(body):
            key = k.strip()
            args[key] = _decode_value(v, types.get(key))
        calls.append(ToolCall(name=name, arguments=args))

    return calls, _CALL_RE.sub("", text)
