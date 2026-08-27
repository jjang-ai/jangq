"""Contract vectors for the Ling-3.0 XML-arg tool parser.

Created by Jinho Jang (eric@jangq.ai). These vectors ARE the spec for the
vmlx-swift port — every case is a shape the model can legally emit.
"""

from jang_tools.ling3.tool_parser import parse_tool_calls

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"},
            "days": {"type": "integer"},
            "metric": {"type": "boolean"},
        }}}},
    {"type": "function", "function": {
        "name": "run_sql",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}}}}},
]


def test_basic_call_with_schema_types():
    text = ("<think>need weather</think>Let me check.\n"
            "<tool_call>get_weather\n"
            "<arg_key>city</arg_key>\n<arg_value>Seoul</arg_value>\n"
            "<arg_key>days</arg_key>\n<arg_value>3</arg_value>\n"
            "<arg_key>metric</arg_key>\n<arg_value>true</arg_value>\n"
            "</tool_call>")
    calls, rest = parse_tool_calls(text, TOOLS)
    assert len(calls) == 1
    c = calls[0]
    assert c.name == "get_weather"
    assert c.arguments == {"city": "Seoul", "days": 3, "metric": True}
    assert "<tool_call>" not in rest and "Let me check." in rest


def test_string_that_looks_like_json_stays_string():
    # schema says `city` is a string; "true" must NOT become a bool
    text = ("<tool_call>get_weather\n"
            "<arg_key>city</arg_key>\n<arg_value>true</arg_value>\n</tool_call>")
    calls, _ = parse_tool_calls(text, TOOLS)
    assert calls[0].arguments == {"city": "true"}


def test_multiline_raw_string_value():
    sql = "SELECT *\nFROM orders\nWHERE total > 10"
    text = (f"<tool_call>run_sql\n<arg_key>query</arg_key>\n"
            f"<arg_value>{sql}</arg_value>\n</tool_call>")
    calls, _ = parse_tool_calls(text, TOOLS)
    assert calls[0].arguments["query"] == sql


def test_multiple_calls_in_one_turn():
    text = ("<tool_call>get_weather\n<arg_key>city</arg_key>\n"
            "<arg_value>Paris</arg_value>\n</tool_call>\n"
            "<tool_call>run_sql\n<arg_key>query</arg_key>\n"
            "<arg_value>SELECT 1</arg_value>\n</tool_call>")
    calls, rest = parse_tool_calls(text, TOOLS)
    assert [c.name for c in calls] == ["get_weather", "run_sql"]
    assert rest.strip() == ""


def test_unknown_function_still_parses():
    text = ("<tool_call>mystery_fn\n<arg_key>x</arg_key>\n"
            "<arg_value>{\"a\": 1}</arg_value>\n</tool_call>")
    calls, _ = parse_tool_calls(text, tools=None)
    assert calls[0].name == "mystery_fn"
    assert calls[0].arguments == {"x": {"a": 1}}


def test_no_args_call():
    calls, _ = parse_tool_calls("<tool_call>list_files\n</tool_call>", TOOLS)
    assert calls[0].name == "list_files"
    assert calls[0].arguments == {}


def test_object_value_via_tojson():
    text = ("<tool_call>mystery\n<arg_key>cfg</arg_key>\n"
            '<arg_value>{"depth": 2, "tags": ["a", "b"]}</arg_value>\n</tool_call>')
    calls, _ = parse_tool_calls(text)
    assert calls[0].arguments["cfg"] == {"depth": 2, "tags": ["a", "b"]}


def test_measured_emission_no_newline_between_arg_pairs():
    """VERBATIM greedy output from the real bf16 model (2026-08-26 live probe).

    Note `</arg_value><arg_key>` with NO separating newline — the model does not
    reliably emit the template's suggested formatting between pairs. Any parser
    that requires whitespace between argument pairs fails on real output.
    """
    text = ("<tool_call>get_weather\n<arg_key>city</arg_key>\n"
            "<arg_value>Seoul</arg_value><arg_key>days</arg_key>\n"
            "<arg_value>3</arg_value>\n</tool_call>")
    calls, _ = parse_tool_calls(text, TOOLS)
    assert calls[0].name == "get_weather"
    assert calls[0].arguments == {"city": "Seoul", "days": 3}


def test_real_osaurus_catalog_schemas_do_not_crash():
    """The full 81-tool osaurus catalog contains non-dict property values;
    schema extraction must degrade to untyped decoding, never crash."""
    tools = [{
        "type": "function",
        "function": {
            "name": "file_read",
            "parameters": {
                "type": "object",
                "properties": {"path": "string"},
            },
        },
    }]
    text = ("<tool_call>file_read\n<arg_key>path</arg_key>\n"
            "<arg_value>a.txt</arg_value>\n</tool_call>")
    calls, _ = parse_tool_calls(text, tools)
    assert calls[0].name == "file_read"
    assert calls[0].arguments == {"path": "a.txt"}


def test_trailing_newline_in_string_value_preserved():
    """Template renders values verbatim — a trailing newline IS the value.
    (Caught by the datagen gate: file_write content ending in \\n was corrupted.)"""
    text = ("<tool_call>run_sql\n<arg_key>query</arg_key>\n"
            "<arg_value>line1\n</arg_value>\n</tool_call>")
    calls, _ = parse_tool_calls(text, TOOLS)
    assert calls[0].arguments["query"] == "line1\n"
