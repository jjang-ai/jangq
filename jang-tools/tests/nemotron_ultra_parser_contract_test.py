import json
import re

from examples.nemotron_ultra.parser_probe import parse_row
from jang_tools.reasoning.deepseek_r1_parser import DeepSeekR1ReasoningParser
from jang_tools.reasoning.qwen3_parser import Qwen3ReasoningParser


def _parse_ultra_xml_tool_calls(text: str):
    pattern = re.compile(
        r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>",
        re.DOTALL,
    )
    calls = []
    for match in pattern.finditer(text):
        args = {}
        for name, value in re.findall(
            r"<parameter=([^>]+)>(.*?)</parameter>", match.group(2), flags=re.DOTALL
        ):
            args[name] = value.strip()
        calls.append(
            {
                "type": "function",
                "function": {
                    "name": match.group(1),
                    "arguments": json.dumps(args, sort_keys=True),
                },
            }
        )
    return pattern.sub("", text).strip(), calls


def test_ultra_think_xml_reasoning_parser_handles_closed_and_truncated_rows():
    parser = Qwen3ReasoningParser()

    reasoning, content = parser.extract_reasoning("<think>check</think>4")
    assert reasoning == "check"
    assert content == "4"

    parser.reset_state(think_in_prompt=True)
    reasoning, content = parser.extract_reasoning("First compute 2+2")
    assert reasoning == "First compute 2+2"
    assert content is None


def test_ultra_deepseek_alias_matches_implicit_closing_think_case():
    qwen = Qwen3ReasoningParser()
    deepseek = DeepSeekR1ReasoningParser()
    output = "scratch work</think>Tokyo is the capital of Japan."

    assert qwen.extract_reasoning(output) == deepseek.extract_reasoning(output)
    assert qwen.extract_reasoning(output) == (
        "scratch work",
        "Tokyo is the capital of Japan.",
    )


def test_ultra_tool_call_format_is_xml_function_not_qwen_json_body():
    content, calls = _parse_ultra_xml_tool_calls(
        "Need weather.\n"
        "<tool_call>\n"
        "<function=get_weather>\n"
        "<parameter=city>\nTokyo\n</parameter>\n"
        "<parameter=unit>celsius</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )

    assert content == "Need weather."
    assert calls == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "Tokyo", "unit": "celsius"}',
            },
        }
    ]


def test_ultra_no_thinking_stray_close_stays_visible_content_not_reasoning():
    parsed = parse_row(
        {
            "id": "nt_capital_default",
            "enable_thinking": False,
            "generated_tokens": 24,
            "ttft_s": 1.342,
            "decode_tps_excluding_first": 3.303,
            "text": "Tokyo is the capital of Japan.</think>Tokyo is the capital of Japan.",
        }
    )

    assert parsed["reasoning_content"] is None
    assert parsed["content"] == "Tokyo is the capital of Japan.Tokyo is the capital of Japan."
    assert parsed["visible_think_marker_leaks"] == ["</think>"]


def test_ultra_thinking_truncated_saved_row_is_reasoning_not_visible_content():
    parsed = parse_row(
        {
            "id": "think_math_default",
            "enable_thinking": True,
            "generated_tokens": 32,
            "ttft_s": 1.503,
            "decode_tps_excluding_first": 3.256,
            "text": "The user is asking a simple arithmetic question",
        }
    )

    assert parsed["reasoning_content"] == "The user is asking a simple arithmetic question"
    assert parsed["content"] is None
    assert parsed["truncated_reasoning"] is True
