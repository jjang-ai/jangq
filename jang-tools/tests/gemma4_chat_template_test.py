"""Prompt-boundary contracts, without loading MLX or model weights."""

import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from jinja2 import Environment

from jang_tools.gemma4_chat_template import _patch_chat_template
from jang_tools.gemma4_chat_template import _TOOL_CHOICE_ANCHOR, _TOOL_CHOICE_PATCH
from jang_tools.gemma4_chat_template import _EMPTY_NO_THINK_TAIL, _SAFE_NO_THINK_TAIL
from jang_tools.gemma4_chat_template import patch_bundle_chat_template


OLD_CLOSE = """{%- if ns.prev_message_type == 'tool_call' and not ns_tr_out.flag -%}
            {{- '<|tool_response>' -}}
        {%- elif not (ns_tr_out.flag and not has_content) -%}
            {{- '<turn|>\\n' -}}
        {%- endif -%}"""

# A small original fixture for the published Gemma turn-ending states. The
# full installed templates are exercised separately by the runtime controls.
FIXTURE = """{%- set ns = namespace(prev_message_type=None) -%}
{%- set loop_messages = messages -%}
{%- for message in loop_messages -%}
  {%- if message.role != 'tool' -%}
    {%- set role = 'model' if message.role == 'assistant' else message.role -%}
    {%- set ns.prev_message_type = 'tool_response' if message.get('result') else ('tool_call' if message.get('tool_calls') else None) -%}
    {%- set ns_tr_out = namespace(flag=message.get('result', false)) -%}
    {%- set has_content = message.get('content', '') | trim | length > 0 -%}
    {{- message.get('content', '') -}}
    {%- if ns_tr_out.flag -%}{{- '<tool_response|>' -}}{%- endif -%}
""" + OLD_CLOSE + """
  {%- endif -%}
{%- endfor -%}"""


def render(messages, source=FIXTURE, **kwargs):
    return Environment().from_string(_patch_chat_template(source)).render(
        messages=messages, **kwargs
    )


class Gemma4ChatTemplateTests(unittest.TestCase):
    def test_tool_result_turn_closes_before_new_user(self):
        for thinking in [False, True]:
            for tool_count in [1, 2]:
                with self.subTest(thinking=thinking, tool_count=tool_count):
                    messages = [
                        {"role": "assistant", "content": "", "result": True, "tool_calls": [{}]},
                        *[{"role": "tool", "content": "result"} for _ in range(tool_count)],
                        {"role": "user", "content": "Next user"},
                    ]
                    self.assertEqual(render(messages, enable_thinking=thinking),
                                     "<tool_response|><turn|>\nNext user<turn|>\n")

    def test_tool_result_stays_open_for_assistant_continuation(self):
        self.assertEqual(render([
            {"role": "assistant", "content": "", "result": True, "tool_calls": [{}]},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "Answer"},
        ]), "<tool_response|>Answer<turn|>\n")

    def test_persisted_runtime_rewrite_closes_before_new_user(self):
        source = FIXTURE.replace(
            "{%- elif not (ns_tr_out.flag and not has_content) -%}",
            "{%- elif not ns_tr_out.flag -%}",
        )
        self.assertEqual(render([
            {"role": "assistant", "content": "", "result": True, "tool_calls": [{}]},
            {"role": "tool", "content": "result"},
            {"role": "user", "content": "Next user"},
        ], source=source), "<tool_response|><turn|>\nNext user<turn|>\n")

    def test_final_tool_result_stays_open(self):
        self.assertEqual(render([
            {"role": "assistant", "content": "", "result": True, "tool_calls": [{}]},
            {"role": "tool", "content": "result"},
        ]), "<tool_response|>")

    def test_plain_turns_still_close(self):
        self.assertEqual(render([{"role": "user", "content": "Hello"}]), "Hello<turn|>\n")

    def test_assistant_fragments_form_one_model_turn(self):
        self.assertEqual(render([
            {"role": "assistant", "content": "First "},
            {"role": "assistant", "content": "answer"},
        ]), "First answer<turn|>\n")

    def test_pending_call_keeps_native_tool_response_open(self):
        self.assertEqual(render([
            {"role": "assistant", "content": "Call", "tool_calls": [{}]},
        ]), "Call<|tool_response>")

    def test_final_result_with_visible_content_closes(self):
        self.assertEqual(render([
            {"role": "assistant", "content": "Visible", "result": True, "tool_calls": [{}]},
        ]), "Visible<tool_response|><turn|>\n")

    def test_export_does_not_invent_required_tool_instructions(self):
        source = """        {%- endfor %}
        {%- set ns.prev_message_type = 'tool' -%}
    {%- endif -%}"""
        self.assertEqual(_patch_chat_template(source), source)

    def test_legacy_injected_instruction_is_removed(self):
        self.assertEqual(_patch_chat_template(_TOOL_CHOICE_PATCH), _TOOL_CHOICE_ANCHOR)

    def test_patching_is_idempotent(self):
        once = _patch_chat_template(FIXTURE)
        self.assertEqual(_patch_chat_template(once), once)

    def test_existing_no_thinking_prefill_policy_is_preserved(self):
        for previous in [None, "tool_response", "tool_call"]:
            values = dict(ns={"prev_message_type": previous},
                          add_generation_prompt=True, enable_thinking=False)
            self.assertEqual(
                Environment().from_string(_patch_chat_template(_EMPTY_NO_THINK_TAIL)).render(**values),
                Environment().from_string(_SAFE_NO_THINK_TAIL).render(**values),
            )

    def test_final_tool_result_uses_canonical_thinking_prefill(self):
        # Google's canonical template opens the requested thought channel after
        # tool results. This is a template prefix, not generated-output repair.
        for content in ["", "Visible"]:
            with self.subTest(content=content):
                value = render([
                    {"role": "assistant", "content": content,
                     "result": True, "tool_calls": [{}]},
                ], source=FIXTURE + _SAFE_NO_THINK_TAIL,
                   enable_thinking=True, add_generation_prompt=True)
                self.assertTrue(value.endswith("<|channel>thought\n"), value)

    def test_thinking_prefill_is_only_for_pending_tool_continuation(self):
        source = FIXTURE + _SAFE_NO_THINK_TAIL
        for add_prompt in [False, True]:
            result = render([
                {"role": "assistant", "content": "", "result": True,
                 "tool_calls": [{}]},
                {"role": "user", "content": "Next user"},
            ], source=source, enable_thinking=True, add_generation_prompt=add_prompt)
            self.assertNotIn("<|channel>thought", result)
        pending = render([
            {"role": "assistant", "content": "Call", "tool_calls": [{}]},
        ], source=source, enable_thinking=True, add_generation_prompt=True)
        self.assertEqual(pending, "Call<|tool_response>")

    def test_legacy_content_is_rendered_after_tool_results(self):
        old_content = """    {%- set has_content = message.get('content', '') | trim | length > 0 -%}
    {{- message.get('content', '') -}}"""
        legacy_capture = """    {%- set captured_content -%}
        {{- message.get('content', '') -}}
    {%- endset -%}
    {{- captured_content -}}
    {%- set has_content = captured_content | trim | length > 0 -%}
    {%- if message['tool_calls'] -%}
        {{- '<|tool_call>call:calculate{}<tool_call|>' -}}
    {%- endif -%}"""
        source = FIXTURE.replace(old_content, legacy_capture) + _SAFE_NO_THINK_TAIL
        self.assertNotEqual(source, FIXTURE + _SAFE_NO_THINK_TAIL)
        value = render([
            {"role": "assistant", "content": "Visible", "result": True,
             "tool_calls": [{}]},
        ], source=source, enable_thinking=True, add_generation_prompt=True)
        self.assertEqual(value,
                         '<|tool_call>call:calculate{}<tool_call|><tool_response|>'
                         'Visible<turn|>\n<|channel>thought\n')
        self.assertEqual(_patch_chat_template(_patch_chat_template(source)),
                         _patch_chat_template(source))


class Gemma4BundleTemplateTests(unittest.TestCase):
    def test_file_and_embedded_template_share_the_correction(self):
        with TemporaryDirectory() as directory:
            bundle = Path(directory)
            (bundle / "chat_template.jinja").write_text(FIXTURE)
            (bundle / "tokenizer_config.json").write_text(json.dumps({
                "chat_template": FIXTURE, "extra": "retained", "eos_token_id": 1,
            }))
            defaults = '{"temperature":1.0,"top_p":0.95,"eos_token_id":[2,3]}'
            (bundle / "generation_config.json").write_text(defaults)
            patch_bundle_chat_template(bundle)
            source = (bundle / "chat_template.jinja").read_text()
            config = json.loads((bundle / "tokenizer_config.json").read_text())
            self.assertEqual(source, config["chat_template"])
            self.assertNotEqual(source, FIXTURE)
            self.assertEqual(config["extra"], "retained")
            self.assertEqual(config["eos_token_id"], [2, 3])
            self.assertEqual((bundle / "generation_config.json").read_text(), defaults)
            before = {p.name: p.read_bytes() for p in bundle.iterdir()}
            patch_bundle_chat_template(bundle)
            self.assertEqual({p.name: p.read_bytes() for p in bundle.iterdir()}, before)

    def test_jinja_only_does_not_invent_tokenizer_configuration(self):
        with TemporaryDirectory() as directory:
            bundle = Path(directory)
            (bundle / "chat_template.jinja").write_text(FIXTURE)
            patch_bundle_chat_template(bundle)
            self.assertFalse((bundle / "tokenizer_config.json").exists())
            self.assertNotEqual((bundle / "chat_template.jinja").read_text(), FIXTURE)

    def test_absent_template_leaves_other_files_unchanged(self):
        with TemporaryDirectory() as directory:
            bundle = Path(directory)
            tokenizer = bundle / "tokenizer_config.json"
            tokenizer.write_text('{"keep":"exact bytes"}')
            patch_bundle_chat_template(bundle)
            self.assertEqual(tokenizer.read_text(), '{"keep":"exact bytes"}')


if __name__ == "__main__":
    unittest.main()
