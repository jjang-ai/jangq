"""Gemma 4 bundle chat-template compatibility, independent of weight conversion."""

import json
from pathlib import Path

_TOOL_CHOICE_ANCHOR = """        {%- endfor %}
        {%- set ns.prev_message_type = 'tool' -%}
    {%- endif -%}"""

_TOOL_CHOICE_PATCH = """        {%- endfor %}
        {%- set ns.prev_message_type = 'tool' -%}
        {%- if tool_choice is defined and ((tool_choice is string and tool_choice == 'required') or (tool_choice is mapping and tool_choice.get('type') == 'required')) -%}
            {{- '\\nTool use is REQUIRED for this turn: you must call exactly one of the declared tools. Output only the tool call, and copy every argument value verbatim \\u2014 character for character, preserving punctuation and newlines, with no added or removed whitespace.' -}}
        {%- endif -%}
    {%- endif -%}"""

_EMPTY_NO_THINK_TAIL = """{%- if add_generation_prompt -%}
    {%- if ns.prev_message_type != 'tool_response' and ns.prev_message_type != 'tool_call' -%}
        {{- '<|turn>model\\n' -}}
        {%- if not enable_thinking | default(false) -%}
            {{- '<|channel>thought\\n<channel|>' -}}
        {%- endif -%}
    {%- endif -%}
{%- endif -%}"""

_SAFE_NO_THINK_TAIL = """{%- if add_generation_prompt -%}
    {%- if ns.prev_message_type != 'tool_response' and ns.prev_message_type != 'tool_call' -%}
        {{- '<|turn>model\\n' -}}
    {%- endif -%}
{%- endif -%}"""

def _remove_legacy_tool_choice_instruction(text: str) -> str:
    """Undo only the exact instruction older exports injected, not user text."""
    return text.replace(_TOOL_CHOICE_PATCH, _TOOL_CHOICE_ANCHOR, 1)


_OLD_TURN_END = """{%- if ns.prev_message_type == 'tool_call' and not ns_tr_out.flag -%}
            {{- '<|tool_response>' -}}
        {%- elif not (ns_tr_out.flag and not has_content) -%}
            {{- '<turn|>\\n' -}}
        {%- endif -%}"""

# Some bundles persisted the former runtime's unconditional-open rewrite.
# That variant also needs the next-role decision, not another blanket close.
_UNCONDITIONAL_OPEN_TURN_END = _OLD_TURN_END.replace(
    "{%- elif not (ns_tr_out.flag and not has_content) -%}",
    "{%- elif not ns_tr_out.flag -%}",
)

# Google Gemma 4 canonical template, 2026-07-09. This is the model's
# turn-boundary contract, not an instruction or a generated-output repair.
# Reference revision: google/gemma-4-12b-it@43c9d014b231474fa7212dbb48a762566c2b255d.
_CANONICAL_TURN_END = """{%- set next_nt = namespace(role=None, found=false) -%}
        {%- for j in range(loop.index0 + 1, loop_messages | length) -%}
            {%- if not next_nt.found and loop_messages[j]['role'] != 'tool' -%}
                {%- set next_nt.role = loop_messages[j]['role'] -%}
                {%- set next_nt.found = true -%}
            {%- endif -%}
        {%- endfor -%}
        {%- set continues_into_next = (
            role == 'model'
            and next_nt.role == 'assistant'
            and (not message.get('tool_calls') or ns_tr_out.flag)
        ) -%}
        {%- if ns.prev_message_type == 'tool_call' and not ns_tr_out.flag -%}
            {{- '<|tool_response>' -}}
        {%- elif continues_into_next -%}
        {%- elif not (ns_tr_out.flag and not has_content and not next_nt.found) -%}
            {{- '<turn|>\\n' -}}
        {%- endif -%}"""

def _patch_chat_template_no_thinking_tail(text: str) -> str:
    """Remove Gemma 4's empty no-thinking thought-channel generation prefill."""
    return text.replace(_EMPTY_NO_THINK_TAIL, _SAFE_NO_THINK_TAIL, 1)

def _patch_chat_template(text: str) -> str:
    """Preserve Gemma's boundary contract without inventing tool instructions.

    INVARIANT: a tool-result turn closes before the next user, but stays open
    for an assistant continuation or a pending generation after final results.
    Apply this known upstream correction at export, never by replacing a valid
    bundle template inside the runtime. Both embedded/file copies are written
    by the shared sidecar exporter. Unknown template layouts remain untouched.
    """
    text = _patch_chat_template_no_thinking_tail(text)
    text = _remove_legacy_tool_choice_instruction(text)
    for legacy_end in (_OLD_TURN_END, _UNCONDITIONAL_OPEN_TURN_END):
        text = text.replace(legacy_end, _CANONICAL_TURN_END, 1)
    return text


def patch_bundle_chat_template(bundle: Path) -> None:
    """Write one corrected template to both existing bundle locations.

    INVARIANT: the file and embedded template must agree after export. Keep
    the existing generation-token metadata propagation; do not alter sampling
    defaults, touch weights or invent a tokenizer config when none exists.
    """
    template = bundle / "chat_template.jinja"
    if not template.exists():
        return
    patched = _patch_chat_template(template.read_text(encoding="utf-8"))
    template.write_text(patched, encoding="utf-8")
    tokenizer = bundle / "tokenizer_config.json"
    if tokenizer.exists():
        config = json.loads(tokenizer.read_text(encoding="utf-8"))
        config["chat_template"] = patched
        generation = bundle / "generation_config.json"
        if generation.exists():
            defaults = json.loads(generation.read_text(encoding="utf-8"))
            for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
                if defaults.get(key) is not None:
                    config[key] = defaults[key]
        tokenizer.write_text(json.dumps(config, indent=2), encoding="utf-8")
