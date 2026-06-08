import json
from pathlib import Path

from jang_tools.convert_gemma4_jang import jang_bits
from jang_tools.convert_gemma4_mxfp import _patch_chat_template, quant_policy


BAD_TAIL = """{%- if add_generation_prompt -%}
    {%- if ns.prev_message_type != 'tool_response' and ns.prev_message_type != 'tool_call' -%}
        {{- '<|turn>model\\n' -}}
        {%- if not enable_thinking | default(false) -%}
            {{- '<|channel>thought\\n<channel|>' -}}
        {%- endif -%}
    {%- endif -%}
{%- endif -%}"""


SAFE_TAIL = """{%- if add_generation_prompt -%}
    {%- if ns.prev_message_type != 'tool_response' and ns.prev_message_type != 'tool_call' -%}
        {{- '<|turn>model\\n' -}}
    {%- endif -%}
{%- endif -%}"""


TOOL_ANCHOR = """        {%- endfor %}
        {%- set ns.prev_message_type = 'tool' -%}
    {%- endif -%}"""


def test_gemma4_template_patch_removes_no_thinking_empty_thought_tail() -> None:
    patched = _patch_chat_template(BAD_TAIL)

    assert "{{- '<|channel>thought\\n<channel|>' -}}" not in patched
    assert SAFE_TAIL in patched


def test_gemma4_template_patch_keeps_explicit_reasoning_and_injects_required_tool_choice() -> None:
    template = (
        "{{- '<|channel>thought\\n' + thinking_text + '\\n<channel|>' -}}\n"
        + TOOL_ANCHOR
        + "\n"
        + BAD_TAIL
    )

    patched = _patch_chat_template(template)

    assert "thinking_text" in patched
    assert "Tool use is REQUIRED" in patched
    assert "tool_choice == 'required'" in patched
    assert "{{- '<|channel>thought\\n<channel|>' -}}" not in patched


def test_gemma4_template_patch_is_idempotent() -> None:
    patched_once = _patch_chat_template(TOOL_ANCHOR + "\n" + BAD_TAIL)
    patched_twice = _patch_chat_template(patched_once)

    assert patched_once == patched_twice


def test_gemma4_sidecar_template_and_tokenizer_config_stay_synced(tmp_path: Path) -> None:
    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir()
    out.mkdir()
    (src / "chat_template.jinja").write_text(TOOL_ANCHOR + "\n" + BAD_TAIL, encoding="utf-8")
    (src / "tokenizer_config.json").write_text(json.dumps({"chat_template": BAD_TAIL}), encoding="utf-8")

    from jang_tools.convert_gemma4_mxfp import _copy_sidecars

    _copy_sidecars(src, out)

    template = (out / "chat_template.jinja").read_text(encoding="utf-8")
    cfg = json.loads((out / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert cfg["chat_template"] == template
    assert "Tool use is REQUIRED" in template
    assert "{{- '<|channel>thought\\n<channel|>' -}}" not in template


def test_gemma4_mxfp_keeps_tied_token_embedding_passthrough() -> None:
    policy = quant_policy("model.language_model.embed_tokens.weight", bits=8)

    assert policy.bits == 16
    assert policy.method == "passthrough"


def test_gemma4_mxfp_default_quantizes_decoder_attention() -> None:
    policy = quant_policy(
        "model.language_model.layers.0.self_attn.q_proj.weight",
        bits=8,
    )

    assert policy.bits == 8
    assert policy.method == "affine"


def test_gemma4_mxfp_audio_probe_can_preserve_attention_fp16() -> None:
    policy = quant_policy(
        "model.language_model.layers.0.self_attn.q_proj.weight",
        bits=8,
        preserve_attention_fp16=True,
    )
    mlp_policy = quant_policy(
        "model.language_model.layers.0.mlp.gate_proj.weight",
        bits=8,
        preserve_attention_fp16=True,
    )

    assert policy.bits == 16
    assert policy.method == "passthrough"
    assert mlp_policy.bits == 8
    assert mlp_policy.method == "affine"


def test_gemma4_mxfp_audio_probe_can_preserve_full_attention_only() -> None:
    full_policy = quant_policy(
        "model.language_model.layers.5.self_attn.q_proj.weight",
        bits=8,
        preserve_full_attention_fp16=True,
    )
    sliding_policy = quant_policy(
        "model.language_model.layers.4.self_attn.q_proj.weight",
        bits=8,
        preserve_full_attention_fp16=True,
    )

    assert full_policy.bits == 16
    assert full_policy.method == "passthrough"
    assert sliding_policy.bits == 8
    assert sliding_policy.method == "affine"


def test_gemma4_mxfp_audio_probe_can_preserve_first_layers() -> None:
    early_policy = quant_policy(
        "model.language_model.layers.2.mlp.down_proj.weight",
        bits=8,
        preserve_first_layers=4,
    )
    later_policy = quant_policy(
        "model.language_model.layers.4.mlp.down_proj.weight",
        bits=8,
        preserve_first_layers=4,
    )

    assert early_policy.bits == 16
    assert early_policy.method == "passthrough"
    assert later_policy.bits == 8
    assert later_policy.method == "affine"


def test_gemma4_jang_keeps_tied_token_embedding_passthrough() -> None:
    assert jang_bits("model.language_model.embed_tokens.weight") is None
