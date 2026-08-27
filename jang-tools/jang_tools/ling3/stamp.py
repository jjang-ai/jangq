"""Stamp the Ling-3.0 JANG bundle contract (reasoning + XML-arg tools).

Created by Jinho Jang (eric@jangq.ai) — 2026-08-26.

Every value here was read off the actual source, not assumed:

* **Sampling** lives in BOTH `generation_config.json` and
  `jang_config.chat.sampling_defaults`, and they must agree
  (`feedback_sampling_defaults_two_file_contract`). Upstream ships a bare
  `{temperature 1.0, top_p 0.95, top_k 20}` with no eos/pad — those are added
  from `tokenizer_config.json` rather than left to a downstream default.

* **Reasoning is ON by default upstream.** `chat_template.jinja` sets
  `thinking_option = 'on'` when neither `enable_thinking` nor `thinking_option`
  is supplied — verified empirically (a no-kwarg render emits
  `detailed thinking on` and opens `<think>`). `preserved_thinking` is
  **hardcoded `true`** at template line 16 and is NOT a caller variable, so
  history reasoning is always retained. The stamp preserves upstream behaviour
  rather than introducing it.

* 🚨 **Tools use an XML-ARG dialect that no existing parser handles.** The
  template instructs a bare function name on the first line followed by
  `<arg_key>`/`<arg_value>` pairs — NOT a JSON object inside `<tool_call>`.
  So `mlx_lm_autodetected` is **false**: mlx_lm picks a tool parser by
  string-searching the chat template, and nothing in this template matches a
  known parser's literal. A bundle shipped without a Ling3-specific parser has
  no working tool calling at all, silently, while the model emits perfectly
  well-formed calls (`project_lfm_tool_parser_fix`).

    python -m jang_tools.ling3.stamp <bundle_dir> [more...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SOURCE_TAG = "inclusionAI/Ling-3.0-tiny generation_config.json + model card"

EOS_IDS = [156895]          # <|role_end|>
PAD_ID = 156892             # <|endoftext|>
BOS_ID = 156891             # <|startoftext|>
THINK_OPEN, THINK_CLOSE = 156903, 156904
TOOL_OPEN, TOOL_CLOSE = 156896, 156897

# Upstream generation_config, verbatim. Not "improved" — the vendor's own
# numbers, so the two files cannot disagree.
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}


def stamp(bundle: Path) -> None:
    cfg_p = bundle / "config.json"
    gen_p = bundle / "generation_config.json"
    jang_p = bundle / "jang_config.json"

    cfg = json.loads(cfg_p.read_text())
    gen = json.loads(gen_p.read_text()) if gen_p.exists() else {}
    jang = json.loads(jang_p.read_text()) if jang_p.exists() else {}

    template = (bundle / "chat_template.jinja")
    if not template.exists():
        raise SystemExit(f"{bundle}: chat_template.jinja missing — tools/reasoning "
                         f"claims below would be unverifiable")
    tpl = template.read_text()

    # Refuse to stamp claims the template does not actually support.
    if "<arg_key>" not in tpl:
        raise SystemExit(f"{bundle}: template has no <arg_key> — the XML-arg tool "
                         f"dialect claim would be false")
    if "detailed thinking " not in tpl:
        raise SystemExit(f"{bundle}: template has no thinking switch — the "
                         f"reasoning claim would be false")

    # ── two-file sampling contract ────────────────────────────────────────
    gen.update({
        "do_sample": True,
        **SAMPLING,
        "eos_token_id": EOS_IDS[0],
        "pad_token_id": PAD_ID,
        "bos_token_id": BOS_ID,
    })
    gen_p.write_text(json.dumps(gen, indent=2))

    chat = jang.get("chat") or {}
    chat["sampling_defaults"] = dict(SAMPLING, source=SOURCE_TAG, mode="default")
    chat["stop_token_ids"] = EOS_IDS
    chat["context_length"] = {"native": cfg.get("max_position_embeddings", 131072)}
    chat["role_framing"] = {
        "style": "bailing_v3",
        "note": ("NOT ChatML: turns are <role>SYSTEM|HUMAN|ASSISTANT</role> ... "
                 "<|role_end|>."),
    }
    jang["chat"] = chat

    # ── reasoning ─────────────────────────────────────────────────────────
    jang["reasoning"] = {
        "supported": True,
        "parser": "bailing_v3",
        "default": "on",
        "think_in_template": True,
        "enable_kwarg": "enable_thinking",
        "think_open_id": THINK_OPEN,
        "think_close_id": THINK_CLOSE,
        "off_is_prefilled_closed_block": True,
        "off_prefill": "<think></think>",
        "preserve_thinking_supported": True,
        "preserve_thinking_default": True,
        "preserve_thinking_transport": "template_constant",
        "note": ("Reasoning is ON by default UPSTREAM: the template sets "
                 "thinking_option='on' when neither enable_thinking nor "
                 "thinking_option is passed, and injects 'detailed thinking on' "
                 "into the SYSTEM turn. Verified empirically. "
                 "preserved_thinking is HARDCODED true in the template (line 16) "
                 "— it is not a caller kwarg, so history <think> blocks are "
                 "always retained, which also improves prefix-cache reuse."),
    }

    # ── tools ─────────────────────────────────────────────────────────────
    jang["tools"] = {
        "supported": True,
        "parser": "bailing_v3_xml_arg",
        "dialect": "xml_arg",
        # The single most important field in this file.
        "mlx_lm_autodetected": False,
        "tool_open_id": TOOL_OPEN,
        "tool_close_id": TOOL_CLOSE,
        "schema_direction": "json_in_xml_out",
        "call_format": (
            "<tool_call>{function-name}\\n"
            "<arg_key>{k}</arg_key>\\n<arg_value>{v}</arg_value>\\n...\\n</tool_call>"
        ),
        "note": ("ASYMMETRIC: tool SCHEMAS are injected as JSON inside <tools>, "
                 "but tool CALLS come back as XML key/value pairs with a BARE "
                 "function name on the first line — not a JSON object. No "
                 "Hermes/Qwen-style JSON parser matches. Values are emitted raw "
                 "when the argument is a string and tojson otherwise, so a "
                 "parser must round-trip both. Ship a Ling3 parser or tool "
                 "calling silently does not work."),
    }

    # ── modality: weight-gated, not config-gated ──────────────────────────
    jang["vision"] = {"supported": False}
    jang["audio"] = {"supported": False}

    jang["runtime"] = {
        "model_type": cfg.get("model_type", "bailing_hybrid"),
        "architecture": "BailingMoeV3",
        "hybrid_attention": {
            "full_attention_layers": [
                i for i in range(cfg["num_hidden_layers"])
                if (i + 1) % cfg["layer_group_size"] == 0
            ],
            "linear_attention": "kda",
            "kda_conv_kernel": cfg.get("short_conv_kernel_size", 4),
            "kda_lower_bound": cfg.get("kda_lower_bound", -5),
            "note": ("KDA (Kimi Delta Attention) — NOT the Ling-2.6 Lightning/GLA "
                     "linear attention. Per-layer decode state is a recurrent "
                     "[H,128,128] tensor PLUS three short-conv ring buffers; it "
                     "does not grow with context."),
        },
        "bundle_has_mtp": False,
        "jang_profile": cfg.get("jang_profile"),
    }

    jang_p.write_text(json.dumps(jang, indent=2))
    print(f"[stamped] {bundle.name}: sampling T={SAMPLING['temperature']} "
          f"top_p={SAMPLING['top_p']} top_k={SAMPLING['top_k']} eos={EOS_IDS} "
          f"reasoning=on preserve_thinking=on tools=xml_arg(custom parser required)")


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    for d in argv:
        stamp(Path(d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
