"""Stamp the GLM-5.3-Flash JANG bundle contract.

VERIFIED against the shipped chat_template.jinja + generation_config
(2026-08-29, zai-org/GLM-5.3-Flash-BF16):
  - vendor generation_config: temperature=1.0, top_p=0.95,
    eos=[154820, 154827, 154829] (no top_k declared — none invented)
  - reasoning_effort in {low, high}, DEFAULT 'max' (anything else → max);
    transport = chat_template kwarg, rendered as
    '<|system|>Reasoning Effort: {Capitalized}'
  - thinking is ON by default (<think> blocks); history thinking is
    PRESERVED by default via clear_thinking=false (kwarg, INVERTED sense
    vs preserve_thinking — clear_thinking=true strips it)
  - tool calls: XML arg_key/arg_value dialect inside <tool_call> — the
    Ling/bailing family parser, NOT hermes, NOT qwen <function=> style
  - VL: vision+video towers shipped (weight-gated below)

  python -m jang_tools.glm5_next.stamp <bundle_dir>
"""

import argparse
import json
from pathlib import Path

EOS_IDS = [154820, 154827, 154829]
SAMPLING = {"temperature": 1.0, "top_p": 0.95}


def stamp(bundle: Path):
    cfg = json.loads((bundle / "config.json").read_text())
    jang = cfg.setdefault("jang_config", {})

    chat = jang.setdefault("chat", {})
    chat["sampling_defaults"] = dict(SAMPLING)
    chat["default_sampling_mode"] = "thinking"
    chat["stop_token_ids"] = EOS_IDS

    jang["reasoning"] = {
        "supported": True,
        "default": "on",
        "supported_reasoning_efforts": ["low", "high", "max"],
        "default_reasoning_effort": "max",
        "reasoning_effort_transport": "chat_template_kwarg",
        "preserve_thinking_supported": True,
        "preserve_thinking_default": True,
        "preserve_thinking_transport": "chat_template_kwarg",
        "preserve_thinking_kwarg": "clear_thinking",
        "preserve_thinking_kwarg_inverted": True,
        "notes": "Template default effort is 'max' (low/high only other legal "
                 "values). clear_thinking=false (default) keeps history "
                 "<think> blocks; set clear_thinking=true to strip them.",
    }

    idx = json.loads((bundle / "model.safetensors.index.json").read_text())
    names = idx["weight_map"].keys()
    has_vision = any(n.startswith("visual.") for n in names)
    has_mtp = any(n.startswith("model.layers.45.") for n in names)
    # vendor consolidates image+video processing into processor_config.json
    proc = {}
    if (bundle / "processor_config.json").exists():
        proc = json.loads((bundle / "processor_config.json").read_text())
    caps = {
        "has_vision": has_vision and "image_processor" in proc,
        "has_video": has_vision and "video_processor" in proc,
        "has_audio": False,
        "supports_thinking": True,
        "default_reasoning": "on",
        "think_in_template": True,
        # VERIFIED by rendering (2026-08-29): generation prompt ends with
        # '<|assistant|><think>' — thinking force-opened at generation.
        "reasoning_parser": "glm_think_block",
        "reasoning_prefill_open_tag": True,
        # arg_key/arg_value XML inside <tool_call>, bare function name, raw
        # values for strings / JSON for non-strings; results render as
        # <|observation|><tool_response>...</tool_response>. VERIFIED by
        # rendering. hermes (JSON) and qwen (<function=>) parsers both fail.
        "tool_parser": "glm_xml_args",
        "tool_response_role": "observation",
    }
    jang["capabilities"] = caps
    cfg["capabilities"] = caps

    jang["mtp"] = ({"mtp_mode": "preserved_enabled", "num_layers": 1,
                    "notes": "MTP layer 45 carried; shares the DSA indexer "
                             "(index_share_for_mtp_iteration). best_depth "
                             "unvalidated until measured on the target runtime."}
                   if has_mtp else {"mtp_mode": "none"})

    ctx = (cfg.get("text_config") or {}).get("max_position_embeddings", 1_048_576)
    jang["context"] = {"native": ctx}
    jang["dsa"] = {
        "index_topk": (cfg.get("text_config") or {}).get("index_topk", 2048),
        "notes": "Dense attention is EXACT for sequences <= index_topk; the "
                 "sparse indexer path (with k-pool compression) is required "
                 "beyond that.",
    }

    (bundle / "config.json").write_text(json.dumps(cfg, indent=1))

    gc = json.loads((bundle / "generation_config.json").read_text())
    gc.update({"temperature": SAMPLING["temperature"], "top_p": SAMPLING["top_p"],
               "do_sample": True, "eos_token_id": EOS_IDS})
    (bundle / "generation_config.json").write_text(json.dumps(gc, indent=2))

    tpl = (bundle / "chat_template.jinja").read_text()
    for lit in ("reasoning_effort", "clear_thinking", "<tool_call>",
                "arg_key", "</think>"):
        assert lit in tpl, f"template missing {lit}"
    print(f"stamped {bundle.name}: T=1.0/p=0.95, efforts low/high/max default "
          f"max, thinking ON, clear_thinking=false, tools=glm_xml_args, "
          f"vision={caps['has_vision']} video={caps['has_video']} "
          f"mtp={jang['mtp']['mtp_mode']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    a = ap.parse_args()
    stamp(Path(a.bundle))
