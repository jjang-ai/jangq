"""Stamp the Qwen3.8-Flash-Next bundle contract.

Everything here was VERIFIED against the shipped tokenizer/template on
2026-08-26 (not assumed):
  - vendor generation_config = thinking preset (T=1.0/top_p .95/top_k 20)
  - card presets: thinking(default) + instruct(0.7/0.80/20, presence 1.5)
  - reasoning_effort in {low, medium, xhigh}, DEFAULT xhigh (xhigh render
    == base render), transport = chat_template kwarg
  - enable_thinking=False → closed think prefill; preserve_thinking
    default ON (history keeps <think>)
  - EOS pair [248046, 248044]; ctx 262144 native / 1M YaRN
Two-file contract: jang_config.chat.sampling_defaults AND
generation_config.json must agree (Laguna lesson).

  python -m jang_tools.qwen4_exp.stamp <bundle_dir> [--no-mtp]
"""

import argparse
import glob
import json
from pathlib import Path

EOS_IDS = [248046, 248044]

SAMPLING_MODES = {
    "thinking": {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                 "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0},
    "instruct": {"temperature": 0.7, "top_p": 0.80, "top_k": 20,
                 "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0},
}
DEFAULT_MODE = "thinking"  # thinking/agentic is the model default (card)


def stamp(bundle: Path, no_mtp: bool = False):
    cfg = json.loads((bundle / "config.json").read_text())
    jang = cfg.setdefault("jang_config", {})

    chat = jang.setdefault("chat", {})
    chat["sampling_defaults"] = dict(SAMPLING_MODES[DEFAULT_MODE])
    chat["sampling_modes"] = SAMPLING_MODES
    chat["default_sampling_mode"] = DEFAULT_MODE
    chat["recommended_max_tokens"] = {"reasoning": 262144, "final_response": 131072}

    jang["reasoning"] = {
        "supported": True,
        "default": "on",
        "supported_reasoning_efforts": ["low", "medium", "xhigh"],
        "default_reasoning_effort": "xhigh",
        "reasoning_effort_transport": "chat_template_kwarg",
        "preserve_thinking_supported": True,
        "preserve_thinking_default": True,
        "preserve_thinking_transport": "chat_template_kwarg",
        "notes": "Thinking ON by default (<think> block). enable_thinking=False "
                 "prefills a closed think block. Efforts verified 2026-08-26: "
                 "low/medium inject system directives, xhigh==default render.",
    }

    # weight-gated capabilities (vestigial-VL rule: claim only what has weights)
    names = set()
    for f in glob.glob(str(bundle / "model-*.safetensors")):
        import mlx.core as mx
        names.update(mx.load(f).keys())
        break_all = False
    has_vision = any(n.startswith("visual.") for n in names) or any(
        "visual." in n for n in names)
    # cheap full check via index
    idx = json.loads((bundle / "model.safetensors.index.json").read_text())
    all_names = idx["weight_map"].keys()
    has_vision = any(n.startswith("visual.") for n in all_names)
    has_mtp = any(n.startswith("mtp.") for n in all_names)
    caps = {
        "has_vision": has_vision,
        "has_video": has_vision and (bundle / "video_preprocessor_config.json").exists(),
        "has_audio": False,
        "supports_thinking": True, "default_reasoning": "on",
        "think_in_template": True, "reasoning_parser": "qwen3",
        "tool_parser": "hermes",  # <tool_call> literals verified in template
    }
    jang["capabilities"] = caps
    cfg["capabilities"] = caps

    jang["mtp"] = ({"mtp_mode": "none"} if (no_mtp or not has_mtp) else {
        "mtp_mode": "preserved_enabled",
        "num_layers": 1,
        "trained_multi_step": True,
        "notes": "MTP head preserved (4-bit gs64 class). best_depth requires a "
                 "measured acceptance+wall-clock sweep on the target runtime "
                 "before any recommendation (unvalidated until then).",
    })

    ctx = cfg.get("text_config", {}).get("max_position_embeddings", 262144)
    jang["context"] = {"native": ctx, "extensible_yarn": 1_000_000}

    (bundle / "config.json").write_text(json.dumps(cfg, indent=1))

    # two-file contract: generation_config must equal the DEFAULT mode
    gc = json.loads((bundle / "generation_config.json").read_text())
    d = SAMPLING_MODES[DEFAULT_MODE]
    gc.update({"temperature": d["temperature"], "top_p": d["top_p"],
               "top_k": d["top_k"], "do_sample": True, "eos_token_id": EOS_IDS})
    (bundle / "generation_config.json").write_text(json.dumps(gc, indent=2))

    # sanity: template contract literals
    tpl = (bundle / "chat_template.jinja").read_text()
    for lit in ("reasoning_effort", "preserve_thinking", "<tool_call>", "</think>"):
        assert lit in tpl, f"template missing {lit}"
    assert gc["eos_token_id"] == EOS_IDS
    print(f"stamped {bundle.name}: default={DEFAULT_MODE} "
          f"(T={d['temperature']}/p={d['top_p']}/k={d['top_k']}), efforts low/medium/xhigh "
          f"default xhigh, preserve_thinking ON, vision={caps['has_vision']} "
          f"video={caps['has_video']} mtp={jang['mtp']['mtp_mode']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--no-mtp", action="store_true")
    ap.add_argument("--source", default="~/models/Qwen3.8-Flash-Next")
    a = ap.parse_args()
    stamp(Path(a.bundle), no_mtp=a.no_mtp)
    from .emit_quant_config import emit
    emit(a.bundle, a.source)
