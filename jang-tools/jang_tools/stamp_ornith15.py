"""Stamp the Ornith 1.5 bundle contract (vision + video + reasoning + MTP).

Created by Jinho Jang (eric@jangq.ai) — 2026-08-19.

Every value here was verified against the downloaded source, not copied from
the Qwen3.8 stamper. Ornith 1.5 is the same `qwen3_5` / `qwen3_5_moe` family,
but the contract differs in three ways that matter:

SAMPLING — the card documents **two** presets (not 3.6's three, not 3.8's
thinking/instruct pair):

    coding (DEFAULT)   temp 0.6  top_p 0.95  top_k 20  min_p 0  presence 0.0
    general            temp 1.0  top_p 0.95  top_k 20  min_p 0  presence 1.5

Coding is the default because Ornith 1.5 IS an agentic coding model (SWE-bench
Verified 79, Terminal-Bench 2.1 67.8). This deliberately diverges from
upstream's generation_config.json, which ships the general numbers; both
presets are stamped and `--default-mode general` restores parity.

repetition_penalty is 1.0 in both. Note the 9B source ships **no
`generation_config.json` at all**, so without this stamp the recommended
sampling would be lost entirely — exactly the Laguna XS `top_k=20` failure
mode (`feedback_sampling_defaults_two_file_contract`). We author both halves
of the two-file contract.

REASONING — thinking is **ON by default**. Verified empirically on the source
tokenizer: the no-kwarg generation prompt is byte-identical to
`enable_thinking=True` and ends `<|im_start|>assistant\\n<think>\\n`;
`enable_thinking=False` instead prefills a *closed* block
`<think>\\n\\n</think>\\n\\n`. So reasoning-off is a prefill, not an omission.

🚨 **Ornith has NO `reasoning_effort`.** The literal does not appear in
`chat_template.jinja`. Do NOT carry over Qwen3.8's low/medium/xhigh tiers —
stamping them would advertise a feature the template cannot honor.

🚨 **No `preserve_thinking` kwarg either**, but history `<think>` blocks are
preserved *unconditionally* (verified: a prior assistant turn's reasoning
survives into the rendered prompt). So it is a property, not a toggle.

TOOLS — `qwen3_coder`. The template contains both the `<tool_call>` and
`<function=` literals, so mlx_lm autodetects it; we stamp it explicitly anyway.

MODALITY — vision **and video**: both `preprocessor_config.json` and
`video_preprocessor_config.json` ship, the template carries `<|image_pad|>`,
`<|video_pad|>` and `add_vision_id`, and the weights carry 333
`model.visual.*` tensors.

🚨 **NO AUDIO.** The tokenizer defines `<|audio_start|>`, `<|audio_end|>` and
`<|audio_pad|>`, but there is **no `audio_config` and no audio-tower weights**.
These are vestigial tokens — the capability gate is weight-gated to False on
purpose. Do not "fix" `has_audio` to True on the strength of the tokens
(`project_vestigial_vl_capability_gate`).

MTP — gated on the actual `mtp.*` tensor count, which differs across the
family: the 35B-A3B ships a real 1-layer MTP head (with its own 256 experts);
the **9B ships none** despite declaring `mtp_num_hidden_layers: 1`, so it
stamps `metadata_only_missing_weights`.

    python -m jang_tools.stamp_ornith15 <bundle_dir> [more...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ornith-1.5 card, "Recommended sampling parameters" — TWO presets.
SAMPLING_MODES = {
    "general": {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                "min_p": 0.0, "presence_penalty": 1.5,
                "repetition_penalty": 1.0},
    "coding": {"temperature": 0.6, "top_p": 0.95, "top_k": 20,
               "min_p": 0.0, "presence_penalty": 0.0,
               "repetition_penalty": 1.0},
}
# Ornith 1.5 is an AGENTIC CODING model — its headline results are SWE-bench
# Verified 79, SWE-bench Pro 59.6, Terminal-Bench 2.1 67.8. Shipping the
# general preset as THE default hands a coding user temperature 1.0 with
# presence_penalty 1.5, which is aggressive for code.
#
# Note the tension, deliberately: upstream's own generation_config.json carries
# the GENERAL numbers (T=1.0, top_k=20, top_p=0.95). We diverge on purpose and
# record both presets, so a runtime can pick and a user can see exactly what
# changed. Override with --default-mode general.
DEFAULT_MODE = "coding"

# config.json top-level eos is 248046 (<|im_end|>); text_config eos is 248044
# (<|endoftext|>). Both must stop generation — the stale-EOS history on this
# tokenizer is why we ship the pair rather than either alone.
EOS_IDS = [248046, 248044]
PAD_ID = 248044
SOURCE_TAG = "ornith15_vendor_card_2026-08-19"
NATIVE_CTX = 262144


def stamp(b: Path, default_mode: str = DEFAULT_MODE) -> dict:
    cfg_p = b / "config.json"
    jang_p = b / "jang_config.json"
    gen_p = b / "generation_config.json"

    cfg = json.loads(cfg_p.read_text())
    jang = json.loads(jang_p.read_text()) if jang_p.exists() else {}
    gen = json.loads(gen_p.read_text()) if gen_p.exists() else {}

    d = SAMPLING_MODES[default_mode]

    # ── two-file sampling contract ────────────────────────────────────────
    # The 9B source ships NO generation_config.json; we author it.
    gen.update({
        "do_sample": True,
        "temperature": d["temperature"],
        "top_p": d["top_p"],
        "top_k": d["top_k"],
        # min_p and presence_penalty are part of the card's recommendation;
        # omitting them here is how the Laguna XS top_k=20 loss happened.
        "min_p": d["min_p"],
        "presence_penalty": d["presence_penalty"],
        "repetition_penalty": d["repetition_penalty"],
        "eos_token_id": EOS_IDS,
        "pad_token_id": PAD_ID,
    })

    chat = jang.get("chat") or {}
    chat["sampling_defaults"] = dict(SAMPLING_MODES[default_mode],
                                     source=SOURCE_TAG, mode=default_mode)
    # Both presets are first-class: a runtime that wants the other one should
    # not have to guess which key holds it.
    chat["sampling_default_mode"] = default_mode
    chat["sampling_alternate_mode"] = ("general" if default_mode == "coding"
                                       else "coding")
    chat["sampling_note"] = (
        "Ornith 1.5 is an agentic coding model; the CODING preset is the "
        "default here (temp 0.6, presence 0.0). Upstream's own "
        "generation_config.json ships the GENERAL preset (temp 1.0, presence "
        "1.5) — switch with sampling_modes.general if you want parity with "
        "vLLM/Transformers defaults.")
    chat["sampling_modes"] = SAMPLING_MODES
    chat["stop_token_ids"] = EOS_IDS
    chat["context_length"] = {"native": NATIVE_CTX}
    jang["chat"] = chat

    # ── reasoning ─────────────────────────────────────────────────────────
    jang["reasoning"] = {
        "supported": True,
        "parser": "qwen3",
        "default": "on",
        "think_in_template": True,
        # Toggleable, but "off" does NOT remove the block — it prefills an
        # EMPTY CLOSED one. A parser that tests "is there a <think> block"
        # will find one in BOTH modes; it must test whether the block has
        # content, or read the post-</think> span.
        "toggleable": True,
        "enable_kwarg": "enable_thinking",
        "off_is_prefilled_closed_block": True,
        "off_prefill": "<think>\n\n</think>\n\n",
        "on_prefill": "<think>\n",
        # Ornith 1.5 does NOT implement reasoning_effort — the literal is
        # absent from chat_template.jinja. Explicit False so nobody wires a
        # UI control for a kwarg the template will ignore.
        "reasoning_effort_supported": False,
        # No kwarg, but history <think> survives rendering unconditionally.
        "preserve_thinking_supported": False,
        "preserve_thinking_behavior": "always_preserved",
        "emits_reasoning_content_key": True,
        "note": ("Thinking is ON by default (no-kwarg render is byte-identical "
                 "to enable_thinking=True); disabling prefills a CLOSED think "
                 "block. NO reasoning_effort tiers on this family. History "
                 "<think> blocks are preserved unconditionally."),
    }

    jang["tools"] = {"supported": True, "parser": "qwen3_coder",
                     "dialect": "xml_function", "mlx_lm_autodetected": True}

    # ── modality: weight-gated ────────────────────────────────────────────
    video_ok = (b / "video_preprocessor_config.json").exists()
    jang["vision"] = {
        "supported": True,
        "video_supported": video_ok,
        "tower": "qwen3_5 ViT (27 layers, hidden 1152, patch 16, merge 2)",
        "processor": "preprocessor_config.json",
        "video_processor": "video_preprocessor_config.json",
    }
    # Vestigial audio tokens with no encoder — see module docstring.
    jang["audio"] = {
        "supported": False,
        "reason": ("tokenizer defines <|audio_start|>/<|audio_end|>/"
                   "<|audio_pad|> but the model has no audio_config and no "
                   "audio-tower weights — vestigial tokens only"),
    }

    # ── MTP: gate on real tensors, stamp the keys the runtimes read ───────
    idx = b / "model.safetensors.index.json"
    mtp_n = 0
    if idx.exists():
        wm = json.loads(idx.read_text()).get("weight_map", {})
        mtp_n = sum(1 for k in wm if k.startswith("mtp."))

    jang["drop_mtp"] = False
    runtime = jang.get("runtime") or {}
    runtime.update({
        "bundle_has_mtp": mtp_n > 0,
        "mtp_layers": 1 if mtp_n else 0,
        "mtp_mode": "preserved_enabled" if mtp_n else "metadata_only_missing_weights",
    })
    jang["runtime"] = runtime
    jang["mtp"] = {
        "num_layers": 1 if mtp_n else 0,
        "artifact_available": mtp_n > 0,
        "tensor_count": mtp_n,
        "runtime_available": False,
        "dedicated_embeddings": False,
        "recommended_num_drafts": 1,
        "notation": ("Draft count is the unambiguous unit: vmlx native-MTP "
                     "'depth' and vLLM num_speculative_tokens both count "
                     "DRAFTS."),
    }

    caps = jang.get("capabilities") or cfg.get("capabilities") or {}
    caps.update({
        "has_vision": True,
        "has_video": bool(video_ok),
        "has_audio": False,
        "modality": "video" if video_ok else "vision",
        "modalities": {"text": True, "vision": True,
                       "video": bool(video_ok), "audio": False},
        "supports_tools": True, "tool_parser": "qwen3_coder",
        "supports_thinking": True, "default_reasoning": "on",
        "think_in_template": True, "reasoning_parser": "qwen3",
        "family": cfg.get("model_type", "qwen3_5"),
        "cache_type": "hybrid",
    })
    jang["capabilities"] = caps
    cfg["capabilities"] = caps

    gen_p.write_text(json.dumps(gen, indent=2) + "\n")
    jang_p.write_text(json.dumps(jang, indent=2) + "\n")
    cfg_p.write_text(json.dumps(cfg, indent=2) + "\n")
    return {"bundle": b.name, "mtp": mtp_n, "video": video_ok}


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    mode = DEFAULT_MODE
    for i, a in enumerate(argv):
        if a == "--default-mode":
            mode = argv[i + 1]
    if mode not in SAMPLING_MODES:
        print(f"  unknown --default-mode {mode!r}; expected one of "
              f"{list(SAMPLING_MODES)}")
        return 1
    d = SAMPLING_MODES[mode]
    for x in argv[1:]:
        if x.startswith("--") or x == mode:
            continue
        r = stamp(Path(x).expanduser(), default_mode=mode)
        print(f"  stamped {r['bundle']:36s} mode={mode} T={d['temperature']} "
              f"top_p={d['top_p']} top_k={d['top_k']} presence={d['presence_penalty']} "
              f"eos={EOS_IDS} reasoning=on(no-effort-tiers) vision=True "
              f"video={r['video']} audio=False mtp={r['mtp']} (+2 sampling modes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
