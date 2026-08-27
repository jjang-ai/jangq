"""Stamp the Nemotron-3-Nano-Omni bundle contract onto a converted bundle.

Created by Jinho Jang (eric@osaurus.ai) — 2026-08-22.

Everything here comes from the model's own README + `generation_config.json`,
not from a sibling bundle:

  | Mode                  | temp | top_p | top_k | max_tokens | budget | grace |
  |-----------------------|------|-------|-------|------------|--------|-------|
  | Thinking (**default**)| 0.6  | 0.95  | —     | 20480      | 16384  | 1024  |
  | Instruct (no-think)   | 0.2  | —     | 1     | 1024       | —      | —     |
  | ASR                   | 1.0  | —     | 1     | —          | —      | —     |

Sampling is written to **both** `jang_config.chat.sampling_defaults` and
`generation_config.json` — the two-file contract. A vendor card that documents
parameters its own `generation_config.json` omits is how the Laguna XS
`top_k=20` was lost across six repos.

Note the source `generation_config.json` ships `reasoning_grace: 512` while the
README recommends **1024** for thinking mode. We stamp 1024 in both files and
record the divergence in `source`, rather than letting the two disagree.

`top_k` is stamped explicitly (0 = disabled) for thinking mode: the vendor
leaves it unspecified, and an unset value lets a downstream server apply its own
(Ollama would force `top_k=40`).

Capabilities are **weight-gated**: vision/audio/video are advertised only if the
tower tensors actually survived into the bundle. A `sound_config` in the JSON is
not proof that a Parakeet encoder shipped.

Idempotent — safe to re-run.

    python -m jang_tools.stamp_nemotron_omni_nano3 <bundle_dir> [more_dirs...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# ── Thinking mode = the default preset ───────────────────────────────────
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 0                 # vendor unspecified -> explicitly disabled
REPETITION_PENALTY = 1.0
MAX_TOKENS = 20480
REASONING_BUDGET = 16384
REASONING_GRACE = 1024    # README thinking-mode value; source gen_config says 512

EOS_IDS = [2, 11]
BOS_ID = 1
PAD_ID = 0

SOURCE_TAG = "vendor_readme_2026-04-28+generation_config(grace 512->1024 per README)"

TOWER_PREFIXES = ("vision_model.", "sound_encoder.", "mlp1.", "sound_projection.")


def _load(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise SystemExit(f"{p}: invalid JSON ({e})")


def _save(p: Path, obj: dict) -> None:
    p.write_text(json.dumps(obj, indent=2) + "\n")


def _towers(bundle: Path) -> dict:
    """Weight-gate the modalities — config is not evidence."""
    idx = bundle / "model.safetensors.index.json"
    wm = json.loads(idx.read_text()).get("weight_map", {}) if idx.exists() else {}
    present = {p.rstrip("."): sum(1 for k in wm if k.startswith(p))
               for p in TOWER_PREFIXES}
    return present


def stamp(bundle: Path) -> dict:
    if not bundle.is_dir():
        raise SystemExit(f"not a directory: {bundle}")

    cfg_p, gen_p, jang_p = (bundle / "config.json",
                            bundle / "generation_config.json",
                            bundle / "jang_config.json")
    cfg, gen, jang = _load(cfg_p), _load(gen_p), _load(jang_p)

    towers = _towers(bundle)
    has_vision = towers.get("vision_model", 0) > 0 and towers.get("mlp1", 0) > 0
    has_audio = towers.get("sound_encoder", 0) > 0 and towers.get("sound_projection", 0) > 0
    has_video = has_vision      # video reuses the vision tower + EVS pruning

    sampling = {
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
        "max_tokens": MAX_TOKENS,
        "source": SOURCE_TAG,
    }

    presets = {
        "thinking": {"temperature": 0.6, "top_p": 0.95, "max_tokens": 20480,
                     "reasoning_budget": REASONING_BUDGET,
                     "reasoning_grace": REASONING_GRACE,
                     "max_model_len": 210000,
                     "note": "default; long-document and multimodal reasoning"},
        "instruct": {"temperature": 0.2, "top_k": 1, "max_tokens": 1024,
                     "enable_thinking": False, "note": "general non-thinking tasks"},
        "asr": {"temperature": 1.0, "top_k": 1, "enable_thinking": False,
                "note": "speech transcription via the Parakeet encoder"},
    }

    reasoning = {
        "supported": True,
        "parser": "deepseek_r1",        # upstream nemotron_v3 == <think></think>
        "default": "on",
        "think_in_template": True,
        "tiers": None,                   # two states only, no low/medium/high
        "budget": "server_side",
        "default_budget": REASONING_BUDGET,
        "default_grace": REASONING_GRACE,
        "enable_kwarg": "enable_thinking",
        "off_is_prefilled_empty_block": True,   # "<think></think>", not omission
    }

    tools = {
        "supported": True,
        "parser": "nemotron",            # XML <tool_call><function=..><parameter=..>
        "dialect": "xml_function",        # upstream vLLM calls this qwen3_coder
        "tools_in_system_prompt": True,
        "tool_results_role": "tool",
        "upstream_parsers": {"tool_call": "qwen3_coder", "reasoning": "nemotron_v3"},
    }

    gen.update({
        "do_sample": True,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "max_new_tokens": MAX_TOKENS,
        "reasoning_budget": REASONING_BUDGET,
        "reasoning_grace": REASONING_GRACE,
        "repetition_penalty": REPETITION_PENALTY,
        "eos_token_id": EOS_IDS,
        "bos_token_id": BOS_ID,
        "pad_token_id": PAD_ID,
    })

    chat = jang.get("chat") or {}
    chat["sampling_defaults"] = sampling
    chat["presets"] = presets
    chat["stop_token_ids"] = EOS_IDS
    jang["chat"] = chat
    jang["reasoning"] = reasoning
    jang["tools"] = tools

    caps = jang.get("capabilities") or {}
    caps.update({
        "has_vision": has_vision, "has_audio": has_audio, "has_video": has_video,
        "modality": "omni" if (has_vision and has_audio) else "text",
        "modalities": {"text": True, "vision": has_vision,
                       "audio": has_audio, "video": has_video},
        "supports_tools": True,
        "supports_thinking": True,
        "default_reasoning": "on",
        "think_in_template": True,
        "reasoning_parser": "deepseek_r1",
        "tool_parser": "nemotron",
        "cache_type": "hybrid",
        "family": "nemotron_h",
        "audio_encoder": "parakeet" if has_audio else None,
        "vision_encoder": "radio" if has_vision else None,
    })
    jang["capabilities"] = caps
    jang["modality"] = caps["modality"]
    jang["multimodal_components"] = [p.rstrip(".") for p in TOWER_PREFIXES
                                     if towers.get(p.rstrip("."), 0) > 0]

    cfg["capabilities"] = caps
    cfg.setdefault("jang_config", {})
    if isinstance(cfg["jang_config"], dict):
        cfg["jang_config"].update(
            {"chat": chat, "reasoning": reasoning, "tools": tools})

    _save(gen_p, gen)
    _save(jang_p, jang)
    _save(cfg_p, cfg)

    return {"bundle": bundle.name, "towers": towers,
            "vision": has_vision, "audio": has_audio, "video": has_video}


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    for d in argv[1:]:
        r = stamp(Path(d).expanduser())
        print(f"  stamped {r['bundle']}: T={TEMPERATURE} top_p={TOP_P} "
              f"top_k={TOP_K} eos={EOS_IDS} reasoning=on "
              f"vision={r['vision']} audio={r['audio']} video={r['video']} "
              f"towers={r['towers']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
