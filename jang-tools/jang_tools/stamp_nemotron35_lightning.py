"""Stamp the Nemotron 3.5 Lightning bundle contract onto a converted bundle.

Applies the following bundle contract:

  1. Sampling defaults in BOTH files (jang_config.chat.sampling_defaults AND
     generation_config.json) — the two-file contract. Vendor card states
     Temperature 1.0 / Top_P 0.95 four separate times; top_k is unspecified
     anywhere, so it is stamped as 0 (disabled) to stop downstream servers
     applying their own default (Ollama would otherwise force top_k=40).
  2. Reasoning: default ON, two states only (no low/med/high tiers),
     think_in_template=True, budget is server-side.
  3. Tools: XML dialect, `nemotron` parser.
  4. Capability gate: text-only, decided on weights not config.
  5. MTP: artifact_available reflects whether mtp.* survived into the bundle;
     runtime_available stays False until a runtime actually decodes with it.

Idempotent — safe to re-run.

    python -m jang_tools.stamp_nemotron35_lightning <bundle_dir> [more_dirs...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# ── Vendor-stated sampling (README + generation_config.json, 2026-08-11) ──
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 0          # unspecified by vendor -> explicitly disabled
REPETITION_PENALTY = 1.0  # vendor specifies none; RL filtered for repetition

EOS_IDS = [2, 11]  # </s> and <|im_end|>. Dropping 11 = model never stops.
BOS_ID = 1         # present, but add_bos_token=False -> do NOT prepend
PAD_ID = 0

SOURCE_TAG = "vendor_card+generation_config_2026-08-11"


def _load(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise SystemExit(f"{p}: invalid JSON ({e})")


def _save(p: Path, obj: dict) -> None:
    p.write_text(json.dumps(obj, indent=2) + "\n")


def _has_mtp(bundle: Path) -> bool:
    idx = bundle / "model.safetensors.index.json"
    if not idx.exists():
        return False
    wm = json.loads(idx.read_text()).get("weight_map", {})
    return any(k.startswith("mtp.") for k in wm)


def stamp(bundle: Path) -> dict:
    if not bundle.is_dir():
        raise SystemExit(f"not a directory: {bundle}")

    cfg_p = bundle / "config.json"
    gen_p = bundle / "generation_config.json"
    jang_p = bundle / "jang_config.json"

    cfg = _load(cfg_p)
    gen = _load(gen_p)
    jang = _load(jang_p)

    mtp_present = _has_mtp(bundle)

    sampling = {
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
        "source": SOURCE_TAG,
    }

    reasoning = {
        "supported": True,
        "parser": "deepseek_r1",       # nemotron_v3 upstream == <think></think>
        "default": "on",               # template: enable_thinking defaults True
        "think_in_template": True,      # template opens the rail itself
        "tiers": None,                  # NO low/medium/high — verified
        "budget": "server_side",        # not a template kwarg
        "enable_kwarg": "enable_thinking",
        "off_is_prefilled_empty_block": True,   # "<think></think>", not omission
        "truncate_history_thinking_default": True,
    }

    tools = {
        "supported": True,
        "parser": "nemotron",           # XML <tool_call><function=..><parameter=..>
        "dialect": "xml_function",       # same string as qwen3_coder upstream
        "tools_in_system_prompt": True,
        "tool_results_role": "tool",     # rendered as <tool_response> in a user turn
        "force_nonempty_content_supported": False,  # documented but absent from template
    }

    # ── generation_config.json: vendor values + explicit top_k ──
    gen.update({
        "do_sample": True,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "eos_token_id": EOS_IDS,
        "bos_token_id": BOS_ID,
        "pad_token_id": PAD_ID,
    })

    # ── jang_config: chat + reasoning + tools + mtp ──
    chat = jang.get("chat") or {}
    chat["sampling_defaults"] = sampling
    chat["add_bos_token"] = False
    chat["stop_token_ids"] = EOS_IDS
    jang["chat"] = chat
    jang["reasoning"] = reasoning
    jang["tools"] = tools
    jang["mtp"] = {
        "artifact_available": mtp_present,
        "runtime_available": False,     # never imply active MTP decode
        "num_nextn_predict_layers": 1,
        "shape": "deepseek_v3_eh_proj",
        "shares_embeddings_and_lm_head": True,
    }

    # The shared nemotron MX converter hardcodes Nemotron-Omni's identity and
    # method="affine" in jang_config. Correct both — a bundle that misreports
    # its source model or codec is the mlxstudio#130 metadata-bug class.
    qmode = (cfg.get("quantization") or {}).get("mode")
    jang["source_model"] = {
        "name": "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        "org": "nvidia",
        "architecture": "nemotron_h",
        "modality": "text",
    }
    if isinstance(jang.get("quantization"), dict) and qmode:
        jang["quantization"]["method"] = qmode
    if qmode:
        jang["profile"] = qmode.upper()

    caps = jang.get("capabilities") or cfg.get("capabilities") or {}
    caps.update({
        "has_vision": False, "has_audio": False, "has_video": False,
        "modality": "text",
        "modalities": {"text": True, "vision": False, "audio": False, "video": False},
        "supports_tools": True,
        "supports_thinking": True,
        "default_reasoning": "on",
        "think_in_template": True,
        "reasoning_parser": "deepseek_r1",
        "tool_parser": "nemotron",
        "cache_type": "hybrid",
        "family": "nemotron_h",
    })
    jang["capabilities"] = caps

    # Mirror onto config.json so runtimes that read either surface agree.
    cfg["capabilities"] = caps
    cfg.setdefault("jang_config", {})
    if isinstance(cfg["jang_config"], dict):
        cfg["jang_config"].update({
            "chat": chat, "reasoning": reasoning, "tools": tools,
            "mtp": jang["mtp"],
        })

    _save(gen_p, gen)
    _save(jang_p, jang)
    _save(cfg_p, cfg)

    return {
        "bundle": bundle.name,
        "mtp_artifact": mtp_present,
        "temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K,
        "eos": EOS_IDS,
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    for d in argv[1:]:
        r = stamp(Path(d).expanduser())
        print(
            f"  stamped {r['bundle']}: T={r['temperature']} top_p={r['top_p']} "
            f"top_k={r['top_k']} eos={r['eos']} reasoning=on "
            f"mtp_artifact={r['mtp_artifact']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
