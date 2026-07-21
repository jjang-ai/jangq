"""Canonical family → parser/cache capability table for JANG / JANGTQ models.

This module owns the mapping from a model's family identifier (HF model_type
or architecture string) to the runtime hints vmlx's `CapabilityDetector`
reads at Tier-1: reasoning parser, tool parser, think-in-template flag,
cache type, modality, supports_tools/supports_thinking.

Cross-checked against:
  - vmlx `Sources/vMLXEngine/ModelCapabilities.swift::ModelTypeTable` (silver)
  - vmlx `Sources/vMLXEngine/Parsers/ParserRegistry.swift` (registered names)
  - vLLM upstream recipes (Qwen 3.5/3.6 → qwen3 + qwen3_coder)

Schemas accepted:
  jang_config["source_model"]["architecture"]      ← string  (JANGTQ converters)
  jang_config["architecture"]["type"]              ← string inside dict (convert.py)

Both shapes resolve through `build_capabilities`. Safe to re-run on already-
stamped artifacts (idempotent).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# (family, reasoning_parser, tool_parser, think_in_template, cache_type)
FAMILY_MAP: dict[str, tuple[str, str, str, bool, str]] = {
    # ZAYA / Zyphra — CCA attention + top-1 MoE. ZAYA and ZAYA1-VL are
    # reasoning-capable and use qwen3 parser metadata. think_in_template stays
    # False because default/no-thinking prompts must start generation in
    # visible content, not in an auto-opened reasoning prefix.
    "zaya":              ("zaya",        "qwen3",       "zaya_xml", False, "hybrid"),
    "zaya1_vl":          ("zaya1_vl",    "qwen3",       "zaya_xml", False, "hybrid"),
    # Qwen 3.5 / 3.6 family (hybrid SSM + attention)
    "qwen3_5":          ("qwen3_5",     "qwen3",       "qwen",     True,  "hybrid"),
    "qwen3_5_text":     ("qwen3_5",     "qwen3",       "qwen",     True,  "hybrid"),
    "qwen3_5_moe":      ("qwen3_5_moe", "qwen3",       "qwen",     True,  "hybrid"),
    "qwen3_5_moe_text": ("qwen3_5_moe", "qwen3",       "qwen",     True,  "hybrid"),
    "qwen3_next":       ("qwen3_next",  "qwen3",       "qwen",     True,  "hybrid"),
    "qwen3":            ("qwen3",       "qwen3",       "qwen",     True,  "kv"),
    # MiniMax M2.x
    "minimax_m2":       ("minimax_m2",  "qwen3",       "minimax",  True,  "kv"),
    "minimax_m2_5":     ("minimax_m2",  "qwen3",       "minimax",  True,  "kv"),
    "minimax":          ("minimax_m2",  "qwen3",       "minimax",  True,  "kv"),
    # MiMo V2.5 — hybrid full-attention + SWA KV. Cache topology details
    # live in the MiMo converter/runtime metadata; the shared capability stamp
    # keeps generic JANG restamping on the XML reasoning/tool parser path.
    "mimo_v2":          ("mimo_v2",     "think_xml",   "xml_function", False, "kv"),
    "mimo_v2_flash":    ("mimo_v2",     "think_xml",   "xml_function", False, "kv"),
    # GLM 5.x (MLA + DSA)
    "glm_moe_dsa":      ("glm5",        "deepseek_r1", "deepseek", True,  "mla"),
    "glm5":             ("glm5",        "deepseek_r1", "deepseek", True,  "mla"),
    # DeepSeek V4 (MLA + CSA/HCA + mHC + hash routing).
    # DSV4 emits its own DSML tool-call format (`<｜DSML｜invoke ...`),
    # NOT the plain DeepSeek-R1 format. The dsml_tool_parser registers
    # both "dsml" and "deepseek_v4" as aliases. Stamping "deepseek" here
    # routed freshly-converted DSV4 bundles through the wrong parser.
    "deepseek_v4":      ("deepseek_v4", "deepseek_r1", "dsml",     True,  "mla"),
    # GLM 4 (dense + MoE, no MLA)
    "glm4":             ("glm4",        "deepseek_r1", "glm47",    False, "kv"),
    "glm4_moe":         ("glm4_moe",    "deepseek_r1", "glm47",    False, "kv"),
    # Nemotron (hybrid SSM)
    "nemotron_h":       ("nemotron_h",  "deepseek_r1", "nemotron", True,  "hybrid"),
    "nemotron_h_v2":    ("nemotron_h",  "deepseek_r1", "nemotron", True,  "hybrid"),
    # Mistral 4 (MLA)
    "mistral3":         ("mistral4",    "mistral",     "mistral",  False, "mla"),
    "mistral4":         ("mistral4",    "mistral",     "mistral",  False, "mla"),
    # Gemma 4 / 3
    # gemma4_unified* is the official 2026-06 release naming for the
    # omni-modal (text+image+audio+video) line; the preview shipped as
    # plain gemma4*. Same backbone — register both so capability
    # resolution + verify_directory don't lean on the substring fallback.
    "gemma4":           ("gemma4",      "gemma4",      "gemma4",   False, "kv"),
    "gemma4_text":      ("gemma4",      "gemma4",      "gemma4",   False, "kv"),
    "gemma4_unified":      ("gemma4",   "gemma4",      "gemma4",   False, "kv"),
    "gemma4_unified_text": ("gemma4",   "gemma4",      "gemma4",   False, "kv"),
    "gemma3":           ("gemma4",      "deepseek_r1", "gemma4",   False, "kv"),
    "gemma3_text":      ("gemma4",      "deepseek_r1", "gemma4",   False, "kv"),
    "gemma3n":          ("gemma4",      "gemma4",      "gemma4",   False, "hybrid"),
    # Bailing v2.5 (Ling) — hybrid MLA + Lightning Linear Attention + MoE + MTP.
    # think_in_template=False because Ling's chat template defaults to
    # `detailed thinking off` and only opens `<think>` when the user supplies
    # `detailed thinking on` in their system message. With think_in_template=True,
    # the deepseek_r1 reasoning parser assumes the assistant turn opens INSIDE
    # a think block and routes ALL output to `reasoning_content`, leaving
    # `content` null — visible as empty UI bubbles on thinking-off prompts.
    "bailing_hybrid":   ("bailing_hybrid", "deepseek_r1", "deepseek", False, "hybrid"),
    "bailing_moe_v2_5": ("bailing_hybrid", "deepseek_r1", "deepseek", False, "hybrid"),
    # Liquid LFM2/LFM2.5 hybrid LIV-conv + GQA + MoE. The template does not
    # pre-open a think block, but model outputs use <think>...</think> and
    # tool calls use the Liquid Python-call format parsed by vmlx as "lfm2".
    "lfm2":             ("lfm2",       "qwen3",       "lfm2",     False, "hybrid"),
    "lfm2_moe":         ("lfm2_moe",   "qwen3",       "lfm2",     False, "hybrid"),
    "lfm2_5":           ("lfm2_moe",   "qwen3",       "lfm2",     False, "hybrid"),
    "lfm25":            ("lfm2_moe",   "qwen3",       "lfm2",     False, "hybrid"),
    # StepFun Step 3.7 Flash is a Step3p7 VLM wrapper around Step3p5 text
    # weights. The chat template opens <think> on assistant prefill and the
    # official serving recipes use the Step3p5 XML tool parser. Attention is
    # standard KV with a full/sliding-window layer pattern, not MLA or SSM.
    "step3p5":          ("step3p7",     "qwen3",       "step3p5",  True,  "kv"),
    "step3p7":          ("step3p7",     "qwen3",       "step3p5",  True,  "kv"),
    # poolside Laguna (XS.2 / M.1 / S-2.1) — text-only MoE, hybrid full+SWA
    # attention with per-layer head counts, sigmoid router + bias, shared
    # expert, softplus attention gating. Chat template is a GLM-thinking
    # derivative ("laguna_glm_thinking_v8"): <think> pre-opened on thinking
    # requests (vendor default ON via default_chat_template_kwargs),
    # closed `</think>` prefill on no-think — same dual convention as glm5,
    # so think_in_template=True. Tool calls are byte-compatible with the
    # glm47 parser (<tool_call>name<arg_key>k</arg_key><arg_value>v…).
    # Vendor (vLLM) calls both parsers "poolside_v1" — recorded in
    # jang_config.chat.vendor_parsers, not here.
    "laguna":           ("laguna",      "deepseek_r1", "glm47",    True,  "kv"),
    # Tencent Hy3-preview (HYV3ForCausalLM) — text-only MoE, 295B/21B active.
    # GQA + qk_norm, sigmoid router with expert_bias (DSV3-style aux-free balancing),
    # 1 shared expert, first_k_dense_replace=1, native MTP layer, 256K context.
    # Reasoning: <think>/</think> qwen3-style + reasoning_effort: no_think|low|high.
    # think_in_template=False because default no_think emits a closed
    # <think></think> prefill; runtime must only seed reasoning when it
    # explicitly passes reasoning_effort=low|high.
    # Tool format: <tool_call><tool_sep><arg_key>/<arg_value> — Tencent-specific
    # ("hunyuan" parser; vLLM names it "hy_v3", SGLang "hunyuan").
    "hy_v3":            ("hy_v3",       "qwen3",       "hunyuan",  False, "kv"),
    # Llama 3.x (dense) — base + instruct
    "llama":            ("llama",       None,          "llama",    False, "kv"),
    "llama3":           ("llama",       None,          "llama",    False, "kv"),
    # idefics3 (SmolVLM) — llama text decoder + SigLIP vision encoder
    "idefics3":         ("idefics3",    None,          "llama",    False, "kv"),
}


def _resolve_family_str(jang: dict, config: dict) -> tuple[str | None, list[str]]:
    """Find the source-arch string from any known location.

    Priority:
      1. `jang_config["source_model"]["architecture"]`  (string form — JANGTQ converters)
      2. `jang_config["architecture"]["type"]`           (string-in-dict — convert.py)
      3. `config["text_config"]["model_type"]`           (HF VLM wrapper)
      4. `config["model_type"]`                          (HF top-level)
    """
    candidates: list[str] = []

    src_dict = jang.get("source_model") or {}
    if isinstance(src_dict.get("architecture"), str):
        candidates.append(src_dict["architecture"])

    arch_dict = jang.get("architecture")
    if isinstance(arch_dict, dict) and isinstance(arch_dict.get("type"), str):
        candidates.append(arch_dict["type"])

    text_cfg = config.get("text_config") or {}
    if isinstance(text_cfg.get("model_type"), str):
        candidates.append(text_cfg["model_type"])

    if isinstance(config.get("model_type"), str):
        candidates.append(config["model_type"])

    # Filter empty strings, dedupe preserving order.
    seen = set()
    unique = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            unique.append(c)

    for c in unique:
        if c.lower() in FAMILY_MAP:
            return c.lower(), unique

    # Substring fallback — match the longest known key contained in any candidate.
    joined = " ".join(c.lower() for c in unique)
    for key in sorted(FAMILY_MAP.keys(), key=len, reverse=True):
        if key in joined:
            return key, unique

    return None, unique


def _resolve_modalities(jang: dict, config: dict) -> dict[str, bool]:
    """Resolve source-backed modal components from JANG and HF config stamps."""
    has_vision = bool(jang.get("has_vision", config.get("has_vision", bool(config.get("vision_config")))))
    has_audio = bool(jang.get("has_audio", config.get("has_audio", bool(config.get("audio_config")))))
    has_video = bool(jang.get("has_video", config.get("has_video", bool(config.get("video_config")))))
    return {
        "text": True,
        "vision": has_vision,
        "audio": has_audio,
        "video": has_video,
    }


def _resolve_modality(jang: dict, config: dict, model_path: Path | None = None) -> str:
    """text | vision | audio | multimodal. jang has_* stamps are authoritative.

    M127 (iter 50): the fallback used to return "vision" if EITHER
    ``text_config`` OR ``vision_config`` appeared in the HF config. But many
    text-only MoE families (qwen3_moe, qwen3_5_moe, glm_moe_dsa, mistral4)
    wrap their text params under ``text_config`` with NO ``vision_config``,
    so any jang_config missing a ``has_vision`` stamp (legacy v1 files,
    third-party JANG models, manually-edited configs) got misclassified as
    vision. vmlx's CapabilityDetector would then route through
    VLMModelFactory and fail to load. Tightened to require ``vision_config``
    specifically — text_config alone is NOT a vision signal.
    """
    modalities = _resolve_modalities(jang, config)
    active_modal = [k for k in ("vision", "audio", "video") if modalities.get(k)]
    if len(active_modal) > 1:
        return "multimodal"
    if active_modal:
        return active_modal[0]

    if "has_vision" in jang:
        return "vision" if jang["has_vision"] else "text"
    arch_dict = jang.get("architecture")
    if isinstance(arch_dict, dict) and "has_vision" in arch_dict:
        return "vision" if arch_dict["has_vision"] else "text"
    if "vision_config" in config:
        return "vision"
    return "text"


def build_capabilities(
    jang: dict,
    config: dict | None = None,
    model_path: Path | None = None,
) -> dict | None:
    """Return the canonical `capabilities` block for this model, or None.

    None means the family couldn't be resolved — the caller should warn and
    skip stamping. The block is safe to assign as `jang["capabilities"] = ...`
    and is idempotent (re-running over an already-stamped jang produces the
    same dict).

    Does NOT mutate any inputs.
    """
    if config is None:
        config = {}
    matched, candidates = _resolve_family_str(jang, config)
    if matched is None:
        return None
    family, reasoning, tool, think_in_template, cache_type = FAMILY_MAP[matched]
    modality = _resolve_modality(jang, config, model_path)
    modalities = _resolve_modalities(jang, config)
    # supports_thinking advertises whether the model architecturally produces
    # chain-of-thought reasoning. ZAYA / ZAYA1-VL DO reason — measured live:
    # `enable_thinking=False` (default template) still produces chain-of-thought
    # output ("Okay, I need to calculate... step by step..."). The earlier
    # exclusion of zaya/zaya1_vl conflated parser-routing concerns
    # (think_in_template=False is correct because the default template emits
    # a closed </think> block) with model-capability claims. Keep
    # think_in_template=False, but mark supports_thinking=True.
    supports_thinking = reasoning is not None
    return {
        "reasoning_parser": reasoning,
        "tool_parser": tool,
        "think_in_template": think_in_template,
        "supports_tools": True,
        "supports_thinking": supports_thinking,
        "family": family,
        "modality": modality,
        "modalities": modalities,
        "has_vision": modalities["vision"],
        "has_audio": modalities["audio"],
        "has_video": modalities["video"],
        "cache_type": cache_type,
    }


# M152 (iter 75): migrated from local M150 implementation to the shared
# tuple-return helper in `_json_utils`. Keeps this module's call sites
# unchanged via the thin alias.
from ._json_utils import read_json_object_safe as _safe_load_json_dict


def verify_directory(model_dir: Path) -> tuple[bool, str]:
    """Re-read jang_config.json after a converter wrote it and confirm:

      1. capabilities block is present
      2. all required keys are populated
      3. parser names are in the registered set
      4. family/cache/modality round-trip via build_capabilities

    Returns (ok, message). Use at the end of every converter:

        from jang_tools.capabilities import verify_directory
        ok, msg = verify_directory(OUT)
        if not ok:
            sys.exit(f'capabilities verify failed: {msg}')
    """
    jang_path = model_dir / "jang_config.json"
    cfg_path = model_dir / "config.json"

    # `convert_mxtq.py` (legacy) inlines jang under config["jang"] — handle both.
    # M125 (iter 48): wrap open() in `with` so fds close deterministically.
    # M150 (iter 72): promote disk / parse failures to (False, msg) instead
    # of raising — matches the function's documented contract. Pre-M150 a
    # corrupt jang_config.json raised JSONDecodeError mid-verify, breaking
    # the CLI harness that expects (ok, msg) for all failure modes.
    if not jang_path.exists():
        if cfg_path.exists():
            cfg, err = _safe_load_json_dict(cfg_path, purpose="config.json (legacy inline jang)")
            if err is not None:
                return False, err
            inline = cfg.get("jang")
            if isinstance(inline, dict):
                jang = inline
                config = cfg
            else:
                return False, f"no jang_config.json and no config.json::jang inline at {model_dir}"
        else:
            return False, f"no jang_config.json at {model_dir}"
    else:
        jang, err = _safe_load_json_dict(jang_path, purpose="jang_config.json")
        if err is not None:
            return False, err
        if cfg_path.exists():
            config, err = _safe_load_json_dict(cfg_path, purpose="config.json")
            if err is not None:
                return False, err
        else:
            config = {}

    caps = jang.get("capabilities")
    if caps is None:
        return False, "missing `capabilities` block (converter forgot to stamp)"

    required = {"reasoning_parser", "tool_parser", "think_in_template",
                "supports_tools", "supports_thinking", "family", "modality",
                "cache_type"}
    missing = required - set(caps.keys())
    if missing:
        return False, f"capabilities missing keys: {sorted(missing)}"

    valid_reasoning = {"qwen3", "deepseek_r1", "mistral", "gemma4",
                       "openai_gptoss", None}
    valid_tool = {"qwen", "qwen3", "hermes", "llama", "mistral", "deepseek",
                  "kimi", "granite", "nemotron", "step3p5", "xlam",
                  "functionary", "glm47", "minimax", "gemma4", "native",
                  "lfm2",
                  # DSV4 native DSML format + Zaya XML tool format. Both
                  # have dedicated parsers in vmlx_engine.tool_parsers.
                  # Without these, verify_directory rejected legitimate
                  # DSV4 / Zaya bundles after stamp.
                  "dsml", "zaya_xml",
                  # Tencent Hy3-preview emits its own XML-like tool tags
                  # (<tool_call><tool_sep><arg_key>/<arg_value>); vLLM
                  # registers this parser as "hy_v3", SGLang as "hunyuan".
                  "hunyuan"}
    valid_cache = {"kv", "hybrid", "mla", "mamba"}
    valid_modality = {"text", "vision", "audio", "multimodal", "embedding", "rerank", "image"}

    if caps["reasoning_parser"] not in valid_reasoning:
        return False, f"reasoning_parser={caps['reasoning_parser']!r} not in {sorted(valid_reasoning - {None})}"
    if caps["tool_parser"] not in valid_tool:
        return False, f"tool_parser={caps['tool_parser']!r} not in {sorted(valid_tool)}"
    if caps["cache_type"] not in valid_cache:
        return False, f"cache_type={caps['cache_type']!r} not in {sorted(valid_cache)}"
    if caps["modality"] not in valid_modality:
        return False, f"modality={caps['modality']!r} not in {sorted(valid_modality)}"

    expected = build_capabilities(jang, config, model_dir)
    if expected is not None and expected != caps:
        # Stamp drift: file says one family, build_capabilities computes another.
        # Most often happens when converter stamps a stale value before later
        # mutating jang_config (e.g. flipping has_vision after capabilities).
        return False, (
            f"capabilities mismatch — stamped block {caps} but recomputing from "
            f"the same jang_config + config yields {expected}. Re-stamp at the "
            "very end of the converter, after all jang_config mutations."
        )

    return True, f"capabilities OK (family={caps['family']})"


def stamp_directory(model_dir: Path, write: bool = False, verbose: bool = True) -> bool:
    """Convenience: read/build/(write-back) jang_config.json for a directory.

    Returns True if a write would happen (or did). Safe to call after a
    converter finishes — reads jang_config.json and config.json from the
    output dir, builds the capabilities block, stamps it back.
    """
    jang_path = model_dir / "jang_config.json"
    cfg_path = model_dir / "config.json"
    if not jang_path.exists():
        if verbose:
            print(f"  [capabilities] SKIP {model_dir.name} — no jang_config.json")
        return False
    # M150 (iter 72): use _safe_load_json_dict so corrupt jang_config.json
    # doesn't crash a batch-stamp run. Matches the verify_directory
    # hardening applied in this iter.
    jang, err = _safe_load_json_dict(jang_path, purpose="jang_config.json")
    if err is not None:
        if verbose:
            print(f"  [capabilities] SKIP {model_dir.name} — {err}")
        return False
    if cfg_path.exists():
        config, err = _safe_load_json_dict(cfg_path, purpose="config.json")
        if err is not None:
            if verbose:
                print(f"  [capabilities] SKIP {model_dir.name} — {err}")
            return False
    else:
        config = {}
    caps = build_capabilities(jang, config, model_dir)
    if caps is None:
        if verbose:
            _, cands = _resolve_family_str(jang, config)
            print(f"  [capabilities] WARN {model_dir.name} — no family match (candidates={cands})")
        return False
    existing = jang.get("capabilities")
    if existing == caps:
        if verbose:
            print(f"  [capabilities] OK   {model_dir.name} (family={caps['family']})")
        return False
    jang["capabilities"] = caps
    if verbose:
        tag = "UPD" if existing else "NEW"
        print(
            f"  [capabilities] {tag}  {model_dir.name} → "
            f"family={caps['family']} reasoning={caps['reasoning_parser']} "
            f"tool={caps['tool_parser']} cache={caps['cache_type']}"
        )
    if write:
        with open(jang_path, "w") as f:
            json.dump(jang, f, indent=2)
            f.write("\n")
    return True
