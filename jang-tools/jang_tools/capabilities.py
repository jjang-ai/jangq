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
import re
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
    "qwen3_5":          ("qwen3_5",     "qwen3",       "qwen3_coder", True, "hybrid"),
    "qwen3_5_text":     ("qwen3_5",     "qwen3",       "qwen3_coder", True, "hybrid"),
    "qwen3_5_moe":      ("qwen3_5_moe", "qwen3",       "qwen3_coder", True, "hybrid"),
    "qwen3_5_moe_text": ("qwen3_5_moe", "qwen3",       "qwen3_coder", True, "hybrid"),
    "qwen3_next":       ("qwen3_next",  "qwen3",       "qwen",     True,  "hybrid"),
    "qwen3":            ("qwen3",       "qwen3",       "qwen",     True,  "kv"),
    # MiniMax M2.x
    "minimax_m2":       ("minimax_m2",  "qwen3",       "minimax",  True,  "kv"),
    "minimax_m2_5":     ("minimax_m2",  "qwen3",       "minimax",  True,  "kv"),
    "minimax":          ("minimax_m2",  "qwen3",       "minimax",  True,  "kv"),
    # MiniMax-M3 — its own reasoning AND tool dialect, not M2.7's. Stamping the
    # M2.7 row here would route M3 bundles through the wrong pair of parsers.
    # think_in_template=False: generation must start in visible content.
    "minimax_m3":       ("minimax_m3",  "minimax_m3",  "minimax_m3", False, "kv"),
    "minimax_m3_vl":    ("minimax_m3",  "minimax_m3",  "minimax_m3", False, "kv"),
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
    # openPangu v2 — deepseek_r1 reasoning with its OWN tool dialect, and
    # think_in_template=True (the template opens the reasoning rail itself).
    "openpangu_v2":     ("openpangu_v2", "deepseek_r1", "openpangu", True, "kv"),
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
    # Muse Glimmer — Gemma-shaped text backbone (mixed sliding/full 3:1,
    # softcap 20) + a Qwen-VL-style windowed ViT, vision AND video.
    # Reasoning is routed by RECIPIENT rather than an inline think pair:
    # "to=self" carries reasoning, "to=user" the answer — hence its own
    # reasoning parser rather than gemma4's. Tools use the ATEM dialect, which
    # is XML-SHAPED but is not valid XML (the template says so outright and it
    # is parsed with regular expressions). think_in_template=False: the
    # template injects no reasoning prefix, so generation must start in
    # visible content.
    "muse_glimmer":      ("muse_glimmer", "muse_glimmer", "atem",   False, "kv"),
    "muse_glimmer_text": ("muse_glimmer", "muse_glimmer", "atem",   False, "kv"),
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
    # TII Falcon-H1 / H1R — hybrid Mamba2 SSM + attention (vmlx routes it
    # through CacheList(ArraysCache, KVCache), so cache stays "hybrid").
    # H1R emits <think>...</think> itself; the chat template never pre-opens
    # a think block (generation prompt is a bare assistant header), so
    # think_in_template stays False and the qwen3 parser extracts the tags.
    # Tool calls are JSON inside <tool_call></tool_call> — the qwen format.
    # Non-R Falcon-H1 shares model_type and simply never emits think tags.
    "falcon_h1":        ("falcon_h1",   "qwen3",       "qwen",     False, "hybrid"),
    # Llama 3.x (dense) — base + instruct
    "llama":            ("llama",       None,          "llama",    False, "kv"),
    "llama3":           ("llama",       None,          "llama",    False, "kv"),
    # idefics3 (SmolVLM) — llama text decoder + SigLIP vision encoder
    "idefics3":         ("idefics3",    None,          "llama",    False, "kv"),
}


def _template_preopens_think(model_dir: Path) -> bool:
    """True when the chat template leaves ``<think>`` OPEN in every generation prompt.

    `think_in_template` in FAMILY_MAP is a per-family default, but some bundles
    ship a template that decides for itself. LFM2.5-2.6B pre-opens the tag
    unconditionally, so the runtime must NOT emit its own opener or the reply
    starts with a doubled tag and the reasoning parser mis-splits the response.

    Only an UNCONDITIONAL, still-open tag counts. Three shapes deliberately do
    not:

    * `assistant\n` with no tag at all — the ordinary case.
    * `{% if enable_thinking %}<think>{% endif %}` (bailing / qwen3 style) — the
      caller chooses per request, so the family default has to stand.
    * `<think></think>` (zaya style) — a CLOSED prefill. The template is
      suppressing reasoning, which is the opposite of pre-opening it, and
      treating it as a pre-open would strip the model's real answer.

    Static analysis only: the template is not rendered, because rendering needs
    a tokenizer and a full message list, and this runs at stamp time.
    """
    template = _read_chat_template(model_dir)
    if not template:
        return False
    body = _generation_prompt_region(template)
    if body is None:
        return False
    literal = _unconditional_literals(body)
    return literal.rstrip().endswith("<think>")


def _read_chat_template(model_dir: Path) -> str:
    """Template text from chat_template.jinja, else tokenizer_config.json."""
    jinja = model_dir / "chat_template.jinja"
    try:
        if jinja.is_file():
            return jinja.read_text(encoding="utf-8")
    except OSError:
        return ""
    try:
        config = json.loads(
            (model_dir / "tokenizer_config.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return ""
    value = config.get("chat_template")
    # Some tokenizers ship a list of named templates.
    if isinstance(value, list):
        for entry in value:
            if isinstance(entry, dict) and entry.get("name") in (None, "default"):
                return str(entry.get("template") or "")
        return ""
    return str(value or "")


def _generation_prompt_region(template: str) -> str | None:
    """The slice guarded by ``if add_generation_prompt``, or None.

    A template with no such guard cannot pre-open anything: the tag would then
    also land on stored assistant turns, which no template does.
    """
    marker = "add_generation_prompt"
    start = template.find(marker)
    if start < 0:
        return None
    return template[start:]


def _unconditional_literals(body: str) -> str:
    """Concatenated literal text that renders REGARDLESS of any inner condition.

    Nested ``{% if %}…{% endif %}`` blocks are dropped whole before the quoted
    literals are collected, which is what separates the unconditional LFM2.5
    shape from the per-request qwen3 one.
    """
    # `if\s` rather than `if\b`: a literal BACKSPACE byte once landed here in
    # place of the word boundary, and a pattern that can never match strips
    # nothing — the conditional qwen3 shape then read as an unconditional
    # pre-open. It was invisible in grep and sed output.
    without_conditionals = re.sub(
        r"\{%-?\s*if\s.*?\{%-?\s*endif\s*-?%\}",
        "",
        body,
        flags=re.S,
    )
    pieces = re.findall(r'"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\'', without_conditionals)
    joined = "".join(double or single for double, single in pieces)
    # Jinja literals carry escaped newlines; unescape so a trailing tag is found.
    return joined.replace("\\n", "\n").replace("\\t", "\t")


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
    caps = jang.get("capabilities")
    caps = caps if isinstance(caps, dict) else {}
    has_vision = bool(jang.get(
        "has_vision",
        caps.get("has_vision", config.get("has_vision", bool(config.get("vision_config")))),
    ))
    has_audio = bool(jang.get(
        "has_audio",
        caps.get("has_audio", config.get("has_audio", bool(config.get("audio_config")))),
    ))
    has_video = bool(jang.get(
        "has_video",
        caps.get("has_video", config.get("has_video", bool(config.get("video_config")))),
    ))
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
    # The FAMILY_MAP value is the per-family DEFAULT. A bundle whose own chat
    # template pre-opens `<think>` unconditionally overrides it, because the
    # runtime must not then emit a second opener — LFM2.5-2.6B ships exactly
    # that shape while the `lfm2` row says False. Only ever an upgrade to True:
    # a family marked True with a template that says nothing keeps its default,
    # since the tag may be injected by the runtime rather than the template.
    if not think_in_template and model_path is not None:
        think_in_template = _template_preopens_think(model_path)
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
from ._json_utils import (
    read_json_object_safe as _safe_load_json_dict,
    write_json_object_atomic,
)


def validate_capabilities_block(caps: Any) -> tuple[bool, str]:
    """Validate the runtime-authoritative capabilities object.

    A top-level ``capabilities`` key is not a friendly feature summary. vMLX
    treats the object as an authoritative parser/cache/family contract. Keep
    this validator independent of a specific model so every converter,
    stamper, and publisher can reject the same malformed shape before users
    discover it at engine startup.
    """
    if not isinstance(caps, dict):
        return False, "capabilities must be a JSON object"

    family = caps.get("family")
    if not isinstance(family, str) or not family.strip():
        return False, "capabilities.family must be a non-empty string"

    required = {
        "reasoning_parser",
        "tool_parser",
        "think_in_template",
        "supports_tools",
        "supports_thinking",
        "family",
        "modality",
        "cache_type",
    }
    missing = required - set(caps)
    if missing:
        return False, f"capabilities missing keys: {sorted(missing)}"

    reasoning_parser = caps.get("reasoning_parser")
    if reasoning_parser is not None and (
        not isinstance(reasoning_parser, str) or not reasoning_parser.strip()
    ):
        return False, "capabilities.reasoning_parser must be null or a non-empty string"

    tool_parser = caps.get("tool_parser")
    if not isinstance(tool_parser, str) or not tool_parser.strip():
        return False, "capabilities.tool_parser must be a non-empty string"

    for key in ("think_in_template", "supports_tools", "supports_thinking"):
        if type(caps.get(key)) is not bool:
            return False, f"capabilities.{key} must be a boolean"

    # Derive parser/cache allowlists from the same family table used to build
    # stamps. Hand-maintained subsets had already drifted: MiniMax-M3, Muse,
    # openPangu, ATEM, and single-video modality were legitimate but absent.
    valid_reasoning = {row[1] for row in FAMILY_MAP.values()} | {
        "openai_gptoss",
        None,
    }
    valid_tool = {row[2] for row in FAMILY_MAP.values()}
    valid_cache = {row[4] for row in FAMILY_MAP.values()} | {"mamba"}
    valid_modality = {
        "text",
        "vision",
        "video",
        "audio",
        "multimodal",
        "embedding",
        "rerank",
        "image",
    }

    if reasoning_parser not in valid_reasoning:
        return False, f"reasoning_parser={reasoning_parser!r} is not registered"
    if tool_parser not in valid_tool:
        return False, f"tool_parser={tool_parser!r} is not registered"
    if caps.get("cache_type") not in valid_cache:
        return False, f"cache_type={caps.get('cache_type')!r} is not registered"
    if caps.get("modality") not in valid_modality:
        return False, f"modality={caps.get('modality')!r} is not registered"

    return True, f"capabilities schema OK (family={family.strip()})"


def validate_jang_runtime_contract(jang: Any) -> tuple[bool, str]:
    """Validate whichever JANG schema makes runtime routing authoritative.

    Pre-capabilities bundles intentionally fall through to ``config.json``.
    Once a bundle declares a capabilities object or chat-authoritative schema,
    however, its family is mandatory because vMLX no longer uses that legacy
    fallback.
    """
    if not isinstance(jang, dict):
        return False, "jang_config.json must contain a JSON object"
    if "capabilities" in jang:
        return validate_capabilities_block(jang.get("capabilities"))
    if isinstance(jang.get("chat"), dict):
        family = jang.get("model_family")
        if not isinstance(family, str) or not family.strip():
            return False, (
                "chat-authoritative jang_config.json requires a non-empty "
                "top-level model_family"
            )
        return True, f"chat-authoritative schema OK (family={family.strip()})"
    return True, "legacy JANG schema; runtime family falls back to config.json"


def _capabilities_match_canonical_fields(caps: dict, expected: dict) -> bool:
    """Allow forward-compatible extras while pinning every canonical field."""
    return all(caps.get(key) == value for key, value in expected.items())


def write_validated_jang_config(
    model_dir: Path,
    jang: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> None:
    """Validate and atomically publish ``jang_config.json`` for a bundle."""
    model_dir = Path(model_dir)
    if config is None:
        cfg_path = model_dir / "config.json"
        if cfg_path.exists():
            config, err = _safe_load_json_dict(cfg_path, purpose="config.json")
            if err is not None:
                raise ValueError(err)
        else:
            config = {}

    caps = jang.get("capabilities")
    ok, message = validate_capabilities_block(caps)
    if not ok:
        raise ValueError(message)

    expected = build_capabilities(jang, config, model_dir)
    if expected is not None and not _capabilities_match_canonical_fields(caps, expected):
        raise ValueError(
            "capabilities mismatch: authoritative fields do not match "
            f"build_capabilities output {expected}"
        )

    write_json_object_atomic(model_dir / "jang_config.json", jang)


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
    ok, message = validate_capabilities_block(caps)
    if not ok:
        return False, message

    expected = build_capabilities(jang, config, model_dir)
    if expected is not None and not _capabilities_match_canonical_fields(caps, expected):
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
        write_validated_jang_config(model_dir, jang, config)
    return True
