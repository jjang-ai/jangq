"""Beginner-friendly recommendation engine.

Reads a source HuggingFace model directory, detects its architecture family
and key properties (expert count, source dtype, VL flags), and returns a
recommended conversion plan in plain English that JANG Studio can surface
as auto-filled defaults + tooltips.

Output schema (JSON from `jang-tools recommend --model <dir> --json`):
{
  "detected": {
    "model_type": "qwen3_5_moe",
    "family_class": "moe_hybrid_ssm",
    "param_count_billions": 35.0,
    "expert_count": 256,
    "is_vl": false,
    "is_video_vl": false,
    "source_dtype": "bfloat16",
    "has_tool_parser": true,
    "has_reasoning_parser": true,
    "is_gated_model": false
  },
  "recommended": {
    "family": "jang",
    "profile": "JANG_4K",
    "method": "mse",
    "hadamard": true,
    "block_size": 64,
    "force_dtype": null,
    "alternatives": [
      {"family": "jang", "profile": "JANG_2L",
       "use_when": "You need the smallest output that still runs well."},
      {"family": "jangtq", "profile": "JANGTQ3",
       "use_when": "You want 30% smaller output and your arch is whitelisted."}
    ]
  },
  "beginner_summary": "Qwen 3.6 35B is a large mixture-of-experts model...",
  "warnings": [
    "256 experts detected — we'll force bfloat16 to avoid overflow."
  ],
  "why_each_choice": {
    "family": "JANG works on every architecture; JANGTQ is a newer, smaller format that we only support for Qwen 3.6 and MiniMax in v1.",
    "profile": "JANG_4K is our default balanced profile — ~4.2 bits per weight, strong quality for MoE models, same size as a 4-bit MLX quant but smarter.",
    ...
  }
}

Covers major architecture families whether we've converted them directly or not:
  - Dense LLMs: llama, mistral, qwen2, qwen3, gemma, gemma2, gemma3, phi, phi3, falcon, stablelm, gpt2, gpt_neox
  - MoE LLMs: qwen2_moe, qwen3_5_moe, mixtral, deepseek_v2, deepseek_v3, deepseek_v32, glm_moe_dsa
  - Hybrid SSM / MoE-Mamba: nemotron_h
  - MLA: deepseek_v2, deepseek_v3, deepseek_v32, glm_moe_dsa, mistral4
  - Proprietary large: minimax_m2
  - VL (image): qwen2_vl, qwen3_vl, idefics3, llava, llava_next, moondream, paligemma
  - VL (video): qwen2_vl (video path), qwen3_vl (video path)

Author: Jinho Jang (eric@jangq.ai)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Any

from .inspect_source import _sniff_dtype, _is_moe, _JANGTQ_V1_WHITELIST
from .architectures import is_verified_minicpm4_05b_config


# ─────────────────────────── Architecture classification ────────────────────

# Family classes drive recommendation logic. Each class has its own default
# profile + warning rules. Maps model_type → family_class.
_MODEL_TYPE_TO_FAMILY_CLASS: dict[str, str] = {
    # Dense LLMs
    "llama": "dense_llm",
    "llama2": "dense_llm",
    "llama3": "dense_llm",
    "llama4": "dense_llm",
    "mistral": "dense_llm",
    "qwen2": "dense_llm",
    "qwen3": "dense_llm",
    "gemma": "dense_llm",
    "gemma2": "dense_llm",
    "gemma3": "dense_llm",
    "gemma4": "dense_llm",
    "phi": "dense_llm",
    "phi3": "dense_llm",
    "phi4": "dense_llm",
    "falcon": "dense_llm",
    "stablelm": "dense_llm",
    "gpt2": "dense_llm",
    "gpt_neox": "dense_llm",
    "olmo": "dense_llm",
    "minicpm": "minicpm_text",

    # Standard MoE LLMs (< 512 experts)
    "qwen2_moe": "moe_standard",
    "mixtral": "moe_standard",

    # MoE with hybrid SSM / linear attention
    "qwen3_5_moe": "moe_hybrid_ssm",     # Qwen 3.6 (256 experts + gated_deltanet)

    # MoE with MLA (Multi-head Latent Attention)
    "deepseek_v2": "moe_mla",
    "deepseek_v3": "moe_mla",
    "deepseek_v32": "moe_mla",
    "mistral4": "moe_mla",

    # 512+ expert MoE (needs special handling)
    "minimax_m2": "moe_large_expert",
    "glm_moe_dsa": "moe_large_expert",

    # MoE + Mamba hybrid (MTP layers)
    "nemotron_h": "hybrid_ssm_mtp",

    # VL — image
    "qwen2_vl": "vl_image",
    "qwen3_vl": "vl_image",
    "idefics3": "vl_image",
    "llava": "vl_image",
    "llava_next": "vl_image",
    "moondream": "vl_image",
    "paligemma": "vl_image",

    # VL — video (uses same model_types but has video_preprocessor_config.json)
    # Detected dynamically.
}


# Known architectures where we force bfloat16 per feedback_bfloat16_fix.md.
_BF16_REQUIRED = {"minimax_m2", "glm_moe_dsa"}

# Models with gated HF access (require token acceptance).
_GATED_PREFIXES = ("meta-llama/", "google/gemma", "mistralai/Mistral", "mistralai/Mixtral")


def _classify_family(model_type: str, expert_count: int, is_vl: bool, is_video_vl: bool) -> str:
    """Pick a family_class for recommendation logic."""
    if is_video_vl:
        return "vl_video"
    if is_vl:
        return _MODEL_TYPE_TO_FAMILY_CLASS.get(model_type, "vl_image")
    if model_type in _MODEL_TYPE_TO_FAMILY_CLASS:
        klass = _MODEL_TYPE_TO_FAMILY_CLASS[model_type]
        # Promote MoE arch to moe_large_expert when expert count is high
        if klass in ("moe_standard", "moe_hybrid_ssm") and expert_count >= 512:
            return "moe_large_expert"
        return klass
    # Fallback — assume dense LLM
    return "dense_llm"


def _estimate_params_billion(cfg: dict) -> float:
    """Very rough parameter count in billions from config hyperparameters.

    For MoE models, counts total expert parameters (not active). Users
    typically care about on-disk size, which reflects total weights.
    """
    text_cfg = cfg.get("text_config", {}) or {}
    hidden = int(cfg.get("hidden_size", 0) or text_cfg.get("hidden_size", 0) or 0)
    layers = int(cfg.get("num_hidden_layers", 0) or text_cfg.get("num_hidden_layers", 0) or 0)
    vocab = int(cfg.get("vocab_size", 0) or text_cfg.get("vocab_size", 0) or 0)
    intermediate = int(cfg.get("intermediate_size", 0) or text_cfg.get("intermediate_size", 0) or 4 * hidden)
    num_experts = int(cfg.get("num_experts") or cfg.get("n_routed_experts") or cfg.get("num_local_experts") or 0)

    if hidden == 0 or layers == 0:
        return 0.0

    # Attention weights: 4 × hidden²
    attn = 4 * hidden * hidden
    # MLP weights per expert: 3 × hidden × intermediate (gate + up + down)
    mlp_per = 3 * hidden * intermediate
    # MoE: total MLP = mlp_per × num_experts + 1 × (shared MLP if any — rough)
    if num_experts > 1:
        mlp = mlp_per * num_experts
    else:
        mlp = mlp_per
    per_layer = attn + mlp

    total = per_layer * layers + 2 * hidden * vocab
    return round(total / 1e9, 2)


# ─────────────────────────── Recommendation tables ──────────────────────────

# Default profile per family_class. Starting point; tuned further by signals.
_DEFAULT_PROFILE_BY_CLASS: dict[str, str] = {
    "dense_llm": "JANG_4K",
    "minicpm_text": "JANG_6M",
    "moe_standard": "JANG_4K",
    "moe_hybrid_ssm": "JANG_4K",
    "moe_mla": "JANG_4K",
    "moe_large_expert": "JANG_2L",     # proven-coherent per memory
    "hybrid_ssm_mtp": "JANG_4K",
    "vl_image": "JANG_4K",
    "vl_video": "JANG_4K",
}

_CLASS_PROSE: dict[str, str] = {
    "dense_llm":
        "A standard dense language model. JANG compresses these to about a "
        "quarter of their FP16 size while keeping quality competitive with "
        "MLX's 4-bit quantization.",
    "minicpm_text":
        "MiniCPM text model whose tied output head needs independent "
        "high-precision quantization. The validated quality floor is JANG_6M; "
        "near-4-bit profiles are not advertised.",
    "moe_standard":
        "A mixture-of-experts model with a moderate number of experts. "
        "JANG handles MoE tensors per-expert so bits go where they matter "
        "most (attention gets 6-8 bits, expert MLPs get 2-4).",
    "moe_hybrid_ssm":
        "A hybrid MoE model (Qwen 3.6 style) that mixes gated DeltaNet / "
        "linear attention with full softmax attention. JANG handles both "
        "path types and preserves the routing metadata.",
    "moe_mla":
        "A MoE model with Multi-head Latent Attention (DeepSeek / Mistral 4 "
        "style). JANG keeps attention precision high because MLA is "
        "sensitive to quantization error.",
    "moe_large_expert":
        "A very large MoE (512+ experts). These need bfloat16 and careful "
        "floor values for expert MLP quantization — JANG applies these "
        "automatically. JANG_2L is proven coherent; go higher only if you "
        "have the RAM.",
    "hybrid_ssm_mtp":
        "A hybrid State-Space + MLP model with multi-token prediction (Nemotron "
        "Cascade / Super). Requires custom modeling_*.py files which JANG "
        "copies over automatically.",
    "vl_image":
        "A vision-language model that accepts image input. JANG copies the "
        "image preprocessor config and keeps the vision tower at appropriate "
        "precision.",
    "vl_video":
        "A vision-language model that also accepts video frames. JANG copies "
        "both preprocessor configs (image + video) so downstream runtimes can "
        "process either input type.",
}


# Profile-specific plain-English descriptions.
_PROFILE_PROSE: dict[str, str] = {
    "JANG_1L": "Most aggressive — 1-bit effective. Smallest output, highest quality loss. For research only.",
    "JANG_2S": "Smallest 2-bit — compact. Good for MoE models with <256 experts when disk is tight.",
    "JANG_2M": "Middle-ground 2-bit — balanced at the tiny end.",
    "JANG_2L": "Best-quality 2-bit — proven coherent even on 512-expert MoE. Use when RAM is limited.",
    "JANG_3S": "Small 3-bit — light protection on attention only.",
    "JANG_3M": "Mid 3-bit — good balance; full attention precision, 3-bit everywhere else.",
    "JANG_3L": "Best 3-bit — attention at 8-bit, embeddings at 4-bit.",
    "JANG_3K": "K-quant 3-bit — budget-neutral sizing; same size as MLX 3-bit but smarter allocation.",
    "JANG_4S": "Small 4-bit — lightweight quality lift.",
    "JANG_4M": "Mid 4-bit — attention at 8-bit, rest at 4-bit. Great for MoE.",
    "JANG_4L": "Best 4-bit — attention at 8-bit, embeddings at 6-bit. Great for dense.",
    "JANG_4K": "DEFAULT — K-quant 4-bit, budget-neutral. Same size as MLX 4-bit quant but with JANG's mixed-precision smart allocation.",
    "JANG_5K": "K-quant 5-bit — higher quality with modest size increase.",
    "JANG_6K": "K-quant 6-bit — near-lossless, small overhead.",
    "JANG_6M": "Near-lossless. Attention and embeddings at 8-bit, everything else at 6-bit.",
    "JANGTQ2": "TurboQuant 2-bit — codebook compression for Qwen 3.6 / MiniMax. Smaller than any JANG at 2-bit.",
    "JANGTQ3": "TurboQuant 3-bit — best size/quality tradeoff for Qwen 3.6 / MiniMax.",
    "JANGTQ4": "TurboQuant 4-bit — near-lossless TQ; smallest high-quality option for whitelisted archs.",
}


def _recommend_family(model_type: str, source_dtype: str) -> tuple[str, list[dict], str]:
    """Pick family (jang vs jangtq) + alternatives + reason."""
    is_whitelisted = model_type in _JANGTQ_V1_WHITELIST
    is_dtype_ok = source_dtype in ("bfloat16", "float8_e4m3fn", "float8_e5m2")
    is_dtype_unknown = source_dtype == "unknown"
    if is_whitelisted and (is_dtype_ok or is_dtype_unknown):
        use_when = "You want 30% smaller output at similar quality — works because your architecture is whitelisted."
        if is_dtype_unknown:
            use_when += " (Source dtype wasn't detected; JANGTQ needs BF16 or FP8.)"
        return (
            "jang",  # keep JANG as default to stay safe for beginners
            [{
                "family": "jangtq",
                "profile": "JANGTQ3",
                "use_when": use_when,
            }],
            "JANG is the safe default that works on every architecture. You can switch to JANGTQ for extra compression since your model is whitelisted.",
        )
    reason = (
        "JANG is the only supported family for your architecture. "
        "JANGTQ currently supports Qwen 3.6 and MiniMax only."
    ) if not is_whitelisted else (
        "JANGTQ requires a BF16 or FP8 source. Your source dtype isn't supported for JANGTQ yet."
    )
    return ("jang", [], reason)


def _recommend_profile(family_class: str, expert_count: int, param_b: float) -> tuple[str, list[dict]]:
    """Pick a default profile + alternatives."""
    default = _DEFAULT_PROFILE_BY_CLASS.get(family_class, "JANG_4K")
    alts: list[dict] = []

    if family_class == "minicpm_text":
        # MiniCPM4-0.5B live validation supports JANG_6M. JANG_4M has a
        # cache-independent multi-turn quality failure and immediate EOS near
        # 32K, so lower profiles must not appear as beginner alternatives.
        return "JANG_6M", []
    if family_class == "moe_large_expert":
        # Default JANG_2L; alts JANG_4M (more quality) and JANG_2S (tight RAM, risk)
        alts.append({"profile": "JANG_4M", "use_when": "You have enough RAM for a larger output and want higher quality."})
        alts.append({"profile": "JANG_2S", "use_when": "Your RAM is very tight. Risk of degraded output on some models."})
    elif family_class in ("moe_standard", "moe_hybrid_ssm", "moe_mla"):
        alts.append({"profile": "JANG_2L", "use_when": "You need the smallest high-quality output. Proven on 2-bit."})
        alts.append({"profile": "JANG_6M", "use_when": "You want near-lossless quality — use if disk isn't a concern."})
    elif family_class == "dense_llm":
        if param_b >= 30:
            alts.append({"profile": "JANG_2L", "use_when": "The model is large; JANG_2L keeps quality high at the smallest size."})
        alts.append({"profile": "JANG_6M", "use_when": "Near-lossless — use if you have disk and want the highest quality."})
    elif family_class == "hybrid_ssm_mtp":
        alts.append({"profile": "JANG_4M", "use_when": "Nemotron-style hybrid models work well at JANG_4M for higher quality."})
    elif family_class in ("vl_image", "vl_video"):
        alts.append({"profile": "JANG_4M", "use_when": "VL models benefit from the extra 8-bit attention JANG_4M provides."})
    return default, alts


def _recommend_hadamard(profile: str) -> tuple[bool, str]:
    """Hadamard helps 3-bit+ and HURTS 2-bit. Default true for 3+, false for 1/2-bit.

    M142 (iter 64): use the authoritative compress-bits from
    `allocate.JANG_PROFILES` instead of a hardcoded profile-name list.
    Same data-driven approach as the Swift-side
    `PreflightRunner.hadamardVsLowBits` to keep the two sides in sync,
    and robust against future profile name additions
    (e.g., JANG_1S, JANGTQ1 — today's hardcoded list would miss them
    and recommend hadamard=True which is wrong at 1-bit).
    """
    from .allocate import JANG_PROFILES, JANG_K_TARGETS
    compress_bits: int
    if profile in JANG_PROFILES:
        compress_bits = JANG_PROFILES[profile][2]  # (critical, important, COMPRESS)
    elif profile in JANG_K_TARGETS:
        # K-quant profiles are uniform at their target bit-width.
        compress_bits = int(round(JANG_K_TARGETS[profile]))
    elif profile.startswith("JANGTQ"):
        # JANGTQ2 / JANGTQ3 / JANGTQ4 — suffix digit is the uniform bit-width.
        suffix = profile.removeprefix("JANGTQ").lstrip("_")
        try:
            compress_bits = int(suffix[0]) if suffix and suffix[0].isdigit() else 4
        except (ValueError, IndexError):
            compress_bits = 4
    else:
        # Unknown profile — match the Swift fallback (pass-through: treat
        # as non-low-bit so hadamard defaults on).
        compress_bits = 4
    if compress_bits <= 2:
        return (False, "Hadamard rotation hurts quality at 2-bit and below, so we leave it off.")
    return (True, "Hadamard rotation reduces quantization error at 3-bit and higher — we turn it on by default.")


def _recommend_dtype(
    model_type: str,
    source_dtype: str,
    expert_count: int = 0,
) -> tuple[str | None, str]:
    """Force bfloat16 for 512+ expert models; otherwise auto (follow source).

    M131 (iter 53): peer-helper parity with `_classify_family` — that helper
    dynamically promotes any MoE model with expert_count >= 512 to
    "moe_large_expert". Pre-iter-53, this helper only checked the named
    set `_BF16_REQUIRED = {"minimax_m2", "glm_moe_dsa"}` — so a future
    512+ expert family (or a custom 512-expert qwen3_5_moe) would get
    force_dtype=None while ``recommend()``'s warning block at the caller
    said "bfloat16 is required to avoid float16 overflow". Self-
    contradicting recommendation; user gets NaN at runtime. Checking
    ``expert_count >= 512`` in-helper aligns with the warning and with
    ``_classify_family``'s dynamic promotion rule.
    """
    if model_type in _BF16_REQUIRED or expert_count >= 512:
        reason_name = model_type if expert_count == 0 else f"{model_type} ({expert_count} experts)"
        return (
            "bfloat16",
            f"{reason_name} has 512+ experts. Float16 overflows at this scale; "
            "we force bfloat16 which has a wider exponent range.",
        )
    return (None, "We use the source model's native dtype — auto-detected.")


# ─────────────────────────── Main entry point ───────────────────────────────


def detect(model_path: Path) -> dict[str, Any]:
    """Inspect the source and return a full detected snapshot."""
    cfg_path = model_path / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found under {model_path}")
    # M120: match inspect_source's error-surface behavior — a bad config.json
    # must produce a diagnostic that includes the path so the wizard's error
    # banner can point the user at the right file. Pre-fix, the top-level
    # `except Exception` printed `JSONDecodeError: ...` with no path, leaving
    # the user to guess which file was malformed.
    try:
        raw = cfg_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"could not read config.json at {cfg_path}: {exc}") from exc
    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"config.json at {cfg_path} is not valid JSON "
            f"(line {exc.lineno}, col {exc.colno}): {exc.msg}"
        ) from exc
    if not isinstance(cfg, dict):
        raise ValueError(
            f"config.json at {cfg_path} has a top-level "
            f"{type(cfg).__name__}, expected a JSON object"
        )
    verified_minicpm = is_verified_minicpm4_05b_config(cfg, model_path)
    model_type = cfg.get("model_type") or (cfg.get("text_config", {}) or {}).get("model_type")
    if not model_type:
        if verified_minicpm:
            model_type = "minicpm"
        else:
            model_type = "unknown"
    expert_count = int(cfg.get("num_experts") or cfg.get("n_routed_experts") or cfg.get("num_local_experts") or 0)
    is_vl = (model_path / "preprocessor_config.json").exists()
    is_video_vl = (model_path / "video_preprocessor_config.json").exists()
    source_dtype = _sniff_dtype(model_path)
    has_tool_parser = bool(cfg.get("tool_call_parser") or cfg.get("tool_choice_parser"))
    has_reasoning_parser = bool(cfg.get("reasoning_parser") or cfg.get("thinking_parser") or cfg.get("enable_thinking"))
    name_or_path = cfg.get("_name_or_path", "") or str(model_path)
    is_gated = any(name_or_path.startswith(p) for p in _GATED_PREFIXES)

    family_class = _classify_family(model_type, expert_count, is_vl, is_video_vl)
    param_b = _estimate_params_billion(cfg)

    return {
        "model_type": model_type,
        "family_class": family_class,
        "param_count_billions": param_b,
        "expert_count": expert_count,
        "is_moe": _is_moe(cfg),
        "is_vl": is_vl,
        "is_video_vl": is_video_vl,
        "source_dtype": source_dtype,
        "has_tool_parser": has_tool_parser,
        "has_reasoning_parser": has_reasoning_parser,
        "is_gated_model": is_gated,
        "name_or_path": name_or_path,
        "verified_minicpm_profile": verified_minicpm,
    }


def recommend(model_path: Path) -> dict[str, Any]:
    """Generate a full recommendation for a source model."""
    detected = detect(model_path)
    model_type = detected["model_type"]
    family_class = detected["family_class"]
    expert_count = detected["expert_count"]
    param_b = detected["param_count_billions"]
    source_dtype = detected["source_dtype"]

    if model_type == "minicpm" and not detected["verified_minicpm_profile"]:
        raise ValueError(
            "No validated JANG profile is registered for this MiniCPM variant; "
            "the current JANG_6M recommendation is limited to MiniCPM4-0.5B"
        )

    family, family_alts, family_reason = _recommend_family(model_type, source_dtype)
    profile, profile_alts = _recommend_profile(family_class, expert_count, param_b)
    hadamard, hadamard_reason = _recommend_hadamard(profile)
    force_dtype, dtype_reason = _recommend_dtype(model_type, source_dtype, expert_count)

    warnings: list[str] = []
    if expert_count >= 512:
        warnings.append(f"{expert_count} experts detected — bfloat16 is required to avoid float16 overflow.")
    if expert_count >= 512:
        warnings.append("Large-expert MoE uses special MLP floor values (gate=4-bit, down=3-bit) automatically.")
    if detected["is_gated_model"]:
        warnings.append(
            f"This model ({detected['name_or_path']}) appears gated on HuggingFace. "
            "You may need to accept terms at the HF repo page before downloading."
        )
    if family_class == "hybrid_ssm_mtp":
        warnings.append(
            "Nemotron-H requires custom modeling_*.py files — JANG copies them automatically."
        )
    if source_dtype == "unknown":
        warnings.append(
            "Source dtype could not be sniffed from the shard header. "
            "If conversion fails, try setting Force dtype in Advanced overrides."
        )
    if model_type == "unknown" or model_type not in _MODEL_TYPE_TO_FAMILY_CLASS:
        warnings.append(
            f"Unknown model_type ({model_type}) — recommendations use generic dense-LLM defaults. "
            "Review the Architecture step carefully before converting."
        )

    class_prose = _CLASS_PROSE.get(family_class, "Your model will be converted using generic defaults.")
    profile_desc = _PROFILE_PROSE.get(profile, "")

    beginner_summary = (
        f"This is a {class_prose.split('.')[0].lower()}. "
        f"We recommend profile {profile} — {profile_desc.split('.')[0].lower()}. "
        f"{'You can switch to JANGTQ for extra compression; see alternatives below.' if family_alts else ''}"
    ).strip()

    alternatives: list[dict] = []
    alternatives.extend(family_alts)
    for alt in profile_alts:
        alternatives.append({
            "family": "jang",
            "profile": alt["profile"],
            "use_when": alt["use_when"],
        })

    return {
        "detected": detected,
        "recommended": {
            "family": family,
            "profile": profile,
            "method": "mse",
            "hadamard": hadamard,
            "block_size": 64,
            "force_dtype": force_dtype,
            "alternatives": alternatives,
        },
        "beginner_summary": beginner_summary,
        "warnings": warnings,
        "why_each_choice": {
            "family": family_reason,
            "profile": f"{_PROFILE_PROSE.get(profile, '')} This was picked because: {class_prose}",
            "method": "MSE (minimum-square-error) weight search is our best-quality method; use RTN only for fastest conversion with slight quality loss.",
            "hadamard": hadamard_reason,
            "block_size": "Block size 64 is our default; it balances quantization granularity with metadata overhead across every architecture.",
            "force_dtype": dtype_reason,
        },
    }


def cmd_recommend(args) -> None:
    """CLI entry: python -m jang_tools recommend --model <dir> [--json]"""
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model dir not found: {model_path}", file=sys.stderr)
        sys.exit(2)
    try:
        rec = recommend(model_path)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(3)
    if args.json:
        print(json.dumps(rec, indent=None))
    else:
        d = rec["detected"]
        r = rec["recommended"]
        print(f"Model: {d['name_or_path']}")
        print(f"Type:  {d['model_type']} ({d['family_class']}, ~{d['param_count_billions']}B params)")
        if d["expert_count"]:
            print(f"       {d['expert_count']} experts")
        if d["is_video_vl"]:
            print("       Vision + Video")
        elif d["is_vl"]:
            print("       Vision (image)")
        print(f"Dtype: {d['source_dtype']}")
        print()
        print("Recommended:")
        print(f"  Family: {r['family']}")
        print(f"  Profile: {r['profile']}")
        print(f"  Method: {r['method']}")
        print(f"  Hadamard: {r['hadamard']}")
        print(f"  Block size: {r['block_size']}")
        if r["force_dtype"]:
            print(f"  Force dtype: {r['force_dtype']}")
        print()
        print(f"{rec['beginner_summary']}")
        if rec["warnings"]:
            print()
            print("Warnings:")
            for w in rec["warnings"]:
                print(f"  * {w}")
        if r["alternatives"]:
            print()
            print("Alternatives:")
            for alt in r["alternatives"]:
                fam = alt.get("family", "jang")
                prof = alt.get("profile", "")
                print(f"  {fam} {prof}: {alt['use_when']}")


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "recommend",
        help="Recommend a conversion plan for a source model (beginner-friendly)",
    )
    p.add_argument("--model", required=True, help="Path to HuggingFace model directory")
    p.add_argument("--json", action="store_true", help="Emit JSON")
    p.set_defaults(func=cmd_recommend)
