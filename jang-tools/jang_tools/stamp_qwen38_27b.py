"""Stamp the Qwen3.8-27B bundle contract (vision + video + MTP + reasoning).

Everything here is taken from the model card and `generation_config.json` of
`Qwen/Qwen3.8-27B`, verified against the downloaded source on 2026-08-14.

SAMPLING — the card documents **three** modes, not one. We stamp all three and
make the thinking/general one the default, because that is what upstream's
`generation_config.json` carries:

    thinking / general   temp 1.0  top_p 0.95  top_k 20  min_p 0  presence 0.0
    thinking / coding    temp 0.6  top_p 0.95  top_k 20  min_p 0  presence 0.0
    instruct (no-think)  temp 0.7  top_p 0.80  top_k 20  min_p 0  presence 1.5

repetition_penalty is 1.0 in all three. Shipping only one number would lose the
coding preset, which is the one that matters most for agentic use — the same
class of loss as the Laguna XS `top_k=20` incident
(`feedback_sampling_defaults_two_file_contract`).

REASONING — thinking is **ON by default**. Verified empirically: the generation
prompt with no kwarg is byte-identical to `enable_thinking=True` and ends
`<|im_start|>assistant\\n<think>\\n`; `enable_thinking=False` instead prefills a
*closed* block `<think>\\n\\n</think>\\n\\n`. So reasoning-off is a prefill, not
an omission. Reasoning parser upstream is `qwen3`.

TOOLS — `qwen3_coder`. mlx_lm already infers this correctly from the template
(it contains the `<tool_call>\\n<function=` literal), so unlike the LFM2.5 line
no detection hint is needed. We still stamp it explicitly.

MODALITY — vision **and video**: the source ships both
`preprocessor_config.json` and `video_preprocessor_config.json`, and carries 333
`model.visual.*` tensors. No audio.

MTP — 15 real `mtp.*` tensors (`mtp.fc` + one transformer block), shared
embeddings (`mtp_use_dedicated_embeddings: False`). Upstream drives it as
`qwen3_next_mtp` with 2 speculative tokens. `runtime_available` stays False
until an MLX runtime actually decodes with it.

    python -m jang_tools.stamp_qwen38_27b <bundle_dir> [more...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ._json_utils import write_json_object_atomic
from .capabilities import write_validated_jang_config

# Qwen3.8's card documents exactly TWO presets (unlike 3.6's three — the
# coding preset does not exist on the 3.8 card; do not carry it over):
#   Thinking:  temp 1.0  top_p 0.95  top_k 20  min_p 0  presence 0.0
#   Instruct:  temp 0.7  top_p 0.80  top_k 20  min_p 0  presence 1.5
SAMPLING_MODES = {
    "thinking": {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                 "min_p": 0.0, "presence_penalty": 0.0,
                 "repetition_penalty": 1.0},
    "instruct_nothinking": {"temperature": 0.7, "top_p": 0.80, "top_k": 20,
                            "min_p": 0.0, "presence_penalty": 1.5,
                            "repetition_penalty": 1.0},
}
DEFAULT_MODE = "thinking"

EOS_IDS = [248046, 248044]
BOS_ID = 248044
PAD_ID = 248044
SOURCE_TAG = "vendor_card+generation_config_2026-08-14"


def _depth(raw: object) -> int | None:
    if type(raw) is not int or not 1 <= raw <= 3:
        return None
    return raw


def _positive_number(raw: object) -> float | None:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    return value if value > 0 else None


def resolve_mtp_tuning_depth(tuning: dict) -> tuple[int, str]:
    """Resolve a safe default draft depth from one tuning sidecar.

    Tokens per verification cycle are acceptance diagnostics, not a throughput
    oracle. A validated recommendation therefore needs measured wall speed,
    and a per-depth speed table must agree with ``best_depth``. Unvalidated
    stamps may only carry the conservative D1 seed.
    """
    if not isinstance(tuning, dict):
        raise ValueError("vmlx_mtp_tuning.json must contain a JSON object")
    if tuning.get("blocked") is True:
        raise ValueError("vmlx_mtp_tuning.json blocks native MTP")

    depth = _depth(tuning.get("best_depth"))
    if depth is None:
        raise ValueError("vmlx_mtp_tuning.json best_depth must be an integer from 1 to 3")

    if tuning.get("validated") is not True:
        if depth != 1:
            raise ValueError("unvalidated native-MTP tuning may only recommend depth 1")
        return 1, "conservative unmeasured D1"

    if tuning.get("output_equivalent") is False:
        raise ValueError("validated native-MTP tuning explicitly failed output equivalence")
    evidence_keys = ("baseline_tok_s", "best_tok_s", "speedup_vs_baseline")
    missing = [key for key in evidence_keys if _positive_number(tuning.get(key)) is None]
    if missing:
        raise ValueError(f"validated native-MTP tuning lacks positive wall-speed evidence: {missing}")

    raw_speeds = tuning.get("measured_tok_s_by_depth")
    if raw_speeds is not None:
        if not isinstance(raw_speeds, dict):
            raise ValueError("measured_tok_s_by_depth must be a JSON object")
        speeds: dict[int, float] = {}
        for raw_key, raw_speed in raw_speeds.items():
            try:
                key = int(raw_key)
            except (TypeError, ValueError):
                continue
            speed = _positive_number(raw_speed)
            if 1 <= key <= 3 and speed is not None:
                speeds[key] = speed
        if depth not in speeds:
            raise ValueError(f"measured_tok_s_by_depth has no valid D{depth} result")
        fastest_speed = max(speeds.values())
        fastest_depth = min(key for key, speed in speeds.items() if speed == fastest_speed)
        if fastest_depth != depth:
            raise ValueError(
                f"best_depth={depth} contradicts measured wall speed; D{fastest_depth} "
                f"is fastest ({fastest_speed:g} tok/s)"
            )

    speedup = _positive_number(tuning.get("speedup_vs_baseline"))
    if depth > 1 and speedup is not None and speedup <= 1.0:
        raise ValueError(
            "validated native-MTP D2/D3 speedup_vs_baseline must exceed 1.0"
        )

    return depth, f"validated measured D{depth}"


def stamp(b: Path) -> dict:
    cfg_p, jang_p, gen_p = (b / "config.json", b / "jang_config.json",
                            b / "generation_config.json")
    cfg = json.loads(cfg_p.read_text())
    jang = json.loads(jang_p.read_text()) if jang_p.exists() else {}
    gen = json.loads(gen_p.read_text()) if gen_p.exists() else {}

    d = SAMPLING_MODES[DEFAULT_MODE]

    # ── two-file sampling contract ────────────────────────────────────────
    gen.update({
        "do_sample": True,
        "temperature": d["temperature"], "top_p": d["top_p"], "top_k": d["top_k"],
        "repetition_penalty": d["repetition_penalty"],
        "eos_token_id": EOS_IDS, "bos_token_id": BOS_ID, "pad_token_id": PAD_ID,
    })
    chat = jang.get("chat") or {}
    chat["sampling_defaults"] = dict(SAMPLING_MODES[DEFAULT_MODE],
                                     source=SOURCE_TAG, mode=DEFAULT_MODE)
    chat["sampling_modes"] = SAMPLING_MODES
    chat["stop_token_ids"] = EOS_IDS
    # Card "Best Practices": within 1M ctx allocate up to 262144 reasoning
    # tokens and 131072 final-response tokens for agentic work.
    chat["recommended_max_tokens"] = {"reasoning": 262144, "final_response": 131072}
    chat["context_length"] = {"native": 262144, "extensible": 1000000}
    jang["chat"] = chat

    # ── reasoning ─────────────────────────────────────────────────────────
    jang["reasoning"] = {
        "supported": True,
        "parser": "qwen3",
        "default": "on",
        "think_in_template": True,
        "enable_kwarg": "enable_thinking",
        "off_is_prefilled_closed_block": True,
        "off_prefill": "<think>\n\n</think>\n\n",
        # SETTLED KEY NAMES (2026-08-14, agreed with the vmlx-side agent):
        # `supported_reasoning_efforts` mirrors the engine's own identifier so
        # wiring is a straight read — the panel must constrain to THESE values
        # (no `high`, no `max`; the template raises on anything else).
        "supported_reasoning_efforts": ["low", "medium", "xhigh"],
        "default_reasoning_effort": "xhigh",
        # transport: the chat template consumes `reasoning_effort` as a
        # template kwarg; API layers surface it as a top-level field and map
        # it into chat_template_kwargs.
        "reasoning_effort_transport": "chat_template_kwarg",
        "preserve_thinking_supported": True,
        "preserve_thinking_default": True,
        "preserve_thinking_transport": "chat_template_kwarg",
        "note": ("Thinking is ON by default; disabling prefills a CLOSED think "
                 "block. NEW in 3.8: reasoning_effort tiers xhigh (default) / "
                 "medium / low — template raises on any other value. "
                 "preserve_thinking is ON BY DEFAULT (verified empirically: "
                 "no-kwarg render keeps history <think> blocks) — the OPPOSITE "
                 "of 3.6-style truncation; stable history improves prefix/KV "
                 "cache reuse and agent decision consistency."),
    }

    jang["tools"] = {"supported": True, "parser": "qwen3_coder",
                     "dialect": "xml_function",
                     "mlx_lm_autodetected": True}

    # ── modality: weight-gated ────────────────────────────────────────────
    jang["vision"] = {
        "supported": True,
        "video_supported": (b / "video_preprocessor_config.json").exists(),
        "tower": "qwen3_5 ViT (27 layers, hidden 1152)",
        "processor": "preprocessor_config.json",
        "video_processor": "video_preprocessor_config.json",
    }

    idx = b / "model.safetensors.index.json"
    mtp_n = 0
    if idx.exists():
        wm = json.loads(idx.read_text()).get("weight_map", {})
        mtp_n = sum(1 for k in wm if k.startswith("mtp."))

    tuning_p = b / "vmlx_mtp_tuning.json"
    tuning_to_write: dict | None = None
    resolved_depth = 1
    depth_basis = "no MTP artifact"
    if mtp_n:
        if tuning_p.exists():
            tuning = json.loads(tuning_p.read_text())
        else:
            q = cfg.get("quantization", {})
            tuning = {
                "best_depth": 1,
                "blocked": False,
                "model_types": ["qwen3_5"],
                "artifact": b.name,
                "quantization_mode": q.get("mode", "affine"),
                "quantization_bits": q.get("bits"),
                "note": (
                    "Conservative UNMEASURED default: 1 draft/step. "
                    "Tokens per cycle do not choose a wall-throughput winner; "
                    "run matched-output D1/D2/D3 timing before validating a deeper depth."
                ),
                "reason": "stamped by stamp_qwen38_27b; recommendation, not measurement",
            }
            tuning_to_write = tuning
        try:
            resolved_depth, depth_basis = resolve_mtp_tuning_depth(tuning)
        except ValueError as exc:
            raise ValueError(f"{tuning_p}: {exc}") from exc
    # ── MTP: stamp the keys the RUNTIMES actually read ────────────────────
    # Verified against both runtimes (2026-08-13):
    #   vmlx  : native_mtp.py reads jang_config.runtime.mtp_layers,
    #           jang_config.mtp.num_layers, jang_config.drop_mtp, and the
    #           vmlx_mtp_tuning.json sidecar (depth = NUMBER OF DRAFTS,
    #           default 3, clamp 1..3).
    #   swift : JangLoader reads jang_config.runtime.{bundle_has_mtp,
    #           mtp_layers, mtp_mode}; NativeMTPTuning reads the same flat
    #           snake_case sidecar and refuses best_depth unless
    #           validated+output_equivalent+measured tok/s are present.
    # NOTATION: draft count is the only unambiguous unit. vmlx "depth" and
    # vLLM num_speculative_tokens both count DRAFTS; the kit docs' D<n>
    # counts tokens/cycle (kit D2 == 1 draft == vmlx depth 1).
    jang["drop_mtp"] = False
    runtime = jang.get("runtime") or {}
    runtime.update({
        "bundle_has_mtp": mtp_n > 0,
        "mtp_layers": 1 if mtp_n else 0,
        "mtp_mode": "preserved_enabled" if mtp_n else "none",
        "mtp_num_speculative_tokens": 2,   # upstream serving config (drafts)
        "mtp_status": (
            "MTP head preserved for native speculative decode; "
            f"recommended {resolved_depth} draft(s)/step from {depth_basis}."
        ),
    })
    jang["runtime"] = runtime
    jang["mtp"] = {
        "num_layers": 1 if mtp_n else 0,   # the key vmlx native_mtp.py reads
        "artifact_available": mtp_n > 0,
        "tensor_count": mtp_n,
        "runtime_available": False,
        "dedicated_embeddings": False,
        "upstream_method": "qwen3_next_mtp",
        "upstream_num_speculative_tokens": 2,   # DRAFTS (== kit D3)
        "trained_multi_step": True,   # card: "MTP: trained with multiple steps"
        "recommended_num_drafts": resolved_depth,
        "notation": ("Draft count is the unambiguous unit. vmlx native-MTP "
                     "'depth' and vLLM num_speculative_tokens both count "
                     "DRAFTS; kit docs' D<n> counts tokens/cycle "
                     "(kit D2 == 1 draft == vmlx depth 1)."),
    }

    caps = jang.get("capabilities") or cfg.get("capabilities") or {}
    caps.update({
        "has_vision": True,
        "has_video": bool(jang["vision"]["video_supported"]),
        "has_audio": False,
        "modality": "multimodal" if jang["vision"]["video_supported"] else "vision",
        "modalities": {"text": True, "vision": True,
                       "video": bool(jang["vision"]["video_supported"]),
                       "audio": False},
        "supports_tools": True, "tool_parser": "qwen3_coder",
        "supports_thinking": True, "default_reasoning": "on",
        "think_in_template": True, "reasoning_parser": "qwen3",
        "family": "qwen3_5", "cache_type": "hybrid",
    })
    jang["capabilities"] = caps
    cfg["capabilities"] = caps

    write_json_object_atomic(gen_p, gen)
    write_json_object_atomic(cfg_p, cfg)
    write_validated_jang_config(b, jang, cfg)
    if tuning_to_write is not None:
        write_json_object_atomic(tuning_p, tuning_to_write)
    return {"bundle": b.name, "mtp": mtp_n,
            "video": jang["vision"]["video_supported"],
            "mtp_depth": resolved_depth, "mtp_depth_basis": depth_basis}


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    d = SAMPLING_MODES[DEFAULT_MODE]
    for x in argv[1:]:
        r = stamp(Path(x).expanduser())
        print(f"  stamped {r['bundle']:34s} T={d['temperature']} top_p={d['top_p']} "
              f"top_k={d['top_k']} eos={EOS_IDS} reasoning=on vision=True "
              f"video={r['video']} mtp={r['mtp']} (+3 sampling modes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
