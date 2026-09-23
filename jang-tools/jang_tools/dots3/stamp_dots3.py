"""Stamp the dots3-note bundle contract (vision+video+audio + MTP + reasoning).

Facts verified against dots-studio/dots3-note-prev source (2026-08-14):

SAMPLING — vendor README quickstart: temperature 1.0, top_p 0.95 (no top_k,
no penalties). generation_config.json upstream carries ONLY eos ids — the
two-file trap — so we stamp full presets in BOTH files. Three modes:

    thinking_general (DEFAULT) 1.0 / 0.95      <- vendor README
    agentic_coding             0.6 / 0.95      <- OUR coding default
       (provenance: DSV4-0731 + Qwen3.6-27B precedent — exact-identifier
        fidelity; vendor publishes no coding preset; vLLM recipe example 0.7)
    instruct_nothinking        0.7 / 0.95      <- vLLM recipe example temp

REASONING — thinking ON by default, native in the template
(`enable_thinking` defaults true; OFF prefills a CLOSED
`<think>\\n\\n</think>\\n\\n` block AND appends `<no_think>` to the user turn).

TOOLS — dots XML dialect (`<dots_function_call>/<invoke>/<parameter>`);
serving parser name upstream: `dots` (SGLang/vLLM `--tool-call-parser dots`).

MODALITY — vision + video + audio, weight-gated on actual tower tensors.

MTP — model.layers.46 (MLA + dense FFN + eh_proj + enorm/hnorm/
shared_head.norm) + model.mtp.embed_tokens (deduped when byte-identical).
Stamped in the schema the runtimes ACTUALLY read (Qwen3.6-27B standard,
commits cb2bbb4/1ea7c96): jang.mtp.num_layers, jang.runtime.{bundle_has_mtp,
mtp_layers,mtp_mode}, drop_mtp, + vmlx_mtp_tuning.json sidecar. Vendor
serving uses 3 drafts; OUR recommendation on Apple silicon is 1 draft
(kit D2) until a measured depth sweep exists — recommendation, not
measurement, and marked as such.

    python -m jang_tools.dots3.stamp_dots3 <bundle_dir> [more...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SAMPLING_MODES = {
    "thinking_general": {"temperature": 1.0, "top_p": 0.95, "top_k": 0,
                         "min_p": 0.0, "presence_penalty": 0.0,
                         "repetition_penalty": 1.0},
    "agentic_coding": {"temperature": 0.6, "top_p": 0.95, "top_k": 0,
                       "min_p": 0.0, "presence_penalty": 0.0,
                       "repetition_penalty": 1.0},
    "instruct_nothinking": {"temperature": 0.7, "top_p": 0.95, "top_k": 0,
                            "min_p": 0.0, "presence_penalty": 0.0,
                            "repetition_penalty": 1.0},
}
DEFAULT_MODE = "thinking_general"
MODE_SOURCES = {
    "thinking_general": "vendor_readme_2026-08-14",
    "agentic_coding": "jang_default_coding_preset (DSV4-0731/Qwen36-27B "
                      "precedent; vendor publishes none; vLLM example 0.7)",
    "instruct_nothinking": "vllm_recipe_example_2026-08-14",
}

EOS_IDS = [151643, 151668]
BOS_ID = 151643
PAD_ID = 151659


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
        "temperature": d["temperature"], "top_p": d["top_p"],
        "top_k": d["top_k"], "repetition_penalty": d["repetition_penalty"],
        "eos_token_id": EOS_IDS, "bos_token_id": BOS_ID, "pad_token_id": PAD_ID,
    })
    chat = jang.get("chat") or {}
    chat["sampling_defaults"] = dict(d, source=MODE_SOURCES[DEFAULT_MODE],
                                     mode=DEFAULT_MODE)
    chat["sampling_modes"] = {
        k: dict(v, source=MODE_SOURCES[k]) for k, v in SAMPLING_MODES.items()}
    chat["stop_token_ids"] = EOS_IDS
    jang["chat"] = chat

    # ── reasoning: ON by default, native ─────────────────────────────────
    jang["reasoning"] = {
        "supported": True,
        "parser": "dots3",
        "default": "on",
        "think_in_template": True,
        "enable_kwarg": "enable_thinking",
        "off_is_prefilled_closed_block": True,
        "off_prefill": "<think>\n\n</think>\n\n",
        "off_user_marker": "<no_think>",
        "tiers": None,
        "note": ("Thinking ON by default: chat_template sets "
                 "enable_thinking=true when undefined. Disabling appends "
                 "<no_think> to the user turn AND prefills a CLOSED think "
                 "block — it is not an omission."),
    }

    jang["tools"] = {"supported": True, "parser": "dots",
                     "dialect": "dots_xml_function_call",
                     "call_open": "<dots_function_call>",
                     "response_wrap": "<dots_function_response>",
                     "mlx_lm_autodetected": False,
                     "detection_hint": ("template literal "
                                        "'<dots_function_call>' — no mlx_lm "
                                        "parser exists for this dialect yet")}

    # ── modality: weight-gated ────────────────────────────────────────────
    idx = b / "model.safetensors.index.json"
    wm = {}
    if idx.exists():
        wm = json.loads(idx.read_text()).get("weight_map", {})
    has_vision = any(k.startswith("vision_encoder.") for k in wm)
    has_audio = any(k.startswith("audio_encoder.") for k in wm)
    mtp_n = sum(1 for k in wm
                if k.startswith(("model.layers.46.", "model.mtp.")))
    jang["vision"] = {
        "supported": has_vision,
        "video_supported": has_vision,      # video shares the vision tower
        "tower": "dots3 MoE-ViT (42 blocks, 608 pyramid experts, 1.2B act)",
        "processor": "preprocessor_config.json",
        "video_processor": "video preprocessing via dots3_note processor "
                           "(config-embedded; no separate file upstream)",
    }
    jang["audio"] = {
        "supported": has_audio,
        "tower": "dots3 speech encoder (32 layers, swiglu whisper-shape, "
                 "conv2d stem, 800M)",
        "config": "config.json audio_config (no separate processor file)",
        "note": "video inputs include their audio track when available",
    }

    # ── MTP: the schema the runtimes ACTUALLY read (qwen36-27b standard) ──
    mtp_embed_shared = bool(jang.get("mtp_embed_shared", False))
    jang["drop_mtp"] = False
    runtime = jang.get("runtime") or {}
    runtime.update({
        "bundle_has_mtp": mtp_n > 0,
        "mtp_layers": 1 if mtp_n else 0,
        "mtp_mode": "preserved_enabled" if mtp_n else "none",
        "mtp_num_speculative_tokens": 1,
        "mtp_status": ("MTP layer 46 preserved for native speculative "
                       "decode; recommended 1 draft/step on Apple silicon "
                       "(unmeasured on this artifact — run a depth sweep "
                       "to validate)."),
    })
    jang["runtime"] = runtime
    jang["mtp"] = {
        "num_layers": 1 if mtp_n else 0,
        "artifact_available": mtp_n > 0,
        "tensor_count": mtp_n,
        "runtime_available": False,
        "dedicated_embeddings": (not mtp_embed_shared) and any(
            k.startswith("model.mtp.embed_tokens") for k in wm),
        "embed_shared_with_backbone": mtp_embed_shared,
        "layout": "dsv3_fusion_at_layer_46 (eh_proj/enorm/hnorm/"
                  "shared_head.norm + MLA + dense FFN)",
        "upstream_method": "nextn/mtp (SGLang NEXTN, vLLM mtp)",
        "upstream_num_speculative_tokens": 3,   # DRAFTS, vendor serving cfg
        "recommended_num_drafts": 1,            # == kit D2 == vmlx depth 1
        "notation": ("Draft count is the unambiguous unit. vmlx native-MTP "
                     "'depth' and vLLM num_speculative_tokens both count "
                     "DRAFTS; kit docs' D<n> counts tokens/cycle "
                     "(kit D2 == 1 draft == vmlx depth 1)."),
    }
    if mtp_n:
        tuning_p = b / "vmlx_mtp_tuning.json"
        if not tuning_p.exists():   # never clobber a real measured sweep
            q = cfg.get("quantization", {})
            tuning_p.write_text(json.dumps({
                "best_depth": 1,
                "blocked": False,
                "model_types": ["dots3_note"],
                "artifact": b.name,
                "quantization_mode": "affine",
                "quantization_bits": q.get("bits"),
                "note": ("Conservative UNMEASURED default: 1 draft/step. "
                         "Run a depth sweep and write validated, "
                         "output_equivalent, baseline_tok_s, best_tok_s, "
                         "speedup_vs_baseline to let Swift honor best_depth."),
                "reason": "stamped by stamp_dots3; recommendation, not measurement",
            }, indent=2) + "\n")

    caps = jang.get("capabilities") or cfg.get("capabilities") or {}
    caps.update({
        "has_vision": has_vision, "has_video": has_vision,
        "has_audio": has_audio,
        "modality": "omni" if (has_vision and has_audio) else (
            "vision" if has_vision else "text"),
        "modalities": {"text": True, "vision": has_vision,
                       "video": has_vision, "audio": has_audio},
        "supports_tools": True, "tool_parser": "dots",
        "supports_thinking": True, "default_reasoning": "on",
        "think_in_template": True, "reasoning_parser": "dots3",
        "family": "dots3_note", "cache_type": "hybrid_full_swa",
        "context_length": 524288,
        "dsa_index_topk": 2048,
    })
    jang["capabilities"] = caps
    cfg["capabilities"] = caps

    gen_p.write_text(json.dumps(gen, indent=2) + "\n")
    jang_p.write_text(json.dumps(jang, indent=2) + "\n")
    cfg_p.write_text(json.dumps(cfg, indent=2) + "\n")
    return {"bundle": b.name, "mtp": mtp_n, "vision": has_vision,
            "audio": has_audio}


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    d = SAMPLING_MODES[DEFAULT_MODE]
    for x in argv[1:]:
        r = stamp(Path(x).expanduser())
        print(f"  stamped {r['bundle']:40s} T={d['temperature']} "
              f"top_p={d['top_p']} eos={EOS_IDS} reasoning=on "
              f"vision={r['vision']} audio={r['audio']} mtp={r['mtp']} "
              f"(+3 sampling modes incl agentic_coding 0.6)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
