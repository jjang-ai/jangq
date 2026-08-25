"""Stamp the Raptor-8B-A1B (lfm2_moe) serving contract into a JANG bundle.

Every value is taken from the source package — `config.json`,
`generation_config.json`, `raptor_generation_profiles.json` and
`lfm25_contract.py` — not invented here.

ARCHITECTURE (Lfm2MoeForCausalLM, model_type `lfm2_moe`)
  24 layers, hybrid `layer_types`: LIV short-conv blocks + 6 full-attention.
  hidden 2048, 32 experts / 4 active (moe_intermediate 1792), 2 dense layers,
  `use_expert_bias`, `norm_topk_prob`, routed_scaling_factor 1.0.
  Tied embeddings, vocab 128000, 128K context.

REASONING DEFAULT
  The vendor's `default` profile sets `reasoning_mode: "off"`, but the chat
  template emits `<think>` unprompted and Eric's call is reasoning ON. So the
  stamped default is the vendor's own `interactive_reasoning` profile —
  greedy + reasoning on — which is a vendor-defined combination, not a
  new one.

SAMPLING
  🚨 The vendor default is GREEDY (`do_sample: false`), not a temperature.
  Only the benchmark profiles sample, and `liquid_native_diagnostic_only`
  (T=0.2/top_k=80/rep 1.05) is explicitly diagnostic — it is carried but
  flagged, never a serving default.

    python -m jang_tools.stamp_raptor8b <bundle> [--src <source_dir>]
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from ._json_utils import write_json_object_atomic
from .capabilities import build_capabilities, write_validated_jang_config

SRC_DEFAULT = Path("/Users/eric/models/Raptor-8B-A1B")

# from the tokenizer, verified 2026-08-24
IM_END, THINK_OPEN, THINK_CLOSE = 124900, 124901, 124902
TOOL_START, TOOL_END = 124905, 124906
BOS, PAD = 124894, 124893


def main(argv) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    bundle = Path(argv[1])
    src = SRC_DEFAULT
    for i, a in enumerate(argv):
        if a == "--src":
            src = Path(argv[i + 1])

    cfg = json.loads((bundle / "config.json").read_text())
    profiles = json.loads((src / "raptor_generation_profiles.json").read_text())

    # carry the vendor profile file into the bundle so the contract is auditable
    for f in ("raptor_generation_profiles.json", "MODEL_STATUS.json"):
        if (src / f).exists():
            shutil.copy2(src / f, bundle / f)

    def mode(name):
        p = dict(profiles[name])
        p.pop("note", None)
        return p

    jang = {
        "format": "JANG", "profile": "MXFP8",
        "source": "Raptor-8B-A1B (Lfm2MoeForCausalLM, lfm2_moe)",
        "chat": {
            # vendor default is GREEDY; stamped default pairs it with reasoning
            # on, which is the vendor's own `interactive_reasoning` profile.
            "sampling_defaults": dict(
                mode("interactive_reasoning"),
                source="raptor_generation_profiles.json:interactive_reasoning",
                mode="interactive_reasoning"),
            "sampling_modes": {k: mode(k) for k in profiles if k != "schema"},
            "diagnostic_only_modes": ["liquid_native_diagnostic_only"],
            "stop_token_ids": [IM_END],
            "context_length": {"native": cfg.get("max_position_embeddings", 128000)},
            "recommended_max_tokens": {
                "default": profiles["default"]["max_new_tokens"],
                "interactive_reasoning": profiles["interactive_reasoning"]["max_new_tokens"],
                "long_reasoning": profiles["long_reasoning"]["max_new_tokens"],
            },
        },
        "reasoning": {
            "supported": True,
            "default": "on",
            "transport": "chat_template_kwarg",
            "enable_kwarg": "enable_thinking",
            "think_in_template": True,
            "think_open_id": THINK_OPEN, "think_close_id": THINK_CLOSE,
            "note": ("Vendor `default` profile is reasoning_mode=off; stamped "
                     "default is the vendor's `interactive_reasoning` profile "
                     "(greedy + reasoning on)."),
        },
        "tools": {
            "supported": True,
            "dialect": "lfm25_native_python_literal",
            "start_id": TOOL_START, "end_id": TOOL_END,
            "contract": "lfm25_contract.py (shipped in bundle)",
            "note": ("Native calls look Python-like but are DATA — parsed by a "
                     "small AST subset, never evaluated. Do not .strip() string "
                     "argument values; a trailing newline is content."),
        },
        "vision": {"supported": False},
        "cache_subtype": "lfm2_moe_hybrid_ssm",
        "runtime": {
            "architecture": "lfm2_moe_hybrid",
            "loads_with": "stock mlx_lm >= 0.31 (mlx_lm.load), no custom code",
        },
        "version": 1,
        "weight_format": "mxfp8",
        # DICT, not a bare string: capabilities._resolve_family_str reads
        # `source_model["architecture"]` as its first family candidate and
        # calls .get() on it, so a string here is an AttributeError at stamp
        # time. This is also the documented priority-1 form.
        "source_model": {
            "name": "Raptor-8B-A1B",
            "architecture": cfg.get("model_type", "lfm2_moe"),
            "release": "RELEASE-CANDIDATE-v112-t609k",
        },
        "has_vision": False,
        "has_audio": False,
        "architecture": {
            "model_type": cfg.get("model_type"),
            "layers": cfg.get("num_hidden_layers"),
            "hybrid_layer_types": True,
            "experts": cfg.get("num_experts"),
            "experts_per_tok": cfg.get("num_experts_per_tok"),
            "dense_layers": cfg.get("num_dense_layers"),
            "tie_word_embeddings": cfg.get("tie_word_embeddings"),
        },
        "quantization": {
            "mode": "mxfp8", "group_size": 32, "bits": 8,
            "fp16_passthrough": [
                "conv.conv (depthwise short-conv kernel)",
                "feed_forward.gate (32-way MoE router)",
                "all norms",
            ],
            "calibrated": False,
            "note": ("MXFP8 is structural — no Hessian allocation, no AWQ, no "
                     "imatrix refit, by design."),
        },
    }
    # weight bytes, as the shipped LFM2.5 stamps carry
    total = sum(p.stat().st_size for p in bundle.glob("*.safetensors"))
    jang["runtime"]["total_weight_bytes"] = total
    jang["runtime"]["total_weight_gb"] = round(total / 1e9, 2)

    # Build the runtime contract through the same family table as every other
    # JANG converter. A hand-authored friendly summary caused the original
    # family-less stamp regression and must never be reintroduced.
    caps = build_capabilities(jang, cfg, bundle)
    if caps is None:
        raise SystemExit(
            "refusing to write an unloadable stamp — capabilities family "
            f"could not be derived from config model_type={cfg.get('model_type')!r}"
        )
    caps["cache_subtype"] = jang["cache_subtype"]
    caps["default_reasoning"] = "on"
    jang["capabilities"] = caps
    write_validated_jang_config(bundle, jang, cfg)

    # two-file contract: generation_config must agree with sampling_defaults
    gen = json.loads((bundle / "generation_config.json").read_text())
    d = jang["chat"]["sampling_defaults"]
    gen.update({
        "do_sample": d["do_sample"],
        "repetition_penalty": d["repetition_penalty"],
        "eos_token_id": IM_END, "bos_token_id": BOS, "pad_token_id": PAD,
    })
    for k in ("temperature", "top_p", "top_k", "min_p"):
        if d.get("do_sample") and k in d:
            gen[k] = d[k]
        else:
            gen.pop(k, None)          # greedy: sampling params must be ABSENT
    write_json_object_atomic(bundle / "generation_config.json", gen)

    print(f"  stamped {bundle.name}")
    print(f"    default mode   : {d['mode']} (do_sample={d['do_sample']}, "
          f"reasoning={d['reasoning_mode']}, max_new={d['max_new_tokens']})")
    print(f"    sampling modes : {list(jang['chat']['sampling_modes'])}")
    print(f"    reasoning      : default={jang['reasoning']['default']}")
    print(f"    tools          : {jang['tools']['dialect']} "
          f"[{TOOL_START}, {TOOL_END}]  eos={IM_END}")
    print(f"    two-file agree : do_sample={gen['do_sample']}, "
          f"sampling keys absent={not any(k in gen for k in ('temperature','top_p','top_k'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
