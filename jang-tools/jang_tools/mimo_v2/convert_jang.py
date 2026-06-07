"""MiMo-V2.5 → JANG bundle converter.

Profiles:
    JANG_2L  JANG 8/6/2 profile; routed experts use 256-expert floors
             gate=4, up=2, down=3, expert group_size=128
    JANG_4M  routed experts: 4-bit affine, everything else 8-bit affine, ViT/audio/o_proj bf16
    JANG_2K  routed experts: gate/up 2-bit, down 4-bit, everything else as above
    JANG_2S  JANG 6/4/2 profile with the same 256-expert floors, text o_proj 8-bit
    JANG_2L_322_G64  fit profile: gate/up/down = 3/2/2, expert group_size=64

Tensor classification (in priority order):

    1. *.weight_scale_inv ............. SKIP (read internally when companion weight is loaded)
    2. *norm.weight, *.bias ........... bf16 passthrough
    3. *.attention_sink_bias .......... bf16 passthrough (SWA layers + MTP layers)
    4. *.e_score_correction_bias ...... fp32 passthrough (routing precision)
    5. mlp.gate.weight (not experts) .. fp32 passthrough (256x4096 router)
    6. visual.* ....................... bf16 passthrough (entire 729M ViT)
    7. audio_encoder.* ................ bf16 passthrough (261M audio encoder)
    8. speech_embeddings.* ............ bf16 passthrough (20 channel embeddings)
    9. *.o_proj.weight ................ bf16 passthrough (49 layers, all bf16 in source)
   10. mtp.*.eh_proj.weight ........... bf16 passthrough (bf16 in source)
   11. mlp.experts.*.{gate,up,down}_proj.weight ..... profile affine, group_size 128
   12. EVERYTHING ELSE .weight ........ 8-bit affine, group_size 64
       (qkv_proj, layer-0 dense MLP, embed_tokens, lm_head, MTP qkv/mlp)

Bundle metadata invariants (set in config.json):
   - quantization.bits = 8
   - quantization.group_size = 64
   - quantization.quant_method = "affine"
   - mxtq_bits = profile bits, or a per-projection dict for mixed K profiles
   - routed_expert_bits = same value as mxtq_bits for routed experts
   - quantization[name] = {bits, group_size, mode} for non-default runtime modules
   - rope_parameters: built from rope_theta + partial_rotary_factor (back-compat with `rope_scaling`)

Usage:
    python -m jang_tools.mimo_v2.convert_jang \\
        --src /Volumes/EricsLLMDrive/jangq-ai/sources/MiMo-V2.5 \\
        --dst ~/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L \\
        --profile 2
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file as sf_save_torch

from .weight_loader import MiMoShardIndex


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


_EXPERT_PAT = re.compile(r"\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight$")
_EXPERT_RUNTIME_PAT = re.compile(
    r"^(model\.layers\.(?P<layer>\d+)\.mlp)\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)
_MIMO_NUM_EXPERTS = 256
_MIMO_EXPERT_GROUP_SIZE = 128
_MIMO_2BIT_EXPERT_FLOORS = {"gate_proj": 4, "down_proj": 3}
_PASSTHROUGH_NAME_TAILS = (
    "norm.weight",
    "post_attention_layernorm.weight",
    "input_layernorm.weight",
    "final_layernorm.weight",
    "pre_mlp_layernorm.weight",
    "enorm.weight",
    "hnorm.weight",
    ".bias",
    "attention_sink_bias",
)


def is_routed_expert_weight(name: str) -> bool:
    return _EXPERT_PAT.search(name) is not None


@dataclass(frozen=True)
class QuantProfile:
    name: str
    routed_expert_bits: int | dict[str, int]
    expert_proj_bits: dict[str, int]
    expert_layer_bits: dict[int, dict[str, int]] | None = None
    bookend_bits: int = 8
    o_proj_bits: int | None = None
    critical_bits: int = 8
    important_bits: int = 6
    compress_bits: int = 2
    expert_group_size: int = _MIMO_EXPERT_GROUP_SIZE
    default_group_size: int = 64

    @classmethod
    def parse(cls, raw: str | int) -> "QuantProfile":
        key = str(raw).strip().lower().replace("_", "").replace("-", "").replace("/", "")
        if key in {"2", "2l", "jang2l"}:
            bits = _mimo_expert_bits_for_compress(2)
            return cls("JANG_2L", bits, bits, critical_bits=8, important_bits=6, compress_bits=2)
        early_match = re.fullmatch(r"(?:jang)?2l?e(?:arly)?(\d+)", key)
        if early_match:
            end_layer = int(early_match.group(1))
            if end_layer < 1 or end_layer > 47:
                raise ValueError(f"invalid early-layer end {end_layer}; expected 1..47")
            base = {"gate_proj": 2, "up_proj": 2, "down_proj": 2}
            early = {"gate_proj": 4, "up_proj": 4, "down_proj": 4}
            return cls(
                f"JANG_2L_E{end_layer}",
                base,
                base,
                {layer: early for layer in range(1, end_layer + 1)},
                critical_bits=8,
                important_bits=6,
                compress_bits=2,
            )
        if key in {"2c4l4", "2crit4last4", "crit4last4", "jang2lc4l4"}:
            base = {"gate_proj": 2, "up_proj": 2, "down_proj": 2}
            critical = {"gate_proj": 8, "up_proj": 8, "down_proj": 8}
            late = {"gate_proj": 4, "up_proj": 4, "down_proj": 4}
            layer_bits = {
                **{layer: critical for layer in range(1, 5)},
                **{layer: late for layer in range(44, 48)},
            }
            return cls(
                "JANG_2L_C4L4",
                base,
                base,
                layer_bits,
                critical_bits=8,
                important_bits=6,
                compress_bits=2,
            )
        if key in {"2g32", "2l32", "jang2lg32"}:
            bits = _mimo_expert_bits_for_compress(2)
            return cls(
                "JANG_2L_G32",
                bits,
                bits,
                critical_bits=8,
                important_bits=6,
                compress_bits=2,
                expert_group_size=32,
            )
        if key in {"2s", "slim", "jang2s"}:
            bits = _mimo_expert_bits_for_compress(2)
            return cls(
                "JANG_2S",
                bits,
                bits,
                bookend_bits=6,
                o_proj_bits=8,
                critical_bits=6,
                important_bits=4,
                compress_bits=2,
            )
        if key in {"4", "4m", "jang4m"}:
            bits = {"gate_proj": 4, "up_proj": 4, "down_proj": 4}
            return cls("JANG_4M", 4, bits, critical_bits=8, important_bits=4, compress_bits=4)
        if key in {"3", "333", "3e", "jang3e"}:
            bits = {"gate_proj": 3, "up_proj": 3, "down_proj": 3}
            return cls("JANG_3E", bits, bits, critical_bits=8, important_bits=4, compress_bits=3)
        if key in {"322", "2fit", "fit", "jang322", "jang2fit"}:
            bits = {"gate_proj": 3, "up_proj": 2, "down_proj": 2}
            return cls("JANG_2L_322", bits, bits, critical_bits=8, important_bits=6, compress_bits=2)
        if key in {"322g64", "2fitg64", "fitg64", "jang322g64", "jang2fitg64"}:
            bits = {"gate_proj": 3, "up_proj": 2, "down_proj": 2}
            return cls(
                "JANG_2L_322_G64",
                bits,
                bits,
                critical_bits=8,
                important_bits=6,
                compress_bits=2,
                expert_group_size=64,
            )
        d3e_match = re.fullmatch(r"(?:jang)?322d(?:own)?3e(?:arly)?(\d+)", key)
        if d3e_match:
            end_layer = int(d3e_match.group(1))
            if end_layer < 1 or end_layer > 47:
                raise ValueError(f"invalid early down3 end {end_layer}; expected 1..47")
            base = {"gate_proj": 3, "up_proj": 2, "down_proj": 2}
            early = {"gate_proj": 3, "up_proj": 2, "down_proj": 3}
            return cls(
                f"JANG_2L_322_D3E{end_layer}",
                base,
                base,
                {layer: early for layer in range(1, end_layer + 1)},
                critical_bits=8,
                important_bits=6,
                compress_bits=2,
            )
        if key in {"323", "jang323"}:
            bits = {"gate_proj": 3, "up_proj": 2, "down_proj": 3}
            return cls("JANG_2L_323", bits, bits, critical_bits=8, important_bits=6, compress_bits=2)
        if key in {"233", "jang233"}:
            bits = {"gate_proj": 2, "up_proj": 3, "down_proj": 3}
            return cls("JANG_233", bits, bits, critical_bits=8, important_bits=6, compress_bits=2)
        if key in {"k", "2k", "224", "jang2k"}:
            bits = {"gate_proj": 2, "up_proj": 2, "down_proj": 4}
            return cls("JANG_2K", bits, bits, critical_bits=8, important_bits=6, compress_bits=2)
        if key in {"422", "jang422"}:
            bits = {"gate_proj": 4, "up_proj": 2, "down_proj": 2}
            return cls("JANG_422", bits, bits, critical_bits=8, important_bits=6, compress_bits=2)
        if key in {"242", "jang242"}:
            bits = {"gate_proj": 2, "up_proj": 4, "down_proj": 2}
            return cls("JANG_242", bits, bits, critical_bits=8, important_bits=6, compress_bits=2)
        if key in {"423", "jang423", "2lfloor", "floor"}:
            bits = {"gate_proj": 4, "up_proj": 2, "down_proj": 3}
            return cls("JANG_423", bits, bits, critical_bits=8, important_bits=6, compress_bits=2)
        raise ValueError(
            f"unknown MiMo quant profile {raw!r}; use 2, 322/2fit, 322g64, 322d3eN, 323, 2e4, 2c4l4, 2g32, 2s, 2k/224, 422, 242, 423, 333, or 4"
        )

    @property
    def default_bits(self) -> int:
        return self.bookend_bits

    def bits_for_expert_name(self, name: str) -> int:
        m = _EXPERT_PAT.search(name)
        if not m:
            raise ValueError(f"not a routed expert weight: {name}")
        lm = _EXPERT_RUNTIME_PAT.match(name)
        if lm and self.expert_layer_bits:
            layer_bits = self.expert_layer_bits.get(int(lm.group("layer")))
            if layer_bits is not None:
                return layer_bits[m.group(1)]
        return self.expert_proj_bits[m.group(1)]


def _mimo_expert_bits_for_compress(compress_bits: int) -> dict[str, int]:
    """Apply current JANG 256-expert MLP floors to MiMo routed experts."""
    bits = {
        "gate_proj": compress_bits,
        "up_proj": compress_bits,
        "down_proj": compress_bits,
    }
    if _MIMO_NUM_EXPERTS >= 256:
        for proj, floor in _MIMO_2BIT_EXPERT_FLOORS.items():
            bits[proj] = max(bits[proj], floor)
    return bits


def runtime_quant_base_for_weight(name: str) -> str:
    """Return the MLX module path that owns a converted affine weight."""
    m = _EXPERT_RUNTIME_PAT.match(name)
    if m:
        return f"{m.group(1)}.switch_mlp.{m.group('proj')}"
    return name[: -len(".weight")] if name.endswith(".weight") else name


def classify(name: str, profile_bits: QuantProfile | int | str) -> tuple[int, str, int]:
    """Return (bits, method, group_size). bits=0 + method='passthrough_bf16'/'passthrough_fp32' = no quant."""
    if not isinstance(profile_bits, QuantProfile):
        profile_bits = QuantProfile.parse(profile_bits)

    if name.endswith(".weight_scale_inv"):
        return 0, "skip", 0

    # fp32 passthrough: router weights + per-expert routing bias correction.
    if name.endswith(".e_score_correction_bias"):
        return 32, "passthrough_fp32", 0
    if name.endswith(".mlp.gate.weight") and ".experts." not in name:
        return 32, "passthrough_fp32", 0

    # bf16 passthrough: norms, biases, sink biases.
    for tail in _PASSTHROUGH_NAME_TAILS:
        if name.endswith(tail):
            return 16, "passthrough_bf16", 0

    # bf16 passthrough: multimodal towers (ViT, audio encoder, speech embeddings).
    if name.startswith("visual.") or name.startswith("audio_encoder.") or name.startswith("speech_embeddings."):
        return 16, "passthrough_bf16", 0

    # MiMo decode pays lm_head every token and embed_tokens is part of the text
    # runtime contract. Keep both as affine bookends so rebuilt bundles do not
    # rely on post-load runtime quantization to repair missing sidecars.
    if name in {"model.embed_tokens.weight", "lm_head.weight"}:
        return profile_bits.default_bits, "affine", profile_bits.default_group_size

    # bf16 passthrough: MTP eh_proj (bf16 in source).
    if name.endswith(".eh_proj.weight"):
        return 16, "passthrough_bf16", 0
    # Text o_proj is bf16 in source ignored_layers. Slim profiles can quantize it
    # explicitly; default profiles keep it passthrough.
    if name.endswith(".o_proj.weight"):
        if profile_bits.o_proj_bits is not None and name.startswith("model.layers."):
            return profile_bits.o_proj_bits, "affine", profile_bits.default_group_size
        return 16, "passthrough_bf16", 0

    # Routed experts → profile_bits affine.
    if is_routed_expert_weight(name):
        return profile_bits.bits_for_expert_name(name), "affine", profile_bits.expert_group_size

    # Everything else (qkv_proj, layer-0 dense MLP, embed, lm_head, MTP qkv/mlp)
    # uses the profile's bookend bit width.
    if name.endswith(".weight"):
        return profile_bits.default_bits, "affine", profile_bits.default_group_size

    # Unknown — passthrough bf16 to be safe.
    return 16, "passthrough_bf16", 0


# --------------------------------------------------------------------------
# Bundle metadata
# --------------------------------------------------------------------------


def _normalize_rope(cfg: dict[str, Any]) -> None:
    """Mirror legacy rope_scaling into transformers 4.50+ rope_parameters."""
    rs = cfg.get("rope_scaling")
    if rs is None:
        rs = {"rope_type": "default", "type": "default"}
    rp = dict(rs)
    if "type" in rp:
        rp["rope_type"] = rp.pop("type")
    if "rope_theta" not in rp:
        rp["rope_theta"] = float(cfg.get("rope_theta", 10000))
    if "partial_rotary_factor" not in rp:
        rp["partial_rotary_factor"] = float(cfg.get("partial_rotary_factor", 1.0))
    for k in ("beta_fast", "beta_slow", "factor"):
        if k in rp:
            rp[k] = float(rp[k])
    cfg["rope_parameters"] = rp


def _write_config_json(
    src: Path,
    dst: Path,
    profile: QuantProfile,
    quant_overrides: dict[str, dict],
    include_mtp: bool = True,
) -> None:
    cfg = json.loads((src / "config.json").read_text())
    cfg.pop("quantization_config", None)
    tokenizer_config = json.loads((src / "tokenizer_config.json").read_text())
    if tokenizer_config.get("chat_template"):
        cfg["chat_template"] = tokenizer_config["chat_template"]
    _normalize_rope(cfg)
    # mlx-lm load_model expects per-tensor overrides AT THE TOP LEVEL of `quantization`,
    # keyed by module path. The `class_predicate` does `config["quantization"][p]`.
    # Nesting under `overrides` makes mlx-lm fall back to default bits → shape mismatch.
    quant_dict: dict[str, Any] = {
        "bits": profile.default_bits,
        "group_size": profile.default_group_size,
        "quant_method": "affine",
        "mode": "affine",
    }
    for path, spec in quant_overrides.items():
        # Inline {bits, group_size}; only carry mode if non-default.
        entry = {"bits": spec["bits"], "group_size": spec["group_size"]}
        if spec.get("mode") and spec["mode"] != "affine":
            entry["mode"] = spec["mode"]
        quant_dict[path] = entry
    cfg["quantization"] = quant_dict
    cfg["mxtq_bits"] = profile.routed_expert_bits
    cfg["routed_expert_bits"] = profile.routed_expert_bits
    cfg["jang_profile"] = profile.name
    cfg["jang_tier_bits"] = {
        "critical": profile.critical_bits,
        "important": profile.important_bits,
        "compress": profile.compress_bits,
    }
    cfg["jang_expert_group_size"] = profile.expert_group_size
    cfg["routed_expert_group_size"] = profile.expert_group_size
    if profile.expert_layer_bits:
        cfg["routed_expert_bit_plan"] = {
            "default": profile.expert_proj_bits,
            "layer_overrides": {
                str(layer): bits for layer, bits in sorted(profile.expert_layer_bits.items())
            },
        }
    cfg["jang_version"] = "v2"
    cfg["capabilities"] = {
        "family": "mimo_v2",
        "modalities": ["text", "vision", "audio"],
        "cache_type": "kv",
        "attention": {
            "full": True,
            "sliding_window": True,
            "sliding_window_size": cfg.get("sliding_window"),
        },
        "reasoning": {"supported": True, "default": True, "parser": "think_xml"},
        "tools": {"supported": True, "parser": "xml_function"},
    }
    cfg["runtime"] = {
        "cache_type": "kv",
        "attention_impl": "hybrid_full_swa_sink",
        "cache_topology": {
            "family": "hybrid_full_swa_kv",
            "prefix_cache": True,
            "l2_disk_cache": True,
            "turboquant_kv": "full_attention_layers_only",
            "swa_layers": "rotating_kv_native",
        },
        "mtp_mode": "preserved_disabled" if include_mtp else "absent",
        "bundle_has_mtp": include_mtp,
        "multimodal_mode": "weights_preserved_text_runtime",
        "quantization_profile": profile.name,
        "routed_expert_bits": profile.routed_expert_bits,
        "routed_expert_group_size": profile.expert_group_size,
        "routed_expert_bit_plan": cfg.get("routed_expert_bit_plan"),
    }
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))


def write_jang_metadata(
    src: Path,
    dst: Path,
    profile: QuantProfile,
    quant_overrides: dict[str, dict],
    include_mtp: bool = False,
) -> None:
    """Write the classic affine MiMo JANG metadata pair.

    ``config.json`` is consumed by MLX/vMLX loaders. ``jang_config.json`` is a
    small explicit contract for release/verifier tooling so classic affine JANG
    bundles are not confused with JANGTQ prestacked expert bundles.
    """
    _write_config_json(src, dst, profile, quant_overrides, include_mtp=include_mtp)
    jang_cfg = {
        "format": "jang",
        "family": "mimo_v2",
        "profile": profile.name,
        "mxtq_bits": profile.routed_expert_bits,
        "routed_expert_bits": profile.routed_expert_bits,
        "routed_expert_group_size": profile.expert_group_size,
        "num_experts": _MIMO_NUM_EXPERTS,
        "expert_layout": "per_expert_affine",
        "runtime_expert_module": "switch_mlp",
        "bookend_bits": profile.default_bits,
        "bookend_group_size": profile.default_group_size,
        "mtp_mode": "preserved_disabled" if include_mtp else "absent",
        "bundle_has_mtp": include_mtp,
    }
    if profile.expert_layer_bits:
        jang_cfg["routed_expert_bit_plan"] = {
            "default": profile.expert_proj_bits,
            "layer_overrides": {
                str(layer): bits for layer, bits in sorted(profile.expert_layer_bits.items())
            },
        }
    (dst / "jang_config.json").write_text(json.dumps(jang_cfg, indent=2))


def _copy_aux_files(src: Path, dst: Path) -> None:
    """Copy tokenizer + chat + preprocessor + custom modeling code + assets."""
    static_files = [
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
        "preprocessor_config.json",
        "configuration_mimo_v2.py",
        "modeling_mimo_v2.py",
        "README.md",
        ".gitattributes",
    ]
    for fn in static_files:
        s = src / fn
        if s.exists():
            shutil.copy2(s, dst / fn)
    # audio_tokenizer/ and assets/ — copy directories whole.
    for sub in ("audio_tokenizer", "assets"):
        if (src / sub).is_dir():
            shutil.copytree(src / sub, dst / sub, dirs_exist_ok=True)
    # Extract chat template to standalone .jinja for visibility (does not override
    # tokenizer_config's embedded copy — that one is canonical for HF loaders).
    tc = json.loads((src / "tokenizer_config.json").read_text())
    if "chat_template" in tc and tc["chat_template"]:
        (dst / "chat_template.jinja").write_text(tc["chat_template"])


# --------------------------------------------------------------------------
# Conversion loop
# --------------------------------------------------------------------------


def convert(
    src: Path,
    dst: Path,
    profile_bits: str | int,
    max_shard_bytes: int = 1_000_000_000,
    include_mtp: bool = False,
) -> None:
    import mlx.core as mx

    profile = QuantProfile.parse(profile_bits)
    idx = MiMoShardIndex(src)
    weight_keys = idx.weight_keys
    required_tensor_names = set()
    for name in weight_keys:
        if not include_mtp and name.startswith("model.mtp."):
            continue
        required_tensor_names.add(name)
        if idx.is_fp8_weight(name):
            required_tensor_names.add(name[: -len(".weight")] + ".weight_scale_inv")
    missing_shards = sorted(
        {
            idx.weight_map[name]
            for name in required_tensor_names
            if not (idx.src / idx.weight_map[name]).exists()
        }
    )
    if missing_shards:
        sample = ", ".join(missing_shards[:5])
        more = "" if len(missing_shards) <= 5 else f", ... +{len(missing_shards) - 5} more"
        raise FileNotFoundError(
            f"MiMo source checkpoint is incomplete: missing {len(missing_shards)} shard file(s): "
            f"{sample}{more}"
        )
    dst.mkdir(parents=True, exist_ok=True)

    print(f"[convert] source: {src}")
    print(f"[convert] target: {dst}")
    print(f"[convert] profile: {profile.name} (routed_experts={profile.routed_expert_bits}, "
          f"tiers={profile.critical_bits}/{profile.important_bits}/{profile.compress_bits}, "
          f"bookend={profile.default_bits}-bit/group{profile.default_group_size}, "
          f"experts_group={profile.expert_group_size})")
    print(f"[convert] MTP tensors: {'preserve' if include_mtp else 'drop'}")
    print(f"[convert] {len(weight_keys)} logical tensors", flush=True)

    shard_idx = 1
    shard_bytes = 0
    shard_buf: dict[str, torch.Tensor] = {}
    shard_map: dict[str, str] = {}
    quant_overrides: dict[str, dict] = {}
    method_totals: dict[str, int] = {"affine": 0, "passthrough_bf16": 0, "passthrough_fp32": 0}
    bit_totals: dict[int, int] = {}
    state_path = dst / ".convert_state.json"
    start_i = 0
    if state_path.exists():
        state = json.loads(state_path.read_text())
        start_i = int(state["next_i"])
        shard_idx = int(state["shard_idx"])
        shard_map = dict(state["shard_map"])
        quant_overrides = dict(state["quant_overrides"])
        method_totals = {k: int(v) for k, v in state["method_totals"].items()}
        bit_totals = {int(k): int(v) for k, v in state["bit_totals"].items()}
        for stale in dst.glob("model-*-of-XXXXX.safetensors"):
            try:
                stale_idx = int(stale.name.split("-")[1])
            except (IndexError, ValueError):
                continue
            if stale_idx >= shard_idx:
                stale.unlink()
        print(f"[convert] resume checkpoint: next tensor {start_i + 1}, next shard {shard_idx}", flush=True)
    t_start = time.time()

    def flush_shard() -> None:
        nonlocal shard_idx, shard_bytes, shard_buf
        if not shard_buf:
            return
        shard_name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        sf_save_torch(shard_buf, str(dst / shard_name))
        for k in shard_buf:
            shard_map[k] = shard_name
        elapsed = time.time() - t_start
        print(f"    shard {shard_idx}: {len(shard_buf)} tensors, "
              f"{shard_bytes / 1e9:.2f} GB  (elapsed {elapsed:.0f}s)", flush=True)
        shard_buf = {}
        shard_bytes = 0
        shard_idx += 1
        gc.collect()

    def add_tensor(name: str, t: torch.Tensor) -> None:
        nonlocal shard_bytes
        # Ensure contiguous + cpu before save.
        if not t.is_contiguous():
            t = t.contiguous()
        shard_buf[name] = t.cpu()
        shard_bytes += t.element_size() * t.numel()
        if shard_bytes >= max_shard_bytes:
            flush_shard()

    def checkpoint(next_i: int) -> None:
        flush_shard()
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "next_i": next_i,
            "shard_idx": shard_idx,
            "shard_map": shard_map,
            "quant_overrides": quant_overrides,
            "method_totals": method_totals,
            "bit_totals": bit_totals,
        }))
        tmp.replace(state_path)

    DEFAULT_BITS = profile.default_bits
    DEFAULT_GROUP = profile.default_group_size

    def _mx_to_torch(arr_mx, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Convert mx.array → torch.Tensor without going through numpy when possible."""
        t = torch.from_numpy(np.array(arr_mx))
        if dtype is not None:
            t = t.to(dtype)
        return t

    def _quantize_affine_cpu_u32(t: torch.Tensor, *, group_size: int, bits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """CPU affine packer for MLX-compatible uint32 quantized weights.

        MLX's runtime kernels only require packed values plus affine
        scales/biases; they do not require that the pack was produced by
        ``mx.quantize``. Use explicit min/max affine for routed experts so the
        built bundle matches the source-side quantization probe. This also
        avoids repeated low-bit Metal quantize work during conversion.
        """
        if bits not in {2, 3, 4, 5, 6, 8}:
            raise ValueError(f"unsupported affine bits={bits}")
        x = t.detach().cpu().float().numpy()
        if x.ndim != 2:
            raise ValueError(f"expected 2D weight for CPU affine pack, got shape {x.shape}")
        rows, cols = x.shape
        if cols % group_size != 0:
            raise ValueError(f"weight cols {cols} not divisible by group_size {group_size}")
        groups = cols // group_size
        if (group_size * bits) % 32 != 0:
            raise ValueError(f"group_size={group_size} and bits={bits} do not pack into whole uint32 words")
        words_per_group = (group_size * bits) // 32
        xr = x.reshape(rows, groups, group_size)
        maxv = xr.max(axis=2).astype(np.float32)
        minv = xr.min(axis=2).astype(np.float32)
        scale = ((minv - maxv) / float((1 << bits) - 1)).astype(np.float32)
        scale = np.minimum(scale, np.float32(-1e-7))
        bias = maxv.astype(np.float32)
        q = np.rint((xr - bias[:, :, None]) / scale[:, :, None])
        q = np.clip(q, 0, (1 << bits) - 1).astype(np.uint32)
        packed_words = np.zeros((rows, groups, words_per_group), dtype=np.uint32)
        mask = np.uint32((1 << bits) - 1)
        for i_col in range(group_size):
            bit_offset = i_col * bits
            word_idx = bit_offset // 32
            shift = bit_offset % 32
            vals = q[:, :, i_col] & mask
            packed_words[:, :, word_idx] |= vals << np.uint32(shift)
            spill = shift + bits - 32
            if spill > 0:
                packed_words[:, :, word_idx + 1] |= vals >> np.uint32(bits - spill)
        packed = packed_words.reshape(rows, groups * words_per_group)
        return (
            torch.from_numpy(packed),
            torch.from_numpy(scale),
            torch.from_numpy(bias),
        )

    for i, name in enumerate(weight_keys):
        if i < start_i:
            continue
        if not include_mtp and name.startswith("model.mtp."):
            continue
        bits, method, group_size = classify(name, profile)

        if method == "skip":
            continue
        if method == "passthrough_bf16":
            t = idx.read_passthrough(name).to(torch.bfloat16)
            add_tensor(name, t)
            method_totals["passthrough_bf16"] += 1
        elif method == "passthrough_fp32":
            t = idx.read_passthrough(name, out_dtype=torch.float32)
            add_tensor(name, t)
            method_totals["passthrough_fp32"] += 1
        elif method == "affine":
            t = idx.read_tensor(name, out_dtype=torch.float32)
            use_cpu_affine = group_size == profile.expert_group_size and ".mlp.experts." in name
            if use_cpu_affine:
                qw_t, qs_t, qb_t = _quantize_affine_cpu_u32(t, group_size=group_size, bits=bits)
            else:
                w = mx.array(t.numpy())
                qw, qs, qb = mx.quantize(w, group_size=group_size, bits=bits)
                qw_t = _mx_to_torch(qw)
                qs_t = _mx_to_torch(qs, torch.float16)
                qb_t = _mx_to_torch(qb, torch.float16)
                del w, qw, qs, qb
            base = name[: -len(".weight")] if name.endswith(".weight") else name
            # mx.quantize returns uint32 packed weights and float sidecars.
            # Store sidecars as f16, matching the other JANG converters. bf16
            # loses too much mantissa for 2-bit affine expert groups.
            add_tensor(f"{base}.weight", qw_t)
            add_tensor(f"{base}.scales", qs_t.to(torch.float16))
            add_tensor(f"{base}.biases", qb_t.to(torch.float16))
            if bits != DEFAULT_BITS or group_size != DEFAULT_GROUP:
                runtime_base = runtime_quant_base_for_weight(name)
                quant_overrides[runtime_base] = {"bits": bits, "group_size": group_size, "mode": "affine"}
            bit_totals[bits] = bit_totals.get(bits, 0) + 1
            method_totals["affine"] += 1
            del t, qw_t, qs_t, qb_t
            try:
                mx.clear_cache()
            except Exception:
                pass
        else:
            raise RuntimeError(f"unknown classification method {method!r} for {name}")

        if (i + 1) % 250 == 0:
            elapsed = time.time() - t_start
            done_pct = 100 * (i + 1) / len(weight_keys)
            rate = (i + 1) / max(elapsed, 1e-3)
            eta = (len(weight_keys) - (i + 1)) / max(rate, 1e-3)
            print(
                f"    [{i+1:6d}/{len(weight_keys)}] {done_pct:.1f}%  "
                f"affine={method_totals['affine']} bf16={method_totals['passthrough_bf16']} "
                f"fp32={method_totals['passthrough_fp32']}  "
                f"({elapsed:.0f}s elapsed, ~{eta:.0f}s left)",
                flush=True,
            )
            checkpoint(i + 1)

    flush_shard()

    # Rename shards to final NNNNN-of-NNNNN form.
    total_shards = shard_idx - 1
    for k in range(1, shard_idx):
        old = dst / f"model-{k:05d}-of-XXXXX.safetensors"
        new = dst / f"model-{k:05d}-of-{total_shards:05d}.safetensors"
        if old.exists():
            old.rename(new)
    final_map = {k: v.replace("XXXXX", f"{total_shards:05d}") for k, v in shard_map.items()}
    total_bytes = sum((dst / fn).stat().st_size for fn in set(final_map.values()))
    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": final_map}, indent=2)
    )

    _copy_aux_files(src, dst)
    write_jang_metadata(src, dst, profile, quant_overrides, include_mtp=include_mtp)
    state_path.unlink(missing_ok=True)

    elapsed = time.time() - t_start
    print()
    print(f"[convert] DONE in {elapsed:.0f}s")
    print(f"[convert] tensors: affine={method_totals['affine']} "
          f"bf16-pt={method_totals['passthrough_bf16']} "
          f"fp32-pt={method_totals['passthrough_fp32']}")
    print(f"[convert] bit distribution (affine only): "
          + ", ".join(f"{b}b={c}" for b, c in sorted(bit_totals.items())))
    print(f"[convert] {total_shards} shards, total {total_bytes / 1e9:.2f} GB")
    print(f"[convert] quant_overrides: {len(quant_overrides)} non-default classifications")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert MiMo-V2.5 source checkpoint to JANG bundle.")
    p.add_argument("--src", required=True, type=Path, help="Source HF checkpoint dir.")
    p.add_argument("--dst", required=True, type=Path, help="Output JANG bundle dir.")
    p.add_argument("--profile", required=True,
                   help="Quant profile: 2, 322/2fit, 322g64, 322d3eN, 323, 2e4, 2c4l4, 2g32, 2s, 2k/224, 422, 242, 423, 333, or 4.")
    p.add_argument("--max-shard-bytes", type=int, default=1_000_000_000,
                   help="Max bytes per output shard (default 1 GB).")
    p.add_argument("--drop-mtp", action="store_true",
                   help="Do not include model.mtp.* speculative decoding tensors. This is the default.")
    p.add_argument("--include-mtp", action="store_true",
                   help="Preserve model.mtp.* tensors as disabled/opaque weights.")
    args = p.parse_args(argv)

    convert(
        args.src.expanduser(),
        args.dst.expanduser(),
        args.profile,
        args.max_shard_bytes,
        include_mtp=bool(args.include_mtp and not args.drop_mtp),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
