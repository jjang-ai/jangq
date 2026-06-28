"""MiMo-V2.5 → JANG bundle converter.

Profiles:
    JANG_2S  sub-105GB target: routed experts gate=4/up=2/down=3 affine,
             qkv + layer-0 dense 6-bit affine, o_proj 8-bit affine,
             token I/O + ViT/audio bf16
    JANG_2C  coherence-leaning sub-105GB target: routed experts 4/3/3,
             qkv + layer-0 dense 6-bit affine, o_proj 8-bit affine,
             token I/O + ViT/audio bf16
    JANG_2X  tighter-size sub-105GB target: routed experts 3/2/3,
             qkv 5-bit + layer-0 dense 6-bit affine, o_proj 8-bit affine,
             token I/O + ViT/audio bf16
    JANG_2F  comfortable sub-105GB target: routed experts 2/2/2,
             text affine bookends/o_proj/qkv/layer0 dense 4-bit,
             ViT/audio bf16
    JANG_2Q  quality-leaning comfortable sub-105GB target: routed experts 2/2/2,
             default text affine/o_proj 4-bit, qkv + layer0 dense 6-bit,
             ViT/audio bf16
    JANG_2L  routed experts: 2-bit affine, everything else 8-bit affine, ViT/audio/o_proj bf16
    JANG_4M  routed experts: 4-bit affine, everything else 8-bit affine, ViT/audio/o_proj bf16
    JANG_2K  routed experts: gate/up 2-bit, down 4-bit, everything else as above

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
   11. mlp.experts.*.{gate,up,down}_proj.weight ..... `profile_bits` affine, group_size 64
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
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file as sf_save_torch

from .affine_codec import quantize_minmax_affine
from .weight_loader import MiMoShardIndex


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


_EXPERT_PAT = re.compile(r"\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight$")
_EXPERT_RUNTIME_PAT = re.compile(
    r"^(model\.layers\.(?P<layer>\d+)\.mlp)\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)
_EXPERT_ANY_PAT = re.compile(
    r"^(model\.layers\.(?P<layer>\d+)\.mlp)\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<suffix>weight|scales|biases)$"
)
_ROUTER_ROW_PAT = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.gate\.(weight|e_score_correction_bias)$"
)
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
class ExpertKeepMap:
    layers: dict[int, list[int]]

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("expert keep map must contain at least one layer")
        counts = {len(v) for v in self.layers.values()}
        if len(counts) != 1:
            raise ValueError(f"expert keep map must use one global keep count, got {sorted(counts)}")
        for layer, experts in self.layers.items():
            if len(set(experts)) != len(experts):
                raise ValueError(f"layer {layer}: duplicate experts in keep map")
            bad = [expert for expert in experts if expert < 0]
            if bad:
                raise ValueError(f"layer {layer}: negative expert ids {bad}")

    @property
    def keep_experts(self) -> int:
        return len(next(iter(self.layers.values())))

    def indices_for_layer(self, layer: int) -> list[int]:
        try:
            return self.layers[int(layer)]
        except KeyError as exc:
            raise KeyError(f"expert keep map missing layer {layer}") from exc

    def remap(self, layer: int, expert: int) -> int | None:
        experts = self.indices_for_layer(layer)
        try:
            return experts.index(int(expert))
        except ValueError:
            return None

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": "mimo-v2-expert-keep-map-v1",
            "keep_experts": self.keep_experts,
            "layers": {str(layer): experts for layer, experts in sorted(self.layers.items())},
        }


def remap_expert_tensor_name(name: str, expert_keep_map: ExpertKeepMap | None) -> str | None:
    if expert_keep_map is None:
        return name
    m = _EXPERT_ANY_PAT.match(name)
    if not m:
        return name
    layer = int(m.group("layer"))
    new_expert = expert_keep_map.remap(layer, int(m.group("expert")))
    if new_expert is None:
        return None
    return (
        f"{m.group(1)}.experts.{new_expert}."
        f"{m.group('proj')}.{m.group('suffix')}"
    )


def slice_router_tensor(name: str, tensor: torch.Tensor, expert_keep_map: ExpertKeepMap | None) -> torch.Tensor:
    if expert_keep_map is None:
        return tensor
    m = _ROUTER_ROW_PAT.match(name)
    if not m:
        return tensor
    keep = torch.tensor(expert_keep_map.indices_for_layer(int(m.group("layer"))), dtype=torch.long)
    return tensor.index_select(0, keep)


def load_expert_keep_map(
    path: Path,
    *,
    keep_experts: int,
    num_layers: int = 48,
    num_experts: int = 256,
) -> ExpertKeepMap:
    data = json.loads(path.read_text(encoding="utf-8"))
    layers_data = data.get("layers")
    if not isinstance(layers_data, dict):
        raise ValueError(f"expert keep map missing layers dict: {path}")

    layers: dict[int, list[int]] = {}
    for layer_idx in range(1, num_layers):
        raw = layers_data.get(str(layer_idx))
        if not isinstance(raw, dict):
            raise ValueError(f"expert keep map missing layer {layer_idx}")
        if isinstance(raw.get("keep"), list):
            ranked = raw["keep"]
        elif isinstance(raw.get("prob_rank"), list):
            ranked = raw["prob_rank"]
        elif isinstance(raw.get("experts"), list):
            ranked = raw["experts"]
        else:
            raise ValueError(f"expert keep map layer {layer_idx} has no keep/prob_rank/experts list")
        selected: list[int] = []
        seen: set[int] = set()
        for value in ranked:
            expert = int(value)
            if expert < 0 or expert >= num_experts:
                raise ValueError(f"expert keep map layer {layer_idx}: expert id {expert} outside 0..{num_experts - 1}")
            if expert not in seen:
                selected.append(expert)
                seen.add(expert)
            if len(selected) >= keep_experts:
                break
        for expert in range(num_experts):
            if len(selected) >= keep_experts:
                break
            if expert not in seen:
                selected.append(expert)
                seen.add(expert)
        if len(selected) != keep_experts:
            raise ValueError(f"expert keep map layer {layer_idx} produced {len(selected)} experts; need {keep_experts}")
        layers[layer_idx] = selected
    return ExpertKeepMap(layers)


@dataclass(frozen=True)
class QuantProfile:
    name: str
    routed_expert_bits: int | dict[str, int]
    expert_proj_bits: dict[str, int]
    bookend_bits: int = 8
    qkv_bits: int = 8
    layer0_dense_bits: int = 8
    o_proj_bits: int | None = None
    token_io_bf16: bool = False
    non_expert_text_bf16: bool = False
    expert_group_size: int = 64
    expert_layer_bits: dict[int, dict[str, int]] = field(default_factory=dict)

    @classmethod
    def parse(cls, raw: str | int) -> "QuantProfile":
        key = str(raw).strip().lower().replace("_", "").replace("-", "").replace("/", "")
        if key in {"1", "1l", "jang1l", "2s", "jang2s"}:
            bits = {"gate_proj": 4, "up_proj": 2, "down_proj": 3}
            return cls(
                "JANG_2S",
                bits,
                bits,
                qkv_bits=6,
                layer0_dense_bits=6,
                o_proj_bits=8,
                token_io_bf16=True,
            )
        if key in {"2c", "jang2c", "coh", "coherence"}:
            bits = {"gate_proj": 4, "up_proj": 3, "down_proj": 3}
            return cls(
                "JANG_2C",
                bits,
                bits,
                qkv_bits=6,
                layer0_dense_bits=6,
                o_proj_bits=8,
                token_io_bf16=True,
            )
        if key in {"2x", "jang2x", "xs", "tight"}:
            bits = {"gate_proj": 3, "up_proj": 2, "down_proj": 3}
            return cls(
                "JANG_2X",
                bits,
                bits,
                qkv_bits=5,
                layer0_dense_bits=6,
                o_proj_bits=8,
                token_io_bf16=True,
            )
        if key in {"2f", "jang2f", "floor", "comfortable"}:
            bits = {"gate_proj": 2, "up_proj": 2, "down_proj": 2}
            return cls(
                "JANG_2F",
                bits,
                bits,
                bookend_bits=4,
                qkv_bits=4,
                layer0_dense_bits=4,
                o_proj_bits=4,
            )
        if key in {"2q", "jang2q", "quality"}:
            bits = {"gate_proj": 2, "up_proj": 2, "down_proj": 2}
            return cls(
                "JANG_2Q",
                bits,
                bits,
                bookend_bits=4,
                qkv_bits=6,
                layer0_dense_bits=6,
                o_proj_bits=4,
            )
        m = re.fullmatch(
            r"([2348][2348][2348])g(32|64|128)l([1-9]|1[0-6])x8"
            r"(?:b([568]))?(?:q([4568]))?(t16|n16)?",
            key,
        )
        if m:
            digits = m.group(1)
            group_size = int(m.group(2))
            late_count = int(m.group(3))
            bookend_bits = int(m.group(4) or 8)
            qkv_bits = int(m.group(5) or bookend_bits)
            bf16_suffix = m.group(6)
            token_io_bf16 = bf16_suffix == "t16"
            non_expert_text_bf16 = bf16_suffix == "n16"
            bits = {
                "gate_proj": int(digits[0]),
                "up_proj": int(digits[1]),
                "down_proj": int(digits[2]),
            }
            late = {"gate_proj": 8, "up_proj": 8, "down_proj": 8}
            return cls(
                f"JANG_{digits.upper()}G{group_size}L{late_count}X8B{bookend_bits}Q{qkv_bits}"
                + ("T16" if token_io_bf16 else "")
                + ("N16" if non_expert_text_bf16 else ""),
                bits,
                bits,
                bookend_bits=bookend_bits,
                qkv_bits=qkv_bits,
                layer0_dense_bits=bookend_bits,
                o_proj_bits=None,
                token_io_bf16=token_io_bf16,
                non_expert_text_bf16=non_expert_text_bf16,
                expert_group_size=group_size,
                expert_layer_bits={layer: late for layer in range(48 - late_count, 48)},
            )
        m = re.fullmatch(
            r"([2348][2348][2348])g(32|64|128)e([1-9]|1[0-6])x8"
            r"(?:b([568]))?(?:q([4568]))?(t16|n16)?",
            key,
        )
        if m:
            digits = m.group(1)
            group_size = int(m.group(2))
            early_count = int(m.group(3))
            bookend_bits = int(m.group(4) or 8)
            qkv_bits = int(m.group(5) or bookend_bits)
            bf16_suffix = m.group(6)
            token_io_bf16 = bf16_suffix == "t16"
            non_expert_text_bf16 = bf16_suffix == "n16"
            bits = {
                "gate_proj": int(digits[0]),
                "up_proj": int(digits[1]),
                "down_proj": int(digits[2]),
            }
            early = {"gate_proj": 8, "up_proj": 8, "down_proj": 8}
            return cls(
                f"JANG_{digits.upper()}G{group_size}E{early_count}X8B{bookend_bits}Q{qkv_bits}"
                + ("T16" if token_io_bf16 else "")
                + ("N16" if non_expert_text_bf16 else ""),
                bits,
                bits,
                bookend_bits=bookend_bits,
                qkv_bits=qkv_bits,
                layer0_dense_bits=bookend_bits,
                o_proj_bits=None,
                token_io_bf16=token_io_bf16,
                non_expert_text_bf16=non_expert_text_bf16,
                expert_group_size=group_size,
                expert_layer_bits={layer: early for layer in range(1, early_count + 1)},
            )
        m = re.fullmatch(r"([2348][2348][2348])(?:g(32|64|128))?(?:b([568]))?(?:q([4568]))?(t16|n16)?", key)
        if m:
            digits = m.group(1)
            group_size = int(m.group(2) or 128)
            bookend_bits = int(m.group(3) or 8)
            qkv_bits = int(m.group(4) or bookend_bits)
            bf16_suffix = m.group(5)
            token_io_bf16 = bf16_suffix == "t16"
            non_expert_text_bf16 = bf16_suffix == "n16"
            bits = {
                "gate_proj": int(digits[0]),
                "up_proj": int(digits[1]),
                "down_proj": int(digits[2]),
            }
            return cls(
                f"JANG_{digits.upper()}G{group_size}B{bookend_bits}Q{qkv_bits}"
                + ("T16" if token_io_bf16 else "")
                + ("N16" if non_expert_text_bf16 else ""),
                bits,
                bits,
                bookend_bits=bookend_bits,
                qkv_bits=qkv_bits,
                layer0_dense_bits=bookend_bits,
                o_proj_bits=None,
                token_io_bf16=token_io_bf16,
                non_expert_text_bf16=non_expert_text_bf16,
                expert_group_size=group_size,
            )
        m = re.fullmatch(r"(?:slim)?322d3e([1-9]|1[0-6])(?:b([568]))?(?:q([4568]))?", key)
        if m:
            end_layer = int(m.group(1))
            bookend_bits = int(m.group(2) or 4)
            qkv_bits = int(m.group(3) or 6)
            base = {"gate_proj": 3, "up_proj": 2, "down_proj": 2}
            early = {"gate_proj": 3, "up_proj": 2, "down_proj": 3}
            return cls(
                f"JANG_2R_322D3E{end_layer}"
                + (f"B{bookend_bits}" if bookend_bits != 4 else "")
                + (f"Q{qkv_bits}" if qkv_bits != 6 else ""),
                base,
                base,
                bookend_bits=bookend_bits,
                qkv_bits=qkv_bits,
                layer0_dense_bits=6,
                o_proj_bits=4,
                expert_group_size=128,
                expert_layer_bits={layer: early for layer in range(1, end_layer + 1)},
            )
        m = re.fullmatch(r"(?:slim)?333e([1-9]|1[0-6])(?:b([568]))?(?:q([4568]))?", key)
        if m:
            end_layer = int(m.group(1))
            bookend_bits = int(m.group(2) or 4)
            qkv_bits = int(m.group(3) or 6)
            base = {"gate_proj": 3, "up_proj": 2, "down_proj": 2}
            early = {"gate_proj": 3, "up_proj": 3, "down_proj": 3}
            return cls(
                f"JANG_2R_333E{end_layer}"
                + (f"B{bookend_bits}" if bookend_bits != 4 else "")
                + (f"Q{qkv_bits}" if qkv_bits != 6 else ""),
                base,
                base,
                bookend_bits=bookend_bits,
                qkv_bits=qkv_bits,
                layer0_dense_bits=6,
                o_proj_bits=4,
                expert_group_size=128,
                expert_layer_bits={layer: early for layer in range(1, end_layer + 1)},
            )
        if key in {"2", "2l", "jang2l"}:
            return cls("JANG_2L", 2, {"gate_proj": 2, "up_proj": 2, "down_proj": 2})
        if key in {"4", "4m", "jang4m"}:
            return cls("JANG_4M", 4, {"gate_proj": 4, "up_proj": 4, "down_proj": 4})
        if key in {"k", "2k", "422", "242", "jang2k"}:
            bits = {"gate_proj": 2, "up_proj": 2, "down_proj": 4}
            return cls("JANG_2K", bits, bits)
        raise ValueError(
            f"unknown MiMo quant profile {raw!r}; use 2s, 2c, 2x, 2q, 2f, "
            "projection profiles like 444g128/448g128 with optional t16/n16, "
            "slim322d3eN, 2, 4, or 2k"
        )

    @property
    def default_bits(self) -> int:
        return self.bookend_bits

    def bits_for_expert_name(self, name: str) -> int:
        runtime_match = _EXPERT_RUNTIME_PAT.match(name)
        if runtime_match:
            layer = int(runtime_match.group("layer"))
            proj = runtime_match.group("proj")
            return self.expert_layer_bits.get(layer, self.expert_proj_bits)[proj]
        m = _EXPERT_PAT.search(name)
        if not m:
            raise ValueError(f"not a routed expert weight: {name}")
        return self.expert_proj_bits[m.group(1)]

    def routed_expert_bit_plan(self) -> dict[str, Any]:
        return {
            "default": self.expert_proj_bits,
            "group_size": self.expert_group_size,
            "layer_overrides": {
                str(layer): bits for layer, bits in sorted(self.expert_layer_bits.items())
            },
        }


def runtime_quant_base_for_weight(name: str) -> str:
    """Return the MLX module path that owns a converted affine weight."""
    m = _EXPERT_RUNTIME_PAT.match(name)
    if m:
        return f"{m.group(1)}.switch_mlp.{m.group('proj')}"
    return name[: -len(".weight")] if name.endswith(".weight") else name



_AWQ_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(post_attention_layernorm\.weight|mlp\.gate\.weight|mlp\.experts\.\d+\.(?:gate_proj|up_proj)\.weight)$")


def load_awq_scales(path: Path) -> dict[int, torch.Tensor]:
    """Load per-layer AWQ MoE input scales saved by awq_source_probe.py."""
    data = json.loads(path.read_text(encoding="utf-8"))
    layers_data = data.get("layers")
    if not isinstance(layers_data, dict) or not layers_data:
        raise ValueError(f"awq scales file missing layers dict: {path}")
    scales: dict[int, torch.Tensor] = {}
    for key, values in layers_data.items():
        s = torch.tensor(values, dtype=torch.float32)
        if s.ndim != 1 or not torch.isfinite(s).all() or (s <= 0).any():
            raise ValueError(f"awq scales layer {key}: must be a finite positive 1-D vector")
        scales[int(key)] = s
    return scales


def apply_awq_fold(name: str, t: torch.Tensor, awq_scales: dict[int, torch.Tensor] | None) -> torch.Tensor:
    """Fold AWQ input scales so the bundle needs no runtime changes.

    MoE input x is produced by post_attention_layernorm and consumed by the
    router and expert gate/up projections. Folding 1/s into the norm weight
    and s into the consumer columns keeps the full-precision function
    identical while shaping quantization error on gate/up:
      norm.weight /= s ; router.weight[:, c] *= s ; gate/up[:, c] *= s.
    down_proj input lives in a different space and is untouched.
    """
    if not awq_scales:
        return t
    m = _AWQ_LAYER_RE.match(name)
    if m is None:
        return t
    layer_idx = int(m.group(1))
    s = awq_scales.get(layer_idx)
    if s is None:
        return t
    if name.endswith("post_attention_layernorm.weight"):
        if t.shape[-1] != s.numel():
            raise ValueError(f"{name}: norm dim {tuple(t.shape)} vs awq scale {s.numel()}")
        return (t.float() / s).to(t.dtype)
    if t.shape[-1] != s.numel():
        raise ValueError(f"{name}: weight cols {tuple(t.shape)} vs awq scale {s.numel()}")
    return (t.float() * s.view(1, -1)).to(t.dtype)


def read_bf16_storage_tensor(idx: MiMoShardIndex, name: str) -> torch.Tensor:
    """Read a tensor for bf16 storage, dequantizing FP8 source weights first."""
    if idx.is_fp8_weight(name):
        return idx.read_tensor(name, out_dtype=torch.bfloat16)
    return idx.read_passthrough(name).to(torch.bfloat16)


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

    if profile_bits.token_io_bf16 and name in {"model.embed_tokens.weight", "lm_head.weight"}:
        return 16, "passthrough_bf16", 0

    if profile_bits.non_expert_text_bf16 and name.endswith(".weight") and not is_routed_expert_weight(name):
        return 16, "passthrough_bf16", 0

    if profile_bits.qkv_bits != profile_bits.default_bits and name.endswith(".self_attn.qkv_proj.weight"):
        return profile_bits.qkv_bits, "affine", 64

    if profile_bits.layer0_dense_bits != profile_bits.default_bits and name.startswith("model.layers.0.mlp.") and name.endswith("_proj.weight"):
        return profile_bits.layer0_dense_bits, "affine", 64

    if profile_bits.o_proj_bits is not None and name.endswith(".o_proj.weight"):
        return profile_bits.o_proj_bits, "affine", 64

    # bf16 passthrough: all o_proj.weight (in source `ignored_layers`) + MTP eh_proj (bf16 in source).
    if name.endswith(".o_proj.weight") or name.endswith(".eh_proj.weight"):
        return 16, "passthrough_bf16", 0

    # Routed experts → profile_bits affine.
    if is_routed_expert_weight(name):
        return profile_bits.bits_for_expert_name(name), "affine", profile_bits.expert_group_size

    # Everything else (qkv_proj, layer-0 dense MLP, embed, lm_head, MTP qkv/mlp)
    # uses the profile default affine bit width.
    if name.endswith(".weight"):
        return profile_bits.default_bits, "affine", 64

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
    routed_group_size: int,
    quant_overrides: dict[str, dict],
    include_mtp: bool = True,
    expert_keep_map: ExpertKeepMap | None = None,
    awq_scales: dict[int, torch.Tensor] | None = None,
) -> None:
    cfg = json.loads((src / "config.json").read_text())
    cfg.pop("quantization_config", None)
    _normalize_rope(cfg)
    # mlx-lm load_model expects per-tensor overrides AT THE TOP LEVEL of `quantization`,
    # keyed by module path. The `class_predicate` does `config["quantization"][p]`.
    # Nesting under `overrides` makes mlx-lm fall back to default bits → shape mismatch.
    quant_dict: dict[str, Any] = {
        "bits": profile.default_bits,
        "group_size": routed_group_size,
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
    cfg["jang_version"] = "v2"
    if expert_keep_map is not None:
        cfg["n_routed_experts"] = expert_keep_map.keep_experts
        text_config = cfg.get("text_config")
        if isinstance(text_config, dict):
            text_config["n_routed_experts"] = expert_keep_map.keep_experts
            if "num_experts" in text_config:
                text_config["num_experts"] = expert_keep_map.keep_experts
    cfg["capabilities"] = {
        "family": "mimo_v2",
        # The released mlx runtime is text-only today. Vision/audio tensors are
        # preserved in the bundle for a future multimodal module, but generated
        # model metadata must not advertise media inference before that forward
        # path exists and is live-proven.
        "modalities": ["text"],
        "preserved_modalities": ["vision", "audio"],
        "unwired_modalities": ["vision", "audio"],
        "cache_type": "kv",
        "attention": {
            "full": True,
            "sliding_window": True,
            "sliding_window_size": cfg.get("sliding_window"),
        },
        "reasoning": {"supported": True, "default": True, "parser": "think_xml"},
        "tools": {"supported": True, "parser": "xml_function"},
        "multimodal_status": "weights_preserved_text_runtime",
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
        "routed_expert_bit_plan": profile.routed_expert_bit_plan(),
    }
    if expert_keep_map is not None:
        cfg["runtime"]["expert_keep_map"] = expert_keep_map.metadata()
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))


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
    include_mtp: bool = True,
    expert_keep_map: ExpertKeepMap | None = None,
    awq_scales: dict[int, torch.Tensor] | None = None,
) -> None:
    profile = QuantProfile.parse(profile_bits)
    dst.mkdir(parents=True, exist_ok=True)
    idx = MiMoShardIndex(src)
    weight_keys = idx.weight_keys

    print(f"[convert] source: {src}")
    print(f"[convert] target: {dst}")
    print(f"[convert] profile: {profile.name} (routed_experts={profile.routed_expert_bit_plan()}, "
          f"bookend={profile.default_bits}-bit, group_size=64)")
    if expert_keep_map is not None:
        print(f"[convert] expert keep map: {expert_keep_map.keep_experts} experts/layer", flush=True)
    print(f"[convert] MTP tensors: {'preserve' if include_mtp else 'drop'}")
    if awq_scales:
        print(f"[convert] AWQ fold: {len(awq_scales)} layers of MoE input scales")
    print(f"[convert] {len(weight_keys)} logical tensors", flush=True)

    shard_idx = 1
    shard_bytes = 0
    shard_buf: dict[str, torch.Tensor] = {}
    shard_map: dict[str, str] = {}
    quant_overrides: dict[str, dict] = {}
    method_totals: dict[str, int] = {"affine": 0, "passthrough_bf16": 0, "passthrough_fp32": 0}
    bit_totals: dict[int, int] = {}
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

    def add_tensor(name: str, t: torch.Tensor) -> None:
        nonlocal shard_bytes
        # Ensure contiguous + cpu before save.
        if not t.is_contiguous():
            t = t.contiguous()
        shard_buf[name] = t.cpu()
        shard_bytes += t.element_size() * t.numel()
        if shard_bytes >= max_shard_bytes:
            flush_shard()

    DEFAULT_BITS = profile.default_bits
    DEFAULT_GROUP = 64

    for i, name in enumerate(weight_keys):
        if not include_mtp and name.startswith("model.mtp."):
            continue
        out_name = remap_expert_tensor_name(name, expert_keep_map)
        if out_name is None:
            continue
        bits, method, group_size = classify(name, profile)

        if method == "skip":
            continue
        if method == "passthrough_bf16":
            t = read_bf16_storage_tensor(idx, name)
            t = apply_awq_fold(name, t, awq_scales)
            t = slice_router_tensor(name, t, expert_keep_map)
            add_tensor(out_name, t)
            method_totals["passthrough_bf16"] += 1
        elif method == "passthrough_fp32":
            t = idx.read_passthrough(name, out_dtype=torch.float32)
            t = apply_awq_fold(name, t, awq_scales)
            t = slice_router_tensor(name, t, expert_keep_map)
            add_tensor(out_name, t)
            method_totals["passthrough_fp32"] += 1
        elif method == "affine":
            t = idx.read_tensor(name, out_dtype=torch.float32)
            t = apply_awq_fold(name, t, awq_scales)
            qw, qs, qb = quantize_minmax_affine(t, group_size=group_size, bits=bits)
            base = out_name[: -len(".weight")] if out_name.endswith(".weight") else out_name
            add_tensor(f"{base}.weight", qw)
            add_tensor(f"{base}.scales", qs)
            add_tensor(f"{base}.biases", qb)
            if bits != DEFAULT_BITS or group_size != DEFAULT_GROUP:
                runtime_base = runtime_quant_base_for_weight(name)
                quant_overrides[runtime_base] = {"bits": bits, "group_size": group_size, "mode": "affine"}
            bit_totals[bits] = bit_totals.get(bits, 0) + 1
            method_totals["affine"] += 1
        else:
            raise RuntimeError(f"unknown classification method {method!r} for {name}")

        if (i + 1) % 500 == 0:
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

    _write_config_json(
        src,
        dst,
        profile,
        DEFAULT_GROUP,
        quant_overrides,
        include_mtp=include_mtp,
        expert_keep_map=expert_keep_map,
    )
    _copy_aux_files(src, dst)

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
                   help="Quant profile: 2s/JANG_2S, grouped forms like 444g64[n16|t16], 2/4, or 2k/422.")
    p.add_argument("--max-shard-bytes", type=int, default=1_000_000_000,
                   help="Max bytes per output shard (default 1 GB).")
    p.add_argument("--drop-mtp", action="store_true",
                   help="Do not include model.mtp.* speculative decoding tensors.")
    p.add_argument("--expert-keep-map", type=Path,
                   help="Router trace or keep-map JSON used to prune routed experts.")
    p.add_argument("--awq-scales", type=Path, default=None,
                   help="JSON of per-layer AWQ MoE input scales; folded into norm/router/gate/up.")
    p.add_argument("--keep-experts", type=int,
                   help="Number of ranked experts to keep per MoE layer when --expert-keep-map is set.")
    args = p.parse_args(argv)

    expert_keep_map = None
    if args.expert_keep_map is not None:
        if args.keep_experts is None:
            p.error("--keep-experts is required with --expert-keep-map")
        expert_keep_map = load_expert_keep_map(
            args.expert_keep_map.expanduser(),
            keep_experts=args.keep_experts,
        )

    convert(
        args.src.expanduser(),
        args.dst.expanduser(),
        args.profile,
        args.max_shard_bytes,
        include_mtp=not args.drop_mtp,
        expert_keep_map=expert_keep_map,
        awq_scales=load_awq_scales(args.awq_scales.expanduser()) if args.awq_scales else None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
