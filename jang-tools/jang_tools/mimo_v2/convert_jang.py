"""MiMo-V2.5 → JANG bundle converter.

Profiles:
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

    @classmethod
    def parse(cls, raw: str | int) -> "QuantProfile":
        key = str(raw).strip().lower().replace("_", "").replace("-", "").replace("/", "")
        if key in {"2", "2l", "jang2l"}:
            return cls("JANG_2L", 2, {"gate_proj": 2, "up_proj": 2, "down_proj": 2})
        if key in {"4", "4m", "jang4m"}:
            return cls("JANG_4M", 4, {"gate_proj": 4, "up_proj": 4, "down_proj": 4})
        if key in {"k", "2k", "422", "242", "jang2k"}:
            bits = {"gate_proj": 2, "up_proj": 2, "down_proj": 4}
            return cls("JANG_2K", bits, bits)
        raise ValueError(f"unknown MiMo quant profile {raw!r}; use 2, 4, or 2k")

    @property
    def default_bits(self) -> int:
        return 8

    def bits_for_expert_name(self, name: str) -> int:
        m = _EXPERT_PAT.search(name)
        if not m:
            raise ValueError(f"not a routed expert weight: {name}")
        return self.expert_proj_bits[m.group(1)]


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

    # bf16 passthrough: all o_proj.weight (in source `ignored_layers`) + MTP eh_proj (bf16 in source).
    if name.endswith(".o_proj.weight") or name.endswith(".eh_proj.weight"):
        return 16, "passthrough_bf16", 0

    # Routed experts → profile_bits affine.
    if is_routed_expert_weight(name):
        return profile_bits.bits_for_expert_name(name), "affine", 64

    # Everything else (qkv_proj, layer-0 dense MLP, embed, lm_head, MTP qkv/mlp) → 8-bit affine.
    if name.endswith(".weight"):
        return 8, "affine", 64

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
    }
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
) -> None:
    import mlx.core as mx

    profile = QuantProfile.parse(profile_bits)
    dst.mkdir(parents=True, exist_ok=True)
    idx = MiMoShardIndex(src)
    weight_keys = idx.weight_keys

    print(f"[convert] source: {src}")
    print(f"[convert] target: {dst}")
    print(f"[convert] profile: {profile.name} (routed_experts={profile.routed_expert_bits}, "
          f"bookend=8-bit, group_size=64)")
    print(f"[convert] MTP tensors: {'preserve' if include_mtp else 'drop'}")
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

    def _mx_to_torch(arr_mx, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Convert mx.array → torch.Tensor without going through numpy when possible."""
        t = torch.from_numpy(np.array(arr_mx))
        if dtype is not None:
            t = t.to(dtype)
        return t

    for i, name in enumerate(weight_keys):
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
            w = mx.array(t.numpy())
            qw, qs, qb = mx.quantize(w, group_size=group_size, bits=bits)
            base = name[: -len(".weight")] if name.endswith(".weight") else name
            # mx.quantize returns: qw=uint32 packed, qs=fp16/fp32 scales, qb=fp16/fp32 biases
            add_tensor(f"{base}.weight", _mx_to_torch(qw))
            add_tensor(f"{base}.scales", _mx_to_torch(qs, torch.bfloat16))
            add_tensor(f"{base}.biases", _mx_to_torch(qb, torch.bfloat16))
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

    _write_config_json(src, dst, profile, DEFAULT_GROUP, quant_overrides, include_mtp=include_mtp)
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
                   help="Quant profile: 2/JANG_2L, 4/JANG_4M, or 2k/422/JANG_2K.")
    p.add_argument("--max-shard-bytes", type=int, default=1_000_000_000,
                   help="Max bytes per output shard (default 1 GB).")
    p.add_argument("--drop-mtp", action="store_true",
                   help="Do not include model.mtp.* speculative decoding tensors.")
    args = p.parse_args(argv)

    convert(
        args.src.expanduser(),
        args.dst.expanduser(),
        args.profile,
        args.max_shard_bytes,
        include_mtp=not args.drop_mtp,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
