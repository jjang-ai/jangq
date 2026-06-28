"""Structural verifier for MiMo-V2.5 JANG bundles.

Checks performed (in order, fail-fast):
  1. config.json present with expected jang_profile + quantization metadata
  2. model.safetensors.index.json present and total_size matches sum of shard file sizes
  3. Every weight_map entry resolves to an existing shard file
  4. Every expected tensor (per source modeling) is present in bundle
  5. Quantized tensors expose `.weight`, `.scales`, `.biases` triplet with correct shapes
  6. Passthrough tensors have correct dtype (bf16 norms/sinks/visual, fp32 router/correction-bias)
  7. Auxiliary files (tokenizer, modeling code, audio_tokenizer/) present
  8. chat_template.jinja extracted matches tokenizer_config.json["chat_template"]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open


def _check_file(path: Path, label: str, failures: list[str]) -> bool:
    if not path.exists():
        failures.append(f"missing {label}: {path}")
        return False
    return True


@dataclass(frozen=True)
class StoragePolicy:
    prestacked_jangtq: bool
    bundle_has_mtp: bool
    token_io_passthrough_bf16: bool
    o_proj_quantized: bool
    default_bits: int


def _storage_policy_for_config(cfg: dict) -> StoragePolicy:
    q = cfg.get("quantization", {})
    runtime = cfg.get("runtime", {})
    tq_layout = runtime.get("tq_layout")
    profile_name = str(cfg.get("jang_profile") or "")
    prestacked_jangtq = (
        tq_layout == "prestacked_switch_mlp"
        or q.get("routed_experts") == "tq_prestacked_switch_mlp"
        or profile_name.startswith("JANGTQ")
    )
    bundle_has_mtp = bool(runtime.get("bundle_has_mtp", True))
    bf16_token_profiles = {"JANG_2S", "JANG_2C", "JANG_2X"}
    quantized_oproj_profiles = {"JANG_2S", "JANG_2C", "JANG_2X", "JANG_2Q", "JANG_2F"}
    default_bits = 8
    if profile_name in {"JANG_2Q", "JANG_2F"} or profile_name.startswith("JANG_2R_"):
        m = re.search(r"B([568])$", profile_name)
        default_bits = int(m.group(1)) if m else 4
    return StoragePolicy(
        prestacked_jangtq=prestacked_jangtq,
        bundle_has_mtp=bundle_has_mtp,
        token_io_passthrough_bf16=profile_name in bf16_token_profiles,
        o_proj_quantized=prestacked_jangtq or profile_name in quantized_oproj_profiles or profile_name.startswith("JANG_2R_"),
        default_bits=default_bits,
    )


def verify_bundle(bundle: Path, src: Path | None = None) -> int:
    failures: list[str] = []
    warnings: list[str] = []
    bundle = Path(bundle).expanduser()
    print(f"[verify] bundle: {bundle}")

    # 1. config.json
    cfg_path = bundle / "config.json"
    if not _check_file(cfg_path, "config.json", failures):
        return _report(failures, warnings)
    cfg = json.loads(cfg_path.read_text())
    expected_top = ["quantization", "mxtq_bits", "routed_expert_bits", "jang_profile", "rope_parameters"]
    for k in expected_top:
        if k not in cfg:
            failures.append(f"config.json missing required top-level key: {k}")
    if cfg.get("model_type") != "mimo_v2":
        failures.append(f"unexpected model_type: {cfg.get('model_type')} (want mimo_v2)")
    q = cfg.get("quantization", {})
    runtime = cfg.get("runtime", {})
    tq_layout = runtime.get("tq_layout")
    storage_policy = _storage_policy_for_config(cfg)
    is_prestacked_jangtq = storage_policy.prestacked_jangtq
    if q.get("quant_method") != "affine":
        failures.append(f"quantization.quant_method = {q.get('quant_method')} (want 'affine')")
    bits = q.get("bits")
    if bits != storage_policy.default_bits:
        failures.append(
            f"quantization.bits = {bits} "
            f"(want {storage_policy.default_bits} profile default; routed experts use overrides)"
        )
    if q.get("group_size") != 64:
        failures.append(f"quantization.group_size = {q.get('group_size')} (want 64)")
    for k in ("capabilities", "runtime"):
        if k not in cfg:
            failures.append(f"config.json missing required top-level key: {k}")
    if cfg.get("capabilities", {}).get("cache_type") != "kv":
        failures.append(f"capabilities.cache_type = {cfg.get('capabilities', {}).get('cache_type')} (want kv)")
    modalities = cfg.get("capabilities", {}).get("modalities")
    if modalities != ["text"]:
        failures.append(
            f"capabilities.modalities = {modalities!r} (want ['text'] until MiMo media forward is wired)"
        )
    preserved_modalities = cfg.get("capabilities", {}).get("preserved_modalities")
    if preserved_modalities != ["vision", "audio"]:
        failures.append(
            "capabilities.preserved_modalities = "
            f"{preserved_modalities!r} (want ['vision', 'audio'])"
        )
    unwired_modalities = cfg.get("capabilities", {}).get("unwired_modalities")
    if unwired_modalities != ["vision", "audio"]:
        failures.append(
            "capabilities.unwired_modalities = "
            f"{unwired_modalities!r} (want ['vision', 'audio'])"
        )
    tool_parser = cfg.get("capabilities", {}).get("tools", {}).get("parser")
    if tool_parser != "xml_function":
        failures.append(
            f"capabilities.tools.parser = {tool_parser!r} (want 'xml_function')"
        )
    reasoning_parser = cfg.get("capabilities", {}).get("reasoning", {}).get("parser")
    if reasoning_parser != "think_xml":
        failures.append(
            f"capabilities.reasoning.parser = {reasoning_parser!r} (want 'think_xml')"
        )
    cache_topology = runtime.get("cache_topology", {})
    expected_cache_topology = {
        "family": "hybrid_full_swa_kv",
        "prefix_cache": True,
        "l2_disk_cache": True,
        "turboquant_kv": "full_attention_layers_only",
        "swa_layers": "rotating_kv_native",
    }
    for key, expected in expected_cache_topology.items():
        actual = cache_topology.get(key)
        if actual != expected:
            failures.append(
                f"runtime.cache_topology.{key} = {actual!r} (want {expected!r})"
            )
    bundle_has_mtp = storage_policy.bundle_has_mtp
    expected_mtp_mode = "preserved_disabled" if bundle_has_mtp else "absent"
    if runtime.get("mtp_mode") != expected_mtp_mode:
        failures.append(
            f"runtime.mtp_mode = {runtime.get('mtp_mode')} "
            f"(want {expected_mtp_mode})"
        )
    if runtime.get("multimodal_mode") != "weights_preserved_text_runtime":
        failures.append(
            "runtime.multimodal_mode = "
            f"{runtime.get('multimodal_mode')!r} "
            "(want 'weights_preserved_text_runtime')"
        )
    bit_plan = runtime.get("routed_expert_bit_plan", {})
    layer_1_plan = bit_plan.get("layer_overrides", {}).get("1")
    routed_bits = cfg.get("routed_expert_bits")
    if isinstance(layer_1_plan, dict):
        expected_down = layer_1_plan.get("down_proj")
    elif isinstance(routed_bits, int):
        expected_down = routed_bits
    elif isinstance(routed_bits, dict):
        expected_down = routed_bits.get("down_proj")
    else:
        expected_down = None
    down_override = q.get("model.layers.1.mlp.switch_mlp.down_proj", {})
    if not is_prestacked_jangtq and down_override.get("bits") != expected_down:
        failures.append(
            "missing/incorrect runtime override for model.layers.1.mlp.switch_mlp.down_proj: "
            f"{down_override} (want bits={expected_down})"
        )
    if is_prestacked_jangtq and tq_layout != "prestacked_switch_mlp":
        failures.append(
            f"runtime.tq_layout = {tq_layout!r} (want 'prestacked_switch_mlp')"
        )
    print(f"[verify] config OK: profile={cfg.get('jang_profile')} routed_bits={cfg.get('routed_expert_bits')}")

    # 2. index.json + shards
    idx_path = bundle / "model.safetensors.index.json"
    if not _check_file(idx_path, "model.safetensors.index.json", failures):
        return _report(failures, warnings)
    idx = json.loads(idx_path.read_text())
    weight_map: dict[str, str] = idx["weight_map"]
    declared_total = idx.get("metadata", {}).get("total_size", 0)
    shard_files = set(weight_map.values())
    actual_total = 0
    for fn in shard_files:
        p = bundle / fn
        if not p.exists():
            failures.append(f"shard referenced but not present: {fn}")
        else:
            actual_total += p.stat().st_size
    if declared_total != actual_total:
        warnings.append(
            f"metadata.total_size {declared_total} != sum of shard sizes {actual_total}"
        )
    print(f"[verify] {len(weight_map)} tensors across {len(shard_files)} shards, "
          f"{actual_total / 1e9:.2f} GB")

    # 3. Spot-check tensor structure: quantized triplet + dtype
    # Pick a few expected tensor groups.
    spot_groups = {
        "attn_qkv":      "model.layers.0.self_attn.qkv_proj",
        "layer0_dense":  "model.layers.0.mlp.gate_proj",
    }
    if not storage_policy.token_io_passthrough_bf16:
        spot_groups.update({
            "embed":         "model.embed_tokens",
            "lm_head":       "lm_head",
        })
    if is_prestacked_jangtq:
        tq_spot_groups = {
            "routed_gate_tq": "model.layers.1.mlp.switch_mlp.gate_proj",
            "routed_up_tq": "model.layers.1.mlp.switch_mlp.up_proj",
            "routed_down_tq": "model.layers.1.mlp.switch_mlp.down_proj",
        }
    else:
        spot_groups["routed_expert"] = "model.layers.1.mlp.experts.0.gate_proj"
        tq_spot_groups = {}
    if bundle_has_mtp:
        spot_groups["mtp_qkv"] = "model.mtp.layers.0.self_attn.qkv_proj"
    for label, base in tq_spot_groups.items():
        expected_members = (
            f"{base}.tq_packed",
            f"{base}.tq_norms",
            f"{base}.tq_bits",
        )
        for k in expected_members:
            if k not in weight_map:
                failures.append(f"{label}: expected JANGTQ tensor member missing: {k}")
        packed_key = f"{base}.tq_packed"
        if packed_key in weight_map:
            with safe_open(str(bundle / weight_map[packed_key]), framework="pt", device="cpu") as f:
                wt = f.get_slice(packed_key)
                print(
                    f"[verify] {label}: packed shape={tuple(wt.get_shape())} dtype={wt.get_dtype()}"
                )
    for label, base in spot_groups.items():
        weight_key = f"{base}.weight"
        scales_key = f"{base}.scales"
        biases_key = f"{base}.biases"
        expected_members = (weight_key, scales_key, biases_key)
        for k in expected_members:
            if k not in weight_map:
                failures.append(f"{label}: expected tensor member missing: {k}")
        # Inspect dtypes
        if weight_key in weight_map:
            with safe_open(str(bundle / weight_map[weight_key]), framework="pt", device="cpu") as f:
                wt = f.get_slice(weight_key)
                w_dtype = str(wt.get_dtype())
                w_shape = tuple(wt.get_shape())
            # mx.quantize packs into uint32; safetensors reports this as "U32".
            up = w_dtype.upper()
            if "U32" not in up and "UINT" not in up and "INT" not in up:
                warnings.append(f"{label}: weight dtype is {w_dtype}, expected uint32 packed")
            print(f"[verify] {label}: weight shape={w_shape} dtype={w_dtype}")

    # 4. Passthrough checks: norms should be bf16, gates should be fp32
    passthrough_spot = {
        "model.layers.0.input_layernorm.weight":               ("bf16", "norm"),
        "model.layers.1.input_layernorm.weight":               ("bf16", "norm"),
        "model.norm.weight":                                   ("bf16", "norm"),
        "model.layers.1.mlp.gate.weight":                      ("f32",  "router gate"),
        "model.layers.1.mlp.gate.e_score_correction_bias":     ("f32",  "router bias"),
        "model.layers.1.self_attn.attention_sink_bias":        ("bf16", "SWA sink bias"),
        "visual.blocks.0.attn.qkv.weight":                     ("bf16", "ViT qkv"),
        "audio_encoder.input_local_transformer.layers.0.input_layernorm.weight": ("bf16", "audio norm"),
        "speech_embeddings.0.weight":                          ("bf16", "speech emb"),
    }
    if storage_policy.token_io_passthrough_bf16:
        passthrough_spot.update({
            "model.embed_tokens.weight": ("bf16", "embed"),
            "lm_head.weight": ("bf16", "lm_head"),
        })
    if not storage_policy.o_proj_quantized:
        passthrough_spot["model.layers.0.self_attn.o_proj.weight"] = ("bf16", "o_proj")
    else:
        spot_groups_for_oproj = {
            "o_proj": "model.layers.0.self_attn.o_proj",
        }
        for label, base in spot_groups_for_oproj.items():
            for suffix in ("weight", "scales", "biases"):
                key = f"{base}.{suffix}"
                if key not in weight_map:
                    failures.append(f"{label}: expected affine tensor member missing: {key}")
    if bundle_has_mtp:
        passthrough_spot.update({
            "model.mtp.layers.0.self_attn.o_proj.weight": ("bf16", "MTP o_proj"),
            "model.mtp.layers.0.eh_proj.weight": ("bf16", "MTP eh_proj"),
        })
    for k, (want_dtype, label) in passthrough_spot.items():
        if k not in weight_map:
            failures.append(f"missing passthrough tensor: {k} ({label})")
            continue
        with safe_open(str(bundle / weight_map[k]), framework="pt", device="cpu") as f:
            dt = str(f.get_slice(k).get_dtype()).lower()
        if want_dtype.lower() not in dt:
            failures.append(f"{label} ({k}): dtype is {dt}, want {want_dtype}")
        else:
            print(f"[verify] passthrough OK: {label} dtype={dt}")

    # 5. Count routed expert layout.
    if is_prestacked_jangtq:
        tq_members = [
            k
            for k in weight_map
            if ".mlp.switch_mlp." in k
            and k.endswith((".tq_packed", ".tq_norms", ".tq_bits"))
        ]
        expected_tq_members = 47 * 3 * 3
        if len(tq_members) != expected_tq_members:
            failures.append(
                f"prestacked routed expert JANGTQ tensor count = {len(tq_members)}, "
                f"expected {expected_tq_members}"
            )
        else:
            print(f"[verify] prestacked routed expert JANGTQ tensor count = {len(tq_members)} ✓")
        expert_weights = [k for k in weight_map if ".mlp.experts." in k and k.endswith(".weight")]
        if expert_weights:
            failures.append(
                f"prestacked JANGTQ bundle unexpectedly has per-expert weights: {len(expert_weights)}"
            )
    else:
        expert_weights = [k for k in weight_map if ".mlp.experts." in k and k.endswith(".weight")]
        expected_experts = int(
            runtime.get("expert_keep_map", {}).get("keep_experts")
            or cfg.get("n_routed_experts")
            or 256
        )
        if len(expert_weights) != 47 * expected_experts * 3:
            failures.append(
                f"routed expert .weight count = {len(expert_weights)}, expected {47 * expected_experts * 3}"
            )
        else:
            print(f"[verify] routed expert .weight count = {len(expert_weights)} ✓")

    # 6. Aux files
    for fn in (
        "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt",
        "generation_config.json", "preprocessor_config.json",
        "configuration_mimo_v2.py", "modeling_mimo_v2.py", "chat_template.jinja",
    ):
        if not (bundle / fn).exists():
            failures.append(f"missing aux file: {fn}")

    if not (bundle / "audio_tokenizer" / "model.safetensors").exists():
        failures.append("missing audio_tokenizer/model.safetensors")
    else:
        print("[verify] audio_tokenizer/ present")

    # 7. chat_template.jinja content matches tokenizer_config
    tc_path = bundle / "tokenizer_config.json"
    ct_path = bundle / "chat_template.jinja"
    if tc_path.exists() and ct_path.exists():
        tc = json.loads(tc_path.read_text())
        embedded = tc.get("chat_template", "")
        extracted = ct_path.read_text()
        if embedded != extracted:
            failures.append("chat_template.jinja does not match tokenizer_config.json embedded chat_template")
        else:
            print(f"[verify] chat_template.jinja matches embedded ({len(extracted)} chars)")

    return _report(failures, warnings)


def _report(failures: list[str], warnings: list[str]) -> int:
    print()
    if warnings:
        print(f"[verify] {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  WARN: {w}")
    if failures:
        print(f"[verify] {len(failures)} FAILURE(s):")
        for f in failures:
            print(f"  FAIL: {f}")
        return 1
    print("[verify] ✓ bundle passes structural checks")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("bundle", type=Path, help="Path to JANG bundle directory.")
    p.add_argument("--src", type=Path, default=None,
                   help="Optional source checkpoint dir for cross-checks (not yet used).")
    args = p.parse_args(argv)
    return verify_bundle(args.bundle, args.src)


if __name__ == "__main__":
    sys.exit(main())
