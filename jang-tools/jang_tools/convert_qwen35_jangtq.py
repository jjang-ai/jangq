"""
Qwen3.5 / Qwen3.6 / Qwen3-Next → JANGTQ Conversion
Created by Jinho Jang (eric@jangq.ai)

Handles the qwen3_5_moe family: hybrid linear_attn + full_attn, 256 routed
experts + 1 shared, pre-stacked `experts.gate_up_proj`/`experts.down_proj`.

Output layout matches what mlx-lm's Qwen3_5MoeModel.sanitize produces so
`load_jangtq.py` can walk modules via plain `get_module`:

  language_model.model.embed_tokens.{weight,scales,biases}       (affine 8-bit)
  language_model.model.layers.L.self_attn.{q,k,v,o}_proj.{...}   (affine 8-bit)
  language_model.model.layers.L.self_attn.{q,k}_norm.weight       (passthrough f16)
  language_model.model.layers.L.linear_attn.in_proj_{qkv,z}.{...} (affine 8-bit)
  language_model.model.layers.L.linear_attn.in_proj_{b,a}.{...}   (affine 8-bit)
  language_model.model.layers.L.linear_attn.out_proj.{...}        (affine 8-bit)
  language_model.model.layers.L.linear_attn.conv1d.weight         (passthrough f16)
  language_model.model.layers.L.linear_attn.A_log                 (passthrough f16)
  language_model.model.layers.L.linear_attn.dt_bias               (passthrough f16)
  language_model.model.layers.L.linear_attn.norm.weight           (passthrough f16)
  language_model.model.layers.L.{input,post_attention}_layernorm.weight (passthrough)
  language_model.model.layers.L.mlp.gate.weight                   (passthrough f16)
  language_model.model.layers.L.mlp.switch_mlp.gate_proj.tq_*     (MXTQ)
  language_model.model.layers.L.mlp.switch_mlp.up_proj.tq_*       (MXTQ, split from gate_up)
  language_model.model.layers.L.mlp.switch_mlp.down_proj.tq_*     (MXTQ)
  language_model.model.layers.L.mlp.shared_expert.{...}           (affine 8-bit)
  language_model.model.layers.L.mlp.shared_expert_gate.weight     (passthrough f16)
  language_model.model.norm.weight                                (passthrough f16)
  language_model.lm_head.{weight,scales,biases}                   (affine 8-bit)
  vision_tower.*                                                  (passthrough, VL preserved)
  mtp.*                                                           (stripped — jang-spec can
                                                                   reload from source separately)

Usage:
  python3 -m jang_tools.convert_qwen35_jangtq <SRC_DIR> <OUT_DIR> [PROFILE]

Profiles:
  JANGTQ_2L  (default) — attn=8, shared=8, expert=2, embed/lm_head=8
  JANGTQ_3L            — expert=3 (more quality, ~30% more disk)
  JANGTQ_4M            — expert=4 (smallest quality delta vs bf16, ~2× disk of 2L)
"""
import sys, json, gc, shutil, re
import argparse
import numpy as np
import mlx.core as mx
from pathlib import Path
from tqdm import tqdm
from safetensors import safe_open
from safetensors.numpy import save_file

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--progress", choices=["json", "off"], default="off")
_ap.add_argument("--quiet-text", action="store_true")
_args, _rest = _ap.parse_known_args()
sys.argv = [sys.argv[0]] + _rest

from jang_tools.progress import ProgressEmitter
progress = ProgressEmitter(
    json_to_stderr=(_args.progress == "json"),
    quiet_text=_args.quiet_text,
)

from jang_tools.calibrate import _load_bf16_tensor
from jang_tools.turboquant.linear import tq_quantize_weight, tq_quantize_experts


SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else None
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else None
PROFILE = sys.argv[3] if len(sys.argv) > 3 else "JANGTQ_2L"
SEED = 42

if SRC is None or OUT is None:
    print(__doc__)
    sys.exit(1)

try:
    OUT.mkdir(parents=True, exist_ok=True)

    with open(SRC / "config.json") as f:
        config = json.load(f)
    text_cfg = config.get("text_config", config)
    n_layers = text_cfg["num_hidden_layers"]
    # Qwen3.6: 256 routed experts + 1 shared
    n_experts = text_cfg.get("num_experts", text_cfg.get("num_local_experts", 0))
    shared_exp_inter = text_cfg.get("shared_expert_intermediate_size", 0)


    # === Bit assignment ===
    # Canonical profile names: JANGTQ{N} where N = routed-expert bit count.
    # Legacy suffix forms (JANGTQ_2L, _4M, etc.) are accepted as aliases for
    # backwards compatibility; the canonical form is what lands in jang_config.
    _EXPERT_BITS_BY_PROFILE = {
        "JANGTQ2": 2, "JANGTQ_2L": 2, "JANGTQ_2S": 2,
        "JANGTQ3": 3, "JANGTQ_3L": 3, "JANGTQ_3S": 3,
        "JANGTQ4": 4, "JANGTQ_4M": 4, "JANGTQ_4K": 4,
        "JANGTQ_K": "mixed", "JANGTQK": "mixed",
    }
    # Case-insensitive lookup; canonicalize to JANGTQ{N}.
    _PROFILE_NORM = PROFILE.upper()
    # M132 (iter 54): peer-helper parity with convert_minimax_jangtq.py:47-48.
    # The MiniMax converter raises ValueError for unknown profiles; this one
    # used to SILENTLY fall back to 2-bit on any unrecognized name — so a
    # typo like "JANGTQ44" (meant JANGTQ4) would produce a 2-bit conversion
    # with a 4-in-its-name label in jang_config. No error, no warning, just
    # a quality regression mislabeled as something higher.
    if _PROFILE_NORM not in _EXPERT_BITS_BY_PROFILE:
        raise ValueError(
            f"unknown profile {PROFILE!r}; expected one of {sorted(_EXPERT_BITS_BY_PROFILE)}"
        )
    EXPERT_BITS = _EXPERT_BITS_BY_PROFILE[_PROFILE_NORM]
    if EXPERT_BITS == "mixed":
        PROFILE = "JANGTQ_K"
        # Same budget as MiniMax JANGTQ_K: gate/up stay 2-bit, down_proj
        # gets 4-bit because it writes directly back into the residual stream.
        _MIXED_EXPERT_BITS = {"gate_proj": 2, "up_proj": 2, "down_proj": 4}
    else:
        _MIXED_EXPERT_BITS = None
        PROFILE = f"JANGTQ{EXPERT_BITS}"


    def get_bits_and_method(tensor_name):
        """Return (bits, method, out_name).

        `method`   ∈ {"passthrough", "affine", "mxtq", "skip"}
        `out_name` = None means use the input name verbatim; else rename.
        """
        name = tensor_name.lower()

        # Drop MTP head — keep source shards around for jang-spec but not in JANGTQ output.
        # (If you want MTP included, change this to "passthrough".)
        if tensor_name.startswith("mtp.") or ".mtp" in tensor_name:
            return (0, "skip", None)

        # Preserve vision tower as-is (bf16 → f16 passthrough) — always-VL rule.
        if tensor_name.startswith("vision_tower") or tensor_name.startswith("model.visual"):
            return (16, "passthrough", None)

        # Norms (input_layernorm, post_attention_layernorm, q_norm, k_norm, linear_attn.norm,
        # RMSNormGated.norm, model.norm, lm_head norm etc) → passthrough fp16
        if tensor_name.endswith("norm.weight") or tensor_name.endswith(".norm"):
            return (16, "passthrough", None)

        # 1-D Mamba-ish tensors: A_log, dt_bias → passthrough fp16
        if tensor_name.endswith(".A_log") or tensor_name.endswith(".dt_bias"):
            return (16, "passthrough", None)

        # conv1d (grouped 1D conv in GatedDeltaNet) → passthrough fp16
        if tensor_name.endswith("conv1d.weight"):
            return (16, "passthrough", None)

        # Router gate (mlp.gate.weight) and shared_expert_gate (sigmoid scalar gate) → passthrough fp16
        if tensor_name.endswith(".mlp.gate.weight") or ".gate.weight" == tensor_name[-len(".gate.weight"):]:
            # Guard against matching e.g. "gate_proj.weight" — that's handled below.
            if "gate_proj" not in tensor_name and "up_proj" not in tensor_name and "down_proj" not in tensor_name:
                return (16, "passthrough", None)
        if tensor_name.endswith("shared_expert_gate.weight"):
            return (16, "passthrough", None)

        # Embeddings / lm_head → affine 8-bit
        if "embed_tokens" in tensor_name or tensor_name.endswith("lm_head.weight") or tensor_name == "lm_head.weight":
            return (8, "affine", None)

        # Full attention q/k/v/o → affine 8-bit
        if ".self_attn." in tensor_name and tensor_name.endswith("_proj.weight"):
            return (8, "affine", None)

        # Linear attention input/output projections → affine 8-bit
        # (conv1d / A_log / dt_bias / norm handled above)
        if ".linear_attn." in tensor_name and tensor_name.endswith(".weight"):
            return (8, "affine", None)

        # Shared expert projections → affine 8-bit (always active — precision matters)
        if ".shared_expert." in tensor_name and tensor_name.endswith(".weight"):
            return (8, "affine", None)

        # Routed experts (pre-stacked) → MXTQ at profile bits
        #   experts.gate_up_proj  → split into switch_mlp.gate_proj + switch_mlp.up_proj
        #   experts.down_proj     → switch_mlp.down_proj
        if tensor_name.endswith(".mlp.experts.gate_up_proj"):
            bits = _MIXED_EXPERT_BITS["gate_proj"] if _MIXED_EXPERT_BITS else EXPERT_BITS
            return (bits, "mxtq", None)
        if tensor_name.endswith(".mlp.experts.down_proj"):
            bits = _MIXED_EXPERT_BITS["down_proj"] if _MIXED_EXPERT_BITS else EXPERT_BITS
            return (bits, "mxtq", None)

        # Anything else → affine 8-bit fallback (keeps conversion total)
        return (8, "affine", None)


    def sanitize_key(hf_key):
        """Map HF checkpoint key to post-sanitize (MLX) key."""
        # `model.language_model.X` → `language_model.model.X`
        if hf_key.startswith("model.language_model."):
            return hf_key.replace("model.language_model", "language_model.model", 1)
        # `model.visual.X` → `vision_tower.X`
        if hf_key.startswith("model.visual"):
            return hf_key.replace("model.visual", "vision_tower", 1)
        # `lm_head.X` at top level → `language_model.lm_head.X` (Qwen3_5_moe's TextModel owns lm_head)
        if hf_key == "lm_head.weight" or hf_key.startswith("lm_head."):
            return "language_model." + hf_key
        return hf_key


    print("=" * 60)
    print(f"  Qwen3.5/3.6 → {PROFILE} JANGTQ Conversion")
    print(f"  Created by Jinho Jang (eric@jangq.ai)")
    print("=" * 60)
    print(f"  Source:  {SRC}")
    print(f"  Output:  {OUT}")
    print(f"  Layers:  {n_layers}")
    print(f"  Experts: {n_experts} routed + 1 shared ({shared_exp_inter}d)")
    if _MIXED_EXPERT_BITS:
        _expert_profile_desc = (
            "MXTQ-K(gate=2, up=2, down=4)"
        )
    else:
        _expert_profile_desc = f"MXTQ-{EXPERT_BITS}"
    print(f"  Profile: expert={_expert_profile_desc}, attn=affine-8, shared=affine-8, gates=fp16")
    print(flush=True)


    # Scan tensors
    progress.phase(1, 3, "scan")
    print("\n  Scanning source shards...", flush=True)
    all_tensors = []
    for sf in sorted(SRC.glob("model-*.safetensors")):
        with safe_open(str(sf), framework="numpy") as f:
            for k in f.keys():
                if k.endswith("_scale_inv"):
                    continue
                shape = list(f.get_slice(k).get_shape())
                all_tensors.append((k, shape, sf))
    print(f"  Found {len(all_tensors)} tensors", flush=True)


    # === Processing state ===
    shard_idx = 0
    shard_tensors = {}
    shard_bytes = 0
    MAX_SHARD = 1_000_000_000  # 1 GB
    total_mxtq = 0
    total_affine = 0
    total_passthrough = 0
    total_skipped = 0
    shard_map = {}


    def flush_shard():
        global shard_idx, shard_tensors, shard_bytes
        if not shard_tensors:
            return
        shard_idx += 1
        fname = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        save_file(shard_tensors, str(OUT / fname))
        for k in shard_tensors:
            shard_map[k] = fname
        print(f"    Shard {shard_idx}: {len(shard_tensors)} tensors, {shard_bytes / 1e9:.1f} GB", flush=True)
        shard_tensors = {}
        shard_bytes = 0


    def add_tensor(name, arr):
        global shard_bytes
        shard_tensors[name] = arr
        shard_bytes += arr.nbytes
        if shard_bytes >= MAX_SHARD:
            flush_shard()


    # === Resume support ===
    done_keys = set()
    existing_shards = sorted(OUT.glob("model-*-of-XXXXX.safetensors"))
    if existing_shards:
        print(f"\n  Resume: found {len(existing_shards)} existing shards", flush=True)
        import struct as _struct
        for sf in existing_shards:
            with open(sf, "rb") as f:
                hsize = _struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(hsize))
            fname = sf.name
            for k in hdr:
                if k == "__metadata__":
                    continue
                done_keys.add(k)
                shard_map[k] = fname
            idx_str = sf.name.split("-")[1]
            shard_idx = max(shard_idx, int(idx_str))
        print(f"  Resume: {len(done_keys)} keys already written, continuing from shard {shard_idx + 1}", flush=True)


    def is_already_done(out_name, method, split_gate_up):
        if method == "skip":
            return True
        if method == "passthrough":
            return out_name in done_keys
        if method == "affine":
            base = out_name[:-7] if out_name.endswith(".weight") else out_name
            return (f"{base}.weight" in done_keys and f"{base}.scales" in done_keys and f"{base}.biases" in done_keys)
        if method == "mxtq":
            if split_gate_up:
                gb = out_name.replace("experts.gate_up_proj", "switch_mlp.gate_proj")
                ub = out_name.replace("experts.gate_up_proj", "switch_mlp.up_proj")
                return (f"{gb}.tq_packed" in done_keys and f"{ub}.tq_packed" in done_keys)
            else:
                base = out_name.replace("experts.down_proj", "switch_mlp.down_proj") \
                       if out_name.endswith("experts.down_proj") else out_name
                return (f"{base}.tq_packed" in done_keys and
                        f"{base}.tq_norms" in done_keys and f"{base}.tq_bits" in done_keys)
        return False


    # === Main loop ===
    progress.phase(2, 3, "convert")
    print("\n  Converting...", flush=True)
    skipped_resume = 0
    for tensor_name, shape, sf_path in tqdm(all_tensors, desc="  Processing"):
        bits, method, _ = get_bits_and_method(tensor_name)
        out_name = sanitize_key(tensor_name)

        if method == "skip":
            total_skipped += 1
            continue

        split_gate_up = tensor_name.endswith("experts.gate_up_proj")

        # Resume check
        if done_keys and is_already_done(out_name, method, split_gate_up):
            skipped_resume += 1
            if method == "mxtq":   total_mxtq += (2 if split_gate_up else 1)
            elif method == "affine": total_affine += 1
            else:                    total_passthrough += 1
            continue

        # Load tensor (bf16 first — Qwen3.6 source is bf16)
        with safe_open(str(sf_path), framework="numpy") as f:
            try:
                tensor = f.get_tensor(tensor_name)
                if not isinstance(tensor, np.ndarray):
                    tensor = np.array(tensor)
            except Exception:
                tensor = _load_bf16_tensor(sf_path, tensor_name, shape)

        if tensor.dtype != np.float32:
            tensor = tensor.astype(np.float32)

        if method == "passthrough":
            # Qwen3.6 patch_embed.proj.weight ships in HF-native layout
            # (out_channels, in_channels, temporal_patch, patch, patch) — shape like
            # (1152, 3, 2, 16, 16). mlx_vlm's Qwen3_5 ViT expects in_channels last:
            # (out_channels, temporal_patch, patch, patch, in_channels). Transpose
            # once at conversion so the shipped model loads without a runtime hack.
            if out_name.endswith("vision_tower.patch_embed.proj.weight") \
                    and tensor.ndim == 5 and tensor.shape[1] == 3:
                tensor = np.ascontiguousarray(
                    np.transpose(tensor, (0, 2, 3, 4, 1))
                )
            add_tensor(out_name, tensor.astype(np.float16))
            total_passthrough += 1

        elif method == "affine":
            w = mx.array(tensor.astype(np.float16))
            qw, qs, qb = mx.quantize(w, group_size=64, bits=bits)
            base = out_name[:-7] if out_name.endswith(".weight") else out_name
            add_tensor(f"{base}.weight", np.array(qw))
            add_tensor(f"{base}.scales", np.array(qs).astype(np.float16))
            add_tensor(f"{base}.biases", np.array(qb).astype(np.float16))
            total_affine += 1
            del w, qw, qs, qb

        elif method == "mxtq":
            # Routed experts come pre-stacked (3D): [n_experts, out, in] in HF.
            # Use tq_quantize_experts (handles 3D); tq_quantize_weight is 2D-only.
            if split_gate_up:
                # tensor shape: (n_experts, 2*inter, hidden)
                mid = tensor.shape[1] // 2
                gate_tensor = tensor[:, :mid, :]
                up_tensor   = tensor[:, mid:, :]
                gate_out = out_name.replace("experts.gate_up_proj", "switch_mlp.gate_proj")
                up_out   = out_name.replace("experts.gate_up_proj", "switch_mlp.up_proj")
                for half_tensor, half_out in ((gate_tensor, gate_out), (up_tensor, up_out)):
                    result = tq_quantize_experts(half_tensor, bits=bits, seed=SEED)
                    add_tensor(f"{half_out}.tq_packed", result["packed"])
                    add_tensor(f"{half_out}.tq_norms",  result["norms"])
                    add_tensor(f"{half_out}.tq_bits",   np.array([bits], dtype=np.uint8))
                    total_mxtq += 1
                del gate_tensor, up_tensor
            else:
                # experts.down_proj pre-stacked (3D)
                down_out = out_name.replace("experts.down_proj", "switch_mlp.down_proj")
                result = tq_quantize_experts(tensor, bits=bits, seed=SEED)
                add_tensor(f"{down_out}.tq_packed", result["packed"])
                add_tensor(f"{down_out}.tq_norms",  result["norms"])
                add_tensor(f"{down_out}.tq_bits",   np.array([bits], dtype=np.uint8))
                total_mxtq += 1

        del tensor
        if (total_mxtq + total_affine) % 200 == 0:
            gc.collect()

    flush_shard()
    if skipped_resume > 0:
        print(f"\n  Resume: skipped {skipped_resume} tensors from previous run", flush=True)


    # Rename shards to final {i}-of-{total}
    progress.phase(3, 3, "write")
    print(f"\n  Renaming {shard_idx} shards...", flush=True)
    for i in range(1, shard_idx + 1):
        old = OUT / f"model-{i:05d}-of-XXXXX.safetensors"
        new = OUT / f"model-{i:05d}-of-{shard_idx:05d}.safetensors"
        if old.exists():
            old.rename(new)
    shard_map = {k: v.replace("XXXXX", f"{shard_idx:05d}") for k, v in shard_map.items()}


    # Write safetensors index
    total_size = 0
    for fname in set(shard_map.values()):
        p = OUT / fname
        if p.exists():
            total_size += p.stat().st_size
    index = {
        "metadata": {"format": "jangtq", "total_size": total_size},
        "weight_map": shard_map,
    }
    # M125 (iter 48): context-managed open so the write is flushed + closed
    # deterministically. See convert_minimax_jangtq.py for the full rationale.
    with open(OUT / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)


    # Write config (strip any fp8/awq sections; force the default bits to expert bits)
    config.pop("quantization_config", None)
    routed_bits_meta = _MIXED_EXPERT_BITS or EXPERT_BITS
    default_expert_bits = 2 if _MIXED_EXPERT_BITS else EXPERT_BITS
    config["quantization"] = {"group_size": 64, "bits": default_expert_bits}
    # Surface the JANGTQ flags at the top level so Swift's
    # `LLMModelFactory.qwen3_5_moe` dispatch can detect MXTQ format
    # without reading the sidecar.
    config["weight_format"] = "mxtq"
    config["mxtq_seed"] = SEED
    config["mxtq_bits"] = routed_bits_meta
    with open(OUT / "config.json", "w") as f:
        json.dump(config, f, indent=2)


    # Write jang_config
    src_arch = text_cfg.get("model_type") or config.get("model_type", "qwen3_5_moe")
    jang_config = {
        "version": 2,
        "weight_format": "mxtq",
        "profile": PROFILE,
        "source_model": {
            "name": SRC.name,
            "architecture": src_arch,
        },
        # has_vision: true now that Swift Qwen35MoEJANGTQ (vMLXVLM) is
        # available. The detector routes the artifact through VLMModelFactory
        # which dispatches qwen3_5_moe + weight_format=mxtq to the
        # vision-aware JANGTQ class. Both Python (load_jangtq_vlm_model) and
        # Swift VLM paths consume the SAME on-disk weights — the LLM-side
        # Swift Qwen35JANGTQModel.sanitize harmlessly strips vision_tower
        # keys when accidentally used.
        "has_vision": True,
        "mxtq_seed": SEED,
        "mxtq_bits": {
            "attention": 8,
            "linear_attention": 8,
            "shared_expert": 8,
            "routed_expert": routed_bits_meta,
            "embed_tokens": 8,
            "lm_head": 8,
        },
        "quantization": {
            "method": "affine+mxtq",
            "group_size": 64,
            "bits_default": default_expert_bits,
        },
    }
    from jang_tools.convert import _build_mtp_runtime_metadata
    _mtp_meta = _build_mtp_runtime_metadata(config, set(shard_map.keys()))
    if _mtp_meta:
        jang_config["runtime"] = _mtp_meta["runtime"]
        jang_config["mtp"] = _mtp_meta["mtp"]
        jang_config["bundle_has_mtp"] = _mtp_meta["bundle_has_mtp"]
        jang_config["mtp_layers"] = _mtp_meta["mtp_layers"]
        print(
            f"  mtp: mode={_mtp_meta['runtime']['mtp_mode']} "
            f"layers={_mtp_meta['runtime']['mtp_layers']} "
            f"tensor_count={_mtp_meta['mtp']['tensor_count']}",
            flush=True,
        )
    # Stamp Tier-1 capabilities (reasoning/tool parser, cache type, modality)
    # so vmlx CapabilityDetector picks the right parsers without falling back
    # to silver/bronze. Idempotent — safe to re-run.
    from jang_tools.capabilities import build_capabilities
    caps = build_capabilities(jang_config, config, OUT)
    if caps is not None:
        jang_config["capabilities"] = caps
        print(f"  capabilities: family={caps['family']} reasoning={caps['reasoning_parser']} "
              f"tool={caps['tool_parser']} cache={caps['cache_type']} modality={caps['modality']}",
              flush=True)
    else:
        print("  WARNING: could not resolve capabilities for this model — vmlx will "
              "fall back to silver/bronze detection. Add the family to "
              "jang_tools/capabilities.py::FAMILY_MAP if this is a new architecture.",
              flush=True)

    with open(OUT / "jang_config.json", "w") as f:
        json.dump(jang_config, f, indent=2)

    # Validate the final jang_config — catch schema drift / typos / missing keys.
    from jang_tools.capabilities import verify_directory
    _ok, _msg = verify_directory(OUT)
    print(f"  verify: {_msg}")
    if not _ok:
        raise SystemExit(f"capabilities verify failed: {_msg}")


    # Copy tokenizer/template/preprocessor files (always-VL rule: keep VL assets)
    for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "generation_config.json", "chat_template.jinja", "merges.txt", "vocab.json",
              "preprocessor_config.json", "processor_config.json",
              "video_preprocessor_config.json",
              "configuration.json",
              # If the model ships custom .py files, preserve them
              f"modeling_{src_arch}.py", f"configuration_{src_arch}.py"]:
        src_f = SRC / f
        if src_f.exists():
            shutil.copy2(str(src_f), str(OUT / f))


    # --- Osaurus / swift-transformers compatibility fix ---------------------------
    # Some source releases ship tokenizer_config.json with
    #   "tokenizer_class": "TokenizersBackend"
    # which swift-transformers (Osaurus, vmlx-swift-lm) can't parse and throws
    #   unsupportedTokenizer("TokenizersBackend")
    # Map it back to a concrete class that swift-transformers knows. For the
    # Qwen 3.5/3.6 family this is Qwen2Tokenizer (same vocab family).
    _tok_cfg = OUT / "tokenizer_config.json"
    if _tok_cfg.exists():
        try:
            with open(_tok_cfg) as f:
                _tc = json.load(f)
            if _tc.get("tokenizer_class") == "TokenizersBackend":
                _tc["tokenizer_class"] = "Qwen2Tokenizer"
                with open(_tok_cfg, "w") as f:
                    json.dump(_tc, f, indent=2)
                print("  [osaurus-fix] tokenizer_class: TokenizersBackend → Qwen2Tokenizer", flush=True)
        except Exception as _e:
            print(f"  [osaurus-fix] skipped: {_e}", flush=True)


    # Build Swift runtime sidecar so Swift runtimes (vmlx-swift-lm, vmlxctl,
    # Osaurus) can load without fatalError. Python loader doesn't need it, but
    # uploading without it is a footgun.
    print(f"\n  Building jangtq_runtime.safetensors sidecar...")
    try:
        from jang_tools.build_jangtq_sidecar import main as _build_sidecar
        _saved_argv = sys.argv
        sys.argv = ["build_jangtq_sidecar", str(OUT)]
        try:
            _build_sidecar()
        finally:
            sys.argv = _saved_argv
    except (Exception, SystemExit) as _e:
        print(f"  [sidecar] FAILED: {_e} — run `python3 -m jang_tools.build_jangtq_sidecar {OUT}` manually before upload", flush=True)


    print(f"\n  Done!")
    print(f"  MXTQ tensors:    {total_mxtq}")
    print(f"  Affine tensors:  {total_affine}")
    print(f"  Passthrough:     {total_passthrough}")
    print(f"  Skipped (MTP):   {total_skipped}")
    print(f"  Output:          {OUT}")
    progress.done(ok=True, output=str(OUT))

except Exception as _exc:
    progress.done(ok=False, error=f"{type(_exc).__name__}: {_exc}")
    raise
