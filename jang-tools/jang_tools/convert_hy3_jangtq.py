"""
Tencent Hy3-preview → JANGTQ Conversion
Created by Jinho Jang (eric@jangq.ai)

Hy3-preview ("hy_v3", HYV3ForCausalLM, 295B/21B active MoE):
  - 80 transformer layers + 1 MTP (next-token prediction) layer for spec decoding
  - GQA: 64 q-heads / 8 KV-heads, head_dim=128, with q_norm/k_norm pre-attn (qwen3-style)
  - MoE: 192 experts, top-8 routing, 1 shared expert, expert_hidden_dim=1536
  - moe_router_use_sigmoid=True (sigmoid routing, like MiniMax M2)
  - moe_router_enable_expert_bias=True + route_norm + router_scaling_factor (DSV3-style aux-free)
  - first_k_dense_replace=1 (layer 0 is dense FFN, not MoE)
  - 256K context, rope_theta=11_158_840
  - enable_lm_head_fp32=True (LM head precision-sensitive)

Mixed-precision TurboQuant for Hy3:
  routed expert MLP weights → MXTQ at the requested profile
    - JANGTQ1: all routed expert projections 1-bit (experimental size floor)
    - JANGTQ2: all routed expert projections 2-bit (128 GB first target)
    - JANGTQ_K: gate/up 2-bit, down 4-bit (quality-first, 192 GB+ preferred)
    - JANGTQ4: all routed expert projections 4-bit (quality reference)
  attention/embed/lm_head/shared_mlp/dense-FFN → 8-bit affine
  MTP layer tensors → same explicit policy as base layers; never silently dropped
  router/expert_bias/norms → bf16/fp16 passthrough

Output is loadable via load_jangtq.py with the TurboQuantLinear Metal kernel; the runtime
adapter is `Hy3JANGTQModel` (TBD in jang-runtime/Sources/JANG/).
"""
import sys
import json
import gc
import argparse
from pathlib import Path

import numpy as np
import mlx.core as mx
from tqdm import tqdm
from safetensors import safe_open
from safetensors.numpy import save_file

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--progress", choices=["json", "off"], default="off")
_ap.add_argument("--quiet-text", action="store_true")
_ap.add_argument("--dry-run", action="store_true")
_args, _rest = _ap.parse_known_args()
sys.argv = [sys.argv[0]] + _rest

from jang_tools.progress import ProgressEmitter
from jang_tools.calibrate import _load_bf16_tensor
from jang_tools.turboquant.linear import tq_quantize_weight

progress = ProgressEmitter(
    json_to_stderr=(_args.progress == "json"),
    quiet_text=_args.quiet_text,
)

if len(sys.argv) < 3:
    print(
        "usage: python -m jang_tools.convert_hy3_jangtq <src_bf16_dir> <out_dir> [profile]\n"
        "  <src_bf16_dir>  path to a Tencent/Hy3-preview BF16 source directory\n"
        "  <out_dir>       output directory for the JANGTQ bundle\n"
        "  [profile]       JANGTQ1 (experimental), JANGTQ2 (default), JANGTQ_K, or JANGTQ4",
        file=sys.stderr,
    )
    sys.exit(2)

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
PROFILE = sys.argv[3] if len(sys.argv) > 3 else "JANGTQ2"
SEED = 42

# 2026-05-20 live vMLX proof:
# Hy3-preview-JANG_2L with source defaults temperature=0.9/top_p=1/top_k=-1
# loops into repeated phrases around 1.5K-2.2K output tokens even with
# reasoning off and prefix cache bypassed. The same bundle at temperature=0.0
# stops cleanly. This is a model-bundle chat default, not a hidden engine guard;
# stamp it into JANG metadata and generation_config so future uploads are
# self-consistent and do not require re-quantizing weights.
HY3_CHAT_SAMPLING_DEFAULTS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "max_new_tokens": 2048,
}
HY3_GENERATION_CONFIG_OVERRIDES = {
    **HY3_CHAT_SAMPLING_DEFAULTS,
    "do_sample": False,
}

_PROFILE_BITS = {
    "JANGTQ1": 1,
    "JANGTQ2": 2,
    "JANGTQ4": 4,
    "JANGTQ_K": "mixed",
    "JANGTQK": "mixed",
}
_PROFILE_NORM = PROFILE.upper()
if _PROFILE_NORM not in _PROFILE_BITS:
    raise SystemExit(
        f"unknown profile {PROFILE!r}; expected one of {sorted(_PROFILE_BITS)}"
    )
EXPERT_BITS = _PROFILE_BITS[_PROFILE_NORM]
if EXPERT_BITS == "mixed":
    PROFILE = "JANGTQ_K"
    _MIXED_PROJ_BITS = {
        "gate_proj": 2,
        "up_proj": 2,
        "down_proj": 4,
    }
else:
    PROFILE = f"JANGTQ{EXPERT_BITS}"
    _MIXED_PROJ_BITS = None

# JANGTQ1 (1-bit MXTQ on routed experts) is experimental: TurboQuantLinear and
# tq_quantize_weight both support bits=1 (verified via smoke test on Hy3
# routed-expert shapes 4096 % 32 == 0, 1536 % 32 == 0), but quality has not
# been measured against reference outputs. The codebook collapses to 2 entries
# (effectively ±v after Hadamard rotation), and routed experts dominate model
# behavior. Expect a meaningful coherence drop vs JANGTQ2.

OUT.mkdir(parents=True, exist_ok=True)

with open(SRC / "config.json") as f:
    config = json.load(f)

if config.get("model_type") != "hy_v3":
    raise SystemExit(
        f"source model_type is {config.get('model_type')!r}; expected 'hy_v3'"
    )

index_path = SRC / "model.safetensors.index.json"
if not _args.dry_run:
    if not index_path.exists():
        raise SystemExit(
            f"missing {index_path}; refusing to convert a partial Hy3 download"
        )
    with open(index_path) as f:
        source_index = json.load(f)
    missing_shards = sorted(
        {
            shard
            for shard in source_index.get("weight_map", {}).values()
            if not (SRC / shard).exists()
        }
    )
    if missing_shards:
        raise SystemExit(
            f"source download incomplete: {len(missing_shards)} shards missing; "
            f"first missing={missing_shards[0]}"
        )

n_layers = int(config["num_hidden_layers"])
n_experts = int(config["num_experts"])
n_experts_per_tok = int(config.get("num_experts_per_tok", 8))
n_shared_experts = int(config.get("num_shared_experts", 1))
hidden_size = int(config["hidden_size"])
intermediate_size = int(config["intermediate_size"])              # dense-FFN width (13312)
moe_intermediate_size = int(config.get("moe_intermediate_size", config.get("expert_hidden_dim", 1536)))
n_mtp_layers = int(config.get("num_nextn_predict_layers", 0))
first_k_dense_replace = int(config.get("first_k_dense_replace", 0))


def _validate_tq_packing_shape(bits: int, in_features: int, role: str) -> None:
    """Fail before a long conversion if vectorized MXTQ packing cannot align.

    `tq_quantize_weight` vectorizes rows by flattening before `pack_bits`, which
    is bit-identical only when each row has an integral number of packed words.
    Hy3 routed expert dims satisfy this for 1/2/4-bit profiles.
    """
    vals_per_u32 = 32 // bits
    if 32 % bits != 0 or in_features % vals_per_u32 != 0:
        raise SystemExit(
            f"{PROFILE} cannot vector-pack {role}: in_features={in_features}, "
            f"bits={bits}, vals_per_u32={vals_per_u32}. Refusing to start a "
            "long conversion that would fail or mis-pack routed experts."
        )


if _MIXED_PROJ_BITS is None:
    _validate_tq_packing_shape(int(EXPERT_BITS), hidden_size, "gate/up routed experts")
    _validate_tq_packing_shape(int(EXPERT_BITS), moe_intermediate_size, "down routed experts")
else:
    _validate_tq_packing_shape(int(_MIXED_PROJ_BITS["gate_proj"]), hidden_size, "gate routed experts")
    _validate_tq_packing_shape(int(_MIXED_PROJ_BITS["up_proj"]), hidden_size, "up routed experts")
    _validate_tq_packing_shape(int(_MIXED_PROJ_BITS["down_proj"]), moe_intermediate_size, "down routed experts")


def get_bits_and_method(name: str) -> tuple[int, str]:
    """Classify a Hy3 tensor into (bits, method ∈ {passthrough, affine, mxtq})."""
    n = name
    is_mtp_tensor = (
        n_mtp_layers > 0
        and (
            n.startswith(f"model.layers.{n_layers}.")
            or ".mtp" in n.lower()
            or "nextn" in n.lower()
            or "mtp_layer" in n.lower()
        )
    )

    # 1D scalars / norms / biases — passthrough
    if n.endswith(".bias") or "norm" in n.lower():
        return (16, "passthrough")
    if n.endswith(".expert_bias") or ".expert_bias" in n or "e_score_correction_bias" in n:
        return (16, "passthrough")
    # MoE router gate (small but precision-critical) — passthrough
    if n.endswith(".mlp.gate.weight") or ".mlp.router.gate.weight" in n:
        return (16, "passthrough")
    # MTP-layer router gate, if present
    if ".router.gate.weight" in n and "mlp." in n:
        return (16, "passthrough")

    # MTP is preserved for future speculative decode but disabled in the first
    # runtime path. Keep its 2D matmuls at 8-bit affine even if the namespace
    # contains `experts.*`; draft quality affects future acceptance rate.
    if is_mtp_tensor:
        return (8, "affine")

    # Embeddings + LM head — 8-bit affine
    if "embed_tokens" in n or n == "lm_head.weight" or n.endswith(".lm_head.weight"):
        return (8, "affine")

    # Attention Q/K/V/O projections — 8-bit affine (qk_norm tensors caught above)
    if "self_attn" in n and any(p in n for p in (".q_proj", ".k_proj", ".v_proj", ".o_proj")):
        return (8, "affine")

    # Dense FFN (first_k_dense_replace layers + MTP layer's MLP if it's dense) — 8-bit affine
    # Pattern: model.layers.0.mlp.{gate,up,down}_proj.weight (NO 'experts.' or 'shared_mlp')
    if ".mlp." in n and any(p in n for p in (".gate_proj", ".up_proj", ".down_proj")):
        if ".experts." in n:
            # Routed expert tensor — MXTQ
            if _MIXED_PROJ_BITS is not None:
                for proj_name, proj_bits in _MIXED_PROJ_BITS.items():
                    if f".{proj_name}.weight" in n or n.endswith(f".{proj_name}.weight"):
                        return (proj_bits, "mxtq")
                return (2, "mxtq")
            return (EXPERT_BITS, "mxtq")
        if ".shared_experts." in n or ".shared_mlp." in n:
            # Shared expert — 8-bit affine
            return (8, "affine")
        # Dense FFN (layer 0 dense or MTP) — 8-bit affine
        return (8, "affine")

    # Default: 8-bit affine for any 2D matmul we missed; passthrough for 1D
    return (8, "affine")


print("=" * 60)
print(f"  Tencent/Hy3-preview → {PROFILE} JANGTQ Conversion")
print(f"  Created by Jinho Jang (eric@jangq.ai)")
print("=" * 60)
print(f"  Source: {SRC}")
print(f"  Output: {OUT}")
print(f"  Layers: {n_layers} + {n_mtp_layers} MTP, Experts: {n_experts} (top-{n_experts_per_tok}, +{n_shared_experts} shared)")
print(f"  Dense layers (first_k_dense_replace): {first_k_dense_replace}")
if _MIXED_PROJ_BITS is not None:
    expert_profile = "mxtq-mixed(gate=2, up=2, down=4)"
else:
    expert_profile = f"mxtq-{EXPERT_BITS}"
print(f"  Profile: attn=affine-8, routed-expert={expert_profile}, shared/dense/MTP=affine-8, router/bias/norms=passthrough")
if _args.dry_run:
    print("  Dry run: no output tensors will be written")
print(flush=True)

# === Scan source ===
progress.phase(1, 3, "scan")
print("\n  Scanning source...", flush=True)
all_tensors: list[tuple[str, list[int], Path]] = []
for sf in sorted(SRC.glob("model-*.safetensors")):
    with safe_open(str(sf), framework="numpy") as f:
        for k in f.keys():
            shape = list(f.get_slice(k).get_shape())
            all_tensors.append((k, shape, sf))
print(f"  Found {len(all_tensors)} tensors", flush=True)

if _args.dry_run:
    counts: dict[str, int] = {"mxtq": 0, "affine": 0, "passthrough": 0}
    sample: dict[str, list[str]] = {"mxtq": [], "affine": [], "passthrough": []}
    for tensor_name, _shape, _sf_path in all_tensors:
        _bits, method = get_bits_and_method(tensor_name)
        counts[method] = counts.get(method, 0) + 1
        if len(sample[method]) < 8:
            sample[method].append(tensor_name)
    print(json.dumps({"counts": counts, "sample": sample}, indent=2))
    progress.done(ok=True, output="dry-run")
    sys.exit(0)

# === Process ===
shard_idx = 0
shard_tensors: dict[str, np.ndarray] = {}
shard_bytes = 0
MAX_SHARD = 1_000_000_000
total_mxtq = 0
total_affine = 0
total_passthrough = 0
shard_map: dict[str, str] = {}


def flush_shard() -> None:
    global shard_idx, shard_tensors, shard_bytes
    if not shard_tensors:
        return
    shard_idx += 1
    fname = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
    save_file(shard_tensors, str(OUT / fname))
    for k in shard_tensors:
        shard_map[k] = fname
    print(f"    Shard {shard_idx}: {len(shard_tensors)} tensors, {shard_bytes/1e9:.2f} GB", flush=True)
    shard_tensors = {}
    shard_bytes = 0


def add_tensor(name: str, arr: np.ndarray) -> None:
    global shard_bytes
    shard_tensors[name] = arr
    shard_bytes += arr.nbytes
    if shard_bytes >= MAX_SHARD:
        flush_shard()


# Resume support
done_keys: set[str] = set()
existing_shards = sorted(OUT.glob("model-*-of-XXXXX.safetensors"))
if existing_shards:
    print(f"\n  Resume: found {len(existing_shards)} existing shards", flush=True)
    import struct as _struct

    for sf in existing_shards:
        with open(sf, "rb") as f:
            hsize = _struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hsize))
        for k in hdr:
            if k != "__metadata__":
                done_keys.add(k)
                shard_map[k] = sf.name
        idx_str = sf.name.split("-")[1]
        shard_idx = max(shard_idx, int(idx_str))
    print(f"  Resume: {len(done_keys)} keys already written, continuing from shard {shard_idx + 1}", flush=True)


def is_already_done(source_name: str, method: str) -> bool:
    if method == "passthrough":
        return source_name in done_keys
    base = source_name.replace(".weight", "") if source_name.endswith(".weight") else source_name
    if method == "affine":
        return all(f"{base}.{k}" in done_keys for k in ("weight", "scales", "biases"))
    if method == "mxtq":
        return all(f"{base}.{k}" in done_keys for k in ("tq_packed", "tq_norms", "tq_bits"))
    return False


progress.phase(2, 3, "convert")
print("\n  Converting...", flush=True)
skipped_resume = 0
for tensor_name, shape, sf_path in tqdm(all_tensors, desc="  Processing"):
    bits, method = get_bits_and_method(tensor_name)

    if done_keys and is_already_done(tensor_name, method):
        skipped_resume += 1
        if method == "mxtq":
            total_mxtq += 1
        elif method == "affine":
            total_affine += 1
        else:
            total_passthrough += 1
        continue

    # Load tensor (Hy3 ships bf16 only; fall back via _load_bf16_tensor if safetensors fails)
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
        add_tensor(tensor_name, tensor.astype(np.float16))
        total_passthrough += 1

    elif method == "affine":
        w = mx.array(tensor.astype(np.float16))
        qw, qs, qb = mx.quantize(w, group_size=64, bits=bits)
        base = tensor_name.replace(".weight", "") if tensor_name.endswith(".weight") else tensor_name
        add_tensor(f"{base}.weight", np.array(qw))
        add_tensor(f"{base}.scales", np.array(qs).astype(np.float16))
        add_tensor(f"{base}.biases", np.array(qb).astype(np.float16))
        total_affine += 1
        del w, qw, qs, qb

    elif method == "mxtq":
        result = tq_quantize_weight(tensor, bits=bits, seed=SEED)
        base = tensor_name.replace(".weight", "") if tensor_name.endswith(".weight") else tensor_name
        add_tensor(f"{base}.tq_packed", result["packed"])
        add_tensor(f"{base}.tq_norms", result["norms"])
        add_tensor(f"{base}.tq_bits", np.array([bits], dtype=np.uint8))
        total_mxtq += 1

    del tensor
    if (total_mxtq + total_affine) % 200 == 0:
        gc.collect()

flush_shard()

if skipped_resume:
    print(f"\n  Resume: skipped {skipped_resume} tensors from previous run", flush=True)

# Rename shards
progress.phase(3, 3, "write")
print(f"\n  Renaming {shard_idx} shards...", flush=True)
for i in range(1, shard_idx + 1):
    old = OUT / f"model-{i:05d}-of-XXXXX.safetensors"
    new = OUT / f"model-{i:05d}-of-{shard_idx:05d}.safetensors"
    if old.exists():
        old.rename(new)
shard_map = {k: v.replace("XXXXX", f"{shard_idx:05d}") for k, v in shard_map.items()}

# total_size from index entries (computed after rename)
total_size = 0
for fname in sorted(set(shard_map.values())):
    p = OUT / fname
    if p.exists():
        total_size += p.stat().st_size

index = {
    "metadata": {"format": "jangtq", "total_size": total_size},
    "weight_map": shard_map,
}
with open(OUT / "model.safetensors.index.json", "w") as f:
    json.dump(index, f, indent=2)

# === Write config + jang_config ===
if _MIXED_PROJ_BITS is not None:
    routed_expert_bits = dict(_MIXED_PROJ_BITS)
    bits_default = 2
else:
    routed_expert_bits = EXPERT_BITS
    bits_default = EXPERT_BITS

mxtq_bits_top = {
    "routed_expert": routed_expert_bits,
    "attention": 8,
    "shared_expert": 8,
    "dense_ffn": 8,
    "mtp": 8,
    "embed_tokens": 8,
    "lm_head": 8,
    "norms_router_biases": 16,
}

config_out = dict(config)
config_out.pop("quantization_config", None)
config_out["weight_format"] = "mxtq"
config_out["mxtq_bits"] = mxtq_bits_top
config_out["mxtq_seed"] = SEED
config_out["quantization"] = {
    "bits": 8,
    "group_size": 64,
    "mode": "affine",
    "routed_expert_bits": routed_expert_bits,
    "mxtq_bits": mxtq_bits_top,
    "expert_layout": "per_expert",
}
config_out["capabilities"] = {
    "reasoning_parser": "qwen3",
    "tool_parser": "hunyuan",
    "think_in_template": False,
    "supports_tools": True,
    "supports_thinking": True,
    "family": "hy_v3",
    "modality": "text",
    "cache_type": "kv",
}
config_out["runtime"] = {
    "bundle_has_mtp": bool(n_mtp_layers),
    "mtp_layers": n_mtp_layers,
    "mtp_mode": "preserved_disabled",
    "mtp_status": (
        "MTP tensors are preserved in the bundle, but the first JANG runtime "
        "path must use normal autoregressive decode until an accept/reject "
        "speculative loop is implemented and tested."
    ),
}
with open(OUT / "config.json", "w") as f:
    json.dump(config_out, f, indent=2)

jang_config = {
    "version": 2,
    "weight_format": "mxtq",
    "profile": PROFILE,
    "cache_subtype": "kv",
    "source_model": {
        "name": "Hy3-preview",
        "org": "tencent",
        "architecture": "hy_v3",
    },
    "expert_layout": "per_expert",
    "mxtq_seed": SEED,
    "mxtq_bits": mxtq_bits_top,
    "quantization": {
        "method": "affine+mxtq",
        "group_size": 64,
        "bits_default": bits_default,
    },
    "runtime": config_out["runtime"],
    "bundle_has_mtp": bool(n_mtp_layers),
    "mtp_layers": n_mtp_layers,
    "capabilities": config_out["capabilities"],
    "chat": {
        "reasoning": {
            "supported": True,
            "parser": "qwen3",
            "default_mode": "no_think",
            "modes": ["no_think", "low", "high"],
        },
        "tool_calling": {
            "supported": True,
            "parser": "hunyuan",
        },
        "sampling_defaults": HY3_CHAT_SAMPLING_DEFAULTS,
    },
}
with open(OUT / "jang_config.json", "w") as f:
    json.dump(jang_config, f, indent=2)

# Sidecar tokenizer / chat template files
SIDECARS = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "chat_template.jinja",
    "chat_template.json",
    "merges.txt",
    "vocab.json",
]
import shutil
for fname in SIDECARS:
    p = SRC / fname
    if p.exists():
        shutil.copy2(str(p), str(OUT / fname))

gen_cfg_path = OUT / "generation_config.json"
try:
    if gen_cfg_path.exists():
        with open(gen_cfg_path) as f:
            gen_cfg = json.load(f)
            if not isinstance(gen_cfg, dict):
                gen_cfg = {}
    else:
        gen_cfg = {}
    gen_cfg.update(HY3_GENERATION_CONFIG_OVERRIDES)
    with open(gen_cfg_path, "w") as f:
        json.dump(gen_cfg, f, indent=2)
except Exception as exc:
    raise SystemExit(
        f"failed to patch Hy3 generation_config.json sampling defaults: {exc}"
    ) from exc

# Inline chat_template into tokenizer_config if jinja exists but tokenizer_config.chat_template is empty
tok_cfg = OUT / "tokenizer_config.json"
template = OUT / "chat_template.jinja"
if tok_cfg.exists() and template.exists():
    with open(tok_cfg) as f:
        tc = json.load(f)
    if not tc.get("chat_template"):
        tc["chat_template"] = template.read_text(encoding="utf-8")
        with open(tok_cfg, "w") as f:
            json.dump(tc, f, indent=2, ensure_ascii=False)

# Build JANGTQ runtime sidecar
print("\n  Building jangtq_runtime.safetensors sidecar...", flush=True)
try:
    from jang_tools.build_jangtq_sidecar import main as build_sidecar
    saved_argv = list(sys.argv)
    sys.argv = ["build_jangtq_sidecar", str(OUT)]
    try:
        build_sidecar()
    finally:
        sys.argv = saved_argv
except (Exception, SystemExit) as exc:
    print(f"  [sidecar] failed: {exc}")
    print(f"  Run manually: python3 -m jang_tools.build_jangtq_sidecar {OUT}")

# Verifier
try:
    from jang_tools.capabilities import verify_directory
    ok, msg = verify_directory(OUT)
    print(f"  verify: ok={ok}  msg={msg}")
    if not ok:
        raise SystemExit(f"capabilities verify failed: {msg}")
except Exception as exc:
    print(f"  [verify] {type(exc).__name__}: {exc}")

print("\n  Done!", flush=True)
print(f"  MXTQ expert tensors: {total_mxtq}", flush=True)
print(f"  Affine tensors:      {total_affine}", flush=True)
print(f"  Passthrough tensors: {total_passthrough}", flush=True)
print(f"  Output:              {OUT}", flush=True)
progress.done(ok=True, output=str(OUT))
