"""Nanbeige 4.x (Looped Transformer) → MXFP8 / JANG_6M / JANG_4M.

Nanbeige4.2-3B is a **dense** Llama-shaped decoder (GQA 48q/8kv, head_dim 128,
SwiGLU, RMSNorm, NeoX half-rotation RoPE θ=7e7) whose 22-layer stack is executed
`num_loops` (=2) times over shared weights.  The loop is a *runtime* property —
it changes nothing about which tensors exist or how they quantize — so
conversion is a straight dense affine/MXFP pass.  What conversion MUST do is
carry the loop contract forward into the bundle metadata so the Swift/Python
runtimes size the KV cache correctly (see `jang_runtime` block below).

Profiles
--------
  MXFP8    every 2-D weight at MXFP8 (mx.quantize mode="mxfp8"), norms fp16.
  JANG_6M  attention (q/k/v/o) 8-bit affine, MLP 6-bit, embed + lm_head 6-bit.
  JANG_4M  attention (q/k/v/o) 8-bit affine, MLP 4-bit, embed + lm_head 4-bit.

Dense caveats that DO apply here (see memory `project_dense_vs_moe`):
JANG's low-bit profiles (2S/2L/3x) must NOT be used on this model — a dense
model has one MLP per layer and no expert redundancy.  4M and 6M are the two
dense-safe tiers and are the only ones this converter accepts.

Norm convention: **plain Llama RMSNorm, NO +1 shift.**  `NanbeigeRMSNorm` is
`weight * x / rms(x)` — identical to `mlx.nn.RMSNorm`.  Adding the Gemma/Qwen-style
+1 here produces garbage.  Norm tensors are copied through unchanged.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from jang_tools.capabilities import build_capabilities
from jang_tools.progress import ProgressEmitter

MAX_SHARD = 4_500_000_000

# (attention, embed/lm_head, mlp) — mirrors allocate.TIER_RULES
# (CRITICAL, IMPORTANT, COMPRESS) for the two dense-safe JANG tiers.
PROFILES: dict[str, dict] = {
    "MXFP8":   {"mode": "mxfp8",  "attn": 8, "embed": 8, "mlp": 8, "top": 8},
    "JANG_6M": {"mode": "affine", "attn": 8, "embed": 6, "mlp": 6, "top": 8},
    "JANG_4M": {"mode": "affine", "attn": 8, "embed": 4, "mlp": 4, "top": 8},
}

# Files copied verbatim from source. `modeling_nanbeige.py` /
# `configuration_nanbeige.py` are deliberately NOT copied: the bundle is an MLX
# artifact, the tokenizer is a plain LlamaTokenizer with no auto_map, and
# shipping remote code would make every load prompt for trust_remote_code.
SIDECAR_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "generation_config.json",
)


def tier_bits(tensor_name: str, profile: dict) -> int | None:
    """Bit width for a tensor, or None for fp16 passthrough."""
    if "norm" in tensor_name.lower() or tensor_name.endswith(".bias"):
        return None
    if len(tensor_name.split(".")) == 1:
        return None
    if ".self_attn." in tensor_name and tensor_name.endswith(
        ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight")
    ):
        return profile["attn"]
    if tensor_name.endswith(("embed_tokens.weight", "lm_head.weight")):
        return profile["embed"]
    return profile["mlp"]


def _scan_source(src: Path) -> list[tuple[str, list[int], Path]]:
    index_path = src / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        files = {src / f for f in weight_map.values()}
    else:
        files = set(src.glob("*.safetensors"))
    out: list[tuple[str, list[int], Path]] = []
    for f in sorted(files):
        with safe_open(str(f), framework="numpy") as sf:
            for key in sf.keys():
                out.append((key, list(sf.get_slice(key).get_shape()), f))
    return sorted(out)


def _load_tensor(sf_path: Path, name: str, shape: list[int]) -> np.ndarray:
    try:
        with safe_open(str(sf_path), framework="numpy") as f:
            t = f.get_tensor(name)
        if not isinstance(t, np.ndarray):
            t = np.array(t)
    except TypeError:
        # numpy has no bfloat16 — read raw and widen to fp32.
        from jang_tools.calibrate import _load_bf16_tensor

        t = _load_bf16_tensor(sf_path, name, tuple(shape))
    return t.astype(np.float32) if t.dtype != np.float32 else t


def _quantize(tensor: np.ndarray, *, bits: int, group_size: int, mode: str):
    """Chunked mx.quantize. Returns (weight, scales, biases|None)."""
    qw, qs, qb = [], [], []
    chunk_rows = max(1, min(tensor.shape[0], 100_000_000 // max(1, tensor.shape[1])))
    for start in range(0, tensor.shape[0], chunk_rows):
        chunk = mx.array(tensor[start : start + chunk_rows].astype(np.float16))
        packed = mx.quantize(chunk, group_size=group_size, bits=bits, mode=mode)
        w, s = packed[0], packed[1]
        b = packed[2] if len(packed) > 2 else None
        mx.eval(w, s, *([] if b is None else [b]))
        qw.append(np.array(w))
        qs.append(np.array(s))
        if b is not None:
            qb.append(np.array(b))
        del chunk, packed, w, s, b
    return (
        np.concatenate(qw, axis=0),
        np.concatenate(qs, axis=0),
        np.concatenate(qb, axis=0) if qb else None,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Nanbeige 4.x source to MXFP8 / JANG_6M / JANG_4M.")
    p.add_argument("src", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--profile", choices=sorted(PROFILES), required=True)
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quiet-text", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gs = args.group_size
    profile_name = args.profile
    profile = PROFILES[profile_name]
    mode = profile["mode"]
    weight_format = "mxfp8" if mode == "mxfp8" else "jang_affine"
    progress = ProgressEmitter(json_to_stderr=False, quiet_text=args.quiet_text)

    src, out = args.src.expanduser(), args.out.expanduser()
    config = json.loads((src / "config.json").read_text(encoding="utf-8"))

    if config.get("model_type") != "nanbeige":
        raise SystemExit(f"expected model_type=nanbeige, got {config.get('model_type')!r}")

    n_layers = int(config["num_hidden_layers"])
    num_loops = int(config.get("num_loops", 1)) or 1
    if config.get("loop_loss_weights"):
        num_loops = len(config["loop_loss_weights"]) + 1
    cache_slots = n_layers * num_loops

    # Architecture features this converter/runtime pair does not implement.
    for key in ("enable_double_loop_split", "enable_hyper_connection", "enable_mhc",
                "enable_depth_attention", "loop_share_kv", "qk_layernorm"):
        if config.get(key):
            raise SystemExit(f"source enables {key}=True — unsupported; port the runtime first")
    if config.get("emb_neighbor_num") is not None:
        raise SystemExit("source enables n-gram embeddings — unsupported; port the runtime first")

    print("=" * 72)
    print(f"  Nanbeige (Looped Transformer) -> {profile_name}")
    print("=" * 72)
    print(f"  Source:      {src}")
    print(f"  Output:      {out}")
    print(f"  Layers:      {n_layers}   num_loops: {num_loops}   CACHE SLOTS: {cache_slots}")
    print(f"  Heads:       {config['num_attention_heads']}q / {config['num_key_value_heads']}kv "
          f"x head_dim {config['head_dim']}  (n_heads*head_dim != hidden_size)")
    print(f"  RoPE theta:  {config['rope_theta']:.0f}   max_pos: {config['max_position_embeddings']}")
    print(f"  Bits:        attn={profile['attn']} mlp={profile['mlp']} embed/lm_head={profile['embed']} "
          f"mode={mode} group_size={gs}")
    print(f"  Norm:        plain RMSNorm, NO +1 shift (fp16 passthrough)")

    progress.phase(1, 3, "scan")
    tensors = _scan_source(src)
    print(f"  Found {len(tensors)} tensors")

    if args.dry_run:
        counts: dict[str, int] = {}
        for name, _shape, _f in tensors:
            b = tier_bits(name, profile)
            k = "fp16" if b is None else f"{mode}-{b}"
            counts[k] = counts.get(k, 0) + 1
        print(json.dumps(counts, indent=2, sort_keys=True))
        progress.done(ok=True, output="dry-run")
        return

    out.mkdir(parents=True, exist_ok=True)
    for stale in list(out.glob("*.safetensors")) + list(out.glob("model.safetensors.index.json")):
        stale.unlink()

    shard_idx = 0
    shard_tensors: dict[str, np.ndarray] = {}
    shard_bytes = 0
    shard_map: dict[str, str] = {}
    overrides: dict[str, dict] = {}
    n_quant = n_pass = 0

    def flush_shard() -> None:
        nonlocal shard_idx, shard_tensors, shard_bytes
        if not shard_tensors:
            return
        shard_idx += 1
        fname = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        save_file(shard_tensors, str(out / fname))
        for k in shard_tensors:
            shard_map[k] = fname
        print(f"    Shard {shard_idx}: {len(shard_tensors)} tensors, {shard_bytes / 1e9:.2f} GB")
        shard_tensors, shard_bytes = {}, 0

    def add_tensor(name: str, arr: np.ndarray) -> None:
        nonlocal shard_bytes
        shard_tensors[name] = arr
        shard_bytes += arr.nbytes
        if shard_bytes >= MAX_SHARD:
            flush_shard()

    progress.phase(2, 3, "convert")
    from tqdm import tqdm

    for name, shape, sf_path in tqdm(tensors, desc="  Processing"):
        bits = tier_bits(name, profile)
        tensor = _load_tensor(sf_path, name, shape)
        if bits is None or tensor.ndim < 2:
            # Norms pass through untouched — NO +1 shift for nanbeige.
            add_tensor(name, tensor.astype(np.float16))
            n_pass += 1
        else:
            w, s, b = _quantize(tensor, bits=bits, group_size=gs, mode=mode)
            base = name[: -len(".weight")]
            add_tensor(f"{base}.weight", w)
            add_tensor(f"{base}.scales", s)
            if b is not None:
                add_tensor(f"{base}.biases", b)
            if bits != profile["top"]:
                overrides[base] = {"group_size": gs, "bits": bits, "mode": mode}
            n_quant += 1
            del w, s, b
        del tensor
        if (n_quant + n_pass) % 40 == 0:
            gc.collect()
            mx.clear_cache()

    flush_shard()

    progress.phase(3, 3, "write")
    for idx in range(1, shard_idx + 1):
        old = out / f"model-{idx:05d}-of-XXXXX.safetensors"
        if old.exists():
            old.rename(out / f"model-{idx:05d}-of-{shard_idx:05d}.safetensors")
    shard_map = {k: v.replace("XXXXX", f"{shard_idx:05d}") for k, v in shard_map.items()}
    total_size = sum((out / f).stat().st_size for f in set(shard_map.values()))
    (out / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"format": weight_format, "total_size": total_size}, "weight_map": shard_map},
            indent=2,
        ),
        encoding="utf-8",
    )

    for fname in SIDECAR_FILES:
        srcf = src / fname
        if srcf.exists():
            shutil.copy2(srcf, out / fname)

    # Nanbeige ships the chat template only inside tokenizer_config.json; the
    # Swift/jinja runtimes expect a standalone chat_template.jinja alongside it.
    tmpl_src = src / "chat_template.jinja"
    if tmpl_src.exists():
        shutil.copy2(tmpl_src, out / "chat_template.jinja")
    else:
        tok_cfg = json.loads((src / "tokenizer_config.json").read_text(encoding="utf-8"))
        template = tok_cfg.get("chat_template")
        if isinstance(template, str) and template.strip():
            (out / "chat_template.jinja").write_text(template, encoding="utf-8")
        else:
            raise SystemExit("no chat_template found in source tokenizer_config.json")

    config.pop("quantization_config", None)
    config.pop("auto_map", None)  # bundle ships no remote code
    config["weight_format"] = weight_format
    quant_block: dict = {"group_size": gs, "bits": profile["top"], "mode": mode}
    quant_block.update(overrides)
    config["quantization"] = quant_block

    # ── THE contract the runtimes must read ───────────────────────────────
    config["jang_runtime"] = {
        "architecture": "looped_transformer",
        "cache_layout": "looped_kv_v1",
        "num_loops": num_loops,
        "num_hidden_layers": n_layers,
        "cache_slots": cache_slots,
        "cache_slot_formula": "layer_idx + loop_idx * num_hidden_layers",
        "loop_final_norm": "skip" if config.get("skip_loop_final_norm") else "every_loop",
        "shared_layer_weights_across_loops": True,
        "position_ids_shared_across_loops": True,
        "norm_convention": "llama_rmsnorm_no_plus_one",
        "rope": {"type": "neox_half_rotation", "theta": config["rope_theta"],
                 "dims": config["head_dim"], "partial_rotary_factor": 1.0},
        "attention": {"type": "gqa", "n_heads": config["num_attention_heads"],
                      "n_kv_heads": config["num_key_value_heads"],
                      "head_dim": config["head_dim"],
                      "qkv_bias": bool(config.get("attention_bias")),
                      "qk_layernorm": bool(config.get("qk_layernorm")),
                      "n_heads_times_head_dim_equals_hidden": (
                          config["num_attention_heads"] * config["head_dim"] == config["hidden_size"]
                      )},
    }
    caps = build_capabilities({}, config, out)
    if caps is not None:
        config["capabilities"] = caps
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    jang_config = {
        "version": 2,
        "weight_format": weight_format,
        "profile": profile_name,
        "source_model": {"name": src.name, "architecture": "nanbeige"},
        "has_vision": False,
        "has_audio": False,
        "quantization": {
            "method": weight_format,
            "quantization_backend": "mx.quantize",
            "mode": mode,
            "group_size": gs,
            "tier_bits": {"attention": profile["attn"], "mlp": profile["mlp"],
                          "embed": profile["embed"]},
            "norm_convention": "llama_rmsnorm_no_plus_one",
            "per_module_override_count": len(overrides),
            "quantized_tensor_count": n_quant,
            "passthrough_tensor_count": n_pass,
        },
        "runtime": {
            "architecture": "looped_transformer",
            "num_loops": num_loops,
            "cache_slots": cache_slots,
            "total_weight_bytes": total_size,
            "total_weight_gb": round(total_size / (1024**3), 2),
        },
    }
    caps = build_capabilities(jang_config, config, out)
    if caps is not None:
        jang_config["capabilities"] = caps
    (out / "jang_config.json").write_text(json.dumps(jang_config, indent=2), encoding="utf-8")

    print(f"\n  Quantized {n_quant} tensors, passed through {n_pass}")
    print(f"  Per-module overrides: {len(overrides)}")
    print(f"  Total: {total_size / 1024**3:.2f} GB across {shard_idx} shard(s)")
    print(f"  cache_slots stamped: {cache_slots}")
    progress.done(ok=True, output=str(out))


if __name__ == "__main__":
    main()
