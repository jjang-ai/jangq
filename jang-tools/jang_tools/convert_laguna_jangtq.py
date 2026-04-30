"""
poolside Laguna-XS.2 → JANGTQ Conversion
Created by Jinho Jang (eric@jangq.ai)

Architecture: model_type=laguna. Hybrid SWA(window=512)+full, per-layer head
count (48 full / 64 SWA), dual RoPE (full=YaRN/swa=default), 256 routed
experts top-8 + 1 shared, layer 0 dense MLP, q_norm/k_norm + per-head
gating (g_proj). bf16 source, NOT FP8.

This converter:
  - reads the per-expert weight files (NOT pre-stacked) from HF and STACKS
    at convert time into (n_experts, 2*inter, hidden) for gate+up + (n_experts,
    hidden, inter) for down → tq_quantize_experts
  - keeps shared_expert + attention + embed/lm_head as affine 8-bit
  - keeps norms + e_score_correction_bias + mlp.gate.weight as fp16
  - keeps layer-0 dense MLP at affine 8-bit
  - bakes mxtq_bits / routed_expert_bits / weight_format into config.json
    (2026-04-25 invariant)

Output keys (canonical, matches Python jang_tools.laguna runtime):
  model.embed_tokens.{weight,scales,biases}                (affine 8-bit)
  model.layers.0.self_attn.{q,k,v,o,g}_proj.{...}           (affine 8-bit)
  model.layers.0.self_attn.{q,k}_norm.weight                (passthrough fp16)
  model.layers.0.mlp.{gate,up,down}_proj.{...}              (affine 8-bit; layer-0 dense)
  model.layers.0.{input,post_attention}_layernorm.weight    (passthrough)
  model.layers.L.self_attn.* (L>=1, same as layer 0)
  model.layers.L.mlp.gate.weight                            (passthrough fp16)
  model.layers.L.mlp.experts.e_score_correction_bias        (passthrough fp16)
  model.layers.L.mlp.experts.gate_up_proj.{tq_packed,tq_norms,tq_bits}  (MXTQ, stacked)
  model.layers.L.mlp.experts.down_proj.{tq_packed,tq_norms,tq_bits}     (MXTQ, stacked)
  model.layers.L.mlp.shared_expert.{...}                    (affine 8-bit)
  model.norm.weight                                          (passthrough)
  lm_head.{weight,scales,biases}                            (affine 8-bit)

Usage:
  python -m jang_tools.convert_laguna_jangtq <SRC> <OUT> [PROFILE]

Profiles: JANGTQ2 (default), JANGTQ3, JANGTQ4
"""
import sys, json, gc, shutil, re
import argparse
from pathlib import Path

import numpy as np
import mlx.core as mx
from tqdm import tqdm
from safetensors import safe_open
from safetensors.numpy import save_file


def _mlx_to_np(t):
    """Convert a torch.Tensor (from safe_open framework='pt') to numpy.
    bf16/f16 -> fp32; uint8 / int / float passthrough."""
    import torch as _torch, numpy as _np
    if not hasattr(t, "dtype"):
        return _np.asarray(t)
    d = t.dtype
    if d in (_torch.bfloat16, _torch.float16):
        return t.to(_torch.float32).numpy()
    if d == _torch.float8_e4m3fn:
        # Keep raw bytes as uint8; downstream FP8 dequant uses .view(np.uint8)
        return t.view(_torch.uint8).numpy()
    return t.numpy()
from jang_tools.turboquant.linear import tq_quantize_weight, tq_quantize_experts


_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--progress", choices=["json", "off"], default="off")
_ap.add_argument("--quiet-text", action="store_true")
_args, _rest = _ap.parse_known_args()
sys.argv = [sys.argv[0]] + _rest

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else None
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else None
PROFILE = sys.argv[3].upper() if len(sys.argv) > 3 else "JANGTQ2"
SEED = 42

if SRC is None or OUT is None:
    print(__doc__); sys.exit(1)

_BITS = {"JANGTQ2": 2, "JANGTQ3": 3, "JANGTQ4": 4}
if PROFILE not in _BITS:
    raise ValueError(f"unknown profile {PROFILE}; expected {sorted(_BITS)}")
EXPERT_BITS = _BITS[PROFILE]


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    config = json.loads((SRC / "config.json").read_text())
    n_layers = config["num_hidden_layers"]
    n_experts = config["num_experts"]
    layer_types = config["layer_types"]
    mlp_layer_types = config["mlp_layer_types"]
    print(f"== Laguna-XS.2 → {PROFILE} ==")
    print(f"  layers={n_layers}, experts={n_experts}, profile={PROFILE} (expert={EXPERT_BITS}-bit)")

    # ---- index source tensors -----------------------------------------------
    idx = json.loads((SRC / "model.safetensors.index.json").read_text())
    weight_map = idx["weight_map"]                         # key -> shard
    by_shard = {}
    for k, sh in weight_map.items():
        by_shard.setdefault(sh, []).append(k)

    # cache of opened safe_opens
    open_cache = {}

    def get_tensor(key: str) -> np.ndarray:
        sh = weight_map[key]
        if sh not in open_cache:
            open_cache[sh] = safe_open(str(SRC / sh), framework="pt")
        return _mlx_to_np(open_cache[sh].get_tensor(key))

    # ---- shard writer -------------------------------------------------------
    shard_idx = 0
    shard_tensors: dict = {}
    shard_bytes = 0
    MAX_SHARD = 1_000_000_000
    shard_map: dict = {}

    def flush():
        nonlocal shard_idx, shard_tensors, shard_bytes
        if not shard_tensors: return
        shard_idx += 1
        fname = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        save_file(shard_tensors, str(OUT / fname))
        for k in shard_tensors: shard_map[k] = fname
        print(f"    shard {shard_idx}: {len(shard_tensors)} tensors, {shard_bytes/1e9:.2f} GB", flush=True)
        shard_tensors = {}; shard_bytes = 0

    def add(name: str, arr: np.ndarray):
        nonlocal shard_bytes
        shard_tensors[name] = arr
        shard_bytes += arr.nbytes
        if shard_bytes >= MAX_SHARD: flush()

    # ---- helpers ------------------------------------------------------------
    def affine_quantize(arr: np.ndarray, bits: int = 8, group: int = 64):
        t = mx.array(arr.astype(np.float32))
        qw, scales, biases = mx.quantize(t, bits=bits, group_size=group)
        mx.eval(qw, scales, biases)
        return (np.asarray(qw, dtype=np.uint32),
                np.asarray(scales).astype(np.float16),
                np.asarray(biases).astype(np.float16))

    def add_affine(out_name: str, arr: np.ndarray, bits: int = 8):
        qw, sc, bi = affine_quantize(arr, bits=bits)
        add(out_name + ".weight", qw)
        add(out_name + ".scales", sc)
        add(out_name + ".biases", bi)

    def add_mxtq_2d(out_name: str, arr: np.ndarray, bits: int):
        res = tq_quantize_weight(arr, bits=bits, seed=SEED)
        add(out_name + ".tq_packed", res["packed"])
        add(out_name + ".tq_norms",  res["norms"])
        add(out_name + ".tq_bits",   np.array([bits], dtype=np.uint8))

    def add_mxtq_3d_stacked(out_name: str, stacked: np.ndarray, bits: int):
        res = tq_quantize_experts(stacked, bits=bits, seed=SEED)
        add(out_name + ".tq_packed", res["packed"])
        add(out_name + ".tq_norms",  res["norms"])
        add(out_name + ".tq_bits",   np.array([bits], dtype=np.uint8))

    # ---- pass 1: process non-expert tensors ---------------------------------
    expert_keys: set = set()
    other_keys: list = []
    for k in weight_map.keys():
        if re.match(r"^model\.layers\.\d+\.mlp\.experts\.\d+\.", k):
            expert_keys.add(k)
        else:
            other_keys.append(k)
    print(f"  per-expert keys: {len(expert_keys)}; other: {len(other_keys)}")

    for k in tqdm(sorted(other_keys), desc="non-expert"):
        arr = get_tensor(k).astype(np.float32) if not k.endswith(".weight") else get_tensor(k)
        # Decide method
        bits, method = decide_method(k)
        if method == "skip":
            continue
        if method == "passthrough":
            add(k, get_tensor(k).astype(np.float16) if get_tensor(k).dtype == np.float32 or
                str(get_tensor(k).dtype).endswith("bfloat16") else get_tensor(k))
            continue
        if method == "affine":
            add_affine(k.rsplit(".weight", 1)[0], get_tensor(k), bits=bits)
            continue
        raise AssertionError(f"unhandled method {method} for {k}")

    # ---- pass 2: stack + quantize routed experts ----------------------------
    # group by (layer, proj-name) -> list of expert idx -> tensor
    re_expert = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(\w+)_proj\.weight$")
    by_layer: dict = {}
    for k in expert_keys:
        m = re_expert.match(k)
        if not m: continue
        L, e, p = int(m.group(1)), int(m.group(2)), m.group(3)
        by_layer.setdefault((L, p), {})[e] = k

    for (L, p), keys in tqdm(sorted(by_layer.items()), desc="stack-experts"):
        n_e = len(keys)
        if n_e != n_experts:
            raise ValueError(f"layer {L} {p}: got {n_e} expert tensors, expected {n_experts}")
        first = get_tensor(keys[0])
        stacked = np.empty((n_experts, *first.shape), dtype=first.dtype)
        for e, k in keys.items():
            stacked[e] = get_tensor(k)
        # Combine gate + up into gate_up at the stacking step (matches modeling_laguna.py
        # LagunaExperts.gate_up_proj which is (n_experts, 2*inter, hidden))
        if p == "gate":
            gate_stack = stacked
            continue  # wait for "up"
        if p == "up":
            up_stack = stacked
            gate_up = np.concatenate([gate_stack, up_stack], axis=1)  # (n_e, 2*inter, hidden)
            out_name = f"model.layers.{L}.mlp.experts.gate_up_proj"
            add_mxtq_3d_stacked(out_name, gate_up, EXPERT_BITS)
            del gate_stack, up_stack, gate_up
            gc.collect()
            continue
        if p == "down":
            out_name = f"model.layers.{L}.mlp.experts.down_proj"
            add_mxtq_3d_stacked(out_name, stacked, EXPERT_BITS)
            del stacked
            gc.collect()
            continue
        raise AssertionError(f"unknown proj {p}")

    flush()

    # ---- rename shards + index ----------------------------------------------
    for i in range(1, shard_idx + 1):
        old = OUT / f"model-{i:05d}-of-XXXXX.safetensors"
        new = OUT / f"model-{i:05d}-of-{shard_idx:05d}.safetensors"
        if old.exists(): old.rename(new)
    fixed_map = {k: v.replace("XXXXX", f"{shard_idx:05d}") for k, v in shard_map.items()}
    total = sum((OUT / fn).stat().st_size for fn in set(fixed_map.values()))
    (OUT / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"format": "jangtq", "total_size": total}, "weight_map": fixed_map},
        indent=2,
    ))

    # ---- write config.json with invariants ---------------------------------
    config.pop("quantization_config", None)
    config["quantization"] = {"group_size": 64, "bits": 8}
    config["weight_format"] = "mxtq"
    config["mxtq_seed"] = SEED
    config["mxtq_bits"] = EXPERT_BITS
    config["routed_expert_bits"] = EXPERT_BITS
    (OUT / "config.json").write_text(json.dumps(config, indent=2))

    # ---- copy ancillary files (chat_template, tokenizer, modeling_laguna) ---
    for name in ("chat_template.jinja", "generation_config.json",
                 "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "configuration_laguna.py", "modeling_laguna.py", "LICENSE.md"):
        src = SRC / name
        if src.exists():
            shutil.copy(src, OUT / name)

    # ---- jang_config.json --------------------------------------------------
    (OUT / "jang_config.json").write_text(json.dumps({
        "version": 2,
        "weight_format": "mxtq",
        "profile": PROFILE,
        "source_model": {"name": SRC.name, "architecture": "laguna"},
        "has_vision": False,
        "has_audio": False,
        "has_video": False,
        "mxtq_seed": SEED,
        "mxtq_bits": {
            "attention": 8,
            "shared_expert": 8,
            "routed_expert": EXPERT_BITS,
            "embed_lm_head": 8,
        },
    }, indent=2))

    print(f"DONE. shards={shard_idx}, total={total/1e9:.2f} GB → {OUT}")


def decide_method(key: str):
    """Return (bits, method). method ∈ {affine, passthrough, skip}.

    Routed expert keys (per-expert, not pre-stacked) are handled in pass 2.
    """
    if key.endswith("e_score_correction_bias"): return (16, "passthrough")
    if key.endswith("norm.weight"):              return (16, "passthrough")
    if key.endswith(".mlp.gate.weight"):
        if "gate_proj" not in key: return (16, "passthrough")
    if "embed_tokens" in key:                    return (8, "affine")
    if key == "lm_head.weight":                  return (8, "affine")
    if ".self_attn." in key and key.endswith("_proj.weight"): return (8, "affine")
    if ".self_attn." in key and (key.endswith("q_norm.weight") or key.endswith("k_norm.weight")):
        return (16, "passthrough")
    if ".shared_expert." in key and key.endswith("_proj.weight"): return (8, "affine")
    # Layer 0 dense MLP: model.layers.0.mlp.{gate,up,down}_proj.weight
    if re.match(r"^model\.layers\.0\.mlp\.(gate|up|down)_proj\.weight$", key):
        return (8, "affine")
    # Default: any other 1-D / scalar tensor → passthrough
    return (16, "passthrough")


if __name__ == "__main__":
    main()
