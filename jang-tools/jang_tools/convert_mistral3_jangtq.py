"""
Mistral-Medium-3.5-128B (mistral3 + ministral3 + pixtral VL) → JANGTQ
Created by Jinho Jang (eric@jangq.ai)

Source format: FP8 e4m3 with PER-TENSOR fp32 scales (weight_block_size: null).
On-disk pairs:
    *.weight          : float8_e4m3fn (out, in)
    *.weight_scale    : float32       (1,)
Modules in `quantization_config.modules_to_not_convert` (vision_tower,
multi_modal_projector, lm_head) are bf16 already — pass through.

Quant strategy:
  - text decoder linears (q/k/v/o + gate/up/down): MXTQ at profile bits
    (after FP8 → bf16 dequant)
  - embed_tokens: affine 8-bit
  - lm_head: PASSTHROUGH bf16 (matches upstream ignored)
  - vision_tower.*, multi_modal_projector.*: PASSTHROUGH bf16
  - all RMSNorms: PASSTHROUGH fp16

Bake invariants:
  - mxtq_bits, routed_expert_bits = profile bits
  - rope_parameters left as-is (already YaRN nested)
  - chat_template + tekken.json carried through

Usage:
  python -m jang_tools.convert_mistral3_jangtq <SRC> <OUT> [PROFILE]
  Profiles: JANGTQ2 (default), JANGTQ3, JANGTQ4
"""
import sys, json, gc, shutil, re
from pathlib import Path

import numpy as np
import mlx.core as mx
import torch  # FP8 decode
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
from jang_tools.turboquant.linear import tq_quantize_weight


SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else None
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else None
PROFILE = sys.argv[3].upper() if len(sys.argv) > 3 else "JANGTQ"
SEED = 42
if SRC is None or OUT is None:
    print(__doc__); sys.exit(1)

_BITS = {"JANGTQ": 2, "JANGTQ2": 2, "JANGTQ3": 3, "JANGTQ4": 4}
if PROFILE not in _BITS:
    raise ValueError(f"unknown profile {PROFILE}")
BITS = _BITS[PROFILE]


def fp8_e4m3_to_fp32(u8: np.ndarray) -> np.ndarray:
    return torch.from_numpy(u8.view(np.uint8)).view(torch.float8_e4m3fn).float().numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((SRC / "config.json").read_text())
    qc = cfg.get("quantization_config") or {}
    ignored = set(qc.get("modules_to_not_convert") or
                  ("model.vision_tower", "model.multi_modal_projector", "lm_head"))

    idx = json.loads((SRC / "model.safetensors.index.json").read_text())
    weight_map = idx["weight_map"]
    open_cache = {}
    def get(k):
        sh = weight_map[k]
        if sh not in open_cache: open_cache[sh] = safe_open(str(SRC / sh), framework="pt")
        return _mlx_to_np(open_cache[sh].get_tensor(k))

    shard_idx = 0; tensors = {}; sb = 0
    MAX = 1_000_000_000; smap = {}

    def flush():
        nonlocal shard_idx, tensors, sb
        if not tensors: return
        shard_idx += 1
        fn = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        save_file(tensors, str(OUT / fn))
        for k in tensors: smap[k] = fn
        print(f"  shard {shard_idx}: {len(tensors)} tensors, {sb/1e9:.2f} GB", flush=True)
        tensors = {}; sb = 0

    def add(name, arr):
        nonlocal sb
        tensors[name] = arr; sb += arr.nbytes
        if sb >= MAX: flush()

    def affine_quantize(arr, bits=8, group=64):
        t = mx.array(arr.astype(np.float32))
        qw, sc, bi = mx.quantize(t, bits=bits, group_size=group)
        mx.eval(qw, sc, bi)
        return (np.asarray(qw, dtype=np.uint32),
                np.asarray(sc).astype(np.float16),
                np.asarray(bi).astype(np.float16))

    def is_ignored(key: str) -> bool:
        base = key.rsplit(".weight", 1)[0]
        return any(base == ig or base.startswith(ig + ".") for ig in ignored)

    # Walk all weight keys; skip *_scale entries (we pair them in)
    all_keys = sorted(weight_map.keys())
    weight_keys = [k for k in all_keys if not k.endswith(".weight_scale") and not k.endswith(".weight_scale_inv") and not k.endswith(".activation_scale")]

    for key in tqdm(weight_keys, desc="convert"):
        # Norms / 1-D / non-decoder scalars → passthrough fp16
        if key.endswith("norm.weight") or key.endswith(".bias"):
            arr = get(key)
            add(key, arr.astype(np.float16) if arr.dtype != np.float16 else arr)
            continue

        if is_ignored(key):
            arr = get(key)
            # vision_tower + projector + lm_head: bf16 → fp16 passthrough
            add(key, arr.astype(np.float16) if arr.dtype != np.float16 else arr)
            continue

        # Embed tokens: affine 8-bit
        if "embed_tokens" in key and key.endswith(".weight"):
            qw, sc, bi = affine_quantize(get(key), bits=8, group=64)
            base = key.rsplit(".weight", 1)[0]
            add(base + ".weight", qw); add(base + ".scales", sc); add(base + ".biases", bi)
            continue

        # Text decoder linear → MXTQ
        if key.endswith(".weight"):
            # Mistral uses .weight_scale_inv for per-tensor dequant scale
            scale_inv_key = key.replace(".weight", ".weight_scale_inv")
            scale_alt_key  = key.replace(".weight", ".weight_scale")
            scale_key = scale_inv_key if scale_inv_key in (weight_map if "weight_map" in dir() else wm) else scale_alt_key
            arr_u8 = get(key)
            if scale_key in weight_map:
                # FP8 dequant (per-tensor scale)
                if arr_u8.dtype != np.uint8:
                    arr_u8 = arr_u8.view(np.uint8)
                w_fp32 = fp8_e4m3_to_fp32(arr_u8)
                scale = float(get(scale_key).reshape(-1)[0])
                arr = (w_fp32 * scale).astype(np.float32)
            else:
                # Already bf16 / fp16
                arr = arr_u8.astype(np.float32)
            base = key.rsplit(".weight", 1)[0]
            res = tq_quantize_weight(arr, bits=BITS, seed=SEED)
            add(base + ".tq_packed", res["packed"])
            add(base + ".tq_norms",  res["norms"])
            add(base + ".tq_bits",   np.array([BITS], dtype=np.uint8))
            del arr
            gc.collect()
            continue

        # Anything else: passthrough
        add(key, get(key).astype(np.float16))

    flush()

    for i in range(1, shard_idx + 1):
        (OUT / f"model-{i:05d}-of-XXXXX.safetensors").rename(
            OUT / f"model-{i:05d}-of-{shard_idx:05d}.safetensors")
    fixed = {k: v.replace("XXXXX", f"{shard_idx:05d}") for k, v in smap.items()}
    total = sum((OUT / fn).stat().st_size for fn in set(fixed.values()))
    (OUT / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"format": "jangtq", "total_size": total}, "weight_map": fixed},
        indent=2))

    cfg.pop("quantization_config", None)
    cfg["quantization"] = {"group_size": 64, "bits": 8}
    cfg["weight_format"] = "mxtq"
    cfg["mxtq_seed"] = SEED
    cfg["mxtq_bits"] = BITS
    cfg["routed_expert_bits"] = BITS  # invariant required even for dense (Swift fallback)
    (OUT / "config.json").write_text(json.dumps(cfg, indent=2))

    for name in ("chat_template.jinja", "SYSTEM_PROMPT.txt", "generation_config.json",
                 "processor_config.json", "params.json",
                 "tokenizer.json", "tokenizer_config.json", "tekken.json", "LICENSE"):
        if (SRC / name).exists(): shutil.copy(SRC / name, OUT / name)

    (OUT / "jang_config.json").write_text(json.dumps({
        "version": 2,
        "weight_format": "mxtq",
        "profile": PROFILE,
        "source_model": {"name": SRC.name, "architecture": "mistral3"},
        "has_vision": True,
        "has_audio": False,
        "has_video": False,
        "vision_arch": "pixtral",
        "mxtq_seed": SEED,
        "mxtq_bits": {
            "text_decoder": BITS,
            "embed_tokens": 8,
            "vision_tower": "passthrough_fp16",
            "multi_modal_projector": "passthrough_fp16",
            "lm_head": "passthrough_fp16",
        },
    }, indent=2))

    print(f"DONE. shards={shard_idx}, total={total/1e9:.2f} GB → {OUT}")


if __name__ == "__main__":
    main()
