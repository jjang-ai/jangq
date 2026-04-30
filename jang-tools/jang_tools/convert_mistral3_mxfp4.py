"""
Mistral-Medium-3.5-128B → MXFP4 (mlx 4-bit affine, group_size=32)
Created by Jinho Jang (eric@jangq.ai)

Same FP8-source dequant as convert_mistral3_jangtq, but quantizes the text
decoder linears with mx.quantize bits=4 group=32 instead of MXTQ. Vision
tower + multi_modal_projector + lm_head stay bf16 (passthrough fp16).

Usage:
  python -m jang_tools.convert_mistral3_mxfp4 <SRC> <OUT> [bits] [group]
  defaults: bits=4, group=32
"""
import sys, json, gc, shutil
from pathlib import Path

import numpy as np
import mlx.core as mx
import torch
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

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else None
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else None
BITS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
GROUP = int(sys.argv[4]) if len(sys.argv) > 4 else 32
if SRC is None or OUT is None:
    print(__doc__); sys.exit(1)


def fp8_to_fp32(u8): return torch.from_numpy(u8.view(np.uint8)).view(torch.float8_e4m3fn).float().numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((SRC / "config.json").read_text())
    qc = cfg.get("quantization_config") or {}
    ignored = set(qc.get("modules_to_not_convert") or
                  ("model.vision_tower", "model.multi_modal_projector", "lm_head"))

    idx = json.loads((SRC / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    cache = {}
    def get(k):
        if wm[k] not in cache:
            cache[wm[k]] = safe_open(str(SRC / wm[k]), framework="pt")
        return _mlx_to_np(cache[wm[k]].get_tensor(k))

    si = 0; tensors = {}; sb = 0; MAX = 1_000_000_000; smap = {}
    def flush():
        nonlocal si, tensors, sb
        if not tensors: return
        si += 1
        fn = f"model-{si:05d}-of-XXXXX.safetensors"
        save_file(tensors, str(OUT / fn))
        for k in tensors: smap[k] = fn
        print(f"  shard {si}: {len(tensors)} tensors, {sb/1e9:.2f} GB", flush=True)
        tensors = {}; sb = 0
    def add(n, a):
        nonlocal sb
        tensors[n] = a; sb += a.nbytes
        if sb >= MAX: flush()

    def is_ignored(key):
        base = key.rsplit(".weight", 1)[0]
        return any(base == ig or base.startswith(ig + ".") for ig in ignored)

    all_k = sorted(wm.keys())
    weight_keys = [k for k in all_k if not k.endswith(".weight_scale") and not k.endswith(".weight_scale_inv") and not k.endswith(".activation_scale")]

    for key in tqdm(weight_keys, desc="mistral3-mxfp4"):
        if key.endswith("norm.weight") or key.endswith(".bias") or is_ignored(key):
            arr = get(key)
            add(key, arr.astype(np.float16) if arr.dtype != np.float16 else arr)
            continue
        if not key.endswith(".weight"):
            arr = get(key)
            add(key, arr.astype(np.float16) if arr.dtype != np.float16 else arr)
            continue

        # Mistral uses .weight_scale_inv for per-tensor dequant scale
        scale_inv_key = key.replace(".weight", ".weight_scale_inv")
        scale_alt_key = key.replace(".weight", ".weight_scale")
        scale_key = scale_inv_key if scale_inv_key in wm else scale_alt_key
        u = get(key)
        if scale_key in wm:
            if u.dtype != np.uint8: u = u.view(np.uint8)
            w_fp32 = fp8_to_fp32(u) * float(get(scale_key).reshape(-1)[0])
        else:
            w_fp32 = u.astype(np.float32)

        t = mx.array(w_fp32)
        qw, sc, bi = mx.quantize(t, bits=BITS, group_size=GROUP)
        mx.eval(qw, sc, bi)
        base = key.rsplit(".weight", 1)[0]
        add(base + ".weight", np.asarray(qw, dtype=np.uint32))
        add(base + ".scales", np.asarray(sc).astype(np.float16))
        add(base + ".biases", np.asarray(bi).astype(np.float16))
        del w_fp32, t; gc.collect()

    flush()
    for i in range(1, si + 1):
        (OUT / f"model-{i:05d}-of-XXXXX.safetensors").rename(
            OUT / f"model-{i:05d}-of-{si:05d}.safetensors")
    fixed = {k: v.replace("XXXXX", f"{si:05d}") for k, v in smap.items()}
    total = sum((OUT / fn).stat().st_size for fn in set(fixed.values()))
    (OUT / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"format": "mxfp4", "total_size": total}, "weight_map": fixed},
        indent=2))

    cfg.pop("quantization_config", None)
    cfg["quantization"] = {"group_size": GROUP, "bits": BITS}
    cfg["weight_format"] = "mxfp4"
    (OUT / "config.json").write_text(json.dumps(cfg, indent=2))
    for name in ("chat_template.jinja", "SYSTEM_PROMPT.txt", "generation_config.json",
                 "processor_config.json", "params.json",
                 "tokenizer.json", "tokenizer_config.json", "tekken.json", "LICENSE"):
        if (SRC / name).exists(): shutil.copy(SRC / name, OUT / name)
    print(f"DONE. shards={si}, total={total/1e9:.2f} GB → {OUT}")


if __name__ == "__main__":
    main()
