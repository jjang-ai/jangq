"""
poolside Laguna-XS.2 → MXFP4 Conversion (mlx 4-bit affine, group_size=32)
Created by Jinho Jang (eric@jangq.ai)

Mirrors convert_nemotron_mxfp4.py for the Laguna arch. No TurboQuant codec;
straight mlx.quantize at bits=4, group_size=32 over every 2-D weight.

What's quantized (4-bit affine g=32):
  - embed_tokens, lm_head
  - all self_attn.{q,k,v,o,g}_proj
  - layer-0 dense mlp.{gate,up,down}_proj
  - shared_expert.{gate,up,down}_proj
  - routed experts STACKED (n_experts, 2*inter, hidden) gate_up + (n_experts, hidden, inter) down

Passthrough fp16:
  - all norms (input_layernorm, post_attention_layernorm, q_norm, k_norm, model.norm)
  - mlp.gate.weight (router)
  - e_score_correction_bias

Usage:
  python -m jang_tools.convert_laguna_mxfp4 <SRC> <OUT> [bits] [group]
  defaults: bits=4, group=32
"""
import sys, json, gc, shutil, re
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

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else None
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else None
BITS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
GROUP = int(sys.argv[4]) if len(sys.argv) > 4 else 32
if SRC is None or OUT is None:
    print(__doc__); sys.exit(1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((SRC / "config.json").read_text())
    n_experts = cfg["num_experts"]

    idx = json.loads((SRC / "model.safetensors.index.json").read_text())
    weight_map = idx["weight_map"]
    open_cache = {}
    def get(k):
        sh = weight_map[k]
        if sh not in open_cache: open_cache[sh] = safe_open(str(SRC / sh), framework="pt")
        return _mlx_to_np(open_cache[sh].get_tensor(k))

    shard_idx = 0; tensors = {}; sb = 0
    MAX = 1_000_000_000
    smap = {}

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
        tensors[name] = arr
        sb += arr.nbytes
        if sb >= MAX: flush()

    def quantize_2d(out_base, arr, bits=BITS, group=GROUP):
        t = mx.array(arr.astype(np.float32))
        qw, sc, bi = mx.quantize(t, bits=bits, group_size=group)
        mx.eval(qw, sc, bi)
        add(out_base + ".weight", np.asarray(qw, dtype=np.uint32))
        add(out_base + ".scales", np.asarray(sc).astype(np.float16))
        add(out_base + ".biases", np.asarray(bi).astype(np.float16))

    def quantize_3d(out_base, stacked, bits=BITS, group=GROUP):
        # mx.quantize works on the last 2 dims; we reshape to (n_e * out, in)
        n_e, o, i = stacked.shape
        flat = stacked.reshape(n_e * o, i)
        t = mx.array(flat.astype(np.float32))
        qw, sc, bi = mx.quantize(t, bits=bits, group_size=group)
        mx.eval(qw, sc, bi)
        # Restore expert axis
        qw_np = np.asarray(qw, dtype=np.uint32).reshape(n_e, o, -1)
        sc_np = np.asarray(sc).astype(np.float16).reshape(n_e, o, -1)
        bi_np = np.asarray(bi).astype(np.float16).reshape(n_e, o, -1)
        add(out_base + ".weight", qw_np)
        add(out_base + ".scales", sc_np)
        add(out_base + ".biases", bi_np)

    PASS = re.compile(r"(norm\.weight$|\.mlp\.gate\.weight$|e_score_correction_bias$)")
    EXP_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")

    expert_groups = {}    # (L, p) -> {e: key}
    other = []
    for k in weight_map:
        m = EXP_RE.match(k)
        if m:
            L, e, p = int(m.group(1)), int(m.group(2)), m.group(3)
            expert_groups.setdefault((L, p), {})[e] = k
        else:
            other.append(k)

    # pass 1: scalars + non-expert linears
    for k in tqdm(sorted(other), desc="non-expert"):
        if PASS.search(k):
            arr = get(k)
            add(k, arr.astype(np.float16) if arr.dtype != np.float16 else arr)
        elif k.endswith(".weight"):
            quantize_2d(k.rsplit(".weight", 1)[0], get(k))
        else:
            arr = get(k)
            add(k, arr.astype(np.float16) if arr.dtype != np.float16 else arr)

    # pass 2: stacked routed experts (gate+up combined → gate_up_proj)
    layers = sorted({L for (L, _) in expert_groups})
    for L in tqdm(layers, desc="experts"):
        gate_keys = expert_groups.get((L, "gate"), {})
        up_keys   = expert_groups.get((L, "up"), {})
        down_keys = expert_groups.get((L, "down"), {})
        if not (len(gate_keys) == len(up_keys) == len(down_keys) == n_experts):
            raise ValueError(f"layer {L} expert tensor count mismatch")
        first = get(gate_keys[0])
        gate = np.empty((n_experts, *first.shape), dtype=first.dtype)
        for e, key in gate_keys.items(): gate[e] = get(key)
        first = get(up_keys[0])
        up = np.empty((n_experts, *first.shape), dtype=first.dtype)
        for e, key in up_keys.items(): up[e] = get(key)
        gate_up = np.concatenate([gate, up], axis=1)
        del gate, up
        quantize_3d(f"model.layers.{L}.mlp.experts.gate_up_proj", gate_up)
        del gate_up; gc.collect()
        first = get(down_keys[0])
        down = np.empty((n_experts, *first.shape), dtype=first.dtype)
        for e, key in down_keys.items(): down[e] = get(key)
        quantize_3d(f"model.layers.{L}.mlp.experts.down_proj", down)
        del down; gc.collect()

    flush()

    for i in range(1, shard_idx + 1):
        (OUT / f"model-{i:05d}-of-XXXXX.safetensors").rename(
            OUT / f"model-{i:05d}-of-{shard_idx:05d}.safetensors")
    fixed = {k: v.replace("XXXXX", f"{shard_idx:05d}") for k, v in smap.items()}
    total = sum((OUT / fn).stat().st_size for fn in set(fixed.values()))
    (OUT / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"format": "mxfp4", "total_size": total}, "weight_map": fixed},
        indent=2))

    cfg.pop("quantization_config", None)
    cfg["quantization"] = {"group_size": GROUP, "bits": BITS}
    cfg["weight_format"] = "mxfp4"
    (OUT / "config.json").write_text(json.dumps(cfg, indent=2))

    for name in ("chat_template.jinja", "generation_config.json",
                 "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "configuration_laguna.py", "modeling_laguna.py", "LICENSE.md"):
        if (SRC / name).exists(): shutil.copy(SRC / name, OUT / name)

    print(f"DONE. shards={shard_idx}, total={total/1e9:.2f} GB → {OUT}")


if __name__ == "__main__":
    main()
