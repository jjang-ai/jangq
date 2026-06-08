"""
MiniMax M2.7 → JANG_K conversion (prestacked switch_mlp via mx.quantize).
Created by Jinho Jang (eric@jangq.ai)

JANG_K bit allocation (mx.quantize affine, group_size=128):
  Routed expert w1 (gate_proj):  2-bit  + AWQ pre-scaling
  Routed expert w2 (down_proj):  4-bit
  Routed expert w3 (up_proj):    2-bit  + AWQ pre-scaling
  Self-attention q/k/v/o:        8-bit
  Embed tokens:                  6-bit
  LM head:                       8-bit
  Norms, biases, router gate:    fp16 passthrough

Output layout matches MiniMax-M2.7-JANG_2L-CRACK convention:
  model.layers.<li>.block_sparse_moe.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}
  shapes: (n_experts, out, in_packed) U32 + (n_experts, out, in/gs) F16 + biases F16

AWQ:
  s_l = clip((max(|x_l|) + eps) ** 0.5, min=1.0)   per-layer
  w1_stack' = w1_stack * s_l[None, None, :]
  w3_stack' = w3_stack * s_l[None, None, :]
  post_attention_layernorm.weight' = post_attention_layernorm.weight / s_l
  → forward math preserved; quant grid spent on high-importance columns.

Loadable via stock mlx_lm.load() — no JANGTQ Swift sidecar.

Usage:
    python3 -m jang_tools.convert_minimax_jang \
        --src  /Volumes/.../MiniMax-SLURPY \
        --out  ~/models/JANGQ/MiniMax-M2.7-JANG_K \
        --awq  ~/jang/awq_scales_minimax_m27.safetensors
"""
import argparse
import gc
import json
import shutil
import time
from pathlib import Path

import numpy as np
import mlx.core as mx
from safetensors import safe_open
from safetensors.numpy import save_file, load_file

from jang_tools.fp8 import load_fp8_tensor
from jang_tools.calibrate import _load_bf16_tensor


# ── Quantization policy ───────────────────────────────────────────
BITS_W1 = 2          # gate_proj  (AWQ-scaled)
BITS_W2 = 4          # down_proj
BITS_W3 = 2          # up_proj    (AWQ-scaled)
BITS_ATTN = 8        # q/k/v/o
BITS_EMBED = 6
BITS_LMHEAD = 8
GROUP_SIZE = 128
SHARD_BYTES = 4_500_000_000


def _parse_args():
    ap = argparse.ArgumentParser(description="MiniMax M2.7 → JANG_K converter")
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--awq", type=Path, required=False,
                    help="AWQ scales file from awq_capture_minimax")
    ap.add_argument("--no-awq", action="store_true")
    ap.add_argument("--shard-bytes", type=int, default=SHARD_BYTES)
    return ap.parse_args()


# ── Source loader ─────────────────────────────────────────────────
def _load_one(src: Path, weight_map: dict, tname: str) -> np.ndarray:
    """Load a single tensor from FP8 / raw / bf16 source."""
    shard = weight_map[tname]
    sf_path = src / shard
    with safe_open(str(sf_path), framework="numpy") as f:
        keys = set(f.keys())
        if tname not in keys:
            raise KeyError(tname)
        shape = list(f.get_slice(tname).get_shape())
        scale_key = tname + "_scale_inv"
        scale = f.get_tensor(scale_key) if scale_key in keys else None
    try:
        return load_fp8_tensor(sf_path, tname, shape, scale).astype(np.float32)
    except Exception:
        pass
    try:
        with safe_open(str(sf_path), framework="numpy") as f:
            t = f.get_tensor(tname)
            if not isinstance(t, np.ndarray):
                t = np.asarray(t)
            return t.astype(np.float32)
    except Exception:
        pass
    return _load_bf16_tensor(sf_path, tname, shape).astype(np.float32)


def _quantize(w_np: np.ndarray, bits: int):
    """mx.quantize(group_size=GROUP_SIZE, bits). Accepts 2D or 3D fp32 input."""
    w = mx.array(w_np.astype(np.float16))
    qw, qs, qb = mx.quantize(w, group_size=GROUP_SIZE, bits=bits)
    out = (
        np.array(qw),
        np.array(qs).astype(np.float16),
        np.array(qb).astype(np.float16),
    )
    del w, qw, qs, qb
    mx.metal.clear_cache()
    return out


# ── Sharded writer ────────────────────────────────────────────────
class ShardedWriter:
    def __init__(self, out_dir: Path, shard_bytes: int):
        self.out = out_dir
        self.shard_bytes = shard_bytes
        self.idx = 0
        self.bytes_in_shard = 0
        self.tensors = {}
        self.placeholder_map = {}  # tname -> placeholder filename
        self.total_written = 0

    def _placeholder(self, idx):
        return f"model-{idx:05d}-of-99999.safetensors"

    def add(self, name: str, arr: np.ndarray):
        self.tensors[name] = arr
        self.bytes_in_shard += arr.nbytes
        if self.bytes_in_shard >= self.shard_bytes:
            self.flush()

    def flush(self):
        if not self.tensors:
            return
        fn = self._placeholder(self.idx + 1)
        save_file(self.tensors, str(self.out / fn))
        for k in self.tensors:
            self.placeholder_map[k] = fn
        sz_gb = self.bytes_in_shard / 1e9
        print(f"      flushed shard {self.idx + 1}: "
              f"{len(self.tensors)} tensors, {sz_gb:.2f} GB", flush=True)
        self.idx += 1
        self.total_written += self.bytes_in_shard
        self.bytes_in_shard = 0
        self.tensors = {}

    def finalize(self):
        self.flush()
        total_shards = self.idx
        weight_map = {}
        # Rename placeholders to model-NNNNN-of-MMMMM.safetensors
        for i in range(1, total_shards + 1):
            old = self.out / self._placeholder(i)
            new_name = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
            old.rename(self.out / new_name)
            for k, v in self.placeholder_map.items():
                if v == self._placeholder(i):
                    weight_map[k] = new_name
        # Compute total disk size
        total_bytes = 0
        for fn in set(weight_map.values()):
            total_bytes += (self.out / fn).stat().st_size
        return total_shards, total_bytes, weight_map


# ── Main ──────────────────────────────────────────────────────────
def main():
    args = _parse_args()
    SRC = args.src.expanduser()
    OUT = args.out.expanduser()
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((SRC / "config.json").read_text())
    n_layers = cfg["num_hidden_layers"]
    n_experts = cfg["num_local_experts"]
    hidden = cfg["hidden_size"]
    inter = cfg["intermediate_size"]
    vocab = cfg["vocab_size"]

    print("=" * 64)
    print("  MiniMax M2.7 → JANG_K")
    print("=" * 64)
    print(f"  src:      {SRC}")
    print(f"  out:      {OUT}")
    print(f"  layers:   {n_layers}  experts: {n_experts}  "
          f"hidden: {hidden}  inter: {inter}  vocab: {vocab}")
    print(f"  policy:   w1={BITS_W1}  w2={BITS_W2}  w3={BITS_W3}  "
          f"attn={BITS_ATTN}  embed={BITS_EMBED}  lm_head={BITS_LMHEAD}  "
          f"gs={GROUP_SIZE}", flush=True)

    awq_per_layer = {}
    if not args.no_awq:
        if args.awq is None:
            raise SystemExit("--awq required (or --no-awq)")
        awq_path = args.awq.expanduser()
        print(f"  AWQ:      {awq_path}", flush=True)
        awq = load_file(str(awq_path))
        for li in range(n_layers):
            k = f"model.layers.{li}.block_sparse_moe.input_scale"
            if k in awq:
                awq_per_layer[li] = awq[k].astype(np.float32)
        print(f"            scales for {len(awq_per_layer)}/{n_layers} layers")
        if len(awq_per_layer) != n_layers:
            missing = [li for li in range(n_layers)
                       if li not in awq_per_layer]
            raise SystemExit(
                f"missing AWQ scales for layers: {missing[:5]}"
                f"{'...' if len(missing) > 5 else ''}"
            )
    else:
        print("  AWQ:      DISABLED (--no-awq)", flush=True)

    weight_map_path = SRC / "model.safetensors.index.json"
    weight_map = json.loads(weight_map_path.read_text())["weight_map"]
    print(f"  source:   {len(weight_map)} tensors across "
          f"{len(set(weight_map.values()))} shards", flush=True)

    writer = ShardedWriter(OUT, args.shard_bytes)
    overrides = {}
    bytes_in, bytes_out = 0, 0
    t0 = time.time()

    # ── Top-level: embed_tokens + lm_head + final norm ──────────
    print("\n  bookends:", flush=True)
    for tname, bits in [
        ("model.embed_tokens.weight", BITS_EMBED),
        ("lm_head.weight", BITS_LMHEAD),
    ]:
        t = _load_one(SRC, weight_map, tname)
        bytes_in += t.nbytes
        qw, qs, qb = _quantize(t, bits)
        base = tname[:-len(".weight")]
        writer.add(f"{base}.weight", qw)
        writer.add(f"{base}.scales", qs)
        writer.add(f"{base}.biases", qb)
        bytes_out += qw.nbytes + qs.nbytes + qb.nbytes
        overrides[base] = {"bits": bits, "group_size": GROUP_SIZE,
                           "mode": "affine"}
        print(f"    {base}: {bits}-bit gs={GROUP_SIZE}  "
              f"in={t.shape}  packed={qw.shape}", flush=True)
        del t, qw, qs, qb

    t = _load_one(SRC, weight_map, "model.norm.weight")
    arr = t.astype(np.float16)
    writer.add("model.norm.weight", arr)
    bytes_in += t.nbytes
    bytes_out += arr.nbytes
    print(f"    model.norm: passthrough fp16 {t.shape}", flush=True)
    del t, arr

    # ── Per-layer loop ───────────────────────────────────────────
    print(f"\n  per-layer ({n_layers}):", flush=True)
    for li in range(n_layers):
        t_layer = time.time()
        prefix = f"model.layers.{li}"

        # Attn projections
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            tname = f"{prefix}.self_attn.{proj}.weight"
            t = _load_one(SRC, weight_map, tname)
            bytes_in += t.nbytes
            qw, qs, qb = _quantize(t, BITS_ATTN)
            base = tname[:-len(".weight")]
            writer.add(f"{base}.weight", qw)
            writer.add(f"{base}.scales", qs)
            writer.add(f"{base}.biases", qb)
            bytes_out += qw.nbytes + qs.nbytes + qb.nbytes
            overrides[base] = {"bits": BITS_ATTN, "group_size": GROUP_SIZE,
                               "mode": "affine"}
            del t, qw, qs, qb

        # Attn norms (passthrough)
        for kn in ["q_norm", "k_norm"]:
            tname = f"{prefix}.self_attn.{kn}.weight"
            if tname in weight_map:
                t = _load_one(SRC, weight_map, tname)
                arr = t.astype(np.float16)
                writer.add(tname, arr)
                bytes_in += t.nbytes
                bytes_out += arr.nbytes
                del t, arr

        # Layer norms
        for ln in ["input_layernorm", "post_attention_layernorm"]:
            tname = f"{prefix}.{ln}.weight"
            t = _load_one(SRC, weight_map, tname)
            bytes_in += t.nbytes
            if ln == "post_attention_layernorm" and li in awq_per_layer:
                # Fold inverse AWQ scale here so forward math is preserved
                t = t / awq_per_layer[li]
            arr = t.astype(np.float16)
            writer.add(tname, arr)
            bytes_out += arr.nbytes
            del t, arr

        # Router gate + correction bias (passthrough)
        for fn in [
            "block_sparse_moe.gate.weight",
            "block_sparse_moe.e_score_correction_bias",
        ]:
            tname = f"{prefix}.{fn}"
            if tname in weight_map:
                t = _load_one(SRC, weight_map, tname)
                arr = t.astype(np.float16)
                writer.add(tname, arr)
                bytes_in += t.nbytes
                bytes_out += arr.nbytes
                del t, arr

        # Routed experts: stack 256 → quantize per projection
        scale_l = awq_per_layer.get(li)  # (hidden,) or None
        scale_apply = (scale_l is not None) and (not args.no_awq)

        for proj_src, proj_dst, bits, awq in [
            ("w1", "gate_proj", BITS_W1, scale_apply),
            ("w2", "down_proj", BITS_W2, False),
            ("w3", "up_proj",   BITS_W3, scale_apply),
        ]:
            t_load = time.time()
            stack = np.empty(
                (n_experts,
                 inter if proj_src in ("w1", "w3") else hidden,
                 hidden if proj_src in ("w1", "w3") else inter),
                dtype=np.float32,
            )
            for e in range(n_experts):
                tname = f"{prefix}.block_sparse_moe.experts.{e}.{proj_src}.weight"
                stack[e] = _load_one(SRC, weight_map, tname)
            t_load_sec = time.time() - t_load
            bytes_in += stack.nbytes

            if awq:
                # Pre-scale on input dim (last axis)
                stack *= scale_l[None, None, :]

            t_q = time.time()
            qw, qs, qb = _quantize(stack, bits)
            t_q_sec = time.time() - t_q
            del stack
            gc.collect()

            base = f"{prefix}.block_sparse_moe.switch_mlp.{proj_dst}"
            writer.add(f"{base}.weight", qw)
            writer.add(f"{base}.scales", qs)
            writer.add(f"{base}.biases", qb)
            bytes_out += qw.nbytes + qs.nbytes + qb.nbytes
            overrides[base] = {"bits": bits, "group_size": GROUP_SIZE,
                               "mode": "affine"}
            print(f"    L{li:2d} {proj_dst:9}: "
                  f"load={t_load_sec:5.1f}s  quant={t_q_sec:4.1f}s  "
                  f"bits={bits}  awq={awq}  packed={qw.shape}", flush=True)
            del qw, qs, qb
            mx.metal.clear_cache()

        print(f"    L{li:2d} done in {time.time()-t_layer:.1f}s   "
              f"running out_bytes={bytes_out/1e9:.2f}GB", flush=True)
        gc.collect()

    # ── Finalize shards + index.json ─────────────────────────────
    print("\n  finalizing shards...", flush=True)
    total_shards, total_size, final_weight_map = writer.finalize()
    index = {"metadata": {"total_size": total_size},
             "weight_map": final_weight_map}
    (OUT / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2))

    # ── config.json ──────────────────────────────────────────────
    print("  writing config.json ...", flush=True)
    out_cfg = dict(cfg)
    out_cfg.pop("quantization_config", None)
    quant_block = {
        "bits": BITS_ATTN,        # default = bookend bit (8)
        "group_size": GROUP_SIZE,
        "mode": "affine",
    }
    quant_block.update(overrides)
    out_cfg["quantization"] = quant_block
    out_cfg["torch_dtype"] = "bfloat16"
    out_cfg["_name_or_path"] = "MiniMax-M2.7-JANG_K"
    with open(OUT / "config.json", "w") as f:
        json.dump(out_cfg, f, indent=2)

    # ── Companion files ──────────────────────────────────────────
    for fn in [
        "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
        "chat_template.jinja", "generation_config.json",
        "configuration_minimax_m2.py", "modeling_minimax_m2.py",
        "special_tokens_map.json",
    ]:
        sfp = SRC / fn
        if sfp.exists():
            shutil.copy2(sfp, OUT / fn)

    for fn in ["prune_info.json"]:
        fp = OUT / fn
        if fp.exists():
            fp.unlink()
            print(f"    stripped {fn}")

    # ── jang_config.json ────────────────────────────────────────
    avg_routed_bits = (BITS_W1 + BITS_W2 + BITS_W3) / 3
    jang_cfg = {
        "format": "jang",
        "format_version": "2.0",
        "quantization": {
            "method": "jang-affine-mixed",
            "profile": "JANG_K",
            "block_size": GROUP_SIZE,
            "mode": "affine",
            "bit_widths_used": sorted({BITS_W1, BITS_W2, BITS_W3,
                                       BITS_ATTN, BITS_EMBED, BITS_LMHEAD}),
            "routed_avg_bits": round(avg_routed_bits, 3),
            "awq": {
                "enabled": (not args.no_awq) and len(awq_per_layer) > 0,
                "alpha": 0.5,
                "scope": "routed_experts.gate_proj+up_proj (joint per-layer)",
                "fold_target": "post_attention_layernorm.weight",
                "scale_floor": 1.0,
            },
        },
        "source_model": {
            "name": "MiniMax-M2.7-FP8",
            "dtype": "fp8_e4m3",
            "parameters": "227.6B",
        },
        "architecture": {
            "type": "moe",
            "attention": "gqa",
            "has_vision": False,
            "has_ssm": False,
            "has_moe": True,
        },
        "capabilities": {
            "reasoning_parser": "qwen3",
            "tool_parser": "minimax",
            "think_in_template": True,
            "supports_tools": True,
            "supports_thinking": True,
            "family": "minimax_m2",
            "modality": "text",
            "cache_type": "kv",
        },
    }
    with open(OUT / "jang_config.json", "w") as f:
        json.dump(jang_cfg, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  bytes_in:  {bytes_in/1e9:7.2f} GB")
    print(f"  bytes_out: {bytes_out/1e9:7.2f} GB  "
          f"(compression: {bytes_in/max(bytes_out,1):.2f}x)")
    print(f"  shards:    {total_shards}  on_disk: {total_size/1e9:.2f} GB")
    print(f"  elapsed:   {elapsed/60:.1f} min ({elapsed:.1f}s)")
    print(f"  DONE → {OUT}")


if __name__ == "__main__":
    main()
