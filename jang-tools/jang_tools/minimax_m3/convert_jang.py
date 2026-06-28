"""MiniMax-M3 (minimax_m3_vl) -> JANG_2L converter (prestacked switch_mlp).

JANG_2L bit allocation (mx.quantize affine, group_size=64):
  Routed expert w1/w2/w3 :  2-bit   (+ optional AWQ pre-scaling on gate/up)
  Shared expert g/u/d    :  6-bit   (always-on -> protected)
  Dense MLP (layers 0-2) :  6-bit   (always-on, only 3 layers)
  Attention q/k/v/o      :  8-bit
  Embed tokens           :  6-bit
  LM head                :  8-bit
  Vision tower + proj    :  8-bit
  Norms, router gate +
    e_score_correction,
    MSA indexer (q/k proj
    + norms)             :  fp16 passthrough (selection quality is critical)

Loadable via stock mlx_lm/mlx_vlm once a minimax_m3_vl model class exists; the
per-module bit map is written into config.json["quantization"] either way.

Routed expert source layout (per expert, Mixtral-style):
  language_model.model.layers.{L}.block_sparse_moe.experts.{e}.{w1,w2,w3}.weight
    w1 = gate [moe_inter, H]   w3 = up [moe_inter, H]   w2 = down [H, moe_inter]
-> stacked + quantized to:
  language_model.model.layers.{L}.block_sparse_moe.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}

Usage:
  python -m jang_tools.minimax_m3.convert_jang \
      --src /Users/eric/models/minimax-m3-src \
      --out ~/.mlxstudio/models/JANGQ-AI/MiniMax-M3-JANG_2L \
      [--awq <scales.safetensors> | --no-awq] [--keep-experts keep.json]

`--keep-experts` (REAP output) maps layer -> kept expert id list; experts not
listed are dropped and the router gate rows are reindexed accordingly.

Created by Jinho Jang (eric@jangq.ai).
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jang_tools.calibrate import _load_bf16_tensor  # noqa: E402

# ── bit policy ────────────────────────────────────────────────────
BITS = {"expert": 2, "shared": 6, "dense": 6, "attn": 8,
        "embed": 6, "lmhead": 8, "vision": 8}
GROUP_SIZE = 64
EXPERT_GS = 64           # PROPER: gs64 (was gs128 size-corner that caused logit collapse)
SHARD_BYTES = 4_500_000_000


def _parse_args():
    ap = argparse.ArgumentParser(description="MiniMax-M3 -> JANG_2L")
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--awq", type=Path, default=None)
    ap.add_argument("--no-awq", action="store_true")
    ap.add_argument("--keep-experts", type=Path, default=None,
                    help="REAP keep map: {layer_idx: [expert_id,...]} JSON")
    ap.add_argument("--down-bits", type=int, default=3,
                    help="down_proj bit width (3=d3 default, 2=all-2-bit)")
    ap.add_argument("--shard-bytes", type=int, default=SHARD_BYTES)
    return ap.parse_args()


def _load_one(src: Path, wm: dict, name: str) -> np.ndarray:
    """Load a tensor as fp32 numpy, dtype-agnostic (bf16/fp16/fp32 all OK).

    torch's safetensors reader handles every dtype natively; the router gate is
    fp32 while experts/attention are bf16, so a bf16-only reader breaks here.
    """
    import torch  # noqa: F401
    sf = src / wm[name]
    with safe_open(str(sf), framework="pt") as f:
        return f.get_tensor(name).float().numpy()


def _quant(w_np: np.ndarray, bits: int, gs: int = GROUP_SIZE):
    # Quantize from fp32 (not fp16) so scale/zero-point estimation uses full
    # precision — matters most for the 2-bit routed experts (arithmetic).
    w = mx.array(w_np.astype(np.float32))
    qw, qs, qb = mx.quantize(w, group_size=gs, bits=bits)
    out = (np.array(qw), np.array(qs).astype(np.float16), np.array(qb).astype(np.float16))
    del w, qw, qs, qb
    mx.metal.clear_cache()
    return out


class ShardedWriter:
    def __init__(self, out_dir: Path, shard_bytes: int):
        self.out = out_dir; self.shard_bytes = shard_bytes
        self.idx = 0; self.bytes_in_shard = 0; self.tensors = {}
        self.placeholder_map = {}; self.total_written = 0

    def _ph(self, i): return f"model-{i:05d}-of-99999.safetensors"

    def add(self, name, arr):
        self.tensors[name] = arr; self.bytes_in_shard += arr.nbytes
        if self.bytes_in_shard >= self.shard_bytes:
            self.flush()

    def flush(self):
        if not self.tensors: return
        fn = self._ph(self.idx + 1)
        save_file(self.tensors, str(self.out / fn))
        for k in self.tensors: self.placeholder_map[k] = fn
        print(f"      shard {self.idx+1}: {len(self.tensors)} tensors "
              f"{self.bytes_in_shard/1e9:.2f}GB", flush=True)
        self.idx += 1; self.total_written += self.bytes_in_shard
        self.bytes_in_shard = 0; self.tensors = {}

    def finalize(self):
        self.flush()
        n = self.idx; wm = {}
        for i in range(1, n + 1):
            new = f"model-{i:05d}-of-{n:05d}.safetensors"
            (self.out / self._ph(i)).rename(self.out / new)
            for k, v in self.placeholder_map.items():
                if v == self._ph(i): wm[k] = new
        total = sum((self.out / f).stat().st_size for f in set(wm.values()))
        return n, total, wm


def main():
    args = _parse_args()
    SRC, OUT = args.src.expanduser(), args.out.expanduser()
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((SRC / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    NL = tc["num_hidden_layers"]
    NE = tc["num_local_experts"]
    H = tc["hidden_size"]
    moe_inter = tc["intermediate_size"]
    moe_freq = tc.get("moe_layer_freq")
    is_moe = (lambda li: bool(moe_freq[li])) if moe_freq else (lambda li: li >= 3)

    keep_map = None
    if args.keep_experts:
        raw = json.loads(args.keep_experts.read_text())
        keep_map = {int(k): list(v) for k, v in raw.items()}
        print(f"  REAP keep map: {len(keep_map)} layers, "
              f"e.g. L3 keeps {len(keep_map.get(3, []))}/{NE}", flush=True)

    awq_layer = {}
    if not args.no_awq and args.awq:
        awq = load_file(str(args.awq.expanduser()))
        for li in range(NL):
            k = f"language_model.model.layers.{li}.block_sparse_moe.input_scale"
            if k in awq:
                awq_layer[li] = awq[k].astype(np.float32)
        print(f"  AWQ: {len(awq_layer)} layers", flush=True)

    wm = json.loads((SRC / "model.safetensors.index.json").read_text())["weight_map"]
    TP = "language_model.model."
    HEAD = "language_model.lm_head.weight"
    writer = ShardedWriter(OUT, args.shard_bytes)
    overrides = {}
    t0 = time.time()

    print(f"  M3 -> JANG_2L  layers={NL} experts={NE} H={H} moe_inter={moe_inter}", flush=True)
    print(f"  policy: {BITS} gs={GROUP_SIZE}", flush=True)

    def emit_quant(name_base, arr_np, bits, gs=GROUP_SIZE):
        qw, qs, qb = _quant(arr_np, bits, gs)
        writer.add(f"{name_base}.weight", qw)
        writer.add(f"{name_base}.scales", qs)
        writer.add(f"{name_base}.biases", qb)
        overrides[name_base] = {"bits": bits, "group_size": gs, "mode": "affine"}
        return qw.nbytes + qs.nbytes + qb.nbytes

    def emit_pass(name):
        t = _load_one(SRC, wm, name).astype(np.float16)
        writer.add(name, t)
        return t.nbytes

    # ── bookends ──
    print("  bookends...", flush=True)
    emit_quant(TP + "embed_tokens", _load_one(SRC, wm, TP + "embed_tokens.weight"), BITS["embed"])
    if HEAD in wm:
        emit_quant("language_model.lm_head", _load_one(SRC, wm, HEAD), BITS["lmhead"])
    emit_pass(TP + "norm.weight")

    # ── decoder layers ──
    for li in range(NL):
        tl = time.time(); pre = f"{TP}layers.{li}"
        # attention
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            emit_quant(f"{pre}.self_attn.{proj}", _load_one(SRC, wm, f"{pre}.self_attn.{proj}.weight"), BITS["attn"])
        # qk-norms + MSA indexer (passthrough, high precision)
        for sub in ("q_norm.weight", "k_norm.weight",
                    "index_q_proj.weight", "index_k_proj.weight",
                    "index_q_norm.weight", "index_k_norm.weight"):
            k = f"{pre}.self_attn.{sub}"
            if k in wm: emit_pass(k)
        # AWQ scale for this MoE layer (per-input-channel), if available.
        scale = awq_layer.get(li) if (not args.no_awq) else None

        # input layernorm: always passthrough.
        emit_pass(f"{pre}.input_layernorm.weight")
        # post_attention_layernorm: feeds the MoE/MLP. With AWQ we fold the
        # inverse scale here so the forward is preserved. M3 is GEMMA-normed
        # (output = x * (1 + w)), so the fold is  (1+w)/s - 1, NOT a plain divide.
        if is_moe(li) and scale is not None:
            pw = _load_one(SRC, wm, f"{pre}.post_attention_layernorm.weight")
            folded = (1.0 + pw) / scale - 1.0
            writer.add(f"{pre}.post_attention_layernorm.weight", folded.astype(np.float16))
        else:
            emit_pass(f"{pre}.post_attention_layernorm.weight")

        if not is_moe(li):
            # dense MLP
            for proj in ("gate_proj", "up_proj", "down_proj"):
                emit_quant(f"{pre}.mlp.{proj}", _load_one(SRC, wm, f"{pre}.mlp.{proj}.weight"), BITS["dense"])
            print(f"    L{li:2d} dense  {time.time()-tl:.1f}s", flush=True)
            continue

        # router gate + correction bias (passthrough; reindex if pruning)
        kept = keep_map.get(li, list(range(NE))) if keep_map else list(range(NE))
        gate = _load_one(SRC, wm, f"{pre}.block_sparse_moe.gate.weight")        # (E,H)
        bias_k = f"{pre}.block_sparse_moe.e_score_correction_bias"
        bias = _load_one(SRC, wm, bias_k) if bias_k in wm else None
        if len(kept) != NE:
            gate = gate[kept]
            if bias is not None: bias = bias[kept]
        g16 = gate.astype(np.float16); writer.add(f"{pre}.block_sparse_moe.gate.weight", g16)
        if bias is not None:
            writer.add(bias_k, bias.astype(np.float16))

        # shared expert (6-bit). It reads the SAME folded post_attn norm output,
        # so gate/up inputs must be scaled by s too (down is post-activation).
        sh = f"{pre}.block_sparse_moe.shared_experts"
        if f"{sh}.gate_proj.weight" in wm:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                w = _load_one(SRC, wm, f"{sh}.{proj}.weight")
                if scale is not None and proj in ("gate_proj", "up_proj"):
                    w = w * scale[None, :]
                emit_quant(f"{sh}.{proj}", w, BITS["shared"])

        # routed experts: stack kept experts per projection, quantize 2-bit
        for src_p, dst_p, awq_on in [("w1", "gate_proj", True), ("w3", "up_proj", True), ("w2", "down_proj", False)]:
            rows = moe_inter if src_p in ("w1", "w3") else H
            cols = H if src_p in ("w1", "w3") else moe_inter
            stack = np.empty((len(kept), rows, cols), dtype=np.float32)
            for i, e in enumerate(kept):
                stack[i] = _load_one(SRC, wm, f"{pre}.block_sparse_moe.experts.{e}.{src_p}.weight")
            if awq_on and scale is not None:
                stack *= scale[None, None, :]
            emit_quant(f"{pre}.block_sparse_moe.switch_mlp.{dst_p}", stack, (args.down_bits if dst_p == "down_proj" else BITS["expert"]), gs=EXPERT_GS)
            del stack; gc.collect(); mx.metal.clear_cache()
        print(f"    L{li:2d} moe keep={len(kept)}/{NE} {time.time()-tl:.1f}s", flush=True)
        gc.collect()

    # ── vision tower + projectors (8-bit linears, passthrough norms/bias) ──
    print("  vision + projectors...", flush=True)
    vision_keys = [k for k in wm if k.startswith("vision_tower.")
                   or k.startswith("multi_modal_projector.") or k.startswith("patch_merge_mlp.")]
    for k in vision_keys:
        if k.endswith(".weight"):
            arr = _load_one(SRC, wm, k)
            # only 2D linear weights get quantized; conv/embeddings/norms pass through
            if arr.ndim == 2 and min(arr.shape) >= GROUP_SIZE and "embeddings" not in k:
                emit_quant(k[:-len(".weight")], arr, BITS["vision"])
                continue
            writer.add(k, arr.astype(np.float16))
            continue
        emit_pass(k)

    print("  finalizing...", flush=True)
    nshard, total, fwm = writer.finalize()
    (OUT / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": fwm}, indent=2))

    # config.json
    out_cfg = dict(cfg)
    out_cfg.pop("quantization_config", None)
    qb = {"bits": 8, "group_size": GROUP_SIZE, "mode": "affine"}
    qb.update(overrides)
    out_cfg["quantization"] = qb
    if keep_map:
        new_ne = len(keep_map.get(3, list(range(NE))))
        out_cfg.setdefault("text_config", {})
        if "text_config" in out_cfg:
            out_cfg["text_config"]["num_local_experts"] = new_ne
    out_cfg["_name_or_path"] = OUT.name
    (OUT / "config.json").write_text(json.dumps(out_cfg, indent=2))

    # companion files (tokenizer + VL processors + chat template + model code)
    for fn in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
               "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
               "generation_config.json", "preprocessor_config.json",
               "configuration_minimax_m3_vl.py", "processing_minimax.py",
               "image_processor.py", "video_processor.py"]:
        if (SRC / fn).exists():
            shutil.copy2(SRC / fn, OUT / fn)

    avg_routed = BITS["expert"]
    jang_cfg = {
        "format": "jang", "format_version": "2.0",
        "quantization": {"method": "jang-affine-mixed", "profile": "JANG_2L",
                         "block_size": GROUP_SIZE, "mode": "affine",
                         "routed_avg_bits": avg_routed,
                         "awq": {"enabled": bool(awq_layer)}},
        "reap": {"pruned": bool(keep_map),
                 "experts_kept": (len(keep_map.get(3, [])) if keep_map else NE),
                 "experts_total": NE},
        "architecture": {"type": "moe", "attention": "gqa+msa_sparse",
                         "has_vision": True, "has_moe": True,
                         "cache_type": "kv+msa_index_dual"},
        "capabilities": {"family": "minimax_m3", "modality": "multimodal",
                         "supports_tools": True, "supports_thinking": True},
    }
    (OUT / "jang_config.json").write_text(json.dumps(jang_cfg, indent=2))

    print(f"\n  shards={nshard} on_disk={total/1e9:.2f}GB elapsed={(time.time()-t0)/60:.1f}min")
    print(f"  DONE -> {OUT}")


if __name__ == "__main__":
    main()
