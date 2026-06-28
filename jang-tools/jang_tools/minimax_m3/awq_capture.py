"""AWQ activation capture for MiniMax-M3 (one weight-streaming pass).

Runs the validated M3 streaming forward over a batch of equal-length Vera
sequences and accumulates, per MoE layer, the per-input-channel activation
maximum of the post_attention_layernorm output (the experts' input). Emits
AWQ scales  s = clip((max|x| + eps)^alpha, min=floor)  in the layout the
converter consumes:

    language_model.model.layers.{li}.block_sparse_moe.input_scale   (hidden,) fp32

Because the batch is equal-length there is no padding to mask out. One pass
over the weights (~one probe forward) yields stats over batch*seq_len tokens.

  python -m jang_tools.minimax_m3.awq_capture \
      --model /Volumes/EricsLLMDrive/sources/minimax-m3 \
      --calib /Volumes/EricsLLMDrive/sources/vera-agentic-coder/vera-keep.jsonl \
      --out   /Users/eric/awq_scales_m3.safetensors \
      --batch 8 --seq-len 512 --device mps

Created by Jinho Jang (eric@jangq.ai).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

from .layer_forward import (M3Cfg, gemma_rms_norm, gqa_attention, swigluoai,
                            moe_forward_observe, precompute_rope)
from .weight_stream import build_index, WeightStreamer
from .probe import _load_tokenizer, _vera_samples

EPS = 1e-6
SCALE_FLOOR = 1.0


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = M3Cfg.from_config_json(Path(args.model) / "config.json")
    idx = build_index(Path(args.model))
    streamer = WeightStreamer(idx, quant="none", device=device, compute_dtype=torch.bfloat16)
    tok = _load_tokenizer(args.model)

    texts = _vera_samples(args.calib, args.batch * 3,
                          domains=("coding", "agentic", "shell", "security", "general", "math", "arithmetic", "stem", "knowledge"))
    seqs = []
    for t in texts:
        ids = tok.encode(t)
        if len(ids) >= args.seq_len:
            seqs.append(ids[:args.seq_len])
        if len(seqs) >= args.batch:
            break
    if len(seqs) < 1:
        raise SystemExit("no Vera sequences >= seq_len; lower --seq-len")
    B, T = len(seqs), args.seq_len
    input_ids = torch.tensor(seqs, device=device)            # (B,T)
    print(f"  AWQ capture: B={B} T={T} ({B*T} tokens) device={device} alpha={args.alpha}", flush=True)

    embed = streamer.embed()
    h = embed[input_ids.reshape(-1)].reshape(B, T, -1).to(device)
    del embed
    cos, sin = precompute_rope(cfg.rotary_dim, T, cfg.rope_theta, device, h.dtype)

    act_max = {}   # li -> (hidden,) fp32 running max of |post_attn_norm|
    t0 = time.time()
    for li in range(cfg.num_hidden_layers):
        tl = time.time()
        inln, postln = streamer.norms(li)
        aw, qn, kn = streamer.attn(li)
        # attention block
        r = h
        hn = gemma_rms_norm(h, inln, cfg.rms_norm_eps)
        hn = gqa_attention(hn, aw["q"], aw["k"], aw["v"], aw["o"], qn, kn, cfg, cos, sin)
        h = r + hn
        # mlp block
        r = h
        hpost = gemma_rms_norm(h, postln, cfg.rms_norm_eps)
        if not cfg.is_dense(li):
            cmax = hpost.float().abs().amax(dim=(0, 1)).cpu().numpy()  # (hidden,)
            prev = act_max.get(li)
            act_max[li] = cmax if prev is None else np.maximum(prev, cmax)
            rw, rb = streamer.router(li)
            sh = streamer.shared_expert(li)
            ld = streamer.make_expert_loader(li)
            mo, _, _ = moe_forward_observe(hpost, rw, rb, ld, sh, cfg, device)
            h = r + mo
        else:
            dm = streamer.dense_mlp(li)
            h = r + swigluoai(hpost, dm["gate_proj"], dm["up_proj"], dm["down_proj"],
                              cfg.swiglu_alpha, cfg.swiglu_limit)
        tag = "moe" if not cfg.is_dense(li) else "dense"
        mx = float(act_max[li].max()) if li in act_max else 0.0
        print(f"    L{li:2d} {tag:5s} {time.time()-tl:5.1f}s act_max={mx:.2f} finite={torch.isfinite(h).all().item()}", flush=True)

    out = {}
    for li, vec in act_max.items():
        s = np.power(vec + EPS, args.alpha).astype(np.float32)
        s = np.maximum(s, SCALE_FLOOR).astype(np.float32)
        out[f"{idx.text_prefix}layers.{li}.block_sparse_moe.input_scale"] = s
        out[f"{idx.text_prefix}layers.{li}.block_sparse_moe.input_max"] = vec.astype(np.float32)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_file(out, args.out)
    print(f"\n  wrote {len(act_max)} layer scales -> {args.out}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
