"""REAP saliency profiling for MiniMax-M3 over the Vera corpus.

Runs the validated streaming forward and accumulates, per MoE layer and expert,
the REAP saliency  S[l,e] = sum over tokens routed to e of  gate_e * ||expert_e(x)||_2
(plus selection count). Processes `batches` groups of equal-length Vera
sequences; each group is one weight-streaming pass.

Output (npz): saliency[L,E] f32, count[L,E] i64, layer_ids[L] (MoE layer indices).

  python -m jang_tools.minimax_m3.reap_profile \
      --model /Volumes/EricsLLMDrive/sources/minimax-m3 \
      --calib /Volumes/.../vera-keep.jsonl \
      --out /Users/eric/m3_reap_saliency.npz \
      --batch 16 --seq-len 512 --batches 4 --device mps

Created by Jinho Jang (eric@jangq.ai).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .layer_forward import (M3Cfg, gemma_rms_norm, gqa_attention, swigluoai,
                            moe_forward_observe, precompute_rope)
from .weight_stream import build_index, WeightStreamer
from .probe import _load_tokenizer, _vera_samples


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = M3Cfg.from_config_json(Path(args.model) / "config.json")
    idx = build_index(Path(args.model))
    streamer = WeightStreamer(idx, quant="none", device=device, compute_dtype=torch.bfloat16)
    tok = _load_tokenizer(args.model)

    need = args.batch * args.batches
    texts = _vera_samples(args.calib, need * 3,
                          domains=("coding", "agentic", "shell", "security", "general", "business", "math", "arithmetic", "stem", "knowledge"))
    seqs = []
    for t in texts:
        ids = tok.encode(t)
        if len(ids) >= args.seq_len:
            seqs.append(ids[:args.seq_len])
        if len(seqs) >= need:
            break
    if len(seqs) < args.batch:
        raise SystemExit(f"only {len(seqs)} Vera seqs >= seq_len; lower --seq-len/--batch")
    groups = [seqs[i:i + args.batch] for i in range(0, len(seqs), args.batch)]
    groups = [g for g in groups if len(g) == args.batch][: args.batches]
    print(f"  REAP profile: {len(groups)} passes x B={args.batch} x T={args.seq_len} "
          f"= {len(groups)*args.batch*args.seq_len} tokens, device={device}", flush=True)

    E = cfg.num_local_experts
    moe_layers = [li for li in range(cfg.num_hidden_layers) if not cfg.is_dense(li)]
    sal = {li: torch.zeros(E, dtype=torch.float64) for li in moe_layers}
    cnt = {li: torch.zeros(E, dtype=torch.int64) for li in moe_layers}

    t0 = time.time()
    for gi, g in enumerate(groups):
        B, T = len(g), args.seq_len
        input_ids = torch.tensor(g, device=device)
        embed = streamer.embed()
        h = embed[input_ids.reshape(-1)].reshape(B, T, -1).to(device)
        del embed
        cos, sin = precompute_rope(cfg.rotary_dim, T, cfg.rope_theta, device, h.dtype)
        for li in range(cfg.num_hidden_layers):
            inln, postln = streamer.norms(li)
            aw, qn, kn = streamer.attn(li)
            r = h
            hn = gqa_attention(gemma_rms_norm(h, inln, cfg.rms_norm_eps),
                               aw["q"], aw["k"], aw["v"], aw["o"], qn, kn, cfg, cos, sin)
            h = r + hn
            r = h
            hp = gemma_rms_norm(h, postln, cfg.rms_norm_eps)
            if cfg.is_dense(li):
                dm = streamer.dense_mlp(li)
                h = r + swigluoai(hp, dm["gate_proj"], dm["up_proj"], dm["down_proj"],
                                  cfg.swiglu_alpha, cfg.swiglu_limit)
            else:
                rw, rb = streamer.router(li)
                sh = streamer.shared_expert(li)
                mo, s_sum, c = moe_forward_observe(hp, rw, rb, streamer.make_expert_loader(li),
                                                   sh, cfg, device)
                sal[li] += s_sum.cpu().double()
                cnt[li] += c.cpu()
                h = r + mo
        print(f"    pass {gi+1}/{len(groups)} done  {time.time()-t0:.0f}s", flush=True)

    layer_ids = np.array(moe_layers, dtype=np.int64)
    sal_arr = np.stack([sal[li].numpy() for li in moe_layers]).astype(np.float32)
    cnt_arr = np.stack([cnt[li].numpy() for li in moe_layers]).astype(np.int64)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, saliency=sal_arr, count=cnt_arr, layer_ids=layer_ids,
             num_experts=E)
    # quick stats
    frac_dead = float((cnt_arr.sum(0) == 0).mean())
    print(f"\n  saved {args.out}  saliency{sal_arr.shape} count_tot={int(cnt_arr.sum())}")
    print(f"  experts never selected (any layer pooled): {frac_dead*100:.1f}%  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
