"""Coherence probe for MiniMax-M3 via streamed forward.

Runs ONE teacher-forced forward over real text and reports next-token top-1
accuracy + perplexity. A coherent model predicts real (esp. code) tokens well;
a quant-broken model is near-random. Compare --quant none (validates the port)
vs --quant 2L (tests the JANG_2L quantization). Optionally a short greedy
continuation for a qualitative read.

  python -m jang_tools.minimax_m3.probe \
      --model /Users/eric/models/minimax-m3-src \
      --calib /Volumes/EricsLLMDrive/sources/vera-agentic-coder/vera-keep.jsonl \
      --quant 2L --seq-len 192 --samples 4 --device cpu [--max-layers N] [--gen 16]

Created by Jinho Jang (eric@jangq.ai).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .layer_forward import M3Cfg, LayerWeights, decoder_layer_forward, gemma_rms_norm, precompute_rope
from .weight_stream import build_index, WeightStreamer


def _load_tokenizer(model_dir: str):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    except Exception:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))

        class _Shim:
            def encode(self, s, **kw):
                return tok.encode(s).ids

            def decode(self, ids, **kw):
                return tok.decode(list(ids))
        return _Shim()


def _vera_samples(calib_path: str, n: int, domains=("coding", "agentic", "shell")):
    out = []
    with open(calib_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("domain") in domains and len(rec.get("text", "")) > 400:
                out.append(rec["text"])
                if len(out) >= n:
                    break
    return out


@torch.no_grad()
def run_forward(model_dir, cfg: M3Cfg, streamer: WeightStreamer, input_ids: torch.Tensor,
                device, max_layers: int | None, verbose=True, keep_map: dict | None = None):
    """input_ids: (1, T) on device. Returns logits (1,T,V) f32 on cpu."""
    T = input_ids.shape[1]
    n_layers = cfg.num_hidden_layers if max_layers is None else min(max_layers, cfg.num_hidden_layers)

    embed = streamer.embed()                                  # (V,H)
    h = embed[input_ids[0]].unsqueeze(0).to(device)           # (1,T,H)
    del embed

    cos, sin = precompute_rope(cfg.rotary_dim, T, cfg.rope_theta, device, h.dtype)

    for li in range(n_layers):
        t0 = time.time()
        inln, postln = streamer.norms(li)
        aw, qn, kn = streamer.attn(li)
        lw = LayerWeights(
            input_layernorm=inln, post_attention_layernorm=postln,
            q_w=aw["q"], k_w=aw["k"], v_w=aw["v"], o_w=aw["o"], q_norm=qn, k_norm=kn,
        )
        if cfg.is_dense(li):
            lw.dense_mlp = streamer.dense_mlp(li)
        else:
            lw.router_w, lw.router_bias = streamer.router(li)
            lw.shared_expert = streamer.shared_expert(li)
            lw.expert_loader = streamer.make_expert_loader(li)
            if keep_map is not None:
                lw.keep_ids = keep_map.get(li) or keep_map.get(str(li))
        h, _, _ = decoder_layer_forward(h, lw, cfg, cos, sin, device)
        if verbose:
            kind = "dense" if cfg.is_dense(li) else "moe"
            fin = torch.isfinite(h).all().item()
            print(f"    L{li:2d} {kind:5s} {time.time()-t0:5.1f}s  "
                  f"hidden|mean|={h.float().abs().mean():.3f} finite={fin}", flush=True)
        del lw

    norm_w = streamer.final_norm()
    h = gemma_rms_norm(h, norm_w, cfg.rms_norm_eps)
    lm = streamer.lm_head().to(device)
    logits = (h @ lm.T.to(h.dtype)).float().cpu()
    return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--quant", choices=["none", "2L"], default="none")
    ap.add_argument("--awq", default=None, help="AWQ scales .safetensors (2L only)")
    ap.add_argument("--keep-experts", default=None, help="REAP keep-map JSON (prune sim)")
    ap.add_argument("--seq-len", type=int, default=192)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-layers", type=int, default=None,
                    help="truncate depth for a fast partial probe")
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = M3Cfg.from_config_json(Path(args.model) / "config.json")
    print("=" * 64)
    print(f"  MiniMax-M3 coherence probe  quant={args.quant}  device={device}")
    print(f"  layers={cfg.num_hidden_layers} dense={cfg.mlp_layer_types.count('dense')} "
          f"experts={cfg.num_local_experts} top-{cfg.num_experts_per_tok} "
          f"H={cfg.hidden_size} heads={cfg.num_attention_heads}/{cfg.num_key_value_heads}")
    if args.max_layers:
        print(f"  TRUNCATED to first {args.max_layers} layers (partial probe)")
    print("=" * 64, flush=True)

    idx = build_index(Path(args.model))
    awq = None
    if args.awq:
        from safetensors.numpy import load_file
        raw = load_file(args.awq)
        awq = {}
        for k, v in raw.items():
            if k.endswith(".block_sparse_moe.input_scale"):
                li = int(k.split(".layers.")[1].split(".")[0])
                awq[li] = v
        print(f"  AWQ: loaded {len(awq)} layer scales from {args.awq}")
    streamer = WeightStreamer(idx, quant=args.quant, device=device,
                              compute_dtype=torch.bfloat16, awq=awq)
    keep_map = None
    if args.keep_experts:
        raw = json.loads(Path(args.keep_experts).read_text())
        keep_map = {int(k): list(v) for k, v in raw.items()}
        print(f"  REAP keep-map: {len(keep_map)} layers, keep {len(next(iter(keep_map.values())))}/expert-count")
    tok = _load_tokenizer(args.model)
    texts = _vera_samples(args.calib, args.samples)
    if not texts:
        raise SystemExit("no Vera samples matched the domain filter")

    tot_correct = tot_tokens = 0
    tot_nll = 0.0
    for si, text in enumerate(texts):
        ids = tok.encode(text)[: args.seq_len]
        if len(ids) < 16:
            continue
        input_ids = torch.tensor([ids], device=device)
        print(f"\n  sample {si} ({len(ids)} tok):", flush=True)
        t0 = time.time()
        logits = run_forward(args.model, cfg, streamer, input_ids, device, args.max_layers,
                             keep_map=keep_map)
        # teacher-forced next-token metrics over positions 0..T-2
        pred = logits[0, :-1].argmax(-1)              # (T-1,)
        tgt = input_ids[0, 1:].cpu()                  # (T-1,)
        correct = (pred == tgt).sum().item()
        logp = torch.log_softmax(logits[0, :-1].float(), -1)
        nll = -logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
        tot_correct += correct
        tot_tokens += tgt.numel()
        tot_nll += nll.sum().item()
        acc = correct / tgt.numel()
        ppl = float(torch.exp(nll.mean()))
        print(f"    -> top1-acc={acc:.3f}  ppl={ppl:.1f}  ({time.time()-t0:.0f}s)", flush=True)
        # qualitative: show 6 mid-sequence predictions
        show = range(max(0, len(ids) // 2), min(len(ids) - 1, len(ids) // 2 + 6))
        for p in show:
            try:
                ctx = tok.decode(ids[max(0, p - 6):p + 1])
                pt = tok.decode([int(pred[p])]); at = tok.decode([int(tgt[p])])
            except Exception:
                ctx, pt, at = "?", str(int(pred[p])), str(int(tgt[p]))
            mark = "OK" if pred[p] == tgt[p] else "  "
            print(f"      [{mark}] …{ctx!r} -> pred={pt!r} actual={at!r}")

    print("\n" + "=" * 64)
    print(f"  OVERALL  quant={args.quant}  top1-acc={tot_correct/max(tot_tokens,1):.3f}  "
          f"ppl={float(torch.exp(torch.tensor(tot_nll/max(tot_tokens,1)))):.1f}  "
          f"(tokens={tot_tokens})")
    print("=" * 64)


if __name__ == "__main__":
    main()
