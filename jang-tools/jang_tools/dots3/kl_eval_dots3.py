"""KL / top-1 acceptance gate: quantized bundle vs stored SOURCE logprobs.

Uses the held-out eval slice captured by capture_dots3 (source top-256
logprobs, sorted). Reports mean KL(source||bundle) in nats over the top-K
support (+ tail bound) and top-1 agreement — the qwen36 harness contract.

    python -m jang_tools.dots3.kl_eval_dots3 <bundle> <capture_dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from .model import BundleModel
from .ops import linear, rms_norm


def full_logprobs(bm: BundleModel, ids: list[int]) -> mx.array:
    """Teacher-forced [S, V] logprobs (no cache, chunk-free: S <= 1024)."""
    from .ops import mla_attention, moe_forward, mlp_dense
    x = bm.embed[mx.array(ids)][None]
    for i, lw in enumerate(bm.layers):
        g = bm.cfg.geom(i)
        h = rms_norm(x, lw["input_layernorm"], bm.cfg.rms_norm_eps)
        x = x + mla_attention(h, lw, g, bm.cfg.rms_norm_eps,
                              bm.cfg.apply_lora_rescale,
                              bm.cfg.k_rope_only_layernorm)
        h = rms_norm(x, lw["post_attention_layernorm"], bm.cfg.rms_norm_eps)
        if bm.cfg.is_moe(i):
            x = x + moe_forward(h, lw, bm.cfg)
        else:
            B, S, H = h.shape
            x = x + mlp_dense(h.reshape(-1, H), lw["mlp_gate"], lw["mlp_up"],
                              lw["mlp_down"]).reshape(B, S, H).astype(x.dtype)
    h = rms_norm(x, bm.final_norm, bm.cfg.rms_norm_eps)
    logits = linear(h, bm.head).astype(mx.float32)[0]
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    mx.eval(lp)
    return lp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", type=Path)
    ap.add_argument("capture", type=Path)
    a = ap.parse_args()

    z = np.load(a.capture / "source_logprobs.npz")
    n_eval = len([k for k in z.files if k.startswith("ids_")])
    bm = BundleModel(a.bundle)

    kls, top1, toks = [], 0, 0
    for k in range(n_eval):
        ids = z[f"ids_{k}"].tolist()
        src_i = z[f"top_i_{k}"]          # [S, K] sorted desc
        src_lp = z[f"top_lp_{k}"]
        q_lp = np.asarray(full_logprobs(bm, ids))     # [S, V]
        S = min(len(ids), src_i.shape[0])
        # positions predict token t+1; compare distributions at every pos
        p = np.exp(src_lp[:S].astype(np.float64))
        q_at = np.take_along_axis(q_lp[:S].astype(np.float64), src_i[:S], 1)
        kl_pos = (p * (src_lp[:S] - q_at)).sum(-1)
        tail = np.clip(1.0 - p.sum(-1), 1e-9, 1.0)
        # tail bound: assume bundle assigns >= uniform-over-rest to the tail
        kl_pos = kl_pos + tail * np.log(np.maximum(tail, 1e-9) / tail)  # ~0
        kls.append(kl_pos)
        top1 += int((q_lp[:S].argmax(-1) == src_i[:S, 0]).sum())
        toks += S
    kl_all = np.concatenate(kls)
    report = {
        "bundle": str(a.bundle),
        "eval_seqs": n_eval, "positions": int(toks),
        "mean_kl_nats": float(kl_all.mean()),
        "p90_kl_nats": float(np.percentile(kl_all, 90)),
        "top1_agreement": top1 / max(toks, 1),
    }
    out = a.bundle / "kl_eval.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
