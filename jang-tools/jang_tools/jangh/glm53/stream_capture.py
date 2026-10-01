"""Layer-streaming bf16 forward of the FULL GLM-5.3-Flash source on a 128 GB Mac.

The 642 GB bf16 source never fits in memory; its 45 decoder layers are loaded ONE AT A TIME from SSD and every
sequence's hidden state (4 mHC streams x T x 4096, bf16) is pushed through that layer before the next is loaded.
Uses the parity-proven jang_tools.glm5_next.modeling (dense-DSA bypass is exact for T <= 2048).

Outputs
  --stats-out : per routed-MoE layer (same schema as the FP8 capture's diag.safetensors so tools can pool them):
                <L>.mlp.experts.{sum_x2, count, expert_sum_x2, expert_count, rows, rows_topk_idx, rows_topk_w}
                + <L>.mlp.switch_mlp.down_proj.expert_sum_a2 (per-expert down-input E[a^2] numerator)
  --ref-out   : for sequences flagged "ref": input_ids + top-128 logprobs per position (klref.safetensors schema),
                i.e. a bf16 REFERENCE for held-out agentic/cybersec prompts.

Input jsonl: {"ids": [...token ids...], "ref": bool, "domain": str}. Build it with build_agentic_corpus.py.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


from jang_tools.glm5_next.modeling import Glm5Args, DecoderLayer, RMSNorm  # noqa: E402
from jang_tools.glm5_next import load as ref_load  # noqa: E402

E = 288
CURRENT = {"ref": False}   # set per sequence during the streamed forward


def layer_weights(src: Path, wm: dict, i: int, cache: dict) -> dict:
    """Source tensors of layer i, sanitized to runtime names, prefix stripped (for DecoderLayer.load_weights)."""
    pre = f"model.language_model.layers.{i}."
    keys = [k for k in wm if k.startswith(pre)]
    raw = {}
    for f in sorted({wm[k] for k in keys}):
        if f not in cache:
            cache.clear(); mx.clear_cache()
            cache[f] = mx.load(str(src / f))
        for k in keys:
            if wm[k] == f:
                raw[k] = cache[f][k]
    san = ref_load.sanitize(raw, num_layers=i + 1)
    out = {}
    for k, v in san.items():
        k2 = k[len(f"model.layers.{i}."):]
        out[k2] = v if v.dtype == mx.float32 or v.dtype == mx.uint32 else v.astype(mx.bfloat16)
    return out


class MoEStats:
    def __init__(self, reservoir: int, seed: int):
        self.rng = np.random.default_rng(seed)
        self.R = reservoir
        self.sum_x2 = np.zeros(4096, np.float64); self.count = 0
        self.e_sum_x2 = np.zeros((E, 4096), np.float64); self.e_count = np.zeros(E, np.float64)
        self.e_sum_a2 = np.zeros((E, 2048), np.float64)
        self.rows, self.tidx, self.tw = [], [], []
        self.seen = 0

    def add(self, x: np.ndarray, idx: np.ndarray, w: np.ndarray, a2_by_expert: dict):
        self.sum_x2 += (x.astype(np.float64) ** 2).sum(0); self.count += x.shape[0]
        for e in np.unique(idx):
            r = np.nonzero((idx == e).any(1))[0]
            self.e_sum_x2[e] += (x[r].astype(np.float64) ** 2).sum(0); self.e_count[e] += len(r)
        for e, a2 in a2_by_expert.items():
            self.e_sum_a2[e] += a2
        for t in range(x.shape[0]):          # reservoir sampling (Algorithm R)
            self.seen += 1
            if len(self.rows) < self.R:
                self.rows.append(x[t]); self.tidx.append(idx[t]); self.tw.append(w[t])
            else:
                j = self.rng.integers(0, self.seen)
                if j < self.R:
                    self.rows[j] = x[t]; self.tidx[j] = idx[t]; self.tw[j] = w[t]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seqs", required=True)
    ap.add_argument("--stats-out", required=True)
    ap.add_argument("--ref-out", required=True)
    ap.add_argument("--reservoir", type=int, default=4096)
    ap.add_argument("--max-layers", type=int, default=0, help="debug: stop after N layers")
    a = ap.parse_args()
    src = Path(a.model)
    cfg = json.loads((src / "config.json").read_text())
    args = Glm5Args.from_config(cfg)
    wm = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    seqs = [json.loads(l) for l in open(a.seqs)]
    for s in seqs:
        assert len(s["ids"]) <= args.index_topk, "dense DSA bypass is exact only for T <= index_topk"
    print(f"{len(seqs)} sequences, {sum(len(s['ids']) for s in seqs)} tokens; ref sequences: {sum(bool(s.get('ref')) for s in seqs)}", flush=True)
    t0 = time.time()
    cache: dict = {}
    # embeddings
    emb = None
    for k in ("model.language_model.embed_tokens.weight",):
        emb = mx.load(str(src / wm[k]))[k].astype(mx.bfloat16)
    streams = []
    for s in seqs:
        x = emb[mx.array(s["ids"])][None]                                  # (1, T, D)
        st = mx.broadcast_to(x[:, :, None, :], (1, x.shape[1], args.hc_mult, x.shape[-1])).astype(mx.bfloat16)
        mx.eval(st); streams.append(st)
    del emb; mx.clear_cache()
    stats_out = {}
    nL = args.num_hidden_layers if not a.max_layers else a.max_layers
    for i in range(nL):
        tl = time.time()
        layer = DecoderLayer(args, i)
        layer.load_weights(list(layer_weights(src, wm, i, cache).items()), strict=True)
        mx.eval(layer.parameters())
        is_moe = i >= args.first_k_dense_replace
        st_i = MoEStats(a.reservoir, seed=i) if is_moe else None
        if is_moe:
            moe = layer.mlp
            orig = type(moe).__call__

            def capture_call(self, x, _st=st_i):
                # held-out ("ref") sequences are never allowed into calibration statistics (no eval contamination)
                collect = not CURRENT["ref"]
                logits = x.astype(mx.float32) @ self.gate.weight.astype(mx.float32).T
                scores = mx.sigmoid(logits)
                choice = scores + self.e_score_correction_bias.astype(mx.float32)
                idx = mx.argpartition(-choice, kth=self.k - 1, axis=-1)[..., : self.k]
                w = mx.take_along_axis(scores, idx, axis=-1)
                if self.norm_topk:
                    w = w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20)
                w = w * self.scaling
                sw = self.switch_mlp
                xe = mx.expand_dims(x, (-2, -3))
                g = sw.gate_proj(xe, idx); u = sw.up_proj(xe, idx)
                act = sw.activation(u, g)                                  # clamped SwiGLU, (B,T,k,1,I)
                y = sw.down_proj(act, idx).squeeze(-2)
                routed = mx.sum(y * w[..., None].astype(y.dtype), axis=-2)
                if not collect:
                    return routed.astype(x.dtype) + self.shared_experts(x)
                xf = np.asarray(x.reshape(-1, x.shape[-1]).astype(mx.float32))
                idn = np.asarray(idx.reshape(-1, self.k)); wn = np.asarray(w.reshape(-1, self.k).astype(mx.float32))
                an = np.asarray(act.reshape(-1, self.k, act.shape[-1]).astype(mx.float32))
                a2 = {}
                for e in np.unique(idn):
                    r, c = np.nonzero(idn == e)
                    a2[int(e)] = (an[r, c].astype(np.float64) ** 2).sum(0)
                if collect:
                    _st.add(xf, idn, wn, a2)
                return routed.astype(x.dtype) + self.shared_experts(x)
            type(moe).__call__ = capture_call
        for j in range(len(streams)):
            CURRENT["ref"] = bool(seqs[j].get("ref"))
            streams[j] = layer(streams[j])
            mx.eval(streams[j])
        if is_moe:
            type(moe).__call__ = orig
        if is_moe and st_i.rows:          # all-ref corpora collect no calibration stats
            base = f"model.language_model.layers.{i}.mlp.experts"
            stats_out[base + ".sum_x2"] = mx.array(st_i.sum_x2.astype(np.float32))
            stats_out[base + ".count"] = mx.array(np.array([st_i.count], np.float32))
            stats_out[base + ".expert_sum_x2"] = mx.array(st_i.e_sum_x2.astype(np.float32))
            stats_out[base + ".expert_count"] = mx.array(st_i.e_count.astype(np.float32))
            stats_out[base + ".rows"] = mx.array(np.stack(st_i.rows).astype(np.float16))
            stats_out[base + ".rows_topk_idx"] = mx.array(np.stack(st_i.tidx).astype(np.int32))
            stats_out[base + ".rows_topk_w"] = mx.array(np.stack(st_i.tw).astype(np.float32))
            stats_out[f"model.language_model.layers.{i}.mlp.switch_mlp.down_proj.expert_sum_a2"] = mx.array(st_i.e_sum_a2.astype(np.float32))
            mx.save_safetensors(a.stats_out, stats_out)
        del layer; mx.clear_cache()
        print(f"layer {i:2d} {'moe' if is_moe else 'dense'} {time.time()-tl:6.1f}s  total {(time.time()-t0)/60:.1f} min", flush=True)
    if a.max_layers:
        return
    # final norm + lm_head -> reference top-128 logprobs for "ref" sequences
    norm = RMSNorm(args.hidden_size, args.rms_norm_eps)
    nk = "model.language_model.norm.weight"
    norm.weight = mx.load(str(src / wm[nk]))[nk].astype(mx.bfloat16)
    head = mx.load(str(src / wm["lm_head.weight"]))["lm_head.weight"].astype(mx.bfloat16)
    ref = {}
    for j, s in enumerate(seqs):
        if not s.get("ref"):
            continue
        h = norm(mx.mean(streams[j], axis=2))
        logits = (h @ head.T)[0].astype(mx.float32)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        top = mx.argsort(-lp, axis=-1)[:, :128]
        ref[f"p{j}.input_ids"] = mx.array(np.array(s["ids"], np.int32))
        ref[f"p{j}.top_ids"] = top.astype(mx.int32)
        ref[f"p{j}.top_logprobs"] = mx.take_along_axis(lp, top, axis=-1)
        mx.eval(ref[f"p{j}.top_logprobs"])
    mx.save_safetensors(a.ref_out, ref, metadata={"reference_precision": "BF16 (layer-streamed source)"})
    print(f"DONE stats {len(stats_out)} tensors, ref prompts {len(ref)//3}, {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
