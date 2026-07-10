"""AWQ activation capture for Tencent Hy3 (one weight-streaming pass, bf16 source).

Created by Jinho Jang (eric@jangq.ai) — 2026-07-09.

Streams the 597 GB BF16 Hy3 source layer-by-layer (never holds the model) over
a batch of equal-length calibration sequences and accumulates, per sparse MoE
layer, the per-input-channel max of |post_attention_layernorm output| — the
input the router + shared expert + routed experts all read. Emits AWQ scales

    s = clip((max|x| + eps)^alpha, min=1.0)

in the layout jang_tools.convert_hy3_jang consumes:

    model.layers.{li}.mlp.input_scale   (hidden,) fp32
    model.layers.{li}.mlp.input_max     (hidden,) fp32   (diagnostics)

The forward mirrors the serving path exactly:
  - standard RMSNorm (not gemma-style)
  - GQA 64/8 heads, head_dim 128, per-head q/k RMSNorm BEFORE RoPE,
    rope_theta 11158840, full-dim rotary
  - sigmoid router + e_score_correction (expert_bias) top-8 selection,
    weights = sigmoid scores of the selected experts (bias only for choice),
    route_norm (normalize to sum 1) then * router_scaling_factor (2.826) —
    same math as mlx_lm.models.dots1 group_expert_select with n_group=1
  - 1 shared expert + silu everywhere; layer 0 dense

Usage (on the box, external drive source):
  python3 -m jang_tools.hy3.awq_capture \
      --model /Volumes/EricsLLMDrive/sources/Hy3 \
      --calib /Volumes/EricsLLMDrive/sources/vera-agentic-coder/vera-keep.jsonl \
      --out   /Volumes/EricsLLMDrive/sources/Hy3-awq-scales.safetensors \
      --batch 8 --seq-len 512 --device mps
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file

EPS = 1e-6
SCALE_FLOOR = 1.0


class Hy3Streamer:
    """Lazy per-tensor loader over the source weight_map (bf16 -> compute dtype)."""

    def __init__(self, src: Path, device, dtype=torch.bfloat16):
        self.src = src
        self.wm = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
        self.device = device
        self.dtype = dtype
        self._handles: dict[str, object] = {}

    def get(self, name: str, dtype=None) -> torch.Tensor:
        fn = self.wm[name]
        if fn not in self._handles:
            self._handles[fn] = safe_open(str(self.src / fn), framework="pt")
        t = self._handles[fn].get_tensor(name)
        return t.to(self.device, dtype or self.dtype)

    def has(self, name: str) -> bool:
        return name in self.wm


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * w.float()).to(x.dtype)


def precompute_rope(head_dim: int, T: int, theta: float, device, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(T, device=device).float()
    freqs = torch.outer(t, inv)                       # (T, hd/2)
    emb = torch.cat([freqs, freqs], dim=-1)           # (T, hd)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, n_heads, T, hd); mlx-style half-rotation
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos[None, None] + rotated * sin[None, None]


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--max-layers", type=int, default=None,
                    help="debug: stop after N layers (partial scales, do NOT convert with them)")
    args = ap.parse_args()

    device = torch.device(args.device)
    src = Path(args.model)
    cfg = json.loads((src / "config.json").read_text())
    assert cfg.get("model_type") == "hy_v3", cfg.get("model_type")

    NL = int(cfg["num_hidden_layers"])
    H = int(cfg["hidden_size"])
    n_heads = int(cfg["num_attention_heads"])
    n_kv = int(cfg["num_key_value_heads"])
    hd = int(cfg["head_dim"])
    NE = int(cfg["num_experts"])
    top_k = int(cfg.get("num_experts_per_tok", 8))
    eps = float(cfg.get("rms_norm_eps", 1e-5))
    theta = float(cfg.get("rope_parameters", {}).get("rope_theta",
                  cfg.get("rope_theta", 10000.0)))
    scaling = float(cfg.get("router_scaling_factor", 1.0))
    route_norm = bool(cfg.get("route_norm", True))

    st = Hy3Streamer(src, device)

    # ── calibration batch (canonical Vera mix, equal-length → no padding) ──
    from jang_tools.minimax_m3.probe import _load_tokenizer, _vera_samples
    tok = _load_tokenizer(str(src))
    texts = _vera_samples(args.calib, args.batch * 3,
                          domains=("coding", "agentic", "shell", "security",
                                   "general", "math", "arithmetic", "stem",
                                   "knowledge"))
    seqs = []
    for t in texts:
        ids = tok.encode(t)
        if len(ids) >= args.seq_len:
            seqs.append(ids[: args.seq_len])
        if len(seqs) >= args.batch:
            break
    if not seqs:
        raise SystemExit("no calibration sequences >= seq_len; lower --seq-len")
    B, T = len(seqs), args.seq_len
    input_ids = torch.tensor(seqs, device=device)
    print(f"  Hy3 AWQ capture: B={B} T={T} ({B*T} tokens) device={device} "
          f"alpha={args.alpha} layers={NL}", flush=True)

    embed = st.get("model.embed_tokens.weight")
    h = embed[input_ids.reshape(-1)].reshape(B, T, H)
    del embed

    cos, sin = precompute_rope(hd, T, theta, device, h.dtype)
    causal = torch.triu(torch.full((T, T), float("-inf"), device=device), 1)

    act_max: dict[int, np.ndarray] = {}
    n_run = NL if args.max_layers is None else min(args.max_layers, NL)
    t0 = time.time()

    for li in range(n_run):
        tl = time.time()
        pre = f"model.layers.{li}"

        # ── attention ──
        r = h
        hn = rms_norm(h, st.get(f"{pre}.input_layernorm.weight"), eps)
        q = (hn @ st.get(f"{pre}.self_attn.q_proj.weight").T)
        k = (hn @ st.get(f"{pre}.self_attn.k_proj.weight").T)
        v = (hn @ st.get(f"{pre}.self_attn.v_proj.weight").T)
        q = q.view(B, T, n_heads, hd)
        k = k.view(B, T, n_kv, hd)
        # per-head q/k RMSNorm BEFORE RoPE
        qn = st.get(f"{pre}.self_attn.q_norm.weight")
        kn = st.get(f"{pre}.self_attn.k_norm.weight")
        q = rms_norm(q, qn, eps).transpose(1, 2)          # (B, nh, T, hd)
        k = rms_norm(k, kn, eps).transpose(1, 2)          # (B, nkv, T, hd)
        v = v.view(B, T, n_kv, hd).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k.repeat_interleave(n_heads // n_kv, dim=1),
            v.repeat_interleave(n_heads // n_kv, dim=1),
            attn_mask=causal,
        )
        out = out.transpose(1, 2).reshape(B, T, n_heads * hd)
        h = r + out @ st.get(f"{pre}.self_attn.o_proj.weight").T

        # ── mlp ──
        r = h
        hpost = rms_norm(h, st.get(f"{pre}.post_attention_layernorm.weight"), eps)
        is_moe = st.has(f"{pre}.mlp.router.gate.weight")

        if not is_moe:
            g = st.get(f"{pre}.mlp.gate_proj.weight")
            u = st.get(f"{pre}.mlp.up_proj.weight")
            d = st.get(f"{pre}.mlp.down_proj.weight")
            h = r + (torch.nn.functional.silu(hpost @ g.T) * (hpost @ u.T)) @ d.T
            del g, u, d
            print(f"    L{li:2d} dense {time.time()-tl:5.1f}s "
                  f"finite={torch.isfinite(h).all().item()}", flush=True)
            continue

        # capture: the experts/router/shared input
        cmax = hpost.float().abs().amax(dim=(0, 1)).cpu().numpy()
        prev = act_max.get(li)
        act_max[li] = cmax if prev is None else np.maximum(prev, cmax)

        # router (dots1 group_expert_select, n_group=1): sigmoid scores;
        # bias ONLY biases the choice; weights are the raw sigmoid scores.
        gate_w = st.get(f"{pre}.mlp.router.gate.weight", dtype=torch.float32)
        bias = st.get(f"{pre}.mlp.expert_bias", dtype=torch.float32)
        logits = hpost.float() @ gate_w.T                    # (B,T,E)
        scores = torch.sigmoid(logits)
        _, sel = (scores + bias).topk(top_k, dim=-1)         # (B,T,K)
        wsel = scores.gather(-1, sel)                        # (B,T,K)
        if route_norm:
            wsel = wsel / (wsel.sum(-1, keepdim=True) + 1e-20)
        wsel = (wsel * scaling).to(h.dtype)

        # shared expert
        sg = st.get(f"{pre}.mlp.shared_mlp.gate_proj.weight")
        su = st.get(f"{pre}.mlp.shared_mlp.up_proj.weight")
        sd = st.get(f"{pre}.mlp.shared_mlp.down_proj.weight")
        moe_out = (torch.nn.functional.silu(hpost @ sg.T) * (hpost @ su.T)) @ sd.T
        del sg, su, sd

        # routed experts — stream one expert at a time, gather its tokens
        flat = hpost.reshape(-1, H)
        sel_flat = sel.reshape(-1, top_k)
        w_flat = wsel.reshape(-1, top_k)
        out_flat = torch.zeros_like(flat)
        for e in range(NE):
            tok_idx, slot = (sel_flat == e).nonzero(as_tuple=True)
            if tok_idx.numel() == 0:
                continue
            xe = flat[tok_idx]
            ge = st.get(f"{pre}.mlp.experts.{e}.gate_proj.weight")
            ue = st.get(f"{pre}.mlp.experts.{e}.up_proj.weight")
            de = st.get(f"{pre}.mlp.experts.{e}.down_proj.weight")
            ye = (torch.nn.functional.silu(xe @ ge.T) * (xe @ ue.T)) @ de.T
            out_flat.index_add_(0, tok_idx, ye * w_flat[tok_idx, slot][:, None])
            del ge, ue, de
        h = r + moe_out + out_flat.reshape(B, T, H)

        mxv = float(act_max[li].max())
        print(f"    L{li:2d} moe   {time.time()-tl:5.1f}s act_max={mxv:.2f} "
              f"finite={torch.isfinite(h).all().item()}", flush=True)

    out = {}
    for li, vec in act_max.items():
        s = np.power(vec + EPS, args.alpha).astype(np.float32)
        s = np.maximum(s, SCALE_FLOOR).astype(np.float32)
        out[f"model.layers.{li}.mlp.input_scale"] = s
        out[f"model.layers.{li}.mlp.input_max"] = vec.astype(np.float32)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_file(out, args.out)
    print(f"\n  wrote {len(act_max)} layer scales -> {args.out} "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
