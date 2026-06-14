"""Single-layer streamed forward for MiniMax-M3 (minimax_m3_vl) in pure torch.

Computes hidden[L+1] = layer_L(hidden[L]) one layer at a time from weights
read off disk. Peak memory is one decoder layer (worst case one MoE layer's
128 experts) plus the cached hidden states. No full model is materialized.

Architecture (text_config), faithful to transformers PR #46600
(`modeling_minimax_m3_vl.py`):

  - RMSNorm: GEMMA style -> x * rsqrt(mean(x^2)+eps) * (1 + weight)
    (use_gemma_norm=True; applies to EVERY norm incl. qk-norm + indexer norms)
  - Attention: GQA (64 q / 4 kv heads, head_dim 128), per-head Gemma qk-norm
    applied BEFORE rope, PARTIAL rope (rotary_dim=64 of 128, rest pass-through),
    rope_theta=5e6, scale = head_dim**-0.5.
    Sparse layers add a Lightning Indexer (MSA) that selects top-k 128-token
    key blocks. At context length < topk_blocks*block_size (=2048) every block
    is visible, so MSA == full causal attention -> the indexer is a NO-OP for
    short-context probing and is skipped here. (Long-context selection
    correctness is a runtime concern owned by the vMLX agents.)
  - MoE (layers >= first dense): sigmoid router + e_score_correction_bias
    (bias used for SELECTION only; gate values are the un-biased sigmoid,
    renormalized over the top-k), routed_scaling_factor on the routed sum,
    one always-on shared expert.
  - swigluoai activation (GPT-OSS style, NOT interleaved), shared by dense MLP,
    routed experts and the shared expert:
        gate = clamp(gate, max=limit)
        up   = clamp(up, -limit, +limit)
        glu  = gate * sigmoid(alpha * gate)
        out  = down( (up + 1.0) * glu )
    alpha=swiglu_alpha (1.702), limit=swiglu_limit (7.0).

For MoE layers the forward also accumulates per-expert REAP saliency
(sum of gate * ||expert(x)||_2) in the same pass.

Created by Jinho Jang (eric@jangq.ai).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


# ----- config -------------------------------------------------------

@dataclass
class M3Cfg:
    hidden_size: int = 6144
    num_hidden_layers: int = 60
    # per-layer dispatch (filled from config)
    mlp_layer_types: list[str] | None = None        # "sparse" | "dense"
    layer_types: list[str] | None = None            # "minimax_m3_sparse" | "full_attention"

    # attention
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    head_dim: int = 128
    rotary_dim: int = 64
    rope_theta: float = 5_000_000.0
    use_qk_norm: bool = True

    # MoE
    num_local_experts: int = 128
    num_experts_per_tok: int = 4
    n_shared_experts: int = 1
    moe_intermediate_size: int = 3072       # routed expert inter (config.intermediate_size)
    shared_intermediate_size: int = 3072
    dense_intermediate_size: int = 12288
    routed_scaling_factor: float = 2.0
    norm_topk_prob: bool = True

    # activation
    swiglu_alpha: float = 1.702
    swiglu_limit: float = 7.0

    rms_norm_eps: float = 1e-6
    vocab_size: int = 200064

    @classmethod
    def from_config_json(cls, cfg_path) -> "M3Cfg":
        d = json.loads(Path(cfg_path).read_text())
        t = d.get("text_config", d)
        nl = t["num_hidden_layers"]
        sparse_cfg = t.get("sparse_attention_config", {}) or {}
        moe_freq = t.get("moe_layer_freq")
        attn_freq = sparse_cfg.get("sparse_attention_freq")
        mlp_types = (["sparse" if f else "dense" for f in moe_freq]
                     if moe_freq is not None else ["sparse"] * nl)
        layer_types = (["minimax_m3_sparse" if f else "full_attention" for f in attn_freq]
                       if attn_freq is not None else ["full_attention"] * nl)
        return cls(
            hidden_size=t["hidden_size"],
            num_hidden_layers=nl,
            mlp_layer_types=mlp_types,
            layer_types=layer_types,
            num_attention_heads=t["num_attention_heads"],
            num_key_value_heads=t["num_key_value_heads"],
            head_dim=t.get("head_dim", t["hidden_size"] // t["num_attention_heads"]),
            rotary_dim=t.get("rotary_dim", int(t.get("partial_rotary_factor", 1.0)
                                               * t.get("head_dim", 128))),
            rope_theta=t.get("rope_theta", 1e4),
            use_qk_norm=t.get("use_qk_norm", True),
            num_local_experts=t["num_local_experts"],
            num_experts_per_tok=t["num_experts_per_tok"],
            n_shared_experts=t.get("n_shared_experts", 1),
            moe_intermediate_size=t["intermediate_size"],
            shared_intermediate_size=t.get("shared_intermediate_size", t["intermediate_size"]),
            dense_intermediate_size=t.get("dense_intermediate_size", t["intermediate_size"]),
            routed_scaling_factor=t.get("routed_scaling_factor", 1.0),
            norm_topk_prob=True,
            swiglu_alpha=t.get("swiglu_alpha", 1.702),
            swiglu_limit=t.get("swiglu_limit", 7.0),
            rms_norm_eps=t.get("rms_norm_eps", 1e-6),
            vocab_size=t["vocab_size"],
        )

    def is_dense(self, li: int) -> bool:
        return self.mlp_layer_types[li] == "dense"


# ----- building blocks ---------------------------------------------

def gemma_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Gemma-style RMSNorm: normalize in fp32, scale by (1 + weight)."""
    in_dtype = x.dtype
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (x32 * (1.0 + weight.float())).to(in_dtype)


def precompute_rope(rotary_dim: int, max_pos: int, base: float,
                    device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin for the rotated sub-dimension. inv_freq has rotary_dim/2 entries."""
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32,
                                            device=device) / rotary_dim))
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)            # (T, rotary_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)     # (T, rotary_dim)  -> rotate_half layout
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_partial_rope(q: torch.Tensor, k: torch.Tensor,
                       cos: torch.Tensor, sin: torch.Tensor,
                       rotary_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """q,k: (B, H, T, head_dim). Rotate only the first `rotary_dim` channels."""
    cos = cos.unsqueeze(0).unsqueeze(0)   # (1,1,T,rotary_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)

    def rope(x):
        x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
        x_rot = (x_rot * cos) + (_rotate_half(x_rot) * sin)
        return torch.cat([x_rot, x_pass], dim=-1)

    return rope(q), rope(k)


def gqa_attention(
    x: torch.Tensor,
    q_w: torch.Tensor, k_w: torch.Tensor, v_w: torch.Tensor, o_w: torch.Tensor,
    q_norm: torch.Tensor | None, k_norm: torch.Tensor | None,
    cfg: M3Cfg, cos: torch.Tensor, sin: torch.Tensor,
) -> torch.Tensor:
    """GQA with per-head Gemma qk-norm + partial rope, full causal SDPA.

    Short-context probe path: MSA block selection is a no-op (all blocks
    visible) so this exact attention equals the sparse layer's attention.
    """
    B, T, H = x.shape
    Hq, Hkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    dt = x.dtype

    q = (x @ q_w.T.to(dt)).view(B, T, Hq, hd)
    k = (x @ k_w.T.to(dt)).view(B, T, Hkv, hd)
    v = (x @ v_w.T.to(dt)).view(B, T, Hkv, hd)

    # Per-head Gemma qk-norm over the head_dim, BEFORE rope.
    if cfg.use_qk_norm and q_norm is not None:
        q = gemma_rms_norm(q, q_norm, cfg.rms_norm_eps)
        k = gemma_rms_norm(k, k_norm, cfg.rms_norm_eps)

    q = q.transpose(1, 2)   # (B,Hq,T,hd)
    k = k.transpose(1, 2)   # (B,Hkv,T,hd)
    v = v.transpose(1, 2)

    q, k = apply_partial_rope(q, k, cos, sin, cfg.rotary_dim)

    # GQA expand kv heads
    r = Hq // Hkv
    k = k.unsqueeze(2).expand(-1, -1, r, -1, -1).reshape(B, Hq, T, hd)
    v = v.unsqueeze(2).expand(-1, -1, r, -1, -1).reshape(B, Hq, T, hd)

    scale = 1.0 / math.sqrt(hd)
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
    attn = attn.masked_fill(mask, float("-inf"))
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(dt)
    out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, Hq * hd)
    return out @ o_w.T.to(dt)


def swigluoai(x: torch.Tensor, gate_w: torch.Tensor, up_w: torch.Tensor,
              down_w: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """GPT-OSS swiglu (non-interleaved), with the M3 (up + 1.0) shift.

    Chunked 2D matmul to stay clear of the MPS INT_MAX matmul buffer cap.
    """
    MAX_ROWS = 16384
    orig = x.shape
    if x.dim() != 2:
        x = x.reshape(-1, orig[-1])
    gate_wT = gate_w.T.to(x.dtype).contiguous()
    up_wT = up_w.T.to(x.dtype).contiguous()
    down_wT = down_w.T.to(x.dtype).contiguous()
    outs = []
    for s in range(0, x.shape[0], MAX_ROWS):
        xi = x[s:s + MAX_ROWS].contiguous()
        gate = xi @ gate_wT
        up = xi @ up_wT
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
        glu = gate * torch.sigmoid(gate * alpha)
        outs.append(((up + 1.0) * glu) @ down_wT)
    y = torch.cat(outs, 0) if len(outs) > 1 else outs[0]
    if len(orig) != 2:
        y = y.reshape(*orig[:-1], y.shape[-1])
    return y


def moe_forward_observe(
    x: torch.Tensor,                 # (B,T,H)
    router_w: torch.Tensor,          # (E,H)
    router_bias: torch.Tensor | None,  # (E,)
    expert_loader,                   # callable(e)->(gate_w,up_w,down_w) or ("__stacked__")
    shared_expert: dict | None,      # {'gate_proj','up_proj','down_proj'}
    cfg: M3Cfg, device,
    keep_ids: list | None = None,    # REAP: restrict routing to these expert ids
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One MoE layer forward with REAP saliency accumulation.

    Returns (out (B,T,H), saliency_sum[E] f32, count[E] i64).
    saliency_sum[e] = sum over tokens routed to e of  gate_e * ||expert_e(x)||_2.
    """
    B, T, H = x.shape
    E, K = cfg.num_local_experts, cfg.num_experts_per_tok
    alpha, limit = cfg.swiglu_alpha, cfg.swiglu_limit
    xf = x.reshape(-1, H)
    N = xf.shape[0]

    logits = xf.float() @ router_w.T.to(device=device, dtype=torch.float32)
    gates = torch.sigmoid(logits)                       # un-biased gate values
    biased = gates + router_bias.to(device, torch.float32) if router_bias is not None else gates
    if keep_ids is not None:
        # REAP prune: dropped experts can never be selected (router rows removed
        # in the bundle); equivalently mask their selection score to -inf.
        mask = torch.full((E,), float("-inf"), device=device)
        mask[torch.tensor(keep_ids, device=device)] = 0.0
        biased = biased + mask
    _, top_idx = torch.topk(biased, K, dim=-1)          # selection uses bias
    top_g = torch.gather(gates, 1, top_idx)             # un-biased weights
    if cfg.norm_topk_prob:
        top_g = top_g / (top_g.sum(-1, keepdim=True) + 1e-20)
    top_g = top_g * cfg.routed_scaling_factor

    out = torch.zeros_like(xf)
    saliency = torch.zeros(E, dtype=torch.float32, device=device)
    count = torch.zeros(E, dtype=torch.int64, device=device)

    flat_e = top_idx.reshape(-1)
    flat_r = torch.arange(N, device=device).unsqueeze(1).expand(-1, K).reshape(-1)
    flat_w = top_g.reshape(-1)
    order = torch.argsort(flat_e, stable=True)
    e_sorted, r_sorted, w_sorted = flat_e[order], flat_r[order], flat_w[order]
    uniq, counts = torch.unique_consecutive(e_sorted, return_counts=True)
    splits = torch.cumsum(counts, 0)
    starts = torch.cat([torch.zeros(1, dtype=splits.dtype, device=device), splits[:-1]])
    uniq_l, starts_l, splits_l = uniq.cpu().tolist(), starts.cpu().tolist(), splits.cpu().tolist()

    stacked = expert_loader("__stacked__", allow_none=True) \
        if getattr(expert_loader, "supports_stacked", False) else None

    for i, e in enumerate(uniq_l):
        s, end = starts_l[i], splits_l[i]
        rows = r_sorted[s:end]
        w_e = w_sorted[s:end]
        x_e = xf[rows].to(device=device, dtype=x.dtype)
        if stacked is not None:
            gw, uw, dw = stacked[0][e], stacked[1][e], stacked[2][e]
        else:
            gw, uw, dw = expert_loader(e)
        f = swigluoai(x_e, gw, uw, dw, alpha, limit)
        norms = torch.linalg.vector_norm(f.float(), ord=2, dim=-1)
        saliency[e] += (w_e * norms).sum()
        count[e] += norms.numel()
        out[rows] += f * w_e.to(f.dtype).unsqueeze(-1)

    if shared_expert is not None:
        out = out + swigluoai(xf, shared_expert["gate_proj"], shared_expert["up_proj"],
                              shared_expert["down_proj"], alpha, limit)
    return out.reshape(B, T, H), saliency, count


@dataclass
class LayerWeights:
    input_layernorm: torch.Tensor
    post_attention_layernorm: torch.Tensor
    q_w: torch.Tensor
    k_w: torch.Tensor
    v_w: torch.Tensor
    o_w: torch.Tensor
    q_norm: torch.Tensor | None = None
    k_norm: torch.Tensor | None = None
    # dense MLP  OR  (router + experts + shared)
    dense_mlp: dict | None = None
    router_w: torch.Tensor | None = None
    router_bias: torch.Tensor | None = None
    expert_loader: Any | None = None
    shared_expert: dict | None = None
    keep_ids: list | None = None


def decoder_layer_forward(
    x: torch.Tensor, lw: LayerWeights, cfg: M3Cfg,
    cos: torch.Tensor, sin: torch.Tensor, device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Full M3 decoder layer. Returns (hidden, saliency|None, count|None)."""
    r = x
    h = gemma_rms_norm(x, lw.input_layernorm, cfg.rms_norm_eps)
    h = gqa_attention(h, lw.q_w, lw.k_w, lw.v_w, lw.o_w,
                      lw.q_norm, lw.k_norm, cfg, cos, sin)
    h = r + h

    r = h
    h = gemma_rms_norm(h, lw.post_attention_layernorm, cfg.rms_norm_eps)
    sal = cnt = None
    if lw.dense_mlp is not None:
        h = swigluoai(h, lw.dense_mlp["gate_proj"], lw.dense_mlp["up_proj"],
                      lw.dense_mlp["down_proj"], cfg.swiglu_alpha, cfg.swiglu_limit)
    else:
        h, sal, cnt = moe_forward_observe(h, lw.router_w, lw.router_bias,
                                          lw.expert_loader, lw.shared_expert, cfg, device,
                                          keep_ids=lw.keep_ids)
    h = r + h
    return h, sal, cnt
