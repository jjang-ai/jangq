"""Plain-torch stand-ins for the release's TileLang kernels (CPU parity tests only).

Activation quant round-trips are identity here, so the reference runs the same
bf16/fp32 numerics as the MLX port. sparse_attn and hc_split_sinkhorn follow the
kernel semantics exactly.
"""

import torch


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    if inplace:
        return x
    raise NotImplementedError("fp8 GEMM path is not used in bf16 parity mode")


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=None):
    if inplace:
        return x
    raise NotImplementedError


def fp8_gemm(*a, **k):
    raise NotImplementedError


def fp4_gemm(*a, **k):
    raise NotImplementedError


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    b, s, h, d = q.shape
    valid = topk_idxs >= 0
    idx = topk_idxs.clamp_min(0).long()
    g = torch.gather(kv.float().unsqueeze(1).expand(b, s, *kv.shape[1:]), 2, idx[..., None].expand(*idx.shape, d))
    scores = torch.einsum("bshd,bskd->bshk", q.float(), g) * softmax_scale
    scores = scores.masked_fill(~valid[:, :, None, :], float("-inf"))
    m = torch.maximum(scores.amax(-1, keepdim=True), attn_sink.float()[None, None, :, None])
    p = torch.exp(scores - m)
    denom = p.sum(-1, keepdim=True) + torch.exp(attn_sink.float()[None, None, :, None] - m)
    o = torch.einsum("bshk,bskd->bshd", p, g) / denom
    return o.to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb
