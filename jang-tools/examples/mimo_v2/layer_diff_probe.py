"""Layer-local source-vs-MLX probe for MiMo-V2.5 JANG bundles.

This is a diagnostic script for coherence bring-up. It streams source FP8
weights layer by layer, runs the in-tree MLX bundle on the same prompt, and
prints relative RMSE after each decoder layer. It is intentionally narrow and
slow; use small ``--layers`` values first.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F
from mlx_lm.utils import load

from jang_tools.mimo_v2 import mlx_register  # noqa: F401
from jang_tools.mimo_v2.weight_loader import MiMoShardIndex


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    y = x.float()
    y = y * torch.rsqrt(torch.mean(y * y, dim=-1, keepdim=True) + eps)
    return y * weight.float()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, *, rope_dim: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = q.shape[-2]
    inv_freq = 1.0 / (base ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
    pos = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().view(1, 1, seq_len, rope_dim)
    sin = emb.sin().view(1, 1, seq_len, rope_dim)
    q_rope, q_nope = q[..., :rope_dim], q[..., rope_dim:]
    k_rope, k_nope = k[..., :rope_dim], k[..., rope_dim:]
    q_out = torch.cat((q_rope * cos + rotate_half(q_rope) * sin, q_nope), dim=-1)
    k_out = torch.cat((k_rope * cos + rotate_half(k_rope) * sin, k_nope), dim=-1)
    return q_out, k_out


class SourceRunner:
    def __init__(self, src: Path):
        self.src = src
        self.idx = MiMoShardIndex(src)
        self.cfg = json.loads((src / "config.json").read_text())
        self.eps = float(self.cfg["layernorm_epsilon"])
        self.hidden = int(self.cfg["hidden_size"])
        self.n_heads = int(self.cfg["num_attention_heads"])
        self.n_kv = int(self.cfg["num_key_value_heads"])
        self.swa_heads = int(self.cfg["swa_num_attention_heads"])
        self.swa_kv = int(self.cfg["swa_num_key_value_heads"])
        self.head_dim = int(self.cfg["head_dim"])
        self.v_dim = int(self.cfg["v_head_dim"])
        self.swa_head_dim = int(self.cfg["swa_head_dim"])
        self.swa_v_dim = int(self.cfg["swa_v_head_dim"])
        self.rope_dim = int(self.head_dim * float(self.cfg["partial_rotary_factor"]))
        self.top_k = int(self.cfg["num_experts_per_tok"])
        self.n_experts = int(self.cfg["n_routed_experts"])

    def tensor(self, name: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return self.idx.read_tensor(name, out_dtype=dtype)

    @lru_cache(maxsize=128)
    def cached_tensor(self, name: str) -> torch.Tensor:
        return self.tensor(name)

    def embed(self, ids: list[int]) -> torch.Tensor:
        w = self.idx.read_passthrough("model.embed_tokens.weight", out_dtype=torch.float32)
        return w[torch.tensor(ids, dtype=torch.long)].unsqueeze(0)

    def attention(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        is_swa = int(self.cfg["hybrid_layer_pattern"][layer_idx]) == 1
        if is_swa:
            n_heads = self.swa_heads
            n_kv = self.swa_kv
            head_dim = self.swa_head_dim
            v_dim = self.swa_v_dim
            theta = float(self.cfg["swa_rope_theta"])
            sliding = int(self.cfg["sliding_window"])
        else:
            n_heads = self.n_heads
            n_kv = self.n_kv
            head_dim = self.head_dim
            v_dim = self.v_dim
            theta = float(self.cfg["rope_theta"])
            sliding = None
        q_size = n_heads * head_dim
        k_size = n_kv * head_dim
        v_size = n_kv * v_dim

        qkv_w = self.tensor(f"model.layers.{layer_idx}.self_attn.qkv_proj.weight")
        qkv = F.linear(x, qkv_w)
        q, k, v = qkv.split([q_size, k_size, v_size], dim=-1)
        bsz, seq_len, _ = x.shape
        q = q.view(bsz, seq_len, n_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, n_kv, head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, n_kv, v_dim).transpose(1, 2)
        v = v * float(self.cfg["attention_value_scale"])
        q, k = apply_rope(q, k, rope_dim=self.rope_dim, base=theta)
        rep = n_heads // n_kv
        if rep != 1:
            k = k[:, :, None, :, :].expand(bsz, n_kv, rep, seq_len, head_dim).reshape(bsz, n_heads, seq_len, head_dim)
            v = v[:, :, None, :, :].expand(bsz, n_kv, rep, seq_len, v_dim).reshape(bsz, n_heads, seq_len, v_dim)
        attn = torch.matmul(q, k.transpose(2, 3)) * (head_dim ** -0.5)
        i = torch.arange(seq_len).view(seq_len, 1)
        j = torch.arange(seq_len).view(1, seq_len)
        allowed = j <= i
        if sliding is not None:
            allowed = allowed & (j >= i - sliding + 1)
        attn = attn.masked_fill(~allowed.view(1, 1, seq_len, seq_len), float("-inf"))
        if is_swa and bool(self.cfg.get("add_swa_attention_sink_bias", False)):
            sink = self.idx.read_passthrough(
                f"model.layers.{layer_idx}.self_attn.attention_sink_bias",
                out_dtype=torch.float32,
            ).view(1, n_heads, 1, 1).expand(bsz, n_heads, seq_len, 1)
            attn = torch.cat((attn, sink), dim=-1)
        probs = F.softmax(attn.float() - attn.float().amax(dim=-1, keepdim=True), dim=-1).to(q.dtype)
        if is_swa and bool(self.cfg.get("add_swa_attention_sink_bias", False)):
            probs = probs[..., :-1]
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(bsz, seq_len, n_heads * v_dim)
        o_w = self.idx.read_passthrough(f"model.layers.{layer_idx}.self_attn.o_proj.weight", out_dtype=torch.float32)
        return F.linear(out, o_w)

    def dense_mlp(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        gate = self.tensor(f"model.layers.{layer_idx}.mlp.gate_proj.weight")
        up = self.tensor(f"model.layers.{layer_idx}.mlp.up_proj.weight")
        down = self.tensor(f"model.layers.{layer_idx}.mlp.down_proj.weight")
        return F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)

    def moe(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden = x.shape
        xf = x.reshape(-1, hidden)
        gate_w = self.idx.read_passthrough(f"model.layers.{layer_idx}.mlp.gate.weight", out_dtype=torch.float32)
        bias = self.idx.read_passthrough(
            f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias",
            out_dtype=torch.float32,
        )
        scores = torch.sigmoid(F.linear(xf.float(), gate_w.float()))
        _, topk_idx = torch.topk(scores + bias.view(1, -1), k=self.top_k, dim=-1, sorted=False)
        topk_w = scores.gather(1, topk_idx)
        topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-20)
        out = torch.zeros_like(xf)
        for expert_idx in torch.unique(topk_idx).tolist():
            slots = topk_idx == int(expert_idx)
            token_idx, slot_idx = torch.where(slots)
            if token_idx.numel() == 0:
                continue
            expert_x = xf[token_idx]
            gate = self.cached_tensor(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight")
            up = self.cached_tensor(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight")
            down = self.cached_tensor(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight")
            expert_y = F.linear(F.silu(F.linear(expert_x, gate)) * F.linear(expert_x, up), down)
            out.index_add_(0, token_idx, expert_y * topk_w[token_idx, slot_idx].unsqueeze(-1))
        return out.view(bsz, seq_len, hidden)

    def layer(self, layer_idx: int, h: torch.Tensor) -> torch.Tensor:
        ln1 = self.idx.read_passthrough(f"model.layers.{layer_idx}.input_layernorm.weight", out_dtype=torch.float32)
        h = h + self.attention(layer_idx, rmsnorm(h, ln1, self.eps))
        ln2 = self.idx.read_passthrough(f"model.layers.{layer_idx}.post_attention_layernorm.weight", out_dtype=torch.float32)
        x = rmsnorm(h, ln2, self.eps)
        if int(self.cfg["moe_layer_freq"][layer_idx]):
            return h + self.moe(layer_idx, x)
        return h + self.dense_mlp(layer_idx, x)


def rel_stats(src: torch.Tensor, actual: mx.array) -> tuple[float, float, float]:
    a = torch.from_numpy(np.array(actual.astype(mx.float32)))
    s = src.float()
    d = s - a
    rmse = torch.sqrt(torch.mean(d * d))
    rms = torch.sqrt(torch.mean(s * s)) + 1e-12
    last_d = s[:, -1, :] - a[:, -1, :]
    last_rmse = torch.sqrt(torch.mean(last_d * last_d))
    last_rms = torch.sqrt(torch.mean(s[:, -1, :] * s[:, -1, :])) + 1e-12
    return float(rmse / rms), float(last_rmse / last_rms), float(d.abs().max())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--prompt", default="What is 2 + 2? Answer in one short sentence.")
    parser.add_argument("--thinking", choices=("default", "on", "off"), default="off")
    args = parser.parse_args()

    model, tokenizer = load(str(args.bundle), lazy=True, tokenizer_config={"trust_remote_code": True})
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if args.thinking == "on":
        template_kwargs["enable_thinking"] = True
    elif args.thinking == "off":
        template_kwargs["enable_thinking"] = False
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}], **template_kwargs)
    ids = tokenizer.encode(prompt)
    mx_ids = mx.array([ids], dtype=mx.int32)

    src = SourceRunner(args.src)
    h_src = src.embed(ids)
    h_mx = model.model.embed_tokens(mx_ids)
    mx.eval(h_mx)
    rel, last_rel, maxerr = rel_stats(h_src, h_mx)
    print(f"embed rel={rel:.6f} last_rel={last_rel:.6f} max={maxerr:.6f}")

    for layer_idx in range(args.layers):
        h_src = src.layer(layer_idx, h_src)
        layer = model.model.layers[layer_idx]
        h_mx = layer(h_mx, mask="causal", cache=None)
        mx.eval(h_mx)
        rel, last_rel, maxerr = rel_stats(h_src, h_mx)
        print(f"layer {layer_idx:02d} rel={rel:.6f} last_rel={last_rel:.6f} max={maxerr:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
