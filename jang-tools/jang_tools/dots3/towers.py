"""dots3 vision + audio towers in MLX (model forward only).

Preprocessing (image patching block-major, video frame sampling, log-mel
chunking) is driven through the PR transformers processor at capture/probe
time; these classes consume its outputs:
  vision: pixel_values [n_patches, 3*14*14], grid_thw [n,3]
  audio:  input_features [n_chunks, 128, frames], chunk lens, chunk counts

Weights arrive as a flat dict of mx.arrays with source names (fp8 repo keeps
both towers bf16; quantized bundles pass QW wrappers through the same ops).
"""
from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from .ops import QW, linear, rms_norm


# --------------------------------------------------------------------- vision
def vision_position_ids(grid_thw: np.ndarray, merge: int) -> np.ndarray:
    """(total_patches, 2) h/w ids, block-major over merge x merge blocks."""
    out = []
    for t, h, w in grid_thw.tolist():
        hh, ww = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        shape = (h // merge, merge, w // merge, merge)
        hh = hh.reshape(shape).transpose(0, 2, 1, 3).reshape(-1)
        ww = ww.reshape(shape).transpose(0, 2, 1, 3).reshape(-1)
        pos = np.stack([hh, ww], -1)
        out.append(np.tile(pos, (t, 1)))
    return np.concatenate(out, 0)


def vision_cu_seqlens(grid_thw: np.ndarray) -> list[int]:
    """Per-frame attention segments (merge_temporal=False convention)."""
    seq = []
    for t, h, w in grid_thw.tolist():
        seq.extend([h * w] * t)
    cu = [0]
    for s in seq:
        cu.append(cu[-1] + s)
    return cu


class VisionTower:
    def __init__(self, weights: dict, vcfg: dict, eps: float = 1e-5):
        self.w = weights
        self.c = vcfg
        self.eps = vcfg.get("rms_norm_eps", eps)
        self.heads = vcfg["num_attention_heads"]
        self.embed = vcfg["embed_dim"]
        self.head_dim = self.embed // self.heads
        self.merge = vcfg["spatial_merge_size"]
        self.n_blocks = vcfg["num_hidden_layers"]
        self.pyramid = vcfg["pyramid_num_routed"]
        self.top_k = int(vcfg.get("capacity_factor", 2))
        inv = 1.0 / (10000.0 ** (np.arange(0, self.head_dim // 2, 2,
                                           dtype=np.float64)
                                 / (self.head_dim // 2)))
        self.inv_freq = inv.astype(np.float32)          # (head_dim/4,)

    def _rope(self, pos_ids: np.ndarray) -> tuple[mx.array, mx.array]:
        # (N,2) x (F,) -> (N, 2F) -> cat -> (N, 4F = head_dim)
        emb = (pos_ids[:, :, None] * self.inv_freq[None, None, :]).reshape(
            pos_ids.shape[0], -1)
        emb = np.concatenate([emb, emb], -1).astype(np.float32)
        return mx.array(np.cos(emb)), mx.array(np.sin(emb))

    @staticmethod
    def _rotate_half(x: mx.array) -> mx.array:
        a, b = mx.split(x, 2, axis=-1)
        return mx.concatenate([-b, a], axis=-1)

    def _attn(self, x: mx.array, bi: int, cu: list[int],
              cos: mx.array, sin: mx.array) -> mx.array:
        p = f"vision_encoder.blocks.{bi}.attn."
        N = x.shape[0]
        qkv = linear(x, self.w[p + "qkv.weight"]).reshape(N, 3, self.heads,
                                                          self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = rms_norm(q, self.w[p + "q_norm.weight"], self.eps)
        k = rms_norm(k, self.w[p + "k_norm.weight"], self.eps)
        cs, sn = cos[:, None, :], sin[:, None, :]
        qf, kf = q.astype(mx.float32), k.astype(mx.float32)
        q = (qf * cs + self._rotate_half(qf) * sn).astype(q.dtype)
        k = (kf * cs + self._rotate_half(kf) * sn).astype(k.dtype)
        outs = []
        scale = self.head_dim ** -0.5
        for s0, s1 in zip(cu[:-1], cu[1:]):
            o = mx.fast.scaled_dot_product_attention(
                q[s0:s1].transpose(1, 0, 2)[None],
                k[s0:s1].transpose(1, 0, 2)[None],
                v[s0:s1].transpose(1, 0, 2)[None], scale=scale, mask=None)
            outs.append(o[0].transpose(1, 0, 2).reshape(s1 - s0, self.embed))
        return linear(mx.concatenate(outs, 0), self.w[p + "proj.weight"])

    def _mlp_dense(self, x: mx.array, pfx: str) -> mx.array:
        g = linear(x, self.w[pfx + "fc1.weight"])
        return linear(mx.multiply(g * mx.sigmoid(g),
                                  linear(x, self.w[pfx + "fc3.weight"])),
                      self.w[pfx + "fc2.weight"])

    def _mlp_moe(self, x: mx.array, bi: int) -> mx.array:
        p = f"vision_encoder.blocks.{bi}.mlp."
        n_exp = self.pyramid[bi]
        gate_w = self.w[p + "gate_weight"].astype(mx.float32)
        bias = self.w[p + "router_bias"].astype(mx.float32)
        logits = x.astype(mx.float32) @ gate_w.T
        probs = mx.sigmoid(logits)
        k = min(self.top_k, n_exp)
        inds = mx.argpartition(-(probs + bias[None]), kth=k - 1,
                               axis=-1)[:, :k]
        rw = mx.take_along_axis(probs, inds, axis=-1)
        rw = rw / (rw.sum(-1, keepdims=True) + 1e-9)      # sigmoid, top_k>1
        out = mx.zeros_like(x.astype(mx.float32))
        wsum = mx.zeros((x.shape[0],), mx.float32)
        inds_np = np.asarray(inds)
        rw_np = np.asarray(rw)
        for e in range(n_exp):
            tok, slot = np.where(inds_np == e)
            if tok.size == 0:
                continue
            t = mx.array(tok)
            we = mx.array(rw_np[tok, slot])
            y = self._mlp_dense(x[t], p + f"experts.{e}.")
            out[t] = out[t] + y.astype(mx.float32) * we[:, None]
            wsum[t] = wsum[t] + we
        out = out / (wsum[:, None] + 1e-9)
        return out.astype(x.dtype)

    def __call__(self, pixel_values: np.ndarray,
                 grid_thw: np.ndarray) -> mx.array:
        c = self.c
        patch = c["patch_size"]
        w = self.w["vision_encoder.patch_embed.proj.weight"]  # [E,3,14,14]
        b = self.w["vision_encoder.patch_embed.proj.bias"]
        pv = mx.array(pixel_values.reshape(-1, 3, patch, patch))
        x = mx.conv2d(pv.transpose(0, 2, 3, 1), w.transpose(0, 2, 3, 1),
                      stride=patch)
        x = x.reshape(-1, self.embed) + b[None]
        x = rms_norm(x, self.w["vision_encoder.patch_embed.norm.weight"],
                     self.eps)
        pos = vision_position_ids(grid_thw, self.merge)
        cu = vision_cu_seqlens(grid_thw)
        cos, sin = self._rope(pos)
        for bi in range(self.n_blocks):
            p = f"vision_encoder.blocks.{bi}."
            h = rms_norm(x, self.w[p + "norm_1.weight"], self.eps)
            x = x + self._attn(h, bi, cu, cos, sin)
            h = rms_norm(x, self.w[p + "norm_2.weight"], self.eps)
            if self.pyramid[bi] >= 1:
                x = x + self._mlp_moe(h, bi)
            else:
                x = x + self._mlp_dense(h, p + "mlp.")
        x = rms_norm(x, self.w["vision_encoder.post_trunk_norm.weight"],
                     self.eps)
        # adapter: LN -> merge 2x2 -> 6144 -> GELU -> 5120
        lnw = self.w["vision_encoder.adapter.ln_q.weight"].astype(mx.float32)
        lnb = self.w["vision_encoder.adapter.ln_q.bias"].astype(mx.float32)
        h = mx.fast.layer_norm(x.astype(mx.float32), lnw, lnb, 1e-6)
        h = h.reshape(-1, self.embed * self.merge ** 2)
        h = linear(h, self.w["vision_encoder.adapter.mlp.0.weight"]) + \
            self.w["vision_encoder.adapter.mlp.0.bias"][None]
        h = 0.5 * h * (1 + mx.erf(h / math.sqrt(2.0)))
        h = linear(h, self.w["vision_encoder.adapter.mlp.2.weight"]) + \
            self.w["vision_encoder.adapter.mlp.2.bias"][None]
        return h


# --------------------------------------------------------------------- audio
class AudioTower:
    def __init__(self, weights: dict, acfg: dict):
        self.w = weights
        wc = acfg["whisper_config"]
        self.d = int(wc["d_model"])
        self.heads = int(wc["encoder_attention_heads"])
        self.head_dim = self.d // self.heads
        self.n_layers = int(wc["encoder_layers"])
        rp = acfg.get("rope_parameters") or {}
        self.rot_dim = (int(self.head_dim *
                            rp.get("partial_rotary_factor", 0.5)) // 2) * 2
        theta = float(rp.get("rope_theta", 10000.0))
        self.inv_freq = (1.0 / theta ** (
            np.arange(0, self.rot_dim, 2, dtype=np.float64) / self.rot_dim
        )).astype(np.float32)
        self.hop = int(acfg.get("hop_length", 160))

    def _conv_stem(self, mel: np.ndarray, sample_lens: np.ndarray) -> mx.array:
        """mel [B, 128, T]; masks between convs per reference."""
        p = "audio_encoder.dots_encoder.speech_encoder."
        x = mx.array(mel)[:, None]                       # [B,1,128,T]
        valid = mx.array((sample_lens // self.hop).astype(np.int32))

        def mask(x, v):
            t = mx.arange(x.shape[-1])[None]
            m = (t < v[:, None]).astype(x.dtype)
            return x * m[:, None, None, :]

        for i in (1, 2, 3):
            x = mask(x, valid)
            wt = self.w[p + f"conv2d{i}.weight"]          # [O,I,3,3]
            bt = self.w[p + f"conv2d{i}.bias"]
            x = mx.conv2d(x.transpose(0, 2, 3, 1), wt.transpose(0, 2, 3, 1),
                          stride=2, padding=1).transpose(0, 3, 1, 2) + \
                bt[None, :, None, None]
            x = 0.5 * x * (1 + mx.erf(x / math.sqrt(2.0)))   # gelu
            valid = (valid + 1) // 2
        x = mask(x, valid)
        B, C, F, T = x.shape
        x = x.transpose(0, 3, 1, 2).reshape(B, T, C * F)
        return linear(x, self.w[p + "conv_out.weight"])

    def _rope(self, T: int, dtype):
        pos = np.arange(T, dtype=np.float32)
        f = pos[:, None] * self.inv_freq[None]
        emb = np.concatenate([f, f], -1)
        return (mx.array(np.cos(emb)).astype(dtype),
                mx.array(np.sin(emb)).astype(dtype))

    @staticmethod
    def _rot_half(x):
        a, b = mx.split(x, 2, axis=-1)
        return mx.concatenate([-b, a], -1)

    def _apply_rope(self, q, k, cos, sin):
        rd = self.rot_dim
        cs, sn = cos[None, None], sin[None, None]        # [1,1,T,rd]
        qr, qp = q[..., :rd], q[..., rd:]
        kr, kp = k[..., :rd], k[..., rd:]
        qr = qr * cs + self._rot_half(qr) * sn
        kr = kr * cs + self._rot_half(kr) * sn
        return (mx.concatenate([qr, qp], -1), mx.concatenate([kr, kp], -1))

    def __call__(self, input_features: np.ndarray, chunk_sample_lens: np.ndarray,
                 chunk_token_lens: np.ndarray, audio_chunk_counts: np.ndarray
                 ) -> tuple[mx.array, np.ndarray]:
        p = "audio_encoder.dots_encoder.speech_encoder."
        x = self._conv_stem(input_features, chunk_sample_lens)
        Tmax = int(chunk_token_lens.max())
        x = x[:, :Tmax]
        cos, sin = self._rope(Tmax, mx.float32)
        pos_mask = mx.array((np.arange(Tmax)[None] <
                             chunk_token_lens[:, None]))
        amask = mx.where(pos_mask[:, None, None, :], 0.0, -mx.inf
                         ).astype(mx.float32)
        B = x.shape[0]
        for i in range(self.n_layers):
            lp = p + f"layers.{i}."
            h = rms_norm(x, self.w[lp + "self_attn_layer_norm.weight"], 1e-6)
            q = (linear(h, self.w[lp + "self_attn.q_proj.weight"]) +
                 self.w[lp + "self_attn.q_proj.bias"][None])
            k = linear(h, self.w[lp + "self_attn.k_proj.weight"])
            v = (linear(h, self.w[lp + "self_attn.v_proj.weight"]) +
                 self.w[lp + "self_attn.v_proj.bias"][None])
            q = q.reshape(B, Tmax, self.heads, self.head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(B, Tmax, self.heads, self.head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(B, Tmax, self.heads, self.head_dim).transpose(0, 2, 1, 3)
            qf, kf = q.astype(mx.float32), k.astype(mx.float32)
            qf, kf = self._apply_rope(qf, kf, cos, sin)
            o = mx.fast.scaled_dot_product_attention(
                qf, kf, v.astype(mx.float32), scale=self.head_dim ** -0.5,
                mask=amask)
            o = o.transpose(0, 2, 1, 3).reshape(B, Tmax, self.d).astype(x.dtype)
            o = (linear(o, self.w[lp + "self_attn.out_proj.weight"]) +
                 self.w[lp + "self_attn.out_proj.bias"][None])
            x = x + o
            h = rms_norm(x, self.w[lp + "final_layer_norm.weight"], 1e-6)
            gu = linear(h, self.w[lp + "fc1.weight"]) + self.w[lp + "fc1.bias"][None]
            g, u = mx.split(gu, 2, axis=-1)
            h = mx.multiply(g * mx.sigmoid(g), u)
            h = linear(h, self.w[lp + "fc2.weight"]) + self.w[lp + "fc2.bias"][None]
            x = x + h
        x = rms_norm(x, self.w[p + "layer_norm.weight"], 1e-6)

        # concat chunks per audio, then adapter
        chunks = [x[i, :int(chunk_token_lens[i])] for i in range(B)]
        embeds, lens, off = [], [], 0
        ap = "audio_encoder.audio_adapter.proj."
        for cnt in audio_chunk_counts.tolist():
            e = mx.concatenate(chunks[off:off + cnt], 0)
            lnw = self.w[ap + "0.weight"].astype(mx.float32)
            lnb = self.w[ap + "0.bias"].astype(mx.float32)
            h = mx.fast.layer_norm(e.astype(mx.float32), lnw, lnb, 1e-5)
            h = linear(h, self.w[ap + "1.weight"]) + self.w[ap + "1.bias"][None]
            h = 0.5 * h * (1 + mx.erf(h / math.sqrt(2.0)))
            h = linear(h, self.w[ap + "3.weight"]) + self.w[ap + "3.bias"][None]
            embeds.append(h)
            lens.append(h.shape[0])
            off += cnt
        return mx.concatenate(embeds, 0), np.array(lens)
