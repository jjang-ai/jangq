"""TQSwitchGLU: drop-in for mlx_lm SwitchGLU on JANGTQ v2 experts.

Tensor names use the `tq2_` prefix ON PURPOSE: released vMLX routes any bundle with `.tq_packed` tensors to the v1
JANGTQ loader, which would silently mis-decode v2. With `tq2_*` names an old runtime fails closed (unknown tensors).

Weights (loaded lazily from the bundle, never repacked at load):
  {gate,up,down}_proj.tq2_packed  uint32 (E, N, K*bits/32)
  {gate,up,down}_proj.tq2_scales  float16 (E, N)
Per-module rotation (config entry "rotation"):
  none       : weights quantized as is
  hadamard32 : weights quantized as W R^T with R = blockwise normalized Walsh-Hadamard over 32-wide input blocks
               (no random signs). Measured 2026-09-25 on GLM-5.3 experts: fixes TQ's per-row error TAILS
               (p99 0.742 -> 0.410, p99.9 0.868 -> 0.497; affine2 g128 0.527 / 0.590) at unchanged byte cost.
               Decode kernels rotate the activation in registers (lane pair); prefill pre-rotates rows.
Paths (same switch point as MLX SwitchGLU: sort when indices.size >= 64):
  decode : fused gate/up/SwiGLU qmv (f32) -> down qmv with the router-weighted sum fused (x.dtype)
  prefill: argsort experts (GPU) -> fused gate/up/SwiGLU NAX qmm -> down NAX qmm -> unsort -> weighted sum
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from . import kernels as K
from .format import codebook

import os

SORT_THRESHOLD = 64
ROTATIONS = ("none", "hadamard32")
# Where decode applies the Hadamard-32: "host" = once per activation row (x once per token, h once per expert-token)
# then the unrotated fast kernels; "kernel" = in-register inside every threadgroup (redundant: measured 1.05-1.15x).
DECODE_ROT = os.environ.get("JANGTQ2_DECODE_ROT", "host")
# Fused Hadamard kernels (2026-09-27): "fused" = one-pass activation rotation kernel + the gate/up kernels rotate
# their OUTPUT (the down input) in the epilogue. "host" = the previous mx.hadamard_transform path (A/B + fallback).
ROT_IMPL = os.environ.get("JANGTQ2_ROT_IMPL", "fused")


class TQSwitchLinear(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, num_experts: int, bits: int, rotation: str = "none"):
        super().__init__()
        if rotation not in ROTATIONS:
            raise ValueError(f"jangtq2: unknown rotation {rotation!r}")
        if rotation == "hadamard32" and input_dims % 32:
            raise ValueError("jangtq2: hadamard32 requires input_dims % 32 == 0")
        self.bits = bits
        self.rotation = rotation
        self.input_dims, self.output_dims, self.num_experts = input_dims, output_dims, num_experts
        self.tq2_packed = mx.zeros((num_experts, output_dims, input_dims * bits // 32), dtype=mx.uint32)
        self.tq2_scales = mx.zeros((num_experts, output_dims), dtype=mx.float16)
        self._cb = mx.array(codebook(bits))

    @property
    def rotated(self) -> bool:
        return self.rotation == "hadamard32"

    def to_quantized(self, **kwargs):
        """Already quantized (JANGH packed codes). Generic loaders (mlx_lm.utils.load_model) call nn.quantize with the
        bundle per-module entry ({"mode": "jangtq2", "bits": b, ...}) for this path; without this method nn.quantize
        raises "Unable to quantize model of type TQSwitchLinear". Fail closed on a mismatching entry."""
        mode, bits = kwargs.get("mode", "jangtq2"), int(kwargs.get("bits", self.bits))
        if mode != "jangtq2" or bits != self.bits or kwargs.get("rotation", self.rotation) != self.rotation:
            raise ValueError(f"jangtq2: bundle entry {kwargs} does not match the installed module (bits {self.bits}, rotation {self.rotation})")
        return self


def rotate_rows(x: mx.array, lin: TQSwitchLinear, out_dtype=None) -> mx.array:
    """Activation rotation: blockwise normalized Hadamard-32. fused impl = one kernel launch (any dtype in/out);
    host impl = float32 mx.hadamard_transform, back to x.dtype."""
    if not lin.rotated:
        return x
    if ROT_IMPL == "fused":
        return K.h32_rows(x, out_dtype or x.dtype)
    shp = x.shape
    return mx.hadamard_transform(x.astype(mx.float32).reshape(*shp[:-1], shp[-1] // 32, 32)).reshape(shp).astype(x.dtype)


class TQSwitchGLU(nn.Module):
    def __init__(self, input_dims: int, hidden_dims: int, num_experts: int, bits_gate_up: int, bits_down: int,
                 swiglu_limit: float = 0.0, rotation_gate_up: str = "none", rotation_down: str = "none",
                 bits_up: int = 0):
        """bits_gate_up = gate bits; bits_up (0 = same as gate) lets gate and up differ (mixed-bit fused kernels)."""
        super().__init__()
        self.gate_proj = TQSwitchLinear(input_dims, hidden_dims, num_experts, bits_gate_up, rotation_gate_up)
        self.up_proj = TQSwitchLinear(input_dims, hidden_dims, num_experts, bits_up or bits_gate_up, rotation_gate_up)
        self.down_proj = TQSwitchLinear(hidden_dims, input_dims, num_experts, bits_down, rotation_down)
        self.limit = float(swiglu_limit)

    def _fused_out(self, K_in, N):
        """True when the gate/up kernels rotate their own output (down input)."""
        return (ROT_IMPL == "fused" and self.down_proj.rotated and DECODE_ROT != "kernel"
                and K.rotout_decode_supported(K_in, N))

    def _decode_h(self, xf, idx):
        """-> (h, h_is_rotated)"""
        g, u = self.gate_proj, self.up_proj
        in_kernel = g.rotated and DECODE_ROT == "kernel"
        if g.rotated and not in_kernel:
            xf = rotate_rows(xf, g, mx.float32 if ROT_IMPL == "fused" else None)
        if self._fused_out(xf.shape[-1], g.output_dims) and not in_kernel:
            return K.gather_qmv_fused_rotout(xf, g.tq2_packed, g.tq2_scales, u.tq2_packed, u.tq2_scales, idx, g.bits,
                                             limit=self.limit, bits_u=u.bits), True
        if g.bits != u.bits:
            # mixed gate/up bits without the rotated-output kernel: two single kernels + host activation
            gg = K.gather_qmv(xf, g.tq2_packed, g.tq2_scales, g._cb, idx, g.bits, x_per_dispatch=False, rotate=in_kernel)
            uu = K.gather_qmv(xf, u.tq2_packed, u.tq2_scales, u._cb, idx, u.bits, x_per_dispatch=False, rotate=in_kernel)
            if self.limit > 0:
                gg = mx.minimum(gg, self.limit); uu = mx.clip(uu, -self.limit, self.limit)
            return gg * mx.sigmoid(gg) * uu, False
        return K.gather_qmv(xf, g.tq2_packed, g.tq2_scales, g._cb, idx, g.bits, x_per_dispatch=False,
                            packed_u=u.tq2_packed, scales_u=u.tq2_scales, limit=self.limit, rotate=in_kernel), False

    def _down_in(self, hr):
        """(h, rotate_in_kernel) for the down projection."""
        h, done = hr
        d = self.down_proj
        if done or not d.rotated:
            return h, False
        if DECODE_ROT != "kernel":
            return rotate_rows(h, d), False
        return h, True

    def _prefill(self, x, idx, kk):
        g, u, d = self.gate_proj, self.up_proj, self.down_proj
        order = mx.argsort(idx)
        inv = mx.argsort(order)
        idx_s = idx[order]
        xs = rotate_rows(x, g)[order // kk]
        fused_out = ROT_IMPL == "fused" and d.rotated and g.output_dims % 32 == 0
        h = K.gather_qmm_sorted(xs, g.tq2_packed, g.tq2_scales, g._cb, idx_s, g.bits,
                                packed_u=u.tq2_packed, scales_u=u.tq2_scales, limit=self.limit, rot_out=fused_out,
                                bits_u=u.bits)
        y = K.gather_qmm_sorted(h if fused_out else rotate_rows(h, d), d.tq2_packed, d.tq2_scales, d._cb, idx_s, d.bits)
        return y[inv]

    def _experts(self, x, indices):
        """x (..., D), indices (..., k) -> (..., k, D) in x.dtype (unweighted, like SwitchGLU)."""
        d = self.down_proj
        lead, kk, D = x.shape[:-1], indices.shape[-1], x.shape[-1]
        xf = x.reshape(-1, D)
        idx = indices.reshape(-1).astype(mx.uint32)
        if idx.size < SORT_THRESHOLD:
            h, rk = self._down_in(self._decode_h(xf, idx))
            y = K.gather_qmv(h, d.tq2_packed, d.tq2_scales, d._cb, idx, d.bits, x_per_dispatch=True, rotate=rk)
            return y.astype(x.dtype).reshape(*lead, kk, D)
        return self._prefill(xf, idx, kk).reshape(*lead, kk, D)

    def __call__(self, x, indices):
        return self._experts(x, indices)

    def routed(self, x, indices, scores):
        """Weighted routed output (..., D) = sum_k scores[...,k] * expert_k(x). Decode fuses the weighted sum
        into the down kernel; prefill uses the sorted NAX path then a weighted reduction."""
        d = self.down_proj
        lead, kk, D = x.shape[:-1], indices.shape[-1], x.shape[-1]
        if indices.size >= SORT_THRESHOLD:
            y = self._experts(x, indices)
            return (y * scores[..., None].astype(y.dtype)).sum(axis=-2)
        idx2 = indices.reshape(-1, kk).astype(mx.uint32)
        h, rk = self._down_in(self._decode_h(x.reshape(-1, D), idx2.reshape(-1)))
        y = K.gather_qmv_weighted_down(h, d.tq2_packed, d.tq2_scales, d._cb, idx2, scores.reshape(-1, kk), d.bits,
                                       x.dtype, rotate=rk)
        return y.reshape(*lead, D)
