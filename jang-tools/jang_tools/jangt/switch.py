"""JTSwitchGLU: routed experts of one MoE layer with per-GROUP formats (gate_up | down), each JANGT or JANGH.

  JANGT group : moe_kernels (decode: rotate + gather kernels; prefill: expert-sorted, groups of experts dequantized to
                bf16 in the ROTATED basis + mx.gather_mm, outlier columns as a small gather_mm)
  JANGH group : jangtq2 (decode gather_qmv / gather_qmv_weighted_down, prefill gather_qmm_sorted, Hadamard-32)
Interface = vMLX Naive MoE: `routed(x, indices, scores) -> sum_k scores_k * expert_k(x)` (x (..., D)).
"""
from __future__ import annotations

import os

import mlx.core as mx
import numpy as np

from jang_tools.jangt import moe_kernels as MK

SORT_THRESHOLD = 64                 # same switch point as mlx-lm SwitchGLU / JANGH
PREFILL_EXPERT_GROUP = int(os.environ.get("JANGT_PREFILL_GROUP", "64"))


class JHGroup:
    """JANGH tensors of one group (gate+up or down), executed with jangtq2 kernels."""

    def __init__(self, packed: dict, scales: dict, kin: int):
        from jang_tools.jangh.format import codebook
        self.packed, self.scales = packed, scales
        self.bits = {p: int(packed[p].shape[-1]) * 32 // kin for p in packed}       # MEASURED from the packed width
        self.cb = {p: mx.array(codebook(b)) for p, b in self.bits.items()}


class JTSwitchGLU:
    is_jangtq2 = True                # the Naive MoE block dispatches to .routed() on this flag

    def __init__(self, layer: int, jt: dict | None, jh: dict | None, D: int = 4096, I: int = 2048, L: int = 12):
        """jt: stage/bundle tensors holding the layer's JANGT groups ({name: array}); jh: {name: array} holding its
        JANGH tensors (tq2_packed / tq2_scales). Each group must be present in exactly one of them."""
        self.D, self.I, self.L = D, I, L
        b = f"model.layers.{layer}.mlp.switch_mlp."
        self.U = MK.load_unit(jt or {}, layer, D, I, L)
        self.jh_gu = self.jh_dn = None
        if self.U["gate"] is None:
            self.jh_gu = JHGroup({p: jh[b + p + "_proj.tq2_packed"] for p in ("gate", "up")},
                                 {p: jh[b + p + "_proj.tq2_scales"] for p in ("gate", "up")}, D)
        if self.U["down"] is None:
            self.jh_dn = JHGroup({"down": jh[b + "down_proj.tq2_packed"]}, {"down": jh[b + "down_proj.tq2_scales"]}, I)

    # ------------------------------------------------------------------ decode
    def _decode_h(self, x2, idx2):
        """-> h (T*k, I) activation in the ORIGINAL basis."""
        if self.jh_gu is None:
            xr = MK.rotate_rows(x2, self.U["su_gu"])
            return MK.gather_gate_up_swiglu_v3(xr, x2, idx2, self.U, out_dtype=mx.float32)
        from jang_tools.jangh import kernels as K
        g = self.jh_gu
        xf = K.h32_rows(x2, mx.float32)
        return K.gather_qmv(xf, g.packed["gate"], g.scales["gate"], g.cb["gate"], idx2.reshape(-1).astype(mx.uint32), g.bits["gate"],
                            x_per_dispatch=False, packed_u=g.packed["up"], scales_u=g.scales["up"], limit=0.0)

    def _decode_down(self, h, idx2, w2, out_dtype):
        if self.jh_dn is None:
            hr = MK.rotate_rows(h, self.U["sv_dn"], idx2.reshape(-1))
            return MK.gather_down_weighted_v3(hr, h, idx2, w2, self.U).astype(out_dtype)
        from jang_tools.jangh import kernels as K
        d = self.jh_dn
        return K.gather_qmv_weighted_down(K.h32_rows(h, mx.float32), d.packed["down"], d.scales["down"], d.cb["down"],
                                          idx2.astype(mx.uint32), w2, d.bits["down"], out_dtype)

    # ------------------------------------------------------------------ prefill
    def _jt_sorted_mm(self, xs_rot, xs_raw, idx_s, P, counts, out_dtype):
        """Expert-sorted rows: y_i = W_rot[idx_i] . xs_rot_i (+ outlier columns from xs_raw). Experts processed in
        groups whose bf16 weights are decoded on the fly (transient ~ group x N x kin x 2 B)."""
        E = P["packed"].shape[0]; N = P["scale"].shape[1]; outs = []
        starts = np.concatenate([[0], np.cumsum(counts)])
        for e0 in range(0, E, PREFILL_EXPERT_GROUP):
            e1 = min(E, e0 + PREFILL_EXPERT_GROUP); r0, r1 = int(starts[e0]), int(starts[e1])
            if r1 == r0:
                continue
            Wg = MK.dequant_rot(P, e0, e1, self.L)                                  # (n_e, N, kin)
            ii = (idx_s[r0:r1] - e0).astype(mx.uint32)
            y = mx.gather_mm(xs_rot[r0:r1, None, :].astype(mx.bfloat16), Wg.swapaxes(-1, -2), rhs_indices=ii, sorted_indices=True)[:, 0, :]
            if P["n_out"]:
                oc = P["ocol"][e0:e1]
                cols = mx.maximum(oc, 0)                                             # (n_e, n_out); -1 entries have wout 0
                xo = mx.take_along_axis(xs_raw[r0:r1], cols[ii.astype(mx.int32)], axis=1)                  # (rows, n_out)
                y = y.astype(mx.float32) + mx.gather_mm(xo[:, None, :].astype(mx.float32),
                                                        P["wout"][e0:e1].astype(mx.float32).swapaxes(-1, -2),
                                                        rhs_indices=ii, sorted_indices=True)[:, 0, :]
            outs.append(y.astype(out_dtype))
            mx.eval(outs[-1]); del Wg
        return mx.concatenate(outs, axis=0)

    def _prefill(self, x2, idx2, w2):
        T, k = idx2.shape
        flat = idx2.reshape(-1)
        order = mx.argsort(flat); idx_s = flat[order]; tok = order // k
        counts = np.bincount(np.array(flat), minlength=256)                         # one host sync per layer (bring-up)
        if self.jh_gu is None:
            xr = MK.rotate_rows(x2, self.U["su_gu"])
            xs_r, xs_x = xr[tok], x2[tok]
            g = self._jt_sorted_mm(xs_r, xs_x, idx_s, self.U["gate"], counts, mx.float32)
            u = self._jt_sorted_mm(xs_r, xs_x, idx_s, self.U["up"], counts, mx.float32)
            h_s = (g * mx.sigmoid(g) * u)
        else:
            from jang_tools.jangh import kernels as K
            gj = self.jh_gu
            xs = K.h32_rows(x2, x2.dtype)[tok]          # bf16: the NAX tiles only fit threadgroup memory for 2-byte x
            h_s = K.gather_qmm_sorted(xs, gj.packed["gate"], gj.scales["gate"], gj.cb["gate"], idx_s.astype(mx.uint32), gj.bits["gate"],
                                      packed_u=gj.packed["up"], scales_u=gj.scales["up"], limit=0.0, rot_out=False, bits_u=gj.bits["up"])
        if self.jh_dn is None:
            hr = MK.rotate_rows(h_s.astype(mx.float32), self.U["sv_dn"], idx_s)
            y_s = self._jt_sorted_mm(hr, h_s, idx_s, self.U["down"], counts, mx.float32)
        else:
            from jang_tools.jangh import kernels as K
            dj = self.jh_dn
            y_s = K.gather_qmm_sorted(K.h32_rows(h_s.astype(mx.bfloat16), mx.bfloat16), dj.packed["down"], dj.scales["down"], dj.cb["down"],
                                      idx_s.astype(mx.uint32), dj.bits["down"])
        y = y_s[mx.argsort(order)].reshape(T, k, -1).astype(mx.float32)
        return (y * w2[..., None].astype(mx.float32)).sum(axis=1)

    # ------------------------------------------------------------------ public
    def routed(self, x, indices, scores):
        lead, kk = x.shape[:-1], indices.shape[-1]
        x2 = x.reshape(-1, self.D); idx2 = indices.reshape(-1, kk).astype(mx.int32); w2 = scores.reshape(-1, kk).astype(mx.float32)
        if idx2.size < SORT_THRESHOLD:
            h = self._decode_h(x2, idx2)
            y = self._decode_down(h, idx2, w2, x.dtype)
        else:
            y = self._prefill(x2, idx2, w2).astype(x.dtype)
        return y.reshape(*lead, self.D)
