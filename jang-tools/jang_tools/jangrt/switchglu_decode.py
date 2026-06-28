"""Reusable installer for the fused-decode `SwitchGLU.__call__` patch.

Background
----------
`mlx_lm.models.switch_layers.SwitchGLU.__call__` runs gate/up/down as
three separate `SwitchLinear` calls — each a separate Metal dispatch.
For a 39-sparse-layer MoE that's 117 dispatches per decode token, which
caps decode throughput to roughly 20 tok/s on Apple-silicon. The
canonical loader `jang_tools.load_jangtq._hydrate_jangtq_model`
class-monkey-patches `SwitchGLU.__call__` so the fast path collapses
gate+up+silu+down into a single `mx.compile`'d closure that runs:

    rotate(x) → fused_gate_up_swiglu → rotate(x_act) → gather_dn(...)

That removes ~3× the dispatch count and lifts decode to ~80 tok/s on
Laguna-class models.

Lightweight runtimes (`jang_tools.laguna.runtime`,
`jang_tools.mistral3.runtime`, …) construct their model directly and
bypass the canonical loader, so they never installed the patch — hence
the 5-20 tok/s decode reports. This module exposes the patch as a
single `install_switchglu_fused_decode()` entry point so any runtime
can opt in after `hydrate_jangtq` runs.

The patch:
  * Is class-level — affects every `SwitchGLU` instance in the process.
  * Falls back to the original `__call__` for any switch layer whose
    `gate_proj` / `up_proj` aren't `TurboQuantSwitchLinear`. Non-TQ
    SwitchGLUs (bf16, JANG affine) are unaffected.
  * Is idempotent — repeated calls re-install on the same hooked
    function and are safe.

Created by Jinho Jang (eric@jangq.ai).
"""
from __future__ import annotations

from typing import Optional

import mlx.core as mx


_INSTALLED = False  # idempotency guard


def install_switchglu_fused_decode() -> bool:
    """Class-monkey-patch `SwitchGLU.__call__` with the JANGTQ fused
    decode path. Returns True on first install, False if already
    installed.

    Safe to call multiple times. Safe to call when no
    `TurboQuantSwitchLinear` instances exist — the patched call
    falls back to the original implementation per-instance.

    Failures (missing `mlx_lm`, missing kernels) propagate as
    `RuntimeError`; the caller is responsible for handling them. The old
    swallow-and-warn behavior masked import errors that meant the
    fused path was silently NEVER active. Codex 2026-05-05 #4.
    """
    global _INSTALLED
    if _INSTALLED:
        return False

    # Imports raise (do NOT swallow) — silent install failures meant
    # Laguna/Mistral users were stuck at unpatched dispatch counts
    # without any way to tell.
    from mlx_lm.models.switch_layers import (
        SwitchGLU,
        _gather_sort,
        _scatter_unsort,
    )
    from jang_tools.turboquant.tq_kernel import TurboQuantSwitchLinear
    from jang_tools.turboquant.fused_gate_up_kernel import (
        fused_gate_up_swiglu_matmul,
        make_fused_gate_up_swiglu_decode,
    )
    from jang_tools.turboquant.gather_tq_kernel import (
        make_gather_tq_decode_per_row,
    )
    from jang_tools.turboquant.hadamard_kernel import hadamard_rotate_metal

    decode_compiled: dict = {}

    def _get_compiled_decode(
        in_f: int,
        out_f: int,
        bits: int,
        K: int,
        swiglu_limit: float = 0.0,
        dp_bits=None,
    ):
        # JANGTQ_K mixed-bit (Codex 2026-05-05 #4): gate/up share `bits`
        # (fused gate+up kernel constraint), but down_proj may differ —
        # MiniMax-M2.7-JANGTQ_K ships gate=2, up=2, down=4. Without
        # threading `dp_bits` through, the down_proj packed tensors are
        # unpacked at the wrong bit width → wrong codebook indices →
        # garbage decode. Defaults to `bits` for legacy uniform callers
        # so JANGTQ2/3/4 paths are byte-identical.
        if dp_bits is None:
            dp_bits = bits
        limit_milli = int(round(float(swiglu_limit or 0.0) * 1000.0))
        key = (in_f, out_f, bits, dp_bits, K, limit_milli)
        if key in decode_compiled:
            return decode_compiled[key]
        fused_gu = make_fused_gate_up_swiglu_decode(
            in_f, out_f, bits, K, swiglu_limit=swiglu_limit
        )
        gather_dn = make_gather_tq_decode_per_row(out_f, in_f, dp_bits, K)

        def _mlp(x_flat, pg, ng, pu, nu, pd, nd, cb_gate, cb_down, signs_in, signs_dn, idx):
            x_rot = hadamard_rotate_metal(x_flat, signs_in)
            x_act = fused_gu(x_rot, pg, ng, pu, nu, cb_gate, idx)        # (K, out_f)
            x_act_rot = hadamard_rotate_metal(x_act, signs_dn)
            y = gather_dn(x_act_rot, pd, nd, cb_down, idx)                # (K, in_f)
            return y

        decode_compiled[key] = mx.compile(_mlp)
        return decode_compiled[key]

    orig_call = SwitchGLU.__call__

    def _fused_switchglu_call(self, x, indices):
        gp = self.gate_proj
        up = self.up_proj
        dp = self.down_proj
        if not isinstance(gp, TurboQuantSwitchLinear) or not isinstance(up, TurboQuantSwitchLinear):
            return orig_call(self, x, indices)
        # Pull DSV4-style limited SwiGLU clamp from the SwitchGLU's
        # activation if present (DSV4 sets `swiglu_limit=10` on its
        # activation module). Codex 2026-05-05 #4 — was missing.
        _activation = getattr(self, "activation", None)
        _swiglu_limit = float(getattr(_activation, "swiglu_limit", 0.0) or 0.0)

        # Decode fast path: batch=1, K=topk, broadcast mode.
        x_sq = x
        while x_sq.ndim > 2 and x_sq.shape[-2] == 1:
            x_sq = x_sq.squeeze(-2)
        x_flat = x_sq.reshape(-1, gp.in_features)
        batch = x_flat.shape[0]
        K = indices.shape[-1] if indices.ndim > 0 else 1
        do_sort_ok = indices.ndim >= 1 and indices.size < 64
        can_fast = (batch == 1 and K > 0 and do_sort_ok and not getattr(self, "training", False))

        if can_fast:
            idx_flat = indices.reshape(-1).astype(mx.uint32)
            compiled_mlp = _get_compiled_decode(
                gp.in_features, gp.out_features, gp.bits, K,
                swiglu_limit=_swiglu_limit, dp_bits=dp.bits,
            )
            y = compiled_mlp(
                x_flat.astype(mx.float32),
                gp.packed, gp.norms, up.packed, up.norms,
                dp.packed, dp.norms,
                gp.codebook, dp.codebook,
                gp.signs, dp.signs, idx_flat,
            )  # (K, in_f)
            out = y.reshape(*indices.shape[:-1], K, 1, gp.in_features)
            if out.dtype != x.dtype:
                out = out.astype(x.dtype)
            return out.squeeze(-2)

        # Slow path: dynamic-shape detection (prefill, sorted routing, training).
        x_exp = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x_exp, idx, inv_order = _gather_sort(x_exp, indices)
        if getattr(self, "training", False):
            idx = mx.stop_gradient(idx)

        x_act = fused_gate_up_swiglu_matmul(
            x_exp,
            gp.packed, gp.norms,
            up.packed, up.norms,
            gp.codebook, gp.signs,
            idx,
            bits=gp.bits,
            swiglu_limit=_swiglu_limit,
        )
        x_out = self.down_proj(x_act, idx, sorted_indices=do_sort)
        if do_sort:
            x_out = _scatter_unsort(x_out, inv_order, indices.shape)
        return x_out.squeeze(-2)

    SwitchGLU.__call__ = _fused_switchglu_call
    _INSTALLED = True
    return True


__all__ = ["install_switchglu_fused_decode"]
