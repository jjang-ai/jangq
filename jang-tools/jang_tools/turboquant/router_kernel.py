"""
Sigmoid-bias top-k router for JANGTQ MoE decode (P15 fast path).

Restores `make_sigmoid_bias_topk_router`, referenced by load_jangtq.py's P15
optimisation (`from jang_tools.turboquant.router_kernel import ...`). With this
module absent (turboquant/* is gitignored), the import failed and the ENTIRE
P15 mx.compile'd-router path was silently skipped ("P15 skipped").

Pure-MLX implementation, numerically equivalent to the reference compiled router
`jang_tools.laguna.model._compiled_sigmoid_bias_topk` (MLX promotes the bias add
to fp32): sigmoid -> add e_score_correction_bias -> top-k via argpartition ->
gather UNBIASED scores -> renormalise. Used directly only by the experimental
JANGTQ_CUSTOM_ROUTER_TOPK single-token decode path; its mere importability
re-enables the (default) inline compiled routers.
"""

import mlx.core as mx

__all__ = ["make_sigmoid_bias_topk_router"]


def make_sigmoid_bias_topk_router(experts: int, top_k: int):
    """Return an mx.compile'd ``router(gates_f32, e_score_bias) -> (inds, scores)``.

    ``experts`` is accepted for signature/interface parity (a Metal-kernel
    variant keys on it); the pure-MLX path is shapeless and does not need it.

    Top-k is taken along the LAST axis. The JANGTQ_CUSTOM_ROUTER_TOPK call site
    passes a single flattened token's gate vector (shape ``(experts,)``). For a
    multi-token batch, pass shape ``(..., experts)`` so per-row top-k is correct
    — do NOT flatten across the token dimension, or routing will be wrong.
    """
    k = int(top_k)

    def _router(gates_f32, e_score_bias):
        scores = mx.sigmoid(gates_f32)
        inds = mx.argpartition(-(scores + e_score_bias), kth=k - 1, axis=-1)[..., :k]
        sel = mx.take_along_axis(scores, inds, axis=-1)  # gather UNBIASED scores
        sel = sel / (mx.sum(sel, axis=-1, keepdims=True) + 1e-20)
        return inds, sel

    return mx.compile(_router)
