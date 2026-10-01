"""Pool two calibration captures with the diag.safetensors schema.

Per routed-MoE layer: reservoir rows / top-k idx / weights are CONCATENATED (both captures' rows remain available
to GPTQ Hessians and held-out splits); E[x^2] numerators and counts are ADDED with the second capture weighted by
`w2` (so a 171k-token agentic capture can count like w2 x its tokens against a 600k-token general capture).
Down-input importance: the agentic capture records expert_sum_a2; the FP8 capture's per-expert down diag lives in
the imatrix file, so the caller keeps using that for down and may blend in `expert_sum_a2 / expert_count`."""
from __future__ import annotations

import mlx.core as mx


def pooled(d1: dict, d2: dict | None, mod: str, w2: float = 2.0):
    """Return (X rows float32, tidx np, tw np, expert_imx (E,4096) float32, pooled_imx (4096,), n_rows_first)."""
    import numpy as np
    X1 = d1[mod + ".rows"].astype(mx.float32)
    t1, w1 = np.asarray(d1[mod + ".rows_topk_idx"]), np.asarray(d1[mod + ".rows_topk_w"])
    es = d1[mod + ".expert_sum_x2"].astype(mx.float32); ec = d1[mod + ".expert_count"].astype(mx.float32)
    ps = d1[mod + ".sum_x2"].astype(mx.float32); pc = d1[mod + ".count"].astype(mx.float32)
    if d2 is None or (mod + ".rows") not in d2:
        return X1, t1, w1, es / mx.maximum(ec, 1.0)[:, None], ps / mx.maximum(pc, 1.0), X1.shape[0]
    X2 = d2[mod + ".rows"].astype(mx.float32)
    t2, w2r = np.asarray(d2[mod + ".rows_topk_idx"]), np.asarray(d2[mod + ".rows_topk_w"])
    es = es + w2 * d2[mod + ".expert_sum_x2"].astype(mx.float32); ec = ec + w2 * d2[mod + ".expert_count"].astype(mx.float32)
    ps = ps + w2 * d2[mod + ".sum_x2"].astype(mx.float32); pc = pc + w2 * d2[mod + ".count"].astype(mx.float32)
    X = mx.concatenate([X1, X2], axis=0)
    return (X, np.concatenate([t1, t2.astype(t1.dtype)]), np.concatenate([w1, w2r.astype(w1.dtype)]),
            es / mx.maximum(ec, 1.0)[:, None], ps / mx.maximum(pc, 1.0), X1.shape[0])
