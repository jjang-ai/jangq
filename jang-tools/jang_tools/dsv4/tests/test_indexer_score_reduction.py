"""A5 — verify _indexer_score_reduction precision.

The Indexer's score reduction:
    scores = relu(q @ k.T) * scale * (weights * n_heads^-0.5).sum(over heads)

is implemented as an @mx.compile graph for fusion. We need to verify
the compile graph produces bit-exact-or-very-close results vs the
expanded reference.

Constants (from DSV4-Flash config):
    index_n_heads = 64
    index_head_dim = 128
    scale = 1/sqrt(128) ≈ 0.0884
    n_heads_inv_sqrt = 1/sqrt(64) = 0.125
"""
from __future__ import annotations
import sys
import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Users/eric/jang/jang-tools")
from jang_tools.dsv4.mlx_model import _indexer_score_reduction


def reference(scores_qkT, weights, scale, n_heads_inv_sqrt):
    """Non-compiled reference. mx.maximum + chained ops."""
    s = mx.maximum(scores_qkT, 0) * scale
    w = weights * n_heads_inv_sqrt
    # (B, n_heads, L, P) * (B, L, n_heads) → broadcast and sum across heads
    return (s * w.swapaxes(-1, -2)[..., None]).sum(axis=1)


def main():
    np.random.seed(0)
    # DSV4-Flash sizes
    B, n_heads, L, P, head_dim = 1, 64, 32, 256, 128
    scale = head_dim ** -0.5
    n_heads_inv_sqrt = n_heads ** -0.5

    rng = np.random.default_rng(0)
    raw = mx.array(rng.standard_normal((B, n_heads, L, P)).astype(np.float32))
    w   = mx.array(rng.standard_normal((B, L, n_heads)).astype(np.float32))

    out_compiled = _indexer_score_reduction(raw, w, scale, n_heads_inv_sqrt)
    out_ref      = reference(raw, w, scale, n_heads_inv_sqrt)
    mx.eval(out_compiled, out_ref)

    diff = mx.abs(out_compiled - out_ref)
    max_d = float(mx.max(diff).item())
    rms = float(mx.sqrt(mx.mean(diff * diff)).item())
    rms_ref = float(mx.sqrt(mx.mean(out_ref * out_ref)).item())
    rel = rms / (rms_ref + 1e-12)
    print(f"  shape: out=({B},{L},{P})  ref RMS={rms_ref:.4f}")
    print(f"  diff RMS={rms:.2e}  max_abs={max_d:.2e}  relative={rel*100:.4f}%")

    # Check sign correctness — out values should track the reference's signs
    # (top-k argpartition is what depends on this; magnitude noise is OK if
    # the partial order is preserved on the topk frontier).
    flat_c = np.array(out_compiled).flatten()
    flat_r = np.array(out_ref).flatten()
    rho = float(np.corrcoef(flat_c, flat_r)[0, 1])
    print(f"  Pearson correlation: {rho:.6f}  (expect > 0.99999)")

    # For the top-512 partition specifically: verify the SET of top-512
    # indices is identical between compiled and reference.
    out_c2 = np.array(out_compiled[0])  # (L, P)
    out_r2 = np.array(out_ref[0])
    set_match = []
    for q in range(L):
        topc = set(np.argpartition(-out_c2[q], min(P-1, 511))[:min(P, 512)])
        topr = set(np.argpartition(-out_r2[q], min(P-1, 511))[:min(P, 512)])
        set_match.append(len(topc & topr) / len(topc))
    print(f"  top-512 set overlap: mean={np.mean(set_match)*100:.2f}%  "
          f"min={min(set_match)*100:.2f}%")

    PASS = max_d < 1e-4 and rho > 0.99999 and min(set_match) >= 0.99
    print(f"\n  {'OK' if PASS else 'FAIL'} — A5 indexer score reduction precision")
    sys.exit(0 if PASS else 1)


if __name__ == "__main__":
    main()
