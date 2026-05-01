"""Synthetic shape regression for HSA + CSA + SWA mask construction.

Goal: exercise the long-ctx mask code paths in `mlx_model.py` lines 1056-1126
WITHOUT loading the full 79 GB DSV4-Flash bundle. We feed mock K/V/q tensors
through the mask helpers and the SDPA call to verify every (B, S, offset,
compress_ratio, has_indexer) combination produces broadcast-compatible
shapes.

If this harness passes for L=256+ but the real-weight prefill still throws
"Shapes (L,L) and (1,H,L,big) cannot be broadcast", the bug is somewhere
upstream of the mask code (e.g. an mlx-lm-supplied mask that we're failing
to discard).
"""
from __future__ import annotations

import sys
import mlx.core as mx

sys.path.insert(0, "/Users/eric/jang/jang-tools")
from jang_tools.dsv4.mlx_model import (
    _build_window_mask,
    _compressed_visibility,
)
import mlx.nn as nn
from mlx.core.fast import scaled_dot_product_attention as sdpa


N_HEADS = 64
HEAD_DIM = 512
SLIDING_WINDOW = 128
SCALE = HEAD_DIM ** -0.5


def _decode_one(B, S, offset, win_len, compressed_len, ratio, has_indexer,
                use_external_mask):
    """Build q + full_kv + mask exactly the way DeepseekV4Attention does."""
    q = mx.zeros((B, N_HEADS, S, HEAD_DIM), dtype=mx.bfloat16)
    full_kv = mx.zeros((B, 1, win_len + compressed_len, HEAD_DIM), dtype=mx.bfloat16)

    # Mask construction (mirror lines 1108-1120 of mlx_model.py)
    comp_visible = _compressed_visibility(B, S, offset, compressed_len, ratio)
    if has_indexer and compressed_len > 0:
        # Fake an indexer top-k that picks every other entry
        idx = mx.arange(compressed_len, dtype=mx.int32)
        sel = (idx % 2 == 0)
        # broadcast to (B, 1, S, P)
        comp_visible = comp_visible & sel[None, None, None, :]

    if compressed_len > 0:
        win_mask = _build_window_mask(B, S, offset, SLIDING_WINDOW, win_len)
        sdpa_mask = mx.concatenate([win_mask, comp_visible], axis=-1)
        external = None
    elif use_external_mask:
        # Plain SWA layer: caller's mlx-lm supplied causal mask of shape
        # (S, win_len + offset) — match what mlx-lm passes during prefill.
        cache_len = win_len
        external = mx.tril(mx.ones((S, cache_len), dtype=mx.bool_),
                           k=cache_len - S)[None, None]
        sdpa_mask = external
    else:
        sdpa_mask = None

    # Run SDPA — this is what fires the broadcast error in the real bug
    out = sdpa(q, full_kv, full_kv, scale=SCALE, mask=sdpa_mask)
    mx.eval(out)
    return out.shape, sdpa_mask.shape if sdpa_mask is not None else None


# Test matrix — every interesting (S, offset, compress_ratio) combo
CASES = [
    # (label,          B, S,   offset, win_len, ratio,  has_indexer,    use_external)
    ("SWA  L=1  off=0",  1, 1,   0, 0, 0, False, True),    # decode first tok
    ("SWA  L=1  off=64", 1, 1,  64, 64, 0, False, True),   # decode mid-window
    ("SWA  L=64 prefill",1, 64,  0, 64, 0, False, True),   # tiny prefill
    ("SWA  L=256 prefill",1, 256, 0, 128, 0, False, True), # prefill > window
    # HSA layers (compress_ratio=128, dense pool, no indexer)
    ("HSA  L=1   off=128 P=1",   1, 1,   128, 128, 128, False, False),
    ("HSA  L=256 prefill P=2",   1, 256, 0,   128, 128, False, False),
    ("HSA  L=512 prefill P=4",   1, 512, 0,   128, 128, False, False),
    ("HSA  L=2048 prefill P=16", 1, 2048,0,   128, 128, False, False),
    # CSA layers (compress_ratio=4, sparse top-k, has_indexer)
    ("CSA  L=1   off=128 P=32",  1, 1,   128, 128, 4, True, False),
    ("CSA  L=256 prefill P=64",  1, 256, 0,   128, 4, True, False),
    ("CSA  L=512 prefill P=128", 1, 512, 0,   128, 4, True, False),
    ("CSA  L=2048 prefill P=512",1, 2048,0,   128, 4, True, False),
]

print(f"{'case':<40s}  {'out_shape':<26s} {'mask_shape':<26s} status")
print("-" * 100)
PASS = 0; FAIL = 0; ERRORS = []
for label, B, S, offset, win_len, ratio, idx, ext in CASES:
    P = 0
    if ratio:
        P = (S + offset) // ratio if (S + offset) >= ratio else 1
    try:
        out_sh, mask_sh = _decode_one(B, S, offset, win_len, P, ratio, idx, ext)
        ok = out_sh == (B, N_HEADS, S, HEAD_DIM)
        status = "PASS" if ok else f"WRONG_SHAPE expected (B,{N_HEADS},{S},{HEAD_DIM})"
        print(f"{label:<40s}  {str(out_sh):<26s} {str(mask_sh):<26s} {status}")
        if ok: PASS += 1
        else: FAIL += 1; ERRORS.append((label, out_sh, mask_sh))
    except Exception as e:
        FAIL += 1; ERRORS.append((label, None, str(e)))
        print(f"{label:<40s}  ERROR: {type(e).__name__}: {e}")

print(f"\n{'='*60}")
print(f"  PASS: {PASS}/{PASS + FAIL}    FAIL: {FAIL}")
if ERRORS:
    print("\n  Failures:")
    for label, sh, msg in ERRORS:
        print(f"    {label}  →  {sh!r}  /  {msg!r}")
sys.exit(0 if FAIL == 0 else 1)
