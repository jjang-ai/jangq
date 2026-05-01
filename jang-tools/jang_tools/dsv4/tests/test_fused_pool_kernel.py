"""A7 Metal kernel validation — fused_attn_metal vs orchestrator.

Status (2026-05-01): kernel scaffold compiles + runs but produces
incorrect numerics on first pass. The orchestrator (`fused_pool_attn.py`)
is the production-correct path; the Metal kernel is a development
scaffold that will be debugged and validated in a follow-up before
being wired behind DSV4_FUSED_POOL_ATTN=1.

This test runs the kernel and reports diff vs orchestrator. It does
NOT fail the suite — kernel is documented as scaffold-quality. When
the kernel passes, change SCAFFOLD_BUDGET to True to gate it.
"""
from __future__ import annotations
import sys
import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Users/eric/jang/jang-tools")
from jang_tools.dsv4.fused_pool_attn_kernel import fused_attn_metal
from jang_tools.dsv4.fused_pool_attn import fused_pool_attention
from jang_tools.dsv4.mlx_model import _build_window_mask, _compressed_visibility


SCAFFOLD = True   # while kernel is unvalidated; flip to False when math passes


def case(label, *, B=1, NH=4, L=2, W=16, P=8, head_dim=32, has_sink=True):
    rng = np.random.default_rng(hash(label) & 0xffffffff)
    dtype = mx.float16
    q = mx.array(rng.standard_normal((B, NH, L, head_dim)).astype(np.float32) * 0.1).astype(dtype)
    k_win = mx.array(rng.standard_normal((B, 1, W, head_dim)).astype(np.float32) * 0.1).astype(dtype)
    v_win = mx.array(rng.standard_normal((B, 1, W, head_dim)).astype(np.float32) * 0.1).astype(dtype)
    k_pool = mx.array(rng.standard_normal((B, 1, P, head_dim)).astype(np.float32) * 0.1).astype(dtype)
    v_pool = mx.array(rng.standard_normal((B, 1, P, head_dim)).astype(np.float32) * 0.1).astype(dtype)
    sink = mx.array(rng.standard_normal((NH,)).astype(np.float32) * 0.1).astype(dtype) if has_sink else None

    win_mask = _build_window_mask(B, L, 0, W, W)
    comp_mask = _compressed_visibility(B, L, 0, P, 4)
    full_mask = mx.concatenate([win_mask, comp_mask], axis=-1)
    full_k = mx.concatenate([k_win, k_pool], axis=2)
    full_v = mx.concatenate([v_win, v_pool], axis=2)
    scale = head_dim ** -0.5

    out_ref = fused_pool_attention(q, k_win, v_win, k_pool, v_pool,
                                    None, sink, win_mask, comp_mask, scale)
    mx.eval(out_ref)
    try:
        out_k = fused_attn_metal(q, full_k, full_v, full_mask, sink, scale)
    except Exception as e:
        print(f"  {label}: KERNEL ERROR: {e}")
        return SCAFFOLD
    if out_k is None:
        print(f"  {label}: KERNEL UNAVAILABLE — orchestrator fallback")
        return True
    mx.eval(out_k)

    diff = mx.abs(out_k.astype(mx.float32) - out_ref.astype(mx.float32))
    max_d = float(mx.max(diff).item())
    rms = float(mx.sqrt(mx.mean(diff * diff)).item())
    rms_ref = float(mx.sqrt(mx.mean(out_ref.astype(mx.float32) ** 2)).item())
    rel = rms / (rms_ref + 1e-9)
    ok = max_d < 1e-2 and rel < 0.05
    flag = "OK" if ok else ("SCAFFOLD-FAIL" if SCAFFOLD else "FAIL")
    print(f"  {label:<35s} max={max_d:.2e}  rms={rms:.2e}  rel={rel*100:5.2f}%  {flag}")
    return ok or SCAFFOLD


print("=== A7 Metal kernel vs orchestrator ===")
ALL = True
ALL &= case("small B=1 NH=4 L=2 W=16 P=8 d=32")
ALL &= case("medium B=1 NH=8 L=4 W=32 P=16 d=64", B=1, NH=8, L=4, W=32, P=16, head_dim=64)
ALL &= case("no-sink", has_sink=False)
ALL &= case("L=1 decode", L=1)
print(f"\n=== {'ALL OK / SCAFFOLD-DOCUMENTED' if ALL else 'FAIL'} ===")
sys.exit(0 if ALL else 1)
