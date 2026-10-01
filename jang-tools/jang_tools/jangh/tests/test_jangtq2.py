"""jangtq2 correctness suite. Run: python -m jang_tools.jangh.tests.test_jangtq2
Fails closed: every check asserts a non-empty, non-trivial baseline."""
import sys, time, numpy as np, mlx.core as mx
from jang_tools.jangh import format as F, kernels as Kn

FAILS = []
def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok: FAILS.append(name)

rng = np.random.default_rng(0)
print("== format")
ref_mse = {2: 0.11748, 3: 0.03462, 4: 0.00969}   # v2 odd-cubic family (2-bit == Lloyd-Max)
for b, want in ref_mse.items():
    got = F.gaussian_mse(b)
    check(f"v2 codebook N(0,1) mse b{b}", abs(got - want) / want < 2e-3, f"{got:.5f} vs {want}")
for b in F.SUPPORTED_BITS:
    for K in (32, 96, 4096):
        q = mx.array(rng.integers(0, 1 << b, size=(5, K)).astype(np.uint8))
        p = F.pack_bitstream(q, b)
        check(f"roundtrip b{b} K{K}", mx.array_equal(F.unpack_bitstream(p, b, K), q).item())
        if K % 64 == 0:   # byte-identical to MLX affine packing: dequantize with scale 1 / bias 0 returns q
            d = mx.dequantize(p, mx.ones((5, K // 64), mx.float32), mx.zeros((5, K // 64), mx.float32), group_size=64, bits=b)
            check(f"MLX-affine-identical b{b} K{K}", mx.array_equal(d, q.astype(mx.float32)).item())
for K in (2048, 4096):
    W = mx.random.normal((8, K))
    s = mx.array(F.make_signs(K, 7).astype(np.float32))
    back = F.unrotate_in(F.rotate_in(W, s), s)
    e = (mx.abs(back - W).max()).item()
    check(f"rotation inverse K{K}", e < 1e-4, f"max|d|={e:.1e}")
    # W x == (W R^T)(R x)
    x = mx.random.normal((K,))
    y1 = W @ x; y2 = F.rotate_in(W, s) @ F.rotate_act(x, s)
    check(f"rotation equivariance K{K}", (mx.linalg.norm(y1 - y2) / mx.linalg.norm(y1)).item() < 1e-5)

def make_expert_bank(E, N, K, b):
    q = mx.array(rng.integers(0, 1 << b, size=(E, N, K)).astype(np.uint8))
    p = F.pack_bitstream(q, b)
    sc = (mx.random.uniform(shape=(E, N)) * 0.02 + 0.005).astype(mx.float16)
    W = F.dequant_rows(q, sc, b)       # (E,N,K) float32 in the rotated basis
    return p, sc, W

print("== decode qmv (float32 out; must match float32 reference ~1e-6)")
for b in F.SUPPORTED_BITS:
    cb = mx.array(F.codebook(b))
    for (E, N, K) in [(16, 2048, 4096), (9, 2050, 1056), (5, 13, 96)]:
        p, sc, W = make_expert_bank(E, N, K, b)
        idx = mx.array(rng.choice(E, 4, replace=E < 4).astype(np.uint32))
        # shared x (gate/up style)
        x = mx.random.normal((1, K))
        y = Kn.gather_qmv(x, p, sc, cb, idx, b, x_per_dispatch=False)
        ref = mx.einsum("dnk,k->dn", W[idx], x[0])
        e = (mx.linalg.norm(y - ref) / mx.linalg.norm(ref)).item()
        check(f"qmv single b{b} E{E} N{N} K{K}", e < 2e-5 and mx.linalg.norm(ref).item() > 0, f"rel {e:.1e}")
        # per-dispatch x (down style)
        xs = mx.random.normal((4, K))
        y = Kn.gather_qmv(xs, p, sc, cb, idx, b, x_per_dispatch=True)
        ref = mx.einsum("dnk,dk->dn", W[idx], xs)
        e = (mx.linalg.norm(y - ref) / mx.linalg.norm(ref)).item()
        check(f"qmv per-dispatch b{b} E{E} N{N} K{K}", e < 2e-5, f"rel {e:.1e}")
        # fused gate/up clamped swiglu
        pu, su, Wu = make_expert_bank(E, N, K, b)
        for lim in (0.0, 10.0, 0.05):
            y = Kn.gather_qmv(x * 30, p, sc, cb, idx, b, x_per_dispatch=False, packed_u=pu, scales_u=su, limit=lim)
            g = mx.einsum("dnk,k->dn", W[idx], x[0] * 30); u = mx.einsum("dnk,k->dn", Wu[idx], x[0] * 30)
            if lim > 0: g = mx.minimum(g, lim); u = mx.clip(u, -lim, lim)
            ref = g * mx.sigmoid(g) * u
            e = (mx.linalg.norm(y - ref) / mx.linalg.norm(ref)).item()
            check(f"qmv fused b{b} N{N} K{K} lim{lim}", e < 1e-4, f"rel {e:.1e}")

print("== prefill NAX qmm (bf16 in/out; reference uses bf16-rounded weights; bound = bf16 output rounding)")
def sorted_rows(M, E):
    idx = np.sort(rng.integers(0, E, size=M)).astype(np.uint32)
    return mx.array(idx)
for b in (2, 3, 4, 8):
    cb = mx.array(F.codebook(b))
    for (E, N, K) in [(12, 2048, 4096), (7, 4096, 2048), (5, 2080, 128)]:
        p, sc, W = make_expert_bank(E, N, K, b)
        pu, su, Wu = make_expert_bank(E, N, K, b)
        Wb = W.astype(mx.bfloat16).astype(mx.float32); Wub = Wu.astype(mx.bfloat16).astype(mx.float32)
        for M in (1, 7, 63, 65, 130, 1000):
            idx = sorted_rows(M, E)
            x = (mx.random.normal((M, K)) * 0.5).astype(mx.bfloat16)
            xf = x.astype(mx.float32)
            y = Kn.gather_qmm_sorted(x, p, sc, cb, idx, b).astype(mx.float32)
            ref = mx.einsum("mnk,mk->mn", Wb[idx], xf)
            e = (mx.linalg.norm(y - ref) / mx.linalg.norm(ref)).item()
            check(f"nax single b{b} N{N} K{K} M{M}", e < 6e-3, f"rel {e:.1e}")
            y = Kn.gather_qmm_sorted(x * 20, p, sc, cb, idx, b, packed_u=pu, scales_u=su, limit=10.0).astype(mx.float32)
            g = mx.minimum(mx.einsum("mnk,mk->mn", Wb[idx], xf * 20), 10.0)
            u = mx.clip(mx.einsum("mnk,mk->mn", Wub[idx], xf * 20), -10.0, 10.0)
            ref = g * mx.sigmoid(g) * u
            e = (mx.linalg.norm(y - ref) / mx.linalg.norm(ref)).item()
            check(f"nax fused b{b} N{N} K{K} M{M}", e < 8e-3, f"rel {e:.1e}")
print("== decode weighted-down (router-weighted sum fused)")
for b in F.SUPPORTED_BITS:
    cb = mx.array(F.codebook(b))
    for (E, N, K, T, kt) in [(16, 4096, 2048, 1, 8), (9, 2050, 1056, 3, 6), (5, 13, 96, 2, 4)]:
        p, sc, W = make_expert_bank(E, N, K, b)
        idx = mx.array(np.stack([rng.choice(E, kt, replace=False) for _ in range(T)]).astype(np.uint32))
        w = mx.random.uniform(shape=(T, kt))
        h = mx.random.normal((T * kt, K))
        for od in (mx.float32, mx.bfloat16):
            y = Kn.gather_qmv_weighted_down(h.astype(od), p, sc, cb, idx, w, b, od).astype(mx.float32)
            ref = mx.einsum("tknk2,tkk2->tn".replace("k2","j").replace("tknj","tknj"), W[idx], h.astype(od).astype(mx.float32).reshape(T, kt, K)) if False else \
                  mx.einsum("tknj,tkj,tk->tn", W[idx], h.astype(od).astype(mx.float32).reshape(T, kt, K), w)
            e = (mx.linalg.norm(y - ref) / mx.linalg.norm(ref)).item()
            tol = 2e-5 if od == mx.float32 else 6e-3
            check(f"wdown b{b} N{N} K{K} T{T} k{kt} {od}", e < tol and mx.linalg.norm(ref).item() > 0, f"rel {e:.1e}")
print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + str(len(FAILS))}")
for f in FAILS: print("   ", f)
sys.exit(1 if FAILS else 0)
