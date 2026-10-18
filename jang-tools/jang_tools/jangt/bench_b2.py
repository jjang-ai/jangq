"""B2: real gather-GEMV speed with memory traffic. JANGT pop1 decode kernel vs JANGH gather_qmv (same skeleton: 2
simdgroups x 4 rows, 16 values per lane, uint32 loads, scale once per row after simd_sum) and MLX affine gather_qmm,
on N0.5 expert shapes (E=256: gate/up 4096->2048, down 2048->4096), dispatches = rows x top-8.
JANGT trick: pop1 values are integers popcount(h) - C, so sum(x*(pop-C)) = sum(x*pop) - C*sum(x): the offset leaves the
inner loop (C*sum(x) is computed once per block and shared by all rows). Bitstream semantics here are speed-only
(random bits); correctness is tested separately against the numpy oracle.
Usage: python -m jang_tools.jangt.bench_b2 [--repeats 7]"""
import argparse, functools, json, sys, time
from pathlib import Path
import mlx.core as mx

from jang_tools.jangh import kernels as JK  # noqa: E402

MULT, ADD = 0x9E3779B1, 0x7F4A7C15


HASHES = {
    "mul_add": "uint h = s * 0x9E3779B1u + 0x7F4A7C15u;",
    "mul":     "uint h = s * 0x9E3779B1u;",
    "mul24":   "uint h = mul24(s, 0x9E3779u);",
    "xorshift": "uint h = s ^ (s << 9); h ^= (h >> 7); h ^= (h << 13);",
    "fp_frac": "float fv = fract(float(s) * 0.6180339887f); uint h = as_type<uint>(fv);",   # speed probe only
}
# ablations (speed only): VAL = how h becomes a float; EXTRACT = how the state is read
VALS = {"popcount": "float(popcount(h))", "shift": "float(h >> 24)"}
EXTRACTS = {
    "win64": "ulong win = (ulong(j == 0u ? prev : wr[j - 1u]) << 32) | ulong(cur); uint s = uint(win >> (32u - KB * (t + 1u))) & 0xFFFFu;",
    "win32": "uint s = (cur >> (KB * t)) & 0xFFFFu;",
}


@functools.lru_cache(maxsize=None)
def jangt_kernel(K_bits: int, hash_name: str = "mul_add", val: str = "popcount", extract: str = "win64"):
    nw = K_bits // 2                       # uint32 words per 16-value lane chunk (K=2: 1, K=4: 2)
    src = f"""
    uint Kd = meta[0], N = meta[1], xdiv = meta[2];
    const uint RW = Kd * {K_bits}u / 32u;              // words per row
    uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    uint row0 = threadgroup_position_in_grid.y * 8u + sgi * 4u;
    uint disp = threadgroup_position_in_grid.z; uint e = idx[disp];
    const device uint* wp = w + ((size_t)e * N + row0) * RW + lane * {nw}u;
    auto xp = x + (size_t)(disp / xdiv) * Kd + lane * 16u;
    float acc[4] = {{0.0f, 0.0f, 0.0f, 0.0f}}; float xsum = 0.0f;
    for (uint k0 = 0; k0 < Kd; k0 += 512u) {{
      float xt[16];
      #pragma unroll
      for (uint i = 0; i < 16u; i++) {{ xt[i] = float(xp[i]); xsum += xt[i]; }}
      #pragma unroll
      for (uint r = 0; r < 4u; r++) {{
        const device uint* wr = wp + r * RW;
        uint prev = lane == 0u && k0 == 0u ? wr[RW - 1u] : wr[-1];   // 16 bits of trellis history (tail-biting)
        float a = 0.0f;
        #pragma unroll
        for (uint j = 0; j < {nw}u; j++) {{
          uint cur = wr[j];
          #pragma unroll
          for (uint t = 0; t < {32 // K_bits}u; t++) {{
            {EXTRACTS[extract].replace("KB", str(K_bits) + "u")}
            {HASHES[hash_name]}
            a = fma(xt[j * {32 // K_bits}u + t], {VALS[val]}, a);
          }}
        }}
        acc[r] += a;
      }}
      wp += {512 * K_bits // 32}u;                       // one block = 512 values = 512*K/32 words
      xp += 512u;
    }}
    float xs = simd_sum(xsum);
    #pragma unroll
    for (uint r = 0; r < 4u; r++) {{
      float s = simd_sum(acc[r]);
      if (lane == 0u && row0 + r < N)
        out[(size_t)disp * N + row0 + r] = (s - 16.0f * xs) * float(sv[(size_t)e * N + row0 + r]);
    }}
    """
    return mx.fast.metal_kernel(name=f"jangt_pop1_qmv_k{K_bits}_{hash_name}_{val}_{extract}", input_names=["x", "w", "sv", "idx", "meta"],
                                output_names=["out"], source=src)


def jangt_gemv(x, w, sv, idx, K_bits, xdiv, hash_name="mul_add", val="popcount", extract="win64"):
    ndisp = idx.size; Kd = x.shape[1]; N = sv.shape[1]
    k = jangt_kernel(K_bits, hash_name, val, extract)
    return k(inputs=[x, w, sv, idx, mx.array([Kd, N, xdiv], dtype=mx.uint32)], grid=(64, N // 8, ndisp),
             threadgroup=(64, 1, 1), output_shapes=[(ndisp, N)], output_dtypes=[mx.float32])[0]


def timeit(fn, idxs, repeats):
    """fn(idx): each call in a timed batch uses a DIFFERENT random expert set over the full bank, so weights come from
    DRAM, not the system cache (the first version reused one set and measured cache-hot >1 TB/s)."""
    mx.eval(fn(idxs[0])); ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        outs = [fn(i) for i in idxs]
        mx.eval(outs); ts.append((time.perf_counter() - t) / len(idxs))
    ts.sort(); return ts[len(ts) // 2]


@functools.lru_cache(maxsize=None)
def jangt_kernel_w16(K_bits: int):
    """Packed 16-bit decode: two states per ushort2 op (16-bit hash, popcount), half2 FMA into a per-block half2
    partial that is folded into fp32 once per block (precision: <=16 products per partial). Speed probe (win32 states)."""
    nw = K_bits // 2; per = 32 // K_bits
    src = f"""
    uint Kd = meta[0], N = meta[1], xdiv = meta[2];
    const uint RW = Kd * {K_bits}u / 32u;
    uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    uint row0 = threadgroup_position_in_grid.y * 8u + sgi * 4u;
    uint disp = threadgroup_position_in_grid.z; uint e = idx[disp];
    const device uint* wp = w + ((size_t)e * N + row0) * RW + lane * {nw}u;
    auto xp = x + (size_t)(disp / xdiv) * Kd + lane * 16u;
    float acc[4] = {{0.0f, 0.0f, 0.0f, 0.0f}}; float xsum = 0.0f;
    for (uint k0 = 0; k0 < Kd; k0 += 512u) {{
      half2 xh[8];
      #pragma unroll
      for (uint i = 0; i < 8u; i++) {{ xh[i] = half2(half(xp[2u*i]), half(xp[2u*i+1u])); xsum += float(xp[2u*i]) + float(xp[2u*i+1u]); }}
      #pragma unroll
      for (uint r = 0; r < 4u; r++) {{
        const device uint* wr = wp + r * RW;
        half2 a2 = half2(0.0h, 0.0h);
        #pragma unroll
        for (uint j = 0; j < {nw}u; j++) {{
          uint cur = wr[j];
          #pragma unroll
          for (uint t = 0; t < {per}u; t += 2u) {{
            ushort2 s2 = ushort2((uint2(cur) >> uint2({K_bits}u * t, {K_bits}u * (t + 1u))) & uint2(0xFFFFu));
            ushort2 h2 = s2 * ushort2(0x9E37) + ushort2(0x7F4A);
            half2 v2 = half2(popcount(h2));
            a2 = fma(xh[(j * {per}u + t) / 2u], v2, a2);
          }}
        }}
        acc[r] += float(a2.x) + float(a2.y);
      }}
      wp += {512 * K_bits // 32}u; xp += 512u;
    }}
    float xs = simd_sum(xsum);
    #pragma unroll
    for (uint r = 0; r < 4u; r++) {{
      float sm = simd_sum(acc[r]);
      if (lane == 0u && row0 + r < N)
        out[(size_t)disp * N + row0 + r] = (sm - 8.0f * xs) * float(sv[(size_t)e * N + row0 + r]);
    }}
    """
    return mx.fast.metal_kernel(name=f"jangt_pop16_qmv_k{K_bits}", input_names=["x", "w", "sv", "idx", "meta"],
                                output_names=["out"], source=src)


def jangt_gemv_w16(x, w, sv, idx, K_bits, xdiv):
    ndisp = idx.size; Kd = x.shape[1]; N = sv.shape[1]
    return jangt_kernel_w16(K_bits)(inputs=[x, w, sv, idx, mx.array([Kd, N, xdiv], dtype=mx.uint32)], grid=(64, N // 8, ndisp),
             threadgroup=(64, 1, 1), output_shapes=[(ndisp, N)], output_dtypes=[mx.float32])[0]


@functools.lru_cache(maxsize=None)
def jangt_kernel_v2(K_bits: int, acc_mode: str = "half2"):
    """Vector trellis V=2: one 16-bit state per PAIR of weights. h = s*M; the 32-bit word (h & MASK) | OR is
    reinterpreted as half2 = the two code values (no popcount, no int->float). acc_mode: 'half2' = half2 FMA into a
    per-block partial folded to fp32 once per block; 'f32' = float2 convert + two fp32 FMAs. Speed probe."""
    nw = K_bits // 2; per = 32 // K_bits                       # weights per word
    acc = ("half2 v2 = as_type<half2>((h & 0x83FF83FFu) | 0x38003800u); a2 = fma(xh[(j * PER + t) / 2u], v2, a2);"
           if acc_mode == "half2" else
           "a = fma(xt[j * PER + t], float(popcount(h & 0xFFFFu)), a); a = fma(xt[j * PER + t + 1u], float(popcount(h >> 16)), a);"
           if acc_mode == "pop2" else
           "half2 v2 = as_type<half2>((h & 0x83FF83FFu) | 0x38003800u); float2 vf = float2(v2); "
           "a = fma(xt[j * PER + t], vf.x, a); a = fma(xt[j * PER + t + 1u], vf.y, a);")
    src = f"""
    uint Kd = meta[0], N = meta[1], xdiv = meta[2];
    const uint RW = Kd * {K_bits}u / 32u;
    uint sgi = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    uint row0 = threadgroup_position_in_grid.y * 8u + sgi * 4u;
    uint disp = threadgroup_position_in_grid.z; uint e = idx[disp];
    const device uint* wp = w + ((size_t)e * N + row0) * RW + lane * {nw}u;
    auto xp = x + (size_t)(disp / xdiv) * Kd + lane * 16u;
    float accr[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
    for (uint k0 = 0; k0 < Kd; k0 += 512u) {{
      float xt[16]; half2 xh[8];
      #pragma unroll
      for (uint i = 0; i < 16u; i++) xt[i] = float(xp[i]);
      #pragma unroll
      for (uint i = 0; i < 8u; i++) xh[i] = half2(half(xt[2u*i]), half(xt[2u*i+1u]));
      #pragma unroll
      for (uint r = 0; r < 4u; r++) {{
        const device uint* wr = wp + r * RW;
        half2 a2 = half2(0.0h, 0.0h); float a = 0.0f;
        #pragma unroll
        for (uint j = 0; j < {nw}u; j++) {{
          uint cur = wr[j];
          #pragma unroll
          for (uint t = 0; t < {per}u; t += 2u) {{
            uint s = (cur >> ({K_bits}u * t)) & 0xFFFFu;
            uint h = s * 0x9E3779B1u;
            {acc.replace("PER", str(per) + "u")}
          }}
        }}
        accr[r] += a + float(a2.x) + float(a2.y);
      }}
      wp += {512 * K_bits // 32}u; xp += 512u;
    }}
    #pragma unroll
    for (uint r = 0; r < 4u; r++) {{
      float sm = simd_sum(accr[r]);
      if (lane == 0u && row0 + r < N) out[(size_t)disp * N + row0 + r] = sm * float(sv[(size_t)e * N + row0 + r]);
    }}
    """
    return mx.fast.metal_kernel(name=f"jangt_v2_qmv_k{K_bits}_{acc_mode}", input_names=["x", "w", "sv", "idx", "meta"],
                                output_names=["out"], source=src)


def jangt_gemv_v2(x, w, sv, idx, K_bits, xdiv, acc_mode):
    ndisp = idx.size; Kd = x.shape[1]; N = sv.shape[1]
    return jangt_kernel_v2(K_bits, acc_mode)(inputs=[x, w, sv, idx, mx.array([Kd, N, xdiv], dtype=mx.uint32)],
             grid=(64, N // 8, ndisp), threadgroup=(64, 1, 1), output_shapes=[(ndisp, N)], output_dtypes=[mx.float32])[0]


def v2_sweep(repeats=7):
    E, N, Kd = 256, 2048, 4096
    sv = mx.random.uniform(shape=(E, N)).astype(mx.float16)
    for b in (2, 3, 4) if False else (2, 4):
        wb = mx.random.randint(0, 2**31, (E, N, Kd * b // 32), dtype=mx.uint32)
        for M in (1, 8):
            idxs = [mx.random.permutation(E)[: M * 8].astype(mx.uint32) for _ in range(48)]; mx.eval(idxs)
            x = mx.random.normal((M, Kd)).astype(mx.bfloat16)
            th = timeit(lambda i: JK.gather_qmv(x, wb, sv, None, i, b, x_per_dispatch=False), idxs, repeats)
            t1 = timeit(lambda i: jangt_gemv(x, wb, sv, i, b, 8, "mul", "popcount", "win32"), idxs, repeats)
            out = f"K={b} M={M}: JANGH {th*1e6:6.1f} us | JANGT V1 pop1 {t1*1e6:6.1f} ({th/t1:4.2f})"
            for am in ("half2", "f32", "pop2"):
                tv = timeit(lambda i: jangt_gemv_v2(x, wb, sv, i, b, 8, am), idxs, repeats)
                out += f" | V2 {am} {tv*1e6:6.1f} ({th/tv:4.2f})"
            print(out, flush=True)


def hash_sweep(repeats=7):
    E, N, Kd, bits, M = 256, 2048, 4096, 2, 1
    wj = mx.random.randint(0, 2**31, (E, N, Kd * bits // 32), dtype=mx.uint32); sv = mx.random.uniform(shape=(E, N)).astype(mx.float16)
    idxs = [mx.random.permutation(E)[: M * 8].astype(mx.uint32) for _ in range(48)]; mx.eval(idxs)
    x = mx.random.normal((M, Kd)).astype(mx.bfloat16)
    th = timeit(lambda i: JK.gather_qmv(x, wj, sv, None, i, bits, x_per_dispatch=False), idxs, repeats)
    print(f"JANGH 2-bit gate_up M1: {th*1e6:.1f} us")
    for hn, val, ex in (("mul", "popcount", "win64"), ("mul", "shift", "win64"), ("mul", "popcount", "win32"),
                        ("mul", "shift", "win32")):
        tj = timeit(lambda i: jangt_gemv(x, wj, sv, i, bits, 8, hn, val, ex), idxs, repeats)
        print(f"  JANGT hash={hn} val={val:8s} extract={ex}: {tj*1e6:6.1f} us  JANGH/JANGT {th/tj:4.2f}", flush=True)
    for b in (2, 4):
        wb = mx.random.randint(0, 2**31, (E, N, Kd * b // 32), dtype=mx.uint32)
        thb = timeit(lambda i: JK.gather_qmv(x, wb, sv, None, i, b, x_per_dispatch=False), idxs, repeats)
        tw = timeit(lambda i: jangt_gemv_w16(x, wb, sv, i, b, 8), idxs, repeats)
        print(f"  JANGT packed-16 K={b}: {tw*1e6:6.1f} us  JANGH {thb*1e6:6.1f} us  JANGH/JANGT {thb/tw:4.2f}", flush=True)


def main():
    if "--hash-sweep" in sys.argv:
        return hash_sweep()
    if "--v2-sweep" in sys.argv:
        return v2_sweep()
    ap = argparse.ArgumentParser(); ap.add_argument("--repeats", type=int, default=7); ap.add_argument("--out")
    a = ap.parse_args(); E = 256; res = {}
    for (name, N, Kd) in (("gate_up", 2048, 4096), ("down", 4096, 2048)):
        for bits in (2, 4):
            wj = mx.random.randint(0, 2**31, (E, N, Kd * bits // 32), dtype=mx.uint32)
            sv = mx.random.uniform(shape=(E, N)).astype(mx.float16)
            # JANGH bank (same bytes): packed (E, N, K*bits/32) uint32, fp16 row scales
            for M in (1, 4, 8):
                idxs = [mx.random.permutation(E)[: M * 8].astype(mx.uint32) for _ in range(48)]
                mx.eval(idxs)
                x = mx.random.normal((M, Kd)).astype(mx.bfloat16)
                tj = timeit(lambda i: jangt_gemv(x, wj, sv, i, bits, 8), idxs, a.repeats)
                th = timeit(lambda i: JK.gather_qmv(x, wj, sv, None, i, bits, x_per_dispatch=False), idxs, a.repeats)
                gbytes = M * 8 * N * Kd * bits / 8 / 1e9
                res[f"{name}_b{bits}_M{M}"] = {"jangt_us": tj * 1e6, "jangh_us": th * 1e6, "ratio_jangh_over_jangt": th / tj,
                                               "jangt_GBps": gbytes / tj, "jangh_GBps": gbytes / th}
                print(f"{name:8s} b{bits} M{M}: JANGT {tj*1e6:7.1f} us ({gbytes/tj:5.0f} GB/s)  JANGH {th*1e6:7.1f} us ({gbytes/th:5.0f} GB/s)  JANGH/JANGT {th/tj:4.2f}", flush=True)
    if a.out: json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
