"""TQSwitchGLU.routed / __call__ correctness vs a per-token dequantized reference (bounded memory), every path:
decode (T*k < 64) and prefill (sorted NAX), bits 2/3/4, mixed gate_up/down bits, bf16 in/out. Fails closed."""
import sys, numpy as np, mlx.core as mx
from jang_tools.jangh import format as F
from jang_tools.jangh.encode import dequant
from jang_tools.jangh.switch import TQSwitchGLU
D, I, E, KT = 1024, 512, 40, 8
rng = np.random.default_rng(1)
FAILS = []
def mk(bgu, bd, rot, bu=0):
    m = TQSwitchGLU(D, I, E, bgu, bd, 10.0, rotation_gate_up=rot, rotation_down=rot, bits_up=bu)
    for lin, b, K in ((m.gate_proj, bgu, D), (m.up_proj, bu or bgu, D), (m.down_proj, bd, I)):
        q = mx.array(rng.integers(0, 1 << b, size=(E, lin.output_dims, K)).astype(np.uint8))
        lin.tq2_packed = F.pack_bitstream(q, b)
        lin.tq2_scales = (mx.random.uniform(shape=(E, lin.output_dims)) * 0.05 + 0.02).astype(mx.float16)
        lin._q = q
    return m
def ref_routed(m, x, idx, w):
    g, u, d = m.gate_proj, m.up_proj, m.down_proj
    out = []
    for t in range(x.shape[0]):
        acc = mx.zeros((D,))
        for j in range(KT):
            e = int(idx[t, j].item())
            Wg = dequant(g._q[e], g.tq2_scales[e], g.bits); Wu = dequant(u._q[e], u.tq2_scales[e], u.bits)
            Wd = dequant(d._q[e], d.tq2_scales[e], d.bits)
            xf = x[t].astype(mx.float32)
            if g.rotated: xf = mx.hadamard_transform(xf.reshape(-1, 32)).reshape(-1)
            gg = mx.minimum(Wg @ xf, 10.0); uu = mx.clip(Wu @ xf, -10.0, 10.0)
            hh = gg * mx.sigmoid(gg) * uu
            if d.rotated: hh = mx.hadamard_transform(hh.reshape(-1, 32)).reshape(-1)
            acc = acc + w[t, j] * (Wd @ hh)
        out.append(acc)
    return mx.stack(out)
CFG = [(2, 2, 'none', 0), (3, 3, 'none', 0), (4, 4, 'none', 0), (2, 3, 'none', 0), (2, 2, 'hadamard32', 0), (2, 3, 'hadamard32', 0), (3, 3, 'hadamard32', 0), (4, 4, 'hadamard32', 0),
       (3, 3, 'hadamard32', 2), (2, 3, 'hadamard32', 3), (4, 2, 'hadamard32', 3), (2, 2, 'none', 3), (3, 4, 'none', 4)]   # mixed gate/up bits
for bgu, bd, rot, bu in CFG:
    m = mk(bgu, bd, rot, bu)
    for T in (1, 3, 7, 8, 16, 33):
        x = (mx.random.normal((T, D)) * 1.5).astype(mx.bfloat16)
        idx = mx.array(np.stack([rng.choice(E, KT, replace=False) for _ in range(T)]).astype(np.uint32))
        w = mx.random.uniform(shape=(T, KT))
        r = ref_routed(m, x, idx, w)
        for name, y in (("routed", m.routed(x, idx, w)), ("call+sum", (m(x, idx).astype(mx.float32) * w[..., None]).sum(-2))):
            y = y.astype(mx.float32)
            e = (mx.sqrt(mx.sum((y - r) ** 2)) / mx.sqrt(mx.sum(r ** 2))).item()
            path = "decode" if T * KT < 64 else "prefill"
            ok = e < (4e-3 if path == "decode" else 1.2e-2) and mx.sum(r ** 2).item() > 0
            print(f"  {'PASS' if ok else 'FAIL'} {rot:10s} g{bgu} u{bu or bgu} d{bd} T{T:2d} {path:7s} {name:8s} rel {e:.1e}", flush=True)
            if not ok: FAILS.append((bgu, bd, T, name))
print("ALL PASS" if not FAILS else f"FAILURES {FAILS}")
sys.exit(1 if FAILS else 0)
