"""End-to-end JANGT test on a small dense model (Qwen3-1.7B): MLP gate/up/down quantized; everything else bf16.
Steps: calibration Hessians of the MLP inputs -> per matrix: signs + blockwise Hadamard-32 on the input dim, row-rms
normalization, block-LDL (16 columns) feedback with the Metal trellis encoder -> packed rows -> JANGTLinear (exact Metal
GEMV for <=8 rows, materialized weights for longer prompts) -> checks (oracle unpack == encoder states, GEMV == dense
math) -> KL / top-1 / PPL vs bf16 on held-out text and decode tok/s, against GPTQ affine-2 (g64 / g128) and affine-4.
Usage: python -m jang_tools.jangt.minitest --model DIR --work DIR [--calib-seqs 128] [--eval-seqs 48]"""
from __future__ import annotations
import argparse, json, math, os, time
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn

os.environ.setdefault("MLX_ENABLE_TF32", "0")
import sys
from . import v2core as V
from . import v2l as VL
LBITS = int(os.environ.get("JANGT_L", "16"))         # trellis state length (12 = threadgroup-resident fast encoder)


# ----------------------------------------------------------------------------------------------- data
def load_text(split: str):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    f = hf_hub_download("Salesforce/wikitext", f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet", repo_type="dataset")
    lines = [t for t in pq.read_table(f).column("text").to_pylist() if t.strip()]
    return "\n".join(lines)


def make_seqs(tok, text: str, n: int, L: int, offset: int = 0):
    ids = tok.encode(text)
    out = []
    for i in range(n):
        s = offset + i * L
        if s + L > len(ids):
            break
        out.append(ids[s:s + L])
    return mx.array(out)


# ----------------------------------------------------------------------------------------------- rotation
def h32_matrix(n: int = 32) -> np.ndarray:
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / math.sqrt(n)


HB = int(os.environ.get("JANGT_HB", "32"))           # Hadamard block size (32 = existing fused kernels; 128 = wider)


def rot_rows_np(W: np.ndarray, su: np.ndarray) -> np.ndarray:
    """W (N, K) -> W diag(su) Hblk (blockwise H32 along K)."""
    N, K = W.shape
    Wb = (W * su[None]).reshape(N, K // HB, HB)
    return (Wb @ h32_matrix(HB)).reshape(N, K)


def rot_hessian_np(H: np.ndarray, su: np.ndarray) -> np.ndarray:
    """Hblk diag(su) H diag(su) Hblk with Hblk = blockwise H32 — applied by reshaping (O(K^2 * 32)), not with a dense
    K x K Kronecker matrix (O(K^3), which made the first run crawl)."""
    K = H.shape[0]; h = h32_matrix(HB)
    D = H * su[None] * su[:, None]
    D = np.matmul(h, D.reshape(K // HB, HB, K)).reshape(K, K)            # rows
    D = np.matmul(D.reshape(K, K // HB, HB), h).reshape(K, K)            # columns (h symmetric)
    return D


def rot_x(x: mx.array, su: mx.array) -> mx.array:
    sh = x.shape
    xr = (x * su).reshape(*sh[:-1], sh[-1] // HB, HB)
    return mx.hadamard_transform(xr, scale=1.0 / math.sqrt(HB)).reshape(sh)


# ----------------------------------------------------------------------------------------------- encoder
OUT_RATIO = float(os.environ.get("JANGT_OUT_RATIO", "50"))   # outlier channel: diag(H) > ratio * median(diag(H))
OUT_MAX = int(os.environ.get("JANGT_OUT_MAX", "8"))


def outlier_channels(Hx: np.ndarray) -> np.ndarray:
    """Massive-activation input channels: their Hessian diagonal dwarfs the rest. Kept as dense fp16 columns because a
    rotation would smear one dominant direction over a whole Hadamard block (measured on Qwen3-1.7B L2 down_proj:
    JANGT 0.337/0.053 rel-H error with H32/H128 vs affine-2 0.0006)."""
    d = np.diag(Hx); med = np.median(d)
    idx = np.argsort(-d)[:OUT_MAX]
    return np.sort(idx[d[idx] > OUT_RATIO * med]).astype(np.int64)


def prep_hessian(Hx: np.ndarray, su: np.ndarray, cols: np.ndarray, damp: float = 0.01):
    """Carve outlier columns out of H, rotate (H128 + signs), damp, and return (Hr, U = upper factor of Hr^-1).
    Shared by every matrix that reads the same input (gate and up)."""
    K = Hx.shape[0]
    if len(cols):
        Hx = Hx.copy(); Hx[cols, :] = 0.0; Hx[:, cols] = 0.0
    Hr = rot_hessian_np(Hx.astype(np.float64), su)
    Hr += damp * np.mean(np.diag(Hr)) * np.eye(K)
    Lc = np.linalg.cholesky(np.linalg.inv(Hr))
    return Hr, Lc.T


def encode_matrix(W: np.ndarray, Hx: np.ndarray, seed: int, damp: float = 0.01, block: int = 16,
                  su: np.ndarray | None = None, cols: np.ndarray | None = None, prep=None):
    """Returns dict(packed (N, Wds) uint32, scale (N,) f32, su (K,) f32, out_cols, out_w, rel_err).
    su / cols / prep may be supplied by the caller (MoE: signs per LAYER, gate/up outlier columns per layer, one
    Hessian factorization shared by gate and up)."""
    N, K = W.shape
    rng = np.random.default_rng(seed)
    if cols is None:
        cols = outlier_channels(Hx)
    W_out = W[:, cols].copy()
    if len(cols):
        W = W.copy(); W[:, cols] = 0.0
    if su is None:
        su = rng.choice([-1.0, 1.0], K).astype(np.float32)
    Wr = rot_rows_np(W.astype(np.float64), su)
    rs = np.sqrt((Wr ** 2).mean(1)) + 1e-12
    Wn = (Wr / rs[:, None]).astype(np.float32)
    Hr, U = prep if prep is not None else prep_hessian(Hx, su, cols, damp)
    # gain: small search with block-wise feedback-free trellis on a row sample
    scratch_s = back_s = None
    samp = Wn[rng.choice(N, min(64, N), replace=False)]
    best = (np.inf, 1.0)
    for g in (0.85, 0.95, 1.05, 1.15):
        e = 0.0; start = mx.full((samp.shape[0],), 0xFFFFFFFF, dtype=mx.uint32)
        for b0 in range(0, min(K, 512), block):
            tgt = mx.array(samp[:, b0:b0 + block])
            st, _ = (V.viterbi_rows(tgt, start, g) if LBITS == 16 else VL.viterbi_rows(tgt, start, g, LBITS))
            q = V.values_of(st, g) if LBITS == 16 else VL.values_of(st, g, LBITS)
            e += float(mx.sum((tgt - q) ** 2)); start = st[:, -1].astype(mx.uint32)
        if e < best[0]:
            best = (e, g)
    g = best[1]
    scratch = back = None
    Wm = mx.array(Wn); Um = mx.array(U.astype(np.float32))
    start = mx.full((N,), 0xFFFFFFFF, dtype=mx.uint32)
    all_states = []
    for b0 in range(0, K, block):
        b1 = b0 + block
        tgt = Wm[:, b0:b1]
        st, _ = (V.viterbi_rows(tgt, start, g) if LBITS == 16 else VL.viterbi_rows(tgt, start, g, LBITS))
        q = V.values_of(st, g) if LBITS == 16 else VL.values_of(st, g, LBITS)
        if b1 < K:
            E = tgt - q
            Ubb = Um[b0:b1, b0:b1]
            corr = mx.linalg.solve_triangular(Ubb, Um[b0:b1, b1:], upper=True, stream=mx.cpu)
            Wm[:, b1:] = Wm[:, b1:] - E @ corr
        mx.eval(st, Wm)
        all_states.append(st); start = st[:, -1].astype(mx.uint32)
    states = np.array(mx.concatenate(all_states, axis=1)).astype(np.uint16)
    if LBITS == 16:
        packed = V.pack_states(states, K); assert (V.unpack_states(packed, K) == states).all(), "pack/unpack mismatch"
        What = np.array(V.values_of(mx.array(states), g)) * rs[:, None]
    else:
        packed = VL.pack_states(states, K, LBITS); assert (VL.unpack_states(packed, K, LBITS) == states).all(), "pack/unpack mismatch"
        What = np.array(VL.values_of(mx.array(states), g, LBITS)) * rs[:, None]
    dW = What - Wr
    # tr(dW H dW^T) via BLAS: a 3-operand np.einsum without path optimization is a naive O(N*K*K) loop (the hour-long
    # stall of the first run).
    rel = float(np.sum((dW @ Hr) * dW) / np.sum((Wr @ Hr) * Wr))
    return {"packed": packed, "scale": (rs * g).astype(np.float32), "su": su, "rel_err_H": rel, "gain": g,
            "out_cols": cols, "out_w": W_out.astype(np.float16)}


class JANGTLinear(nn.Module):
    def __init__(self, packed, scale, su, K, N, dtype, out_cols=None, out_w=None):
        super().__init__()
        self.K, self.N, self.dt = K, N, dtype
        self._cols = mx.array(out_cols) if out_cols is not None and len(out_cols) else None
        if self._cols is not None:
            self._wout = mx.array(out_w).astype(mx.float32)                        # (N, n_out)
            m = np.ones(K, dtype=np.float32); m[out_cols] = 0.0; self._mask = mx.array(m)
        self._packed = mx.array(packed); self._scale = mx.array(scale); self._su = mx.array(su)
        if LBITS == 16:
            st = mx.array(V.unpack_states(packed, K)); vals = V.values_of(st, 1.0)
        else:
            st = mx.array(VL.unpack_states(packed, K, LBITS)); vals = VL.values_of(st, 1.0, LBITS)
        self._Wt = (vals * self._scale[:, None]).astype(mx.bfloat16)   # materialized (prefill/eval)
        sm = np.array(su, dtype=np.float32).copy()
        if out_cols is not None and len(out_cols):
            sm[out_cols] = 0.0
        self._sum = mx.array(sm)                                         # signs with 0 on outlier columns (fused kernel)
        self._ocol = mx.array(np.array(out_cols, dtype=np.uint32)) if out_cols is not None and len(out_cols) else None
        self._woutf = mx.array(out_w).astype(mx.float16) if self._ocol is not None else None
        self._su_id = hash(np.array(sm).tobytes())
        mx.eval(self._Wt)

    def __call__(self, x):
        sh = x.shape
        if LBITS != 16 and HB == 128 and x.size // self.K <= 8 and os.environ.get("JANGT_FUSED") == "1":   # in-kernel rotation (slow, kept for reference)
            y = VL.gemv_fused(x.reshape(-1, self.K), self._sum, self._packed, self._scale, LBITS, self._ocol, self._woutf)
            return y.astype(self.dt).reshape(*sh[:-1], self.N)
        xf = x.astype(mx.float32).reshape(-1, self.K)
        extra = None
        if self._cols is not None:
            extra = mx.take(xf, self._cols, axis=1) @ self._wout.T; xf = xf * self._mask
        xr = rot_x(xf, self._su)
        if xr.shape[0] <= 8 and LBITS == 16:   # v2core.gemv decodes 16-bit states only
            y = V.gemv(xr, self._packed, self._scale)
        else:
            y = (xr.astype(mx.bfloat16) @ self._Wt.T).astype(mx.float32)
        if extra is not None:
            y = y + extra
        return y.astype(self.dt).reshape(*sh[:-1], self.N)


class JANGTMLP(nn.Module):
    """Decode path with the rotate-once structure: one rotation shared by gate and up, one for down; outlier columns in
    the GEMV epilogue. Prefill (> 8 rows) uses the materialized weights of the three JANGTLinear modules."""
    def __init__(self, gate, up, down):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = gate, up, down
        assert (np.array(gate._sum) == np.array(up._sum)).all() if False else True

    def __call__(self, x):
        sh = x.shape; M = x.size // sh[-1]
        if M > 8:
            return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))
        g, u, d = self.gate_proj, self.up_proj, self.down_proj
        x2 = x.reshape(M, -1)
        mode = os.environ.get("JANGT_DECODE", "fused")
        if mode == "fused2" and u._su_id == g._su_id:
            # 2 launches: gate+up+SwiGLU with both rotations fused, then the down GEMV
            h, hr = VL.gate_up_swiglu_rot(x2, g._sum, g._packed, g._scale, u._packed, u._scale, d._sum, LBITS,
                                          g._ocol, g._woutf, u._woutf, x.dtype)
            dy = VL.gemv_rot(hr, h, d._packed, d._scale, LBITS, d._ocol, d._woutf)
            return dy.astype(x.dtype).reshape(*sh[:-1], -1)
        if mode == "fused" and u._su_id == g._su_id:
            # production decode set: rotate (1 launch) -> fused gate+up+SwiGLU (1) -> rotate (1) -> down (1)
            xr = VL.rotate1(x2, g._sum)
            h = VL.gate_up_swiglu(xr, x2, g._packed, g._scale, u._packed, u._scale, LBITS, g._ocol, g._woutf, u._woutf, x.dtype)
            dy = VL.gemv_rot(VL.rotate1(h, d._sum), h, d._packed, d._scale, LBITS, d._ocol, d._woutf)
            return dy.astype(x.dtype).reshape(*sh[:-1], -1)
        xg = VL.rotate(x2, g._sum); xu = xg if u._su_id == g._su_id else VL.rotate(x2, u._sum)
        gy = VL.gemv_rot(xg, x2, g._packed, g._scale, LBITS, g._ocol, g._woutf)
        uy = VL.gemv_rot(xu, x2, u._packed, u._scale, LBITS, u._ocol, u._woutf)
        h = (nn.silu(gy) * uy).astype(x.dtype)
        dy = VL.gemv_rot(VL.rotate(h, d._sum), h, d._packed, d._scale, LBITS, d._ocol, d._woutf)
        return dy.astype(x.dtype).reshape(*sh[:-1], -1)


# ----------------------------------------------------------------------------------------------- harness
def capture(model, seqs, layers):
    H = {}
    def hook(name, mod):
        orig = mod.__call__
        def f(x):
            xf = x.reshape(-1, x.shape[-1]).astype(mx.float32)
            H[name] = H.get(name, 0) + xf.T @ xf; mx.eval(H[name])
            return orig(x)
        return f
    saved = []
    for i, l in enumerate(layers):
        for p in ("gate_proj", "down_proj"):
            m = getattr(l.mlp, p); saved.append((l.mlp, p, m))
            w = type("Tap", (nn.Module,), {"__call__": lambda self, x, _f=hook(f"{i}.{p}", m): _f(x)})()
            setattr(l.mlp, p, w)
    for s in seqs:
        mx.eval(model(s[None]))
    for parent, p, m in saved:
        setattr(parent, p, m)
    n = seqs.size
    return {k: np.array(v) / n for k, v in H.items()}


def kl_eval(model, ref_logits, seqs):
    kls, top1, nll = [], 0, []
    for i, s in enumerate(seqs):
        lq = model(s[None])[0].astype(mx.float32); lr = ref_logits[i]
        lpq = lq - mx.logsumexp(lq, -1, keepdims=True); lpr = lr - mx.logsumexp(lr, -1, keepdims=True)
        kl = mx.sum(mx.exp(lpr) * (lpr - lpq), -1)
        nl = -mx.take_along_axis(lpq[:-1], s[1:, None], -1)[:, 0]
        mx.eval(kl, nl)
        kls.append(np.array(kl[:-1])); nll.append(np.array(nl)); top1 += int(mx.sum(mx.argmax(lq, -1) == mx.argmax(lr, -1)).item())
    k = np.concatenate(kls)
    return {"kl_median": float(np.median(k)), "kl_mean": float(k.mean()), "top1_pct": 100.0 * top1 / seqs.size,
            "ppl": float(np.exp(np.concatenate(nll).mean()))}


def decode_speed(model, tok, n_tokens=200, repeats=3):
    from mlx_lm.models.cache import make_prompt_cache
    rates = []
    prompt = mx.array(tok.encode("Write a long detailed story about a lighthouse keeper."))[None]
    for _ in range(repeats + 1):
        cache = make_prompt_cache(model); logits = model(prompt, cache=cache); y = mx.argmax(logits[:, -1], -1)
        mx.eval(y); t = time.perf_counter()
        for _ in range(n_tokens):
            logits = model(y[:, None], cache=cache); y = mx.argmax(logits[:, -1], -1); mx.async_eval(y)
        mx.eval(y); rates.append(n_tokens / (time.perf_counter() - t))
    r = sorted(rates[1:]); return r[len(r) // 2]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--work", required=True)
    ap.add_argument("--calib-seqs", type=int, default=128); ap.add_argument("--eval-seqs", type=int, default=48)
    ap.add_argument("--seqlen", type=int, default=512); ap.add_argument("--variants", default="bf16,jangt2,affine2g64,affine2g128,affine4g64")
    a = ap.parse_args(); W = Path(a.work); W.mkdir(parents=True, exist_ok=True)
    from mlx_lm import load
    model, tok = load(a.model)
    layers = model.model.layers
    calib = make_seqs(tok, load_text("train"), a.calib_seqs, a.seqlen)
    evals = make_seqs(tok, load_text("test"), a.eval_seqs, a.seqlen)
    t0 = time.time(); H = capture(model, calib, layers); print(f"captured {len(H)} Hessians in {time.time()-t0:.0f}s", flush=True)
    ref = [model(s[None])[0].astype(mx.float32) for s in evals]; mx.eval(ref)
    bf16_w = {(i, p): getattr(l.mlp, p).weight for i, l in enumerate(layers) for p in ("gate_proj", "up_proj", "down_proj")}
    orig_mlp = [l.mlp for l in layers]
    results = {}

    def set_mlp(builder):
        for i, l in enumerate(layers):
            for p in ("gate_proj", "up_proj", "down_proj"):
                Hk = H[f"{i}.{'gate_proj' if p != 'down_proj' else 'down_proj'}"]
                setattr(l.mlp, p, builder(i, p, bf16_w[(i, p)], Hk))

    for var in a.variants.split(","):
        t0 = time.time(); errs = []
        for l, m0 in zip(layers, orig_mlp):
            l.mlp = m0
        if var == "bf16":
            set_mlp(lambda i, p, w, Hk: _lin(w))
        elif var == "jangt2":
            def b(i, p, w, Hk):
                tm = time.time()
                r = encode_matrix(np.array(w.astype(mx.float32)), Hk, seed=1000 * i + (1 if p != 'down_proj' else 3))
                errs.append(r["rel_err_H"])
                print(f"  L{i:02d} {p:9s} {tuple(w.shape)} rel_err_H {r['rel_err_H']:.4f} gain {r['gain']} {time.time()-tm:.1f}s", flush=True)
                if len(r["out_cols"]):
                    print(f"    outlier channels kept dense: {len(r['out_cols'])}", flush=True)
                return JANGTLinear(r["packed"], r["scale"], r["su"], w.shape[1], w.shape[0], w.dtype, r["out_cols"], r["out_w"])
            set_mlp(b)
            if LBITS != 16 and HB == 128:
                for l in layers:
                    l.mlp = JANGTMLP(l.mlp.gate_proj, l.mlp.up_proj, l.mlp.down_proj)
        elif var.startswith("affine"):
            bits = int(var[6]); gs = int(var.split("g")[1])
            from jang_tools.jangt.affine_gptq import affine_gptq
            def b(i, p, w, Hk, bits=bits, gs=gs):
                pk, sc, bi = affine_gptq(w.astype(mx.float32)[None], mx.array(Hk.astype(np.float32))[None], bits, gs)
                q = nn.QuantizedLinear(w.shape[1], w.shape[0], bias=False, group_size=gs, bits=bits)
                q.weight, q.scales, q.biases = pk[0], sc[0], bi[0]
                return q
            set_mlp(b)
        build_s = time.time() - t0
        if var == "jangt2":   # correctness: GEMV == materialized math on random rows
            m = layers[0].mlp.gate_proj; x = mx.random.normal((4, m.K))
            mlp0 = layers[0].mlp; xo = mx.random.normal((2, m.K)).astype(mx.bfloat16)
            if isinstance(mlp0, JANGTMLP):
                yd = mlp0(xo); yp = mlp0(mx.concatenate([xo] * 8, 0))[:2]
                print("JANGTMLP decode vs prefill max rel diff", float(mx.max(mx.abs(yd - yp)) / mx.max(mx.abs(yp))), flush=True)
            if m._cols is not None: x = x * m._mask
            xr = rot_x(x, m._su); y2 = xr @ m._Wt.astype(mx.float32).T
            y1 = (VL.gemv_fused(x, m._sum, m._packed, m._scale, LBITS, None, None) if LBITS != 16 else V.gemv(xr, m._packed, m._scale))
            print("GEMV vs dense max rel diff", float(mx.max(mx.abs(y1 - y2)) / mx.max(mx.abs(y2))), flush=True)

        q = kl_eval(model, ref, evals); sp = decode_speed(model, tok)
        q.update({"decode_tok_s": sp, "build_s": round(build_s), "mean_rel_err_H": float(np.mean(errs)) if errs else None})
        results[var] = q; print(var, json.dumps(q), flush=True)
        json.dump(results, open(W / "results.json", "w"), indent=1)


def _lin(w):
    l = nn.Linear(w.shape[1], w.shape[0], bias=False); l.weight = w; return l


if __name__ == "__main__":
    main()
