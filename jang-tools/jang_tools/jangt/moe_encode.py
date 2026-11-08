"""JANGT encoder for the routed experts of ONE MoE layer (first used on Naive-N0.5-Flash).

Per layer:
  signs        su_gu (D,), su_dn (I,) — per LAYER (seeded by the layer index), so one rotation of the layer input serves
               every expert at decode.
  outliers     gate/up: per LAYER, from the pooled UNcentered second moment (massive channels of these models are
               layer-wide constants); down: per EXPERT (hidden units are expert-specific). Kept as fp16 columns.
  Hessian      per expert = the JANGH recipe (pooled shrinkage + mu mu^T, jang_tools.jangh.n05.recipe.hessian), one factorization
               shared by gate and up.
  encode       minitest.encode_matrix (H128 + signs, per-row RMS x gain, block-LDL feedback, L-bit trellis K=2 V=2).
Output: <out>/L##.safetensors
  model.layers.L.mlp.switch_mlp.{proj}.jt_packed (E, N, words) u32, .jt_scale (E, N) f32, .jt_su (K,) f32, .jt_ocol (E, n_out) i32 (-1 = none),
  ....{proj}.jt_wout (E, N, n_out) f16
  plus <out>/L##.json : per expert rel-H error, gain, outlier columns, timings.
Run with MLX_ENABLE_TF32=0, JANGT_HB=128, JANGT_L=12.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import mlx.core as mx

from jang_tools.jangt import minitest as MT
from jang_tools.jangt import v2l as VL

E_DEFAULT = 256


def dequant(enc: dict, K: int) -> np.ndarray:
    """Packed trellis -> dense W in the ORIGINAL basis (outlier columns restored)."""
    kb = enc.get("kbits", 2)
    st = mx.array(VL.unpack_states(enc["packed"], K, MT.LBITS, kb))
    Wr = np.array(VL.values_of(st, 1.0, MT.LBITS, kb)).astype(np.float64) * enc["scale"][:, None]
    N = Wr.shape[0]; h = MT.h32_matrix(MT.HB)
    W = (Wr.reshape(N, K // MT.HB, MT.HB) @ h).reshape(N, K) * enc["su"][None]     # h symmetric orthonormal
    if len(enc["out_cols"]):
        W[:, enc["out_cols"]] = enc["out_w"].astype(np.float64)
    return W


def dequant_ragged(sf: dict, m: str, e0: int, e1: int, K: int) -> mx.array:
    """Experts e0..e1-1 of a ragged stage tensor set (jt_packed flat, jt_off, jt_kbits, jt_cs) -> (n, N, K) f32."""
    off = np.array(sf[m + ".jt_off"]); kb = np.array(sf[m + ".jt_kbits"]); flat = sf[m + ".jt_packed"]
    scale, su, ocol, wout = sf[m + ".jt_scale"], sf[m + ".jt_su"], sf[m + ".jt_ocol"], sf[m + ".jt_wout"]
    cs = sf.get(m + ".jt_cs"); N = scale.shape[1]; out = []
    for e in range(e0, e1):
        k = float(kb[e]); k = int(k) if k.is_integer() else k
        W = dequant_mx(flat[int(off[e]):int(off[e + 1])].reshape(N, -1)[None], scale[e:e + 1], su, ocol[e:e + 1], wout[e:e + 1], K, k)[0]
        out.append(W * cs[e][None, :].astype(mx.float32) if cs is not None else W)
    return mx.stack(out)


def dequant_mx(packed: mx.array, scale: mx.array, su: mx.array, ocol: mx.array, wout: mx.array, K: int, kbits=2) -> mx.array:
    """(E, N, words) packed experts -> (E, N, K) f32 in the ORIGINAL basis on the GPU (outlier columns restored;
    ocol = -1 marks padding)."""
    E_, N = packed.shape[:2]; L = MT.LBITS; out = []
    oc_np = np.array(ocol)
    for e in range(E_):
        st = mx.array(VL.unpack_states(np.array(packed[e]), K, L, kbits))
        Wr = VL.values_of(st, 1.0, L, kbits) * scale[e][:, None]
        W = mx.hadamard_transform(Wr.reshape(N, K // MT.HB, MT.HB), scale=MT.HB ** -0.5).reshape(N, K) * su[None]
        c = oc_np[e][oc_np[e] >= 0]
        if len(c):
            W[:, mx.array(c)] = wout[e][:, :len(c)].astype(mx.float32)
        out.append(W)
    return mx.stack(out)


def col_rms_scales(W: np.ndarray, cols) -> np.ndarray:
    """Per-input-channel scales c = column RMS of W, geomean-normalized; outlier columns fixed at 1. The encoder
    quantizes W / c against H' = diag(c) H diag(c); at decode x is multiplied by c inside the rotation (free)."""
    c = np.sqrt((W.astype(np.float64) ** 2).mean(0)) + 1e-12
    c = c / np.exp(np.mean(np.log(c)))
    c[np.asarray(cols, dtype=np.int64)] = 1.0
    return c


def col_refit(W: np.ndarray, A: np.ndarray, H: np.ndarray, cols) -> np.ndarray:
    """Least-squares per-column scale s minimizing tr((W - A diag(s)) H (W - A diag(s))^T) (outlier columns fixed at
    1): (H o A^T A) s = diag(H W^T A). Measured on N0.5 L33/L46 down: 1.075 -> 1.045 rel-H vs JANGH-2 (1.007 together
    with col_rms_scales)."""
    M = H * (A.T @ A); b = np.einsum("jk,kj->j", H, W.T @ A)
    keep = np.ones(W.shape[1], bool); keep[np.asarray(cols, dtype=np.int64)] = False
    s = np.ones(W.shape[1]); Mk = M[np.ix_(keep, keep)]
    Mk = Mk + 1e-6 * np.mean(np.diag(Mk)) * np.eye(Mk.shape[0])
    s[keep] = np.linalg.solve(Mk, b[keep] - M[np.ix_(keep, ~keep)] @ s[~keep])
    return s


def unit_gain_rows(W: np.ndarray, What: np.ndarray, H: np.ndarray) -> np.ndarray:
    WH = W @ H
    return np.sum(WH * W, 1) / np.maximum(np.sum(WH * What, 1), 1e-30)


def layer_outlier_cols(mean_e: np.ndarray, wsum: np.ndarray, pooled_cov: np.ndarray) -> np.ndarray:
    """Pooled UNcentered second moment diag = diag(S_pool) + weighted mean of mu_e^2."""
    w = wsum / max(wsum.sum(), 1e-300)
    d = np.diag(pooled_cov) + (w[:, None] * mean_e ** 2).sum(0)
    return MT.outlier_channels(np.diag(d))


# ------------------------------------------------------------------------------------------- batched encoder
GAINS = (0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45)   # K=2.5/3 optimum sits at 1.2-1.3 (Gaussian test)


def ldl_feedback(U: np.ndarray, block: int = 16) -> np.ndarray:
    """T with T[b0:b1, b1:] = U_bb^-1 U[b0:b1, b1:] for every 16-column block (float64 CPU, once per Hessian), so the
    block-LDL update of the GPU loop is one batched matmul: W[:, b1:] -= E_block @ T[b0:b1, b1:]."""
    from scipy.linalg import solve_triangular
    K = U.shape[0]; T = np.zeros((K, K), np.float32)
    for b0 in range(0, K - block, block):
        b1 = b0 + block
        T[b0:b1, b1:] = solve_triangular(U[b0:b1, b0:b1], U[b0:b1, b1:], lower=False)
    return T


def encode_batch(Ws, sus, colss, preps, Ts, block: int = 16, gains=GAINS, refit: bool = True, kbits=2):
    """Encode B matrices of one shape at once (the trellis kernel runs over B*N rows; the gain is folded into the
    target, Viterbi at gain 1, so every row may carry its own gain). Same result as minitest.encode_matrix with the
    same gain. Returns a list of dicts like encode_matrix."""
    B = len(Ws); N, K = Ws[0].shape; L = MT.LBITS
    Wn, rss, outs = [], [], []
    for W, sgn, cols in zip(Ws, sus, colss):
        W_out = W[:, cols].copy()
        if len(cols):
            W = W.copy(); W[:, cols] = 0.0
        Wr = MT.rot_rows_np(W.astype(np.float64), sgn)
        rs = np.sqrt((Wr ** 2).mean(1)) + 1e-12
        Wn.append((Wr / rs[:, None]).astype(np.float32)); rss.append(rs); outs.append((W_out, Wr))
    Wm = mx.array(np.stack(Wn))                                              # (B, N, K)
    # gain: feedback-free trellis error on 64 sampled rows x the first 512 columns, every candidate, all matrices
    rng = np.random.default_rng(0); ridx = mx.array(rng.choice(N, min(64, N), replace=False).astype(np.int32))
    samp = Wm[:, ridx, :min(K, 512)].reshape(-1, min(K, 512)); errs = []
    for g in gains:
        start = mx.full((samp.shape[0],), 0xFFFFFFFF, dtype=mx.uint32); e = mx.zeros((samp.shape[0],))
        for b0 in range(0, samp.shape[1], block):
            t = samp[:, b0:b0 + block] / g
            st, _ = VL.viterbi_rows(t, start, 1.0, L, kbits); q = VL.values_of(st, g, L, kbits)
            e = e + mx.sum((samp[:, b0:b0 + block] - q) ** 2, axis=1); start = st[:, -1].astype(mx.uint32)
        errs.append(e.reshape(B, -1).sum(1))
    gsel = np.array(gains, np.float32)[np.argmin(np.array(mx.stack(errs)), axis=0)]          # (B,)
    grow = mx.array(np.repeat(gsel, N))[:, None]                                            # (B*N, 1)
    Tm = mx.array(np.stack(Ts))                                                             # (B, K, K)
    start = mx.full((B * N,), 0xFFFFFFFF, dtype=mx.uint32); all_states = []
    for b0 in range(0, K, block):
        b1 = b0 + block
        tgt = Wm[:, :, b0:b1].reshape(B * N, block)
        st, _ = VL.viterbi_rows(tgt / grow, start, 1.0, L, kbits)
        q = VL.values_of(st, 1.0, L, kbits) * grow
        if b1 < K:
            Eb = (tgt - q).reshape(B, N, block)
            Wm[:, :, b1:] = Wm[:, :, b1:] - Eb @ Tm[:, b0:b1, b1:]
        mx.eval(st, Wm)
        all_states.append(st); start = st[:, -1].astype(mx.uint32)
    states = np.array(mx.concatenate(all_states, axis=1)).astype(np.uint16).reshape(B, N, K // 2)
    res = []
    for b in range(B):
        packed = VL.pack_states(states[b], K, L, kbits)
        What = np.array(VL.values_of(mx.array(states[b]), float(gsel[b]), L, kbits)).astype(np.float64) * rss[b][:, None]
        Hr = preps[b][0]; Wr = outs[b][1]
        Hm = mx.array(Hr.astype(np.float32)); Wf = mx.array(Wr.astype(np.float32)); Qf = mx.array(What.astype(np.float32))
        WH = Wf @ Hm; ww = mx.sum(WH * Wf, axis=1); wq = mx.sum(WH * Qf, axis=1)
        # UNIT-GAIN row refit (not the MSE-optimal <w,q>/<q,q>, which shrinks rows — the GLM-5.3 </think> trap):
        # measured on N0.5 L1 within 1 % of the MSE refit, -6 % (up) / -4 % (down) rel-H vs no refit
        f = np.array(mx.where(wq > 0, ww / mx.maximum(wq, 1e-30), 1.0)).astype(np.float64) if refit else np.ones(N)
        What = What * f[:, None]
        dW = mx.array((What - Wr).astype(np.float32)); num = mx.sum((dW @ Hm) * dW); den = mx.sum(WH * Wf)
        res.append({"packed": packed, "scale": (rss[b] * gsel[b] * f).astype(np.float32), "su": sus[b], "rel_err_H": float(num / den),
                    "gain": float(gsel[b]), "out_cols": colss[b], "out_w": outs[b][0].astype(np.float16), "kbits": kbits})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--headers", required=True)
    ap.add_argument("--hess-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--m-min-gu", type=float, default=0.5); ap.add_argument("--m-min-dn", type=float, default=0.5)
    ap.add_argument("--experts", default="all", help="debug subset, e.g. 0-15")
    ap.add_argument("--block-gu", type=int, default=32); ap.add_argument("--block-dn", type=int, default=16)
    ap.add_argument("--kbits-gu", type=float, default=2); ap.add_argument("--kbits-dn", type=float, default=2)
    ap.add_argument("--expert-k", default=None, help='json {"L": {"gu": [K_e x E], "dn": [K_e x E]}} per-expert widths')
    ap.add_argument("--no-colnorm-dn", action="store_true", help="disable column-RMS scaling + LS column refit on down")
    ap.add_argument("--plan", default=None, help="JANGH plan json: take m_min per layer from its recipe_m_min (same as the reference)")
    a = ap.parse_args()
    assert os.environ.get("MLX_ENABLE_TF32") == "0", "run with MLX_ENABLE_TF32=0"
    assert MT.HB == 128 and MT.LBITS == 12, "JANGT v1: JANGT_HB=128 JANGT_L=12"
    mx.set_cache_limit(2 * 2 ** 30)   # MLX keeps freed buffers; per-width batches of varying size bloat it
    from jang_tools.jangh.n05.hess_cpu import HessStore
    from jang_tools.jangh.n05.recipe import hessian, TAU

    src, out = Path(a.model), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    H = json.loads(Path(a.headers).read_text())
    cache = {}

    def get(name):
        f = H[name][0]
        if f not in cache:
            cache.clear(); mx.clear_cache(); cache[f] = mx.load(str(src / f))
        return cache[f][name]

    import struct as _struct
    _hdrs = {}

    def get_np(name):
        """Thread-safe bf16 -> f32 read through a private memmap (worker threads must not touch the MLX cache)."""
        f = H[name][0]; path = str(src / f)
        if f not in _hdrs:
            with open(path, "rb") as fh:
                n = _struct.unpack("<Q", fh.read(8))[0]; _hdrs[f] = (n, json.loads(fh.read(n)))
        n, hdr = _hdrs[f]; v = hdr[name]; assert v["dtype"] == "BF16", v["dtype"]
        a0, a1 = v["data_offsets"]
        u = np.memmap(path, dtype=np.uint16, mode="r", offset=8 + n + a0, shape=tuple(v["shape"]))
        return (np.asarray(u, dtype=np.uint32) << 16).view(np.float32)

    for L in [int(x) for x in a.layers.split(",")]:
        dst = out / f"L{L:02d}.safetensors"
        if dst.exists():
            print(f"L{L}: exists, skip", flush=True); continue
        t0 = time.time()
        if a.plan:
            mm = json.loads(Path(a.plan).read_text())["recipe_m_min"][str(L)]
            a.m_min_gu, a.m_min_dn = float(mm["gu"]), float(mm["dn"])
        hf = str(Path(a.hess_dir) / f"L{L:02d}.safetensors")
        HS = {w: HessStore(hf, w) for w in ("gu", "dn")}
        pooled = {w: HS[w].pooled() for w in HS}
        E = HS["gu"].E
        ex = range(E) if a.experts == "all" else range(int(a.experts.split("-")[0]), int(a.experts.split("-")[1]) + 1)
        rng = np.random.default_rng(10_000 + L)
        su = {"gu": rng.choice([-1.0, 1.0], HS["gu"].n).astype(np.float32), "dn": rng.choice([-1.0, 1.0], HS["dn"].n).astype(np.float32)}
        cols_gu = layer_outlier_cols(HS["gu"].mean, HS["gu"].wsum, pooled["gu"])
        print(f"L{L}: E={E} gate/up outlier columns (layer) {cols_gu.tolist()}  pooled ready {time.time()-t0:.0f}s", flush=True)
        base = f"model.layers.{L}.mlp.experts"
        res = {p: {"packed": [], "scale": [], "ocol": [], "wout": [], "k": [], "cs": []} for p in ("gate", "up", "down")}
        EK = json.loads(Path(a.expert_k).read_text()).get(str(L)) if a.expert_k else None
        kofs = lambda g, e: (EK[g][e] if EK else (a.kbits_gu if g == "gu" else a.kbits_dn))
        kn = lambda k: int(k) if float(k).is_integer() else float(k)
        rep = {"layer": L, "expert_k": EK, "colnorm_dn": not a.no_colnorm_dn, "kbits": {"gu": a.kbits_gu, "dn": a.kbits_dn}, "m_min": {"gu": a.m_min_gu, "dn": a.m_min_dn}, "outlier_cols_gu": cols_gu.tolist(), "block_gu": a.block_gu, "block_dn": a.block_dn,
               "row_refit": "unit-gain", "experts": {}}

        do_gu = not (EK is not None and EK.get("gu") is None)            # null group in --expert-k = JANGH elsewhere
        do_dn = not (EK is not None and EK.get("dn") is None)

        def hess_job(e):
            pg = Tg = pd = Td = Hd = cdn = None; cd = np.zeros(0, np.int64)
            if do_gu:
                Hg = hessian(HS["gu"].cov(e), HS["gu"].mean[e], HS["gu"].rows[e], pooled["gu"], a.m_min_gu, TAU["gu"])
                pg = MT.prep_hessian(Hg, su["gu"], cols_gu); Tg = ldl_feedback(pg[1], a.block_gu)
            if do_dn:
                Hd = hessian(HS["dn"].cov(e), HS["dn"].mean[e], HS["dn"].rows[e], pooled["dn"], a.m_min_dn, TAU["dn"])
                cd = MT.outlier_channels(Hd)
                cdn = np.ones(Hd.shape[0]) if a.no_colnorm_dn else col_rms_scales(np.array(get_np(f"{base}.{e}.down_proj.weight")), cd)
                pd = MT.prep_hessian(Hd * np.outer(cdn, cdn), su["dn"], cd); Td = ldl_feedback(pd[1], a.block_dn)
            return e, pg, Tg, cd, pd, Td, Hd, cdn

        exl = list(ex); BE = 8
        with ThreadPoolExecutor(max_workers=4) as pool:                  # CPU float64 factorizations ahead of the GPU
            # sliding window: ~420 MB of float64 factors per expert, so never more than 2 batches in flight
            # (a run that kept dense f32 copies of every expert + 3 batches in flight thrashed the Mac into a watchdog panic)
            futs = [None] * len(exl); nxt = 0
            for c0 in range(0, len(exl), BE):
                while nxt < min(len(exl), c0 + 2 * BE):
                    futs[nxt] = pool.submit(hess_job, exl[nxt]); nxt += 1
                te = time.time(); chunk = [futs[k].result() for k in range(c0, min(len(exl), c0 + BE))]
                PR = (("gate", "up") if do_gu else ()) + (("down",) if do_dn else ())
                Wl = {p: [np.array(get(f"{base}.{c[0]}.{p}_proj.weight").astype(mx.float32)) for c in chunk] for p in PR}
                n = len(chunk)
                encs = {"gate": [None] * n, "up": [None] * n, "down": [None] * n}
                for k in (sorted({kofs("gu", c[0]) for c in chunk}) if do_gu else []):   # one batch per width
                    ix = [i_ for i_, c in enumerate(chunk) if kofs("gu", c[0]) == k]
                    gu = encode_batch([Wl["gate"][i_] for i_ in ix] + [Wl["up"][i_] for i_ in ix], [su["gu"]] * 2 * len(ix),
                                      [cols_gu] * 2 * len(ix), [chunk[i_][1] for i_ in ix] * 2, [chunk[i_][2] for i_ in ix] * 2,
                                      block=a.block_gu, kbits=kn(k))
                    for q_, i_ in enumerate(ix):
                        encs["gate"][i_], encs["up"][i_] = gu[q_], gu[len(ix) + q_]
                for k in (sorted({kofs("dn", c[0]) for c in chunk}) if do_dn else []):
                    ix = [i_ for i_, c in enumerate(chunk) if kofs("dn", c[0]) == k]
                    dn = encode_batch([Wl["down"][i_] / chunk[i_][7][None, :] for i_ in ix], [su["dn"]] * len(ix), [chunk[i_][3] for i_ in ix],
                                      [chunk[i_][4] for i_ in ix], [chunk[i_][5] for i_ in ix], block=a.block_dn, kbits=kn(k))
                    for q_, i_ in enumerate(ix):
                        enc = dn[q_]; c = chunk[i_]; cs = c[7].copy()
                        if not a.no_colnorm_dn:                                       # LS column refit + unit-gain rows
                            W0 = Wl["down"][i_].astype(np.float64); A0 = dequant(enc, W0.shape[1]) * cs[None, :]
                            s_ = col_refit(W0, A0, c[6], c[3]); cs = cs * s_
                            f_ = unit_gain_rows(W0, A0 * s_[None, :], c[6]); enc["scale"] = (enc["scale"] * f_).astype(np.float32)
                        enc["cs"] = cs.astype(np.float32); encs["down"][i_] = enc
                for k_, c in enumerate(chunk):
                    e = c[0]; r = {}
                    for p in PR:
                        enc = encs[p][k_]
                        res[p]["packed"].append(enc["packed"]); res[p]["scale"].append(enc["scale"]); res[p]["k"].append(enc["kbits"])
                        res[p]["cs"].append(enc.get("cs", np.ones(Wl[p][k_].shape[1], np.float32)))
                        res[p]["ocol"].append(np.asarray(enc["out_cols"], np.int32)); res[p]["wout"].append(enc["out_w"])
                        r[p] = {"rel_err_H": enc["rel_err_H"], "gain": enc["gain"], "n_out": int(len(enc["out_cols"])), "kbits": enc["kbits"]}
                    r["dn_outlier_cols"] = c[3].tolist(); rep["experts"][str(e)] = r
                    futs[c0 + k_] = None
                print(f"L{L} experts {chunk[0][0]}-{chunk[-1][0]}: " + " ".join(f"{p} {np.median([encs[p][k]['rel_err_H'] for k in range(n)]):.5f}" for p in PR)
                      + f"  {(time.time()-te)/n:.2f}s/expert  elapsed {(time.time()-t0)/60:.1f} min", flush=True)
        tens = {}
        for p in [p for p in res if res[p]["packed"]]:
            m = f"model.layers.{L}.mlp.switch_mlp.{p}_proj"
            words = [x.size for x in res[p]["packed"]]
            tens[m + ".jt_packed"] = mx.array(np.concatenate([x.ravel() for x in res[p]["packed"]]))
            tens[m + ".jt_off"] = mx.array(np.concatenate([[0], np.cumsum(words)]).astype(np.int64))
            tens[m + ".jt_kbits"] = mx.array(np.array(res[p]["k"], np.float32))
            if p == "down":
                tens[m + ".jt_cs"] = mx.array(np.stack(res[p]["cs"]).astype(np.float32))
            tens[m + ".jt_scale"] = mx.array(np.stack(res[p]["scale"]))
            tens[m + ".jt_su"] = mx.array(su["dn" if p == "down" else "gu"])
            nmax = max(1, max(len(c) for c in res[p]["ocol"]))
            oc = np.full((len(res[p]["ocol"]), nmax), -1, np.int32); wo = np.zeros((len(res[p]["ocol"]), res[p]["scale"][0].shape[0], nmax), np.float16)
            for i, (c, w) in enumerate(zip(res[p]["ocol"], res[p]["wout"])):
                oc[i, :len(c)] = c; wo[i, :, :len(c)] = w
            tens[m + ".jt_ocol"] = mx.array(oc); tens[m + ".jt_wout"] = mx.array(wo)
        mx.eval(list(tens.values()))
        tmp = out / f".L{L:02d}.tmp.safetensors"; mx.save_safetensors(str(tmp), tens); tmp.replace(dst)
        errs = {p: float(np.median([v[p]["rel_err_H"] for v in rep["experts"].values()])) for p in ("gate", "up", "down")
                if res[p]["packed"]}
        rep["median_rel_err_H"] = errs; rep["minutes"] = (time.time() - t0) / 60
        (out / f"L{L:02d}.json").write_text(json.dumps(rep, indent=1))
        print(f"L{L} DONE {rep['minutes']:.1f} min  median rel-H err {errs}", flush=True)
        del res, tens; mx.clear_cache()


if __name__ == "__main__":
    main()
