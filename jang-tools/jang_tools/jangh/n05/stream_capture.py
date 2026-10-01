"""Layer-streaming BF16 forward of the FULL Naive-N0.5-Flash source on a 128 GB Mac — capture v2.

The 575 GiB source never fits; its 48 layers are loaded one at a time and every sequence's hidden state is pushed
through that layer before the next is loaded. Uses the parity-tested n05.model (DSA top-2048 path active for T>2048).

Input: corpus jsonl from build_corpus.py ({"ids", "ref", "set", "domain", "decisions"}).
Sequence roles
  held-out reference (ref=true: klref / agentic_ref) : produce BF16 top-128 reference logprobs; NEVER enter statistics
  calibration, every 10th (cal_te)                     : rows go ONLY to the held-out row reservoir (te_rows)
  calibration, the rest (cal_tr)                       : diag stats, train reservoir, FULL per-expert Hessians
Outputs
  --stats-out : per MoE layer L
      model.layers.L.mlp.experts.{sum_x2 (D), count (1), expert_sum_x2 (E,D), expert_count (E),
                                  rows (R,D) f16, rows_topk_idx (R,k), rows_topk_w (R,k),
                                  te_rows (Rte,D) f16, te_rows_topk_idx, te_rows_topk_w}
      model.layers.L.mlp.switch_mlp.down_proj.{expert_sum_a2 (E,I), expert_count (E)}
  --hess-dir/L{L:02d}.safetensors : per-expert MEAN + CENTERED covariance of the projection inputs, router-weight^2
      weighted (the true contribution of an expert's error to the residual stream), normalized by sum w^2.
      Why centered: 96-99% of the MoE-input energy of this model is a CONSTANT vector (massive channels are
      constants: e.g. mean 19.2, std 0.95). Its quantization error is removed exactly by a per-row bias
      (b = (W - W_hat) mu); the quantizer objective is then the centered covariance, whose diagonal spans ~1e3
      instead of ~1e5-1e6 (fp32-safe). Uncentered second moment = cov + mu mu^T.
        gu_mean (E,D) f32, dn_mean (E,I) f32
        gu_diag (E,D) f32, gu_tri (E, D(D+1)/2) i16   gate/up input   H = sqrt(d_i d_j) * C, C = tri/32767 upper-tri
        dn_diag (E,I) f32, dn_tri (E, I(I+1)/2) i16   down input (silu(g)*u of the BF16 expert)
        (16-bit FIXED POINT, not fp16: uniform 1.5e-5 resolution -> ~1e-3 eigenvalue noise in correlation space;
         fp16's relative rounding gave ~1e-2, the same size as the 1% GPTQ damping)
        {gu,dn}_top_idx (E,32) i32 + {gu,dn}_top_rows (E,32,n) f32   EXACT rows of the 32 largest-diagonal channels
        wsum (E,) f32, rows (E,) f32
      Stored as diagonal + 16-bit CORRELATION because the inputs have massive channels (diag range > 1e5).
  --ref-out   : per held-out sequence j: p{j}.input_ids, p{j}.top_ids, p{j}.top_logprobs (top-128)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


from jang_tools.jangh.n05.model import Args, DecoderLayer, route, sanitize_layer  # noqa: E402
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort  # noqa: E402

POOL_TOKENS = 40_000
MIN_PROVISIONAL = 64      # rows an expert must have before its provisional centering vector is fixed


def load_layer_raw(src: Path, H: dict, i: int, cache: dict) -> dict:
    pre = f"model.layers.{i}."
    files = sorted({v[0] for k, v in H.items() if k.startswith(pre)})
    raw = {}
    for f in files:
        if f not in cache:
            if len(cache) > 2:
                cache.pop(next(iter(cache)))
            cache[f] = mx.load(str(src / f))
        raw.update({k: v for k, v in cache[f].items() if k.startswith(pre)})
    return raw


def tri_index(n: int) -> mx.array:
    r, c = np.triu_indices(n)
    return mx.array((r.astype(np.int64) * n + c).astype(np.int32))


NTOP = 32   # rows/cols of the largest-diagonal channels are stored EXACTLY (fp32)


def unpack_hess(diag: mx.array, tri: mx.array, iu: mx.array, top_idx: mx.array | None = None,
                top_rows: mx.array | None = None) -> mx.array:
    """(n,) f32 diag + (n(n+1)/2,) f16 upper-tri correlation (+ exact fp32 rows of the NTOP largest-diagonal
    channels) -> (n,n) f32 Hessian. The exact rows matter: fp16 rounding of a cross term between a MASSIVE channel
    m and a normal channel j is ~5e-4*sqrt(d_m d_j), i.e. up to ~15% of d_j, enough to make H indefinite."""
    n = diag.shape[0]
    tf = tri.astype(mx.float32) * (1.0 / 32767.0) if tri.dtype == mx.int16 else tri.astype(mx.float32)
    C = mx.zeros((n * n,), mx.float32).at[iu].add(tf).reshape(n, n)
    C = C + C.T - mx.diag(mx.diag(C))
    s = mx.sqrt(mx.maximum(diag.astype(mx.float32), 0.0))
    Hm = C * s[:, None] * s[None, :]
    if top_idx is not None:
        keep = mx.ones((n,), mx.float32).at[top_idx].add(-1.0)                 # 0 on the exact channels
        Hm = Hm * keep[:, None] * keep[None, :]
        R = top_rows.astype(mx.float32)                                        # (NTOP, n) exact rows
        Rk = R * keep[None, :]                                                 # cross terms with non-top columns
        Z = mx.zeros((n, n), mx.float32).at[top_idx].add(Rk)
        Hm = Hm + Z + Z.T
        Z2 = mx.zeros((n, n), mx.float32).at[top_idx].add(R * (1.0 - keep)[None, :])   # top x top block (once)
        Hm = Hm + Z2
    return Hm


def load_hess(hs: dict, which: str, e: int, iu: mx.array) -> mx.array:
    """CENTERED covariance of expert e's projection input ("gu" or "dn"). Mean: hs[f"{which}_mean"][e]."""
    return unpack_hess(hs[f"{which}_diag"][e], hs[f"{which}_tri"][e], iu, hs[f"{which}_top_idx"][e], hs[f"{which}_top_rows"][e])


class Reservoir:
    def __init__(self, R, D, k, rng):
        self.R, self.rng = R, rng
        self.keys = np.zeros((0,), np.float64); self.rows = np.zeros((0, D), np.float16)
        self.tidx = np.zeros((0, k), np.int32); self.tw = np.zeros((0, k), np.float32)

    def add(self, x, idx, w):
        N = x.shape[0]
        keys = self.rng.random(N)
        allk = np.concatenate([self.keys, keys])
        keep = np.argsort(allk)[: self.R]
        old, new = keep[keep < len(self.keys)], keep[keep >= len(self.keys)] - len(self.keys)
        if len(new) == 0:
            return
        sel = mx.array(np.sort(new).astype(np.int32)); new = np.sort(new)
        xn = np.asarray(x[sel].astype(mx.float16))
        self.keys = np.concatenate([self.keys[old], keys[new]])
        self.rows = np.concatenate([self.rows[old], xn]); self.tidx = np.concatenate([self.tidx[old], idx[new]])
        self.tw = np.concatenate([self.tw[old], w[new]])


class Stats:
    def __init__(self, E, D, I, k, R, Rte, seed, hess):
        self.E, self.D, self.I, self.k, self.hess = E, D, I, k, hess
        self.sum_x2 = mx.zeros((D,), mx.float32); self.count = 0
        self.e_x2 = mx.zeros((E, D), mx.float32); self.e_cnt = mx.zeros((E,), mx.float32)
        self.e_a2 = mx.zeros((E, I), mx.float32)
        rng = np.random.default_rng(seed)
        self.tr, self.te = Reservoir(R, D, k, rng), Reservoir(Rte, D, k, rng)
        if hess:
            self.H = [mx.zeros((D, D), mx.float32) for _ in range(E)]
            self.Hd = [mx.zeros((I, I), mx.float32) for _ in range(E)]
            self.m1 = [mx.zeros((D,), mx.float32) for _ in range(E)]       # sum w^2 x   (first moments)
            self.m1d = [mx.zeros((I,), mx.float32) for _ in range(E)]
            self.wsum = np.zeros(E, np.float64); self.nrows = np.zeros(E, np.float64)
            self.pool, self.pool_tokens = [], 0
            self.mu0 = [None] * E; self.mu0d = [None] * E; self.carry = [None] * E

    def add(self, x, idx, w, act, heldout):
        """x (T,D) bf16, idx (T,k), w (T,k) f32, act (T,k,I) bf16."""
        idn = np.asarray(idx).astype(np.int32); wn = np.asarray(w.astype(mx.float32))
        if heldout:
            self.te.add(x, idn, wn)
            return
        T, k = idn.shape
        xf = x.astype(mx.float32); x2 = xf * xf
        self.sum_x2 = self.sum_x2 + mx.sum(x2, axis=0); self.count += T
        flat = idx.reshape(-1)
        self.e_x2 = self.e_x2.at[flat].add(mx.repeat(x2, k, axis=0))
        self.e_cnt = self.e_cnt.at[flat].add(mx.ones(flat.shape, mx.float32))
        a = act.reshape(T * k, -1).astype(mx.float32)
        self.e_a2 = self.e_a2.at[flat].add(a * a)
        mx.eval(self.sum_x2, self.e_x2, self.e_cnt, self.e_a2)
        self.tr.add(x, idn, wn)
        if self.hess:
            ab = act.reshape(T * k, -1).astype(mx.bfloat16); mx.eval(ab)
            self.pool.append((x, idn, wn, ab)); self.pool_tokens += T
            if self.pool_tokens >= POOL_TOKENS:
                self.flush()

    def flush(self, final: bool = False):
        if not self.hess or (not self.pool and not final):
            return
        if not self.pool:
            self.pool = [(mx.zeros((0, self.D), mx.bfloat16), np.zeros((0, self.k), np.int32), np.zeros((0, self.k), np.float32), mx.zeros((0, self.I), mx.bfloat16))]
        k = self.k
        X = mx.concatenate([p[0] for p in self.pool], axis=0)
        A = mx.concatenate([p[3] for p in self.pool], axis=0)
        idx = np.concatenate([p[1] for p in self.pool]); w = np.concatenate([p[2] for p in self.pool])
        flat = idx.ravel()
        order = np.argsort(flat, kind="stable")
        bounds = np.searchsorted(flat[order], np.arange(self.E + 1))
        ws = w.ravel()[order].astype(np.float32)
        pend = []
        for e in range(self.E):
            b0, b1 = int(bounds[e]), int(bounds[e + 1])
            if b1 == b0 and not (final and self.carry[e] is not None):
                continue
            o = order[b0:b1]
            ww = mx.array(ws[b0:b1])[:, None]
            Xe = X[mx.array((o // k).astype(np.int32))].astype(mx.float32)
            Ae = A[mx.array(o.astype(np.int32))].astype(mx.float32)
            if self.carry[e] is not None:
                cx, ca, cw = self.carry[e]
                Xe, Ae, ww = mx.concatenate([cx, Xe]), mx.concatenate([ca, Ae]), mx.concatenate([cw, ww])
                self.carry[e] = None
            if self.mu0[e] is None:
                if Xe.shape[0] < MIN_PROVISIONAL and not final:
                    # too few rows to set this expert's provisional mean: a mean taken from 1-2 rows can be an outlier
                    # row (huge activations), and every later row then cancels against it (measured: correlation
                    # eigenvalues down to -1.6 for such experts). Hold the rows until there are enough.
                    mx.eval(Xe, Ae, ww); self.carry[e] = (Xe, Ae, ww)
                    continue
                # PER-EXPERT provisional centering vectors. Inputs are constant-dominated (|mean|^2 / variance up to
                # 1e2 overall and >1e6 for near-constant units); accumulating raw second moments and subtracting
                # mu mu^T afterwards cancels catastrophically in fp32. The down input is each expert's OWN hidden
                # layer, so a layer-wide mean does not center it.
                w2e = ww * ww
                self.mu0[e] = mx.sum(Xe * w2e, axis=0) / mx.sum(w2e)
                self.mu0d[e] = mx.sum(Ae * w2e, axis=0) / mx.sum(w2e)
                mx.eval(self.mu0[e], self.mu0d[e])
            Ye = (Xe - self.mu0[e]) * ww                                   # symmetric Gram form Y^T Y
            Be = (Ae - self.mu0d[e]) * ww
            self.H[e] = self.H[e] + Ye.T @ Ye
            self.Hd[e] = self.Hd[e] + Be.T @ Be
            self.m1[e] = self.m1[e] + mx.sum(Ye * ww, axis=0)
            self.m1d[e] = self.m1d[e] + mx.sum(Be * ww, axis=0)
            wnp = np.asarray(ww).astype(np.float64)
            self.wsum[e] += float((wnp ** 2).sum()); self.nrows[e] += Xe.shape[0]
            pend += [self.H[e], self.Hd[e], self.m1[e], self.m1d[e]]
            if len(pend) >= 32:
                mx.eval(pend); pend = []
        mx.eval(pend)
        self.pool, self.pool_tokens = [], 0
        mx.clear_cache()

    def save_hess(self, path: Path, iu_d, iu_i):
        self.flush(final=True)
        self.nonpos = [0, 0]      # units whose computed variance is <= 0 (gate/up input, down input): cancellation alarm
        self.cancel = [0.0, 0.0]  # max over experts/units of (provisional-mean error)^2 / variance: must stay << 1e4
        gd, gt, dd, dt, gi, gr, di, dr, gm, dm = [], [], [], [], [], [], [], [], [], []
        for e in range(self.E):
            n = max(self.wsum[e], 1e-30)
            if self.mu0[e] is None:                                           # expert never routed in calibration
                self.mu0[e] = mx.zeros((self.D,), mx.float32); self.mu0d[e] = mx.zeros((self.I,), mx.float32)
            for Hm, m1, mu0, iu, dl, tl, il, rl, ml in ((self.H[e], self.m1[e], self.mu0[e], iu_d, gd, gt, gi, gr, gm),
                                                       (self.Hd[e], self.m1d[e], self.mu0d[e], iu_i, dd, dt, di, dr, dm)):
                dm_ = m1 / n                                                   # mean of (x - mu0): small
                Hn = Hm / n - dm_[:, None] * dm_[None, :]                      # CENTERED covariance (w^2-weighted)
                mu = mu0 + dm_
                d = mx.maximum(mx.diag(Hn), 0.0)
                s = mx.where(d > 0, mx.rsqrt(mx.maximum(d, 1e-30)), 0.0)
                tri = mx.round(mx.clip((Hn * s[:, None] * s[None, :]).reshape(-1)[iu], -1.0, 1.0) * 32767.0).astype(mx.int16)
                top = mx.argsort(-d)[:NTOP].astype(mx.int32)
                rows = Hn[top]
                mx.eval(d, tri, top, rows, mu); dl.append(d); tl.append(tri); il.append(top); rl.append(rows); ml.append(mu)
                if self.nrows[e] > 0:
                    gi_ = 0 if iu is iu_d else 1
                    self.nonpos[gi_] += int(mx.sum(mx.diag(Hn) <= 0).item())
                    ratio = float(mx.max(mx.where(d > 0, dm_ * dm_ / mx.maximum(d, 1e-30), 0.0)).item())
                    self.cancel[gi_] = max(self.cancel[gi_], ratio)
            self.H[e] = None; self.Hd[e] = None
        out = {"gu_diag": mx.stack(gd), "gu_tri": mx.stack(gt), "dn_diag": mx.stack(dd), "dn_tri": mx.stack(dt),
               "gu_top_idx": mx.stack(gi), "gu_top_rows": mx.stack(gr), "dn_top_idx": mx.stack(di), "dn_top_rows": mx.stack(dr),
               "gu_mean": mx.stack(gm), "dn_mean": mx.stack(dm),
               "wsum": mx.array(self.wsum.astype(np.float32)), "rows": mx.array(self.nrows.astype(np.float32))}
        tmp = path.with_suffix(".tmp.safetensors")
        mx.save_safetensors(str(tmp), out); tmp.replace(path)
        del out, gd, gt, dd, dt; mx.clear_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--headers", required=True)
    ap.add_argument("--seqs", required=True)
    ap.add_argument("--stats-out", required=True); ap.add_argument("--ref-out", required=True)
    ap.add_argument("--hess-dir", default=None)
    ap.add_argument("--reservoir", type=int, default=8192); ap.add_argument("--reservoir-te", type=int, default=4096)
    ap.add_argument("--max-layers", type=int, default=0)
    ap.add_argument("--limit-seqs", type=int, default=0)
    ap.add_argument("--fingerprint", default=None, help="debug: write per-layer per-sequence hidden-state fingerprints (json)")
    a = ap.parse_args()
    src = Path(a.model)
    A = Args.from_config(json.loads((src / "config.json").read_text()))
    H = json.loads(Path(a.headers).read_text())
    seqs = [json.loads(l) for l in open(a.seqs)]
    if a.limit_seqs:
        seqs = seqs[: a.limit_seqs]
    n_cal = 0
    for s in seqs:
        s["cal_te"] = False
        if not s["ref"]:
            s["cal_te"] = n_cal % 10 == 9; n_cal += 1
    hess_dir = Path(a.hess_dir) if a.hess_dir else None
    if hess_dir:
        hess_dir.mkdir(parents=True, exist_ok=True)
        iu_d, iu_i = tri_index(A.hidden_size), tri_index(A.moe_intermediate_size)
    tok = lambda f: sum(len(s["ids"]) for s in seqs if f(s))
    print(f"{len(seqs)} sequences, {tok(lambda s: True)} tokens; held-out reference {tok(lambda s: s['ref'])}, "
          f"cal_tr {tok(lambda s: not s['ref'] and not s['cal_te'])}, cal_te {tok(lambda s: s['cal_te'])}; "
          f"hessians {'ON -> ' + str(hess_dir) if hess_dir else 'off'}", flush=True)
    t0 = time.time()
    emb = mx.load(str(src / H["model.embed_tokens.weight"][0]))["model.embed_tokens.weight"]
    hs = []
    for s in seqs:
        h = emb[mx.array(s["ids"])][None].astype(mx.bfloat16); mx.eval(h); hs.append(h)
    del emb; mx.clear_cache()
    stats_out, cache = {}, {}
    nL = a.max_layers or A.num_hidden_layers
    k = A.num_experts_per_tok
    for i in range(nL):
        tl = time.time()
        layer = DecoderLayer(A, i)
        raw = load_layer_raw(src, H, i, cache)
        layer.load_weights(list(sanitize_layer(raw, i, A.n_routed_experts).items()), strict=True)
        mx.eval(layer.parameters())
        is_moe = bool(A.moe_layer_freq[i])
        st = Stats(A.n_routed_experts, A.hidden_size, A.moe_intermediate_size, k, a.reservoir, a.reservoir_te,
                   seed=1000 + i, hess=hess_dir is not None) if is_moe else None
        for j, s in enumerate(seqs):
            x = hs[j]
            x = x + layer.self_attn(layer.input_layernorm(x))
            xn = layer.post_attention_layernorm(x)
            if is_moe:
                moe = layer.mlp
                idx, w = route(moe.gate, xn, k, A.norm_topk_prob, A.routed_scaling_factor)
                sw = moe.switch_mlp
                xe = mx.expand_dims(xn, (-2, -3))
                xs, ids_s, inv = _gather_sort(xe, idx)                                # same sorted path as SwitchGLU
                act_s = sw.activation(sw.up_proj(xs, ids_s, sorted_indices=True), sw.gate_proj(xs, ids_s, sorted_indices=True))
                y = _scatter_unsort(sw.down_proj(act_s, ids_s, sorted_indices=True), inv, idx.shape).squeeze(-2)
                x = x + mx.sum(y * w[..., None].astype(y.dtype), axis=-2).astype(x.dtype)
                mx.eval(x)
                if not s["ref"]:
                    T = xn.shape[1]
                    act = _scatter_unsort(act_s, inv, idx.shape)                      # silu(g)*u  (B,T,k,1,I)
                    xr = xn.reshape(T, -1); mx.eval(xr)
                    st.add(xr, idx.reshape(T, -1), w.reshape(T, -1), act.reshape(T, k, -1), s["cal_te"])
            else:
                x = x + layer.mlp(xn)
            mx.eval(x); hs[j] = x
        extra = ""
        if is_moe:
            b = f"model.layers.{i}.mlp.experts"
            if hess_dir:
                st.save_hess(hess_dir / f"L{i:02d}.safetensors", iu_d, iu_i)
                extra = (f"  H rows/expert min {int(st.nrows.min())} median {int(np.median(st.nrows))}"
                         f"  var<=0 units gu {st.nonpos[0]} dn {st.nonpos[1]}  max dm^2/var gu {st.cancel[0]:.1f} dn {st.cancel[1]:.1f}")
            stats_out.update({b + ".sum_x2": st.sum_x2, b + ".count": mx.array([float(st.count)]),
                              b + ".expert_sum_x2": st.e_x2, b + ".expert_count": st.e_cnt,
                              b + ".rows": mx.array(st.tr.rows), b + ".rows_topk_idx": mx.array(st.tr.tidx),
                              b + ".rows_topk_w": mx.array(st.tr.tw),
                              b + ".te_rows": mx.array(st.te.rows), b + ".te_rows_topk_idx": mx.array(st.te.tidx),
                              b + ".te_rows_topk_w": mx.array(st.te.tw),
                              f"model.layers.{i}.mlp.switch_mlp.down_proj.expert_sum_a2": st.e_a2,
                              f"model.layers.{i}.mlp.switch_mlp.down_proj.expert_count": st.e_cnt})
            tmp = a.stats_out + ".tmp.safetensors"
            mx.save_safetensors(tmp, stats_out); Path(tmp).replace(a.stats_out)
            cov = int((np.asarray(st.e_cnt) >= 1000).sum())
            extra = f"  experts>=1000 rows: {cov}/{A.n_routed_experts}" + extra
        if a.fingerprint:
            fp = json.loads(Path(a.fingerprint).read_text()) if Path(a.fingerprint).exists() else {}
            fp[str(i)] = [[float(mx.sum(h.astype(mx.float32)).item()), float(mx.sum(mx.abs(h.astype(mx.float32))).item()),
                           float(mx.max(mx.abs(h.astype(mx.float32))).item())] for h in hs]
            Path(a.fingerprint).write_text(json.dumps(fp))
        del layer, raw, st; mx.clear_cache()
        print(f"layer {i:2d} {'moe' if is_moe else 'dense'} {'SWA' if A.hybrid_layer_pattern[i] else 'DSA'} "
              f"{time.time()-tl:6.1f}s  total {(time.time()-t0)/60:.1f} min" + extra, flush=True)
    if a.max_layers:
        print("DEBUG STOP", flush=True); return
    normw = mx.load(str(src / H["model.norm.weight"][0]))["model.norm.weight"]
    head = mx.load(str(src / H["lm_head.weight"][0]))["lm_head.weight"].astype(mx.bfloat16)
    ref = {}
    for j, s in enumerate(seqs):
        if not s["ref"]:
            continue
        h = mx.fast.rms_norm(hs[j], normw.astype(mx.bfloat16), A.layernorm_epsilon)
        logits = (h @ head.T)[0].astype(mx.float32)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        top = mx.argsort(-lp, axis=-1)[:, :128]
        ref[f"p{j}.input_ids"] = mx.array(np.array(s["ids"], np.int32))
        ref[f"p{j}.top_ids"] = top.astype(mx.int32)
        ref[f"p{j}.top_logprobs"] = mx.take_along_axis(lp, top, axis=-1)
        mx.eval(ref[f"p{j}.top_logprobs"])
    mx.save_safetensors(a.ref_out, ref, metadata={"reference_precision": "BF16 (layer-streamed source)"})
    print(f"DONE stats {len(stats_out)} tensors, ref sequences {len(ref)//3}, {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
