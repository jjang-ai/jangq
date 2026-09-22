"""GPTQ for MiMo-V2.6 routed experts (batched over experts, MLX).

One layer-streamed pass over the calibration corpus on the EXACT source:
  1. capture, per expert, the full input Hessian of its routed tokens
       gate/up : H_e = E[x x^T] over tokens routed to e     (d = 4096)
       down    : H_e = E[h h^T] over (token, e) pairs         (d = 2048)
  2. shrink toward the layer-pooled Hessian: rare experts are rank-deficient
       H'_e = (n_e H_e + tau H_pool) / (n_e + tau),  tau = d tokens
     then damp: H'_e += 0.01 * mean(diag) * I
  3. for every AFFINE unit of the plan: fixed grid = the same imatrix fit the
     converter uses (bytes/scales identical), then GPTQ error-compensated
     rounding onto that grid (block 128, Cholesky of H^-1).
     AWQ: gate/up weights are W*s and see x/s, so H -> S^-1 H S^-1.
  4. never-worse guard per expert: keep GPTQ codes only if the Hessian-weighted
     reconstruction error tr(D H D^T) beats the fitted-RTN codes.
Codes are written to <out>/L{L}.{proj}.safetensors (weight/scales/biases) with
a per-unit report; convert_v26 consumes them via plan["gptq_dir"].
Source activations are propagated between layers.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

from .convert_v26 import load_plan
from .mxfp4_codec import mxfp4_raw_to_mlx
from .v26_quant import fit_affine, pack_codes
from .v26_source import SourceStream, layer_masks
from .v26_sweep import awq_scale, expert_importance

PROJS = ("gate_proj", "up_proj", "down_proj")
EXP_CHUNK = 32


def _seg_hessians(xs: mx.array, idx_sorted: mx.array, E: int, acc: list, cnt: np.ndarray, valid=None):
    """xs [R, d] rows sorted by expert id; acc = list of E [d, d] fp32 sums (updated in place)."""
    counts = np.bincount(np.array(idx_sorted), minlength=E)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    for e in range(E):
        n = int(counts[e])
        if n:
            x = xs[int(starts[e]): int(starts[e]) + n].astype(mx.float32)
            if valid is not None:
                mask = valid[int(starts[e]):int(starts[e]) + n]
                x = x * mx.array(mask[:, None], dtype=mx.float32)
                counts[e] = int(mask.sum())
            acc[e] = acc[e] + x.T @ x
        # Bound the Metal command buffer; a lazy 256-expert Hessian update
        # timed out on the real 4096-dimensional layer despite fitting RAM.
        if e % 8 == 7:
            mx.eval(acc[e-7:e+1])
    mx.eval(acc)
    cnt += counts


def gptq_round(W, H, scales, biases, bits, group_size, block=128):
    """W [E,O,D] fp32 (already AWQ-scaled), H [E,D,D] damped, grid from the fit.
    Returns integer codes [E,O,D] (uint32)."""
    E, O, D = W.shape
    maxq = float((1 << bits) - 1)
    s = scales.astype(mx.float32)
    b = biases.astype(mx.float32)
    # U = upper Cholesky factor of H^-1  (CPU linalg)
    Hinv = mx.linalg.inv(H, stream=mx.cpu)
    Hinv = (Hinv + mx.swapaxes(Hinv, -1, -2)) / 2
    U = mx.swapaxes(mx.linalg.cholesky(Hinv, stream=mx.cpu), -1, -2)  # upper
    mx.eval(U)
    W = mx.array(W)
    codes_cols = []
    for i1 in range(0, D, block):
        i2 = min(i1 + block, D)
        W1 = W[:, :, i1:i2]
        U1 = U[:, i1:i2, i1:i2]
        errs, cols = [], []
        for j in range(i2 - i1):
            w = W1[:, :, j]
            sj, bj = s[:, :, (i1 + j) // group_size], b[:, :, (i1 + j) // group_size]
            q = mx.clip(mx.round((w - bj) / sj), 0, maxq)
            cols.append(q)
            d = U1[:, j, j][:, None]
            err = (w - (q * sj + bj)) / d
            if j + 1 < i2 - i1:
                W1 = mx.concatenate([W1[:, :, : j + 1],
                                     W1[:, :, j + 1:] - err[:, :, None] * U1[:, j, j + 1:][:, None, :]], axis=-1)
            errs.append(err)
            if j % 8 == 7:
                mx.eval(W1, errs, cols)
        Err = mx.stack(errs, axis=-1)  # [E,O,blk]
        if i2 < D:
            W = mx.concatenate([W[:, :, :i2], W[:, :, i2:] - Err @ U[:, i1:i2, i2:]], axis=-1)
        codes_cols.append(mx.stack(cols, axis=-1))
        mx.eval(W, codes_cols[-1])
    return mx.concatenate(codes_cols, axis=-1).astype(mx.uint32)


def h_err(Wq, W, H):
    Dm = Wq - W
    return ((Dm @ H) * Dm).sum(axis=(-1, -2))  # [E]


def solve_unit(ss, L, proj, spec, Hs, a_scale, imp, out_dir, report):
    E = ss.args.n_routed_experts
    bits, gs = spec["bits"], spec["group_size"]
    names = [f"model.layers.{L}.mlp.experts.{e}.{proj}.weight" for e in range(E)]
    packed_all, sc_all, bi_all, kept = [], [], [], 0
    gain = []
    for c0 in range(0, E, EXP_CHUNK):
        raw = [mxfp4_raw_to_mlx(*ss.idx.read_mxfp4_raw(n)) for n in names[c0:c0 + EXP_CHUNK]]
        W = mx.dequantize(mx.array(np.stack([r[0] for r in raw])), mx.array(np.stack([r[1] for r in raw])),
                          group_size=32, bits=4, mode="mxfp4").astype(mx.float32)
        if a_scale is not None and proj != "down_proj":
            W = W * a_scale
        cimp = imp[c0:c0 + EXP_CHUNK]
        p_fit, s_fit, b_fit = fit_affine(W, cimp, bits=bits, group_size=gs)
        H = Hs(c0, min(c0 + EXP_CHUNK, E))
        print(f"[gptq] solve L{L}.{proj} experts {c0}:{min(c0+EXP_CHUNK,E)}", flush=True)
        mx.eval(W, H, p_fit, s_fit, b_fit)
        code_parts = []
        for e0 in range(0, W.shape[0], 4):
            print(f"[gptq] round experts {c0+e0}:{c0+min(e0+4,W.shape[0])}", flush=True)
            code_parts.append(gptq_round(W[e0:e0+4], H[e0:e0+4],
                                        s_fit[e0:e0+4], b_fit[e0:e0+4], bits, gs))
            mx.eval(code_parts[-1])
        codes = mx.concatenate(code_parts)
        Wq_fit = mx.dequantize(p_fit, s_fit, b_fit, group_size=gs, bits=bits, mode="affine").astype(mx.float32)
        p_g = pack_codes(codes, bits)
        # Compare both candidates through the same stored bf16 affine grid.
        # Float32 code*scale+bias only on the GPTQ arm biases this guard.
        Wq_g = mx.dequantize(p_g, s_fit, b_fit, group_size=gs, bits=bits, mode="affine").astype(mx.float32)
        e_fit, e_g = h_err(Wq_fit, W, H), h_err(Wq_g, W, H)
        use_g = e_g < e_fit
        packed = mx.where(use_g[:, None, None], p_g, p_fit)
        mx.eval(packed, e_fit, e_g)
        kept += int(use_g.sum())
        gain += (1 - np.array(mx.minimum(e_g, e_fit)) / np.maximum(np.array(e_fit), 1e-30)).tolist()
        packed_all.append(packed); sc_all.append(s_fit); bi_all.append(b_fit)
        del W, H
    t = {"weight": mx.concatenate(packed_all), "scales": mx.concatenate(sc_all), "biases": mx.concatenate(bi_all)}
    mx.save_safetensors(str(out_dir / f"L{L}.{proj}.safetensors"), t,
                        metadata={"bits": str(bits), "group_size": str(gs), "awq": str(a_scale is not None)})
    report[f"L{L}.{proj}"] = {"spec": spec, "gptq_kept": kept, "experts": E,
                              "mean_herr_reduction": float(np.mean(gain)), "median_herr_reduction": float(np.median(gain))}


def run(src, tokens, plan_path, out_dir, batch=1, tau_scale=1.0, damp=0.01, max_units=0):
    from .v26_gptq_provenance import bind_run
    bind_run(out_dir, src, tokens, json.loads(plan_path.read_text()), tau_scale=tau_scale, damp=damp)
    out_dir.mkdir(parents=True, exist_ok=True)
    ss = SourceStream(src)
    a = ss.args
    plan = load_plan(plan_path, a.num_hidden_layers, [L for L in range(a.num_hidden_layers) if a.moe_layer_freq[L]])
    stats = mx.load(plan["stats"])
    alpha = {int(k): float(v) for k, v in (plan.get("awq_alpha") or {}).items()}
    pad = 151643
    real = np.array([len(r) - int(np.argmax(r[::-1] != pad)) for r in tokens])
    order = np.argsort(-real)
    batches = [tokens[order[i:i + batch], : int(real[order[i:i + batch]].max())] for i in range(0, len(tokens), batch)]
    hs = [ss.embed(mx.array(b)) for b in batches]
    masks = {}
    report = {}
    rp = out_dir / "gptq_report.json"
    capture_path = out_dir / "hessian_capture_report.json"
    capture_report = json.loads(capture_path.read_text()) if capture_path.exists() else {}
    if rp.exists():
        report = json.loads(rp.read_text())
    t0 = time.time()
    solved = 0
    H_, I_, E = a.hidden_size, a.moe_intermediate_size, a.n_routed_experts
    for L in range(a.num_hidden_layers):
        lay = ss.build_layer(L)
        moe = bool(a.moe_layer_freq[L])
        todo = [p for p in PROJS if moe and plan["_units"][(L, p)]["mode"] == "affine"
                and not (out_dir / f"L{L}.{p}.safetensors").exists()]
        need_gu = any(p in todo for p in ("gate_proj", "up_proj"))
        need_d = "down_proj" in todo
        if need_gu:
            Hgu, Cgu = [mx.zeros((H_, H_)) for _ in range(E)], np.zeros(E)
        if need_d:
            Hd, Cd = [mx.zeros((I_, I_)) for _ in range(E)], np.zeros(E)
        propagation_error = None
        for i, h in enumerate(hs):
            T = h.shape[1]
            if T not in masks:
                masks[T] = layer_masks(a, T)
            full_m, swa_m = masks[T]
            h = h + lay.self_attn(lay.input_layernorm(h), swa_m if lay.is_swa else full_m)
            xin = lay.post_attention_layernorm(h)
            if moe and (need_gu or need_d):
                print(f"[gptq] capture L{L} batch {i+1}/{len(hs)}", flush=True)
                sw = lay.mlp.switch_mlp
                xf = xin.reshape(-1, H_)
                idx, wts = lay.mlp.gate(xf)
                xe = mx.expand_dims(xf, (-2, -3))
                xs, idx_s, inv = _gather_sort(xe, idx)
                # Padding participates in propagation but never in calibration.
                lengths = real[order[i*batch:i*batch+batch]]
                valid = (np.arange(T)[None, :] < lengths[:, None]).reshape(-1)
                sorted_valid = np.repeat(valid, idx.shape[-1])[np.array(mx.argsort(inv))]
                if need_gu:
                    _seg_hessians(xs.reshape(-1, H_), idx_s, E, Hgu, Cgu, sorted_valid)
                g = sw.gate_proj(xs, idx_s, sorted_indices=True)
                u = sw.up_proj(xs, idx_s, sorted_indices=True)
                hh = nn.silu(g) * u
                if need_d:
                    _seg_hessians(hh.reshape(-1, I_), idx_s, E, Hd, Cd, sorted_valid)
                mx.eval(hh)
                y = _scatter_unsort(sw.down_proj(hh, idx_s, sorted_indices=True), inv, idx.shape)
                y = (y.squeeze(-2).astype(mx.float32) * wts[..., None]).sum(-2).astype(h.dtype)
                if i == 0:
                    reference_y = lay.mlp(xin)
                    error = mx.abs(reference_y.astype(mx.float32)-y.reshape(h.shape).astype(mx.float32)).max()
                    scale = mx.maximum(mx.abs(reference_y.astype(mx.float32)).max(), 1e-8)
                    propagation_error = float((error/scale).item())
                    if propagation_error > 1e-5:
                        raise ValueError(f"Hessian capture changes source propagation at L{L}: {propagation_error}")
                hs[i] = h + y.reshape(h.shape)
            else:
                hs[i] = h + lay.mlp(xin)
            mx.eval(hs[i], *(Hgu if need_gu else []), *(Hd if need_d else []))
        if todo:
            expected_count = int(real.sum()) * a.num_experts_per_tok
            observed = {}
            for label, needed, counts in (("gate_up", need_gu, Cgu if need_gu else None),
                                          ("down", need_d, Cd if need_d else None)):
                if needed:
                    if int(counts.sum()) != expected_count:
                        raise ValueError(f"Hessian routed-token coverage mismatch L{L}.{label}")
                    observed[label] = {"routed_tokens": int(counts.sum()),
                                       "expert_min": int(counts.min()), "expert_max": int(counts.max()),
                                       "experts_with_tokens": int((counts>0).sum())}
            capture_report[str(L)] = {"valid_tokens": int(real.sum()), "counts": observed,
                                      "source_propagation_rel_max": propagation_error,
                                      "peak_gib": mx.get_peak_memory()/2**30}
            capture_path.write_text(json.dumps(capture_report, indent=2)+"\n")
            s = awq_scale(stats, L, alpha[L]) if alpha.get(L) else None
            for p in todo:
                Hlist, C = (Hd, Cd) if p == "down_proj" else (Hgu, Cgu)
                d = Hlist[0].shape[-1]
                pool = sum(Hlist) / max(float(C.sum()), 1.0)
                mx.eval(pool)
                tau = tau_scale * d
                inv_s = (1.0 / s) if (s is not None and p != "down_proj") else None

                def shrunk(e0, e1, Hlist=Hlist, C=C, pool=pool, tau=tau, inv_s=inv_s, d=d):
                    He = mx.stack([(Hlist[e] + tau * pool) / (float(C[e]) + tau) for e in range(e0, e1)])
                    if inv_s is not None:
                        He = He * inv_s[None, :, None] * inv_s[None, None, :]
                    diag = mx.diagonal(He, axis1=-2, axis2=-1)
                    return He + (damp * diag.mean(-1) + 1e-8)[:, None, None] * mx.eye(d)[None]

                imp = expert_importance(stats, L, p, s)
                solve_unit(ss, L, p, plan["_units"][(L, p)], shrunk, s, imp, out_dir, report)
                rp.write_text(json.dumps(report, indent=1))
                print(f"[gptq] L{L} {p}: kept {report[f'L{L}.{p}']['gptq_kept']}/{E} "
                      f"median H-err reduction {report[f'L{L}.{p}']['median_herr_reduction']:.3f} ({time.time()-t0:.0f}s)", flush=True)
                solved += 1
                if max_units and solved >= max_units:
                    print(f"[gptq] UNIT LIMIT {solved}; partial run, not DONE", flush=True)
                    return
        del lay
        mx.clear_cache()
        print(f"[gptq] layer {L} done {time.time()-t0:.0f}s", flush=True)
    print("[gptq] DONE", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--tokens", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=1, help="Sequences per Hessian capture batch; 1 is the measured M5 Max setting")
    ap.add_argument("--limit", type=int, default=0, help="Limit calibration sequences; use a separate smoke output directory")
    ap.add_argument("--max-units", type=int, default=0, help="Stop after this many newly solved projection units (0 = all)")
    a = ap.parse_args(argv)
    toks = np.load(a.tokens)
    if a.limit:
        toks = toks[: a.limit]
    if a.limit < 0 or a.max_units < 0:
        ap.error("limits must be nonnegative")
    run(a.src, toks, a.plan, a.out, a.batch, max_units=a.max_units)


if __name__ == "__main__":
    main()
