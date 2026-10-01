"""GLM-5.3-Flash -> JANGTQ v2 bundle.

Inputs: BF16 source, plan.json (jang_tools.jangh.glm53.plan), FP8-reference calibration (diag reservoirs + routing, imatrix).
Per-tensor formats come from the plan (exact, header-derived). Routed experts:
  gate/up : GPTQ with the v2 codebook quantizer, H_e = routed-rows Hessian + lam * (tr H / tr D) * D_e,
            D_e = the expert's 600k-token imatrix diagonal (lam = 4: held-out 0.62-0.87x vs RTN, never worse, 2026-09-25).
            Rows with too little data fall back to the prior (the prior term always keeps H well conditioned).
  down    : same GPTQ only in layers where a held-out check (Hessian on reservoir rows [0,3072), score on
            [3072,4096) routed rows) wins: mean ratio < 0.98 and <= 1 of 8 sampled experts worse. Else RTN.
            (Measured: L5 down GPTQ is worse at every lam; L20 0.94x; L44 0.63x.)
  Row scales: exact H-weighted refit for the chosen codes (GPTQ) / imatrix-weighted LS (RTN). Stored float16.
No AWQ (TQ-objective alpha search gain <= 0.1% on all layers). No MTP (layer 45 dropped). Vision kept bf16.
Shards: written to a staging file, then streamed into an alignment-safe container (never rewritten in place),
alignment verified per shard.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from jang_tools.jangh.encode import encode, dequant
from jang_tools.jangh.format import pack_bitstream, codebook, cubic_params, h32, h32_both
from jang_tools.jangh.gptq import gptq_encode, _hinv_upper


from jang_tools.format.aligned_safetensors import rewrite_aligned_safetensors, verify_safetensors_alignment  # noqa: E402

E, D_IN, D_H, LIMIT = 288, 4096, 2048, 10.0
SHARD_BYTES = 5 * 2**30
FP32_SUFFIXES = ("A_log", "dt_bias", "e_score_correction_bias", "hc_base", "hc_scale")
EXPERT_CHUNK = 16


# ------------------------------------------------------------------ naming (identical to the shipped JANG bundle)
def runtime_name(k: str) -> str:
    k = k.replace("model.language_model.", "model.")
    k = k.replace("model.visual.", "visual.")
    k = re.sub(r"\.hc_(attn|ffn)_(base|fn|scale)$", r".\1_hc.hc_\2", k)
    k = k.replace(".mlp.gate.e_score_correction_bias", ".mlp.e_score_correction_bias")
    if k.endswith(("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight")):
        k = k[: -len(".weight")]
    if k.endswith("self_attn.o_norm.weight"):
        k = k[: -len(".weight")]
    return k


# ------------------------------------------------------------------ shard writer
class ShardWriter:
    def __init__(self, out_dir: Path):
        self.out, self.buf, self.nbytes, self.i, self.wm, self.total = out_dir, {}, 0, 0, {}, 0

    def add(self, name, arr):
        mx.eval(arr)
        assert name not in self.wm and name not in self.buf, f"duplicate tensor {name}"
        self.buf[name] = arr
        self.nbytes += arr.nbytes
        self.total += arr.nbytes
        if self.nbytes >= SHARD_BYTES:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        self.i += 1
        stage = self.out / f".staging-{self.i:05d}.safetensors"
        final = self.out / f"model-{self.i:05d}.safetensors"
        mx.save_safetensors(str(stage), self.buf)
        rewrite_aligned_safetensors(stage, final)
        n, bad = verify_safetensors_alignment(final)
        assert bad == 0 and n == len(self.buf), f"alignment check failed for {final}: {bad}/{n}"
        stage.unlink()
        for k in self.buf:
            self.wm[k] = final.name
        self.buf, self.nbytes = {}, 0
        mx.clear_cache()

    def finish(self):
        self.flush()
        files = sorted(self.out.glob("model-*.safetensors"))
        ren = {}
        for i, f in enumerate(files, 1):
            new = f"model-{i:05d}-of-{len(files):05d}.safetensors"
            ren[f.name] = new
            f.rename(self.out / new)
        wm = {k: ren[v] for k, v in self.wm.items()}
        (self.out / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": self.total, "format": "jangtq2"}, "weight_map": wm}, indent=1))
        return len(files)


# ------------------------------------------------------------------ non-expert formats
def mxfp8_pack(w: mx.array):
    """OUR mxfp8 packer: per-32 e8m0 scale = best of {ceil, ceil-1} by group error (mx.quantize picks one step too
    small). Returns (weight uint32 (N, K/4), scales uint8 (N, K/32)) in MLX mxfp8 layout."""
    W = w.astype(mx.float32)
    o, i = W.shape
    G = W.reshape(o, i // 32, 32)
    a = mx.maximum(mx.abs(G).max(-1, keepdims=True), 1e-30)
    e = mx.ceil(mx.log2(a / 448.0))
    best = None
    for de in (0, -1):
        sc = 2.0 ** (e + de)
        q8 = mx.to_fp8(mx.clip(G / sc, -448, 448))
        err = ((G - mx.from_fp8(q8, dtype=mx.float32) * sc) ** 2).sum(-1, keepdims=True)
        cand = (q8, e + de, err)
        if best is None:
            best = cand
        else:
            pick = err < best[2]
            best = (mx.where(pick, q8, best[0]), mx.where(pick, e + de, best[1]), mx.minimum(err, best[2]))
    q8, ex, _ = best
    return q8.reshape(o, i).view(mx.uint32), (ex.reshape(o, i // 32) + 127).astype(mx.uint8)


# ------------------------------------------------------------------ expert Hessians
def routed_hessians(X, tidx, tw, experts, rows_sel=None):
    """H_e = sum_r w_r x_r x_r^T / sum_r w_r over reservoir rows routed to e. Returns list of (H, n_rows)."""
    out = []
    for e in experts:
        m = tidx == e
        if rows_sel is not None:
            m = m & rows_sel[:, None]
        r, s = np.nonzero(m)
        if len(r) == 0:
            out.append((mx.zeros((X.shape[1], X.shape[1])), 0))
            continue
        Xe = X[mx.array(r.astype(np.uint32))]
        w = mx.array(tw[r, s].astype(np.float32))
        out.append(((Xe * w[:, None]).T @ Xe / float(w.sum().item()), len(r)))
    return out


def with_prior(H, n, D, lam):
    Dm = mx.diag(D)
    if n == 0:
        return Dm
    return H + lam * (mx.trace(H) / mx.maximum(mx.trace(Dm), 1e-30)) * Dm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--diag", required=True)
    ap.add_argument("--imatrix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lam", type=float, default=4.0)
    ap.add_argument("--no-gptq", action="store_true", help="RTN (imatrix-weighted LS scale) everywhere: control build")
    ap.add_argument("--layers", default="all", help="debug: subset of MoE layers (bundle incomplete)")
    ap.add_argument("--rotation", default="none", choices=["none", "hadamard32"])
    ap.add_argument("--diag2", default=None, help="second capture (same schema) pooled into Hessians / priors")
    ap.add_argument("--w2", type=float, default=2.0, help="weight of --diag2 statistics vs --diag")
    a = ap.parse_args()
    t0 = time.time()
    src = Path(a.model)
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to write into non-empty {out} (never rewrite artifacts in place)")
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(Path(a.plan).read_text())
    fmts = plan["tensor_formats"]
    ebits = plan["expert_bits"]
    wm = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    diag = mx.load(a.diag)
    diag2 = mx.load(a.diag2) if a.diag2 else None
    from jang_tools.jangh.glm53.pool_stats import pooled
    imat = mx.load(a.imatrix)
    cfg = json.loads((src / "config.json").read_text())
    report = {"layers": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "rotation": a.rotation,
              "diag2": a.diag2, "w2": a.w2 if a.diag2 else None}
    ROT = a.rotation == "hadamard32"
    rotW = (lambda M: h32(M)) if ROT else (lambda M: M)
    rotH = (lambda M: h32_both(M)) if ROT else (lambda M: M)

    cache: dict = {}

    def get(name):
        f = wm[name]
        if f not in cache:
            cache.clear()
            mx.clear_cache()
            cache[f] = mx.load(str(src / f))
        return cache[f][name]

    W = ShardWriter(out)
    qcfg = {"group_size": 64, "bits": 8, "mode": "affine"}      # top-level default for loaders that expect it
    n_fmt = {}

    # ---------------- non-expert tensors, grouped by layer for locality
    def layer_of(k):
        m = re.search(r"layers\.(\d+)\.", k)
        return int(m.group(1)) if m else -1
    keys = sorted((k for k in wm if fmts.get(k) not in ("expert", "drop_mtp")),
                  key=lambda k: (0 if not k.startswith("model.visual.") else 1, layer_of(k), k))
    want_layers = None if a.layers == "all" else {int(x) for x in a.layers.split(",")}
    for k in keys:
        f = fmts[k]
        name = runtime_name(k)
        w = get(k)
        n_fmt[f] = n_fmt.get(f, 0) + 1
        if f in ("keep", "vision_bf16"):
            if k.endswith(("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight")):
                w = w.reshape(w.shape[0], w.shape[-1])
            W.add(name, w)
            continue
        mod = name[: -len(".weight")]
        if f == "mxfp8":
            wq, sc = mxfp8_pack(w)
            W.add(mod + ".weight", wq); W.add(mod + ".scales", sc)
            qcfg[mod] = {"group_size": 32, "bits": 8, "mode": "mxfp8"}
        elif f == "affine8":
            wb = w.astype(mx.bfloat16)                      # bf16 scales/biases: no fp32 promotion under bf16 acts
            wq, sc, bi = mx.quantize(wb, group_size=64, bits=8)
            W.add(mod + ".weight", wq); W.add(mod + ".scales", sc); W.add(mod + ".biases", bi)
            qcfg[mod] = {"group_size": 64, "bits": 8, "mode": "affine"}
        else:
            raise ValueError(f"unknown format {f} for {k}")
    print(f"non-expert tensors done: {n_fmt} ({(time.time()-t0)/60:.1f} min)", flush=True)

    # ---------------- routed experts per MoE layer
    moe_layers = sorted({int(k.split(":")[0]) for k in ebits})
    for L in moe_layers:
        if want_layers is not None and L not in want_layers:
            continue
        tl = time.time()
        mod = f"model.language_model.layers.{L}.mlp.experts"
        X, tidx, tw, imx_gu, _, n_first = pooled(diag, diag2, mod, a.w2)
        imx_d = imat[f"model.layers.{L}.mlp.switch_mlp.down_proj.expert_diag"].astype(mx.float32)
        b_gu, b_d = ebits[f"{L}:gate_up"], ebits[f"{L}:down"]
        base = f"model.language_model.layers.{L}.mlp.experts"
        rec = {"bits_gate_up": b_gu, "bits_down": b_d}

        def load_proj(proj, e0, e1):
            return mx.stack([get(f"{base}.{e}.{proj}_proj.weight") for e in range(e0, e1)]).astype(mx.float32)

        # ---- down GPTQ held-out decision for this layer
        down_gptq = False
        if not a.no_gptq:
            counts = np.bincount(tidx[:3072].ravel(), minlength=E)
            sample = [int(e) for e in np.argsort(-counts)[:4]] + [int(e) for e in np.random.default_rng(L).choice(E, 4, replace=False)]
            ratios = []
            for e in sample:
                Wg = get(f"{base}.{e}.gate_proj.weight").astype(mx.float32); Wu = get(f"{base}.{e}.up_proj.weight").astype(mx.float32)
                Wd = get(f"{base}.{e}.down_proj.weight").astype(mx.float32)
                r_tr, s_tr = np.nonzero(tidx == e); keep = (r_tr < 3072) | (r_tr >= n_first); r_tr, s_tr = r_tr[keep], s_tr[keep]
                r_te, s_te = np.nonzero(tidx[3072:n_first] == e); r_te = r_te + 3072
                if len(r_te) < 4 or len(r_tr) < 8:
                    continue
                act = lambda Xr: (lambda g, u: g * mx.sigmoid(g) * u)(mx.minimum(Xr @ Wg.T, LIMIT), mx.clip(Xr @ Wu.T, -LIMIT, LIMIT))
                Atr = act(X[mx.array(r_tr.astype(np.uint32))]); Ate = act(X[mx.array(r_te.astype(np.uint32))])
                wtr = mx.array(tw[r_tr, s_tr].astype(np.float32)); wte = mx.array(tw[r_te, s_te].astype(np.float32))
                Hd = (Atr * wtr[:, None]).T @ Atr / float(wtr.sum().item())
                ref = Ate @ Wd.T; den = float((wte[:, None] * ref ** 2).sum())
                q, s = encode(rotW(Wd), b_d, None if ROT else imx_d[e])
                rtn = float((wte[:, None] * (rotW(Ate) @ dequant(q, s, b_d).T - ref) ** 2).sum()) / den
                Q, S = gptq_encode(rotW(Wd)[None], rotH(with_prior(Hd, len(r_tr), imx_d[e], a.lam))[None], b_d)
                gq = float((wte[:, None] * (rotW(Ate) @ dequant(Q[0], S[0], b_d).T - ref) ** 2).sum()) / den
                ratios.append(gq / rtn)
            down_gptq = len(ratios) >= 4 and float(np.mean(ratios)) < 0.98 and sum(r > 1 for r in ratios) <= 1
            rec["down_heldout_ratios"] = ratios
        rec["down_method"] = "gptq" if down_gptq else "rtn"
        rec["gate_up_method"] = "rtn" if a.no_gptq else "gptq"

        # ---- encode all experts, chunked
        outs = {p: {"packed": [], "scales": []} for p in ("gate", "up", "down")}
        for e0 in range(0, E, EXPERT_CHUNK):
            e1 = min(E, e0 + EXPERT_CHUNK)
            ex = list(range(e0, e1))
            Wg, Wu, Wd = load_proj("gate", e0, e1), load_proj("up", e0, e1), load_proj("down", e0, e1)
            Wg_raw, Wu_raw = Wg, Wu                      # un-rotated copies drive the down-input activations
            Wg, Wu, Wd = rotW(Wg), rotW(Wu), rotW(Wd)
            if a.no_gptq and ROT:
                for p, Wp, b in (("gate", Wg, b_gu), ("up", Wu, b_gu), ("down", Wd, b_d)):
                    q, s = encode(Wp, b)
                    outs[p]["packed"].append(pack_bitstream(q, b)); outs[p]["scales"].append(s.astype(mx.float16))
            elif a.no_gptq:
                for p, Wp, D, b in (("gate", Wg, imx_gu[e0:e1][:, None, :], b_gu), ("up", Wu, imx_gu[e0:e1][:, None, :], b_gu),
                                    ("down", Wd, imx_d[e0:e1][:, None, :], b_d)):
                    q, s = encode(Wp, b, D)
                    outs[p]["packed"].append(pack_bitstream(q, b)); outs[p]["scales"].append(s.astype(mx.float16))
            else:
                Hs = routed_hessians(X, tidx, tw, ex)
                Hgu = rotH(mx.stack([with_prior(H, n, imx_gu[e], a.lam) for (H, n), e in zip(Hs, ex)]))
                Ugu = _hinv_upper(Hgu, 0.01)                    # shared by gate and up (same input)
                for p, Wp in (("gate", Wg), ("up", Wu)):
                    Q, S = gptq_encode(Wp, Hgu, b_gu, U=Ugu)
                    outs[p]["packed"].append(pack_bitstream(Q, b_gu)); outs[p]["scales"].append(S.astype(mx.float16))
                if down_gptq:
                    Hd_list = []
                    for e in ex:
                        r, s_ = np.nonzero(tidx == e)
                        if len(r) == 0:
                            Hd_list.append(mx.diag(imx_d[e])); continue
                        Xe = X[mx.array(r.astype(np.uint32))]; we = mx.array(tw[r, s_].astype(np.float32))
                        g = mx.minimum(Xe @ Wg_raw[e - e0].T, LIMIT); u = mx.clip(Xe @ Wu_raw[e - e0].T, -LIMIT, LIMIT)
                        A = g * mx.sigmoid(g) * u
                        Hd_list.append(with_prior((A * we[:, None]).T @ A / float(we.sum().item()), len(r), imx_d[e], a.lam))
                    Q, S = gptq_encode(Wd, rotH(mx.stack(Hd_list)), b_d)
                else:
                    Q, S = encode(Wd, b_d, None if ROT else imx_d[e0:e1][:, None, :])
                outs["down"]["packed"].append(pack_bitstream(Q, b_d)); outs["down"]["scales"].append(S.astype(mx.float16))
            for p in outs:
                mx.eval(outs[p]["packed"][-1], outs[p]["scales"][-1])
            del Wg, Wu, Wd, Wg_raw, Wu_raw
            mx.clear_cache()
        for p, b in (("gate", b_gu), ("up", b_gu), ("down", b_d)):
            mname = f"model.layers.{L}.mlp.switch_mlp.{p}_proj"
            W.add(mname + ".tq2_packed", mx.concatenate(outs[p]["packed"], axis=0))
            W.add(mname + ".tq2_scales", mx.concatenate(outs[p]["scales"], axis=0))
            qcfg[mname] = {"mode": "jangtq2", "bits": b, "rotation": a.rotation}
        rec["seconds"] = round(time.time() - tl, 1)
        report["layers"][str(L)] = rec
        (out / "jangtq2_build_report.json").write_text(json.dumps(report, indent=1))
        print(f"L{L:2d} gu tq{b_gu} ({rec['gate_up_method']}) down tq{b_d} ({rec['down_method']}"
              + (f", held-out {np.mean(rec.get('down_heldout_ratios', [1])):.3f}" if not a.no_gptq else "")
              + f") {rec['seconds']}s  total {(time.time()-t0)/60:.1f} min", flush=True)

    nshards = W.finish()

    # ---------------- configs + sidecars
    cfg.setdefault("text_config", {})
    cfg["text_config"]["num_nextn_predict_layers"] = 0                  # MTP dropped
    if "num_nextn_predict_layers" in cfg:
        cfg["num_nextn_predict_layers"] = 0
    cfg["quantization"] = qcfg
    cfg["quantization_config"] = qcfg
    jtq = {"version": 2, "packing": "lsb-bitstream", "scale_dtype": "float16", "rotation": a.rotation,
           "codebook_family": "odd-cubic", "codebooks": {str(b): {"alpha": cubic_params(b)[0], "beta": cubic_params(b)[1],
                                                                  "levels": codebook(b).tolist()} for b in (2, 3, 4)}}
    cfg["jangtq"] = jtq
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    jang_cfg = {
        "format": "jangtq2", "format_version": 2, "jangtq": jtq,
        "source": {"repo": "zai-org/GLM-5.3-Flash-BF16", "revision": "a5b45eb41df6402735dedc900be14a42e8d5e538"},
        "mtp": "dropped", "vision": "bf16", "video": "supported (shared vision tower)",
        "calibration": {"reference": "FP8 release", "tokens": 600064, "mix": "web50/code25/chat15/math10",
                        "awq": plan.get("awq"), "gptq": None if a.no_gptq else {"lam": a.lam, "prior": "per-expert imatrix diagonal",
                                                                                 "gate_up": "all layers", "down": "held-out gated per layer"},
                        "imatrix": "per-expert (x-side) importance: GPTQ prior + RTN scale weights"},
        "plan": {k: plan[k] for k in ("total_gib", "fixed_gib_by_format", "experts_gib", "expert_bits_histogram")},
        "expert_bits": ebits,
    }
    (out / "jang_config.json").write_text(json.dumps(jang_cfg, indent=1))
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "processor_config.json"):
        if (src / f).exists():
            shutil.copy2(src / f, out / f)
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S"); report["shards"] = nshards; report["total_bytes"] = W.total
    (out / "jangtq2_build_report.json").write_text(json.dumps(report, indent=1))
    print(f"DONE {nshards} shards, {W.total/2**30:.3f} GiB tensor payload, {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
