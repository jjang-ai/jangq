"""Naive-N0.5-Flash -> JANGH bundle (on-disk identifiers: jangtq v2 / mode "jangtq2" / tq2_packed, tq2_scales).
Run with MLX_ENABLE_TF32=0.

Inputs : BF16 source + headers.json, plan.json (n05.plan), capture-v3 Hessian directory.
Experts: per layer L, per projection group (gate/up share the input; down has its own):
           H_e from jang_tools.jangh.n05.recipe.hessian (m_min chosen per layer on held-out rows, recorded in the plan)
           float64 CPU factorization in worker processes (n05.gptq_stable.prepare) -> U, H in the rotated+permuted basis
           GPU: Hadamard-32 rotation of W, act-order GPTQ with the JANGH codebook, exact H-weighted row scale,
                codes un-permuted to storage order, LSB bitstream packing, fp16 row scales.
         --rtn : no GPTQ (codebook RTN in the rotated basis) — control build.
Non-experts: plan formats (affine8 g64 with bf16 scales/biases | mxfp8 g32 with OUR packer | keep).
Resumable: every finished layer is written to <out>/.stage/L##.safetensors (tmp + rename); a rerun skips them.
Final assembly: each stage file is streamed into an alignment-safe shard and verified; never rewritten in place.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent



E, D, I = 256, 4096, 2048
CHUNK = 16
_W = {}


def _worker(job):
    """CPU float64: Hessian of one expert -> upper factor in the rotated + act-ordered basis, written to shared memory."""
    (hess_file, which, e, slot, m_min, tau, pool_path, shm_u, shm_h, K, nslots, damp) = job
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
    from jang_tools.jangh.n05.hess_cpu import HessStore, prepare
    from jang_tools.jangh.n05.recipe import hessian
    key = (hess_file, which)
    if key not in _W:
        _W.clear(); _W[key] = HessStore(hess_file, which)
    HS = _W[key]
    pool = np.load(pool_path, mmap_mode="r")
    Hm = hessian(HS.cov(e), HS.mean[e], HS.rows[e], pool, m_min, tau)
    U, Hp, perm, extra = prepare(Hm, True, damp=damp)
    su, sh = shared_memory.SharedMemory(name=shm_u), shared_memory.SharedMemory(name=shm_h)
    np.ndarray((nslots, K, K), np.float32, buffer=su.buf)[slot] = U
    np.ndarray((nslots, K, K), np.float32, buffer=sh.buf)[slot] = Hp
    su.close(); sh.close()
    return slot, perm, extra


def _pool_part(job):
    hess_file, which, e0, e1 = job
    from jang_tools.jangh.n05.hess_cpu import HessStore
    HS = HessStore(hess_file, which)
    acc = np.zeros((HS.n, HS.n)); tot = 0.0
    for e in range(e0, e1):
        if HS.wsum[e] > 0:
            acc += HS.wsum[e] * HS.cov(e); tot += HS.wsum[e]
    return acc, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--headers", required=True)
    ap.add_argument("--plan", required=True); ap.add_argument("--hess-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--rtn", action="store_true"); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stats", default=None, help="capture stats (held-out rows): every layer is VERIFIED on unseen rows from the packed tensors")
    ap.add_argument("--m-min-gu", type=float, default=None, help="EXPERIMENT: override the plan's pooled-covariance floor for gate/up")
    ap.add_argument("--m-min-dn", type=float, default=None, help="EXPERIMENT: override the plan's pooled-covariance floor for down")
    ap.add_argument("--damp", type=float, default=0.01, help="diagonal damping in the rotated basis (recipe: 0.01)")
    ap.add_argument("--skip-nonexpert", action="store_true", help="EXPERIMENT: expert stage files only")
    ap.add_argument("--keep-stage", action="store_true", help="keep <out>/.stage (per-layer files) after assembly: sens_real.py reads them")
    ap.add_argument("--layers", default="all", help="debug: subset of MoE layers (bundle incomplete, no assembly)")
    a = ap.parse_args()
    assert os.environ.get("MLX_ENABLE_TF32") == "0", "run with MLX_ENABLE_TF32=0"
    import mlx.core as mx
    from jang_tools.jangh.encode import encode
    from jang_tools.jangh.format import pack_bitstream, unpack_bitstream, codebook, cubic_params, h32
    from jang_tools.jangh.encode import dequant
    from jang_tools.jangh.gptq import gptq_encode
    from jang_tools.jangh.glm53.convert_tq import mxfp8_pack
    from jang_tools.format.aligned_safetensors import rewrite_aligned_safetensors, verify_safetensors_alignment

    t0 = time.time()
    src, out = Path(a.model), Path(a.out)
    stage = out / ".stage"; stage.mkdir(parents=True, exist_ok=True)
    if any(out.glob("model-*.safetensors")):
        raise SystemExit(f"refusing: {out} already holds final shards (never rewrite artifacts in place)")
    H = json.loads(Path(a.headers).read_text()); plan = json.loads(Path(a.plan).read_text())
    fmts, ebits, mmin = plan["tensor_formats"], plan["expert_bits"], plan["recipe_m_min"]
    rep_path = out / "jangh_build_report.json"
    report = json.loads(rep_path.read_text()) if rep_path.exists() else {"layers": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                                        "method": "rtn" if a.rtn else "gptq", "rotation": "hadamard32"}
    cache = {}

    def get(name):
        f = H[name][0]
        if f not in cache:
            cache.clear(); mx.clear_cache()
            cache[f] = mx.load(str(src / f))
        return cache[f][name]

    def save_stage(name, tensors):
        tmp = stage / f".{name}.tmp.safetensors"
        mx.save_safetensors(str(tmp), tensors); tmp.replace(stage / f"{name}.safetensors")

    stats = mx.load(a.stats) if a.stats else None
    qcfg = {"group_size": 64, "bits": 8, "mode": "affine"}
    # ---------------- non-expert tensors (one stage file per ~5 GiB)
    ne_done = stage / "nonexpert.done.json"
    if not ne_done.exists() and not a.skip_nonexpert:
        keys = sorted(k for k in H if fmts[k] != "expert")
        buf, nb, part, q_ne, counts = {}, 0, 0, {}, {}
        for k in keys:
            f = fmts[k]; w = get(k); counts[f] = counts.get(f, 0) + 1
            if f == "keep":
                buf[k] = w; nb += w.nbytes
            else:
                mod = k[: -len(".weight")]
                if f == "mxfp8":
                    wq, sc = mxfp8_pack(w); buf[mod + ".weight"] = wq; buf[mod + ".scales"] = sc
                    q_ne[mod] = {"group_size": 32, "bits": 8, "mode": "mxfp8"}
                else:
                    wq, sc, bi = mx.quantize(w.astype(mx.bfloat16), group_size=64, bits=8)
                    buf[mod + ".weight"] = wq; buf[mod + ".scales"] = sc; buf[mod + ".biases"] = bi
                    q_ne[mod] = {"group_size": 64, "bits": 8, "mode": "affine"}
                mx.eval(list(buf.values())[-3:]); nb += wq.nbytes + sc.nbytes
            if nb >= 3 * 2**30:
                save_stage(f"N{part:02d}", buf); buf, nb, part = {}, 0, part + 1
        if buf:
            save_stage(f"N{part:02d}", buf)
        ne_done.write_text(json.dumps({"quant": q_ne, "counts": counts}, indent=1))
        print(f"non-expert tensors done: {counts} ({(time.time()-t0)/60:.1f} min)", flush=True)
    if ne_done.exists():
        qcfg.update(json.loads(ne_done.read_text())["quant"])
    for L_ in mmin:
        if a.m_min_gu is not None:
            mmin[L_]["gu"] = a.m_min_gu
        if a.m_min_dn is not None:
            mmin[L_]["dn"] = a.m_min_dn

    # ---------------- routed experts
    layers = list(range(1, 48)) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    ctx = mp.get_context("spawn")
    pool_exec = None if a.rtn else ctx.Pool(a.workers)
    for L in layers:
        if (stage / f"L{L:02d}.safetensors").exists():
            continue
        tl = time.time()
        base = f"model.layers.{L}.mlp.experts"
        hess_file = str(Path(a.hess_dir) / f"L{L:02d}.safetensors")
        bits = {p: int(ebits[f"{L}:{p}"]) for p in ("gate", "up", "down")}
        rec = {"bits": bits, "m_min": mmin[str(L)], "damp": a.damp, "extra_damp": {}}
        outs = {p: {"packed": [], "scales": []} for p in ("gate", "up", "down")}
        for grp, names, K, which in (("gu", ("gate", "up"), D, "gu"), ("dn", ("down",), I, "dn")):
            if not a.rtn:
                parts = pool_exec.map(_pool_part, [(hess_file, which, e0, min(E, e0 + 32)) for e0 in range(0, E, 32)])
                pooled = sum(p[0] for p in parts) / max(sum(p[1] for p in parts), 1e-300)
                pool_path = str(stage / f".pool_{which}.npy"); np.save(pool_path, pooled); del parts, pooled
                extras = []
            for e0 in range(0, E, CHUNK):
                ex = list(range(e0, min(E, e0 + CHUNK)))
                Ws = {n: h32(mx.stack([get(f"{base}.{e}.{n}_proj.weight") for e in ex]).astype(mx.float32)) for n in names}
                mx.eval(list(Ws.values()))
                if a.rtn:
                    for n in names:
                        q, s = encode(Ws[n], bits[n])
                        outs[n]["packed"].append(pack_bitstream(q, bits[n])); outs[n]["scales"].append(s.astype(mx.float16))
                        mx.eval(outs[n]["packed"][-1], outs[n]["scales"][-1])
                    continue
                su = shared_memory.SharedMemory(create=True, size=len(ex) * K * K * 4)
                sh = shared_memory.SharedMemory(create=True, size=len(ex) * K * K * 4)
                try:
                    jobs = [(hess_file, which, e, i, float(mmin[str(L)][grp]), 4096.0 if grp == "gu" else 2048.0, pool_path,
                             su.name, sh.name, K, len(ex), a.damp) for i, e in enumerate(ex)]
                    res = sorted(pool_exec.map(_worker, jobs))
                    perm = mx.array(np.stack([r[1] for r in res])); extras += [r[2] for r in res]
                    U = mx.array(np.ndarray((len(ex), K, K), np.float32, buffer=su.buf))
                    Hp = mx.array(np.ndarray((len(ex), K, K), np.float32, buffer=sh.buf))
                    mx.eval(U, Hp)
                finally:
                    su.close(); su.unlink(); sh.close(); sh.unlink()
                inv = mx.argsort(perm, axis=-1)
                for n in names:
                    Wp = mx.take_along_axis(Ws[n], mx.broadcast_to(perm[:, None, :], Ws[n].shape), axis=2)
                    Q, S = gptq_encode(Wp, Hp, bits[n], U=U)
                    Q = mx.take_along_axis(Q, mx.broadcast_to(inv[:, None, :], Q.shape), axis=2)
                    outs[n]["packed"].append(pack_bitstream(Q, bits[n])); outs[n]["scales"].append(S.astype(mx.float16))
                    mx.eval(outs[n]["packed"][-1], outs[n]["scales"][-1])
                del U, Hp, Ws; mx.clear_cache()
            if not a.rtn:
                ex_arr = np.array(extras)
                rec["extra_damp"][grp] = {"experts_escalated": int((ex_arr > 0).sum()), "max": float(ex_arr.max())}
                os.unlink(pool_path)
        tens = {}
        for n in ("gate", "up", "down"):
            m = f"model.layers.{L}.mlp.switch_mlp.{n}_proj"
            tens[m + ".tq2_packed"] = mx.concatenate(outs[n]["packed"], axis=0)
            tens[m + ".tq2_scales"] = mx.concatenate(outs[n]["scales"], axis=0)
            assert tens[m + ".tq2_packed"].shape == (E, I if n != "down" else D, (D if n != "down" else I) * bits[n] // 32)
            assert bool(mx.all(mx.isfinite(tens[m + ".tq2_scales"].astype(mx.float32))).item()), f"non-finite scale in {m}"
        if stats is not None:
            # VERIFY from the packed tensors (what the runtime will read), on rows of unseen documents
            b_ = f"model.layers.{L}.mlp.experts"
            Xte = stats[b_ + ".te_rows"].astype(mx.float32); ti = np.asarray(stats[b_ + ".te_rows_topk_idx"]); tw = np.asarray(stats[b_ + ".te_rows_topk_w"])
            tc = np.bincount(ti.ravel(), minlength=E)
            elig = np.nonzero(tc >= 16)[0]
            assert len(elig) > 0, f"L{L}: no expert has 16 held-out rows — verification would be vacuous"
            sample = sorted({int(e) for e in np.argsort(-tc)[:3]} | {int(e) for e in np.random.default_rng(L).choice(elig, min(3, len(elig)), replace=False)})
            num = den = 0.0
            for e in sample:
                r_, s_ = np.nonzero(ti == e)
                X = Xte[mx.array(r_.astype(np.int32))]; w2 = mx.array((tw[r_, s_] ** 2).astype(np.float32))[:, None]
                Wb = {n: get(f"{base}.{e}.{n}_proj.weight").astype(mx.float32) for n in ("gate", "up", "down")}
                g, u = X @ Wb["gate"].T, X @ Wb["up"].T
                y = (g * mx.sigmoid(g) * u) @ Wb["down"].T
                Wq = {}
                for n in ("gate", "up", "down"):
                    m_ = f"model.layers.{L}.mlp.switch_mlp.{n}_proj"
                    K_ = I if n == "down" else D
                    Wq[n] = dequant(unpack_bitstream(tens[m_ + ".tq2_packed"][e], bits[n], K_), tens[m_ + ".tq2_scales"][e], bits[n])
                xr = h32(X); gq, uq = xr @ Wq["gate"].T, xr @ Wq["up"].T
                yq = h32(gq * mx.sigmoid(gq) * uq) @ Wq["down"].T
                num += float((w2 * (yq - y) ** 2).sum()); den += float((w2 * y ** 2).sum())
            rec["heldout_nmse_packed"] = num / max(den, 1e-30)
            if not np.isfinite(rec["heldout_nmse_packed"]) or rec["heldout_nmse_packed"] > 1.0:
                raise SystemExit(f"L{L}: held-out verification failed (NMSE {rec['heldout_nmse_packed']})")
        save_stage(f"L{L:02d}", tens)
        rec["seconds"] = round(time.time() - tl, 1)
        report["layers"][str(L)] = rec
        rep_path.write_text(json.dumps(report, indent=1))
        print(f"L{L:2d} gate tq{bits['gate']} up tq{bits['up']} down tq{bits['down']} m_min {rec['m_min']} held-out NMSE {rec.get('heldout_nmse_packed', float('nan')):.4f} "
              f"escalated {rec['extra_damp']} {rec['seconds']}s  total {(time.time()-t0)/60:.1f} min", flush=True)
        del outs, tens; mx.clear_cache()
    if pool_exec is not None:
        pool_exec.close(); pool_exec.join()
    if a.layers != "all":
        print("PARTIAL (debug) — no assembly", flush=True); return

    # ---------------- assembly: stage files -> alignment-safe shards
    files = sorted(stage.glob("N*.safetensors")) + sorted(stage.glob("L*.safetensors"))
    wm, total = {}, 0
    for i, f in enumerate(files, 1):
        final = out / f"model-{i:05d}-of-{len(files):05d}.safetensors"
        rewrite_aligned_safetensors(f, final)
        n, bad = verify_safetensors_alignment(final)
        assert bad == 0, f"{final}: {bad} misaligned tensors"
        import struct
        with open(final, "rb") as fh:
            hn = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(hn))
        for k, v in hdr.items():
            if k != "__metadata__":
                assert k not in wm, f"duplicate tensor {k}"
                wm[k] = final.name; total += v["data_offsets"][1] - v["data_offsets"][0]
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total, "format": "jangtq2"}, "weight_map": wm}, indent=1))
    for L in range(1, 48):
        for n in ("gate", "up", "down"):
            qcfg[f"model.layers.{L}.mlp.switch_mlp.{n}_proj"] = {"mode": "jangtq2", "bits": int(ebits[f"{L}:{n}"]), "rotation": "hadamard32"}
    cfg = json.loads((src / "config.json").read_text())
    cfg.pop("auto_map", None)                       # no remote code: the runtime owns the architecture
    cfg["quantization"] = qcfg; cfg["quantization_config"] = qcfg
    jtq = {"version": 2, "packing": "lsb-bitstream", "scale_dtype": "float16", "rotation": "hadamard32", "codebook_family": "odd-cubic",
           "codebooks": {str(b): {"alpha": cubic_params(b)[0], "beta": cubic_params(b)[1], "levels": codebook(b).tolist()} for b in (2, 3, 4)}}
    cfg["jangtq"] = jtq
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    (out / "jang_config.json").write_text(json.dumps({
        "format": "jangtq2", "format_version": 2, "format_name": "JANGH", "jangtq": jtq,
        "source": {"repo": "NaiveAI/Naive-N0.5-Flash", "revision": "0235b3b"},
        "plan": {k: plan[k] for k in ("total_gib", "fixed_gib_by_format", "experts_gib", "expert_bits_histogram", "nonexpert")},
        "expert_bits": ebits}, indent=1))
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "vocab.json", "merges.txt",
              "special_tokens_map.json", "LICENSE"):
        if (src / f).exists():
            shutil.copy2(src / f, out / f)
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S"); report["shards"] = len(files); report["total_bytes"] = total
    rep_path.write_text(json.dumps(report, indent=1))
    if a.keep_stage:
        stage.rename(out.parent / (out.name + ".stage"))          # NOT inside the bundle: the root holds shards + one index only
    else:
        shutil.rmtree(stage)
    print(f"DONE {len(files)} shards, {total/2**30:.3f} GiB tensor payload, {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
