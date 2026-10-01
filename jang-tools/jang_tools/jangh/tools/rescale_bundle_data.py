"""Per-expert gains measured on real activations (GLM-5.3-Flash layout). Codes and all other tensors unchanged.

For every routed expert, on the rows routed to it in the pooled captures (FP8 calibration, agentic, long reasoning):
  gate / up : g = <y, y> / <y, yq>            (pre-activation of the source vs the quantized matrix)
  down      : g = <o, o> / <o, oq>            (whole expert output, given the corrected gate / up)
clipped to GCLIP; experts with fewer than 24 rows take the layer median. New scale = old scale x g (per expert).
Same rule as hybrid3.py GAIN=data. Writes a NEW bundle directory.
usage: rescale_bundle_data.py <src bundle> <dst bundle> <capture,capture,...>
"""
import os
import json, shutil, sys, time
from pathlib import Path
import numpy as np, mlx.core as mx

from jang_tools.format.aligned_safetensors import rewrite_aligned_safetensors, verify_safetensors_alignment
from jang_tools.jangh.format import unpack_bitstream, codebook, h32

SRC = Path(os.environ["JANGH_SOURCE"])
B, D, CAPS = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].split(",")
GCLIP = (0.9, 1.35); LIM = 10.0
assert B.resolve() != D.resolve() and not (D.exists() and any(D.iterdir())), "never rewrite a bundle in place"
D.mkdir(parents=True, exist_ok=True)
cfg = json.loads((B / "config.json").read_text()); bq = cfg["quantization"]
idx = json.loads((B / "model.safetensors.index.json").read_text()); bwm = idx["weight_map"]
swm = json.loads((SRC / "model.safetensors.index.json").read_text())["weight_map"]
caps = [mx.load(p) for p in CAPS]; files, bfiles = {}, {}


def src(k):
    f = swm[k]
    if f not in files: files.clear(); mx.clear_cache(); files[f] = mx.load(str(SRC / f))
    return files[f][k].astype(mx.float32)


def bt(k):
    f = bwm[k]
    if f not in bfiles:
        if len(bfiles) > 3: bfiles.clear()
        bfiles[f] = mx.load(str(B / f))
    return bfiles[f][k]


act = lambda a_, u_: (lambda gg, uu: gg * mx.sigmoid(gg) * uu)(mx.minimum(a_, LIM), mx.clip(u_, -LIM, LIM))
layers = sorted({int(k.split(".")[2]) for k in bq if ".mlp.switch_mlp.gate_proj" in k})
assert len(layers) == 42
NEW, report, t0 = {}, {}, time.time()
for L in layers:
    b = f"model.language_model.layers.{L}.mlp.experts"
    X = mx.concatenate([c[b + ".rows"].astype(mx.float32) for c in caps if b + ".rows" in c], axis=0)
    ti = np.concatenate([np.asarray(c[b + ".rows_topk_idx"]) for c in caps if b + ".rows" in c], axis=0)
    assert X.shape[0] >= 8192, (L, X.shape)
    pk, sc, spec = {}, {}, {}
    for p in ("gate", "up", "down"):
        n = f"model.layers.{L}.mlp.switch_mlp.{p}_proj"; spec[p] = bq[n]; pk[p] = bt(n + ".tq2_packed"); sc[p] = bt(n + ".tq2_scales")
    E = pk["gate"].shape[0]; g = {p: np.full(E, np.nan, np.float32) for p in pk}; small = 0

    def dec(p, e):
        K = pk[p].shape[-1] * 32 // spec[p]["bits"]
        w = mx.array(codebook(spec[p]["bits"]))[unpack_bitstream(pk[p][e], spec[p]["bits"], K)] * sc[p][e].astype(mx.float32)[..., None]
        return h32(w) if spec[p].get("rotation") == "hadamard32" else w
    for e in range(E):
        r = np.nonzero((ti == e).any(1))[0]
        if len(r) < 24: small += 1; continue
        Xe = X[mx.array(r.astype(np.uint32))]
        w = {p: src(f"model.language_model.layers.{L}.mlp.experts.{e}.{p}_proj.weight") for p in pk}; q = {p: dec(p, e) for p in pk}
        yg, yu = Xe @ w["gate"].T, Xe @ w["up"].T; qg, qu = Xe @ q["gate"].T, Xe @ q["up"].T
        cg = float(np.clip(float(mx.sum(yg * yg) / mx.maximum(mx.sum(yg * qg), 1e-9)), *GCLIP)); cu = float(np.clip(float(mx.sum(yu * yu) / mx.maximum(mx.sum(yu * qu), 1e-9)), *GCLIP))
        y = act(yg, yu) @ w["down"].T; yq = act(qg * cg, qu * cu) @ q["down"].T
        cd = float(np.clip(float(mx.sum(y * y) / mx.maximum(mx.sum(y * yq), 1e-9)), *GCLIP))
        g["gate"][e], g["up"][e], g["down"][e] = cg, cu, cd
        if e % 64 == 0: mx.clear_cache()
    for p in pk:
        g[p] = np.where(np.isnan(g[p]), np.nanmedian(g[p]), g[p]).astype(np.float32)
        n = f"model.layers.{L}.mlp.switch_mlp.{p}_proj"
        s2 = (sc[p].astype(mx.float32) * mx.array(g[p])[:, None]).astype(mx.float16); mx.eval(s2)
        assert s2.shape == sc[p].shape and bool(mx.all(mx.isfinite(s2.astype(mx.float32))))
        NEW[n + ".tq2_scales"] = s2
        report[n] = {"bits": spec[p]["bits"], "gain_mean": float(g[p].mean()), "gain_min": float(g[p].min()), "gain_max": float(g[p].max()), "experts_lt24_rows": small, "rows": int(X.shape[0])}
    print(f"L{L:2d} gains gate {g['gate'].mean():.3f} up {g['up'].mean():.3f} down {g['down'].mean():.3f} (experts < 24 rows {small}) {(time.time()-t0)/60:.1f} min", flush=True)
    del X; mx.clear_cache()
assert len(NEW) == 126
bfiles.clear(); files.clear(); shards = sorted(set(bwm.values())); total = 0; used = 0
for n, f in enumerate(shards, 1):
    t = mx.load(str(B / f)); out = {k: (NEW[k] if k in NEW else v) for k, v in t.items()}; used += sum(k in NEW for k in t)
    stage = D / f".staging-{f}"; mx.save_safetensors(str(stage), out); rewrite_aligned_safetensors(stage, D / f); stage.unlink()
    cnt, bad = verify_safetensors_alignment(D / f); assert bad == 0 and cnt == len(out), (f, bad, cnt)
    chk = mx.load(str(D / f))
    for k, v in t.items():
        assert chk[k].dtype == v.dtype and chk[k].shape == v.shape, k
        if k not in NEW and v.size <= 1 << 24:
            a_, b_ = (chk[k], v) if v.dtype not in (mx.bfloat16, mx.float16, mx.float32) else (chk[k].astype(mx.float32), v.astype(mx.float32))
            assert bool(mx.all(a_ == b_)), f"{k} changed"
    total += sum(v.nbytes for v in out.values()); del t, out, chk; mx.clear_cache()
    print(f"[{n}/{len(shards)}] {f} written, {(time.time()-t0)/60:.1f} min", flush=True)
assert used == 126, used
(D / "model.safetensors.index.json").write_text(json.dumps(idx, indent=1))
for f in B.iterdir():
    if f.is_file() and not f.name.startswith("model-") and f.name != "model.safetensors.index.json" and not f.name.startswith("."):
        shutil.copy2(f, D / f.name)
jc = json.loads((D / "jang_config.json").read_text())
jc["scale_correction"] = {"revision": 3, "rule": "per-expert gain measured on routed activations: gate/up <y,y>/<y,yq>, down unit gain of the expert output", "clip": list(GCLIP),
                          "captures": ["FP8 calibration (600k tokens)", "agentic (171k tokens)", "long reasoning (12 transcripts, 5k tokens each)"], "codes_changed": False}
(D / "jang_config.json").write_text(json.dumps(jc, indent=1))
(D / "jangh_scale_correction_report.json").write_text(json.dumps(report, indent=1))
print(f"DONE {len(shards)} shards, payload {total/2**30:.3f} GiB, {(time.time()-t0)/60:.1f} min", flush=True)
