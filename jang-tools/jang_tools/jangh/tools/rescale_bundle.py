"""Gain-corrected row scales for a JANGH bundle (GLM-5.3-Flash layout). Codes (tq2_packed) and every other tensor unchanged.

Why: MSE-optimal row scales shrink every quantized matrix along its source row (<w,q>/<w,w> = 0.88 at 2-bit, 0.965 at
3-bit). Three matrices per expert, 42 layers: deep features are attenuated; after long reasoning P(</think>) is
suppressed by up to 7.7 nats and reasoning never closes. New scale = old scale * (<w,w>/<w,q>)^POW per output row,
clipped to [0.5, 2.0] (identical to the rule measured in hybrid2.py). Rotation is orthogonal: the inner products are
taken in the stored (rotated) basis.
Writes a NEW bundle directory (never in place); shards go through a staging file and the alignment-safe rewriter.
usage: rescale_bundle.py <src bundle> <dst bundle> <pow> <which: gate,up,down>
"""
import os
import json, shutil, sys, time, hashlib
from pathlib import Path
import numpy as np, mlx.core as mx

from jang_tools.format.aligned_safetensors import rewrite_aligned_safetensors, verify_safetensors_alignment
from jang_tools.jangh.format import unpack_bitstream, codebook, h32

SRC = Path(os.environ["JANGH_SOURCE"])
B, D, POW, WHICH = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]), sys.argv[4].split(",")
assert B.resolve() != D.resolve() and not (D.exists() and any(D.iterdir())), "never rewrite a bundle in place"
D.mkdir(parents=True, exist_ok=True)
cfg = json.loads((B / "config.json").read_text()); bq = cfg["quantization"]
idx = json.loads((B / "model.safetensors.index.json").read_text()); bwm = idx["weight_map"]
swm = json.loads((SRC / "model.safetensors.index.json").read_text())["weight_map"]
files = {}


def src(k):
    f = swm[k]
    if f not in files: files.clear(); mx.clear_cache(); files[f] = mx.load(str(SRC / f))
    return files[f][k].astype(mx.float32)


def new_scales(mod, pk, sc):
    spec = bq[mod]; assert spec["mode"] == "jangtq2" and sc.dtype == mx.float16
    L = int(mod.split(".")[2]); p_ = mod.rsplit(".", 1)[1][:-len("_proj")]
    K = pk.shape[-1] * 32 // spec["bits"]; cb = mx.array(codebook(spec["bits"])); out, gs = [], []
    for e0 in range(0, pk.shape[0], 16):
        e1 = min(e0 + 16, pk.shape[0])
        q = cb[unpack_bitstream(pk[e0:e1], spec["bits"], K)] * sc[e0:e1].astype(mx.float32)[..., None]
        w = mx.stack([src(f"model.language_model.layers.{L}.mlp.experts.{e}.{p_}_proj.weight") for e in range(e0, e1)])
        if spec.get("rotation") == "hadamard32": w = h32(w)
        c = mx.sum(w * w, axis=-1) / mx.maximum(mx.sum(w * q, axis=-1), 1e-12)
        c = mx.clip(c, 0.5, 2.0) ** POW
        s2 = (sc[e0:e1].astype(mx.float32) * c).astype(mx.float16); mx.eval(s2, c); out.append(s2); gs.append(c)
        del q, w; mx.clear_cache()
    g = mx.concatenate([x.reshape(-1) for x in gs]); s2 = mx.concatenate(out, axis=0)
    assert s2.shape == sc.shape and bool(mx.all(mx.isfinite(s2.astype(mx.float32))))
    return s2, {"bits": spec["bits"], "gain_mean": float(mx.mean(g)), "gain_min": float(mx.min(g)), "gain_max": float(mx.max(g)), "clipped_rows": int(mx.sum((g <= 0.5 ** POW) | (g >= 2.0 ** POW)))}


t0 = time.time(); report = {}; shards = sorted(set(bwm.values())); total = 0
for n, f in enumerate(shards, 1):
    t = mx.load(str(B / f)); out = {}; changed = 0
    for k, v in t.items():
        if k.endswith(".tq2_scales") and ".mlp.switch_mlp." in k and k.rsplit(".", 2)[1][:-len("_proj")] in WHICH:
            mod = k[:-len(".tq2_scales")]; pkk = mod + ".tq2_packed"
            pk = t[pkk] if pkk in t else mx.load(str(B / bwm[pkk]))[pkk]
            out[k], report[mod] = new_scales(mod, pk, v); changed += 1
        else:
            out[k] = v
    stage = D / f".staging-{f}"; mx.save_safetensors(str(stage), out)
    rewrite_aligned_safetensors(stage, D / f); stage.unlink()
    cnt, bad = verify_safetensors_alignment(D / f); assert bad == 0 and cnt == len(out), (f, bad, cnt, len(out))
    # unchanged tensors must be bit-identical to the source bundle
    chk = mx.load(str(D / f))
    for k, v in t.items():
        assert chk[k].dtype == v.dtype and chk[k].shape == v.shape, k
        if not (k.endswith(".tq2_scales") and ".mlp.switch_mlp." in k) and v.size <= 1 << 24:
            a_, b_ = (chk[k], v) if v.dtype not in (mx.bfloat16, mx.float16, mx.float32) else (chk[k].astype(mx.float32), v.astype(mx.float32))
            assert bool(mx.all(a_ == b_)), f"{k} changed"
    total += sum(v.nbytes for v in out.values()); del t, out, chk; mx.clear_cache()
    print(f"[{n}/{len(shards)}] {f}: {changed} scale tensors corrected, {(time.time()-t0)/60:.1f} min", flush=True)
assert len(report) == 42 * len(WHICH), len(report)
(D / "model.safetensors.index.json").write_text(json.dumps(idx, indent=1))
for f in B.iterdir():
    if f.is_file() and not f.name.startswith("model-") and f.name != "model.safetensors.index.json" and not f.name.startswith("."):
        shutil.copy2(f, D / f.name)
jc = json.loads((D / "jang_config.json").read_text())
jc["scale_correction"] = {"revision": 2, "rule": "row scale *= (<w,w>/<w,q>)^pow, clipped to [0.5, 2.0]", "pow": POW, "projections": WHICH,
                          "reason": "MSE-optimal scales attenuate deep features; close-reasoning decision restored", "codes_changed": False}
(D / "jang_config.json").write_text(json.dumps(jc, indent=1))
(D / "jangh_scale_correction_report.json").write_text(json.dumps(report, indent=1))
print(f"DONE {len(shards)} shards, payload {total/2**30:.3f} GiB, {(time.time()-t0)/60:.1f} min", flush=True)
