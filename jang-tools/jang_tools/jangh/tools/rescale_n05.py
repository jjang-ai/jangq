"""Naive-N0.5-Flash JANGH bundle with unit-gain row scales (scale x <w,w>/<w,q>, clip 0.5-2.0).
Codes and every other tensor unchanged. Writes a NEW directory. Same rule as GLM-5.3-Flash-JANGH2 revision 2.
usage: rescale_n05.py <src bundle> <dst bundle>"""
import os
import json, shutil, sys, time
from pathlib import Path
import mlx.core as mx

from jang_tools.format.aligned_safetensors import rewrite_aligned_safetensors, verify_safetensors_alignment
from jang_tools.jangh.format import unpack_bitstream, codebook, h32
SRC = Path(os.environ["JANGH_SOURCE"]); H = json.loads(Path(os.environ["JANGH_HEADERS"]).read_text())
B, D = Path(sys.argv[1]), Path(sys.argv[2])
assert B.resolve() != D.resolve() and not (D.exists() and any(D.iterdir()))
D.mkdir(parents=True, exist_ok=True)
bq = json.loads((B / "config.json").read_text())["quantization"]; idx = json.loads((B / "model.safetensors.index.json").read_text()); bwm = idx["weight_map"]
files = {}
def src(k):
    f = H[k][0]
    if f not in files:
        if len(files) > 2: files.pop(next(iter(files)))
        files[f] = mx.load(str(SRC / f))
    return files[f][k].astype(mx.float32)
def new_scales(mod, pk, sc):
    spec = bq[mod]; assert spec["mode"] == "jangtq2" and sc.dtype == mx.float16
    L = int(mod.split(".")[2]); p = mod.rsplit(".", 1)[1][:-len("_proj")]
    K = pk.shape[-1] * 32 // spec["bits"]; cb = mx.array(codebook(spec["bits"])); out, clipped, n = [], 0, 0
    for e0 in range(0, pk.shape[0], 16):
        e1 = min(e0 + 16, pk.shape[0])
        q = cb[unpack_bitstream(pk[e0:e1], spec["bits"], K)] * sc[e0:e1].astype(mx.float32)[..., None]
        w = mx.stack([src(f"model.layers.{L}.mlp.experts.{e}.{p}_proj.weight") for e in range(e0, e1)])
        if spec.get("rotation") == "hadamard32": w = h32(w)
        c = mx.sum(w * w, axis=-1) / mx.maximum(mx.sum(w * q, axis=-1), 1e-12)
        cc = mx.clip(c, 0.5, 2.0); clipped += int(mx.sum(cc != c)); n += c.size
        s2 = (sc[e0:e1].astype(mx.float32) * cc).astype(mx.float16); mx.eval(s2); out.append(s2); del q, w
    mx.clear_cache(); s2 = mx.concatenate(out, axis=0); assert s2.shape == sc.shape and bool(mx.all(mx.isfinite(s2.astype(mx.float32))))
    return s2, {"bits": spec["bits"], "rows": n, "clipped_rows": clipped}
t0 = time.time(); rep = {}; shards = sorted(set(bwm.values()))
for n, f in enumerate(shards, 1):
    t = mx.load(str(B / f)); out = {}
    for k, v in t.items():
        if k.endswith(".tq2_scales"):
            mod = k[:-len(".tq2_scales")]; pkk = mod + ".tq2_packed"; pk = t[pkk] if pkk in t else mx.load(str(B / bwm[pkk]))[pkk]
            out[k], rep[mod] = new_scales(mod, pk, v)
        else: out[k] = v
    st = D / f".staging-{f}"; mx.save_safetensors(str(st), out); rewrite_aligned_safetensors(st, D / f); st.unlink()
    cnt, bad = verify_safetensors_alignment(D / f); assert bad == 0 and cnt == len(out)
    del t, out; mx.clear_cache(); print(f"[{n}/{len(shards)}] {f} {(time.time()-t0)/60:.1f} min", flush=True)
(D / "model.safetensors.index.json").write_text(json.dumps(idx, indent=1))
for f in B.iterdir():
    if f.is_file() and not f.name.startswith("model-") and f.name != "model.safetensors.index.json" and not f.name.startswith("."): shutil.copy2(f, D / f.name)
with open(D / "scale_correction_report.json", "w") as fh:
    json.dump(rep, fh, indent=1)
print(f"DONE modules {len(rep)} clipped rows {sum(v['clipped_rows'] for v in rep.values())} of {sum(v['rows'] for v in rep.values())}, {(time.time()-t0)/60:.1f} min", flush=True)
