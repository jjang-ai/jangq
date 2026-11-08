"""Assemble a mixed JANGT/JANGH N0.5 bundle from stage directories.

Sources (first holder wins per routed projection, exactly as n05_streval scored it):
  --jt     JANGT stage dir (moe_encode.py, ragged jt_* tensors; every unit must have ONE width -> stored (E, N, W))
  --pick   re-quantized JANGH downs (jh_pick*)
  --base   the published JANGH2 bundle (non-experts, tokenizer/template files, remaining JANGH units)
  --plan   allocation json + key ("97.55:free") -> cross-checked against what the sources hold (measured widths)
Output: aligned shards (~--shard-gib), model.safetensors.index.json, config.json with the mixed per-module contract
and a `jangt` block, jang_config.json, tokenizer/template files, and (only with --drafter) a DSpark drafter under
dspark/. Bundles ship WITHOUT a drafter by default (2026-10-09). Never rewrites an existing output.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np

from jang_tools.format.aligned_safetensors import rewrite_aligned_safetensors, verify_safetensors_alignment
from jang_tools.jangt import v2l as VL

PROJ_KIN = {"gate": 4096, "up": 4096, "down": 2048}
COPY_FILES = ("chat_template.jinja", "generation_config.json", "merges.txt", "special_tokens_map.json",
              "tokenizer_config.json", "tokenizer.json", "vocab.json", "LICENSE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jt", required=True); ap.add_argument("--pick", required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--plan", required=True); ap.add_argument("--plan-key", required=True)
    ap.add_argument("--drafter", default=None, help="optional DSpark drafter dir (not shipped by default)"); ap.add_argument("--out", required=True)
    ap.add_argument("--shard-gib", type=float, default=5.0)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing: {out} exists and is not empty")
    stage = out.parent / (out.name + ".staging"); stage.mkdir(parents=True, exist_ok=False); out.mkdir(parents=True, exist_ok=True)
    base, jtd, pkd = Path(a.base), Path(a.jt), Path(a.pick)
    wm = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    cfg = json.loads((base / "config.json").read_text())
    plan = json.loads(Path(a.plan).read_text())[a.plan_key]["formats"]
    quant = dict(cfg["quantization"]); outliers = {}
    # (name, file, source key, reshape) — NO array references are held between shards (a version that captured
    # arrays in closures kept every evaluated tensor resident: memguard trip at 70 GB)
    tensors: list[tuple[str, Path, str, tuple | None]] = []
    mx.set_cache_limit(2 * 2**30)

    def load(path: Path):
        return mx.load(str(path))

    # non-expert tensors: everything in the base bundle except routed switch_mlp projections
    for k in sorted(wm):
        if ".mlp.switch_mlp." not in k:
            tensors.append((k, base / wm[k], k, None))
    counts = {"jt": 0, "pick": 0, "base": 0}
    for L in range(1, 48):
        jf, pf = jtd / f"L{L:02d}.safetensors", pkd / f"L{L:02d}.safetensors"
        jsf = load(jf) if jf.exists() else {}
        psf = load(pf) if pf.exists() else {}
        for p in ("gate", "up", "down"):
            m = f"model.layers.{L}.mlp.switch_mlp.{p}_proj"; grp = "gate_up" if p != "down" else "down"; want = plan[f"{L}:{grp}"]
            if m + ".jt_packed" in jsf:
                kb = np.array(jsf[m + ".jt_kbits"]).reshape(-1); assert (kb == kb[0]).all(), f"{m}: per-expert widths"
                K = float(kb[0]); K = int(K) if K.is_integer() else K
                assert want == f"jt{K}", (m, want, K)
                scale = jsf[m + ".jt_scale"]; E, N = scale.shape; W = VL.words_per_row(PROJ_KIN[p], 12, K)
                flat = jsf[m + ".jt_packed"]; assert flat.size == E * N * W, (m, flat.size, E * N * W)
                oc = jsf[m + ".jt_ocol"]
                tensors.append((m + ".jt_packed", jf, m + ".jt_packed", (E, N, W)))
                for s in ("jt_scale", "jt_su", "jt_ocol", "jt_wout") + (("jt_cs",) if p == "down" else ()):
                    tensors.append((f"{m}.{s}", jf, f"{m}.{s}", None))
                quant[m] = {"mode": "jangt", "kbits": K}
                outliers[m] = int(oc.shape[-1]) if int(mx.max(oc).item()) >= 0 else 0
                counts["jt"] += 1
                continue
            src = psf if (m + ".tq2_packed") in psf else None
            for s in ("tq2_packed", "tq2_scales"):
                key = f"{m}.{s}"
                tensors.append((key, pf if src is not None else base / wm[key], key, None))
            pk = (psf if src is not None else load(base / wm[m + ".tq2_packed"]))[m + ".tq2_packed"]
            bits = int(pk.shape[-1]) * 32 // PROJ_KIN[p]
            assert want == f"jh{bits}", (m, want, bits)
            quant[m] = {"mode": "jangtq2", "bits": bits, "rotation": "hadamard32"}
            counts["pick" if src is not None else "base"] += 1
    print(f"routed projections: {counts}", flush=True)
    jsf = psf = None

    # stream into staging shards, then aligned final shards
    limit = int(a.shard_gib * 2**30); shards, cur, cur_b = [], {}, 0
    def flush():
        nonlocal cur, cur_b
        if cur:
            i = len(shards) + 1; p_ = stage / f"part-{i:05d}.safetensors"
            mx.eval(list(cur.values())); mx.save_safetensors(str(p_), cur); shards.append((p_, list(cur))); cur, cur_b = {}, 0
            print(f"  staged shard {i} ({p_.stat().st_size / 2**30:.2f} GiB)", flush=True)
    open_files = {}
    for name, f, key, shape in tensors:
        if f not in open_files:
            if len(open_files) > 3:
                open_files.pop(next(iter(open_files)))
            open_files[f] = load(f)
        t = open_files[f][key]
        if shape is not None:
            t = t.reshape(*shape)
        cur[name] = t; cur_b += t.nbytes
        if cur_b >= limit:
            flush(); open_files.clear(); mx.clear_cache()
    flush(); open_files.clear()
    n = len(shards); weight_map = {}; total = 0
    for i, (p_, names) in enumerate(shards, 1):
        dst = out / f"model-{i:05d}-of-{n:05d}.safetensors"
        rewrite_aligned_safetensors(p_, dst); verify_safetensors_alignment(dst); p_.unlink()
        total += dst.stat().st_size
        for k in names:
            weight_map[k] = dst.name
    stage.rmdir()
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=1))
    cfg["quantization"] = quant
    cfg["jangt"] = {"version": 1, "code": "v2_halfbits", "state_bits": 12, "rotation": "hadamard128-signs",
                    "outlier_columns": outliers}
    if a.drafter:
        cfg["jangt"]["drafter"] = "dspark/"
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    jc = json.loads((base / "jang_config.json").read_text())
    jc["format_name"] = "JANGT (mixed JANGT/JANGH experts)"; jc["jangt"] = cfg["jangt"]
    (out / "jang_config.json").write_text(json.dumps(jc, indent=1))
    for f in COPY_FILES:
        if (base / f).exists():
            shutil.copy2(base / f, out / f)
    if not a.drafter:
        print(f"DONE: {n} shards, {total / 2**30:.3f} GiB weights (no drafter) -> {out}", flush=True)
        return
    dd = out / "dspark"; dd.mkdir()
    for f in Path(a.drafter).iterdir():
        if f.is_file() and not f.name.startswith("."):
            if f.suffix == ".safetensors":
                # a quantized drafter written by mx.save_safetensors is NOT aligned (U32 tensors at odd offsets): the
                # vMLX loader then repairs it on disk at first load (2026-10-09). Write it aligned here instead.
                rewrite_aligned_safetensors(f, dd / f.name); verify_safetensors_alignment(dd / f.name)
            else:
                shutil.copy2(f, dd / f.name)
    print(f"DONE: {n} shards, {total / 2**30:.3f} GiB weights + drafter {sum(f.stat().st_size for f in dd.iterdir()) / 2**30:.3f} GiB -> {out}", flush=True)


if __name__ == "__main__":
    main()
