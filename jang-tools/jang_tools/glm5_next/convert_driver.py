"""Segmented convert driver for glm5_next (fresh subprocess per tensor range;
merges segments, quant entries, and stamps config). Same rationale as the
qwen4_exp driver: monolithic Metal sessions die; segments replay clean.

  python -m jang_tools.glm5_next.convert_driver --model <src> --out <bundle> \
      --bit-map <map.json> [--awq-scales s.safetensors] [--imatrix diag] \
      [--seg-size 120]
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def count_tensors(model_dir, no_mtp=False) -> int:
    import glob

    import mlx.core as mx

    from .convert import sanitize_bundle

    weights = {}
    for f in sorted(glob.glob(str(Path(model_dir) / "model-*.safetensors"))):
        weights.update(mx.load(f))
    return len(sanitize_bundle(weights, no_mtp=no_mtp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bit-map", required=True)
    ap.add_argument("--awq-scales", default=None)
    ap.add_argument("--imatrix", default=None)
    ap.add_argument("--seg-size", type=int, default=120)
    ap.add_argument("--no-mtp", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = count_tensors(args.model, args.no_mtp)
    print(f"{n} tensors → segments of {args.seg_size}")

    seg_names = []
    t0 = time.time()
    for start in range(0, n, args.seg_size):
        end = min(start + args.seg_size, n)
        seg = f"seg_{start:05d}"
        seg_names.append(seg)
        if (out_dir / seg / "map.json").exists():
            print(f"{seg}: already done, skipping")
            continue
        cmd = [sys.executable, "-u", "-m", "jang_tools.glm5_next.convert",
               "--model", args.model, "--out", str(out_dir),
               "--bit-map", args.bit_map,
               "--tensor-start", str(start), "--tensor-end", str(end),
               "--segment", seg]
        if args.awq_scales:
            cmd += ["--awq-scales", args.awq_scales]
        if args.imatrix:
            cmd += ["--imatrix", args.imatrix]
        if args.no_mtp:
            cmd.append("--no-mtp")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"{seg} FAILED (exit {r.returncode}) — rerun driver to resume")
            raise SystemExit(r.returncode)
        print(f"{seg} done ({(time.time()-t0)/60:.1f} min elapsed)", flush=True)

    weight_map, total, quant_entries = {}, 0, {}
    idx = 0
    for seg in seg_names:
        seg_dir = out_dir / seg
        m = json.loads((seg_dir / "map.json").read_text())
        renames = {}
        for old in sorted(set(m["weight_map"].values())):
            idx += 1
            renames[old] = f"model-{idx:05d}.safetensors"
        for old, new in renames.items():
            shutil.move(str(seg_dir / old), str(out_dir / new))
        for k, v in m["weight_map"].items():
            weight_map[k] = renames[v]
        total += m["total"]
        quant_entries.update(m.get("quant_entries", {}))
        shutil.rmtree(seg_dir)
    nfiles = idx
    final = {}
    for i, old in enumerate(sorted({v for v in weight_map.values()}), 1):
        final[old] = f"model-{i:05d}-of-{nfiles:05d}.safetensors"
    for old, new in final.items():
        shutil.move(str(out_dir / old), str(out_dir / new))
    weight_map = {k: final[v] for k, v in weight_map.items()}
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=1))

    from .convert import _finalize_configs
    _finalize_configs(Path(args.model), out_dir,
                      json.loads(Path(args.bit_map).read_text()),
                      quant_entries, args, bool(args.awq_scales))
    print(f"MERGED: {nfiles} shards, {total/2**30:.2f} GiB "
          f"in {(time.time()-t0)/60:.1f} min → {out_dir}")


if __name__ == "__main__":
    main()
