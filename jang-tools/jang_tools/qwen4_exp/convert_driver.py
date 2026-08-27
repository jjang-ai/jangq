"""Segmented convert driver: runs convert.py in FRESH subprocesses per tensor
range (the monolithic build gets OOM-killed ~tensor 520 while every range
replays clean in a new process), then merges segments into the final bundle.

  python -m jang_tools.qwen4_exp.convert_driver --model <src> --out <bundle> \
      --bit-map <map.json> [--awq-scales s.safetensors] [--imatrix diag] \
      [--mtp] [--seg-size 250]
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def count_tensors(model_dir, mtp: bool) -> int:
    import glob

    import mlx.core as mx

    from .convert import sanitize_stream

    weights = {}
    for f in sorted(glob.glob(str(Path(model_dir) / "model-*.safetensors"))):
        weights.update(mx.load(f))
    return len(sanitize_stream(weights, keep_mtp=mtp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bit-map", required=True)
    ap.add_argument("--awq-scales", default=None)
    ap.add_argument("--imatrix", default=None)
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--seg-size", type=int, default=250)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = count_tensors(args.model, args.mtp)
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
        cmd = [sys.executable, "-u", "-m", "jang_tools.qwen4_exp.convert",
               "--model", args.model, "--out", str(out_dir),
               "--bit-map", args.bit_map,
               "--tensor-start", str(start), "--tensor-end", str(end),
               "--segment", seg]
        if args.awq_scales:
            cmd += ["--awq-scales", args.awq_scales]
        if args.imatrix:
            cmd += ["--imatrix", args.imatrix]
        if args.mtp:
            cmd.append("--mtp")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"{seg} FAILED (exit {r.returncode}) — rerun driver to resume")
            raise SystemExit(r.returncode)
        print(f"{seg} done ({(time.time()-t0)/60:.1f} min elapsed)", flush=True)

    # merge: renumber shards, concat weight maps
    weight_map, total = {}, 0
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
        shutil.rmtree(seg_dir)
    final = {}
    nfiles = idx
    for i, old in enumerate(sorted({v for v in weight_map.values()}), 1):
        final[old] = f"model-{i:05d}-of-{nfiles:05d}.safetensors"
    for old, new in final.items():
        shutil.move(str(out_dir / old), str(out_dir / new))
    weight_map = {k: final[v] for k, v in weight_map.items()}
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=1))

    # support files + jang_config stamp
    model_dir = Path(args.model)
    for f in ("config.json", "generation_config.json", "tokenizer.json",
              "tokenizer_config.json", "vocab.json", "merges.txt",
              "chat_template.jinja", "preprocessor_config.json",
              "video_preprocessor_config.json", "LICENSE"):
        if (model_dir / f).exists():
            shutil.copy(model_dir / f, out_dir / f)
    cfg = json.loads((out_dir / "config.json").read_text())
    bit_map = json.loads(Path(args.bit_map).read_text())
    cfg["jang_config"] = {
        "format": "jang_v2",
        "family": "qwen4_exp",
        "norm_convention": "runtime_plus1_applied",
        "bit_map": bit_map,
        "created": time.strftime("%Y-%m-%d"),
        "quantization": {
            "calibrated": bool(args.imatrix),
            "imatrix": Path(args.imatrix).name if args.imatrix else None,
            "imatrix_refit": bool(args.imatrix),
            "awq_folded": bool(args.awq_scales),
            "awq_scales": Path(args.awq_scales).name if args.awq_scales else None,
            "hessian_allocation": Path(args.bit_map).name,
            "gptq": False,
        },
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    print(f"MERGED: {nfiles} shards, {total/2**30:.2f} GiB "
          f"in {(time.time()-t0)/60:.1f} min → {out_dir}")


if __name__ == "__main__":
    main()
