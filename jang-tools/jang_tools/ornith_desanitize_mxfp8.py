"""Reverse a baked-in `qwen3_5.sanitize()` in an MXFP8 bundle.

Created by Jinho Jang (eric@jangq.ai) — 2026-08-19.

`convert_qwen35_mxfp8` (via `convert_qwen35_mxfp4`) loads the source through
mlx_vlm, which applies `qwen3_5.sanitize()` — **+1.0 to every 1-D norm** and
**moveaxis(2,1) on conv1d** — and then saves the in-memory params without
reversing either. The next `mlx_vlm.load()` applies both a SECOND time, so the
runtime sees norms at source+2 and conv1d transposed back.

Measured on Ornith-1.5-9B-MXFP8 before this fix:

    65 norms   mean(bundle - source) = +1.0000  (exactly)
    conv1d     (8192, 4, 1)  vs source (8192, 1, 4)

versus the D-lane JANG bundles built by `qwen36_build`, which reverse it:

    65 norms   mean(bundle - source) = -0.0000
    conv1d     (8192, 1, 4)  ✓

Symptom is the trap-#2 signature: every Linear dequantizes correctly, yet
generation emits invalid UTF-8 and the detokenizer raises UnicodeDecodeError.

This is written as a standalone post-pass rather than a change to
`convert_qwen35_mxfp4`, because that converter is shared with other model
families whose runtimes may expect the sanitized convention; fixing it there
would silently re-convention every bundle built from it. The JANG contract is
that bundles store the SOURCE convention and the runtime owns the shift.

Idempotent: it compares against the source and only subtracts where the norm
is actually ~+1.0 off, so running twice is a no-op.

    python -m jang_tools.ornith_desanitize_mxfp8 <bundle> --source <src>
"""
from __future__ import annotations

import glob
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx

NORM_SUFFIXES = (".input_layernorm.weight", ".post_attention_layernorm.weight",
                 "model.norm.weight", ".q_norm.weight", ".k_norm.weight")

TOL = 0.05   # how close to +1.0 the offset must be to count as sanitized


def _src_key(bundle_key: str) -> str:
    return bundle_key.replace("language_model.model.", "model.language_model.")


def main(argv) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    bundle = Path(argv[1]).expanduser()
    source = None
    for i, a in enumerate(argv):
        if a == "--source":
            source = Path(argv[i + 1]).expanduser()
    if source is None:
        print("  --source <original bf16 dir> is required (used to verify the "
              "offset is really +1 before touching anything)")
        return 1

    src = {}
    for f in glob.glob(f"{source}/*.safetensors"):
        src.update(mx.load(f))

    n_norm = n_conv = 0
    skipped_norm = 0
    for f in sorted(glob.glob(f"{bundle}/*.safetensors")):
        w = mx.load(f)
        mx.eval(list(w.values()))          # force before rewriting the mmap
        changed = False
        for k, v in list(w.items()):
            if not hasattr(v, "ndim"):
                continue
            if any(k.endswith(s) for s in NORM_SUFFIXES) and v.ndim == 1:
                s = src.get(_src_key(k))
                if s is None or s.shape != v.shape:
                    continue
                off = float(mx.mean(v.astype(mx.float32) - s.astype(mx.float32)))
                if abs(off - 1.0) <= TOL:
                    w[k] = v - 1.0
                    n_norm += 1
                    changed = True
                else:
                    skipped_norm += 1       # already correct, or not sanitized
            elif "conv1d.weight" in k and v.ndim == 3 and v.shape[-1] == 1:
                w[k] = v.moveaxis(1, 2)
                n_conv += 1
                changed = True
        if not changed:
            continue
        mx.eval(list(w.values()))
        # lazy-mmap self-clobber: never rewrite a shard in place while loaded
        # arrays still reference its mapping. temp + atomic rename.
        # mx.save_safetensors enforces the .safetensors extension, so the temp
        # name must keep it rather than ending in .tmp.
        tmp = f.replace(".safetensors", ".tmp.safetensors")
        mx.save_safetensors(tmp, w, metadata={"format": "pt"})
        shutil.move(tmp, f)
        print(f"  rewrote {Path(f).name}", flush=True)

    print(f"\n  de-sanitized: {n_norm} norms (-1.0), {n_conv} conv1d "
          f"(moveaxis back); {skipped_norm} norms already in source convention")

    cfg_p = bundle / "config.json"
    if cfg_p.exists() and (n_norm or n_conv):
        cfg = json.loads(cfg_p.read_text())
        cfg["desanitized"] = {"norms": n_norm, "conv1d": n_conv,
                              "note": "reversed baked-in qwen3_5.sanitize(); "
                                      "bundle stores SOURCE convention"}
        cfg_p.write_text(json.dumps(cfg, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
