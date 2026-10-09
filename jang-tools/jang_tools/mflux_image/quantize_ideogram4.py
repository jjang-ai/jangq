"""Quantize Ideogram 4 to an mflux bundle (MLX affine, group size 64).

    python -m jang_tools.mflux_image.quantize_ideogram4 --bits 8 --source <hf snapshot dir> --out <bundle dir>

The tokenizer is copied next to the weights so the bundle is self-contained. Ideogram 4 is prompted with a JSON
caption; plain-text prompts can return the model's learned "blocked by safety filter" card, so test with a caption.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bits", type=int, required=True, choices=(3, 4, 5, 6, 8))
    ap.add_argument("--source", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from mflux.models.ideogram4.variants.txt2img.ideogram4 import Ideogram4

    src, out = Path(a.source), Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    t = time.time()
    model = Ideogram4(quantize=a.bits, model_path=str(src))
    model.save_model(str(out))
    if (src / "tokenizer").is_dir() and not (out / "tokenizer").exists():
        shutil.copytree(src / "tokenizer", out / "tokenizer")
    print(f"SAVED {out} bits={model.bits} in {time.time() - t:.1f} s", flush=True)


if __name__ == "__main__":
    main()
