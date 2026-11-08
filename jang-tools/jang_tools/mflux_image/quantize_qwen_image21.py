"""Quantize Qwen-Image-2.1 or Qwen-Image-2.1-Turbo to an mflux bundle (MLX affine, group size 64).

    python -m jang_tools.mflux_image.quantize_qwen_image21 --bits 8 --source <hf snapshot dir> --out <bundle dir>

Saves through mflux's QwenImage21Edit, which keeps the Qwen3-VL vision tower: one bundle serves text-to-image AND
reference editing. QwenImage21Edit.save_model copies model_index.json and scheduler/ with the weights, so a Turbo
source keeps its fixed `sample_sigmas` grid (asserted below). The Turbo source needs the scheduler patch in
patches/ only for generation, not for quantization.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

LAYOUT = ("model_index.json", "transformer", "text_encoder", "vae", "scheduler", "processor/tokenizer.json",
          "processor/preprocessor_config.json")


def sample_sigmas(model_dir: Path):
    index = model_dir / "model_index.json"
    return json.loads(index.read_text()).get("sample_sigmas") if index.exists() else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bits", type=int, required=True, choices=(3, 4, 5, 6, 8))
    ap.add_argument("--source", required=True, help="Qwen/Qwen-Image-2.1 or Qwen/Qwen-Image-2.1-Turbo snapshot")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit

    src, out = Path(a.source), Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    want = sample_sigmas(src)
    t = time.time()
    model = QwenImage21Edit(quantize=a.bits, model_path=str(src))
    print(f"loaded + quantized in {time.time() - t:.1f} s, bits={model.bits}", flush=True)
    model.save_model(str(out))
    missing = [p for p in LAYOUT if not (out / p).exists()]
    if missing:
        raise SystemExit(f"saved bundle is missing {missing}")
    got = sample_sigmas(out)
    if got != want:
        raise SystemExit(f"model_index.json sample_sigmas changed on save: source {want}, bundle {got}")
    kind = f"Turbo, {len(got)}-step grid" if got else "base, shifted 40-step schedule"
    print(f"SAVED {out} ({kind}) in {time.time() - t:.1f} s", flush=True)


if __name__ == "__main__":
    main()
