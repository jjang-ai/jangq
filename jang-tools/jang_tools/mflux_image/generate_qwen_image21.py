"""Reference generation / edit with an mflux Qwen-Image-2.1 or -Turbo bundle (or source snapshot).

    python -m jang_tools.mflux_image.generate_qwen_image21 --model <dir> --out a.png --seed 7 --prompt "..."
    python -m jang_tools.mflux_image.generate_qwen_image21 --model <dir> --out b.png --seed 7 --prompt "..." --ref a.png

A checkpoint whose model_index.json pins `sample_sigmas` (Turbo) samples on that grid: steps = grid length,
guidance 1.0, scheduler qwen21_turbo (needs patches/mflux-03700b9-qwen21-turbo-scheduler.patch). Otherwise the base
model's defaults apply: 40 steps on the shifted linear schedule.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from jang_tools.mflux_image.quantize_qwen_image21 import sample_sigmas


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", required=True); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--width", type=int, default=1024); ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=None); ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--ref", nargs="*", default=[], help="ordered reference images (edit), up to 10")
    ap.add_argument("--bits", type=int, default=None, help="quantize on load (source snapshots only)")
    a = ap.parse_args(argv)
    from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit

    grid = sample_sigmas(Path(a.model))
    kw = {}
    if grid:
        from mflux.models.qwen21.model.qwen21_scheduler import Qwen21TurboScheduler  # noqa: F401  registers qwen21_turbo
        steps = len(grid); kw["scheduler"] = "qwen21_turbo"
        if a.steps not in (None, steps):
            raise SystemExit(f"this checkpoint samples on its fixed {steps}-node grid; drop --steps")
    else:
        steps = a.steps or 40
    t = time.time()
    model = QwenImage21Edit(quantize=a.bits, model_path=a.model)
    print(f"loaded in {time.time() - t:.1f} s", flush=True)
    t = time.time()
    image = model.generate_image(seed=a.seed, prompt=a.prompt, num_inference_steps=steps, guidance=a.guidance,
                                 width=a.width, height=a.height, image_paths=a.ref or None, **kw)
    image.save(a.out)
    print(f"GENERATED {a.out} ({steps} steps) in {time.time() - t:.1f} s", flush=True)


if __name__ == "__main__":
    main()
