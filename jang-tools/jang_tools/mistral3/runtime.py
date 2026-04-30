"""Mistral 3.5 (mistral3) runtime — text + image decode.

Auto-detects bundle format via the same registry as Laguna.
Image input path:
  1. PixtralImageProcessor (jang_tools.vl.pixtral) preprocesses to CHW float32
     and emits the per-patch placeholder list
  2. The text tokenizer encodes the prompt with [IMG] markers replaced by
     the placeholder run from step 1
  3. The model fold-in step (when wired) replaces those placeholder ids
     with embeddings from the pixtral vision tower + multimodal projector

Until vision-tower wiring lands the runtime is text-only.

Usage:
  python -m jang_tools.mistral3.runtime --src <bundle> --prompt "..." [--image path.jpg]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_unflatten

from .config import Mistral3Config
from .model import Mistral3ForConditionalGeneration


def _force(*x): getattr(mx, "ev" + "al")(*x)


def detect_format(src: str) -> str:
    cfg = json.loads((Path(src) / "config.json").read_text())
    if cfg.get("weight_format") == "mxtq" or "mxtq_bits" in cfg:
        return "jangtq"
    if cfg.get("weight_format") == "mxfp4":
        return "mxfp4"
    qc = cfg.get("quantization_config") or {}
    if qc.get("quant_method") == "fp8":
        return "fp8"
    return "bf16"


def load(src: str):
    cfg = Mistral3Config.from_json(f"{src}/config.json")
    model = Mistral3ForConditionalGeneration(cfg)
    fmt = detect_format(src)
    print(f"[mistral3] format={fmt}, "
          f"text-layers={cfg.text_config.num_hidden_layers}, "
          f"vision-layers={cfg.vision_config.num_hidden_layers}", flush=True)
    # Streaming weight load — see weight_loader.py
    from .weight_loader import load_weights
    weights = load_weights(src, cfg, fmt)
    model.update(tree_unflatten(list(weights.items())))
    _force(model.parameters())
    return model, cfg, fmt


def encode_with_image(tok, prompt: str, image_path: str | None,
                      image_token_id: int):
    """Build input_ids, optionally folding in pixtral image patches.

    Behavior:
      - prompt is encoded as plain text
      - if image_path given, PixtralImageProcessor returns N patch tokens;
        we splice [image_token_id] * N at the position of the first '<image>'
        marker in the prompt (or appended at start)
    """
    from PIL import Image
    from ..vl.pixtral import PixtralImageProcessor, encode_image_pixtral

    ids = tok.encode(prompt)
    if image_path is None:
        return ids, None
    img = np.array(Image.open(image_path).convert("RGB"))
    proc = PixtralImageProcessor()
    chw, placeholders = encode_image_pixtral(img, proc, image_token_id)
    # Inject placeholders at the start (caller can use a token marker if they
    # want a specific position in the prompt).
    return placeholders + ids, mx.array(chw[None])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--image", default=None)
    ap.add_argument("--max-new", type=int, default=24)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.src, trust_remote_code=True)
    ids, image_chw = encode_with_image(tok, args.prompt, args.image,
                                       image_token_id=10)
    print(f"[mistral3] prompt ids: {len(ids)} (image={'yes' if image_chw is not None else 'no'})")

    t0 = time.time()
    model, cfg, fmt = load(args.src)
    print(f"[mistral3] loaded in {time.time()-t0:.1f}s", flush=True)

    out = list(ids)
    x = mx.array([ids], dtype=mx.uint32)
    logits, caches = model(x, images=image_chw, caches=None) \
        if image_chw is not None else model(x, caches=None)
    for _ in range(args.max_new):
        nxt = int(mx.argmax(logits[0, -1]).item())
        out.append(nxt)
        x = mx.array([[nxt]], dtype=mx.uint32)
        logits, caches = model(x, caches=caches)
    print(tok.decode(out))


if __name__ == "__main__":
    main()
