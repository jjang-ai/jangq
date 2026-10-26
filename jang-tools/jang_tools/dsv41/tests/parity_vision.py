"""Numeric parity: MLX vision tower (ViT+Aligner) vs the release torch reference.

Loads the real BF16 vision weights from a JANG bundle into both the release torch modules and the
MLX port, runs identical random patches through, and reports max/mean relative error on the ViT
output and the final aligner output. This validates the 2D RoPE, the attention, the SwiGLU MLP and
the 3x3 unfold merge independently of the main backbone forward.

  python -m jang_tools.dsv41.tests.parity_vision --bundle <bundle> \
      --ref ~/models/DeepSeek-V4.1-Flash-1017/inference
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--ref", required=True, help="release inference/ dir with vision.py")
    ap.add_argument("--nh", type=int, default=4)
    ap.add_argument("--nw", type=int, default=6)
    a = ap.parse_args()

    import torch

    bundle = Path(a.bundle).expanduser()
    cfg = json.loads((bundle / "config.json").read_text())
    from jang_tools.dsv41.vision import ViT as MLViT, Aligner as MLAligner, VisionArgs

    va = VisionArgs.from_config(cfg)

    # locate + load the vision/aligner BF16 tensors from the bundle
    index = json.loads((bundle / "model.safetensors.index.json").read_text())["weight_map"]
    files = sorted({f for t, f in index.items() if t.startswith(("vision.", "aligner.", "image_"))})
    W = {}
    for f in files:
        W.update({k: v.astype(mx.float32) for k, v in mx.load(str(bundle / f)).items()
                  if k.startswith(("vision.", "aligner.", "image_"))})  # f32: clean parity (bf16 values exact)
    print(f"loaded {len(W)} vision tensors from {files}")

    # --- MLX side ---
    ml_vit, ml_al = MLViT(va), MLAligner(va)
    ml_vit.load_weights([(k[len("vision."):], v) for k, v in W.items() if k.startswith("vision.")], strict=True)
    ml_al.load_weights([(k[len("aligner."):], v) for k, v in W.items() if k.startswith("aligner.")], strict=True)
    mx.eval(ml_vit.parameters(), ml_al.parameters())

    # --- torch reference side ---
    sys.path.insert(0, str(Path(a.ref).expanduser()))
    import vision as refv  # release inference/vision.py

    rargs = SimpleNamespace(
        vision_dim=va.dim, vision_n_heads=va.n_heads, vision_inter_dim=va.inter_dim,
        vision_patch_size=va.patch_size, vision_rope_theta=va.rope_theta,
        vision_n_layers=va.n_layers, vision_downsample_ratio=va.downsample_ratio, dim=va.text_dim,
    )
    ref_vit, ref_al = refv.ViT(rargs), refv.Aligner(rargs)

    def to_t(v):
        return torch.from_numpy(np.array(v.astype(mx.float32)))  # f32

    sd_vit = {k[len("vision."):]: to_t(v) for k, v in W.items() if k.startswith("vision.")}
    sd_al = {k[len("aligner."):]: to_t(v) for k, v in W.items() if k.startswith("aligner.")}
    ref_vit.load_state_dict(sd_vit, strict=True)
    ref_al.load_state_dict(sd_al, strict=True)
    ref_vit.eval(); ref_al.eval()

    # identical random patches
    nh, nw = a.nh, a.nw
    p = va.patch_size
    rng = np.random.default_rng(0)
    patches = rng.standard_normal((nh * nw, 3, p, p)).astype(np.float32) * 0.5

    with torch.inference_mode():
        r_vit = ref_vit(torch.from_numpy(patches), nh, nw)
        r_al = ref_al(r_vit, nh, nw)
    m_vit = ml_vit(mx.array(patches), nh, nw)
    m_al = ml_al(m_vit, nh, nw)
    mx.eval(m_vit, m_al)

    def rel(ref_t, ml_a, name):
        r = ref_t.float().numpy()
        m = np.array(ml_a.astype(mx.float32))
        assert r.shape == m.shape, f"{name} shape {r.shape} vs {m.shape}"
        num = np.linalg.norm((r - m).ravel())
        den = np.linalg.norm(r.ravel()) + 1e-9
        mx_abs = np.abs(r - m).max()
        print(f"{name:12s} shape={r.shape} rel-L2={num/den:.3e} max-abs={mx_abs:.3e}")
        return num / den

    print(f"\ngrid n_vit={nh}x{nw} -> {nh*nw} patches; aligner -> {m_al.shape[0]} llm tokens")
    e1 = rel(r_vit, m_vit, "ViT out")
    e2 = rel(r_al, m_al, "Aligner out")
    ok = e1 < 2e-2 and e2 < 2e-2
    print("\nPARITY", "PASS" if ok else "FAIL", "(bf16 tolerance 2e-2 rel-L2)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
