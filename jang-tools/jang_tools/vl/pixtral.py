"""Pixtral image preprocessor (used by Mistral 3.5 and any pixtral VL).

Per upstream config (Mistral-Medium-3.5-128B):
    vision_config.model_type = "pixtral"
    image_size = 1540, patch_size = 14, spatial_merge_size = 2
    hidden_size = 1664, num_hidden_layers = 48, num_attention_heads = 16

Pixtral takes variable-aspect images: it pads each image to (H', W') where
H', W' are multiples of patch_size, then emits H'*W'/(patch_size**2) tokens
that get spatially merged by `spatial_merge_size` before joining the LM.

Token layout for the LM (single image):
    <BOI> [N_tokens placeholders, one per merged patch] <EOI>

`image_token_index = 10` is the single placeholder ID inserted by the LM
processor; the vision tower fills in their embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PixtralImageProcessor:
    image_size: int = 1540
    patch_size: int = 14
    spatial_merge_size: int = 2
    image_mean: tuple = (0.48145466, 0.4578275, 0.40821073)
    image_std: tuple = (0.26862954, 0.26130258, 0.27577711)
    rescale_factor: float = 1 / 255.0

    def preprocess(self, img: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """img: HxWx3 uint8 -> CxHxW float32 normalized + (H_patch, W_patch).

        Variable aspect: shorter side scales to fit; longer side is bucketed
        into a multiple of patch_size up to image_size. Aspect preserved.
        """
        H, W = img.shape[:2]
        scale = self.image_size / max(H, W)
        new_H = max(1, int(round(H * scale)))
        new_W = max(1, int(round(W * scale)))
        # Round up to patch grid
        ps = self.patch_size
        H_ = (new_H + ps - 1) // ps * ps
        W_ = (new_W + ps - 1) // ps * ps
        out = np.zeros((H_, W_, 3), dtype=np.float32)
        # Naive nearest resize; production uses PIL.LANCZOS
        ys = (np.arange(new_H) * H / new_H).astype(np.int64)
        xs = (np.arange(new_W) * W / new_W).astype(np.int64)
        out[:new_H, :new_W] = img[ys][:, xs].astype(np.float32)
        out = out * self.rescale_factor
        mean = np.array(self.image_mean, dtype=np.float32)
        std = np.array(self.image_std, dtype=np.float32)
        out = (out - mean) / std
        out = out.transpose(2, 0, 1)  # HWC -> CHW
        return out, (H_ // ps, W_ // ps)

    def num_image_tokens(self, h_patch: int, w_patch: int) -> int:
        s = self.spatial_merge_size
        return (h_patch // s) * (w_patch // s)


def encode_image_pixtral(img: np.ndarray,
                         processor: Optional[PixtralImageProcessor] = None,
                         image_token_id: int = 10) -> tuple[np.ndarray, list[int]]:
    """Returns (CHW float32 array, [image_token_id] * num_tokens)."""
    p = processor or PixtralImageProcessor()
    chw, (hp, wp) = p.preprocess(img)
    n = p.num_image_tokens(hp, wp)
    return chw, [image_token_id] * n
