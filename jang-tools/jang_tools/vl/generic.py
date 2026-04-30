"""Helpers shared by all VL processors."""

from __future__ import annotations

import numpy as np


def pad_image_to_grid(img: np.ndarray, ps: int, fill: float = 0.0) -> np.ndarray:
    """Pad to (Hpad, Wpad) so both are multiples of `ps`."""
    H, W = img.shape[-2:]
    Hpad = (H + ps - 1) // ps * ps
    Wpad = (W + ps - 1) // ps * ps
    if (Hpad, Wpad) == (H, W):
        return img
    pad = ((0, 0),) * (img.ndim - 2) + ((0, Hpad - H), (0, Wpad - W))
    return np.pad(img, pad, constant_values=fill)


def place_image_token(input_ids: list[int], image_token_id: int,
                      image_tokens: list[int]) -> list[int]:
    """Replace the first occurrence of `image_token_id` in `input_ids`
    with the per-patch placeholder list."""
    if image_token_id not in input_ids:
        return list(input_ids) + image_tokens
    out = []
    inserted = False
    for t in input_ids:
        if t == image_token_id and not inserted:
            out.extend(image_tokens)
            inserted = True
        else:
            out.append(t)
    return out
