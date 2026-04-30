"""Image / VL preprocessing for JANG models.

Pixtral path covers Mistral 3.5 (mistral3) and any pixtral-style VL.
generic.py is for Qwen-style native-resolution + spatial_merge schemes.
audio.py + video.py are stubs for when we hit those modalities (not yet
needed by Mistral 3.5 / Laguna XS.2).
"""

from .pixtral import PixtralImageProcessor, encode_image_pixtral
from .generic import pad_image_to_grid, place_image_token

__all__ = [
    "PixtralImageProcessor",
    "encode_image_pixtral",
    "pad_image_to_grid",
    "place_image_token",
]
