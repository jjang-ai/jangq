"""Top-level MiMo-V2.5 VLM model (mlx_vlm-shaped) for MLX.

Design constraint: the tracked module tree exposes text weights at the SAME
paths the bundle uses (``model.*``, ``lm_head.*``, ``visual.*``). This makes
mlx_vlm's per-module quantization overrides (keyed ``model.layers...``) and
``load_weights`` resolve without any key remapping. The ``language_model``
attribute mlx_vlm's generate loop requires is an UNTRACKED shim (plain
object), so no parameters are duplicated.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from jang_tools.mimo_v2 import mlx_model as text_mlx_model

from .config import ModelConfig
from .vision import VisionModel


def _resolve_text_params(config) -> dict:
    if config is None:
        return {}
    if isinstance(config, dict):
        return config
    raw = getattr(config, "raw", None)
    if raw:
        return raw
    if hasattr(config, "__dict__"):
        return {k: v for k, v in vars(config).items() if not k.startswith("_")}
    return {}


class _LanguageModelShim:
    """Plain-object adapter satisfying mlx_vlm's model.language_model API."""

    def __init__(self, owner: "Model"):
        self._owner = owner

    def __call__(self, inputs=None, inputs_embeds=None, cache=None, mask=None, **kwargs):
        return self._owner.text(inputs, inputs_embeds=inputs_embeds, cache=cache)

    @property
    def layers(self):
        return self._owner.text.layers

    def make_cache(self):
        return self._owner.text.make_cache()

    def parameters(self):
        return self._owner.parameters()


class LanguageModel:
    """API placeholder for mlx_vlm module-shape expectations.

    Deliberately NOT an nn.Module and deliberately without ``sanitize`` —
    mlx_vlm would otherwise instantiate it with an empty injected text_config
    just to call sanitize. All real work lives in :class:`Model`.
    """

    def __init__(self, *args, **kwargs):
        raise TypeError("LanguageModel is a placeholder; use Model")


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = getattr(config, "model_type", "mimo_v2")
        params = _resolve_text_params(config)
        if "hybrid_layer_pattern" not in params:
            raise ValueError(
                f"MiMo text config resolution failed: got {type(config).__name__} "
                f"with {len(params)} usable keys (missing hybrid_layer_pattern)"
            )
        args = text_mlx_model.ModelArgs.from_dict(params)
        text = text_mlx_model.Model(args)
        # Track text submodules at bundle-native paths.
        self.model = text.model
        self.lm_head = text.lm_head
        self.visual = VisionModel(config.vision_config)
        # Untracked references (underscore attrs are not registered by mlx.nn).
        self._text = text
        self._lm_shim = _LanguageModelShim(self)

    # --- mlx_vlm API surface -------------------------------------------------

    @property
    def text(self):
        return self._text

    @property
    def language_model(self):
        return self._lm_shim

    @property
    def layers(self):
        return self._text.layers

    def make_cache(self):
        return self._text.make_cache()

    def sync_text_refs(self):
        """Re-point the text wrapper at the tracked (possibly re-quantized) modules.

        Module replacement during quantization swaps direct children on THIS
        module's tracked tree; the untracked ``_text`` wrapper keeps stale
        references to the originals (notably ``lm_head``), which silently
        produces garbage logits if not re-synced.
        """
        self._text.model = self.model
        self._text.lm_head = self.lm_head

    # --- multimodal merge -----------------------------------------------------

    def _merge_modal(self, input_ids: mx.array, embeds: mx.array, token_id: int, feats: mx.array) -> mx.array:
        mask = input_ids == token_id
        n_slots = int(mask.sum().item())
        if n_slots == 0:
            return embeds
        if feats.shape[0] != n_slots:
            raise ValueError(
                f"modal embedding count mismatch for token_id={token_id}: "
                f"{n_slots} placeholders vs {feats.shape[0]} features"
            )
        flat_mask = mask.reshape(-1)
        positions = mx.array([i for i, m in enumerate(flat_mask.tolist()) if m])
        flat = embeds.reshape(-1, embeds.shape[-1])
        flat[positions] = feats.astype(flat.dtype)
        return flat.reshape(embeds.shape)

    def get_input_embeddings(
        self,
        input_ids: mx.array = None,
        pixel_values: mx.array = None,
        image_grid_thw: mx.array = None,
        pixel_values_videos: mx.array = None,
        video_grid_thw: mx.array = None,
        **kwargs,
    ) -> mx.array:
        embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            feats = self.visual(pixel_values, image_grid_thw)
            embeds = self._merge_modal(input_ids, embeds, self.config.image_token_id, feats)
        if pixel_values_videos is not None:
            feats = self.visual(pixel_values_videos, video_grid_thw)
            embeds = self._merge_modal(input_ids, embeds, self.config.video_token_id, feats)
        return embeds

    def __call__(self, input_ids: mx.array, pixel_values: mx.array = None, mask=None, cache=None, **kwargs):
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        inputs_embeds = None
        if pixel_values is not None or pixel_values_videos is not None:
            inputs_embeds = self.get_input_embeddings(
                input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )
        return self._text(input_ids, inputs_embeds=inputs_embeds, cache=cache)

    # --- weights ---------------------------------------------------------------

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        text_weights = {}
        vision_weights = {}
        for k, v in weights.items():
            if k.startswith("visual."):
                vision_weights[k] = v
            elif k.startswith(("audio_encoder.", "speech_embeddings.", "model.mtp.")):
                continue  # preserved in bundles, unwired in this runtime
            else:
                text_weights[k] = v
        out = dict(self._text.sanitize(text_weights))
        out.update(self.visual.sanitize(vision_weights))
        return out
