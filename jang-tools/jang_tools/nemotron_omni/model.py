"""NemotronHOmni — native MLX multimodal wrapper.

Combines:
  - LLM:               mlx_lm.models.nemotron_h.Model (loaded via mlx_lm.load
                       or jang_tools.load_jangtq.load_jangtq_model)
  - Vision tower:      jang_tools.nemotron_omni.radio.RADIOVisionModel
  - Vision projector:  jang_tools.nemotron_omni.projectors.VisionMLPProjector
  - Audio encoder:     jang_tools.nemotron_omni.parakeet.ParakeetEncoder
  - Sound projector:   jang_tools.nemotron_omni.projectors.SoundProjector
  - Image preprocess:  jang_tools.nemotron_omni.image_processor.preprocess_images
  - Audio preprocess:  jang_tools.nemotron_omni.audio_features.extract_mel_features

The orchestrator builds the user prompt with multimodal placeholder tokens,
runs encoders, injects embeddings at placeholder positions, prefills via
inputs_embeds, then decodes token-by-token using the LLM's hybrid
Mamba+Attention cache.

Stage 2 native MLX. Drop-in replacement for `OmniChat`/`OmniSession` once
parity is fully validated. Today's status:
  - Vision tower: WORKING (matches PyTorch within bf16 noise)
  - Projectors: WORKING (small modules, trivially correct)
  - Image preprocess: WORKING (numpy NVLM tiling)
  - Audio mel: WORKING (numpy STFT + librosa filterbank)
  - Parakeet body: APPROXIMATE (simplified rel-pos attention; close but not bit-exact)

Use `OmniChat` (jang_tools.nemotron_omni_chat) for production today; this
class is the future-state native MLX path.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

import mlx.core as mx
import numpy as np
from PIL import Image

from .radio import RADIOVisionModel, map_radio_weights, pixel_shuffle
from .parakeet import ParakeetEncoder, map_parakeet_weights
from .projectors import (
    VisionMLPProjector, SoundProjector,
    map_mlp1_weights, map_sound_projection_weights,
)
from .image_processor import preprocess_images
from .audio_features import extract_mel_features


def _load_safetensors_with_prefix(
    bundle_path: Path, prefix: str,
) -> dict[str, mx.array]:
    """Load all tensors with a given prefix from a sharded bundle."""
    from safetensors import safe_open
    idx_path = bundle_path / "model.safetensors.index.json"
    idx = json.loads(idx_path.read_text())["weight_map"]
    needed_shards = sorted({v for k, v in idx.items() if k.startswith(prefix)})
    out: dict[str, mx.array] = {}
    for shard in needed_shards:
        with safe_open(str(bundle_path / shard), framework="mlx") as f:
            for k in f.keys():
                if k.startswith(prefix):
                    out[k] = f.get_tensor(k)
    return out


def _cast_to_fp32(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    return {k: v.astype(mx.float32) if v.dtype == mx.bfloat16 else v
            for k, v in weights.items()}


def _cast_to(weights: dict[str, mx.array], dtype: mx.Dtype) -> dict[str, mx.array]:
    """Cast weights to `dtype`. Honors caller's `dtype` arg (was previously
    ignored by `_cast_to_fp32` which always upcast bf16→fp32, doubling encoder
    RAM regardless of constructor `dtype`)."""
    return {k: (v.astype(dtype) if v.dtype != dtype else v)
            for k, v in weights.items()}


class NemotronHOmni:
    """Native MLX multimodal Nemotron-3-Nano-Omni wrapper.

    Args:
        bundle_path: omni-merged bundle (LLM + vision + sound + projectors).
        dtype: mx dtype for encoders. Default fp32 for stability; bf16 once
            stable.
        device: ignored — MLX runs on Metal automatically.
    """

    def __init__(
        self,
        bundle_path: Union[str, Path],
        *,
        dtype: mx.Dtype = mx.float32,
    ):
        self.bundle_path = Path(bundle_path)
        self.dtype = dtype

        with open(self.bundle_path / "config_omni.json") as f:
            self.omni_config = json.load(f)

        self.img_context_token_id = self.omni_config["img_context_token_id"]
        self.video_context_token_id = self.omni_config.get(
            "video_context_token_id", self.img_context_token_id,
        )
        self.sound_context_token_id = self.omni_config["sound_context_token_id"]
        self.downsample_ratio = self.omni_config.get("downsample_ratio", 0.5)
        self.image_size = self.omni_config.get("force_image_size", 512)

        # Load LLM via mlx_lm or load_jangtq
        self._load_llm()
        # Load vision tower + mlp1 projector
        self._load_vision()
        # Load sound encoder + sound_projection
        self._load_sound()
        # Tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.bundle_path), trust_remote_code=True,
        )
        # Persistent multi-turn cache (None = fresh)
        self._cache = None
        self._eos_ids = {11}

    def _load_llm(self):
        try:
            with open(self.bundle_path / "jang_config.json") as f:
                jc = json.load(f)
            wf = jc.get("weight_format", "mlx")
        except (OSError, ValueError, TypeError, AttributeError):
            wf = "mlx"
        if wf == "mxtq":
            from jang_tools.load_jangtq import load_jangtq_model
            self.llm, _ = load_jangtq_model(str(self.bundle_path))
        else:
            from jang_tools.nemotron_omni_chat import (
                _load_mlx_lm_ignoring_omni_extras,
            )

            self.llm, _ = _load_mlx_lm_ignoring_omni_extras(self.bundle_path)

    def _load_vision(self):
        # RADIO
        self.vision_model = RADIOVisionModel(apply_input_conditioner=False)
        radio_w = _load_safetensors_with_prefix(
            self.bundle_path, "vision_model.radio_model.",
        )
        radio_mapped = _cast_to(map_radio_weights(radio_w), self.dtype)
        self.vision_model.load_weights(list(radio_mapped.items()), strict=False)

        # mlp1 projector — derive dims from config + ratio
        vit_hidden = self.omni_config.get("vit_hidden_size", 1280)
        proj_hidden = self.omni_config.get("projector_hidden_size", 20480)
        llm_hidden = self.omni_config.get("llm_config", {}).get("hidden_size", 2688)
        post_shuffle_dim = vit_hidden * int(1 / self.downsample_ratio) ** 2  # 1280*4=5120

        self.mlp1 = VisionMLPProjector(post_shuffle_dim, proj_hidden, llm_hidden)
        mlp1_w = _load_safetensors_with_prefix(self.bundle_path, "mlp1.")
        mlp1_mapped = _cast_to(map_mlp1_weights(mlp1_w), self.dtype)
        self.mlp1.load_weights(list(mlp1_mapped.items()), strict=False)

    def _load_sound(self):
        # Parakeet encoder
        self.sound_encoder = ParakeetEncoder()
        se_w = _load_safetensors_with_prefix(
            self.bundle_path, "sound_encoder.encoder.",
        )
        se_mapped = _cast_to(map_parakeet_weights(se_w), self.dtype)
        self.sound_encoder.load_weights(list(se_mapped.items()), strict=False)

        # Sound projection
        self.sound_projection = SoundProjector()
        sp_w = _load_safetensors_with_prefix(self.bundle_path, "sound_projection.")
        sp_mapped = _cast_to(map_sound_projection_weights(sp_w), self.dtype)
        self.sound_projection.load_weights(list(sp_mapped.items()), strict=False)

    # ── Embedding extraction (native MLX) ──────────────────────────────────

    def extract_image_embeds(self, pil_images: List[Image.Image]) -> mx.array:
        """PIL images → image embeds (num_total_tiles_tokens, llm_hidden)."""
        pixel_values_np, tile_counts = preprocess_images(
            pil_images, image_size=self.image_size, max_num=1, use_thumbnail=False,
        )
        pv = mx.array(pixel_values_np)            # (N_tiles, 3, H, W)
        feats = self.vision_model(pv)              # (N_tiles, n_cls + N_patches, D)
        # Strip cls/register tokens (first 10)
        feats = feats[:, self.vision_model.num_cls_tokens:, :]
        # pixel_shuffle: (N, n_patches, D) → (N, h, w, D) → (N, h*r, w*r, D/r²)
        N, P, D = feats.shape
        h = w = int(np.sqrt(P))
        feats = feats.reshape(N, h, w, D)
        feats = pixel_shuffle(feats, scale_factor=self.downsample_ratio)
        feats = feats.reshape(N, -1, feats.shape[-1])  # (N, h*w*r², D/r²)
        # Project via mlp1 to LLM hidden dim
        feats = self.mlp1(feats)                   # (N_tiles, n_tokens, llm_hidden)
        return feats

    def extract_audio_embeds(self, audio_path_or_array) -> mx.array:
        """Audio file or 1-D numpy → audio embeds (n_subsampled, llm_hidden)."""
        if isinstance(audio_path_or_array, (str, Path)):
            import soundfile as sf
            audio, sr = sf.read(str(audio_path_or_array))
            if audio.ndim > 1:
                audio = audio.mean(axis=-1)
            if sr != 16000:
                from scipy import signal
                audio = signal.resample_poly(audio, 16000, sr)
        else:
            audio = audio_path_or_array
        mel_np = extract_mel_features(audio.astype(np.float32))  # (1, F, 128)
        mel = mx.array(mel_np)
        feats = self.sound_encoder(mel)                          # (1, F_sub, 1024)
        feats = self.sound_projection(feats)                     # (1, F_sub, llm_hidden)
        return feats

    def extract_video_embeds(
        self, video_path: Path | str, *, target_frames: int = 32,
        apply_evs: bool = True, evs_pruning_rate: float = 0.7,
    ) -> mx.array:
        """Video file → video embeds (n_total_tokens, llm_hidden).

        Uses RADIO with `video=True` to swap in `video_embedder` for the
        T-frame channel-stacked input. After the ViT, apply optional EVS
        token retention mask, then pixel-shuffle + mlp1.
        """
        from .video_processor import preprocess_video, compute_evs_retention_mask

        pixel_values_np, meta = preprocess_video(
            video_path, image_size=self.image_size, target_frames=target_frames,
            video_temporal_patch_dim=2,
        )
        # pixel_values_np: (n_groups, T*3=6, H, W). RADIO expects (B, 3, H, W)
        # for the standard embedder. The video_embedder handles the T*3 channels.
        # Our `_im_to_patches` reshapes via `(B, C, py, P, px, P)` — for the
        # video path, we treat C=T*3=6 so the patch becomes 6*P*P=1536, which
        # matches video_embedder.weight shape (1280, 1536).
        pv = mx.array(pixel_values_np)
        feats = self.vision_model(pv, video=True)  # (n_groups, n_cls + N_patches, D)
        # Strip cls/register tokens
        feats = feats[:, self.vision_model.num_cls_tokens:, :]
        # pixel_shuffle: (N, n_patches, D) → (N, h*r, w*r, D/r²) flat → (N, h*w*r², D/r²)
        N, P, D = feats.shape
        h = w = int(np.sqrt(P))
        feats = feats.reshape(N, h, w, D)
        feats = pixel_shuffle(feats, scale_factor=self.downsample_ratio)
        feats = feats.reshape(N, -1, feats.shape[-1])  # (N, 256, 5120) if 512×512

        # Project via mlp1 to LLM hidden dim
        feats = self.mlp1(feats)            # (n_groups, n_tokens_per_group, llm_hidden)

        # EVS pruning: drop ~70% of redundant tokens (high cosine similarity
        # to previous frame's same spatial position)
        if apply_evs and N >= 2:
            from .video_processor import compute_evs_retention_mask
            n_groups, tokens_per, hidden = feats.shape
            # spatial_merge_size = 1/downsample_ratio = 2 (after pixel_shuffle)
            grid_full = int(np.sqrt(P))     # 32
            grid_post = int(np.sqrt(tokens_per))  # 16 after shuffle
            # Flatten (n_groups, n_tokens, D) → (n_groups*n_tokens, D)
            feats_flat = np.array(feats.astype(mx.float32)).reshape(-1, hidden)
            mask = compute_evs_retention_mask(
                feats_flat, n_temporal_groups=n_groups,
                grid_h=grid_full, grid_w=grid_full,
                spatial_merge_size=grid_full // grid_post,  # 32/16 = 2
                q=evs_pruning_rate,
            )
            kept_idx = np.where(mask)[0]
            feats_pruned = feats_flat[kept_idx]
            feats = mx.array(feats_pruned).reshape(1, -1, hidden)
        return feats

    # ── Multi-turn chat ────────────────────────────────────────────────────

    def reset(self):
        self._cache = None

    def _ensure_cache(self):
        if self._cache is None:
            self._cache = self.llm.make_cache()

    def turn(
        self,
        text: str,
        images: Optional[Sequence[Union[str, Path]]] = None,
        audio: Optional[Union[str, Path]] = None,
        video: Optional[Union[str, Path]] = None,
        max_tokens: int = 256,
        temperature: float = 0.6,
        top_p: float = 0.95,
        enable_thinking: bool = True,
        video_target_frames: int = 32,
        video_apply_evs: bool = True,
        token_callback: Optional[Callable[[Optional[int], str], None]] = None,
    ) -> str:
        self._ensure_cache()

        # Encode multimodal inputs
        image_embeds = video_embeds = audio_embeds = None
        n_image_tokens = n_video_tokens = n_audio_tokens = 0
        if images:
            pil = [Image.open(str(p)).convert("RGB") for p in images]
            ie = self.extract_image_embeds(pil)
            image_embeds = np.array(ie.astype(mx.float32))
            n_image_tokens = image_embeds.shape[0] * image_embeds.shape[1]
        if video is not None:
            ve = self.extract_video_embeds(
                video, target_frames=video_target_frames,
                apply_evs=video_apply_evs,
            )
            video_embeds = np.array(ve.astype(mx.float32))
            n_video_tokens = video_embeds.shape[0] * video_embeds.shape[1]
        if audio is not None:
            ae = self.extract_audio_embeds(audio)
            audio_embeds = np.array(ae.astype(mx.float32))
            n_audio_tokens = audio_embeds.shape[0] * audio_embeds.shape[1]

        # Build prompt. Note: video uses `<image>` (id=18) tokens too — the
        # tokenizer has no real `<video>` token despite img_context_token_id
        # being 131081 in config. Source `processing.py:297-300` says the
        # 131081 id doesn't decode to any printable string, so they reuse
        # `<image>` as the placeholder. The model distinguishes image vs video
        # by which embeds were extracted (we just place them in order).
        media = ""
        if n_image_tokens > 0:
            media += "<img>" + ("<image>" * n_image_tokens) + "</img>\n"
        if n_video_tokens > 0:
            # Reuse <img>/<image> for video frames per source convention.
            media += "<img>" + ("<image>" * n_video_tokens) + "</img>\n"
        if n_audio_tokens > 0:
            media += "<sound>" + ("<so_embedding>" * n_audio_tokens) + "</sound>\n"
        msg = media + text
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": msg}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        input_ids = self.tokenizer(prompt, return_tensors="np")["input_ids"]
        self._last_prompt_tokens = int(input_ids.shape[-1])

        # Embed text + inject multimodal
        ids_mx = mx.array(input_ids)
        text_embeds = self.llm.backbone.embeddings(ids_mx)
        embeds_np = np.array(text_embeds.astype(mx.float32))
        ids = input_ids[0]
        # Image and video both use img_context_token_id=18; assemble the
        # combined image+video placeholder slots in prompt order. We placed
        # images first then video in the prompt, so the matching positions
        # array is `[img_positions, video_positions]` in that order.
        image_video_embeds_concat = None
        if image_embeds is not None or video_embeds is not None:
            parts = []
            if image_embeds is not None:
                parts.append(image_embeds.reshape(-1, image_embeds.shape[-1]))
            if video_embeds is not None:
                parts.append(video_embeds.reshape(-1, video_embeds.shape[-1]))
            image_video_embeds_concat = np.concatenate(parts, axis=0)

        for embeds_flat, tok_id in [
            (image_video_embeds_concat, self.img_context_token_id),
            (audio_embeds.reshape(-1, audio_embeds.shape[-1])
             if audio_embeds is not None else None, self.sound_context_token_id),
        ]:
            if embeds_flat is None:
                continue
            positions = np.where(ids == tok_id)[0]
            if len(positions) != embeds_flat.shape[0]:
                raise ValueError(
                    f"placeholder count mismatch: positions={len(positions)} "
                    f"vs embeds={embeds_flat.shape[0]}"
                )
            embeds_np[0, positions, :] = embeds_flat
        embeds = mx.array(embeds_np, dtype=text_embeds.dtype)

        # Decode
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask
        backbone = self.llm.backbone

        h = embeds
        attn_mask = create_attention_mask(h, self._cache[backbone.fa_idx])
        ssm_mask = create_ssm_mask(h, self._cache[backbone.ssm_idx])
        ci = 0
        for layer in backbone.layers:
            if layer.block_type in ("M", "*"):
                c = self._cache[ci]; ci += 1
                m = attn_mask if layer.block_type == "*" else ssm_mask
                h = layer(h, mask=m, cache=c)
            else:
                h = layer(h)
        h = backbone.norm_f(h)
        logits = self.llm.lm_head(h)
        next_logit = logits[:, -1, :]

        def sample(logit, temp, tp):
            if temp <= 0:
                return mx.argmax(logit, axis=-1, keepdims=True)
            logit = logit / temp
            if tp >= 1.0:
                return mx.random.categorical(logit)[..., None]
            sorted_idx = mx.argsort(-logit, axis=-1)
            sorted_logits = mx.take_along_axis(logit, sorted_idx, axis=-1)
            sorted_probs = mx.softmax(sorted_logits, axis=-1)
            cumprobs = mx.cumsum(sorted_probs, axis=-1)
            keep = mx.concatenate(
                [mx.ones_like(cumprobs[..., :1]) > 0,
                 cumprobs[..., :-1] <= tp],
                axis=-1,
            )
            neg_inf = mx.full(sorted_logits.shape, -1e9, dtype=sorted_logits.dtype)
            filtered = mx.where(keep, sorted_logits, neg_inf)
            tok_in_sorted = mx.random.categorical(filtered)[..., None]
            return mx.take_along_axis(sorted_idx, tok_in_sorted, axis=-1)

        tokens: List[int] = []
        detokenizer = None
        if token_callback is not None:
            from mlx_lm.tokenizer_utils import TokenizerWrapper

            detokenizer = TokenizerWrapper(
                self.tokenizer,
                eos_token_ids=self._eos_ids,
            ).detokenizer

        def record_token(token) -> None:
            token_id = int(token.item())
            tokens.append(token_id)
            segment = ""
            if detokenizer is not None and token_id not in self._eos_ids:
                detokenizer.add_token(token_id)
                segment = detokenizer.last_segment
            if token_callback is not None:
                token_callback(token_id, segment)

        tok = sample(next_logit, temperature, top_p)
        record_token(tok)
        for _ in range(max_tokens - 1):
            if tokens[-1] in self._eos_ids:
                break
            h = backbone.embeddings(tok)
            attn_mask = create_attention_mask(h, self._cache[backbone.fa_idx])
            ssm_mask = create_ssm_mask(h, self._cache[backbone.ssm_idx])
            ci = 0
            for layer in backbone.layers:
                if layer.block_type in ("M", "*"):
                    c = self._cache[ci]; ci += 1
                    m = attn_mask if layer.block_type == "*" else ssm_mask
                    h = layer(h, mask=m, cache=c)
                else:
                    h = layer(h)
            h = backbone.norm_f(h)
            logits = self.llm.lm_head(h)
            tok = sample(logits[:, -1, :], temperature, top_p)
            record_token(tok)

        if detokenizer is not None:
            detokenizer.finalize()
            final_segment = detokenizer.last_segment
            if final_segment and token_callback is not None:
                token_callback(None, final_segment)

        self._last_completion_tokens = len(tokens)
        self._last_finish_reason = (
            "stop" if tokens and tokens[-1] in self._eos_ids else "length"
        )
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
