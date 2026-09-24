"""MiMo-V2.6 omni front-end: text model + lazily loaded vision / audio towers.

Towers load on first use, so text-only serving keeps the full memory headroom
(stock-macOS GPU budget, README §7). Placeholders emitted by the chat template

    image  <|vision_start|><|image_pad|><|vision_end|>
    video  <|vision_start|><|video_pad|><|vision_end|>
    audio  <|mimo_audio_start|><|audio_pad|><|mimo_audio_end|>

are expanded (processor token counts; SGLang layout, see VISION.md/AUDIO.md),
embedded, and the pad positions are overwritten with tower outputs in order.
The backbone uses plain 1-D positions for every token (reference forward).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from . import v26_audio as A
from . import v26_vision as V

AUDIO_START_ID, AUDIO_PAD_ID, AUDIO_END_ID = 151673, 151669, 151674


@dataclass
class Media:
    images: list = field(default_factory=list)   # PIL.Image / HxWx3 uint8 arrays
    videos: list = field(default_factory=list)   # (frames [T,H,W,3] uint8, timestamps seconds [T])
    audios: list = field(default_factory=list)   # (wav np.ndarray, sample_rate)


class MiMoV26Omni:
    def __init__(self, bundle: str | Path, model: Any, tokenizer: Any, *, vision_dtype=mx.bfloat16,
                 merger_norm: str = "rms", audio_tokenizer_dtype=mx.float32):
        self.bundle = Path(bundle)
        self.model, self.tokenizer = model, tokenizer
        self.vision_dtype, self.merger_norm = vision_dtype, merger_norm
        self.audio_tokenizer_dtype = audio_tokenizer_dtype  # fp32: bf16 codes are noisy even in the reference
        self._vt = self._vp = self._atok = self._aenc = None

    # -------------------------------------------------------------- lazy towers
    def vision(self):
        if self._vt is None:
            cfg = V.MiMoV26VisionConfig.from_model_dir(self.bundle)
            tower = V.MiMoVisionTower(cfg, merger_norm=self.merger_norm)
            tower.load_source_weights(V.load_visual_weights_from_source(self.bundle), dtype=self.vision_dtype)
            self._vt, self._vp = tower, V.MiMoV26VisionProcessor.from_model_dir(self.bundle)
        return self._vt, self._vp

    def audio(self):
        if self._atok is None:
            self._atok = A.load_audio_tokenizer_encoder(str(self.bundle), dtype=self.audio_tokenizer_dtype)
            self._aenc = A.load_audio_encoder(str(self.bundle))
        return self._atok, self._aenc

    # ----------------------------------------------------------------- prompt
    @staticmethod
    def _find(ids: list[int], triple: tuple[int, int, int]) -> list[int]:
        a, b, c = triple
        return [i for i in range(len(ids) - 2) if ids[i] == a and ids[i + 1] == b and ids[i + 2] == c]

    def build(self, prompt_ids: list[int], media: Media) -> tuple[mx.array, mx.array | None]:
        """-> (input_ids [L], inputs_embeds [L, H] or None when text-only)."""
        ids = list(prompt_ids)
        if not (media.images or media.videos or media.audios):
            return mx.array(ids), None
        spans = []  # (pos, kind)
        spans += [(p, "image") for p in self._find(ids, (V.VISION_START_ID, V.IMAGE_PAD_ID, V.VISION_END_ID))]
        spans += [(p, "video") for p in self._find(ids, (V.VISION_START_ID, V.VIDEO_PAD_ID, V.VISION_END_ID))]
        spans += [(p, "audio") for p in self._find(ids, (AUDIO_START_ID, AUDIO_PAD_ID, AUDIO_END_ID))]
        spans.sort()
        want = {"image": len(media.images), "video": len(media.videos), "audio": len(media.audios)}
        have = {k: sum(1 for _, s in spans if s == k) for k in want}
        if want != have:
            raise ValueError(f"placeholders {have} != media {want}")
        it = {"image": iter(media.images), "video": iter(media.videos), "audio": iter(media.audios)}
        vis_items, aud_codes, out = [], [], []
        cursor = 0
        for pos, kind in spans:
            out += ids[cursor:pos]
            if kind == "audio":
                wav, sr = next(it["audio"])
                atok, _ = self.audio()
                codes, n = A.prepare_audio(np.asarray(wav), int(sr), atok)
                aud_codes.append(codes)
                out += A.build_audio_prompt_ids(n)
            else:
                _, vp = self.vision()
                if kind == "image":
                    item = vp.preprocess_image(next(it["image"]))
                else:
                    frames, ts = next(it["video"])
                    item = vp.preprocess_video(frames, ts)
                vis_items.append(item)
                out += vp.expand_tokens(item, encode=lambda s: self.tokenizer.encode(s, add_special_tokens=False))
            cursor = pos + 3
        out += ids[cursor:]
        input_ids = mx.array(out)
        emb = self.model.model.embed_tokens(input_ids)
        if vis_items:
            tower, _ = self.vision()
            pv, grid = V.batch_items(vis_items)
            feats = tower(mx.array(pv).astype(self.vision_dtype), grid).astype(emb.dtype)
            pos = np.where(np.isin(np.array(out), [V.IMAGE_PAD_ID, V.VIDEO_PAD_ID]))[0]
            if feats.shape[0] != len(pos):
                raise AssertionError(f"vision rows {feats.shape[0]} != pad tokens {len(pos)}")
            emb[mx.array(pos)] = feats
        if aud_codes:
            _, aenc = self.audio()
            feats = mx.concatenate([A.embed_audio(c, aenc) for c in aud_codes], axis=0).astype(emb.dtype)
            pos = np.where(np.array(out) == AUDIO_PAD_ID)[0]
            if feats.shape[0] != len(pos):
                raise AssertionError(f"audio rows {feats.shape[0]} != pad tokens {len(pos)}")
            emb[mx.array(pos)] = feats
        return input_ids, emb

    # ---------------------------------------------------------------- generate
    def generate(self, prompt_ids: list[int], media: Media | None = None, *, max_tokens: int = 256,
                 temperature: float = 1.0, top_p: float = 0.95, stop_ids=(151643, 151645, 151672)):
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=temperature, top_p=top_p if temperature > 0 else 1.0)
        ids, emb = self.build(prompt_ids, media or Media())
        cache = self.model.make_cache()
        logits = self.model(ids[None], cache=cache, input_embeddings=None if emb is None else emb[None])
        out = []
        for _ in range(max_tokens):
            tok = sampler(logits[:, -1, :])
            t = int(tok.item())
            if t in stop_ids:
                break
            out.append(t)
            logits = self.model(tok.reshape(1, 1), cache=cache)
        return self.tokenizer.decode(out), len(ids)
