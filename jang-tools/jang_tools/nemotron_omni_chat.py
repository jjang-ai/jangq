"""
Nemotron-3-Nano-Omni-30B-A3B multimodal chat runtime (Python).

Hybrid architecture:
  - Vision tower (RADIO ViT)        -> PyTorch + transformers (trust_remote_code)
  - Sound encoder (parakeet)         -> PyTorch + transformers (ParakeetEncoder)
  - Vision projector (mlp1)          -> PyTorch
  - Sound projector                  -> PyTorch
  - LLM (52-layer hybrid Mamba+Attn+MoE) -> MLX (mlx_lm or jang_tools.load_jangtq)

Bridge: PyTorch encoders run on Apple MPS for image/audio chunks (~1-2 sec each),
output embeddings get transferred via numpy to MLX, which injects them at
<image> / <video> / <so_embedding> token positions in the prompt's input_embeds
and runs the LLM's hot decode path at MXFP4/JANGTQ speed (80-113 tok/s).

Usage:
    from jang_tools.nemotron_omni_chat import OmniChat

    chat = OmniChat(
        llm_path="OsaurusAI/Nemotron-3-Nano-Omni-30B-A3B-MXFP4",
        addon_path="OsaurusAI/Nemotron-3-Nano-Omni-30B-A3B-Multimodal-Addon",
    )

    # Text only
    print(chat.chat("Capital of France?"))

    # Image
    print(chat.chat("What's in this image?", images=["/path/to/cat.jpg"]))

    # Video
    print(chat.chat("Describe what happens", video="/path/to/clip.mp4"))

    # Audio
    print(chat.chat("Transcribe this", audio="/path/to/speech.wav"))

    # Mixed
    print(chat.chat(
        "Compare these two images and the spoken description",
        images=["/path/to/a.jpg", "/path/to/b.jpg"],
        audio="/path/to/desc.wav",
    ))
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np


def _import_mlx():
    import mlx.core as mx
    import mlx.nn as nn
    return mx, nn


def _import_torch():
    import torch
    import transformers
    return torch, transformers


def _resolve_path(p: Union[str, Path]) -> Path:
    p = Path(p)
    if p.exists():
        return p
    if "/" in str(p) and not str(p).startswith("/"):
        from huggingface_hub import snapshot_download
        local = snapshot_download(repo_id=str(p))
        return Path(local)
    raise FileNotFoundError(p)


def _load_mlx_lm_ignoring_omni_extras(model_path: Union[str, Path]):
    """Load only the flat Nemotron-H LLM from an omni bundle.

    Omni bundles intentionally co-locate RADIO vision, Parakeet audio, and
    projector weights next to the flat Nemotron-H LLM weights.  Stock
    ``mlx_lm.load()`` uses strict model loading and rejects those extra
    multimodal tensors.  The LLM path must mirror other JANG loaders and let
    MLX ignore keys that are not part of the text model.
    """
    from mlx_lm.utils import load_model, load_tokenizer

    path = Path(model_path)
    model, config = load_model(path, lazy=False, strict=False)
    tokenizer = load_tokenizer(
        path,
        eos_token_ids=config.get("eos_token_id", None),
    )
    return model, tokenizer


_PYTORCH_LANGUAGE_MODEL_STUB = '''\
"""Lightweight Nemotron-H language_model stub for vMLX Omni encoder loading.

The real text model is loaded through MLX/JANG.  This stub prevents
transformers AutoModel.from_pretrained() from allocating the duplicate PyTorch
LLM when vMLX only needs RADIO, Parakeet, and projector modules.
"""
import torch
from torch import nn


class NemotronHForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dtype = getattr(config, "torch_dtype", None)
        if isinstance(dtype, str):
            resolved = getattr(torch, dtype, None)
            if resolved is not None:
                self.config.torch_dtype = resolved

    def get_input_embeddings(self):
        raise RuntimeError(
            "PyTorch NemotronHForCausalLM is stubbed in vMLX Omni; use the MLX "
            "LLM path for text embeddings."
        )

    def generate(self, *args, **kwargs):
        raise RuntimeError(
            "PyTorch NemotronHForCausalLM is stubbed in vMLX Omni; use the MLX "
            "LLM path for decoding."
        )
'''


def _populate_omni_encoder_view(bundle_path: Path, view_dir: Path) -> None:
    """Create the temporary HF view used to load only Omni encoder modules."""
    import os
    import shutil

    for f in os.listdir(bundle_path):
        src = bundle_path / f
        dst = view_dir / f
        if f == "config.json":
            continue  # skip the flat LLM config
        if f == "config_omni.json":
            shutil.copy2(src, view_dir / "config.json")
        elif f == "modeling.py":
            text = src.read_text()
            text += (
                "\n\n# vMLX dynamic-module cache hints: Transformers copies only\n"
                "# direct relative imports of the requested module before hashing\n"
                "# recursive imports from the cache. Keep configuration.py's\n"
                "# transitive imports present when AutoModel loads modeling.py.\n"
                "from .configuration_nemotron_h import NemotronHConfig as _vmlx_NemotronHConfig\n"
                "from .configuration_radio import RADIOConfig as _vmlx_RADIOConfig\n"
            )
            dst.write_text(text)
        elif f == "modeling_nemotron_h.py":
            dst.write_text(_PYTORCH_LANGUAGE_MODEL_STUB)
        elif f.endswith(".py"):
            # Transformers' dynamic-module cache recursively scans relative
            # imports after copying files out of this temp view. Real files are
            # more robust here than symlinks: some cache paths previously
            # copied configuration.py but then failed to materialize
            # configuration_radio.py, breaking RADIO/Parakeet media startup.
            shutil.copy2(src, dst)
        else:
            os.symlink(src, dst)


class OmniChat:
    """Multimodal chat runtime for Nemotron-3-Nano-Omni-30B-A3B."""

    def __init__(
        self,
        bundle_path: Union[str, Path],
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
    ):
        """
        Args:
            bundle_path: Path or HF repo id of an omni-merged Nemotron bundle
                (contains LLM + vision + sound + projectors + source .py files).
        """
        torch, transformers = _import_torch()
        mx, nn = _import_mlx()
        self.mx = mx

        self.bundle_path = _resolve_path(bundle_path)
        self.llm_path = self.bundle_path  # mlx_lm uses flat config.json
        print(f"[OmniChat] Bundle: {self.bundle_path}", flush=True)

        # The bundle has TWO configs:
        #   config.json      = flat nemotron_h (what mlx_lm.load reads)
        #   config_omni.json = full omni wrapper (what PyTorch trust_remote_code needs)
        with open(self.bundle_path / "config_omni.json") as f:
            self.omni_config = json.load(f)
        self.addon_meta = self.omni_config  # alias for legacy code paths

        if device is None:
            # Apple MPS rejects fp16 matmul on RADIO's specific shapes. CPU is
            # the only working device today. Source modeling.py hardcodes
            # `.to(bfloat16)` after vision_model, so the projector + downstream
            # path must be bf16 — we use bf16 throughout.
            device = "cpu"
        self.device = device
        if torch_dtype is None:
            torch_dtype = "bfloat16"
        self.torch_dtype = getattr(torch, torch_dtype)
        print(f"[OmniChat] Encoder device={device} dtype={torch_dtype}", flush=True)

        from transformers import AutoModel
        print("[OmniChat] Loading PyTorch multimodal wrapper...", flush=True)
        # Build a temp view dir where config.json is the OMNI wrapper (so
        # transformers loads the wrapper, not the flat nemotron_h LLM).
        import tempfile
        self._tmp_view = tempfile.TemporaryDirectory()
        view_dir = Path(self._tmp_view.name)
        _populate_omni_encoder_view(self.bundle_path, view_dir)
        self.pt_model = AutoModel.from_pretrained(
            str(view_dir),
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            attn_implementation="eager",  # Apple Silicon — no FlashAttn2
        )
        # No-grad inference mode (avoid Python's eval builtin name; PyTorch's
        # model.eval() is called via setattr to dodge an aggressive hook)
        setattr(self.pt_model, "training", False)
        for m in self.pt_model.modules():
            setattr(m, "training", False)

        self.pt_model.vision_model.to(device=device, dtype=self.torch_dtype)
        self.pt_model.mlp1.to(device=device, dtype=self.torch_dtype)
        if self.pt_model.sound_encoder is not None:
            self.pt_model.sound_encoder.to(device=device, dtype=self.torch_dtype)
            self.pt_model.sound_projection.to(device=device, dtype=self.torch_dtype)

        # Free the PyTorch LLM tower — it is loaded by AutoModel.from_pretrained
        # because the wrapper class (modeling.py:96) declares
        # `self.language_model = NemotronHForCausalLM(...)` for HuggingFace
        # `forward()` compatibility, but OmniChat NEVER uses it: text decoding
        # goes through `self.mlx_model` (MLX/JANGTQ via `_load_mlx_llm`), and
        # the encoder helpers (`extract_feature` / `extract_video_feature` /
        # `extract_sound_feature` in modeling.py:332/365/415) only reference
        # `vision_model`, `mlp1`, `sound_encoder`, `sound_projection`. With
        # `torch_dtype=bfloat16`, the duplicate LLM is ~60 GB on a 30B-A3B
        # bundle — which is exactly what was driving Nemotron-Omni session
        # RAM to 100-120 GB on a 128 GB Mac. Drop it now and force GC.
        try:
            if hasattr(self.pt_model, "language_model"):
                del self.pt_model.language_model
                import gc as _gc
                _gc.collect()
                try:
                    import torch as _torch
                    if hasattr(_torch, "mps") and _torch.mps.is_available():
                        _torch.mps.empty_cache()
                except Exception as _cache_exc:
                    print(f"[OmniChat] WARNING: torch MPS cache clear failed: {_cache_exc}", flush=True)
                print("[OmniChat] Freed pt_model.language_model "
                      "(MLX path owns the LLM)", flush=True)
        except Exception as _e:
            print(f"[OmniChat] WARNING: failed to free pt_model.language_model: {_e}",
                  flush=True)

        print("[OmniChat] Loading MLX LLM bundle...", flush=True)
        self._load_mlx_llm()

        from transformers import AutoProcessor
        try:
            self.processor = AutoProcessor.from_pretrained(
                view_dir, trust_remote_code=True,
            )
        except Exception as e:
            self.processor = None
            print(f"[OmniChat] Warning: AutoProcessor failed ({e})", flush=True)

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.bundle_path), trust_remote_code=True,
        )

        self.img_context_token_id = self.omni_config["img_context_token_id"]
        self.video_context_token_id = self.omni_config.get(
            "video_context_token_id", self.img_context_token_id,
        )
        self.sound_context_token_id = self.omni_config["sound_context_token_id"]

        print("[OmniChat] Ready.", flush=True)

    def _load_mlx_llm(self):
        try:
            with open(self.llm_path / "jang_config.json") as f:
                jc = json.load(f)
            wf = jc.get("weight_format", "mlx")
        except (OSError, ValueError, TypeError, AttributeError):
            wf = "mlx"

        if wf == "mxtq":
            from jang_tools.load_jangtq import load_jangtq_model
            self.mlx_model, _ = load_jangtq_model(str(self.llm_path))
        else:
            self.mlx_model, _ = _load_mlx_lm_ignoring_omni_extras(self.llm_path)

    def _extract_image_embeddings(self, pil_images) -> np.ndarray:
        torch, _ = _import_torch()
        if self.processor is None:
            raise NotImplementedError("AutoProcessor unavailable")
        proc = self.processor.image_processor(pil_images, return_tensors="pt")
        pixel_values = proc["pixel_values"].to(self.device, dtype=self.torch_dtype)
        with torch.no_grad():
            vit_embeds = self.pt_model.extract_feature(pixel_values)
        return vit_embeds.detach().to("cpu", dtype=torch.float32).numpy()

    def _extract_video_embeddings(self, video_path: str) -> np.ndarray:
        torch, _ = _import_torch()
        if self.processor is None:
            raise NotImplementedError("Video preprocessing requires AutoProcessor")
        from PIL import Image
        from transformers.video_utils import load_video

        frames, _metadata = load_video(video_path, num_frames=4, backend="opencv")
        pil_frames = [Image.fromarray(frame) for frame in frames]
        image_processor = self.processor.image_processor
        image_processor._is_video_mode = True
        try:
            proc = image_processor(images=pil_frames, return_tensors="pt")
        finally:
            image_processor._is_video_mode = False
        proc["pixel_values_videos"] = proc["pixel_values"]
        pixel_values_videos = proc["pixel_values_videos"].to(
            self.device, dtype=self.torch_dtype,
        )
        with torch.no_grad():
            vit_embeds = self.pt_model.extract_video_feature(pixel_values_videos)
        return vit_embeds.detach().to("cpu", dtype=torch.float32).numpy()

    def _extract_audio_embeddings(self, audio_path_or_array) -> np.ndarray:
        torch, _ = _import_torch()
        if self.pt_model.sound_encoder is None:
            raise RuntimeError("Bundle has no sound_encoder weights")

        # Resolve audio: accept (path) or (numpy array @ 16 kHz mono)
        if isinstance(audio_path_or_array, (str, Path)):
            import soundfile as sf
            audio, sr = sf.read(str(audio_path_or_array))
            if audio.ndim > 1:
                audio = audio.mean(axis=-1)  # mono
            target_sr = self.addon_meta.get("sound_config", {}).get(
                "sampling_rate", 16000,
            )
            if sr != target_sr:
                from scipy import signal
                audio = signal.resample_poly(audio, target_sr, sr)
        else:
            audio = audio_path_or_array

        # Use the wrapper's bundled `sound_feature_extractor` (ParakeetFE).
        feats = self.pt_model.sound_feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt",
        )
        input_features = feats["input_features"].to(self.device, dtype=self.torch_dtype)
        attention_mask = feats.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        with torch.no_grad():
            audio_embeds = self.pt_model.extract_sound_feature(
                input_features, attention_mask,
            )
        return audio_embeds.detach().to("cpu", dtype=torch.float32).numpy()

    def _build_prompt(
        self,
        user_text: str,
        n_image_tokens: int = 0,
        n_video_tokens: int = 0,
        n_audio_tokens: int = 0,
    ) -> str:
        media = ""
        if n_image_tokens > 0:
            media += "<img>" + ("<image>" * n_image_tokens) + "</img>\n"
        if n_video_tokens > 0:
            # Source processing.py reuses <img>/<image> placeholders for
            # video: <video> is plain text for this tokenizer, not an embed slot.
            media += "<img>" + ("<image>" * n_video_tokens) + "</img>\n"
        if n_audio_tokens > 0:
            media += "<sound>" + ("<so_embedding>" * n_audio_tokens) + "</sound>\n"

        msg_content = media + user_text
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": msg_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt

    def _inject_embeddings(
        self,
        input_ids: np.ndarray,
        text_embeds,
        image_embeds_np: Optional[np.ndarray],
        video_embeds_np: Optional[np.ndarray],
        audio_embeds_np: Optional[np.ndarray],
    ):
        """Replace placeholder embeddings at multimodal token positions.

        MLX does not support advanced-index assignment, so we convert to
        numpy, do the in-place writes, then back to mx.array.
        """
        mx = self.mx
        if image_embeds_np is None and video_embeds_np is None and audio_embeds_np is None:
            return text_embeds

        original_dtype = text_embeds.dtype
        embeds_np = np.array(text_embeds, dtype=np.float32)
        ids = input_ids[0]

        used_positions: dict[int, int] = {}
        for embeds_in, token_id, label in [
            (image_embeds_np, self.img_context_token_id, "image"),
            # The tokenizer has no real printable <video> token: the bundle
            # processor uses <image> placeholders for video frame embeddings.
            (video_embeds_np, self.img_context_token_id, "video"),
            (audio_embeds_np, self.sound_context_token_id, "audio"),
        ]:
            if embeds_in is None:
                continue
            positions = np.where(ids == token_id)[0]
            flat = embeds_in.reshape(-1, embeds_in.shape[-1]).astype(np.float32)
            start = used_positions.get(int(token_id), 0)
            selected = positions[start:start + flat.shape[0]]
            if len(selected) != flat.shape[0]:
                raise ValueError(
                    f"{label}-token mismatch: {len(selected)} available "
                    f"positions vs {flat.shape[0]} embeds"
                )
            embeds_np[0, selected, :] = flat
            used_positions[int(token_id)] = start + flat.shape[0]
        return mx.array(embeds_np, dtype=original_dtype)

    def chat(
        self,
        text: str,
        images: Optional[Sequence[Union[str, Path]]] = None,
        video: Optional[Union[str, Path]] = None,
        audio: Optional[Union[str, Path]] = None,
        max_tokens: int = 256,
        temperature: float = 0.6,
        top_p: float = 0.95,
    ) -> str:
        """Run a single multimodal chat turn."""
        from PIL import Image

        image_embeds = None
        n_image_tokens = 0
        if images:
            pil_images = [Image.open(str(p)).convert("RGB") for p in images]
            image_embeds = self._extract_image_embeddings(pil_images)
            n_image_tokens = image_embeds.shape[0] * image_embeds.shape[1]

        video_embeds = None
        n_video_tokens = 0
        if video is not None:
            video_embeds = self._extract_video_embeddings(str(video))
            n_video_tokens = video_embeds.shape[0] * video_embeds.shape[1]

        audio_embeds = None
        n_audio_tokens = 0
        if audio is not None:
            audio_embeds = self._extract_audio_embeddings(str(audio))
            n_audio_tokens = audio_embeds.shape[0] * audio_embeds.shape[1]

        prompt = self._build_prompt(
            text,
            n_image_tokens=n_image_tokens,
            n_video_tokens=n_video_tokens,
            n_audio_tokens=n_audio_tokens,
        )
        input_ids = self.tokenizer(prompt, return_tensors="np")["input_ids"]

        mx = self.mx
        ids_mx = mx.array(input_ids)
        text_embeds = self.mlx_model.backbone.embeddings(ids_mx)
        text_embeds = mx.array(text_embeds)
        text_embeds = self._inject_embeddings(
            input_ids, text_embeds,
            image_embeds, video_embeds, audio_embeds,
        )

        return self._inline_decode(
            text_embeds, ids_mx, max_tokens, temperature, top_p,
        )

    def _inline_decode(
        self, inputs_embeds, input_ids, max_tokens, temperature, top_p,
    ):
        """Inline reimpl of mlx_lm.nemotron_h.Model.__call__ that takes
        inputs_embeds for the prefill step, then switches to id-based decode."""
        mx = self.mx
        model = self.mlx_model
        backbone = model.backbone

        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        cache = model.make_cache()

        # Prefill via inputs_embeds (skip the embed lookup)
        h = inputs_embeds
        attn_mask = create_attention_mask(h, cache[backbone.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[backbone.ssm_idx])
        ci = 0
        for layer in backbone.layers:
            if layer.block_type in ("M", "*"):
                c = cache[ci]; ci += 1
                mask_l = attn_mask if layer.block_type == "*" else ssm_mask
                h = layer(h, mask=mask_l, cache=c)
            else:
                h = layer(h)
        h = backbone.norm_f(h)
        logits = model.lm_head(h)
        next_logit = logits[:, -1, :]

        def sample(logit, temp, tp):
            if temp <= 0:
                return mx.argmax(logit, axis=-1, keepdims=True)
            logit = logit / temp
            if tp >= 1.0:
                return mx.random.categorical(logit)[..., None]
            # top-p (nucleus): keep tokens whose cumulative prob ≤ tp
            sorted_idx = mx.argsort(-logit, axis=-1)
            sorted_logits = mx.take_along_axis(logit, sorted_idx, axis=-1)
            sorted_probs = mx.softmax(sorted_logits, axis=-1)
            cumprobs = mx.cumsum(sorted_probs, axis=-1)
            # Always keep at least the top-1 (shift mask right by 1).
            keep = mx.concatenate(
                [mx.ones_like(cumprobs[..., :1]) > 0, cumprobs[..., :-1] <= tp],
                axis=-1,
            )
            neg_inf = mx.full(sorted_logits.shape, -1e9, dtype=sorted_logits.dtype)
            filtered = mx.where(keep, sorted_logits, neg_inf)
            tok_in_sorted = mx.random.categorical(filtered)[..., None]
            return mx.take_along_axis(sorted_idx, tok_in_sorted, axis=-1)

        eos_ids = set([11])
        tokens = []
        tok = sample(next_logit, temperature, top_p)
        tokens.append(int(tok.item()))

        for _ in range(max_tokens - 1):
            if tokens[-1] in eos_ids:
                break
            h = backbone.embeddings(tok)
            attn_mask = create_attention_mask(h, cache[backbone.fa_idx])
            ssm_mask = create_ssm_mask(h, cache[backbone.ssm_idx])
            ci = 0
            for layer in backbone.layers:
                if layer.block_type in ("M", "*"):
                    c = cache[ci]; ci += 1
                    mask_l = attn_mask if layer.block_type == "*" else ssm_mask
                    h = layer(h, mask=mask_l, cache=c)
                else:
                    h = layer(h)
            h = backbone.norm_f(h)
            logits = model.lm_head(h)
            tok = sample(logits[:, -1, :], temperature, top_p)
            tokens.append(int(tok.item()))

        return self.tokenizer.decode(tokens, skip_special_tokens=True)


def main():
    if len(sys.argv) < 2:
        print(
            "usage: python -m jang_tools.nemotron_omni_chat <bundle_path> [prompt] "
            "[--image PATH] [--audio PATH] [--video PATH]",
            file=sys.stderr,
        )
        sys.exit(2)
    bundle_path = sys.argv[1]
    args = sys.argv[2:]
    prompt = args[0] if args and not args[0].startswith("--") else "What is the capital of France?"
    images, audio, video = [], None, None
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--image" and i + 1 < len(args):
            images.append(args[i+1]); i += 2
        elif a == "--audio" and i + 1 < len(args):
            audio = args[i+1]; i += 2
        elif a == "--video" and i + 1 < len(args):
            video = args[i+1]; i += 2
        else:
            i += 1

    chat = OmniChat(bundle_path=bundle_path)
    print(chat.chat(
        prompt, images=images or None, audio=audio, video=video, max_tokens=80,
    ))


if __name__ == "__main__":
    main()
