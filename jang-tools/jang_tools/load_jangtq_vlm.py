"""Load a JANGTQ-quantized VLM (e.g., Qwen 3.5/3.6 Vision MoE).

Sibling of `load_jangtq.load_jangtq_model` but builds the model skeleton via
`mlx_vlm` so the vision_tower + processor are wired up. Returns a tuple
`(model, processor)` ready for `mlx_vlm.generate`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

# We import the heavy lifters from load_jangtq so behavior stays in one place.
# Anything that touches model internals (TQ replacement, SwitchGLU monkey
# patch, router compile, MLA-bits fix, P18 QKV fusion, _fix_quantized_bits)
# is re-used by calling a helper carved out of load_jangtq_model.


def _affine_quantize_mode(quantization: dict | None) -> str:
    """Return the MLX affine quantization mode for non-TQ VLM modules.

    JANGTQ VLM bundles use ``mode="affine+mxtq"`` at the container level:
    affine modules such as embeddings, attention projections, and lm_head are
    still ordinary MLX affine quantized weights, while routed experts are later
    replaced by TurboQuant modules.  Passing the container mode directly to
    ``nn.quantize`` raises ``KeyError`` in MLX and can also mask bundle/runtime
    drift.  Normalize only the container marker; leave real affine modes alone.
    """

    if not isinstance(quantization, dict):
        return "affine"
    mode = str(quantization.get("mode", "affine")).lower()
    if mode == "affine+mxtq":
        return "affine"
    return mode


def _vlm_quant_weight_key_candidates(module_path: str, model_type: str = "") -> set[str]:
    """Return possible on-disk ``.scales`` keys for a VLM module path."""

    candidates = {f"{module_path}.scales"}
    if str(model_type or "").lower() == "zaya1_vl":
        if module_path.startswith("language_model.model."):
            raw = module_path.replace("language_model.model.", "model.", 1)
            candidates.add(f"{raw}.scales")
            if ".mlp.zaya_block." in raw:
                candidates.add(f"{raw.replace('.mlp.zaya_block.', '.zaya_block.', 1)}.scales")
        if module_path.startswith("vision_tower."):
            candidates.add(f"model.visual{module_path[len('vision_tower'):]}.scales")
            candidates.add(f"model.vision_tower{module_path[len('vision_tower'):]}.scales")
    return candidates


class _Zaya1VLImageProcessorProxy:
    """Delegate image preprocessing while bypassing mlx-vlm's BaseImageProcessor shortcut."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __call__(self, *args, **kwargs):
        return self._inner(*args, **kwargs)

    def preprocess(self, *args, **kwargs):
        return self._inner.preprocess(*args, **kwargs)


class _Zaya1VLProcessor:
    def __init__(self, tokenizer, image_processor, chat_template: str | None):
        self.tokenizer = tokenizer
        self._image_processor = image_processor
        self.image_processor = _Zaya1VLImageProcessorProxy(image_processor)
        self.chat_template = chat_template
        self.image_token = "<image>"
        self.video_token = "<video>"

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def _normalize_content(self, content):
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if not isinstance(content, list):
            if content is None:
                return []
            return [{"type": "text", "text": str(content)}]

        normalized = []
        for item in content:
            if isinstance(item, str):
                normalized.append({"type": "text", "text": item})
                continue
            if not isinstance(item, dict):
                normalized.append({"type": "text", "text": str(item)})
                continue
            item_type = str(item.get("type", ""))
            if item_type in ("image", "image_url", "input_image"):
                normalized.append({"type": "image"})
            elif item_type in ("text", "input_text"):
                text = item.get("text", "") or item.get("content", "")
                normalized.append({"type": "text", "text": str(text)})
            else:
                text = item.get("text", "") or item.get("content", "")
                if text:
                    normalized.append({"type": "text", "text": str(text)})
        return normalized

    def _normalize_messages(self, messages):
        return [
            {**message, "content": self._normalize_content(message.get("content"))}
            if isinstance(message, dict)
            else message
            for message in messages
        ]

    def apply_chat_template(self, messages, *args, **kwargs):
        messages = self._normalize_messages(messages)
        if self.chat_template is not None:
            kwargs.setdefault("chat_template", self.chat_template)
        return self.tokenizer.apply_chat_template(messages, *args, **kwargs)

    def __call__(
        self,
        text=None,
        images=None,
        padding=True,
        padding_side="left",
        add_special_tokens=True,
        return_tensors="np",
        **kwargs,
    ):
        kwargs.pop("audio", None)
        kwargs.pop("audios", None)
        if images is None:
            return self.tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                padding_side=padding_side,
                return_tensors=return_tensors,
                **kwargs,
            )

        import numpy as np

        if not isinstance(images, list):
            images = [images]
        prompts = text if isinstance(text, list) else [text]
        vision = self._image_processor(images=images, return_tensors="np")
        grids = np.asarray(vision["image_grid_thw"], dtype=np.int64)
        merge = int(getattr(self._image_processor, "merge_size", 2) or 2)
        image_repeats = [
            int(t * h * w // (merge * merge)) for t, h, w in grids
        ]
        image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)

        expanded_ids = []
        image_cursor = 0
        for prompt in prompts:
            ids = self.tokenizer.encode(
                prompt or "",
                add_special_tokens=add_special_tokens,
            )
            out_ids = []
            for token_id in ids:
                if token_id == image_token_id:
                    if image_cursor >= len(image_repeats):
                        raise ValueError(
                            "ZAYA1-VL prompt has more <image> tokens than images"
                        )
                    out_ids.extend([image_token_id] * image_repeats[image_cursor])
                    image_cursor += 1
                else:
                    out_ids.append(token_id)
            expanded_ids.append(out_ids)
        if image_cursor != len(image_repeats):
            raise ValueError("ZAYA1-VL image count does not match <image> tokens in prompt")

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        max_len = max(len(ids) for ids in expanded_ids)
        input_ids = []
        attention = []
        for ids in expanded_ids:
            pad = [pad_id] * (max_len - len(ids))
            if padding and padding_side == "left":
                row = pad + ids
                mask = [0] * len(pad) + [1] * len(ids)
            elif padding:
                row = ids + pad
                mask = [1] * len(ids) + [0] * len(pad)
            else:
                row = ids
                mask = [1] * len(ids)
            input_ids.append(row)
            attention.append(mask)

        return {
            "input_ids": np.asarray(input_ids, dtype=np.int64),
            "attention_mask": np.asarray(attention, dtype=np.int64),
            "pixel_values": vision["pixel_values"],
            "image_grid_thw": grids,
        }


def _load_zaya1_vl_chat_template(model_path: Path) -> str | None:
    chat_template_path = model_path / "chat_template.json"
    if chat_template_path.exists():
        data = json.loads(chat_template_path.read_text())
        if isinstance(data, dict) and data.get("chat_template"):
            return data["chat_template"]
    tok_config_path = model_path / "tokenizer_config.json"
    if tok_config_path.exists():
        data = json.loads(tok_config_path.read_text())
        if isinstance(data, dict):
            return data.get("chat_template")
    return None


def _build_zaya1_vl_processor(model_path: Path):
    from transformers import AutoImageProcessor, AutoTokenizer
    from mlx_vlm.tokenizer_utils import load_tokenizer as vlm_load_tokenizer
    from mlx_vlm.utils import StoppingCriteria

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    chat_template = _load_zaya1_vl_chat_template(model_path)
    if chat_template is not None:
        tokenizer.chat_template = chat_template
    image_processor = AutoImageProcessor.from_pretrained(model_path)
    processor = _Zaya1VLProcessor(tokenizer, image_processor, chat_template)
    detokenizer_class = vlm_load_tokenizer(model_path, return_tokenizer=False)
    processor.detokenizer = detokenizer_class(tokenizer)
    tokenizer.stopping_criteria = StoppingCriteria(tokenizer.eos_token_id, tokenizer)
    return processor


def _mlx_vlm_skeleton(model_path: Path):
    """Build the mlx_vlm model + processor without loading weights.

    Replicates the relevant parts of `mlx_vlm.utils.load_model` and
    `mlx_vlm.utils.load` up to the point where weights would be loaded,
    plus the `nn.quantize` step (without it, affine-quantized modules like
    `embed_tokens`, `q_proj`, `shared_expert.*` would stay as plain
    Embedding/Linear and silently consume packed uint32 weights as if they
    were floats — producing 512-d embeddings instead of 2048-d).
    """
    from mlx_vlm.utils import (
        load_config, get_model_and_args, update_module_configs,
        load_processor,
    )
    config = load_config(model_path)
    if str(config.get("model_type", "")).lower() == "zaya1_vl":
        import jang_tools.zaya1_vl  # noqa: F401  (registers on import)

    model_class, _ = get_model_and_args(config=config)

    config.setdefault("text_config", config.pop("llm_config", {}))
    config.setdefault("vision_config", {})
    config.setdefault("audio_config", {})

    model_config = model_class.ModelConfig.from_dict(config)
    model_config = update_module_configs(
        model_config, model_class, config,
        ["text", "vision", "perceiver", "projector", "audio"],
    )

    model = model_class.Model(model_config)

    # Apply nn.quantize for affine-quantized modules. Use the same
    # class_predicate pattern as mlx_vlm.utils.load_model: only quantize
    # modules whose `.scales` key actually exists in the on-disk weights.
    # TQ-replaced switch_mlp modules will be skipped (no `.scales`); they
    # get handled later by `_hydrate_jangtq_model`'s TurboQuant pass.
    quantization = config.get("quantization")
    if quantization is not None:
        from safetensors import safe_open
        weight_files = sorted(model_path.glob("model-*.safetensors"))
        # Cheap weight-key index — just enumerate keys, don't materialize tensors.
        weight_keys = set()
        for wf in weight_files:
            with safe_open(str(wf), framework="numpy") as f:
                weight_keys.update(f.keys())

        # mlx_vlm uses a shared sanitize-aware key-rewrite. Our artifact is
        # already in post-sanitize form so the keys we'll see are
        # `language_model.model.embed_tokens.weight` etc. The class_predicate
        # receives the *module path*, which mirrors the post-sanitize key.
        def get_class_predicate(p, m):
            if p in quantization:
                return quantization[p]
            if not hasattr(m, "to_quantized"):
                return False
            if hasattr(m, "weight") and m.weight.size % 64 != 0:
                return False
            return bool(
                _vlm_quant_weight_key_candidates(
                    p,
                    str(config.get("model_type", "")),
                )
                & weight_keys
            )

        nn.quantize(
            model,
            group_size=quantization.get("group_size", 64),
            bits=quantization.get("bits", 2),
            mode=_affine_quantize_mode(quantization),
            class_predicate=get_class_predicate,
        )

    # Kimi K2.6 (and other recent VLMs) ship a custom processor
    # (`kimi_k25_processor.py` etc.) that AutoProcessor refuses to load
    # without trust_remote_code. Bundles in this loader are opt-in — we're
    # already executing their safetensors — so forward trust_remote_code=True.
    if str(config.get("model_type", "")).lower() == "zaya1_vl":
        processor = _build_zaya1_vl_processor(model_path)
    else:
        processor = load_processor(
            model_path, add_generation_prompt=True, trust_remote_code=True,
        )
    _install_video_fallback(processor)
    return model, processor, config, model_config


def _install_video_fallback(processor):
    """Enable video inputs without torchvision.

    transformers' AutoVideoProcessor requires torchvision, which is not in
    the bundled Python vMLX ships. Qwen3VLProcessor.__call__ then falls back
    to ``self.image_processor(videos=videos)`` which raises TypeError because
    Qwen2VLImageProcessor's __call__ has no ``videos`` kwarg.

    This adapter routes video frames through image_processor (one image per
    frame) and promotes the resulting grid to video_grid_thw. Temporal merging
    (temporal_patch_size) is preserved by rewriting the grid's time axis so
    the <|video_pad|> expansion in the chat template produces the correct
    token count. Callers still pass ``videos=[[frame1, frame2, ...]]`` — the
    wrapper then converts internally.
    """
    if not hasattr(processor, "image_processor"):
        return

    ip = processor.image_processor
    proc_cls = processor.__class__
    orig_call = proc_cls.__call__
    if getattr(orig_call, "_jangtq_video_fallback", False):
        return  # already patched on this class

    temporal_patch = int(getattr(ip, "temporal_patch_size", 2) or 2)

    def _is_video_processor_dependency_error(exc: Exception) -> bool:
        if isinstance(exc, (ImportError, ModuleNotFoundError)):
            return True
        msg = str(exc)
        if not msg:
            return False
        return (
            "PyAV is not installed" in msg
            or "PyAV" in msg
            or "video operations in torchvision" in msg
            or "torchvision.io" in msg
            and "read_video" in msg
        )

    def _is_video_processor_frame_list_error(exc: Exception) -> bool:
        msg = str(exc)
        return (
            isinstance(exc, ValueError)
            and "Expected video as (T, C, H, W)" in msg
            and "got shape" in msg
        )

    def _patched_call(self, images=None, text=None, videos=None, **kwargs):
        if videos is None:
            return orig_call(self, images=images, text=text, videos=videos, **kwargs)

        video_processor = getattr(self, "video_processor", None)
        if video_processor is None:
            pass
        else:
            try:
                return orig_call(self, images=images, text=text, videos=videos, **kwargs)
            except Exception as exc:  # pragma: no cover - environment dependent
                if not (
                    _is_video_processor_dependency_error(exc)
                    or _is_video_processor_frame_list_error(exc)
                ):
                    raise
                # Fall through to image-processor fallback for optional-dependency
                # missing, or mlx-vlm video processors that expect a pre-decoded
                # (T, C, H, W) array while callers pass frame path lists.

        # Lift each video's frames into a flat image batch, one row per frame.
        # image_processor produces image_grid_thw shape (sum_frames, 3) with t=1.
        # We collapse each video's rows into a single row with
        # t = ceil(n_frames / temporal_patch_size), keeping h, w from the
        # first frame. Preserves total patch count when n_frames % tp == 0.
        import mlx.core as _mx
        import numpy as _np

        flat_frames = []
        frames_per_video = []
        for v in videos:
            fs = v if isinstance(v, (list, tuple)) else [v]
            flat_frames.extend(fs)
            frames_per_video.append(len(fs))

        image_out = ip(images=flat_frames)
        pv = image_out["pixel_values"]           # (sum_patches, D)
        grid = image_out["image_grid_thw"]       # (sum_frames, 3)  t=1 each

        # Build video_grid_thw per video by merging frames along t-axis.
        g_np = grid if isinstance(grid, _np.ndarray) else _np.asarray(grid)
        video_rows = []
        cursor = 0
        for n in frames_per_video:
            if n == 0:
                continue
            frame_rows = g_np[cursor:cursor + n]  # (n, 3)
            cursor += n
            h, w = int(frame_rows[0, 1]), int(frame_rows[0, 2])
            t_patches = max(1, (n + temporal_patch - 1) // temporal_patch)
            video_rows.append([t_patches, h, w])
        v_grid = _np.asarray(video_rows, dtype=_np.int64)

        videos_inputs_synth = {
            "pixel_values_videos": _mx.array(pv) if not isinstance(pv, _mx.array) else pv,
            "video_grid_thw": _mx.array(v_grid),
        }

        # Splice synthesized video_inputs into the parent __call__ by keeping
        # videos=None (so original path skips real video_processor) and
        # post-merging. Simplest: call parent with images only, then merge.
        base = orig_call(self, images=images, text=text, videos=None, **kwargs)

        # Token-expansion for <|video_pad|>: same math as the real path.
        # If `base` already tokenized text without video markers, we must
        # re-tokenize with expanded markers. Easiest: do the text-mutation
        # here before calling orig_call, not after. Restart the call.
        #
        # Because tokenization happens inside orig_call, we re-enter with
        # the pre-expanded text.
        if text is not None:
            texts = text if isinstance(text, list) else [text]
            mutated = []
            merge_len = ip.merge_size ** 2
            idx = 0
            for t_str in texts:
                s = t_str
                while self.video_token in s and idx < len(video_rows):
                    num_video_tokens = int(_np.prod(v_grid[idx])) // merge_len
                    s = s.replace(self.video_token,
                                  "<|placeholder|>" * num_video_tokens, 1)
                    idx += 1
                s = s.replace("<|placeholder|>", self.video_token)
                mutated.append(s)
            # Re-tokenize with expanded markers, drop video token-expansion
            # that orig_call would have done (we already did it).
            tok_kwargs = dict(kwargs)
            tok_kwargs.pop("return_tensors", None)
            tokenized = self.tokenizer(mutated, **tok_kwargs)
            # Re-wrap as BatchFeature like orig_call does.
            from transformers.feature_extraction_utils import BatchFeature
            from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import to_mlx
            merged = {**tokenized, **videos_inputs_synth}
            if images is not None:
                # image_inputs already merged into `base`; copy the image keys.
                for k in ("pixel_values", "image_grid_thw"):
                    if k in base:
                        merged[k] = base[k]
            return BatchFeature(data=to_mlx(merged))

        # No text — just merge and return
        for k, v in videos_inputs_synth.items():
            base[k] = v
        return base

    _patched_call._jangtq_video_fallback = True
    proc_cls.__call__ = _patched_call


def load_jangtq_vlm_model(model_path) -> Tuple[nn.Module, object]:
    """Load JANGTQ VLM (model + processor)."""
    model_path = Path(model_path)
    # M125 (iter 48): context-manage reads so fds close deterministically.
    with open(model_path / "config.json") as f:
        config = json.load(f)
    jang_cfg_path = model_path / "jang_config.json"
    if jang_cfg_path.exists():
        with open(jang_cfg_path) as f:
            jang_cfg = json.load(f)
    else:
        jang_cfg = {}
    mxtq_seed = jang_cfg.get("mxtq_seed", 42)
    mxtq_bits_map = jang_cfg.get("mxtq_bits", {})

    print(f"Loading JANGTQ VLM: {model_path.name}", flush=True)
    print(f"  seed={mxtq_seed}, bits_map={mxtq_bits_map}", flush=True)

    model, processor, _, model_config = _mlx_vlm_skeleton(model_path)
    print(f"  Built mlx_vlm skeleton: {type(model).__name__}", flush=True)
    vision_module = getattr(model, "vision_tower", None) or getattr(model, "visual", None)
    vision_config = getattr(vision_module, "config", None)
    vision_depth = getattr(vision_config, "depth", None)
    if vision_depth is not None:
        print(f"  vision module depth: {vision_depth} layers", flush=True)
    else:
        print("  vision module depth: unavailable", flush=True)
    text_config = getattr(getattr(model, "config", None), "text_config", None)
    text_layers = getattr(text_config, "num_hidden_layers", None)
    if text_layers is not None:
        print(f"  language_model layers: {text_layers}", flush=True)
    else:
        print("  language_model layers: unavailable", flush=True)

    # Now apply the same TQ + quantization treatment as load_jangtq_model.
    # We delegate to a helper inside load_jangtq that does everything *except*
    # building the skeleton.
    from jang_tools.load_jangtq import _hydrate_jangtq_model
    _hydrate_jangtq_model(
        model=model,
        model_path=model_path,
        mxtq_seed=mxtq_seed,
        mxtq_bits_map=mxtq_bits_map,
        model_config=model_config,
    )
    return model, processor
