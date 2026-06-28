"""Self-contained loader + generation for MiMo-V2.5 JANG VLM bundles.

Mirrors mlx_lm's proven quantized-bundle loading (per-module overrides) for
the text side and hydrates the bf16 vision tower, avoiding mlx_vlm's loader
entirely. Generation is a small prefill-with-embeds + cached decode loop on
the existing text decoder.
"""

from __future__ import annotations

import glob
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .model import Model


def load_vlm(bundle: str | Path):
    bundle = Path(bundle)
    config = json.loads((bundle / "config.json").read_text())
    model_config = ModelConfig.from_dict(config)
    model = Model(model_config)

    weights: dict[str, mx.array] = {}
    for f in sorted(glob.glob(str(bundle / "model-*.safetensors"))):
        weights.update(mx.load(f))
    weights = model.sanitize(weights)

    quantization = config.get("quantization")
    if quantization is not None:
        # Manual module replacement: nn.quantize's update_modules silently
        # drops conversions when the predicate is mixed (some False) on this
        # tree shape, so we convert and re-assign through parents directly.
        def convert(parent, attr, module, path):
            override = quantization.get(path)
            if isinstance(override, dict):
                kwargs = dict(override)
            elif hasattr(module, "to_quantized") and f"{path}.scales" in weights:
                kwargs = {"group_size": quantization["group_size"], "bits": quantization["bits"]}
            else:
                return 0
            kwargs.setdefault("mode", quantization.get("mode", "affine"))
            qmod = module.to_quantized(**kwargs)
            if isinstance(parent, list):
                parent[attr] = qmod
            else:
                parent[attr] = qmod  # Module is dict-backed; triggers re-registration
            return 1

        def walk(module, prefix):
            count = 0
            for key, child in list(module.items()):
                path = f"{prefix}.{key}" if prefix else str(key)
                if isinstance(child, nn.Module):
                    n = convert(module, key, child, path)
                    count += n
                    if n == 0:
                        count += walk(child, path)
                elif isinstance(child, list):
                    changed = False
                    for i, item in enumerate(child):
                        ipath = f"{path}.{i}"
                        if isinstance(item, nn.Module):
                            n = convert(child, i, item, ipath)
                            if n:
                                changed = True
                                count += n
                            else:
                                count += walk(item, ipath)
                    if changed:
                        setattr(module, key, child)
            return count

        n_quantized = walk(model, "")
        import os
        if os.environ.get("MIMO_VLM_DEBUG"):
            print(f"manually quantized modules: {n_quantized}", flush=True)

    import os
    if os.environ.get("MIMO_VLM_DEBUG"):
        print("post-quantize types:",
              type(model.model.layers[1].mlp.switch_mlp.gate_proj).__name__,
              type(model.model.layers[1].self_attn.qkv_proj).__name__,
              type(model.model.embed_tokens).__name__, flush=True)
    model.load_weights(list(weights.items()), strict=True)
    model.sync_text_refs()
    mx.eval(model.parameters())
    model.eval()

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(bundle, trust_remote_code=True)
    return model, processor


def generate_vl(
    model: Model,
    processor,
    messages: list[dict],
    images=None,
    videos=None,
    max_tokens: int = 128,
    temp: float = 1.0,
    top_p: float = 0.95,
    enable_thinking: bool = False,
) -> str:
    from mlx_lm.sample_utils import make_sampler

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    inputs = processor(
        text=[text],
        images=images,
        videos=videos,
        return_tensors="np",
    )
    input_ids = mx.array(inputs["input_ids"])
    kwargs = {}
    if "pixel_values" in inputs:
        kwargs["pixel_values"] = mx.array(inputs["pixel_values"])
        kwargs["image_grid_thw"] = mx.array(inputs["image_grid_thw"])
    if "pixel_values_videos" in inputs:
        kwargs["pixel_values_videos"] = mx.array(inputs["pixel_values_videos"])
        kwargs["video_grid_thw"] = mx.array(inputs["video_grid_thw"])

    embeds = model.get_input_embeddings(input_ids, **kwargs)
    cache = model.make_cache()
    sampler = make_sampler(temp=temp, top_p=top_p)

    out = model.text(input_ids, inputs_embeds=embeds, cache=cache)
    logits = out.logits if hasattr(out, "logits") else out
    tok = sampler(logits[:, -1, :].astype(mx.float32) if logits.dtype != mx.float32 else logits[:, -1, :])
    mx.eval(tok)

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    extra_eos = getattr(model.config, "eos_token_id", None)
    if isinstance(extra_eos, int):
        eos_ids.add(extra_eos)
    elif isinstance(extra_eos, (list, tuple)):
        eos_ids.update(int(e) for e in extra_eos)

    generated = []
    cur = tok
    for _ in range(max_tokens):
        tid = int(cur.item())
        if tid in eos_ids:
            break
        generated.append(tid)
        out = model.text(cur.reshape(1, 1), inputs_embeds=None, cache=cache)
        logits = out.logits if hasattr(out, "logits") else out
        cur = sampler(logits[:, -1, :].astype(mx.float32))
        mx.eval(cur)
    return tokenizer.decode(generated)
