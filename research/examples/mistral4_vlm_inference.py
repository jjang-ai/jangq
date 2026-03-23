"""
Mistral Small 4 (119B) — VLM (Vision) Inference Example
JANG quantized model with Pixtral vision on Apple Silicon via MLX

Requirements:
  pip install mlx mlx-lm mlx-vlm>=0.4.1 transformers safetensors
  pip install jang  # or add jang-tools to path

Model directory must contain:
  - config.json, jang_config.json, tokenizer files
  - processor_config.json, preprocessor_config.json (for VLM)
  - model-*.safetensors (weight shards)
  - Vision conv weights must be in MLX OHWI format

Created by Jinho Jang (eric@jangq.ai)
"""
import sys
import json
import gc
from pathlib import Path

sys.path.insert(0, "/Users/eric/jang/jang-tools")  # Adjust path

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.utils import (
    get_model_and_args, update_module_configs,
    skip_multimodal_module, load_processor,
)
from mlx_vlm import generate
from mlx_vlm.prompt_utils import apply_chat_template
from jang_tools.loader import _get_v2_weight_files, _fix_quantized_bits

MODEL_PATH = Path("/Users/eric/models/Mistral-Small-4-119B-JANG_2L")


def load_jang_vlm(model_path):
    """Load JANG Mistral 4 as a full VLM (vision + text)."""
    model_path = Path(model_path)
    vlm_config = json.loads((model_path / "config.json").read_text())

    # Create VLM model (vision + text + projector)
    model_class, _ = get_model_and_args(config=vlm_config)
    model_config = model_class.ModelConfig.from_dict(vlm_config)
    modules = ["text", "vision", "perceiver", "projector", "audio"]
    model_config = update_module_configs(
        model_config, model_class, vlm_config, modules
    )
    model = model_class.Model(model_config)

    # Quantize text layers only (skip vision encoder + MoE gate)
    def class_predicate(path, module):
        if skip_multimodal_module(path):
            return False
        if "gate" in path and "gate_proj" not in path:
            return False
        return hasattr(module, "to_quantized")

    qbits = vlm_config.get("quantization", {}).get("bits", 2)
    nn.quantize(model, group_size=64, bits=qbits, class_predicate=class_predicate)

    # Load all weights (text + vision)
    weight_files = _get_v2_weight_files(model_path)
    for sf in weight_files:
        weights = mx.load(str(sf))
        clean = {
            k: v for k, v in weights.items()
            if not k.endswith(".importance")
            and not k.startswith("mtp.")
            and "activation_scale" not in k
            and "scale_inv" not in k
        }
        try:
            clean = model.sanitize(clean)
        except (KeyError, ValueError):
            pass
        model.load_weights(list(clean.items()), strict=False)
        del clean, weights
        gc.collect()

    _fix_quantized_bits(model, {})
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())

    # Load processor
    processor = load_processor(model_path, processor_config=vlm_config)
    model.config = model_config

    return model, processor, model_config


if __name__ == "__main__":
    print("Loading JANG VLM...")
    model, processor, config = load_jang_vlm(MODEL_PATH)
    print("Loaded!")

    # Test with image
    image_path = str(MODEL_PATH / "images" / "aime.png")

    prompt = apply_chat_template(
        processor, config=config,
        prompt="Describe what you see in this image briefly.",
        images=[image_path],
    )
    print(f"\nPrompt: {prompt[:80]}...")

    output = generate(
        model, processor, prompt,
        max_tokens=100, verbose=True,
        image=[image_path],
    )
    print(f"\nVLM Output: {output.text}")

    # Text-only through VLM model (also works)
    prompt_text = apply_chat_template(
        processor, config=config,
        prompt="What is the capital of Japan?",
    )
    output_text = generate(model, processor, prompt_text, max_tokens=20, verbose=True)
    print(f"\nText Output: {output_text.text}")
