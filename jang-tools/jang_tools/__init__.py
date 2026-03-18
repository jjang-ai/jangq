"""
JANG — Jang Adaptive N-bit Grading
Mixed-Precision Quantization for Apple Silicon
Created by Jinho Jang (eric@jangq.ai)

The GGUF equivalent for MLX. Models stay quantized in GPU memory
and dequantize on-the-fly during inference at full Metal speed.

v2.0: MLX-native safetensors — models load instantly via mmap.

Quick start:
    from jang_tools import convert_model, JANG_PROFILES, profile_for_bits

    # Convert any HuggingFace model
    convert_model("path/to/model", "output/path", profile="JANG_2L")

    # Or pick a profile by number (1-8)
    profile = profile_for_bits(2)  # → "JANG_2S"
"""

__version__ = "2.1.1"
__author__ = "Jinho Jang"
__email__ = "eric@jangq.ai"

# Core conversion
from .convert import convert_model

# Profiles and allocation
from .allocate import (
    JANG_PROFILES,
    JANG_K_TARGETS,
    BIT_TO_PROFILE,
    Tier,
    classify_tensor,
    allocate_bits_profile,
    allocate_bits_budget,
    profile_for_bits,
    is_k_quant,
    k_quant_target,
    estimate_size_gb,
)

# Quantization
from .quantize import QuantizedTensor, quantize_tensor, dequantize_tensor

# Format I/O
from .format.reader import load_jang_model, is_jang_model, JANGModel
from .format.writer import write_jang_model, write_jang_v2_model

# Calibration
from .calibrate import calibrate_from_weights

# Architecture detection
from .architectures import detect_architecture, ArchConfig, ArchType

# Inference loader (requires mlx, mlx-lm — Apple Silicon only)
try:
    from .loader import load_jang_model as load_for_inference
    from .loader import load_jang_vlm_model, upgrade_v1_to_v2
except ImportError:
    load_for_inference = None
    load_jang_vlm_model = None
    upgrade_v1_to_v2 = None

__all__ = [
    # Conversion
    "convert_model",
    # Profiles
    "JANG_PROFILES",
    "JANG_K_TARGETS",
    "BIT_TO_PROFILE",
    "Tier",
    "classify_tensor",
    "allocate_bits_profile",
    "allocate_bits_budget",
    "profile_for_bits",
    "is_k_quant",
    "k_quant_target",
    "estimate_size_gb",
    # Quantization
    "QuantizedTensor",
    "quantize_tensor",
    "dequantize_tensor",
    # Format I/O
    "load_jang_model",
    "is_jang_model",
    "JANGModel",
    "write_jang_model",
    "write_jang_v2_model",
    # Calibration
    "calibrate_from_weights",
    # Architecture
    "detect_architecture",
    "ArchConfig",
    "ArchType",
    # Inference (None if mlx not installed)
    "load_for_inference",
    "load_jang_vlm_model",
    "upgrade_v1_to_v2",
]
