"""JANG runtime — unified loader for JANG (mx.quantize affine) + JANGTQ
(TurboQuant) bundles, with auto-detect.

This is the Python reference; vmlx-swift mirrors it under
swift/Sources/JANGRuntime.

Usage:
    from jang_tools.jangrt import load_bundle
    model, cfg, tokenizer = load_bundle("/path/to/JANGQ-AI/Some-Model-JANGTQ2")
    out = model.generate("hello", max_new=32)
"""

from .loader import load_bundle, detect_format
from .linear import JANGLinear, JANGTQLinear

__all__ = ["load_bundle", "detect_format", "JANGLinear", "JANGTQLinear"]
