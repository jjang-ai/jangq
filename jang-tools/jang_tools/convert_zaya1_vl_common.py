"""Shared helpers for Zyphra/ZAYA1-VL-8B conversion.

The VL branch is text+vision. We intentionally keep Vision/LoRA related tensors
as fp16 passthrough and preserve all processor sidecars required by mlx-vlm.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .convert_zaya_common import (
    CAPABILITIES,
    copy_sidecars_with_template,
    is_passthrough as is_text_passthrough,
)


VL_PASSTHROUGH_PREFIXES = (
    "model.visual",
    "model.vision",
    "vision_tower",
    "visual.",
    "vision.",
    "mm_projector",
)

VL_PASSTHROUGH_SUBSTRINGS = (
    ".lora_",
    ".lora.",
    "_lora",
    "vision_projection",
)

VL_SIDECARS = (
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "configuration.json",
)


def zaya1_vl_capabilities() -> dict:
    # VL trunk wraps the same ZAYA LM that reasons by default (measured live:
    # enable_thinking=False default render still produces chain-of-thought).
    # Match the text-bundle stamp so picker UIs and the test suite agree.
    caps = CAPABILITIES.copy()
    caps["family"] = "zaya1_vl"
    caps["modality"] = "vision"
    return caps


def is_passthrough(name: str) -> bool:
    lname = name.lower()
    if is_text_passthrough(name):
        return True
    if any(lname.startswith(prefix) for prefix in VL_PASSTHROUGH_PREFIXES):
        return True
    if any(token in lname for token in VL_PASSTHROUGH_SUBSTRINGS):
        return True
    # Keep ViT projection blocks and visual encoder branches untouched.
    if "vision_tower." in lname:
        return True
    return False


def copy_zaya1_vl_sidecars(src: Path, out: Path) -> None:
    copy_sidecars_with_template(src, out)
    for file_name in VL_SIDECARS:
        src_file = src / file_name
        if src_file.exists():
            shutil.copy2(str(src_file), str(out / file_name))

    # Zaya1-VL releases often include architecture-specific Python files. Preserve
    # them if present; downstream runtime loaders can ignore unknown entries.
    for py_name in sorted(src.glob("configuration_*.py")):
        shutil.copy2(str(py_name), str(out / py_name.name))
    for py_name in sorted(src.glob("modeling_*.py")):
        shutil.copy2(str(py_name), str(out / py_name.name))
