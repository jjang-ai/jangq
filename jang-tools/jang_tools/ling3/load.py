"""Load a Ling-3.0 / BailingMoeV3 checkpoint into the MLX model.

Created by Jinho Jang (eric@jangq.ai)
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from jang_tools.ling3.model import Model, ModelArgs


def load_config(path: str | Path) -> ModelArgs:
    with open(Path(path) / "config.json") as f:
        return ModelArgs.from_dict(json.load(f))


def load_weights(path: str | Path) -> dict[str, mx.array]:
    files = sorted(glob.glob(str(Path(path) / "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors under {path}")
    weights: dict[str, mx.array] = {}
    for f in files:
        weights.update(mx.load(f))
    return weights


def load_model(path: str | Path, dtype: mx.Dtype = mx.bfloat16, lazy: bool = False) -> Model:
    """Build the model and load the checkpoint.

    Raises if the checkpoint and the module tree disagree in either direction.
    A silently-dropped tensor is the failure mode that produces a model which
    loads, runs, and is quietly wrong — so this refuses to be lenient.
    """
    args = load_config(path)
    model = Model(args)
    weights = model.sanitize(load_weights(path))

    expected = dict(tree_flatten(model.parameters()))
    missing = sorted(set(expected) - set(weights))
    unexpected = sorted(set(weights) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"checkpoint/module mismatch\n"
            f"  missing ({len(missing)}): {missing[:8]}\n"
            f"  unexpected ({len(unexpected)}): {unexpected[:8]}"
        )

    bad_shape = [
        (k, tuple(weights[k].shape), tuple(expected[k].shape))
        for k in expected
        if tuple(weights[k].shape) != tuple(expected[k].shape)
    ]
    if bad_shape:
        raise ValueError(f"shape mismatch on {len(bad_shape)} tensors: {bad_shape[:8]}")

    # Router and KDA gate params stay in high precision — their error compounds
    # through a recurrence or flips which experts run, rather than averaging out.
    keep_fp32 = (".gate.weight", ".gate.expert_bias", ".A_log", ".dt_bias")
    weights = {
        k: (v.astype(mx.float32) if k.endswith(keep_fp32) else v.astype(dtype))
        for k, v in weights.items()
    }

    model.update(tree_unflatten(list(weights.items())))
    if not lazy:
        mx.eval(model.parameters())
    model.eval()
    return model
