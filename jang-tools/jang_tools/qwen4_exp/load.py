"""Checkpoint loading / sanitization for Qwen4-Exp.

Key transformations (HF checkpoint → MLX module tree):
  - model.language_model.*            → language_model.*
  - mlp.experts.gate_up_proj [E,in,2h]→ mlp.switch_mlp.gate_proj/up_proj [E,h,in]
  - mlp.experts.down_proj    [E,h,in']→ mlp.switch_mlp.down_proj [E,in',h]  (transposed to SwitchLinear layout)
  - ple.ple_embedding.ngram_embedding.shard_{0..127} → ple.ngram_embedding.weight (row concat, numeric order)
  - ple.conv1d.weight [C,1,K]         → ple.conv1d_weight [C,K]
  - linear_attn.conv1d.weight [C,1,K] → [C,K,1] (mlx Conv1d layout, moveaxis)
  - +1 shift on Qwen4ExpTextRMSNorm-family weights (hc_norm, ple norms,
    q/k norms, indexer layernorms). GDN linear_attn.norm NOT shifted.
  - mtp.* and model.visual.* set aside (returned separately).
"""

import glob
import json
import re
from pathlib import Path

import mlx.core as mx

from .modeling import Model, Qwen4ExpTextArgs

# suffixes storing (weight - 1); module uses plain RMSNorm ⇒ add 1 at load
PLUS_ONE_SUFFIXES = (
    ".hc_norm.weight",
    ".norm_key.weight",
    ".norm_query.weight",
    ".norm_conv.weight",
    ".q_norm.weight",
    ".k_norm.weight",
    ".q_layernorm.weight",
    ".k_layernorm.weight",
)

_SHARD_RE = re.compile(r"^(.*\.ple)\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$")


def sanitize(weights: dict, args: Qwen4ExpTextArgs, keep_visual: bool = False):
    """Returns (text_weights, mtp_weights, visual_weights)."""
    text, mtp, visual = {}, {}, {}

    for k, v in weights.items():
        if k.startswith("mtp."):
            mtp[k] = v
            continue
        if k.startswith("model.visual."):
            if keep_visual:
                visual[k.replace("model.visual.", "visual.")] = v
            continue
        if k.startswith("model.language_model."):
            k = k.replace("model.language_model.", "language_model.")
        m = _SHARD_RE.match(k)
        if m:
            # keep the checkpoint row-shards — the runtime embedding is
            # sharded so lookups only page in gathered rows (never concat:
            # that materializes ~95 GiB)
            text[f"{m.group(1)}.ngram_embedding.shards.{int(m.group(2))}.weight"] = v
            continue
        if k.endswith("ple.ple_embedding.layer_multipliers") or k.endswith(
            "ple.ple_embedding.ngram_heads_offsets"
        ) or k.endswith("ple.ple_embedding.ngram_heads_vocab_sizes"):
            # recomputed exactly from config (verified vs HF); keep for assertion
            text["__buffer__" + k] = v
            continue
        if k.endswith("ple.conv1d.weight"):
            # [C,1,K] → [C,K]
            text[k.replace("ple.conv1d.weight", "ple.conv1d_weight")] = v.squeeze(1)
            continue
        if k.endswith("linear_attn.conv1d.weight") and v.shape[-1] != 1:
            text[k] = v.moveaxis(2, 1)
            continue
        if k.endswith("mlp.experts.gate_up_proj"):
            # HF layout [E, out=2h, in] (nn.Linear-style, verified vs HF tiny
            # model parity) → split the out axis into gate / up halves.
            two_h = v.shape[1]
            h = two_h // 2
            base = k.replace("mlp.experts.gate_up_proj", "mlp.switch_mlp")
            text[base + ".gate_proj.weight"] = v[:, :h, :]
            text[base + ".up_proj.weight"] = v[:, h:, :]
            continue
        if k.endswith("mlp.experts.down_proj"):
            # HF [E, out=hidden, in=h] == SwitchLinear layout; pass through
            text[k.replace("mlp.experts.down_proj", "mlp.switch_mlp.down_proj.weight")] = v
            continue
        text[k] = v

    for k in list(text):
        if any(k.endswith(sfx) for sfx in PLUS_ONE_SUFFIXES):
            text[k] = text[k] + 1.0

    return text, mtp, visual


def load_config(model_dir: str) -> dict:
    with open(Path(model_dir) / "config.json") as f:
        return json.load(f)


def load_bundle(bundle_dir: str, lazy: bool = True) -> Model:
    """Load a JANG v2 qwen4_exp bundle (runtime naming, +1 already applied,
    mixed-bit quantization recorded in config.jang_config.bit_map)."""
    import mlx.nn as nn

    cfg = load_config(bundle_dir)
    args = Qwen4ExpTextArgs.from_config(cfg)
    model = Model(args)

    weights = {}
    for f in sorted(glob.glob(str(Path(bundle_dir) / "model-*.safetensors"))):
        weights.update(mx.load(f))
    # bundles ship visual.* and mtp.* (contract: always include); the text
    # runtime loads only its own modules
    weights = {k: v for k, v in weights.items()
               if not k.startswith(("visual.", "mtp."))}

    # n-gram hash buffers are stored for self-description; the runtime
    # recomputes them from config — verify then drop
    ple = model.language_model.layers[args.ple_layer_ids[0] - 1].ple
    import numpy as np

    for name, ref in (
        ("layer_multipliers", ple.hasher.layer_multipliers),
        ("ngram_heads_offsets", np.array(ple.hasher.head_offsets)),
        ("ngram_heads_vocab_sizes", np.array(ple.hasher.head_vocab_sizes)),
    ):
        for k in [k for k in weights if k.endswith(f"ple.{name}")]:
            got = np.asarray(weights.pop(k)).astype(np.int64)
            assert (got == np.asarray(ref)).all(), f"{name} mismatch vs config"

    quant_bases = {k[: -len(".scales")] for k in weights if k.endswith(".scales")}
    bit_map = (cfg.get("jang_config") or {}).get("bit_map", {})

    from .convert import match_spec  # longest-prefix resolution

    def class_predicate(path, module):
        base = path + ".weight" if hasattr(module, "weight") else path
        if path in quant_bases or base in quant_bases or (path + ".weight") in quant_bases:
            spec = match_spec(path + ".weight", bit_map) if bit_map else None
            if isinstance(spec, dict):
                return {"group_size": spec["group_size"], "bits": spec["bits"]}
            return True
        return False

    nn.quantize(model, class_predicate=class_predicate)
    model.load_weights(list(weights.items()), strict=True)
    if not lazy:
        mx.eval(model.parameters())
    return model


def load_model(model_dir: str, lazy: bool = True, strict: bool = True) -> Model:
    cfg = load_config(model_dir)
    args = Qwen4ExpTextArgs.from_config(cfg)
    model = Model(args)

    weights = {}
    for f in sorted(glob.glob(str(Path(model_dir) / "model-*.safetensors"))):
        weights.update(mx.load(f, return_metadata=False) if False else mx.load(f))

    text, _mtp, _visual = sanitize(weights, args)

    # assert recomputed n-gram buffers match the checkpoint, then drop them
    for k in [k for k in text if k.startswith("__buffer__")]:
        v = text.pop(k)
        name = k.split(".")[-1]
        ple = model.language_model.layers[args.ple_layer_ids[0] - 1].ple
        import numpy as np

        if name == "layer_multipliers":
            ref = ple.hasher.layer_multipliers
        elif name == "ngram_heads_offsets":
            ref = np.array(ple.hasher.head_offsets)
        else:
            ref = np.array(ple.hasher.head_vocab_sizes)
        got = np.asarray(v.astype(mx.int64) if v.dtype != mx.int64 else v)
        assert (got == np.asarray(ref)).all(), f"{name} mismatch vs recomputed"

    model.load_weights(list(text.items()), strict=strict)

    # page-granular table lookups (MLX gather would materialize whole shards)
    from .table_reader import FileBackedNGramTable

    lid = args.ple_layer_ids[0] - 1
    fmt = f"model.language_model.layers.{lid}.ple.ple_embedding.ngram_embedding.shard_{{}}.weight"
    try:
        table = FileBackedNGramTable(model_dir, fmt, args.split_ngram_parts)
        model.language_model.layers[lid].ple.ngram_embedding.set_file_backed(table)
    except Exception as e:
        print(f"file-backed ngram table unavailable ({e}); falling back to MLX gathers")

    if not lazy:
        mx.eval(model.parameters())
    return model
