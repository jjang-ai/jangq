"""Gemma 4 (gemma4_unified) omni-modal -> MXFP4/MXFP8 conversion.

This is NOT a JANGTQ/TurboQuant converter. It emits MLX MX-quantized weights
(`weight`/`scales`[/`biases`]) with `weight_format=mxfp4|mxfp8`, quantizes the
text decoder linears, and preserves the tied token embedding/output projection
plus the thin multimodal *embedders* as fp16 passthrough.

Gemma 4 12B is an *early-fusion* omni-modal model: there is no standalone ViT
or audio conformer stack in the released checkpoint. Image patches and audio
frames are linearly embedded (`vision_embedder.*` / `embed_vision.*` /
`embed_audio.*`) straight into the shared decoder, which performs the actual
cross-modal encoding (`use_bidirectional_attention="vision"`). So "preserve
the omni-modal stack" == keep those small embedder tensors fp16.

Two Gemma-4-specific traps this converter handles (vs the Qwen MXFP path):

  1. RMSNorm convention CHANGED. Gemma 1/2/3 used `x * (1 + w)` (scale_shift=1).
     Gemma 4 uses stock `nn.RMSNorm` => `x * w` (scale_shift=0). So norm weights
     are passed through *as-is*; we do NOT add 1.0. (mlx_vlm/mlx_lm gemma4 use
     `from mlx.nn import RMSNorm` directly — see RMSNormZeroShift docstring.)

  2. attention_k_eq_v on the 8 full-attention layers (5,11,...,47): those layers
     have NO v_proj weight (V is derived from the pre-k_norm K projection at
     runtime). We simply never see a v_proj tensor for them — no special-casing
     needed beyond not assuming every layer has one.

There is NO native MTP in the 12B checkpoint (no `mtp.*` tensors).

Key remap mirrors mlx_vlm.models.gemma4 sanitize():
  strip leading `model.`, then `language_model.` -> `language_model.model.`.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mlx.core as mx
from safetensors import safe_open
from safetensors.numpy import save_file
from tqdm import tqdm

from jang_tools.calibrate import _load_bf16_tensor
from jang_tools.capabilities import build_capabilities, verify_directory
from jang_tools.convert import _remove_stale_jang_artifacts
from jang_tools.progress import ProgressEmitter
from jang_tools.ssm_layout import prepare_mlx_passthrough_tensor


MAX_SHARD = 1_000_000_000

# Gemma 4 ships an HF `processor_config.json` (NOT preprocessor_config.json)
# plus a `chat_template.jinja`. Keep both so the bundle is self-describing for
# the VL/audio processor and the chat/tool/thinking template.
SIDECAR_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "chat_template.jinja",
    "chat_template.json",
    "processor_config.json",
    "preprocessor_config.json",
    "configuration.json",
    "README.md",
    "LICENSE",
]

# Tensor-name fragments that mark the thin multimodal embedders / any
# (future) tower. Always fp16 passthrough — never quantized.
_MULTIMODAL_FRAGMENTS = (
    "vision_embedder",
    "embed_vision",
    "embed_audio",
    "vision_tower",
    "audio_tower",
)


@dataclass(frozen=True)
class QuantPolicy:
    bits: int
    method: str  # "affine" | "passthrough" | "skip"


def _decoder_layer_subpath(tensor_name: str) -> tuple[int, str] | None:
    marker = ".language_model.layers."
    if marker not in tensor_name:
        return None
    tail = tensor_name.split(marker, 1)[1]
    layer_s, dot, subpath = tail.partition(".")
    if not dot:
        return None
    try:
        return int(layer_s), subpath
    except ValueError:
        return None


def _is_decoder_attention_weight(tensor_name: str) -> bool:
    layer = _decoder_layer_subpath(tensor_name)
    return bool(
        layer is not None
        and ".self_attn." in f".{layer[1]}"
        and tensor_name.endswith(".weight")
    )


def _is_full_attention_layer(tensor_name: str) -> bool:
    layer = _decoder_layer_subpath(tensor_name)
    return bool(layer is not None and layer[0] in {5, 11, 17, 23, 29, 35, 41, 47})


def sanitize_key(hf_key: str) -> str:
    """HF gemma4_unified key -> mlx_vlm gemma4 key layout."""
    k = hf_key
    if k.startswith("model."):
        k = k[len("model.") :]
    if k.startswith("language_model.") and not k.startswith("language_model.model."):
        k = "language_model.model." + k[len("language_model.") :]
    return k


def quant_policy(
    tensor_name: str,
    bits: int = 4,
    *,
    preserve_attention_fp16: bool = False,
    preserve_full_attention_fp16: bool = False,
    preserve_first_layers: int = 0,
) -> QuantPolicy:
    name = tensor_name.lower()
    if tensor_name.endswith("_scale_inv"):
        return QuantPolicy(0, "skip")

    # Multimodal embedders / towers stay fp16 so the omni-modal path survives.
    if any(frag in name for frag in _MULTIMODAL_FRAGMENTS):
        return QuantPolicy(16, "passthrough")

    # Gemma 4 12B ties input embeddings to output logits (no lm_head tensor).
    # Keep that matrix dense; quantizing it poisons both token lookup and the
    # final vocab projection.
    if tensor_name.endswith("embed_tokens.weight"):
        return QuantPolicy(16, "passthrough")

    # Norms (input/post_attn/pre+post_ffw/q_norm/k_norm/final norm), biases,
    # the per-layer learned scalar, and positional embeddings stay fp16.
    if (
        "norm" in name
        or tensor_name.endswith(".bias")
        or tensor_name.endswith("layer_scalar")
        or tensor_name.endswith("pos_embedding")
        or tensor_name.endswith("embed_scale")
    ):
        return QuantPolicy(16, "passthrough")

    # Bare single-token names (defensive).
    if len(tensor_name.split(".")) == 1:
        return QuantPolicy(16, "passthrough")

    layer = _decoder_layer_subpath(tensor_name)
    if layer is not None and tensor_name.endswith(".weight"):
        layer_idx, _subpath = layer
        if preserve_first_layers > 0 and layer_idx < preserve_first_layers:
            return QuantPolicy(16, "passthrough")
        if preserve_attention_fp16 and _is_decoder_attention_weight(tensor_name):
            return QuantPolicy(16, "passthrough")
        if (
            preserve_full_attention_fp16
            and _is_full_attention_layer(tensor_name)
            and _is_decoder_attention_weight(tensor_name)
        ):
            return QuantPolicy(16, "passthrough")

    # Everything else 2D = decoder linears (q/k/v/o, gate/up/down) -> MX affine
    # quantized.
    return QuantPolicy(bits, "affine")


def _load_tensor(sf_path: Path, tensor_name: str, shape: list[int]) -> np.ndarray:
    with safe_open(str(sf_path), framework="numpy") as f:
        try:
            tensor = f.get_tensor(tensor_name)
            if not isinstance(tensor, np.ndarray):
                tensor = np.array(tensor)
        except Exception:
            # bf16 source: numpy can't represent it, read raw bytes.
            tensor = _load_bf16_tensor(sf_path, tensor_name, shape)
    if tensor.dtype != np.float32:
        tensor = tensor.astype(np.float32)
    return tensor


def _affine_quantize(
    tensor: np.ndarray,
    *,
    bits: int,
    group_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if tensor.ndim >= 3:
        original_shape = tensor.shape
        tensor = tensor.reshape(-1, tensor.shape[-1])
    else:
        original_shape = None

    q_weights: list[np.ndarray] = []
    q_scales: list[np.ndarray] = []
    q_biases: list[np.ndarray] = []
    mode = f"mxfp{bits}"
    chunk_rows = max(1, min(tensor.shape[0], 100_000_000 // max(1, tensor.shape[1])))
    for start in range(0, tensor.shape[0], chunk_rows):
        chunk = mx.array(tensor[start : start + chunk_rows].astype(np.float16))
        quantized = mx.quantize(chunk, group_size=group_size, bits=bits, mode=mode)
        qw, qs = quantized[:2]
        qb = quantized[2] if len(quantized) > 2 else None
        q_weights.append(np.array(qw))
        q_scales.append(np.array(qs))
        if qb is not None:
            q_biases.append(np.array(qb))
        mx.eval(qw, qs, *([] if qb is None else [qb]))
        del chunk, qw, qs, qb

    weight = np.concatenate(q_weights, axis=0)
    scales = np.concatenate(q_scales, axis=0)
    biases = np.concatenate(q_biases, axis=0) if q_biases else None
    if original_shape is not None:
        weight = weight.reshape(original_shape[0], original_shape[1], -1)
        scales = scales.reshape(original_shape[0], original_shape[1], -1)
        if biases is not None:
            biases = biases.reshape(original_shape[0], original_shape[1], -1)
    return weight, scales, biases


def _prepare_passthrough(out_name: str, tensor: np.ndarray) -> np.ndarray:
    # NOTE: Gemma 4 uses scale_shift=0 RMSNorm -> do NOT add 1.0 to norm
    # weights (unlike Gemma 1-3 / the Qwen converter convention).
    return prepare_mlx_passthrough_tensor(out_name, tensor)


def _scan_source(src: Path) -> list[tuple[str, list[int], Path]]:
    items: list[tuple[str, list[int], Path]] = []
    index_path = src / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        by_shard: dict[str, list[str]] = {}
        for key, shard in index.get("weight_map", {}).items():
            by_shard.setdefault(shard, []).append(key)
        for shard, keys in sorted(by_shard.items()):
            sf_path = src / shard
            with sf_path.open("rb") as f:
                hsize = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(hsize))
            for key in sorted(keys):
                if key.endswith("_scale_inv"):
                    continue
                items.append((key, list(header[key].get("shape", [])), sf_path))
        return items

    # Single-file (gemma-4-12B-it ships one model.safetensors) or sharded.
    single = src / "model.safetensors"
    shards = sorted(src.glob("model-*.safetensors"))
    files = [single] if single.exists() else shards
    for sf_path in files:
        with safe_open(str(sf_path), framework="numpy") as f:
            for key in f.keys():
                if key.endswith("_scale_inv"):
                    continue
                items.append((key, list(f.get_slice(key).get_shape()), sf_path))
    return items


# Gemma 4's shipped chat template lists tools but has NO `tool_choice` branch,
# so OpenAI-style `tool_choice: "required"` is not encoded in the prompt and the
# model is never told it MUST call a tool / copy args verbatim. We inject a
# conditional required-tool stanza (native Gemma format, not JSON), firing ONLY
# on tool_choice==required so normal optional-tool turns are unchanged.
_TOOL_CHOICE_ANCHOR = """        {%- endfor %}
        {%- set ns.prev_message_type = 'tool' -%}
    {%- endif -%}"""

_TOOL_CHOICE_PATCH = """        {%- endfor %}
        {%- set ns.prev_message_type = 'tool' -%}
        {%- if tool_choice is defined and ((tool_choice is string and tool_choice == 'required') or (tool_choice is mapping and tool_choice.get('type') == 'required')) -%}
            {{- '\\nTool use is REQUIRED for this turn: you must call exactly one of the declared tools. Output only the tool call, and copy every argument value verbatim \\u2014 character for character, preserving punctuation and newlines, with no added or removed whitespace.' -}}
        {%- endif -%}
    {%- endif -%}"""


_EMPTY_NO_THINK_TAIL = """{%- if add_generation_prompt -%}
    {%- if ns.prev_message_type != 'tool_response' and ns.prev_message_type != 'tool_call' -%}
        {{- '<|turn>model\\n' -}}
        {%- if not enable_thinking | default(false) -%}
            {{- '<|channel>thought\\n<channel|>' -}}
        {%- endif -%}
    {%- endif -%}
{%- endif -%}"""

_SAFE_NO_THINK_TAIL = """{%- if add_generation_prompt -%}
    {%- if ns.prev_message_type != 'tool_response' and ns.prev_message_type != 'tool_call' -%}
        {{- '<|turn>model\\n' -}}
    {%- endif -%}
{%- endif -%}"""


def _patch_chat_template_tool_choice(text: str) -> str:
    """Inject the conditional tool_choice==required stanza (idempotent)."""
    if "Tool use is REQUIRED" in text:
        return text
    if _TOOL_CHOICE_ANCHOR in text:
        return text.replace(_TOOL_CHOICE_ANCHOR, _TOOL_CHOICE_PATCH, 1)
    return text


def _patch_chat_template_no_thinking_tail(text: str) -> str:
    """Remove Gemma 4's empty no-thinking thought-channel generation prefill."""
    return text.replace(_EMPTY_NO_THINK_TAIL, _SAFE_NO_THINK_TAIL, 1)


def _patch_chat_template(text: str) -> str:
    """Apply all Gemma 4 chat-template compatibility patches (idempotent)."""
    text = _patch_chat_template_no_thinking_tail(text)
    text = _patch_chat_template_tool_choice(text)
    return text


def _copy_sidecars(src: Path, out: Path) -> None:
    for file_name in SIDECAR_FILES:
        src_file = src / file_name
        if src_file.exists():
            shutil.copy2(str(src_file), str(out / file_name))
    # Patch the jinja template (tool_choice==required + safe no-thinking
    # generation tail) and fold it into tokenizer_config so runtimes that read
    # tokenizer_config.chat_template get the same, patched template.
    tok_cfg = out / "tokenizer_config.json"
    template = out / "chat_template.jinja"
    if template.exists():
        patched = _patch_chat_template(template.read_text(encoding="utf-8"))
        template.write_text(patched, encoding="utf-8")
        if tok_cfg.exists():
            cfg = json.loads(tok_cfg.read_text(encoding="utf-8"))
            cfg["chat_template"] = patched
            gen_cfg = out / "generation_config.json"
            if gen_cfg.exists():
                gen = json.loads(gen_cfg.read_text(encoding="utf-8"))
                for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
                    if gen.get(key) is not None:
                        cfg[key] = gen[key]
            tok_cfg.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def parse_args(default_bits: int = 4) -> argparse.Namespace:
    fmt = f"MXFP{default_bits}"
    parser = argparse.ArgumentParser(
        description=f"Convert Gemma 4 (gemma4_unified) source to {fmt}."
    )
    parser.add_argument("src", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--bits", type=int, default=default_bits, choices=[4, 8])
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument(
        "--preserve-attention-fp16",
        action="store_true",
        help=(
            "Keep all decoder self-attention projections fp16 while MXFP-quantizing "
            "the remaining decoder linears. Diagnostic/audio-safe profile; not "
            "the default uniform MXFP artifact."
        ),
    )
    parser.add_argument(
        "--preserve-full-attention-fp16",
        action="store_true",
        help=(
            "Keep only full/global attention layer projections fp16. Useful for "
            "Gemma 4 12B audio/root-cause probes."
        ),
    )
    parser.add_argument(
        "--preserve-first-layers",
        type=int,
        default=0,
        help=(
            "Keep all decoder projection weights in the first N layers fp16. "
            "Useful for staged audio-quality probes."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress", choices=["json", "off"], default="off")
    parser.add_argument("--quiet-text", action="store_true")
    return parser.parse_args()


def main(default_bits: int = 4) -> None:
    args = parse_args(default_bits=default_bits)
    weight_format = f"mxfp{args.bits}"
    profile_suffixes = []
    if args.preserve_attention_fp16:
        profile_suffixes.append("ATTNFP16")
    if args.preserve_full_attention_fp16:
        profile_suffixes.append("FULLATTNFP16")
    if args.preserve_first_layers:
        profile_suffixes.append(f"L0_{args.preserve_first_layers}_FP16")
    profile = f"MXFP{args.bits}" + (("_" + "_".join(profile_suffixes)) if profile_suffixes else "")
    progress = ProgressEmitter(
        json_to_stderr=(args.progress == "json"),
        quiet_text=args.quiet_text,
    )
    src = args.src.expanduser()
    out = args.out.expanduser()

    config = json.loads((src / "config.json").read_text(encoding="utf-8"))
    text_cfg = config.get("text_config", config)
    has_vision = bool(config.get("vision_config"))
    has_audio = bool(config.get("audio_config"))
    has_video = bool(config.get("video_config"))
    n_layers = int(text_cfg.get("num_hidden_layers", 0))
    layer_types = text_cfg.get("layer_types") or []
    n_full = sum(1 for t in layer_types if t == "full_attention")

    print("=" * 70)
    print(f"  Gemma 4 (gemma4_unified) -> {profile} omni-modal")
    print("=" * 70)
    print(f"  Source:  {src}")
    print(f"  Output:  {out}")
    print(f"  Layers:  {n_layers}  (full-attn {n_full} / sliding {n_layers - n_full})")
    print(f"  Norm:    scale_shift=0 (NO +1)")
    active_modal = [name for name, enabled in (("vision", has_vision), ("audio", has_audio), ("video", has_video)) if enabled]
    print(f"  Modal:   {','.join(active_modal) if active_modal else 'text'} embedders preserved fp16 where present")
    if profile_suffixes:
        print(f"  Repair:  selective fp16 passthrough profile={profile}")
    print(f"  MTP:     none")

    progress.phase(1, 3, "scan")
    tensors = _scan_source(src)
    print(f"  Found {len(tensors)} tensors")
    if args.dry_run:
        counts: dict[str, int] = {}
        for name, _shape, _sf in tensors:
            policy = quant_policy(
                name,
                bits=args.bits,
                preserve_attention_fp16=args.preserve_attention_fp16,
                preserve_full_attention_fp16=args.preserve_full_attention_fp16,
                preserve_first_layers=args.preserve_first_layers,
            )
            key = f"{policy.method}-{policy.bits}"
            counts[key] = counts.get(key, 0) + 1
        print(json.dumps(counts, indent=2, sort_keys=True))
        progress.done(ok=True, output="dry-run")
        return

    out.mkdir(parents=True, exist_ok=True)
    removed = _remove_stale_jang_artifacts(out)
    if removed:
        print(f"  Removed {len(removed)} stale output file(s)")

    shard_idx = 0
    shard_tensors: dict[str, np.ndarray] = {}
    shard_bytes = 0
    shard_map: dict[str, str] = {}
    total_affine = 0
    total_passthrough = 0
    total_skipped = 0

    def flush_shard() -> None:
        nonlocal shard_idx, shard_tensors, shard_bytes
        if not shard_tensors:
            return
        shard_idx += 1
        name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        mx.save_safetensors(
            str(out / name),
            {
                k: (mx.array(v) if isinstance(v, np.ndarray) else v)
                for k, v in shard_tensors.items()
            },
        )
        for key in shard_tensors:
            shard_map[key] = name
        print(f"    Shard {shard_idx}: {len(shard_tensors)} tensors, {shard_bytes / 1e9:.2f} GB")
        shard_tensors = {}
        shard_bytes = 0

    def add_tensor(name: str, arr: np.ndarray) -> None:
        nonlocal shard_bytes
        shard_tensors[name] = arr
        shard_bytes += arr.nbytes
        if shard_bytes >= MAX_SHARD:
            flush_shard()

    progress.phase(2, 3, "convert")
    for tensor_name, shape, sf_path in tqdm(tensors, desc="  Processing"):
        policy = quant_policy(
            tensor_name,
            bits=args.bits,
            preserve_attention_fp16=args.preserve_attention_fp16,
            preserve_full_attention_fp16=args.preserve_full_attention_fp16,
            preserve_first_layers=args.preserve_first_layers,
        )
        if policy.method == "skip":
            total_skipped += 1
            continue

        out_name = sanitize_key(tensor_name)
        tensor = _load_tensor(sf_path, tensor_name, shape)
        if policy.method == "passthrough" or tensor.ndim < 2:
            tensor = _prepare_passthrough(out_name, tensor)
            if any(frag in out_name.lower() for frag in _MULTIMODAL_FRAGMENTS):
                add_tensor(out_name, mx.array(tensor.astype(np.float32), dtype=mx.bfloat16))
            else:
                add_tensor(out_name, tensor.astype(np.float16))
            total_passthrough += 1
        else:
            qw, qs, qb = _affine_quantize(
                tensor,
                bits=policy.bits,
                group_size=args.group_size,
            )
            base = out_name[: -len(".weight")] if out_name.endswith(".weight") else out_name
            add_tensor(f"{base}.weight", qw)
            add_tensor(f"{base}.scales", qs)
            if qb is not None:
                add_tensor(f"{base}.biases", qb)
            total_affine += 1
            del qw, qs, qb
        del tensor
        if (total_affine + total_passthrough) % 200 == 0:
            gc.collect()
            mx.clear_cache()

    flush_shard()

    progress.phase(3, 3, "write")
    for idx in range(1, shard_idx + 1):
        old = out / f"model-{idx:05d}-of-XXXXX.safetensors"
        new = out / f"model-{idx:05d}-of-{shard_idx:05d}.safetensors"
        if old.exists():
            old.rename(new)
    shard_map = {key: value.replace("XXXXX", f"{shard_idx:05d}") for key, value in shard_map.items()}
    total_size = sum((out / name).stat().st_size for name in set(shard_map.values()))
    (out / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"format": weight_format, "total_size": total_size}, "weight_map": shard_map},
            indent=2,
        ),
        encoding="utf-8",
    )

    config.pop("quantization_config", None)
    config["weight_format"] = weight_format
    config["has_vision"] = has_vision
    config["has_audio"] = has_audio
    config["has_video"] = has_video
    config["modalities"] = {
        "text": True,
        "vision": has_vision,
        "audio": has_audio,
        "video": has_video,
    }
    config["quantization"] = {
        "bits": args.bits,
        "group_size": args.group_size,
        "mode": weight_format,
        "quantization_backend": "mx.quantize",
        "norm_convention": "gemma4_scale_shift_zero",
        "tied_embedding": "fp16_passthrough",
        "multimodal": "fp16_passthrough_embedders_early_fusion",
        "mtp": "none",
    }
    if profile_suffixes:
        config["quantization"]["selective_passthrough"] = {
            "preserve_attention_fp16": bool(args.preserve_attention_fp16),
            "preserve_full_attention_fp16": bool(args.preserve_full_attention_fp16),
            "preserve_first_layers": int(args.preserve_first_layers),
            "reason": "gemma4_12b_audio_quality_probe",
        }
    caps = build_capabilities({}, config, out)
    if caps is not None:
        config["capabilities"] = caps
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    jang_config = {
        "version": 2,
        "weight_format": weight_format,
        "profile": profile,
        "source_model": {
            "name": src.name,
            "architecture": text_cfg.get("model_type", config.get("model_type", "gemma4_unified_text")),
        },
        "has_vision": has_vision,
        "has_audio": has_audio,
        "has_video": has_video,
        "modalities": {
            "text": True,
            "vision": has_vision,
            "audio": has_audio,
            "video": has_video,
        },
        "quantization": {
            "method": weight_format,
            "quantization_backend": "mx.quantize",
            "mode": weight_format,
            "group_size": args.group_size,
            "bits": args.bits,
            "norm_convention": "gemma4_scale_shift_zero",
            "tied_embedding": "fp16_passthrough",
            "multimodal": "fp16_passthrough_embedders_early_fusion",
            "mtp_policy": "none",
            "passthrough_bit_widths_used": [16],
            "passthrough_tensor_count": total_passthrough,
        },
        "runtime": {
            "total_weight_bytes": total_size,
            "total_weight_gb": round(total_size / (1024 ** 3), 2),
            "attention": "hybrid_swa_full_5to1",
            "sliding_window": text_cfg.get("sliding_window"),
            "attention_k_eq_v_on_full_layers": bool(text_cfg.get("attention_k_eq_v")),
            "full_attention_layers": [i for i, t in enumerate(layer_types) if t == "full_attention"],
        },
    }
    if profile_suffixes:
        jang_config["quantization"]["selective_passthrough"] = {
            "preserve_attention_fp16": bool(args.preserve_attention_fp16),
            "preserve_full_attention_fp16": bool(args.preserve_full_attention_fp16),
            "preserve_first_layers": int(args.preserve_first_layers),
            "reason": "gemma4_12b_audio_quality_probe",
        }
    caps = build_capabilities(jang_config, config, out)
    if caps is not None:
        jang_config["capabilities"] = caps
    (out / "jang_config.json").write_text(json.dumps(jang_config, indent=2), encoding="utf-8")
    _copy_sidecars(src, out)

    ok, msg = verify_directory(out)
    print(f"  verify: {msg}")
    if not ok:
        raise SystemExit(f"capabilities verify failed: {msg}")

    print("\n  Done!")
    print(f"  Affine tensors:      {total_affine}")
    print(f"  Passthrough tensors: {total_passthrough}")
    print(f"  Skipped tensors:     {total_skipped}")
    print(f"  Output:              {out}")
    progress.done(ok=True, output=str(out))


if __name__ == "__main__":
    main()
