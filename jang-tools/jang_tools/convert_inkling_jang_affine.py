"""Build the approximately 90-GB Inkling-Small JANG affine bundle.

Inkling stores every fused ``w13`` matrix with row-interleaved gate/up values
(``g0,u0,g1,u1,...``). Routed banks are de-interleaved into separately packed
gate/up modules so they can carry different widths; dense, shared, and MTP
``w13`` tensors retain native layout and are de-interleaved by the runtime.

Storage is spent on the more-sensitive ``up`` leg, not the down projection:

* routed gate/up/down: 2/3/2-bit affine on layers 6..41
* layers 2..5: all three routed legs are 2-bit because their learned routed
  scale is only 0.007..0.020
* routed-expert group size: 128
* attention, embeddings, head, dense MLPs, and shared experts: 8-bit affine,
  group size 64
* vision/audio towers: float16 passthrough
* all eight MTP heads: 4-bit affine, group size 64
* activation-aware scaling is folded into each MoE norm, router, routed w13,
  and shared w13, so no AWQ-specific runtime operation is required

The resulting plan is approximately 89.91 GB decimal and uses only the ordinary
MLX affine representation (uint32 weight + fp16 scales/biases).

Usage:
    python -m jang_tools.convert_inkling_jang_affine SRC OUT --plan-only
    python -m jang_tools.convert_inkling_jang_affine SRC OUT \
      --awq-stats /path/to/Inkling-Small-BF16-awq-rms.safetensors
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file

from .awq import compute_awq_scales


SHARD_BYTES = 4_000_000_000
EXPERT_GROUP_SIZE = 128
CONTROL_GROUP_SIZE = 64
CONTROL_BITS = 8
MTP_BITS = 4
MTP_GROUP_SIZE = 64
AWQ_ALPHA = 0.25
DEFAULT_HIGH_LAYERS = frozenset(range(6, 42))
LAYER_RE = re.compile(r"model\.llm\.layers\.(\d+)\.")
INKLING_EOS_TOKEN_ID = 200006
INKLING_EOS_TOKEN = "<|content_model_end_sampling|>"
INKLING_REASONING_PARSER = "inkling"
INKLING_TOOL_PARSER = "inkling"
INKLING_DEFAULT_REASONING_EFFORT = "high"
INKLING_EFFORT_MAP = {
    "none": 0.0,
    "minimal": 0.1,
    "low": 0.2,
    "medium": 0.7,
    "high": 0.9,
    "max": 0.99,
}

_VENDOR_EFFORT_MAP_LINE = (
    '{%- set effort_map = {"none": 0.0, "minimal": 0.1, "low": 0.2, '
    '"medium": 0.7, "high": 0.9, "max": 0.99} -%}'
)
_COMPAT_EFFORT_MAP_LINE = (
    '{%- set effort_map = {"none": 0.0, "minimal": 0.1, "low": 0.2, '
    '"medium": 0.7, "high": 0.9, "xhigh": 0.99, "max": 0.99} -%}'
)
_VENDOR_EFFORT_DEFAULT_LINE = (
    "    {%- set eff = reasoning_effort if reasoning_effort is defined "
    "and reasoning_effort is not none else 0.9 -%}"
)
_COMPAT_EFFORT_DEFAULT_BLOCK = """\
    {%- if reasoning_effort is defined and reasoning_effort is not none -%}
        {%- set eff = reasoning_effort -%}
    {%- elif enable_thinking is defined -%}
        {%- set eff = 0.9 if enable_thinking else 0.0 -%}
    {%- else -%}
        {%- set eff = 0.9 -%}
    {%- endif -%}"""

# These are state/control tensors, not affine Linear matrices.
PASSTHROUGH_SUBSTRINGS = (
    "_norm.weight",
    "norm.weight",
    "_sconv.weight",
    "rel_logits_proj.proj",
    "global_scale",
    ".gate.bias",
    ".gate.weight",
)
TOWER_PREFIXES = ("model.visual", "model.audio")


@dataclass
class SizePlan:
    source_tensors: int = 0
    output_tensor_groups: int = 0
    quantized_parameters: int = 0
    passthrough_parameters: int = 0
    packed_bytes: int = 0
    affine_metadata_bytes: int = 0
    passthrough_bytes: int = 0
    mtp_quantized_parameters: int = 0

    @property
    def projected_bytes(self) -> int:
        return self.packed_bytes + self.affine_metadata_bytes + self.passthrough_bytes

    def to_dict(self) -> dict:
        result = asdict(self)
        result["projected_bytes"] = self.projected_bytes
        result["projected_gb"] = self.projected_bytes / 1_000_000_000
        result["projected_gib"] = self.projected_bytes / (1024**3)
        return result


def _read_json_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _normalize_inkling_chat_template(template: str) -> str:
    """Preserve Inkling's native effort-scalar template and default.

    Inkling does not use the common boolean `enable_thinking` contract. Its
    source template accepts `reasoning_effort` and defaults an omitted value to
    0.9 (`high`). The compatibility strings below only identify and undo the
    earlier local boolean patch; fresh source templates are returned unchanged.
    Any other vendor revision fails loudly for review.
    """
    if _COMPAT_EFFORT_MAP_LINE in template:
        template = template.replace(
            _COMPAT_EFFORT_MAP_LINE, _VENDOR_EFFORT_MAP_LINE, 1
        )
    if _COMPAT_EFFORT_DEFAULT_BLOCK in template:
        template = template.replace(
            _COMPAT_EFFORT_DEFAULT_BLOCK, _VENDOR_EFFORT_DEFAULT_LINE, 1
        )

    if template.count(_VENDOR_EFFORT_MAP_LINE) != 1:
        raise ValueError("unrecognized Inkling effort_map in chat template")
    if template.count(_VENDOR_EFFORT_DEFAULT_LINE) != 1:
        raise ValueError("unrecognized Inkling reasoning-effort default block")
    if "enable_thinking" in template:
        raise ValueError(
            "Inkling source template unexpectedly references enable_thinking; "
            "preserve its native reasoning_effort contract instead"
        )
    return template


def _effort_value(value: object) -> float:
    if isinstance(value, str):
        if value not in INKLING_EFFORT_MAP:
            raise ValueError(f"unknown Inkling reasoning effort: {value!r}")
        return INKLING_EFFORT_MAP[value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if 0.0 <= number <= 0.99:
            return number
    raise ValueError(f"invalid Inkling reasoning effort: {value!r}")


def repair_inkling_bundle_metadata(bundle: str | Path) -> dict:
    """Deterministically stamp the complete Inkling serving contract.

    This touches JSON/template sidecars only. Safetensor shards and the tensor
    index are never opened for writing. It is safe to run after conversion or
    on an already-built affine bundle, and a second run is byte-idempotent.
    """
    bundle = Path(bundle).expanduser().resolve()
    required = (
        "config.json",
        "jang_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
    )
    missing = [name for name in required if not (bundle / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Inkling bundle is missing required files: {', '.join(missing)}"
        )

    config = _read_json_object(bundle / "config.json")
    jang = _read_json_object(bundle / "jang_config.json")
    tokenizer_config = _read_json_object(bundle / "tokenizer_config.json")
    special_tokens = _read_json_object(bundle / "special_tokens_map.json")

    if config.get("model_type") not in {"inkling", "inkling_mm_model"}:
        raise ValueError(
            f"not an Inkling bundle: model_type={config.get('model_type')!r}"
        )
    if config.get("eos_token_id") != INKLING_EOS_TOKEN_ID:
        raise ValueError(
            "Inkling EOS mismatch: config.json must declare "
            f"{INKLING_EOS_TOKEN_ID}, got {config.get('eos_token_id')!r}"
        )

    decoder = tokenizer_config.get("added_tokens_decoder") or {}
    eos_entry = decoder.get(str(INKLING_EOS_TOKEN_ID))
    if not isinstance(eos_entry, dict) or eos_entry.get("content") != INKLING_EOS_TOKEN:
        raise ValueError(
            "tokenizer vocabulary does not map Inkling EOS id "
            f"{INKLING_EOS_TOKEN_ID} to {INKLING_EOS_TOKEN!r}"
        )

    template_path = bundle / "chat_template.jinja"
    template = _normalize_inkling_chat_template(
        template_path.read_text(encoding="utf-8")
    )
    template_path.write_text(template, encoding="utf-8")

    tokenizer_config["eos_token"] = INKLING_EOS_TOKEN
    tokenizer_config["chat_template"] = template
    _write_json(bundle / "tokenizer_config.json", tokenizer_config)

    special_tokens["eos_token"] = INKLING_EOS_TOKEN
    _write_json(bundle / "special_tokens_map.json", special_tokens)

    generation_path = bundle / "generation_config.json"
    generation = (
        _read_json_object(generation_path) if generation_path.is_file() else {}
    )
    existing_eos = generation.get("eos_token_id")
    if existing_eos is not None:
        eos_values = existing_eos if isinstance(existing_eos, list) else [existing_eos]
        if INKLING_EOS_TOKEN_ID not in eos_values:
            raise ValueError(
                "generation_config.json EOS disagrees with config.json: "
                f"{existing_eos!r}"
            )
    else:
        generation["eos_token_id"] = INKLING_EOS_TOKEN_ID

    for key, expected in (
        ("reasoning_parser", INKLING_REASONING_PARSER),
        ("tool_call_parser", INKLING_TOOL_PARSER),
    ):
        existing = generation.get(key)
        if existing not in (None, expected):
            raise ValueError(
                f"generation_config.json {key}={existing!r}, expected {expected!r}"
            )
        generation[key] = expected

    template_kwargs = dict(generation.get("default_chat_template_kwargs") or {})
    declared_effort = template_kwargs.get("reasoning_effort")
    if declared_effort is not None and _effort_value(declared_effort) != 0.9:
        raise ValueError(
            "generation_config reasoning default disagrees with the Inkling "
            f"template default: {declared_effort!r}"
        )
    template_kwargs["reasoning_effort"] = INKLING_DEFAULT_REASONING_EFFORT
    generation["default_chat_template_kwargs"] = template_kwargs
    _write_json(generation_path, generation)

    config["reasoning_parser"] = INKLING_REASONING_PARSER
    config["tool_call_parser"] = INKLING_TOOL_PARSER
    config["default_chat_template_kwargs"] = {
        "reasoning_effort": INKLING_DEFAULT_REASONING_EFFORT
    }

    jang["chat"] = {
        "reasoning": {
            "supported": True,
            "parser": INKLING_REASONING_PARSER,
            "mechanism": "effort_scalar",
            "default_mode": INKLING_DEFAULT_REASONING_EFFORT,
            "default_effort": 0.9,
            "effort_map": INKLING_EFFORT_MAP,
            "valid_range": [0.0, 0.99],
            "thinking_channel_token": "<|content_thinking|>",
            "visible_channel_token": "<|content_text|>",
            "message_end_token": "<|end_message|>",
            "message_field": "reasoning_content",
            "generation_prompt_ends_with": "<|message_model|>",
            "control_argument": "reasoning_effort",
            "boolean_enable_thinking_supported": False,
            "omitted_argument_effort": 0.9,
        },
        "tool_calling": {
            "supported": True,
            "parser": INKLING_TOOL_PARSER,
            "invoke_json_token": "<|content_invoke_tool_json|>",
        },
        "stop": {
            "eos_token_id": INKLING_EOS_TOKEN_ID,
            "eos_token": INKLING_EOS_TOKEN,
        },
        "default_chat_template_kwargs": {
            "reasoning_effort": INKLING_DEFAULT_REASONING_EFFORT
        },
        "sampling_defaults": {},
        "sampling_defaults_source": (
            "vendor-unspecified: the source bundle has no generation_config.json "
            "and publishes no numeric sampling defaults; no temperature, top_p, "
            "top_k, repetition penalty, or token cap was invented"
        ),
    }

    # Stamp after all config/chat mutations and after the index/sidecars exist.
    from .capabilities import build_capabilities, verify_directory

    capabilities = build_capabilities(jang, config, bundle)
    if capabilities is None:
        raise ValueError("could not resolve Inkling capabilities")
    jang["capabilities"] = capabilities
    config["capabilities"] = capabilities
    _write_json(bundle / "config.json", config)
    _write_json(bundle / "jang_config.json", jang)

    ok, message = verify_directory(bundle)
    if not ok:
        raise ValueError(f"Inkling capabilities verification failed: {message}")
    return {
        "bundle": str(bundle),
        "eos_token_id": INKLING_EOS_TOKEN_ID,
        "eos_token": INKLING_EOS_TOKEN,
        "reasoning_parser": INKLING_REASONING_PARSER,
        "tool_call_parser": INKLING_TOOL_PARSER,
        "default_reasoning_effort": INKLING_DEFAULT_REASONING_EFFORT,
        "capabilities": capabilities,
        "verification": message,
    }


def _is_mtp(name: str) -> bool:
    return name.startswith("model.mtp") or ".mtp." in name


def _is_routed_expert(name: str) -> bool:
    return name.endswith(("mlp.experts.w13_weight", "mlp.experts.w2_weight"))


def _is_routed_w13(name: str) -> bool:
    return name.endswith("mlp.experts.w13_weight")


def _is_routed_w2(name: str) -> bool:
    return name.endswith("mlp.experts.w2_weight")


def _layer(name: str) -> int:
    match = LAYER_RE.search(name)
    if match is None:
        raise ValueError(f"routed expert tensor has no base-layer index: {name}")
    return int(match.group(1))


def _is_passthrough(name: str, shape: tuple[int, ...]) -> bool:
    if len(shape) == 1:
        return True
    if any(marker in name for marker in PASSTHROUGH_SUBSTRINGS):
        return True
    return name.startswith(TOWER_PREFIXES)


def _quant_spec(
    name: str,
    shape: tuple[int, ...],
    high_layers: frozenset[int],
) -> tuple[int, int] | None:
    if _is_passthrough(name, shape):
        return None
    if _is_mtp(name):
        return MTP_BITS, MTP_GROUP_SIZE
    if _is_routed_w13(name):
        raise ValueError("routed w13 is emitted as separately quantized gate/up legs")
    if _is_routed_w2(name):
        return 2, EXPERT_GROUP_SIZE
    return CONTROL_BITS, CONTROL_GROUP_SIZE


def _source_metadata(src: Path) -> tuple[dict, dict[str, tuple[tuple[int, ...], str]]]:
    index = json.loads((src / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    metadata: dict[str, tuple[tuple[int, ...], str]] = {}
    for shard_name in sorted(set(weight_map.values())):
        with safe_open(str(src / shard_name), framework="numpy") as handle:
            for name in handle.keys():
                tensor = handle.get_slice(name)
                metadata[name] = (
                    tuple(int(v) for v in tensor.get_shape()),
                    str(tensor.get_dtype()).upper(),
                )
    missing = set(weight_map) - set(metadata)
    if missing:
        raise ValueError(f"source index/header mismatch, missing e.g. {sorted(missing)[:5]}")
    return weight_map, metadata


def plan_conversion(src: str | Path, high_layers: frozenset[int]) -> SizePlan:
    src = Path(src).expanduser()
    _weight_map, metadata = _source_metadata(src)
    plan = SizePlan(source_tensors=len(metadata))
    for name, (shape, source_dtype) in metadata.items():
        elements = int(np.prod(shape, dtype=np.int64))
        if _is_routed_w13(name):
            if shape[-2] % 2:
                raise ValueError(f"{name}: interleaved row count must be even, got {shape[-2]}")
            if shape[-1] % EXPERT_GROUP_SIZE:
                raise ValueError(
                    f"{name}: input dimension {shape[-1]} is not divisible by "
                    f"group size {EXPERT_GROUP_SIZE}"
                )
            leg_elements = elements // 2
            leg_rows = leg_elements // int(shape[-1])
            leg_groups = leg_rows * (int(shape[-1]) // EXPERT_GROUP_SIZE)
            up_bits = 3 if _layer(name) in high_layers else 2
            plan.output_tensor_groups += 2
            plan.quantized_parameters += elements
            plan.packed_bytes += leg_elements * (2 + up_bits) // 8
            plan.affine_metadata_bytes += 2 * leg_groups * 4
            continue
        spec = _quant_spec(name, shape, high_layers)
        plan.output_tensor_groups += 1
        if spec is None:
            plan.passthrough_parameters += elements
            plan.passthrough_bytes += elements * (4 if source_dtype == "F32" else 2)
            continue
        bits, group_size = spec
        if shape[-1] % group_size:
            raise ValueError(
                f"{name}: input dimension {shape[-1]} is not divisible by group size {group_size}"
            )
        rows = elements // int(shape[-1])
        groups_per_row = int(shape[-1]) // group_size
        plan.quantized_parameters += elements
        if _is_mtp(name):
            plan.mtp_quantized_parameters += elements
        plan.packed_bytes += elements * bits // 8
        plan.affine_metadata_bytes += rows * groups_per_row * 4
    return plan


def _parse_layers(raw: str) -> frozenset[int]:
    values: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lo, hi = (int(v) for v in item.split("-", 1))
            values.update(range(lo, hi + 1))
        else:
            values.add(int(item))
    bad = sorted(v for v in values if v < 2 or v > 41)
    if bad:
        raise ValueError(f"high routed-layer indices must be in 2..41, got {bad}")
    return frozenset(values)


def _load_awq_scales(path: Path, hidden_size: int) -> dict[int, np.ndarray]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"AWQ activation stats not found: {path}")
    raw = load_file(str(path))
    scales: dict[int, np.ndarray] = {}
    for layer_idx in range(2, 42):
        key = f"layers.{layer_idx}.experts_input"
        if key not in raw:
            raise ValueError(f"AWQ stats are incomplete: missing {key} in {path}")
        act = np.asarray(raw[key], dtype=np.float32)
        if act.shape != (hidden_size,):
            raise ValueError(
                f"{key}: expected {(hidden_size,)}, got {act.shape}"
            )
        if not np.isfinite(act).all() or np.any(act < 0):
            raise ValueError(f"{key}: activation statistics must be finite and nonnegative")
        scale = compute_awq_scales(act, alpha=AWQ_ALPHA)
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError(f"{key}: computed invalid AWQ scales")
        scales[layer_idx] = np.ascontiguousarray(scale)
    return scales


def _apply_awq_fold_inplace(
    name: str,
    value,
    awq_scales: dict[int, np.ndarray],
) -> bool:
    """Fold one MoE-input scale into a float32 torch tensor in place."""
    match = LAYER_RE.search(name)
    if match is None:
        return False
    scale_np = awq_scales.get(int(match.group(1)))
    if scale_np is None:
        return False

    inverse = name.endswith(".mlp_norm.weight")
    router = name.endswith(".mlp.gate.weight")
    w13 = name.endswith(
        (
            ".mlp.experts.w13_weight",
            ".mlp.shared_experts.shared_w13_weight",
        )
    )
    if not (inverse or router or w13):
        return False

    torch = __import__("torch")
    scale = torch.from_numpy(scale_np)
    if inverse:
        if value.ndim != 1 or value.shape[0] != scale.shape[0]:
            raise ValueError(f"{name}: AWQ norm shape mismatch {tuple(value.shape)}")
        value.div_(scale)
    elif router:
        if value.ndim != 2 or value.shape[-1] != scale.shape[0]:
            raise ValueError(f"{name}: AWQ router shape mismatch {tuple(value.shape)}")
        value.mul_(scale.view(1, -1))
    else:
        if value.ndim != 3 or value.shape[-1] != scale.shape[0]:
            raise ValueError(f"{name}: AWQ w13 shape mismatch {tuple(value.shape)}")
        value.mul_(scale.view(1, 1, -1))
    return True


def convert(
    src: str | Path,
    out: str | Path,
    *,
    high_layers: frozenset[int] = DEFAULT_HIGH_LAYERS,
    awq_stats: str | Path,
) -> Path:
    src = Path(src).expanduser().resolve()
    out = Path(out).expanduser().resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"source model not found: {src}")
    if out == src:
        raise ValueError("output must differ from source")
    existing_artifacts = (
        list(out.glob("model-*.safetensors"))
        + list(out.glob("model.safetensors.index.json"))
        + list(out.glob("jang_config.json"))
    ) if out.exists() else []
    if existing_artifacts:
        raise FileExistsError(
            f"output already contains model artifacts; choose an empty directory: {out}"
        )
    out.mkdir(parents=True, exist_ok=True)

    size_plan = plan_conversion(src, high_layers)
    if size_plan.projected_bytes >= 95_000_000_000:
        raise ValueError(
            f"projected bundle is {size_plan.projected_bytes / 1e9:.3f} GB, "
            "which violates the 95 GB cap"
        )

    cfg = json.loads((src / "config.json").read_text())
    text_cfg = cfg.get("text_config", cfg)
    hidden_size = int(text_cfg["hidden_size"])
    moe_intermediate_size = int(text_cfg["intermediate_size"])
    awq_stats = Path(awq_stats).expanduser().resolve()
    awq_scales = _load_awq_scales(awq_stats, hidden_size)
    weight_map_in = json.loads(
        (src / "model.safetensors.index.json").read_text()
    )["weight_map"]
    names = list(weight_map_in)
    # Cheap/control tensors first, expert banks last.
    names.sort(key=lambda name: (_is_routed_expert(name), _is_mtp(name), name))

    print(f"  src: {src}")
    print(f"  out: {out}")
    print(f"  routed up_proj @ 3 bit: {sorted(high_layers)}")
    print("  every gate/down + remaining up_proj @ 2 bit")
    print(f"  control matrices @ 8 bit; MTP matrices @ {MTP_BITS} bit")
    print(f"  AWQ fold: {awq_stats} ({len(awq_scales)} MoE layers, alpha={AWQ_ALPHA})")
    print(
        f"  projected: {size_plan.projected_bytes / 1e9:.3f} GB / "
        f"{size_plan.projected_bytes / 2**30:.3f} GiB"
    )

    shard: dict[str, np.ndarray] = {}
    shard_bytes = 0
    shard_idx = 0
    weight_map_out: dict[str, str] = {}
    quantization: dict[str, dict] = {}
    counts = {
        "affine2": 0,
        "affine3": 0,
        "affine4": 0,
        "affine8": 0,
        "fp16": 0,
        "fp32": 0,
        "awq_folded": 0,
    }

    def flush() -> None:
        nonlocal shard, shard_bytes, shard_idx
        if not shard:
            return
        shard_idx += 1
        filename = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        mx.save_safetensors(
            str(out / filename),
            {key: mx.array(value) for key, value in shard.items()},
        )
        for key in shard:
            weight_map_out[key] = filename
        print(
            f"    shard {shard_idx}: {len(shard)} tensors, "
            f"{shard_bytes / 1e9:.2f} GB",
            flush=True,
        )
        shard = {}
        shard_bytes = 0

    def add(name: str, value: np.ndarray) -> None:
        nonlocal shard_bytes
        value = np.ascontiguousarray(value)
        shard[name] = value
        shard_bytes += value.nbytes
        if shard_bytes >= SHARD_BYTES:
            flush()

    def quantize_add(base: str, value_torch, bits: int, group_size: int) -> None:
        value = mx.array(value_torch.numpy())
        packed, scales, biases = mx.quantize(
            value, group_size=group_size, bits=bits
        )
        add(f"{base}.weight", np.array(packed))
        add(f"{base}.scales", np.array(scales).astype(np.float16))
        add(f"{base}.biases", np.array(biases).astype(np.float16))
        quantization[base] = {
            "mode": "affine",
            "bits": bits,
            "group_size": group_size,
        }
        counts[f"affine{bits}"] += 1
        del value, packed, scales, biases
        mx.clear_cache()

    handles: dict[str, object] = {}
    t0 = time.time()
    try:
        for index, name in enumerate(names):
            shard_name = weight_map_in[name]
            if shard_name not in handles:
                handles[shard_name] = safe_open(str(src / shard_name), framework="pt")
            tensor = handles[shard_name].get_tensor(name)
            shape = tuple(int(v) for v in tensor.shape)
            source_is_fp32 = tensor.dtype == __import__("torch").float32

            if _is_routed_w13(name):
                folded = tensor.float()
                if _apply_awq_fold_inplace(name, folded, awq_scales):
                    counts["awq_folded"] += 1
                if folded.shape[-2] != 2 * moe_intermediate_size:
                    raise ValueError(
                        f"{name}: expected {2 * moe_intermediate_size} interleaved "
                        f"rows, got {folded.shape[-2]}"
                    )
                paired = folded.reshape(
                    *folded.shape[:-2],
                    moe_intermediate_size,
                    2,
                    folded.shape[-1],
                )
                prefix = name[: -len("experts.w13_weight")]
                gate = paired[..., 0, :].contiguous()
                quantize_add(
                    prefix + "switch_mlp.gate_proj",
                    gate,
                    2,
                    EXPERT_GROUP_SIZE,
                )
                del gate
                up = paired[..., 1, :].contiguous()
                quantize_add(
                    prefix + "switch_mlp.up_proj",
                    up,
                    3 if _layer(name) in high_layers else 2,
                    EXPERT_GROUP_SIZE,
                )
                del up, paired, folded
            else:
                spec = _quant_spec(name, shape, high_layers)

            if _is_routed_w13(name):
                pass
            elif spec is None:
                torch = __import__("torch")
                # Preserve source FP32 controls exactly. Inkling's 40 router
                # correction biases and 40 learned global scales are FP32; the
                # bias participates directly in top-k selection, so casting it
                # to fp16/bf16 changes expert sets and destroys generation.
                passthrough_dtype = (
                    torch.float32 if source_is_fp32 else torch.float16
                )
                folded = tensor.float()
                if _apply_awq_fold_inplace(name, folded, awq_scales):
                    counts["awq_folded"] += 1
                value = folded.to(passthrough_dtype).numpy()
                if (
                    name.endswith("_sconv.weight")
                    and value.ndim == 3
                    and value.shape[-1] != 1
                ):
                    value = np.ascontiguousarray(value.transpose(0, 2, 1))
                add(name, value)
                counts["fp32" if passthrough_dtype == torch.float32 else "fp16"] += 1
                del folded
            else:
                bits, group_size = spec
                folded = tensor.float()
                if _apply_awq_fold_inplace(name, folded, awq_scales):
                    counts["awq_folded"] += 1
                base = name[: -len(".weight")] if name.endswith(".weight") else name
                quantize_add(base, folded, bits, group_size)
                del folded

            del tensor
            if index % 100 == 0:
                gc.collect()
                print(
                    f"  [{index}/{len(names)}] {counts} "
                    f"{time.time() - t0:.0f}s",
                    flush=True,
                )
    finally:
        handles.clear()

    flush()
    for index in range(1, shard_idx + 1):
        old = out / f"model-{index:05d}-of-XXXXX.safetensors"
        new = out / f"model-{index:05d}-of-{shard_idx:05d}.safetensors"
        old.rename(new)
    weight_map_out = {
        key: filename.replace("XXXXX", f"{shard_idx:05d}")
        for key, filename in weight_map_out.items()
    }
    total_size = sum(
        (out / filename).stat().st_size for filename in set(weight_map_out.values())
    )
    (out / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map_out},
            indent=1,
        )
    )

    out_cfg = dict(cfg)
    out_cfg.pop("quantization_config", None)
    out_cfg["weight_format"] = "affine"
    out_cfg["quantization"] = quantization
    out_cfg["routed_expert_bit_plan"] = {
        "strategy": "protect_up_by_router_global_scale",
        "gate_proj_bits": 2,
        "up_proj_3bit_layers": sorted(high_layers),
        "up_proj_remaining_bits": 2,
        "down_proj_bits": 2,
        "group_size": EXPERT_GROUP_SIZE,
        "storage_layout": "split_gate_up_down",
    }
    out_cfg["mtp_quantization"] = {
        "bits": MTP_BITS,
        "group_size": MTP_GROUP_SIZE,
        "layers": list(range(int(cfg.get("mtp_config", {}).get("num_nextn_predict_layers", 0)))),
    }
    out_cfg["awq"] = {
        "enabled": True,
        "folded": True,
        "alpha": AWQ_ALPHA,
        "layers": sorted(awq_scales),
        "activation_stat": "per-channel RMS at post_attention_layernorm output",
    }
    (out / "config.json").write_text(json.dumps(out_cfg, indent=1))

    mtp_layers = list(
        range(int(cfg.get("mtp_config", {}).get("num_nextn_predict_layers", 0)))
    )
    (out / "jang_config.json").write_text(
        json.dumps(
            {
                "format": "jang",
                "format_version": 2,
                "weight_format": "affine",
                "profile": "INKLING_JANG_1L_AWQ_90GB",
                "source_model": str(src),
                "quantization": {
                    "scheme": "asymmetric",
                    "backend": "mx.quantize",
                    "preserve_source_fp32_controls": True,
                    "control_bits": CONTROL_BITS,
                    "control_group_size": CONTROL_GROUP_SIZE,
                    "routed_expert_bit_plan": out_cfg["routed_expert_bit_plan"],
                    "mtp": out_cfg["mtp_quantization"],
                    "awq": {
                        **out_cfg["awq"],
                        "source_stats": str(awq_stats),
                    },
                },
                "native_w13_layout": "row_interleaved_gate_up",
                "bundle_has_vision": True,
                "bundle_has_audio": True,
                "bundle_has_mtp": bool(mtp_layers),
                "mtp_layers": mtp_layers,
                "size_plan": size_plan.to_dict(),
            },
            indent=1,
        )
    )

    for extra in (
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "processor_config.json",
        "generation_config.json",
        ".gitattributes",
    ):
        if (src / extra).is_file():
            shutil.copy2(src / extra, out / extra)
    if (src / "tiktoken").is_dir():
        shutil.copytree(src / "tiktoken", out / "tiktoken", dirs_exist_ok=True)

    metadata_result = repair_inkling_bundle_metadata(out)
    print(
        "  metadata: "
        f"eos={metadata_result['eos_token_id']} "
        f"reasoning={metadata_result['reasoning_parser']} "
        f"tools={metadata_result['tool_call_parser']} "
        f"default_effort={metadata_result['default_reasoning_effort']}",
        flush=True,
    )

    print(f"\n  done in {(time.time() - t0) / 60:.1f} min — {counts}")
    print(
        f"  bundle: {total_size / 1e9:.3f} GB / "
        f"{total_size / 2**30:.3f} GiB in {shard_idx} shards"
    )
    if total_size >= 95_000_000_000:
        raise RuntimeError(f"actual bundle violates 95 GB cap: {total_size}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument(
        "--high-layers",
        default=",".join(str(v) for v in sorted(DEFAULT_HIGH_LAYERS)),
        help="comma/range list of routed up_proj layers stored at 3 bits",
    )
    parser.add_argument(
        "--awq-stats",
        type=Path,
        help=(
            "per-layer post-attention RMS stats; defaults to "
            "<SRC parent>/<SRC name>-awq-rms.safetensors"
        ),
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    high_layers = _parse_layers(args.high_layers)
    if args.plan_only:
        print(json.dumps(plan_conversion(args.src, high_layers).to_dict(), indent=2))
        return 0
    awq_stats = args.awq_stats
    if awq_stats is None:
        source = args.src.expanduser().resolve()
        awq_stats = source.parent / f"{source.name}-awq-rms.safetensors"
    convert(args.src, args.out, high_layers=high_layers, awq_stats=awq_stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
