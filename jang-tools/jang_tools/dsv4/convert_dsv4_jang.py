"""DSV4-Flash → JANG_2L (standard mx.quantize path, not MXTQ codebook).

Differences vs convert_dsv4_jangtq.py:
  - Routed experts use mx.quantize(bits=2) affine (standard mlx-lm) instead
    of MXTQ codebook. Larger size vs JANGTQ2, but works with stock mlx-lm
    loader if our custom runtime isn't ready yet.
  - Attention + shared + embed + lm_head: 8-bit affine (same).
  - Norms / router / mHC: fp16 passthrough (same).

Usage:
  python -m jang_tools.dsv4.convert_dsv4_jang \\
      --src <path/to/DeepSeek-V4-Flash> \\
      --dst ~/.mlxstudio/models/JANGQ-AI/DSV4-Flash-JANG_2L \\
      --profile 2
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file as sf_save_np

from jang_tools.dsv4.chat_template import DSV4_CHAT_TEMPLATE_JINJA
from jang_tools.dsv4.weight_loader import ShardIndex


CRITICAL_F32_RE = re.compile(
    r"^(hc_head_(?:fn|base|scale)|"
    r"layers\.\d+\.hc_(?:attn|ffn)_(?:fn|base|scale)|"
    r"layers\.\d+\.attn\.attn_sink|"
    r"layers\.\d+\.ffn\.gate\.bias|"
    r"mtp\.\d+\.hc_(?:attn|ffn)_(?:fn|base|scale)|"
    r"mtp\.\d+\.hc_head_(?:fn|base|scale)|"
    r"mtp\.\d+\.attn\.attn_sink|"
    r"mtp\.\d+\.ffn\.gate\.bias)$"
)


def is_routed_expert_weight(name: str) -> bool:
    return re.search(r"ffn\.experts\.\d+\.(w1|w2|w3)\.weight$", name) is not None


def is_token_bookend_weight(name: str) -> bool:
    """Return true for input/output token projection weights."""
    return name in {"embed.weight", "head.weight"} or name.endswith(
        (
            ".embed_tokens.weight",
            ".tok_embeddings.weight",
            ".lm_head.weight",
            ".output.weight",
        )
    )


def is_attention_weight(name: str) -> bool:
    """Return true for DSV4 attention/compressor/indexer matrix weights."""
    return re.match(r"^(layers\.\d+|mtp\.\d+)\.attn\..*\.weight$", name) is not None


def parse_routed_4bit_layers(value: str | None) -> dict[int, int]:
    """Parse a comma/space-separated main-layer list into an affine bit plan."""
    if not value:
        return {}
    out: dict[int, int] = {}
    for part in re.split(r"[,\s]+", value.strip()):
        if not part:
            continue
        layer = int(part)
        if layer < 0:
            raise ValueError(f"invalid negative routed layer index: {layer}")
        out[layer] = 4
    return dict(sorted(out.items()))


def parse_routed_down_4bit_layers(value: str | None) -> dict[int, int]:
    """Parse main-layer list whose routed w2/down projections should be 4-bit."""
    return parse_routed_4bit_layers(value)


def parse_routed_projection_bits(value: str | None) -> dict[str, int]:
    """Parse a routed projection bit plan such as ``down=4`` or ``2/4/2``.

    Projection names use DSV4 source names internally:
      - ``w1`` / ``gate`` / ``gate_proj``
      - ``w2`` / ``down`` / ``down_proj``
      - ``w3`` / ``up`` / ``up_proj``
    """
    if not value:
        return {}
    aliases = {
        "w1": "w1",
        "gate": "w1",
        "gate_proj": "w1",
        "w2": "w2",
        "down": "w2",
        "down_proj": "w2",
        "w3": "w3",
        "up": "w3",
        "up_proj": "w3",
    }
    value = value.strip()
    if re.fullmatch(r"\d+\s*[/,:-]\s*\d+\s*[/,:-]\s*\d+", value):
        parts = [int(p) for p in re.split(r"\s*[/,:-]\s*", value)]
        return {proj: bits for proj, bits in zip(("w1", "w2", "w3"), parts)}

    out: dict[str, int] = {}
    for part in re.split(r"[,\s]+", value):
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"invalid routed projection bit entry {part!r}; use down=4 or 2/4/2"
            )
        raw_proj, raw_bits = part.split("=", 1)
        proj = aliases.get(raw_proj.strip().lower())
        if proj is None:
            raise ValueError(f"invalid routed projection {raw_proj!r}")
        bits = int(raw_bits)
        if bits not in (2, 3, 4):
            raise ValueError(f"invalid routed projection bits for {proj}: {bits}")
        out[proj] = bits
    return dict(sorted(out.items()))


_PROJECTION_ALIASES = {
    "w1": "w1",
    "gate": "w1",
    "gate_proj": "w1",
    "w2": "w2",
    "down": "w2",
    "down_proj": "w2",
    "w3": "w3",
    "up": "w3",
    "up_proj": "w3",
}


def _canonical_projection(raw: str) -> str:
    proj = _PROJECTION_ALIASES.get(raw.strip().lower())
    if proj is None:
        raise ValueError(f"invalid routed projection {raw!r}")
    return proj


def _merge_projection_layer_bits(
    left: dict[str, dict[int, int]],
    right: dict[str, dict[int, int]],
) -> dict[str, dict[int, int]]:
    out: dict[str, dict[int, int]] = {
        proj: dict(layer_bits) for proj, layer_bits in left.items()
    }
    for proj, layer_bits in right.items():
        out.setdefault(proj, {}).update(layer_bits)
    return {
        proj: dict(sorted(layer_bits.items()))
        for proj, layer_bits in sorted(out.items())
        if layer_bits
    }


def parse_routed_projection_layer_bits(
    value: str | None,
) -> dict[str, dict[int, int]]:
    """Parse selected main-layer projection bits.

    Accepted entries are comma/space-separated forms like:

      ``down:7=3,down:8=3,gate:12=3``

    Projection aliases match ``--routed-projection-bits``. Only main
    ``layers.N`` routed experts are affected; preserved MTP routed experts
    stay at the global projection/default bit plan.
    """
    if not value:
        return {}
    out: dict[str, dict[int, int]] = {}
    for part in re.split(r"[,\s]+", value.strip()):
        if not part:
            continue
        match = re.fullmatch(r"([A-Za-z0-9_]+):(\d+)=(\d+)", part)
        if not match:
            raise ValueError(
                f"invalid routed projection layer bit entry {part!r}; "
                "use down:7=3"
            )
        proj = _canonical_projection(match.group(1))
        layer = int(match.group(2))
        bits = int(match.group(3))
        if bits not in (2, 3, 4):
            raise ValueError(f"invalid routed projection bits for {proj}: {bits}")
        out.setdefault(proj, {})[layer] = bits
    return {
        proj: dict(sorted(layer_bits.items()))
        for proj, layer_bits in sorted(out.items())
    }


def parse_routed_projection_layer_bits_file(
    path: Path | None,
) -> dict[str, dict[int, int]]:
    """Read selected projection/layer bit plan JSON.

    Supported JSON shapes:

      ``{"w2": {"7": 3, "8": 3}, "gate": {"12": 3}}``
      ``{"routed_projection_layer_bits": {...}}``
    """
    if path is None:
        return {}
    raw = json.loads(path.read_text())
    data = raw.get("routed_projection_layer_bits", raw)
    out: dict[str, dict[int, int]] = {}
    for raw_proj, raw_layer_bits in data.items():
        proj = _canonical_projection(str(raw_proj))
        if not isinstance(raw_layer_bits, dict):
            raise ValueError(f"layer bit plan for {raw_proj!r} must be an object")
        for raw_layer, raw_bits in raw_layer_bits.items():
            layer = int(raw_layer)
            bits = int(raw_bits)
            if layer < 0:
                raise ValueError(f"invalid negative layer index: {layer}")
            if bits not in (2, 3, 4):
                raise ValueError(f"invalid routed projection bits for {proj}: {bits}")
            out.setdefault(proj, {})[layer] = bits
    return {
        proj: dict(sorted(layer_bits.items()))
        for proj, layer_bits in sorted(out.items())
    }


def parse_routed_projection_group_sizes(value: str | None) -> dict[str, int]:
    """Parse projection-specific affine group sizes such as ``gate=32,down=64``."""
    if not value:
        return {}
    aliases = {
        "w1": "w1",
        "gate": "w1",
        "gate_proj": "w1",
        "w2": "w2",
        "down": "w2",
        "down_proj": "w2",
        "w3": "w3",
        "up": "w3",
        "up_proj": "w3",
    }
    out: dict[str, int] = {}
    for part in re.split(r"[,\s]+", value.strip()):
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"invalid routed projection group-size entry {part!r}; use gate=32"
            )
        raw_proj, raw_gs = part.split("=", 1)
        proj = aliases.get(raw_proj.strip().lower())
        if proj is None:
            raise ValueError(f"invalid routed projection {raw_proj!r}")
        group_size = int(raw_gs)
        if group_size not in (32, 64, 128):
            raise ValueError(
                f"invalid routed projection group size for {proj}: {group_size}"
            )
        out[proj] = group_size
    return dict(sorted(out.items()))


def routed_bits_for_name(
    name: str,
    profile_bits: int,
    routed_layer_bits: dict[int, int] | None = None,
    routed_projection_bits: dict[str, int] | None = None,
    routed_down_layer_bits: dict[int, int] | None = None,
    routed_projection_layer_bits: dict[str, dict[int, int]] | None = None,
) -> int:
    """Return routed expert bits for source tensor name.

    The selected-layer compromise applies only to main `layers.N` routed
    experts and wins over projection defaults. Down-layer plans are narrower:
    they lift only `w2` / down projections for selected main layers. Projection
    defaults implement pure JANG_K style plans such as `w1/w2/w3 = 2/4/2`;
    they apply to both main and preserved MTP routed experts because they
    describe the projection contract, not a single main-layer exception.
    """
    m = re.match(r"^(layers\.(\d+)|mtp\.\d+)\.ffn\.experts\.\d+\.(w[123])\.weight$", name)
    if not m:
        return profile_bits
    if m.group(2) is not None and routed_layer_bits:
        layer_bits = routed_layer_bits.get(int(m.group(2)))
        if layer_bits is not None:
            return int(layer_bits)
    if m.group(2) is not None and routed_projection_layer_bits:
        layer_bits = routed_projection_layer_bits.get(m.group(3))
        if layer_bits is not None:
            bits = layer_bits.get(int(m.group(2)))
            if bits is not None:
                return int(bits)
    if m.group(2) is not None and m.group(3) == "w2" and routed_down_layer_bits:
        layer_bits = routed_down_layer_bits.get(int(m.group(2)))
        if layer_bits is not None:
            return int(layer_bits)
    if routed_projection_bits:
        return int(routed_projection_bits.get(m.group(3), profile_bits))
    return profile_bits


def routed_group_size_for_name(
    name: str,
    routed_group_size: int,
    routed_projection_group_sizes: dict[str, int] | None = None,
) -> int:
    """Return routed expert affine group size for source tensor name."""
    m = re.match(r"^(layers\.(\d+)|mtp\.\d+)\.ffn\.experts\.\d+\.(w[123])\.weight$", name)
    if not m or not routed_projection_group_sizes:
        return routed_group_size
    return int(routed_projection_group_sizes.get(m.group(3), routed_group_size))


def compatible_group_size(in_dim: int, requested: int) -> int:
    """Use the requested affine group size, falling back only when required.

    DSV4 routed expert dimensions are compatible with 128-wide groups, which
    trims the scale/bias sidecars materially versus the older 64-wide build.
    The fallback keeps odd non-routed tensors convertible without silently
    increasing group size beyond the caller's requested policy.
    """
    for gsz in (requested, 64, 32):
        if gsz <= requested and in_dim % gsz == 0:
            return gsz
    raise ValueError(f"no compatible affine group size for dim={in_dim}, requested={requested}")


def read_passthrough(idx: ShardIndex, name: str) -> np.ndarray:
    """Read tensors that should not be quantized.

    DSV4 control tensors are small but numerically load-bearing. Keep true
    source F32 for mHC/Sinkhorn/sink/router controls instead of rounding them
    through fp16 while building the affine JANG bundle.
    """
    if idx.dtype_of(name) == torch.float32 or CRITICAL_F32_RE.match(name):
        return idx.read_tensor(name, out_dtype=torch.float32).numpy().astype(np.float32)
    src_dtype = idx.dtype_of(name)
    if src_dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
        return idx.read_tensor(name).numpy()
    tensor = idx.read_tensor(name, out_dtype=torch.float16)
    if tensor.dtype == torch.bfloat16:
        return tensor.float().numpy().astype(np.float16)
    return tensor.numpy()


def classify(
    name: str,
    profile_bits: int,
    bookend_bits: int = 8,
    routed_group_size: int = 64,
    bookend_group_size: int = 64,
    attention_bits: int | None = None,
    attention_group_size: int | None = None,
    token_bookend_bits: int | None = None,
    token_bookend_group_size: int | None = None,
    routed_layer_bits: dict[int, int] | None = None,
    routed_projection_bits: dict[str, int] | None = None,
    routed_down_layer_bits: dict[int, int] | None = None,
    routed_projection_layer_bits: dict[str, dict[int, int]] | None = None,
    routed_projection_group_sizes: dict[str, int] | None = None,
) -> tuple[int, str, int]:
    """Same rules as convert_dsv4_jangtq.classify but all quantizable
    weights go through `affine` (mx.quantize). bookend_bits controls
    everything that isn't a routed expert (attn, shared expert, embed,
    lm_head, MTP matmuls, Compressor/Indexer)."""
    if ("norm" in name or name.endswith(".bias") or "attn_sink" in name
            or ".ape" in name or "tid2eid" in name or name.startswith("hc_")
            or re.search(r"^layers\.\d+\.hc_", name)
            or re.search(r"^mtp\.\d+\.hc_", name)):
        return 16, "passthrough", 0
    if name.endswith(".gate.weight") and "experts" not in name:
        return 16, "passthrough", 0
    # Routed expert → affine at profile_bits (2, 3, or 4)
    if is_routed_expert_weight(name):
        bits = routed_bits_for_name(
            name,
            profile_bits,
            routed_layer_bits,
            routed_projection_bits,
            routed_down_layer_bits,
            routed_projection_layer_bits,
        )
        group_size = routed_group_size_for_name(
            name,
            routed_group_size,
            routed_projection_group_sizes,
        )
        return bits, "affine", group_size
    if token_bookend_bits is not None and is_token_bookend_weight(name):
        return token_bookend_bits, "affine", (
            token_bookend_group_size or bookend_group_size
        )
    if attention_bits is not None and is_attention_weight(name):
        return attention_bits, "affine", attention_group_size or bookend_group_size
    # Everything else (incl. MTP matmuls) → bookend_bits affine
    if name.endswith(".weight"):
        return bookend_bits, "affine", bookend_group_size
    return 16, "passthrough", 0


def convert(src: Path, dst: Path, profile_bits: int,
            bookend_bits: int = 8,
            routed_group_size: int = 64,
            bookend_group_size: int = 64,
            attention_bits: int | None = None,
            attention_group_size: int | None = None,
            routed_layer_bits: dict[int, int] | None = None,
            routed_projection_bits: dict[str, int] | None = None,
            routed_down_layer_bits: dict[int, int] | None = None,
            routed_projection_group_sizes: dict[str, int] | None = None,
            token_bookend_bits: int | None = None,
            token_bookend_group_size: int | None = None,
            routed_projection_layer_bits: dict[str, dict[int, int]] | None = None,
            drop_mtp: bool = False) -> None:
    import mlx.core as mx

    routed_layer_bits = dict(sorted((routed_layer_bits or {}).items()))
    routed_projection_bits = dict(sorted((routed_projection_bits or {}).items()))
    routed_down_layer_bits = dict(sorted((routed_down_layer_bits or {}).items()))
    routed_projection_layer_bits = {
        proj: dict(sorted(layer_bits.items()))
        for proj, layer_bits in sorted((routed_projection_layer_bits or {}).items())
    }
    routed_projection_group_sizes = dict(
        sorted((routed_projection_group_sizes or {}).items())
    )
    dst.mkdir(parents=True, exist_ok=True)
    idx = ShardIndex(src)
    print(f"[convert] source: {src}")
    print(f"[convert] target: {dst}")
    print(f"[convert] profile: JANG_{profile_bits}L (all-affine, "
          f"routed_gs={routed_group_size}, bookend={bookend_bits}-bit/"
          f"gs={bookend_group_size})")
    if attention_bits is not None:
        print(
            "[convert] attention override: "
            f"{attention_bits}-bit/gs={attention_group_size or bookend_group_size}"
        )
    if token_bookend_bits is not None:
        print("[convert] token bookend override: "
              f"{token_bookend_bits}-bit/gs="
              f"{token_bookend_group_size or bookend_group_size}")
    if routed_layer_bits:
        print(f"[convert] routed layer bit plan: {routed_layer_bits}")
    if routed_projection_bits:
        print(f"[convert] routed projection bit plan: {routed_projection_bits}")
    if routed_down_layer_bits:
        print(f"[convert] routed down-projection layer bit plan: {routed_down_layer_bits}")
    if routed_projection_layer_bits:
        print(
            "[convert] routed projection/layer bit plan: "
            f"{routed_projection_layer_bits}"
        )
    if routed_projection_group_sizes:
        print(
            "[convert] routed projection group-size plan: "
            f"{routed_projection_group_sizes}"
        )
    if drop_mtp:
        print("[convert] MTP tensors: DROP", flush=True)
    weight_keys = [k for k in idx.keys if not k.endswith(".scale")]
    if drop_mtp:
        before = len(weight_keys)
        weight_keys = [k for k in weight_keys if not k.startswith("mtp.")]
        print(f"[convert] dropped {before - len(weight_keys)} mtp.* tensors")
    print(f"[convert] {len(weight_keys)} logical tensors")

    MAX_SHARD_BYTES = 1_000_000_000
    shard_idx = 1
    shard_bytes = 0
    shard_buf: dict[str, np.ndarray] = {}
    shard_map: dict[str, str] = {}
    totals = {"affine": 0, "passthrough": 0}
    group_totals: dict[str, int] = {}
    quant_overrides: dict[str, dict] = {}
    t_start = time.time()

    def flush_shard():
        nonlocal shard_idx, shard_bytes, shard_buf
        if not shard_buf:
            return
        shard_name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        sf_save_np(shard_buf, str(dst / shard_name))
        for k in shard_buf:
            shard_map[k] = shard_name
        print(f"    shard {shard_idx}: {len(shard_buf)} tensors, "
              f"{shard_bytes / 1e9:.2f} GB  "
              f"(elapsed {time.time() - t_start:.0f}s)", flush=True)
        shard_buf = {}
        shard_bytes = 0
        shard_idx += 1

    def add_tensor(name: str, arr: np.ndarray):
        nonlocal shard_bytes
        shard_buf[name] = arr
        shard_bytes += arr.nbytes
        if shard_bytes >= MAX_SHARD_BYTES:
            flush_shard()

    for i, name in enumerate(weight_keys):
        bits, method, requested_gs = classify(
            name,
            profile_bits,
            bookend_bits,
            routed_group_size,
            bookend_group_size,
            attention_bits,
            attention_group_size,
            token_bookend_bits,
            token_bookend_group_size,
            routed_layer_bits,
            routed_projection_bits,
            routed_down_layer_bits,
            routed_projection_layer_bits,
            routed_projection_group_sizes,
        )
        if method == "passthrough":
            arr = read_passthrough(idx, name)
            add_tensor(name, arr)
            totals["passthrough"] += 1
        else:  # affine
            t = idx.read_tensor(name, out_dtype=torch.float32)
            gsz = compatible_group_size(int(t.shape[-1]), requested_gs)
            w = mx.array(t.numpy())
            qw, qs, qb = mx.quantize(w, group_size=gsz, bits=bits)
            base = name[:-len(".weight")] if name.endswith(".weight") else name
            add_tensor(f"{base}.weight", np.array(qw))
            add_tensor(f"{base}.scales", np.array(qs).astype(np.float16))
            add_tensor(f"{base}.biases", np.array(qb).astype(np.float16))
            if bits != profile_bits or gsz != routed_group_size:
                quant_overrides[base] = {"bits": bits, "group_size": gsz, "mode": "affine"}
            group_key = f"{bits}b_g{gsz}"
            group_totals[group_key] = group_totals.get(group_key, 0) + 1
            totals["affine"] += 1
        if (i + 1) % 500 == 0:
            print(f"    processed {i + 1}/{len(weight_keys)}  "
                  f"affine={totals['affine']} passthrough={totals['passthrough']}  "
                  f"({time.time() - t_start:.0f}s)", flush=True)
    flush_shard()

    for k in range(1, shard_idx):
        old = dst / f"model-{k:05d}-of-XXXXX.safetensors"
        new = dst / f"model-{k:05d}-of-{shard_idx - 1:05d}.safetensors"
        if old.exists():
            old.rename(new)
    final_map = {k: v.replace("XXXXX", f"{shard_idx - 1:05d}") for k, v in shard_map.items()}
    total_bytes = sum((dst / fn).stat().st_size for fn in set(final_map.values()))
    (dst / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": total_bytes},
        "weight_map": final_map,
    }, indent=2))

    src_cfg = json.loads((src / "config.json").read_text())
    src_cfg.pop("quantization_config", None)
    mtp_layers = int(src_cfg.get("num_nextn_predict_layers", 0))
    # transformers 4.45+ renamed `rope_scaling` -> `rope_parameters` and
    # `type` -> `rope_type` inside the dict. Add the newer spelling for
    # compatibility, but never remove rope_scaling: jang_tools.dsv4.mlx_model
    # consumes it directly for DSV4 Flash compressed-context YaRN.
    if "rope_scaling" in src_cfg and "rope_parameters" not in src_cfg:
        rs = dict(src_cfg["rope_scaling"])
        rp = dict(rs)
        if "type" in rp:
            rp["rope_type"] = rp.pop("type")
        if "rope_theta" not in rp:
            rp["rope_theta"] = float(src_cfg.get("rope_theta", 10000))
        for k in ("beta_fast", "beta_slow", "factor"):
            if k in rp:
                rp[k] = float(rp[k])
        src_cfg["rope_parameters"] = rp
    # Keep MTP tensors and source config fields in the bundle, but do not
    # claim runtime self-spec activation here. The normal autoregressive
    # runtime must ignore MTP unless an explicit accept/reject speculative
    # decode path is implemented and selected.
    quant_cfg: dict[str, object] = {
        "bits": profile_bits,
        "group_size": routed_group_size,
        "mode": "affine",
        "routed_expert_bits": profile_bits,
        "routed_expert_group_size": routed_group_size,
        "bookend_bits": bookend_bits,
        "bookend_group_size": bookend_group_size,
    }
    if token_bookend_bits is not None:
        quant_cfg["token_bookend_bits"] = token_bookend_bits
        quant_cfg["token_bookend_group_size"] = (
            token_bookend_group_size or bookend_group_size
        )
    if attention_bits is not None:
        quant_cfg["attention_bits"] = attention_bits
        quant_cfg["attention_group_size"] = attention_group_size or bookend_group_size
    routed_expert_bit_plan = None
    if (
        routed_layer_bits
        or routed_projection_bits
        or routed_down_layer_bits
        or routed_projection_layer_bits
        or routed_projection_group_sizes
    ):
        routed_expert_bit_plan = {
            "default_bits": profile_bits,
            "codec": "affine",
            "group_size": routed_group_size,
            "routed_layer_bits": {str(k): int(v) for k, v in routed_layer_bits.items()},
            "routed_projection_bits": {
                str(k): int(v) for k, v in routed_projection_bits.items()
            },
            "mtp_routed_bits": profile_bits,
            "mtp_routed_projection_bits": {
                str(k): int(v) for k, v in routed_projection_bits.items()
            },
            "routed_down_layer_bits": {
                str(k): int(v) for k, v in routed_down_layer_bits.items()
            },
            "routed_projection_layer_bits": {
                str(proj): {str(k): int(v) for k, v in layer_bits.items()}
                for proj, layer_bits in routed_projection_layer_bits.items()
            },
            "routed_projection_group_sizes": {
                str(k): int(v) for k, v in routed_projection_group_sizes.items()
            },
        }
        quant_cfg["routed_expert_bit_plan"] = routed_expert_bit_plan
    quant_cfg.update(quant_overrides)
    src_cfg["quantization"] = quant_cfg
    src_cfg["weight_format"] = "affine"
    src_cfg["routed_expert_bits"] = profile_bits
    src_cfg["routed_expert_group_size"] = routed_group_size
    if routed_expert_bit_plan:
        src_cfg["routed_expert_bit_plan"] = routed_expert_bit_plan
    src_cfg["group_size"] = routed_group_size
    profile_suffix = f"JANG_{profile_bits}L_GS{routed_group_size}"
    if routed_layer_bits:
        layers_tag = "-".join(str(k) for k in routed_layer_bits)
        profile_suffix += f"_L{layers_tag}x4"
    if routed_projection_bits:
        proj_tags = {"w1": "G", "w2": "D", "w3": "U"}
        proj_tag = "-".join(
            f"{proj_tags.get(k, k)}{v}" for k, v in routed_projection_bits.items()
        )
        profile_suffix += f"_P{proj_tag}"
    if routed_down_layer_bits:
        layers_tag = "-".join(str(k) for k in routed_down_layer_bits)
        profile_suffix += f"_D{layers_tag}x4"
    if routed_projection_layer_bits:
        profile_suffix += "_ProjLayerBits"
    if routed_projection_group_sizes:
        proj_tags = {"w1": "G", "w2": "D", "w3": "U"}
        gs_tag = "-".join(
            f"{proj_tags.get(k, k)}gs{v}"
            for k, v in routed_projection_group_sizes.items()
        )
        profile_suffix += f"_{gs_tag}"
    if bookend_group_size != routed_group_size:
        profile_suffix += f"_BKGS{bookend_group_size}"
    if bookend_bits != 8:
        profile_suffix += f"_bk{bookend_bits}"
    if attention_bits is not None:
        profile_suffix += (
            f"_Attn{attention_bits}g"
            f"{attention_group_size or bookend_group_size}"
        )
    if token_bookend_bits is not None:
        profile_suffix += (
            f"_Tok{token_bookend_bits}g"
            f"{token_bookend_group_size or bookend_group_size}"
        )
    if drop_mtp:
        profile_suffix += "_NoMTP"
        src_cfg["num_nextn_predict_layers"] = 0
        src_cfg["mtp_num_hidden_layers"] = None
        src_cfg["use_mtp"] = False
        runtime_cfg = src_cfg.setdefault("runtime", {})
        runtime_cfg.update({
            "bundle_has_mtp": False,
            "mtp_layers": 0,
            "mtp_mode": "dropped",
        })
    if mtp_layers > 0 and not drop_mtp:
        profile_suffix += "_MTP"
        src_cfg.setdefault("runtime", {})
        src_cfg["runtime"].update({
            "bundle_has_mtp": True,
            "mtp_layers": mtp_layers,
            "mtp_mode": "preserved_disabled",
        })
    src_cfg["_name_or_path"] = f"DSV4-Flash-{profile_suffix}"
    (dst / "config.json").write_text(json.dumps(src_cfg, indent=2))

    (dst / "jang_config.json").write_text(json.dumps({
        "weight_format": "affine",
        "profile": profile_suffix,
        "source_model": str(src),
        "critical_f32_preserved": True,
        "dsv4_runtime_requirements": {
            "limited_swiglu_tq_patch": False,
            "generic_mlx_sinks": False,
            "native_cache_schema": "deepseek_v4_v7",
            "generic_turboquant_kv": False,
            "long_ctx_default": True,
            "pool_quant_default": False,
            "max_num_seqs": 1,
        },
        "affine_bits": {
            "routed_expert": profile_bits,
            "attention": attention_bits or bookend_bits,
            "shared_expert": bookend_bits,
            "embed_tokens": token_bookend_bits or bookend_bits,
            "lm_head": token_bookend_bits or bookend_bits,
            "mtp_matmul": bookend_bits,
            "norms_router_hc": 16,
        },
        "affine_group_size": {
            "routed_expert": routed_group_size,
            "attention": attention_group_size or bookend_group_size,
            "shared_expert": bookend_group_size,
            "embed_tokens": token_bookend_group_size or bookend_group_size,
            "lm_head": token_bookend_group_size or bookend_group_size,
            "mtp_matmul": bookend_group_size,
        },
        "quantization": {
            "method": "affine",
            "top_level_default": {
                "bits": profile_bits,
                "group_size": routed_group_size,
                "mode": "affine",
            },
            "routed_experts": {
                "bits": profile_bits,
                "codec": "affine",
                "group_size": routed_group_size,
                "bit_plan": routed_expert_bit_plan,
            },
            "non_routed": {
                "bits": bookend_bits,
                "codec": "affine",
                "group_size": bookend_group_size,
            },
            "token_bookends": {
                "bits": token_bookend_bits or bookend_bits,
                "codec": "affine",
                "group_size": token_bookend_group_size or bookend_group_size,
                "override": token_bookend_bits is not None,
            },
            "attention": {
                "bits": attention_bits or bookend_bits,
                "codec": "affine",
                "group_size": attention_group_size or bookend_group_size,
                "override": attention_bits is not None,
            },
            "critical_control_tensors": "source-f32",
            "override_count": len(quant_overrides),
            "group_totals": group_totals,
        },
        "routed_layer_bits": (
            {str(k): int(v) for k, v in routed_layer_bits.items()}
            if routed_layer_bits else {}
        ),
        "routed_projection_bits": (
            {str(k): int(v) for k, v in routed_projection_bits.items()}
            if routed_projection_bits else {}
        ),
        "routed_down_layer_bits": (
            {str(k): int(v) for k, v in routed_down_layer_bits.items()}
            if routed_down_layer_bits else {}
        ),
        "routed_projection_layer_bits": (
            {
                str(proj): {str(k): int(v) for k, v in layer_bits.items()}
                for proj, layer_bits in routed_projection_layer_bits.items()
            }
            if routed_projection_layer_bits else {}
        ),
        "routed_projection_group_sizes": (
            {str(k): int(v) for k, v in routed_projection_group_sizes.items()}
            if routed_projection_group_sizes else {}
        ),
        "cache": {
            "schema": "deepseek_v4_v7",
            "components": ["swa", "csa", "hsa"],
            "sliding_window": src_cfg.get("sliding_window"),
            "compress_ratios": src_cfg.get("compress_ratios"),
            "generic_turboquant_kv": False,
            "pool_quant_default": False,
            "mtp_activation_requires_draft_cache": True,
        },
        "mtp": {
            "preserved": mtp_layers > 0 and not drop_mtp,
            "runtime_self_spec_enabled": False,
            "mode": "dropped" if drop_mtp else (
                "preserved_disabled" if mtp_layers > 0 else "absent"
            ),
            "num_nextn_predict_layers": 0 if drop_mtp else mtp_layers,
            "activation_requires": (
                "separate MTP drafter, draft cache, accept/reject verifier, "
                "and DSV4 SWA+CSA/HSA composite-cache-safe rollback"
            ),
        },
        "source_config": {
            "n_routed_experts": src_cfg.get("n_routed_experts"),
            "num_experts_per_tok": src_cfg.get("num_experts_per_tok"),
            "num_hidden_layers": src_cfg.get("num_hidden_layers"),
            "num_nextn_predict_layers": mtp_layers,
            "sliding_window": src_cfg.get("sliding_window"),
            "compress_ratios": src_cfg.get("compress_ratios"),
            "hc_mult": src_cfg.get("hc_mult"),
            "hc_sinkhorn_iters": src_cfg.get("hc_sinkhorn_iters"),
            "swiglu_limit": src_cfg.get("swiglu_limit"),
            "routed_scaling_factor": src_cfg.get("routed_scaling_factor"),
        },
        "model_family": "deepseek_v4",
        "chat": {
            "encoder": "encoding_dsv4",
            "encoder_fn": "encode_messages",
            "chat_template_source": "tokenizer_config",
            "has_tokenizer_chat_template": True,
            "bos_token": "<｜begin▁of▁sentence｜>",
            "eos_token": "<｜end▁of▁sentence｜>",
            "bos_token_id": 0,
            "eos_token_id": 1,
            "role_tokens": {
                "user": "<｜User｜>",
                "assistant": "<｜Assistant｜>",
                "latest_reminder": "<｜latest_reminder｜>",
            },
            "reasoning": {
                "supported": True,
                "modes": ["chat", "thinking"],
                "default_mode": "chat",
                "thinking_start": "<think>",
                "thinking_end": "</think>",
                "reasoning_effort_levels": ["max", "high", None],
                "drop_earlier_reasoning": True,
            },
            "tool_calling": {
                "supported": True,
                "parser": "dsml",
                "dsml_token": "｜DSML｜",
                "tool_calls_block": "tool_calls",
                "invoke_block": "invoke",
                "parameter_block": "parameter",
                "tool_output_tag": "tool_result",
            },
            "sampling_defaults": {
                "temperature": 0.6,
                "top_p": 0.95,
                "repetition_penalty": 1.0,
                "repetition_penalty_thinking": 1.0,
                "repetition_penalty_chat": 1.05,
                "max_new_tokens": 4096,
            },
        },
    }, indent=2, ensure_ascii=False))

    copied = 0
    for p in src.iterdir():
        if p.is_file() and not p.name.endswith(".safetensors") \
                and p.name not in ("config.json", "model.safetensors.index.json"):
            shutil.copy2(p, dst / p.name)
            copied += 1
    enc = src / "encoding"
    if enc.is_dir():
        shutil.copytree(enc, dst / "encoding", dirs_exist_ok=True)
        copied += 1
    print(f"[convert] copied {copied} aux files/dirs")

    tok_cfg = dst / "tokenizer_config.json"
    if tok_cfg.exists():
        data = json.loads(tok_cfg.read_text())
        data["chat_template"] = DSV4_CHAT_TEMPLATE_JINJA
        tok_cfg.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print("[convert] injected DSV4 chat_template into tokenizer_config.json")

    # Force eos_token_id to a list including <|User|> when the tokenizer can
    # resolve it, preventing generated assistant turns from rolling into a fake
    # user turn after <|end_of_sentence|>.
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(dst), trust_remote_code=True)
        eos_id = tok.convert_tokens_to_ids("<｜end▁of▁sentence｜>")
        user_id = tok.convert_tokens_to_ids("<｜User｜>")
        if eos_id is not None and user_id is not None and eos_id != user_id:
            eos_list = sorted({int(eos_id), int(user_id)})
            for fn in ("config.json", "generation_config.json", "tokenizer_config.json"):
                p = dst / fn
                if not p.exists():
                    continue
                data = json.loads(p.read_text())
                data["eos_token_id"] = eos_list
                p.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            print(f"[convert] forced eos_token_id={eos_list} in config files")
    except Exception as eos_err:
        print(f"[convert] WARN eos list patch failed: {eos_err}")

    elapsed = time.time() - t_start
    print(f"\nDONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  affine={totals['affine']}  passthrough={totals['passthrough']}")
    print(f"  group_totals={group_totals}")
    print(f"  quant_overrides={len(quant_overrides)}")
    print(f"  output size: {total_bytes / 1e9:.1f} GB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--profile", type=int, default=2, choices=(2, 3, 4),
                    help="routed-expert bits")
    ap.add_argument("--bookend-bits", type=int, default=8, choices=(4, 6, 8),
                    help="non-routed bits (attn / shared / embed / lm_head / "
                         "mtp matmuls). Default 8. Use 4 for the smallest "
                         "coherent bundle (~10-15 GB savings vs 8).")
    ap.add_argument("--routed-group-size", type=int, default=64, choices=(32, 64, 128),
                    help="affine group size for routed experts. Use 128 for "
                         "the compact M5 affine DSV4 build.")
    ap.add_argument("--bookend-group-size", type=int, default=64, choices=(32, 64, 128),
                    help="affine group size for non-routed quantized tensors.")
    ap.add_argument("--attention-bits", type=int, choices=(4, 6, 8),
                    help="optional bits just for attention/compressor/indexer "
                         "matrix weights. Leave unset to use --bookend-bits.")
    ap.add_argument("--attention-group-size", type=int, choices=(32, 64, 128),
                    help="optional group size just for attention/compressor/"
                         "indexer matrix weights. Defaults to "
                         "--bookend-group-size.")
    ap.add_argument("--token-bookend-bits", type=int, choices=(4, 6, 8),
                    help="optional bits just for token input/output projections "
                         "(embed/head or embed_tokens/lm_head). Leave unset to "
                         "use --bookend-bits for these tensors.")
    ap.add_argument("--token-bookend-group-size", type=int, choices=(32, 64, 128),
                    help="optional group size just for token input/output "
                         "projections. Defaults to --bookend-group-size.")
    ap.add_argument("--routed-4bit-layers", default="",
                    help="comma/space-separated main layer indexes whose routed "
                         "experts should use 4-bit affine while the rest stay "
                         "at --profile bits. MTP routed experts stay at the "
                         "default profile bits.")
    ap.add_argument("--routed-projection-bits", default="",
                    help="projection-specific routed bits such as down=4 or "
                         "2/4/2 for w1/w2/w3. Selected --routed-4bit-layers "
                         "override this for main layers.")
    ap.add_argument("--routed-down-4bit-layers", default="",
                    help="comma/space-separated main layer indexes whose routed "
                         "w2/down projections should use 4-bit affine while "
                         "w1/gate and w3/up stay at --profile bits. This does "
                         "not affect MTP routed experts.")
    ap.add_argument("--routed-projection-layer-bits", default="",
                    help="comma/space-separated selected projection/layer bits "
                         "such as down:7=3,gate:8=3. Applies only to main "
                         "layers.N routed experts; preserved MTP routed experts "
                         "stay at the projection/default bit plan.")
    ap.add_argument("--routed-projection-layer-bits-file", type=Path,
                    help="JSON file for selected projection/layer bits, either "
                         "{'down': {'7': 3}} or "
                         "{'routed_projection_layer_bits': {...}}.")
    ap.add_argument("--routed-projection-group-sizes", default="",
                    help="projection-specific routed expert affine group sizes "
                         "such as gate=32,up=64,down=64. This is useful for "
                         "DQ-style all-affine size experiments.")
    ap.add_argument("--drop-mtp", action="store_true",
                    help="drop mtp.* tensors and mark the bundle as no-MTP. "
                         "Keep unset for preserved-disabled MTP artifacts.")
    args = ap.parse_args()
    routed_projection_layer_bits = _merge_projection_layer_bits(
        parse_routed_projection_layer_bits(args.routed_projection_layer_bits),
        parse_routed_projection_layer_bits_file(args.routed_projection_layer_bits_file),
    )
    convert(
        src=args.src,
        dst=args.dst,
        profile_bits=args.profile,
        bookend_bits=args.bookend_bits,
        routed_group_size=args.routed_group_size,
        bookend_group_size=args.bookend_group_size,
        attention_bits=args.attention_bits,
        attention_group_size=args.attention_group_size,
        routed_layer_bits=parse_routed_4bit_layers(args.routed_4bit_layers),
        routed_projection_bits=parse_routed_projection_bits(
            args.routed_projection_bits
        ),
        routed_down_layer_bits=parse_routed_down_4bit_layers(
            args.routed_down_4bit_layers
        ),
        routed_projection_group_sizes=parse_routed_projection_group_sizes(
            args.routed_projection_group_sizes
        ),
        token_bookend_bits=args.token_bookend_bits,
        token_bookend_group_size=args.token_bookend_group_size,
        routed_projection_layer_bits=routed_projection_layer_bits,
        drop_mtp=args.drop_mtp,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
