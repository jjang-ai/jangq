"""DeepSeek-V4-Flash FP4+FP8 source → JANGTQ bundle.

No REAP prune — converts all 256 experts as-is. Profile:
  JANGTQ2 (default): routed experts 2-bit MXTQ, attention + embed +
                     lm_head + shared experts 8-bit affine, norms +
                     router + mHC params fp16 passthrough
  JANGTQ4: routed experts 4-bit MXTQ (bigger bundle, better fidelity)

Usage:
  python -m jang_tools.dsv4.convert_dsv4_jangtq \\
      --src <path/to/DeepSeek-V4-Flash> \\
      --dst ~/.mlxstudio/models/JANGQ-AI/DSV4-Flash-JANGTQ \\
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

from jang_tools.dsv4.weight_loader import ShardIndex


SEED = 42
FORMAT = "jangtq"  # set by main() from --format flag: "jang" or "jangtq"

# DSV4 canonical chat template (Jinja port of encoding_dsv4.encode_messages).
# Bundles ship this baked into tokenizer_config.json so Swift consumers don't
# need the legacy-bundle injection path. Mirrors the template embedded in
# vMLX swift's `CapabilityDetector.deepseekV4BuiltinTemplate`. Single source
# of truth for both: keep them in lockstep when editing.
DSV4_CHAT_TEMPLATE_JINJA = (
    "{%- set thinking_on = (enable_thinking is defined and enable_thinking) -%}"
    "{%- set effort = (reasoning_effort if reasoning_effort is defined else None) -%}"
    "{%- set ns = namespace(last_user_idx=-1) -%}"
    "{%- for m in messages -%}"
    "{%- if m.role == 'user' or m.role == 'developer' -%}"
    "{%- set ns.last_user_idx = loop.index0 -%}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "<｜begin▁of▁sentence｜>"
    "{%- if thinking_on and effort == 'max' -%}"
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n"
    "{% endif -%}"
    "{%- for m in messages -%}"
    "{%- set idx = loop.index0 -%}"
    "{%- if m.role == 'system' -%}"
    "{{- m.content -}}"
    "{%- elif m.role == 'user' or m.role == 'developer' -%}"
    "<｜User｜>{{- m.content -}}"
    "{%- elif m.role == 'assistant' -%}"
    "<｜Assistant｜>"
    "{%- set rc = m.reasoning_content if m.reasoning_content is defined else '' -%}"
    "{%- if thinking_on and rc and idx > ns.last_user_idx -%}"
    "<think>{{- rc -}}</think>"
    "{%- endif -%}"
    "{{- m.content -}}"
    "<｜end▁of▁sentence｜>"
    "{%- elif m.role == 'tool' -%}"
    "<tool_result>{{- m.content -}}</tool_result>"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "<｜Assistant｜>"
    "{%- if thinking_on -%}<think>{%- else -%}</think>{%- endif -%}"
    "{%- endif -%}"
)


def classify(name: str, profile_bits: int) -> tuple[int, str, int]:
    """Map tensor name → (bits, method, group_size).

    **group_size=128 default (2026-04-24)**: cross-model evidence from
    CRACK_abliteration/docs/minimax_m25/MINIMAX_SPEED_BUG.md shows
    `gather_qmm` with group_size=64 had +15-25% cache pressure; switching
    to 128 eliminated it. Override via DSV4_GROUP_SIZE env var.

    - FP4-origin routed experts:
      * profile 8 → 8-bit affine g=128 (max fidelity)
      * profile 4 + format=jangtq → 4-bit MXTQ codebook (FIX 2026-04-24)
      * profile 4 + format=jang   → 4-bit affine g=128 (matches FP4 source)
      * profile 2 + format=jangtq → 2-bit MXTQ codebook (aggressive; lossy)
      * profile 2 + format=jang   → 2-bit affine g=128
    - FP8-origin attention/shared/embed/head: 8-bit affine g=128
    - Norms/small: fp16 passthrough
    """
    import os as _os
    n = name
    # gsz=32 default. Tried gsz=128 (CRACK_abliteration MiniMax precedent
    # claimed +15-25%) but it BROKE DSV4 output (gibberish on JANGTQ4 bench
    # 2026-04-24). Working JANGTQ uses gsz=32, that's the verified-coherent
    # value for DSV4. Keep override env for future experiments.
    gsz_default = int(_os.environ.get("DSV4_GROUP_SIZE", "32"))

    # Norms + small tensors → fp16 passthrough
    if ("norm" in n or n.endswith(".bias") or "attn_sink" in n
            or ".ape" in n or "tid2eid" in n or n.startswith("hc_")
            or re.search(r"^layers\.\d+\.hc_", n)
            or re.search(r"^mtp\.\d+\.hc_", n)
            or re.search(r"^layers\.\d+\.ffn\.gate\.(weight|bias)$", n)
            ):
        return 16, "passthrough", 0

    # Router gate.weight → fp16 passthrough
    if n.endswith(".gate.weight") and "experts" not in n:
        return 16, "passthrough", 0

    # Routed experts
    if re.search(r"ffn\.experts\.\d+\.(w1|w2|w3)\.weight$", n):
        # JANG format: always use standard affine (all bit-widths).
        # JANGTQ format: MXTQ codebook for 2-bit AND 4-bit, affine for higher.
        # FIX 2026-04-24: previous code routed profile=4 to affine even for
        # jangtq → produced JANGTQ4 bundles WITHOUT .tq_packed tensors, broke
        # Swift sidecar build. Now profile_bits in (2, 4) → mxtq codebook
        # when FORMAT=jangtq. profile_bits in (3, 5, 6, 8) → affine since
        # TurboQuant codebook only supports 2- and 4-bit cleanly.
        eff_bits = profile_bits
        # First-N MoE 4-bit override (#39, B.6): hash-routed layers (default
        # `num_hash_layers=3`) are deterministic per-token, no smoothing
        # across multiple expert outputs → quantization noise compounds
        # harder. Bump them to higher bits via env DSV4_HASH_BITS=4 (default
        # off; opt-in until empirical pass@1 delta is measured).
        hash_bits = int(_os.environ.get("DSV4_HASH_BITS", "0"))
        n_hash_layers = int(_os.environ.get("DSV4_NUM_HASH_LAYERS", "3"))
        if hash_bits:
            layer_m = re.match(r"^layers\.(\d+)\.", n)
            if layer_m and int(layer_m.group(1)) < n_hash_layers:
                eff_bits = max(profile_bits, hash_bits)
        if FORMAT == "jang":
            return eff_bits, "affine", gsz_default
        if eff_bits == 2:
            return eff_bits, "mxtq", 0
        return eff_bits, "affine", gsz_default

    # Shared experts + attention + embed + head: default 8-bit affine g=32.
    # Env DSV4_HIGH_PRECISION=1 keeps these at bf16 (no quant) to eliminate
    # compound quant error for arithmetic-sensitive reasoning. Trade-off:
    # bundle +5-8 GB, but removes all quant noise from non-routed path.
    # Env DSV4_LOW_BITS=1 cuts attention/shared to 4-bit affine — speeds
    # decode by ~20-30% (bandwidth reduction on the heaviest matmul wq_b
    # 1024→32768 which is 45% of per-layer time per profiler 2026-04-24).
    # Quality risk: attention quant noise could compound but DSV4 was
    # trained with FP8 activations so 4-bit affine is well within tolerance.
    hp = _os.environ.get("DSV4_HIGH_PRECISION", "0") == "1"
    lb = _os.environ.get("DSV4_LOW_BITS", "0") == "1"
    nonroute_bits = 16 if hp else (4 if lb else 8)
    nonroute_mode = "passthrough" if hp else "affine"
    nonroute_gsz = 0 if hp else gsz_default
    if "shared_experts" in n and n.endswith(".weight"):
        return nonroute_bits, nonroute_mode, nonroute_gsz
    if re.search(r"attn\.(wq_a|wq_b|wkv|wo_a|wo_b)\.weight$", n):
        return nonroute_bits, nonroute_mode, nonroute_gsz
    if n == "embed.weight" or n == "head.weight":
        return nonroute_bits, nonroute_mode, nonroute_gsz

    if n.endswith(".weight"):
        return nonroute_bits, nonroute_mode, nonroute_gsz

    return 16, "passthrough", 0


def convert(
    src: Path, dst: Path, profile_bits: int,
    awq_norms_path: Path | None = None, awq_alpha: float = 0.25,
) -> None:
    import mlx.core as mx
    from jang_tools.turboquant.linear import tq_quantize_weight

    dst.mkdir(parents=True, exist_ok=True)
    idx = ShardIndex(src)
    print(f"[convert] source: {src}")
    print(f"[convert] target: {dst}")
    profile_name = f"JANG_{profile_bits}L" if FORMAT == "jang" else f"JANGTQ{profile_bits}"
    print(f"[convert] profile: {profile_name} (format={FORMAT})")

    # Optional AWQ pre-scaling: caller may pass a path to an
    # awq_activations.safetensors (output of `python3 -m
    # jang_tools.awq_capture <bundle>`). For each routed-expert mxtq
    # weight we look up `<module_name>` (e.g.
    # `model.layers.5.mlp.switch_mlp.gate_proj`) → per-channel max(|x|),
    # compute scale `s = (norms + eps)^alpha`, multiply weight columns
    # by `s` BEFORE quantization, and save the scale to
    # `awq_scales.safetensors` so the runtime loader (`load_jangtq.py`)
    # can divide activations by the same `s` at inference.
    awq_norms: dict[str, np.ndarray] = {}
    awq_scales_out: dict[str, np.ndarray] = {}
    awq_layernorm_fold: dict[int, np.ndarray] = {}  # layer_idx -> scale
    awq_eps = 1e-8
    if awq_norms_path is not None:
        from safetensors.numpy import load_file as _safe_load
        awq_norms = _safe_load(str(awq_norms_path))
        print(f"[convert] AWQ pre-scaling: {len(awq_norms)} module activation "
              f"norms loaded, alpha={awq_alpha}")
        # Zero-runtime AWQ (#44): switch_mlp.gate_proj, switch_mlp.up_proj,
        # shared_experts.{gate,up}_proj all consume the SAME input
        # (post_attention_layernorm output → MoE), so they share one AWQ
        # scale per layer. Fold that scale's INVERSE into
        # post_attention_layernorm.weight at convert time:
        #   y = (RMSNorm(x) * w_norm / s) @ Q(W * s)  ≈  RMSNorm(x) * w_norm @ W
        # The runtime never divides — just multiplies a smaller w_norm.
        # Saves 4 elementwise-divide dispatches per MoE layer × 41 layers
        # = 164 dispatches/token.
        # down_proj inputs are swiglu(gate*up), a different distribution,
        # so down_proj scales stay in `awq_scales_out` for runtime apply.
        n_layers = 0  # set below from src config
        try:
            _src_cfg_for_layers = json.loads((src / "config.json").read_text())
            n_layers = int(_src_cfg_for_layers.get("num_hidden_layers", 43))
        except Exception:
            n_layers = 43
        for layer_idx in range(n_layers):
            gate_key = f"model.layers.{layer_idx}.mlp.switch_mlp.gate_proj"
            norms = awq_norms.get(gate_key)
            if norms is not None:
                scale = (norms.astype(np.float32) + awq_eps) ** awq_alpha
                awq_layernorm_fold[layer_idx] = scale
        if awq_layernorm_fold:
            print(f"[convert] AWQ fold: {len(awq_layernorm_fold)} layer norms "
                  f"will be pre-divided by gate/up scales")

    print(f"[convert] scanning for .weight keys (skip sibling .scale)...")
    weight_keys = [k for k in idx.keys if not k.endswith(".scale")]
    print(f"[convert] {len(weight_keys)} logical tensors to process")

    MAX_SHARD_BYTES = 1_000_000_000
    shard_idx = 1
    shard_bytes = 0
    shard_buf: dict[str, np.ndarray] = {}
    shard_map: dict[str, str] = {}

    totals = {"mxtq": 0, "affine": 0, "passthrough": 0}
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
        bits, method, gsz = classify(name, profile_bits)

        if method == "passthrough":
            t = idx.read_tensor(name, out_dtype=torch.float16)
            arr = t.numpy() if t.dtype != torch.bfloat16 else t.float().numpy().astype(np.float16)
            # AWQ fold (#44): if this is a post-attention layernorm (source
            # key `layers.N.ffn_norm.weight`) and we have a folded scale
            # for layer N, divide the layernorm weight by that scale.
            # Equivalent to inserting a runtime `x / s` before MoE matmuls,
            # but free at inference (just a smaller layernorm weight).
            if awq_layernorm_fold:
                m = re.match(r"^layers\.(\d+)\.ffn_norm\.weight$", name)
                if m:
                    layer_idx = int(m.group(1))
                    fold_scale = awq_layernorm_fold.get(layer_idx)
                    if fold_scale is not None:
                        arr = (arr.astype(np.float32) / fold_scale).astype(arr.dtype)
            add_tensor(name, arr)
            totals["passthrough"] += 1

        elif method == "affine":
            t = idx.read_tensor(name, out_dtype=torch.float32)
            w = mx.array(t.numpy())
            # FP4-origin routed experts at 4-bit: use MXFP4 mode which
            # exactly replicates source FP4 16-level log-spaced codebook.
            # All other tensors (FP8-origin attention, shared, embed, head):
            # use standard affine — they're FP8 source so linear 8-bit
            # affine represents them losslessly within 0.5% RMS.
            is_routed_expert = re.search(r"ffn\.experts\.\d+\.(w1|w2|w3)\.weight$", name) is not None
            # Direct-copy only applies if source is still in FP4 format (int8
            # packed + float8_e8m0fnu scale). BF16-dequant sources have no
            # .scale sibling — fall through to mx.quantize on the bf16 tensor.
            raw_w = idx.read_raw(name) if is_routed_expert and bits == 4 else None
            src_is_fp4 = raw_w is not None and raw_w.dtype == torch.int8
            if is_routed_expert and bits == 4 and src_is_fp4:
                # BIT-EXACT preservation: source is already MXFP4 format
                # (int8 packed FP4 + UE8M0 scale). MLX's mxfp4 uint32 layout
                # matches source int8 byte-for-byte (little-endian packing of
                # 4 source bytes per uint32, nibbles LSB→MSB). So we can
                # DIRECT-COPY without going through bf16 intermediate.
                sk = name[:-len(".weight")] + ".scale" if name.endswith(".weight") else name + ".scale"
                raw_s = idx.read_raw(sk)    # float8_e8m0fnu torch tensor
                # int8 (out, in/2) → reinterpret as uint8 bytes → pack into uint32
                w_bytes = raw_w.numpy().view(np.uint8)  # (out, in/2)
                out_dim, packed_in = w_bytes.shape
                in_dim = packed_in * 2
                assert in_dim % 8 == 0, f"in_dim {in_dim} not multiple of 8"
                # View as (out, in/8, 4 bytes) then as little-endian uint32
                w_u32 = w_bytes.reshape(out_dim, in_dim // 8, 4).copy().view(np.uint32).reshape(out_dim, in_dim // 8)
                # float8_e8m0fnu doesn't support .numpy() directly; reinterpret as uint8
                s_bytes = raw_s.view(torch.uint8).numpy()   # (out, in/32)
                assert s_bytes.shape == (out_dim, in_dim // 32), f"scale shape {s_bytes.shape}"
                base = name[:-len(".weight")] if name.endswith(".weight") else name
                add_tensor(f"{base}.weight", np.ascontiguousarray(w_u32))
                add_tensor(f"{base}.scales", np.ascontiguousarray(s_bytes))
                totals["affine"] += 1
            else:
                # AWQ fold consistency (#44): if `ffn_norm.weight` was
                # divided by scale `s`, every consumer of post_attn_layernorm
                # output must have its weight pre-multiplied by `s` (column
                # axis = in_features) so the runtime math stays equivalent.
                # shared_experts.w1 (gate) and w3 (up) consume the same input;
                # bake the scale into their weights here. w2 (down) consumes
                # swiglu(w1·w3) which is unchanged by the fold, so don't touch.
                if awq_layernorm_fold:
                    sm = re.match(
                        r"^layers\.(\d+)\.ffn\.shared_experts\.(w[13])\.weight$",
                        name,
                    )
                    if sm:
                        layer_idx = int(sm.group(1))
                        fold_scale = awq_layernorm_fold.get(layer_idx)
                        if fold_scale is not None and fold_scale.shape[-1] == w.shape[-1]:
                            w = w * mx.array(fold_scale)[None, :].astype(w.dtype)
                qw, qs, qb = mx.quantize(w, group_size=gsz or 64, bits=bits)
                base = name[:-len(".weight")] if name.endswith(".weight") else name
                add_tensor(f"{base}.weight", np.array(qw))
                add_tensor(f"{base}.scales", np.array(qs).astype(np.float16))
                add_tensor(f"{base}.biases", np.array(qb).astype(np.float16))
                totals["affine"] += 1

        elif method == "mxtq":
            t = idx.read_tensor(name, out_dtype=torch.float32)
            arr = t.numpy()
            base = name[:-len(".weight")] if name.endswith(".weight") else name

            # AWQ pre-scaling: experts in the same SwitchGLU layer share
            # the same input distribution (router applies the same x to
            # every active expert), so the scale belongs to the SwitchGLU
            # layer, not the per-expert weight. For DSV4 routed experts
            # the source key is `model.layers.L.mlp.ffn.experts.E.wK.weight`
            # but the runtime module is `model.layers.L.mlp.switch_mlp.{gate,up,down}_proj`.
            # Map projection: w1→gate_proj, w3→up_proj, w2→down_proj.
            if awq_norms:
                m = re.match(
                    r"^(model\.layers\.\d+)\.mlp\.ffn\.experts\.\d+\.(w[123])\.weight$",
                    name,
                )
                if m:
                    layer_pfx, w_name = m.group(1), m.group(2)
                    proj = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}[w_name]
                    runtime_key = f"{layer_pfx}.mlp.switch_mlp.{proj}"
                    norms = awq_norms.get(runtime_key)
                    if norms is not None and norms.shape[-1] == arr.shape[-1]:
                        scale = (norms.astype(np.float32) + awq_eps) ** awq_alpha
                        arr = arr * scale[np.newaxis, :]
                        # Zero-runtime AWQ (#44): gate_proj + up_proj scales
                        # are folded into the upstream post_attention_layernorm
                        # weight; do NOT emit them in awq_scales_out.
                        # down_proj input is swiglu(gate*up) — distinct
                        # distribution, must apply at runtime.
                        if proj == "down_proj":
                            awq_scales_out[runtime_key] = scale.astype(np.float32)

            result = tq_quantize_weight(arr, bits=bits, seed=SEED)
            add_tensor(f"{base}.tq_packed", np.asarray(result["packed"]))
            add_tensor(f"{base}.tq_norms", np.asarray(result["norms"]))
            add_tensor(f"{base}.tq_bits", np.array([bits], dtype=np.int32))
            totals["mxtq"] += 1
        else:
            raise ValueError(f"unknown method {method} for {name}")

        if (i + 1) % 500 == 0:
            print(f"    processed {i + 1}/{len(weight_keys)}  "
                  f"mxtq={totals['mxtq']} affine={totals['affine']} "
                  f"passthrough={totals['passthrough']}  "
                  f"({time.time() - t_start:.0f}s)", flush=True)

    flush_shard()

    # Rename shards to final count
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

    # config.json + jang_config.json
    src_cfg = json.loads((src / "config.json").read_text())
    src_cfg.pop("quantization_config", None)
    # ROOT-CAUSE FIX 2026-04-24: top-level MUST stay bits=8 mode=affine to
    # match the dominant 8-bit-affine layout (attention/shared/embed/head/
    # compressor/indexer). Setting top-level bits=2 (old JANGTQ2 + JANG_2L
    # behavior) made MLX dequantize 8-bit-stored weights with the 2-bit
    # kernel → silent garbage. See research/JANGTQ-CONFIG-METADATA-BUG-2026-04-24.md.
    quant_cfg: dict = {"group_size": 32, "bits": 8, "mode": "affine"}
    n_layers = src_cfg.get("num_hidden_layers", 43)
    if FORMAT == "jangtq" and profile_bits == 4:
        # Routed experts → MXFP4 codebook (per-module override).
        for L in range(n_layers):
            for proj in ("gate_proj", "down_proj", "up_proj"):
                path = f"model.layers.{L}.mlp.switch_mlp.{proj}"
                quant_cfg[path] = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
    elif FORMAT == "jangtq" and profile_bits == 2:
        # Routed experts → MXTQ codebook stored as tq_packed/tq_norms/tq_bits
        # (no `.weight` key). MLX's standard quant loader cannot see them.
        # Runtime swaps in TurboQuantSwitchLinear from sanitize().
        # No per-module override needed — non-expert layers correctly inherit
        # the 8-bit affine top-level default.
        pass
    else:
        # JANG_2L (FORMAT=jang): routed experts at profile_bits affine,
        # everything else at 8-bit affine. Per-module overrides REQUIRED
        # so MLX dequantizes each at the correct bit-width.
        for L in range(n_layers):
            for proj in ("gate_proj", "down_proj", "up_proj"):
                path = f"model.layers.{L}.mlp.switch_mlp.{proj}"
                quant_cfg[path] = {
                    "group_size": 32, "bits": profile_bits, "mode": "affine",
                }
    src_cfg["quantization"] = quant_cfg
    src_cfg["_name_or_path"] = f"DSV4-Flash-{profile_name}"
    # Swift DeepseekV4JANGTQConfiguration expects these top-level keys (Codable
    # auto-decode requires the field even with a Swift default; quantization
    # nested block isn't enough). Fix Mac Studio Swift bench failures observed
    # 2026-04-24: "Missing field 'routed_expert_bits'", then "'group_size'".
    src_cfg["routed_expert_bits"] = profile_bits
    src_cfg["group_size"] = quant_cfg.get("group_size", 32)
    src_cfg["mxtq_seed"] = SEED

    # transformers ≥4.50 expects `rope_parameters` (with rope_theta inside)
    # rather than the legacy `rope_scaling` + top-level `rope_theta` pair.
    # Without this mirror, mlx-lm's `load_tokenizer` raises "Missing required
    # keys in `rope_parameters` for 'rope_type'='yarn': {'rope_theta'}" and
    # falls back to bare PreTrainedTokenizerFast — which drops chat_template,
    # special tokens, and BOS/EOS handling. Bench harness has to manually
    # re-inject chat_template every time.
    if "rope_parameters" not in src_cfg:
        rs = src_cfg.get("rope_scaling")
        if isinstance(rs, dict):
            rp = dict(rs)
            rp.setdefault("rope_type", rp.pop("type", "yarn"))
            rp.setdefault("rope_theta", src_cfg.get("rope_theta", 10000))
        else:
            rp = {
                "rope_type": "default",
                "rope_theta": src_cfg.get("rope_theta", 10000),
            }
        # transformers validation rejects ints for these fields → cast to float.
        for k in ("factor", "beta_fast", "beta_slow", "rope_theta"):
            if k in rp and not isinstance(rp[k], float):
                rp[k] = float(rp[k])
        src_cfg["rope_parameters"] = rp
    (dst / "config.json").write_text(json.dumps(src_cfg, indent=2))

    # Bake DSV4 chat_template into tokenizer_config.json so Swift consumers
    # don't need a runtime fallback. Upstream DSV4 ships only the Python
    # encoding_dsv4.py module — no Jinja template — so for Swift bundles
    # we inject a faithful Jinja port of encoding_dsv4.encode_messages
    # (system / user / assistant + thinking_mode + tool_result).
    # See research/DSV4-RUNTIME-ARCHITECTURE.md §31. The vMLX Swift runtime
    # has the same template baked into CapabilityDetector.deepseekV4BuiltinTemplate
    # as a legacy-bundle compatibility path; once all bundles ship the
    # baked template the Swift injection becomes a true fallback.
    tok_cfg_path = dst / "tokenizer_config.json"
    if tok_cfg_path.exists():
        try:
            tok_cfg = json.loads(tok_cfg_path.read_text())
        except Exception:
            tok_cfg = {}
        if not tok_cfg.get("chat_template"):
            tok_cfg["chat_template"] = DSV4_CHAT_TEMPLATE_JINJA
            tok_cfg_path.write_text(json.dumps(tok_cfg, indent=2, ensure_ascii=False))
            print(f"  [chat-template] injected DSV4 canonical template into tokenizer_config.json", flush=True)

    (dst / "jang_config.json").write_text(json.dumps({
        "weight_format": "mxfp4_mixed" if FORMAT == "jangtq" and profile_bits == 4 else (
            "affine" if FORMAT == "jang" else "mxtq"
        ),
        "profile": profile_name,
        "mxtq_seed": SEED,
        "source_model": str(src),
        "source_config": {
            "n_routed_experts": src_cfg.get("n_routed_experts"),
            "num_hidden_layers": src_cfg.get("num_hidden_layers"),
            "n_hash_layers": src_cfg.get("num_hash_layers"),
        },
        "mxtq_bits": {
            "routed_expert": profile_bits,
            "attention": 8,
            "shared_expert": 8,
            "embed_tokens": 8,
            "lm_head": 8,
            "norms_router_hc": 16,
        },
        # DSV4-Flash chat + reasoning + tool-parser metadata for runtime wiring.
        # Python loader can use this to auto-wire chat encoding; Swift port
        # reads this to know which encoder to use.
        "model_family": "deepseek_v4",
        "chat": {
            "encoder": "encoding_dsv4",  # Python module in ./encoding/
            "encoder_fn": "encode_messages",
            "chat_template_source": "builtin_encoding_module",
            "has_tokenizer_chat_template": False,  # tokenizer_config.json has no chat_template
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
                # "chat" mode: prompt ends with <Assistant></think> (empty reasoning closed)
                # "thinking" mode: prompt ends with <Assistant><think> (open, model fills)
                "reasoning_effort_levels": ["max", "high", None],
                "drop_earlier_reasoning": True,  # drop_thinking in encode_messages
            },
            "tool_calling": {
                "supported": True,
                "parser": "dsml",  # DeepSeek Markup Language (｜DSML｜...)
                "dsml_token": "｜DSML｜",
                "tool_calls_block": "tool_calls",
                "invoke_block": "invoke",
                "parameter_block": "parameter",
                "tool_output_tag": "tool_result",
            },
            "sampling_defaults": {
                "temperature": 0.6,   # from inference/generate.py default
                "top_p": 0.95,
                "max_new_tokens": 300,
            },
        },
    }, indent=2))

    # Copy aux files (tokenizer, chat template, modeling files if any)
    copied = 0
    for p in src.iterdir():
        if p.is_file() and not p.name.endswith(".safetensors") \
                and p.name not in ("config.json", "model.safetensors.index.json"):
            shutil.copy2(p, dst / p.name)
            copied += 1
    # Also copy encoding/ directory (DSV4's Python chat-template impl)
    enc = src / "encoding"
    if enc.is_dir():
        shutil.copytree(enc, dst / "encoding", dirs_exist_ok=True)
        copied += 1
    print(f"[convert] copied {copied} aux files/dirs")

    # AWQ scales sidecar — load_jangtq.py reads this at runtime and
    # divides x by the per-channel scale before each TQ matmul. Each
    # routed-expert weight in this layer was already pre-scaled at
    # quantize time (see mxtq path above), so the math is:
    #   y = (x / s) @ Q(W * s)  ≈  x @ W
    # with outlier input channels preserved at higher effective resolution.
    if awq_scales_out:
        from safetensors.numpy import save_file as _safe_save
        _safe_save(awq_scales_out, str(dst / "awq_scales.safetensors"))
        print(f"  AWQ scales: wrote {len(awq_scales_out)} entries to "
              f"awq_scales.safetensors")

    elapsed = time.time() - t_start
    print(f"\nDONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  mxtq={totals['mxtq']}  affine={totals['affine']}  passthrough={totals['passthrough']}")
    print(f"  output size: {total_bytes / 1e9:.1f} GB")

    # Build Swift runtime sidecar so Swift runtimes (vmlx-swift-lm, vmlxctl,
    # Osaurus) can load JANGTQ bundles without fatalError on missing
    # `jangtq_runtime.safetensors`. Python loader doesn't need it, but
    # uploading without it bricks Swift/Osaurus consumers. Mirrors the
    # pattern in `convert_qwen35_jangtq.py:520-533` and the JANGTQ
    # converters for MiniMax / GLM-5.1.
    if FORMAT == "jangtq":
        print(f"\n  Building jangtq_runtime.safetensors sidecar...", flush=True)
        try:
            from jang_tools.build_jangtq_sidecar import main as _build_sidecar
            _saved_argv = sys.argv
            sys.argv = ["build_jangtq_sidecar", str(dst)]
            try:
                _build_sidecar()
            finally:
                sys.argv = _saved_argv
            print(f"  [sidecar] OK", flush=True)
        except (Exception, SystemExit) as _e:
            print(f"  [sidecar] FAILED: {_e}", flush=True)
            print(f"  [sidecar] run manually before upload:"
                  f" `python3 -m jang_tools.build_jangtq_sidecar {dst}`", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--profile", type=int, default=2, choices=(2, 3, 4, 5, 6, 8),
                    help="Routed-expert bit count.")
    ap.add_argument("--format", default="jangtq", choices=("jang", "jangtq"),
                    help="jang=standard affine everywhere; jangtq=MXTQ for 2-bit routed.")
    ap.add_argument("--awq-norms", type=Path, default=None,
                    help="Optional awq_activations.safetensors from "
                         "`python3 -m jang_tools.awq_capture`. When provided, "
                         "routed-expert mxtq weights are pre-scaled by "
                         "(norms+eps)^alpha before quantization, and the scale "
                         "is written to awq_scales.safetensors for runtime use.")
    ap.add_argument("--awq-alpha", type=float, default=0.25,
                    help="AWQ alpha exponent. 0.25 empirically optimal, 1/3 "
                         "theoretically. Only relevant with --awq-norms.")
    args = ap.parse_args()
    global FORMAT
    FORMAT = args.format
    convert(args.src, args.dst, args.profile,
            awq_norms_path=args.awq_norms, awq_alpha=args.awq_alpha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
