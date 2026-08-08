"""LiquidAI LFM2.5 dense hybrid → MXFP8 / JANG_6M with AWQ folds + GPTQ QAT.

Created by Jinho Jang (eric@jangq.ai) — 2026-08-04.

Target source: LiquidAI/LFM2.5-2.6B (release 2026-07-28, bf16, ~5.3 GB,
model_type="lfm2", Lfm2ForCausalLM). Dense hybrid: 22 double-gated short-conv
(LIV) blocks + 8 GQA attention layers (32q/8kv, head_dim 64, per-head q/k
RMSNorm, NeoX RoPE θ=1e7), SwiGLU FFN (w1/w3 -> w2, inter 10752), tied
embeddings (vocab 128000), 128K context, plain Llama RMSNorm — NO +1 shift.

Profiles (dense model — only the dense-safe near-lossless tiers exist here;
see memory `project_dense_vs_moe`):
  MXFP8    every 2-D weight mx.quantize mode="mxfp8" gs32 (uniform 8-bit,
           e4m3 codes + e8m0 scales); conv depthwise kernels + norms fp16.
  JANG_6M  affine gs32: token-mixing operators (attention q/k/v/out and
           conv in/out projections) 8-bit CRITICAL, FFN w1/w2/w3 6-bit,
           tied embedding 6-bit; conv kernels + norms fp16.

AWQ (both profiles, function-preserving folds — no runtime cost, no sidecar):
  s1 = clip(geomean-normalized max|ffn_in|^0.25, 0.5..2)   per layer
       w1/w3 input columns *= s1, ffn_norm.weight /= s1
  s2 = clip(geomean-normalized max|swiglu|^0.25, 0.5..2)   per layer
       w2 input columns *= s2, w3 output rows /= s2
  Attention/conv projections stay 8-bit RTN in both profiles: measured 8-bit
  affine/mxfp8 error is already near-lossless there, and the o_proj fold
  would need kv-head-tiled scale constraints for zero gain.

QAT (both profiles): GPTQ codes-only learned rounding on the FIXED storage
grid for w1/w3/w2 per layer, BRECQ-sequenced (w2 inputs derived through the
already-quantized folded w1/w3 + SwiGLU), with a per-tensor best-of-RTN
guard. The tied embedding keeps RTN: it serves both the input lookup and the
lm_head matmul, and GPTQ would trade lookup accuracy for logit accuracy.

Output loads with STOCK mlx_lm (`mlx_lm.load` / vmlx) — config carries the
same key set the vendor's own MLX export uses (block_ff_dim, top-level
rope_theta, eos list, quantization + quantization_config stanzas).

Usage:
  python -m jang_tools.lfm25.convert \
      --src ~/.mlxstudio/models/LiquidAI/LFM2.5-2.6B \
      --out ~/.mlxstudio/models/JANGQ-AI/LFM2.5-2.6B-JANG_6M \
      --profile JANG_6M --calib /path/lfm25_calib.safetensors
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file as sf_load
from safetensors.numpy import save_file

from jang_tools.capabilities import build_capabilities
from jang_tools.lfm25.qat_gptq import gptq_codes, make_grid, pack_self_test

MAX_SHARD = 4_500_000_000

PROFILES: dict[str, dict] = {
    #             mode      operator  ffn  embed  top
    "MXFP8":   {"mode": "mxfp8",  "operator": 8, "ffn": 8, "embed": 8, "top": 8},
    "JANG_6M": {"mode": "affine", "operator": 8, "ffn": 6, "embed": 6, "top": 8},
}

# Sampling parameters LiquidAI documents on the model card. For LFM2.5-2.6B
# the card and generation_config.json agree (temperature/top_k/
# repetition_penalty) — kept as an explicit constant + hard gate anyway so a
# future vendor revision that drops one from generation_config cannot ship a
# silently different sampling contract (the Laguna-XS top_k=20 lesson,
# memory `feedback_sampling_defaults_two_file_contract`).
CARD_DOCUMENTED_SAMPLING = {
    "temperature": 0.1,
    "top_k": 50,
    "repetition_penalty": 1.1,
}

SIDECAR_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "chat_template.jinja",
    "LICENSE",
)

_ATTN_PROJ = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "out_proj.weight")
_FFN_PROJ = ("w1", "w2", "w3")


def tier_bits(name: str, profile: dict) -> int | None:
    """Bit width for a tensor, or None for fp16 passthrough.

    Raises on unknown tensor names — this converter refuses to guess
    (memory `feedback_structural_verification_not_enough`).
    """
    if name.endswith("conv.conv.weight"):
        return None  # depthwise 3-tap Conv1d — tiny, not group-quantizable
    if ".self_attn." in name and name.endswith(("q_layernorm.weight", "k_layernorm.weight")):
        return None
    if "norm" in name.split(".")[-2]:
        return None  # operator_norm / ffn_norm / embedding_norm
    if ".self_attn." in name and name.endswith(_ATTN_PROJ):
        return profile["operator"]
    if ".conv." in name and name.endswith(("in_proj.weight", "out_proj.weight")):
        return profile["operator"]
    if ".feed_forward." in name and name.endswith(
        ("w1.weight", "w2.weight", "w3.weight")
    ):
        return profile["ffn"]
    if name.endswith("embed_tokens.weight"):
        return profile["embed"]
    raise SystemExit(f"unknown tensor name (refusing to guess a tier): {name}")


def _scan_source(src: Path) -> dict[str, tuple[list[int], Path]]:
    index_path = src / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    files = {src / f for f in weight_map.values()}
    out: dict[str, tuple[list[int], Path]] = {}
    for f in sorted(files):
        with safe_open(str(f), framework="numpy") as sf:
            for key in sf.keys():
                out[key] = (list(sf.get_slice(key).get_shape()), f)
    return out


def _load_tensor(sf_path: Path, name: str, shape: list[int]) -> np.ndarray:
    try:
        with safe_open(str(sf_path), framework="numpy") as f:
            t = f.get_tensor(name)
        if not isinstance(t, np.ndarray):
            t = np.array(t)
    except TypeError:
        # numpy has no bfloat16 — read raw and widen to fp32.
        from jang_tools.calibrate import _load_bf16_tensor

        t = _load_bf16_tensor(sf_path, name, tuple(shape))
    if t.dtype != np.float32:
        return t.astype(np.float32)
    return t if t.flags.writeable else t.copy()


def _awq_scale(max_abs: np.ndarray, alpha: float = 0.25,
               clip_lo: float = 0.5, clip_hi: float = 2.0) -> np.ndarray:
    """Geomean-normalized AWQ channel scales from per-channel max|x|."""
    s = np.power(np.maximum(max_abs.astype(np.float64), 1e-8), alpha)
    s = s / np.exp(np.mean(np.log(s)))
    return np.clip(s, clip_lo, clip_hi).astype(np.float32)


def _silu(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = x[pos] / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = x[~pos] * ex / (1.0 + ex)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert LFM2.5 source to MXFP8 / JANG_6M with AWQ + QAT."
    )
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--profile", choices=sorted(PROFILES), required=True)
    p.add_argument("--calib", type=Path, default=None,
                   help="lfm25_calib.safetensors from jang_tools.lfm25.calibrate")
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--no-awq", action="store_true")
    p.add_argument("--no-qat", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:  # noqa: C901
    args = parse_args()
    src, out = args.src.expanduser(), args.out.expanduser()
    profile_name = args.profile
    profile = PROFILES[profile_name]
    mode = profile["mode"]
    gs = args.group_size
    if mode == "mxfp8" and gs != 32:
        raise SystemExit("MXFP8 requires group_size=32")
    weight_format = "mxfp8" if mode == "mxfp8" else "jang_affine"
    use_awq = not args.no_awq
    use_qat = not args.no_qat
    if (use_awq or use_qat) and args.calib is None:
        raise SystemExit("--calib is required unless --no-awq AND --no-qat")

    config = json.loads((src / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "lfm2":
        raise SystemExit(f"expected model_type=lfm2, got {config.get('model_type')!r}")
    if config.get("architectures") != ["Lfm2ForCausalLM"]:
        raise SystemExit(f"unexpected architectures: {config.get('architectures')!r}")
    if not config.get("tie_word_embeddings"):
        raise SystemExit("expected tie_word_embeddings=true (no lm_head tensor)")
    layer_types = config["layer_types"]
    n_layers = int(config["num_hidden_layers"])
    assert len(layer_types) == n_layers
    full_attn_idxs = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    hidden = int(config["hidden_size"])
    inter = int(config["intermediate_size"])

    print("=" * 72)
    print(f"  LFM2.5 (hybrid LIV-conv + GQA) -> {profile_name}")
    print("=" * 72)
    print(f"  Source:   {src}")
    print(f"  Output:   {out}")
    print(f"  Layers:   {n_layers} ({n_layers - len(full_attn_idxs)} conv + "
          f"{len(full_attn_idxs)} attention @ {full_attn_idxs})")
    print(f"  Bits:     operator={profile['operator']} ffn={profile['ffn']} "
          f"embed={profile['embed']} mode={mode} gs={gs}")
    print(f"  AWQ:      {'fold w1/w3 (ffn_norm) + w2 (w3 rows)' if use_awq else 'OFF'}")
    print(f"  QAT:      {'GPTQ codes-only on w1/w2/w3, best-of-RTN' if use_qat else 'OFF'}")
    print(f"  Norms:    plain RMSNorm, NO +1 shift, fp16 passthrough")

    tensors = _scan_source(src)
    print(f"  Found {len(tensors)} tensors")

    if args.dry_run:
        counts: dict[str, int] = {}
        for name in tensors:
            b = tier_bits(name, profile)
            k = "fp16" if b is None else f"{mode}-{b}"
            counts[k] = counts.get(k, 0) + 1
        print(json.dumps(counts, indent=2, sort_keys=True))
        return

    pack_self_test()

    calib: dict[str, np.ndarray] = {}
    s1_all: dict[int, np.ndarray] = {}
    s2_all: dict[int, np.ndarray] = {}
    if args.calib is not None:
        calib = sf_load(str(args.calib.expanduser()))
        for i in range(n_layers):
            if use_awq:
                s1_all[i] = _awq_scale(calib[f"layers.{i}.ffn_input_max"])
                s2_all[i] = _awq_scale(calib[f"layers.{i}.ffn_intermediate_max"])
            else:
                s1_all[i] = np.ones(hidden, dtype=np.float32)
                s2_all[i] = np.ones(inter, dtype=np.float32)

    out.mkdir(parents=True, exist_ok=True)
    for stale in list(out.glob("*.safetensors")) + list(
        out.glob("model.safetensors.index.json")
    ):
        stale.unlink()

    shard_idx = 0
    shard_tensors: dict[str, np.ndarray] = {}
    shard_bytes = 0
    shard_map: dict[str, str] = {}
    overrides: dict[str, dict] = {}
    n_quant = n_pass = 0
    qat_report: dict[str, dict] = {}
    t_start = time.time()

    def flush_shard() -> None:
        nonlocal shard_idx, shard_tensors, shard_bytes
        if not shard_tensors:
            return
        shard_idx += 1
        fname = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        save_file(shard_tensors, str(out / fname))
        for k in shard_tensors:
            shard_map[k] = fname
        print(f"    Shard {shard_idx}: {len(shard_tensors)} tensors, "
              f"{shard_bytes / 1e9:.2f} GB")
        shard_tensors, shard_bytes = {}, 0

    def add_arrays(arrs: dict[str, np.ndarray]) -> None:
        nonlocal shard_bytes
        for k, v in arrs.items():
            shard_tensors[k] = v
            shard_bytes += v.nbytes
        if shard_bytes >= MAX_SHARD:
            flush_shard()

    def note_override(base: str, bits: int) -> None:
        if bits != profile["top"]:
            overrides[base] = {"group_size": gs, "bits": bits, "mode": mode}

    def quant_rtn(name: str, w: np.ndarray, bits: int) -> None:
        nonlocal n_quant
        base = name[: -len(".weight")]
        grid = make_grid(w, mode, gs, bits)
        add_arrays(grid.emit(base, grid.rtn_codes(w)))
        note_override(base, bits)
        n_quant += 1

    def passthrough(name: str, w: np.ndarray) -> None:
        nonlocal n_pass
        add_arrays({name: w.astype(np.float16)})
        n_pass += 1

    # ── embed (tied; RTN by design — dual lookup/lm_head use) ────────────
    print("  [1/3] embedding")
    name = "model.embed_tokens.weight"
    shape, f = tensors.pop(name)
    quant_rtn(name, _load_tensor(f, name, shape), profile["embed"])
    gc.collect()
    mx.clear_cache()

    # ── per-layer pass ────────────────────────────────────────────────────
    print("  [2/3] layers")
    for li in range(n_layers):
        prefix = f"model.layers.{li}."
        layer_names = sorted(n for n in tensors if n.startswith(prefix))
        s1, s2 = s1_all.get(li), s2_all.get(li)

        # FFN trio handled together (folds + BRECQ-sequenced QAT)
        ffn = {p: f"{prefix}feed_forward.{p}.weight" for p in _FFN_PROJ}
        w1 = _load_tensor(tensors[ffn["w1"]][1], ffn["w1"], tensors[ffn["w1"]][0])
        w3 = _load_tensor(tensors[ffn["w3"]][1], ffn["w3"], tensors[ffn["w3"]][0])
        w2 = _load_tensor(tensors[ffn["w2"]][1], ffn["w2"], tensors[ffn["w2"]][0])
        if use_awq:
            w1 *= s1[None, :]
            w3 *= s1[None, :]
            w3 /= s2[:, None]
            w2 *= s2[None, :]

        ffn_bits = profile["ffn"]
        grids = {p: make_grid(w, mode, gs, ffn_bits)
                 for p, w in (("w1", w1), ("w3", w3), ("w2", w2))}

        if use_qat:
            x1 = calib[f"layers.{li}.x1"].astype(np.float32)
            if use_awq:
                x1 = x1 / s1[None, :]
            codes1, st1 = gptq_codes(w1, grids["w1"], x1,
                                     label=f"L{li}.w1 {mode}{ffn_bits}",
                                     verbose=(li % 10 == 0))
            codes3, st3 = gptq_codes(w3, grids["w3"], x1,
                                     label=f"L{li}.w3 {mode}{ffn_bits}",
                                     verbose=False)
            # BRECQ: w2 sees the swiglu output of the QUANTIZED w1/w3
            g_act = x1 @ grids["w1"].dequant(codes1).T
            u_act = x1 @ grids["w3"].dequant(codes3).T
            x2 = _silu(g_act) * u_act
            del g_act, u_act
            codes2, st2 = gptq_codes(w2, grids["w2"], x2,
                                     label=f"L{li}.w2 {mode}{ffn_bits}",
                                     verbose=(li % 10 == 0))
            del x1, x2
            qat_report[str(li)] = {"w1": st1, "w2": st2, "w3": st3}
        else:
            codes1 = grids["w1"].rtn_codes(w1)
            codes3 = grids["w3"].rtn_codes(w3)
            codes2 = grids["w2"].rtn_codes(w2)

        for p, codes in (("w1", codes1), ("w2", codes2), ("w3", codes3)):
            base = ffn[p][: -len(".weight")]
            add_arrays(grids[p].emit(base, codes))
            note_override(base, ffn_bits)
            n_quant += 1
        del w1, w2, w3, grids, codes1, codes2, codes3

        # remaining layer tensors
        for name in layer_names:
            if name in (ffn["w1"], ffn["w2"], ffn["w3"]):
                tensors.pop(name)
                continue
            shape, f = tensors.pop(name)
            w = _load_tensor(f, name, shape)
            bits = tier_bits(name, profile)
            if name.endswith("ffn_norm.weight") and use_awq:
                # inverse of the w1/w3 input fold — function-preserving
                w = w / s1
            if bits is None or w.ndim < 2 or name.endswith("conv.conv.weight"):
                passthrough(name, w)
            else:
                quant_rtn(name, w, bits)
            del w
        gc.collect()
        mx.clear_cache()
        if li % 10 == 9:
            print(f"    layer {li + 1}/{n_layers} done "
                  f"({time.time() - t_start:.0f}s elapsed)")

    # ── remaining top-level tensors ───────────────────────────────────────
    for name in sorted(tensors):
        shape, f = tensors[name]
        passthrough(name, _load_tensor(f, name, shape))
    flush_shard()

    # ── index ─────────────────────────────────────────────────────────────
    print("  [3/3] metadata")
    for idx in range(1, shard_idx + 1):
        old = out / f"model-{idx:05d}-of-XXXXX.safetensors"
        if old.exists():
            old.rename(out / f"model-{idx:05d}-of-{shard_idx:05d}.safetensors")
    shard_map = {k: v.replace("XXXXX", f"{shard_idx:05d}") for k, v in shard_map.items()}
    total_size = sum((out / f).stat().st_size for f in set(shard_map.values()))
    (out / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"format": weight_format, "total_size": total_size},
             "weight_map": shard_map},
            indent=2,
        ),
        encoding="utf-8",
    )

    # ── sidecars ──────────────────────────────────────────────────────────
    for fname in SIDECAR_FILES:
        srcf = src / fname
        if srcf.exists():
            shutil.copy2(srcf, out / fname)
    tmpl_p = out / "chat_template.jinja"
    if not tmpl_p.exists():
        raise SystemExit("source ships no chat_template.jinja — refusing to build "
                         "a bundle without the reasoning-default-on template")
    template_text = tmpl_p.read_text(encoding="utf-8")

    # Inline the template into tokenizer_config.json too: transformers v5
    # reads the standalone .jinja, but older consumers and several MLX
    # runtimes only look at tokenizer_config (laguna include-stub lesson).
    tok_cfg_p = out / "tokenizer_config.json"
    tok_cfg = json.loads(tok_cfg_p.read_text(encoding="utf-8"))
    if not tok_cfg.get("chat_template"):
        tok_cfg["chat_template"] = template_text
        tok_cfg_p.write_text(json.dumps(tok_cfg, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print("  chat_template: inlined .jinja into tokenizer_config.json")

    # ── reasoning default derived from the SHIPPED template ──────────────
    # (commit de67683 lesson: never hardcode; render the actual artifact)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(out))
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "ping"}],
        add_generation_prompt=True, tokenize=False,
    )
    thinking_always_on = rendered.rstrip("\n").endswith("<think>")
    if not thinking_always_on:
        raise SystemExit(
            "shipped chat template does not pre-open <think> on the generation "
            f"prompt — LFM2.5-2.6B is an always-thinking model. Rendered tail: "
            f"{rendered[-120:]!r}"
        )
    print("  reasoning: template pre-opens <think> — always-on confirmed")

    # ── config.json (vendor-MLX-parity keys + JANG contract) ─────────────
    config.pop("quantization_config", None)
    config.pop("auto_map", None)
    # keys the vendor's own MLX export adds for mlx_lm ModelArgs compat
    config.setdefault("block_ff_dim", inter)
    if "rope_parameters" in config:
        config.setdefault("rope_theta", config["rope_parameters"].get("rope_theta"))
    if not isinstance(config.get("eos_token_id"), list):
        config["eos_token_id"] = [config["eos_token_id"]]
    config["weight_format"] = weight_format
    quant_block: dict = {"group_size": gs, "bits": profile["top"], "mode": mode}
    quant_block.update(overrides)
    config["quantization"] = quant_block
    config["quantization_config"] = json.loads(json.dumps(quant_block))

    config["jang_runtime"] = {
        "architecture": "lfm2_hybrid",
        "cache_layout": "lfm2_hybrid_v1",
        "num_hidden_layers": n_layers,
        "full_attention_layers": full_attn_idxs,
        "num_conv_layers": n_layers - len(full_attn_idxs),
        "conv_L_cache": int(config["conv_L_cache"]),
        "conv_kernel_layout": "hf_channels_first",  # (C,1,K) on disk; mlx_lm sanitize() transposes
        "cache": {"full_attention": "kv", "conv": "arrays_state_size_1"},
        "norm_convention": "llama_rmsnorm_no_plus_one",
        "qk_norm": "per_head_rmsnorm",
        "rope": {"type": "neox", "theta": config["rope_theta"],
                 "dims": hidden // int(config["num_attention_heads"]),
                 "applied_after_qk_norm": True},
        "attention": {"type": "gqa",
                      "n_heads": int(config["num_attention_heads"]),
                      "n_kv_heads": int(config["num_key_value_heads"]),
                      "head_dim": hidden // int(config["num_attention_heads"])},
        "tied_embeddings": True,
    }

    # ── jang_config.json ──────────────────────────────────────────────────
    gen_cfg = json.loads((out / "generation_config.json").read_text(encoding="utf-8"))
    sampling_defaults = {
        k: gen_cfg[k]
        for k in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty")
        if k in gen_cfg
    }
    for k, v in CARD_DOCUMENTED_SAMPLING.items():
        sampling_defaults.setdefault(k, v)

    chat_block = {
        "reasoning": {
            "supported": True,
            # vmlx registry parser names: think-tag parsing is the qwen3
            # parser; tool calls use the Liquid Python-call "lfm2" parser.
            "parser": "qwen3",
            "default_enabled": thinking_always_on,
            "always_on": thinking_always_on,
            "default_mode": "think",
            # No off-switch: the template opens <think> unconditionally and
            # exposes no enable_thinking kwarg (only preserve_thinking).
            "modes": ["think"],
        },
        "tool_calling": {"supported": True, "parser": "lfm2"},
        "sampling_defaults": sampling_defaults,
        "template_kwargs_defaults": {"preserve_thinking": False},
    }

    # sampling two-file reconciliation + hard gate
    gen_out_p = out / "generation_config.json"
    added = {}
    for k, v in sampling_defaults.items():
        if k not in gen_cfg:
            gen_cfg[k] = v
            added[k] = v
    if added:
        gen_out_p.write_text(json.dumps(gen_cfg, indent=2) + "\n", encoding="utf-8")
        print(f"  generation_config: added card-documented {added}")
    disagree = {
        k: (v, gen_cfg.get(k))
        for k, v in sampling_defaults.items()
        if gen_cfg.get(k) != v
    }
    if disagree:
        raise SystemExit(
            "sampling defaults disagree between jang_config.chat and "
            f"generation_config.json (key: jang vs gen): {disagree}"
        )

    awq_block = {
        "applied": use_awq,
        "method": "activation-max channel folds (alpha=0.25, clip 0.5..2.0, "
                  "geomean-normalized)",
        "folds": ["w1/w3 input <- ffn_norm", "w2 input <- w3 rows"] if use_awq else [],
        "calibration": (calib and args.calib.name) or None,
    }
    if use_qat:
        imp = [s["improvement"] for lr in qat_report.values() for s in lr.values()]
        used_gptq = sum(1 for lr in qat_report.values()
                        for s in lr.values() if s["used"] == "gptq")
        qat_block = {
            "applied": True,
            "method": "gptq-codes-only-fixed-grid (BRECQ-sequenced w1/w3->w2, "
                      "best-of-RTN guard)",
            "tensors": len(imp),
            "tensors_using_gptq": used_gptq,
            "mean_recon_improvement": round(float(np.mean(imp)), 4),
            "report": "qat_report.json",
        }
        (out / "qat_report.json").write_text(
            json.dumps(qat_report, indent=1) + "\n", encoding="utf-8")
    else:
        qat_block = {"applied": False}

    jang_config = {
        "version": 2,
        "weight_format": weight_format,
        "profile": profile_name,
        "source_model": {"name": src.name, "architecture": "lfm2",
                         "vendor": "LiquidAI"},
        "has_vision": False,
        "has_audio": False,
        "quantization": {
            "method": weight_format,
            "quantization_backend": "mx.quantize",
            "mode": mode,
            "group_size": gs,
            "tier_bits": {"operator": profile["operator"], "ffn": profile["ffn"],
                          "embed": profile["embed"]},
            "norm_convention": "llama_rmsnorm_no_plus_one",
            "per_module_override_count": len(overrides),
            "quantized_tensor_count": n_quant,
            "passthrough_tensor_count": n_pass,
            "awq": awq_block,
            "qat": qat_block,
        },
        "chat": chat_block,
        "runtime": {
            "architecture": "lfm2_hybrid",
            "loads_with": "stock mlx_lm >= 0.31 (mlx_lm.load), no custom code",
            "total_weight_bytes": total_size,
            "total_weight_gb": round(total_size / (1024 ** 3), 2),
        },
    }

    # capabilities LAST, after all jang_config/config mutations (verify gate
    # recomputes and requires exact equality). Text-only is weight-gated:
    # no vision/audio tensors exist; the template's <image> item handling is
    # inert placeholder text on this model.
    caps = build_capabilities(jang_config, config, out)
    if caps is None:
        raise SystemExit("capabilities: family resolution failed for lfm2")
    if not caps["think_in_template"]:
        raise SystemExit(
            "capabilities stamped think_in_template=False but the template "
            "pre-opens <think> — capabilities.py template-evidence override "
            "missing?"
        )
    if caps["has_vision"] or caps["has_audio"] or caps["has_video"]:
        raise SystemExit(f"capabilities claim non-text modality on a text-only "
                         f"model: {caps}")
    config["capabilities"] = caps
    jang_config["capabilities"] = caps

    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out / "jang_config.json").write_text(json.dumps(jang_config, indent=2),
                                          encoding="utf-8")

    print(f"\n  Quantized {n_quant} tensors, passed through {n_pass}")
    print(f"  Per-module overrides: {len(overrides)}")
    if use_qat:
        print(f"  QAT: {qat_block['tensors_using_gptq']}/{qat_block['tensors']} "
              f"tensors ship GPTQ codes, mean recon improvement "
              f"{qat_block['mean_recon_improvement']:+.1%}")
    print(f"  Total: {total_size / 1024 ** 3:.2f} GB across {shard_idx} shard(s)")
    print(f"  Done in {time.time() - t_start:.0f}s -> {out}")


if __name__ == "__main__":
    main()
