"""poolside Laguna (XS.2 / M.1 / S-2.1) -> all-affine JANG converter.

Created by Jinho Jang (eric@jangq.ai) — 2026-07-21.

Target source: poolside/Laguna-S-2.1 (release 2026-07-21, BF16, ~235 GB,
model_type="laguna", LagunaForCausalLM). Also loads the smaller family
members (XS.2 33B, M.1) — everything below is config-driven.

Architecture (verified against S-2.1 config.json + safetensors index):
  - 48 decoder layers, layer 0 dense MLP (mlp_layer_types[0]="dense",
    intermediate 12288), layers 1..47 sparse MoE
  - MoE: 256 routed experts (moe_intermediate_size 1024), top-10, sigmoid
    router + e_score_correction_bias (DSV3 aux-free), norm_topk_prob,
    routed_scaling_factor 2.5, 1 shared expert (inter 1024)
  - Hybrid attention: full_attention (48 heads, YaRN theta 500k,
    partial_rotary 0.5) / sliding_attention (72 heads, window 512, default
    RoPE theta 10k) — per-layer head counts, GQA 8 kv-heads, head_dim 128
  - per-head q_norm/k_norm RMSNorm + softplus g_proj attention gating
    (gating="per-head" on S-2.1; per-element on M.1 — runtime branches on
    the gate width, converter just quantizes whatever shape g_proj has)
  - untied embeddings, vocab 100352, 1M ctx, eos [2, 24]

Profiles (all mx.quantize affine, group_size 64 = the proven M.1 recipe;
spec: tests/test_laguna_jang_affine_policy.py):
  JANG_2L (smallest, default):
      routed gate/up/down 2/2/3 · attention (q/k/v/o/g) 8 · shared expert 6 ·
      dense FFN 6 · embed 6 · lm_head 8 · router/bias/norms fp16 passthrough
      (byte-for-byte the policy of the shipped Laguna-M.1-JANG_2L bundle)
  JANG_3L: routed 3/3/4, rest as 2L
  JANG_4M: routed 4/4/4, shared/dense 8, rest as 2L

AWQ: optional (--awq <scales.safetensors>, hy3 key convention
`model.layers.{L}.mlp.input_scale`). Laguna-M.1-JANG_2L shipped no-AWQ and
decodes coherently, so unlike hy3 this converter does NOT refuse low-bit
routed conversion without scales — it warns. Standard RMSNorm -> the
inverse fold is a plain divide on post_attention_layernorm; the router
gate is scaled by s so routing logits stay bit-identical.

Output layout (loads via jang_tools.laguna.runtime with ZERO runtime
changes — matches the Laguna-M.1-JANG_2L bundle contract exactly):
  - routed experts prestacked: model.layers.{L}.mlp.switch_mlp.{proj}.{weight,scales,biases}
  - everything else keeps source names (mlp.gate.weight router,
    mlp.experts.e_score_correction_bias, mlp.shared_expert.*, q/k norms,
    g_proj, ...) — runtime.py remaps at load
  - config.json[quantization] = per-module {bits, group_size, mode} map
    (top-level bits=8 default; runtime derives true bits from packed
    shapes as a belt-and-braces check)

Usage:
  python -m jang_tools.convert_laguna_jang \
      --src ~/models/poolside/Laguna-S-2.1 \
      --out ~/.mlxstudio/models/JANGQ-AI/Laguna-S-2.1-JANG_2L \
      --profile JANG_2L
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PROJS = ("gate_proj", "up_proj", "down_proj")


# ── policy ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LagunaJangPolicy:
    profile: str
    group_size: int
    routed_bits: dict
    attention_bits: int
    shared_expert_bits: int
    dense_ffn_bits: int
    embed_bits: int
    lm_head_bits: int


_PROFILES = {
    # Proven recipe: shipped Laguna-M.1-JANG_2L uses exactly this map
    # (routed_avg_bits 2.333, gs 64, no AWQ) and decodes coherently.
    "JANG_2L": dict(
        group_size=64,
        routed_bits={"gate_proj": 2, "up_proj": 2, "down_proj": 3},
        attention_bits=8,
        shared_expert_bits=6,
        dense_ffn_bits=6,
        embed_bits=6,
        lm_head_bits=8,
    ),
    "JANG_3L": dict(
        group_size=64,
        routed_bits={"gate_proj": 3, "up_proj": 3, "down_proj": 4},
        attention_bits=8,
        shared_expert_bits=6,
        dense_ffn_bits=6,
        embed_bits=6,
        lm_head_bits=8,
    ),
    "JANG_4M": dict(
        group_size=64,
        routed_bits={"gate_proj": 4, "up_proj": 4, "down_proj": 4},
        attention_bits=8,
        shared_expert_bits=8,
        dense_ffn_bits=8,
        embed_bits=6,
        lm_head_bits=8,
    ),
}


def profile_policy(profile: str) -> LagunaJangPolicy:
    key = profile.upper()
    if key not in _PROFILES:
        raise ValueError(
            f"unknown Laguna JANG profile {profile!r}; expected one of {sorted(_PROFILES)}"
        )
    spec = _PROFILES[key]
    return LagunaJangPolicy(
        profile=key,
        group_size=spec["group_size"],
        routed_bits=dict(spec["routed_bits"]),
        attention_bits=spec["attention_bits"],
        shared_expert_bits=spec["shared_expert_bits"],
        dense_ffn_bits=spec["dense_ffn_bits"],
        embed_bits=spec["embed_bits"],
        lm_head_bits=spec["lm_head_bits"],
    )


def build_chat_block(gen_cfg: dict) -> dict:
    """jang_config['chat'] from the vendor generation_config.json — verbatim
    passthrough, nothing invented. See main() for the enable_thinking trap
    (vendor default true via default_chat_template_kwargs; template fallback
    false)."""
    tpl_kwargs = dict(gen_cfg.get("default_chat_template_kwargs") or {})
    sampling_defaults = {
        k: gen_cfg[k]
        for k in ("temperature", "top_p", "top_k", "min_p")
        if k in gen_cfg
    }
    thinking_on = bool(tpl_kwargs.get("enable_thinking", False))
    return {
        "reasoning": {
            "supported": True,
            # RUNTIME parser names (vmlx registry), like the hy3 stamp.
            # Laguna's template is a GLM-thinking derivative → deepseek_r1
            # think-tag parsing; tools are glm47 arg_key/arg_value format.
            # The vendor's own (vLLM) parser names go in vendor_parsers.
            "parser": "deepseek_r1",
            "default_enabled": thinking_on,
            # hy3-style mode fields for engines that read modes, not bools.
            "default_mode": "think" if thinking_on else "no_think",
            "modes": ["think", "no_think"],
        },
        "tool_calling": {
            "supported": True,
            "parser": "glm47",
        },
        "vendor_parsers": {
            "reasoning": gen_cfg.get("reasoning_parser", "poolside_v1"),
            "tool": gen_cfg.get("tool_call_parser", "poolside_v1"),
        },
        "sampling_defaults": sampling_defaults,
        "template_kwargs_defaults": tpl_kwargs,
    }


def _is_passthrough(name: str) -> bool:
    n = name
    if n.endswith(".bias"):
        return True
    if "norm" in n.lower():  # input/post_attention/q/k norms, model.norm
        return True
    if "e_score_correction_bias" in n:
        return True
    # Router gate — NOT gate_proj (that's an FFN matmul). The router reads
    # the post-attn residual and picks experts; keep it exact.
    if n.endswith(".mlp.gate.weight"):
        return True
    return False


def classify_tensor(name: str, policy: LagunaJangPolicy) -> tuple[int, str]:
    """Classify a Laguna source tensor -> (bits, method in {affine, passthrough})."""
    n = name

    if _is_passthrough(n):
        return (16, "passthrough")

    if ".mlp.experts." in n and any(f".{p}.weight" in n for p in _PROJS):
        for p in _PROJS:
            if f".{p}.weight" in n:
                return (policy.routed_bits[p], "affine")

    if ".shared_expert." in n:
        return (policy.shared_expert_bits, "affine")

    if ".mlp." in n and any(f".{p}.weight" in n for p in _PROJS):
        return (policy.dense_ffn_bits, "affine")

    if "embed_tokens" in n:
        return (policy.embed_bits, "affine")

    if n == "lm_head.weight" or n.endswith(".lm_head.weight"):
        return (policy.lm_head_bits, "affine")

    # g_proj rides with attention: it gates the attention output per head
    # (or per element on M.1) and errors there scale the whole residual
    # write — same sensitivity class as o_proj.
    if "self_attn" in n and any(
        f".{p}" in n for p in ("q_proj", "k_proj", "v_proj", "o_proj", "g_proj")
    ):
        return (policy.attention_bits, "affine")

    # Any 2D matmul we missed stays safe at 8-bit affine.
    return (8, "affine")


# ── conversion ────────────────────────────────────────────────────────────

SHARD_BYTES = 4_500_000_000


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="poolside Laguna -> all-affine JANG")
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--profile", default="JANG_2L")
    ap.add_argument("--awq", type=Path, default=None,
                    help="AWQ scales safetensors (hy3 key convention: "
                         "model.layers.{L}.mlp.input_scale)")
    ap.add_argument("--group-size", type=int, default=None,
                    help="override policy group_size")
    ap.add_argument("--shard-bytes", type=int, default=SHARD_BYTES)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def _load_pt(src: Path, wm: dict, name: str) -> np.ndarray:
    """fp32 numpy load, dtype-agnostic (bf16 weights load via torch)."""
    import torch  # noqa: F401
    from safetensors import safe_open

    with safe_open(str(src / wm[name]), framework="pt") as f:
        return f.get_tensor(name).float().numpy()


class _ShardedWriter:
    def __init__(self, out_dir: Path, shard_bytes: int):
        self.out = out_dir
        self.shard_bytes = shard_bytes
        self.idx = 0
        self.bytes_in_shard = 0
        self.tensors: dict[str, np.ndarray] = {}
        self.placeholder_map: dict[str, str] = {}

    def _ph(self, i: int) -> str:
        return f"model-{i:05d}-of-99999.safetensors"

    def add(self, name: str, arr: np.ndarray) -> None:
        self.tensors[name] = arr
        self.bytes_in_shard += arr.nbytes
        if self.bytes_in_shard >= self.shard_bytes:
            self.flush()

    def flush(self) -> None:
        if not self.tensors:
            return
        from safetensors.numpy import save_file

        fn = self._ph(self.idx + 1)
        save_file(self.tensors, str(self.out / fn))
        for k in self.tensors:
            self.placeholder_map[k] = fn
        print(f"      shard {self.idx + 1}: {len(self.tensors)} tensors "
              f"{self.bytes_in_shard / 1e9:.2f}GB", flush=True)
        self.idx += 1
        self.bytes_in_shard = 0
        self.tensors = {}

    def finalize(self) -> tuple[int, int, dict[str, str]]:
        self.flush()
        n = self.idx
        wm: dict[str, str] = {}
        for i in range(1, n + 1):
            new = f"model-{i:05d}-of-{n:05d}.safetensors"
            (self.out / self._ph(i)).rename(self.out / new)
            for k, v in self.placeholder_map.items():
                if v == self._ph(i):
                    wm[k] = new
        total = sum((self.out / f).stat().st_size for f in set(wm.values()))
        return n, total, wm


def main(argv=None) -> None:
    import mlx.core as mx

    args = _parse_args(argv)
    SRC, OUT = args.src.expanduser(), args.out.expanduser()

    cfg = json.loads((SRC / "config.json").read_text())
    if cfg.get("model_type") != "laguna":
        raise SystemExit(f"source model_type={cfg.get('model_type')!r}; expected 'laguna'")

    NL = int(cfg["num_hidden_layers"])
    NE = int(cfg["num_experts"])
    H = int(cfg["hidden_size"])
    moe_inter = int(cfg["moe_intermediate_size"])

    policy = profile_policy(args.profile)
    gs = args.group_size or policy.group_size

    index_path = SRC / "model.safetensors.index.json"
    if not index_path.exists():
        raise SystemExit(f"missing {index_path}; refusing a partial download")
    wm = json.loads(index_path.read_text())["weight_map"]
    if not args.dry_run:
        # Dry-run only reads the index, so it may run mid-download.
        missing = sorted({s for s in wm.values() if not (SRC / s).exists()})
        if missing:
            raise SystemExit(
                f"source incomplete: {len(missing)} shards missing; first={missing[0]}"
            )

    # ── AWQ scales (optional for laguna: M.1-JANG_2L shipped no-AWQ coherent) ──
    min_routed = min(policy.routed_bits.values())
    awq_layer: dict[int, np.ndarray] = {}
    if args.awq is not None:
        from safetensors.numpy import load_file

        raw = load_file(str(args.awq.expanduser()))
        for li in range(NL):
            k = f"model.layers.{li}.mlp.input_scale"
            if k in raw:
                awq_layer[li] = raw[k].astype(np.float32)
        n_sparse = sum(
            1 for li in range(NL)
            if f"model.layers.{li}.mlp.gate.weight" in wm
        )
        if len(awq_layer) != n_sparse:
            raise SystemExit(
                f"AWQ scales cover {len(awq_layer)}/{n_sparse} sparse layers — "
                f"refusing a partial fold (file: {args.awq})"
            )
        print(f"  AWQ: {len(awq_layer)} layer scales loaded", flush=True)
    elif min_routed <= 3:
        print(f"  WARNING: routed experts at {policy.routed_bits} bits with NO "
              "AWQ scales. The M.1 2L bundle shipped this way and decodes "
              "coherently, but scales measurably protect low-bit arithmetic "
              "margins — consider capturing them for the campaign bundle.",
              flush=True)

    if args.dry_run:
        counts: dict[str, int] = {}
        sample: dict[str, list[str]] = {}
        for name in wm:
            bits, method = classify_tensor(name, policy)
            key = f"{method}:{bits}"
            counts[key] = counts.get(key, 0) + 1
            sample.setdefault(key, [])
            if len(sample[key]) < 5:
                sample[key].append(name)
        print(json.dumps({
            "profile": policy.profile, "group_size": gs,
            "awq_layers": len(awq_layer),
            "counts": counts, "sample": sample,
        }, indent=2))
        return

    OUT.mkdir(parents=True, exist_ok=True)
    writer = _ShardedWriter(OUT, args.shard_bytes)
    overrides: dict[str, dict] = {}
    t0 = time.time()

    def _quant(w_np: np.ndarray, bits: int):
        # Quantize from fp32 so scale/zero estimation keeps full precision —
        # matters most for the 2-bit routed experts.
        w = mx.array(w_np.astype(np.float32))
        qw, qs, qb = mx.quantize(w, group_size=gs, bits=bits)
        out = (np.array(qw), np.array(qs).astype(np.float16),
               np.array(qb).astype(np.float16))
        del w, qw, qs, qb
        mx.clear_cache()
        return out

    def emit_quant(base: str, arr: np.ndarray, bits: int) -> None:
        qw, qs, qb = _quant(arr, bits)
        writer.add(f"{base}.weight", qw)
        writer.add(f"{base}.scales", qs)
        writer.add(f"{base}.biases", qb)
        overrides[base] = {"bits": bits, "group_size": gs, "mode": "affine"}

    def emit_pass(name: str, arr: np.ndarray | None = None) -> None:
        t = (_load_pt(SRC, wm, name) if arr is None else arr).astype(np.float16)
        writer.add(name, t)

    print(f"  Laguna -> {policy.profile}  layers={NL} experts={NE} "
          f"H={H} moe_inter={moe_inter} gs={gs}", flush=True)
    print(f"  routed={policy.routed_bits} attn={policy.attention_bits} "
          f"shared={policy.shared_expert_bits} dense={policy.dense_ffn_bits} "
          f"embed={policy.embed_bits} lm_head={policy.lm_head_bits} "
          f"awq={'on' if awq_layer else 'OFF'}", flush=True)

    # ── bookends ──
    print("  bookends...", flush=True)
    emit_quant("model.embed_tokens", _load_pt(SRC, wm, "model.embed_tokens.weight"),
               policy.embed_bits)
    emit_quant("lm_head", _load_pt(SRC, wm, "lm_head.weight"), policy.lm_head_bits)
    emit_pass("model.norm.weight")

    # ── layers ──
    def convert_layer(li: int) -> None:
        tl = time.time()
        pre = f"model.layers.{li}"

        # attention: q/k/v/o + softplus gate g_proj all at attention bits;
        # per-head q/k norms pass through.
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj", "g_proj"):
            k = f"{pre}.self_attn.{proj}.weight"
            if k in wm:
                emit_quant(f"{pre}.self_attn.{proj}", _load_pt(SRC, wm, k),
                           policy.attention_bits)
        for sub in ("q_norm", "k_norm"):
            k = f"{pre}.self_attn.{sub}.weight"
            if k in wm:
                emit_pass(k)

        emit_pass(f"{pre}.input_layernorm.weight")

        is_moe = f"{pre}.mlp.gate.weight" in wm
        scale = awq_layer.get(li) if is_moe else None

        # post_attention_layernorm feeds router + shared + routed experts.
        # Standard RMSNorm -> AWQ inverse fold is a plain divide.
        pw = _load_pt(SRC, wm, f"{pre}.post_attention_layernorm.weight")
        if scale is not None:
            pw = pw / scale
        emit_pass(f"{pre}.post_attention_layernorm.weight", pw)

        if not is_moe:
            for proj in _PROJS:
                emit_quant(f"{pre}.mlp.{proj}",
                           _load_pt(SRC, wm, f"{pre}.mlp.{proj}.weight"),
                           policy.dense_ffn_bits)
            print(f"    L{li:2d} dense  {time.time() - tl:.1f}s", flush=True)
            return

        # router gate: scale by s so routing logits stay bit-identical to
        # the unfolded forward. Bias key keeps its source name — the laguna
        # runtime remaps mlp.experts.e_score_correction_bias at load.
        gate = _load_pt(SRC, wm, f"{pre}.mlp.gate.weight")  # (E, H)
        if scale is not None:
            gate = gate * scale[None, :]
        emit_pass(f"{pre}.mlp.gate.weight", gate)
        emit_pass(f"{pre}.mlp.experts.e_score_correction_bias")

        # shared expert reads the same folded stream -> scale gate/up inputs.
        for proj in _PROJS:
            w = _load_pt(SRC, wm, f"{pre}.mlp.shared_expert.{proj}.weight")
            if scale is not None and proj in ("gate_proj", "up_proj"):
                w = w * scale[None, :]
            emit_quant(f"{pre}.mlp.shared_expert.{proj}", w,
                       policy.shared_expert_bits)

        # routed experts: stack -> prestacked switch_mlp, quantized per policy.
        for proj in _PROJS:
            rows = moe_inter if proj in ("gate_proj", "up_proj") else H
            cols = H if proj in ("gate_proj", "up_proj") else moe_inter
            stack = np.empty((NE, rows, cols), dtype=np.float32)
            for e in range(NE):
                stack[e] = _load_pt(SRC, wm, f"{pre}.mlp.experts.{e}.{proj}.weight")
            if scale is not None and proj in ("gate_proj", "up_proj"):
                stack *= scale[None, None, :]
            emit_quant(f"{pre}.mlp.switch_mlp.{proj}", stack,
                       policy.routed_bits[proj])
            del stack
            gc.collect()
            mx.clear_cache()
        print(f"    L{li:2d} moe    {time.time() - tl:.1f}s", flush=True)
        gc.collect()

    for li in range(NL):
        convert_layer(li)

    print("  finalizing...", flush=True)
    nshard, total, fwm = writer.finalize()
    (OUT / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": fwm}, indent=2))

    # ── config.json ──
    out_cfg = dict(cfg)
    out_cfg.pop("quantization_config", None)
    qb = {"bits": 8, "group_size": gs, "mode": "affine"}
    qb.update(overrides)
    out_cfg["quantization"] = qb
    out_cfg["_name_or_path"] = OUT.name

    # Seed block. Canonicalized below via jang_tools.capabilities so the
    # modalities/has_* fields match what verify_directory recomputes.
    capabilities = {
        "family": "laguna",
        "modality": "text",
        "supports_tools": True,
        "supports_thinking": True,
        "cache_type": "kv",
    }

    # ── vendor generation params: pass through, never invent ──
    # S-2.1 ships temp 1.0 / top_p 1.0 / min_p 0.0 / top_k 20, parsers
    # "poolside_v1", and default_chat_template_kwargs.enable_thinking=true.
    # The chat template's OWN fallback is enable_thinking=false, so a
    # consumer that ignores default_chat_template_kwargs silently runs
    # no-think — stamp the kwargs into jang_config so engines see them.
    gen_cfg: dict = {}
    gen_p = SRC / "generation_config.json"
    if gen_p.exists():
        gen_cfg = json.loads(gen_p.read_text())
    else:
        print("  WARNING: source has no generation_config.json — chat block "
              "will carry no vendor sampling defaults", flush=True)
    chat_block = build_chat_block(gen_cfg)
    # EOS consistency: config.json vs generation_config.json. The template
    # emits 〈|EOS|〉 (id 2) as BOS and stops on [2, 24]; a mismatch here is
    # how bundles end up generating past end-of-turn.
    def _as_eos_set(v):
        if v is None:
            return set()
        return set(v) if isinstance(v, (list, tuple)) else {v}
    cfg_eos = _as_eos_set(out_cfg.get("eos_token_id"))
    gen_eos = _as_eos_set(gen_cfg.get("eos_token_id"))
    if gen_cfg and cfg_eos != gen_eos:
        raise SystemExit(
            f"eos_token_id mismatch: config.json={sorted(cfg_eos)} vs "
            f"generation_config.json={sorted(gen_eos)} — refusing to ship"
        )

    # ── jang_config.json ──
    bits_map = {
        "routed_expert": (policy.routed_bits
                          if len(set(policy.routed_bits.values())) > 1
                          else min_routed),
        "attention": policy.attention_bits,
        "shared_expert": policy.shared_expert_bits,
        "dense_mlp": policy.dense_ffn_bits,
        "embed_tokens": policy.embed_bits,
        "lm_head": policy.lm_head_bits,
        "norms_router": 16,
    }
    bit_widths_used = sorted(
        {policy.attention_bits, policy.shared_expert_bits, policy.dense_ffn_bits,
         policy.embed_bits, policy.lm_head_bits, *policy.routed_bits.values()}
    )
    jang_cfg = {
        "format": "jang",
        "format_version": "2.0",
        "profile": policy.profile,
        "cache_subtype": "kv",
        "source_model": {
            "name": SRC.name, "org": "poolside", "architecture": "laguna",
        },
        "quantization": {
            "method": "jang-affine-mixed",
            "profile": policy.profile,
            "block_size": gs,
            "group_size": gs,
            "mode": "affine",
            "bits": 8,  # conservative default for un-overridden modules
            "bits_by_role": bits_map,
            "bit_widths_used": bit_widths_used,
            "routed_avg_bits": round(sum(policy.routed_bits.values()) / 3.0, 3),
            "awq": {
                "enabled": bool(awq_layer),
                "scope": ("routed+shared gate/up + router gate fold"
                          if awq_layer else None),
            },
        },
        "architecture": {
            "type": "moe",
            "attention": "gqa+gated",
            # Weight-gated (vestigial-VL rule): the S-2.1 index carries ZERO
            # vision/audio/video tensors and config has no vision_config —
            # text-only is a verified fact, not a card claim.
            "has_vision": False,
            "has_audio": False,
            "has_video": False,
            "has_moe": True,
            "cache_type": "kv",
        },
        # Verbatim vendor values from generation_config.json — no audit has
        # been run on the quantized tail yet, so nothing is invented or
        # floored here. If a loop audit later shows the low-bit tail needs
        # floors (cf. hy3 2026-07-10), stamp them THEN, with data.
        "chat": chat_block,
    }

    # Canonicalize capabilities from the FINAL jang_config + config + the
    # written tensor index. Must run after every jang_config mutation.
    try:
        from jang_tools.capabilities import build_capabilities

        caps = build_capabilities(jang_cfg, out_cfg, OUT)
        if caps:
            jang_cfg["capabilities"] = caps
            out_cfg["capabilities"] = caps
        else:
            print("  [capabilities] WARN: family unresolved; keeping seed block",
                  flush=True)
    except Exception as exc:  # pragma: no cover
        print(f"  [capabilities] {type(exc).__name__}: {exc}", flush=True)

    (OUT / "config.json").write_text(json.dumps(out_cfg, indent=2))
    (OUT / "jang_config.json").write_text(json.dumps(jang_cfg, indent=2))

    # ── sidecars (incl. trust_remote_code modules for AutoTokenizer/Config) ──
    for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json",
               "merges.txt", "special_tokens_map.json", "added_tokens.json",
               "chat_template.jinja", "chat_template.json",
               "generation_config.json",
               "configuration_laguna.py", "modeling_laguna.py",
               "LICENSE.md"):
        if (SRC / fn).exists():
            shutil.copy2(SRC / fn, OUT / fn)

    # inline chat_template into tokenizer_config when absent there
    tok_cfg_p = OUT / "tokenizer_config.json"
    tpl_p = OUT / "chat_template.jinja"
    if tok_cfg_p.exists() and tpl_p.exists():
        tc = json.loads(tok_cfg_p.read_text())
        if not tc.get("chat_template"):
            tc["chat_template"] = tpl_p.read_text(encoding="utf-8")
            tok_cfg_p.write_text(json.dumps(tc, indent=2, ensure_ascii=False))

    # ── chat-template round-trip gate ──
    # Structural presence is not enough (feedback_structural_verification_
    # not_enough): load the tokenizer from the WRITTEN bundle and render the
    # template both with and without thinking. Catches missing sidecars,
    # a template that fails to jinja-compile, and the think-tag protocol
    # regressing (GLM-style: '<think>' when thinking, bare '</think>' when
    # not).
    print("  chat-template round-trip...", flush=True)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(OUT), trust_remote_code=True)
    msgs = [{"role": "user", "content": "ping"}]
    think = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    nothink = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    for label, rendered, tail in (("think", think, "<assistant><think>"),
                                  ("no-think", nothink, "<assistant></think>")):
        if "<user>ping</user>" not in rendered or not rendered.endswith(tail):
            raise SystemExit(
                f"chat template round-trip FAILED ({label}): got {rendered!r}"
            )
    # apply_chat_template(tokenize=True) returns list[int] on most tokenizer
    # classes but a list of tokenizers.Encoding on some fast-tokenizer paths
    # (bit the first S-2.1 build) — normalize before checking.
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    if ids and not isinstance(ids[0], int):
        first = ids[0]
        ids = list(getattr(first, "ids", first))
    bos = out_cfg.get("bos_token_id")
    if bos is not None:
        if not ids or ids[0] != bos:
            raise SystemExit(
                f"encoded prompt does not start with bos_token_id={bos} "
                f"(template leads with 〈|EOS|〉): head={ids[:4]}"
            )
        if len(ids) > 1 and ids[1] == bos:
            raise SystemExit(
                f"DOUBLE BOS: template emits 〈|EOS|〉 AND the tokenizer "
                f"prepends bos — head={ids[:4]}. Fix tokenizer_config "
                "(add_bos_token) before shipping."
            )
    print(f"    ok — think/no-think render + bos={bos} head verified", flush=True)

    print(f"\n  shards={nshard} on_disk={total / 1e9:.2f}GB "
          f"elapsed={(time.time() - t0) / 60:.1f}min")

    # Verify LAST and make failure loud.
    from jang_tools.capabilities import verify_directory

    ok, msg = verify_directory(OUT)
    print(f"  verify: ok={ok}  msg={msg}", flush=True)
    if not ok:
        raise SystemExit(f"capabilities verify FAILED for {OUT}: {msg}")

    print(f"  DONE -> {OUT}")


if __name__ == "__main__":
    main()
