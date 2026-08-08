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
import re
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
    "JANG_6M": dict(
        group_size=64,
        routed_bits={"gate_proj": 6, "up_proj": 6, "down_proj": 6},
        attention_bits=8,
        shared_expert_bits=8,
        dense_ffn_bits=8,
        embed_bits=6,
        lm_head_bits=8,
    ),
}


# ── QAT grid experts (Raptor-1.0-16B) ────────────────────────────────────
def _to_bf16(x: np.ndarray) -> np.ndarray:
    """Round fp32 -> bf16 -> fp32. The QAT forward used a bf16 scale; using
    an fp32 scale here would reproduce a slightly different function."""
    import torch

    return torch.from_numpy(np.ascontiguousarray(x)).bfloat16().float().numpy()


def qat_snap_int4_g128(w: np.ndarray, group_size: int = 128):
    """Snap onto the QAT grid per CONVERT.md Path A — an APPROXIMATION.

    Symmetric int4, groups of ``group_size`` along the INPUT (last) dim,
    ``scale = bf16(group_absmax / 7.5)``, ``q = clamp(round(w/scale), -8, 7)``.

    ⚠ Measured against poolside's packed W4A16 codes (2026-07-29, three expert
    projections across layers 1/20/39): this reproduces ~99.4% of codes, but
    ~0.6% differ by exactly ±1 because their bf16 scale rounding differs from
    ours by up to one ULP (max rel 3.8e-3..5.3e-3). It is therefore NOT the
    certified function. Use ``--qat-packed`` (Path B) for anything that ships
    under the certified numbers; this path is for analysis only.

    Returns (q_sym int8 in [-8,7], scale fp32-holding-bf16-values).
    """
    if w.shape[-1] % group_size:
        raise ValueError(
            f"last dim {w.shape[-1]} not divisible by group_size {group_size}")
    grp = w.reshape(*w.shape[:-1], w.shape[-1] // group_size, group_size)
    scale = _to_bf16(np.abs(grp).max(-1, keepdims=True) / 7.5)
    safe = np.where(scale > 0, scale, np.float32(1.0))
    q = np.clip(np.rint(grp / safe), -8, 7).astype(np.int8)
    return q.reshape(w.shape), scale.squeeze(-1).astype(np.float32)


def pack_affine4(q_affine: np.ndarray) -> np.ndarray:
    """Pack 4-bit codes (0..15) into uint32, 8 per word, little nibble first
    — MLX's affine layout, which is also the W4A16 packed layout."""
    a = q_affine.reshape(*q_affine.shape[:-1], -1, 8).astype(np.uint32)
    shifts = (np.arange(8, dtype=np.uint32) * np.uint32(4))
    return (a << shifts).sum(-1).astype(np.uint32)


def qat_expert_affine(w: np.ndarray, group_size: int = 128):
    """Snap to the QAT grid and express it losslessly as a jang affine block.

    MLX dequantizes affine as ``q*scale + bias``; with ``q = q_sym + 8`` and
    ``bias = -8*scale`` that is exactly ``q_sym * scale`` — the symmetric QAT
    grid, carried in the affine container the runtime already understands.
    """
    q_sym, scale = qat_snap_int4_g128(w, group_size)
    packed = pack_affine4((q_sym.astype(np.int16) + 8).astype(np.uint8))
    return packed, scale, (-8.0 * scale)


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


def _template_default_enable_thinking(template_text: str | None) -> bool:
    """Read Laguna's literal Jinja fallback without inventing a policy."""
    compact = "".join((template_text or "").split())
    true_marker = "enable_thinking=enable_thinking|default(true)"
    false_marker = "enable_thinking=enable_thinking|default(false)"
    if true_marker in compact:
        return True
    if false_marker in compact:
        return False
    return False


# Matches only the `enable_thinking = enable_thinking | default(<bool>)`
# assignment — not the many other `| default(...)` calls in the template.
_THINKING_DEFAULT_RE = re.compile(
    r"(enable_thinking\s*=\s*enable_thinking\s*\|\s*default\(\s*)"
    r"(true|false)"
    r"(\s*\))"
)


def _set_template_thinking_default(template_text: str, want: bool) -> tuple[str, int]:
    """Rewrite the template's literal thinking fallback to ``want``.

    Returns (new_text, n_substitutions) so the caller can refuse to ship when
    the fallback is absent or ambiguous rather than silently doing nothing.
    """
    new_text, n = _THINKING_DEFAULT_RE.subn(
        lambda m: f"{m.group(1)}{'true' if want else 'false'}{m.group(3)}",
        template_text,
    )
    return new_text, n


# Sampling values poolside documents on the model card but omits from some
# generation_config.json files. S-2.1 ships top_k=20; XS-2.1 does NOT, even
# though its own card states "The same sampling parameters were used for all
# Laguna XS 2.1 benchmarking: temperature=1.0, top_k=20 and top_p=1" and both
# usage snippets pass top_k=20. Omission there is a vendor inconsistency, not
# intent to disable top_k: with top_k unset at temperature 1.0 / top_p 1.0 a
# runtime samples the full 100352-wide distribution unfiltered, which is not
# how the model was evaluated (and it is worst on the low-bit profiles).
# These fill ONLY missing keys — an explicit vendor value always wins.
_CARD_DOCUMENTED_SAMPLING = {"top_k": 20}


def build_chat_block(
    gen_cfg: dict,
    *,
    template_text: str | None = None,
) -> dict:
    """jang_config['chat'] from the vendor generation_config.json.

    Vendor values pass through verbatim; the only additions are sampling keys
    poolside documents on the card but leaves out of generation_config
    (``_CARD_DOCUMENTED_SAMPLING``), and only where the vendor said nothing.

    An explicit ``default_chat_template_kwargs.enable_thinking`` remains
    authoritative. If it is absent, mirror the effective template's literal
    ``default(true|false)`` fallback. Poolside changed S-2.1 from false to true
    in revision e80da38, so assuming either value independently of the copied
    template makes ``jang_config`` disagree with the actual prompt.
    """
    tpl_kwargs = dict(gen_cfg.get("default_chat_template_kwargs") or {})
    sampling_defaults = {
        k: gen_cfg[k]
        for k in ("temperature", "top_p", "top_k", "min_p")
        if k in gen_cfg
    }
    for k, v in _CARD_DOCUMENTED_SAMPLING.items():
        sampling_defaults.setdefault(k, v)
    if "enable_thinking" in tpl_kwargs:
        thinking_on = bool(tpl_kwargs["enable_thinking"])
    else:
        thinking_on = _template_default_enable_thinking(template_text)
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
    ap.add_argument("--qat-packed", type=Path, default=None,
                    help="QAT W4A16 artifact dir (e.g. Raptor-1.0-16B-A3B-"
                         "W4A16). Routed experts are IMPORTED from its packed "
                         "int4 codes instead of being re-quantized, so the "
                         "bundle reproduces the certified QAT function "
                         "bit-exactly (CONVERT.md Path B).")
    ap.add_argument("--shard-bytes", type=int, default=SHARD_BYTES)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def _load_raw_pt(src: Path, wm: dict, name: str):
    """Load a tensor without forcing fp32 — packed codes must stay integral."""
    import torch  # noqa: F401
    from safetensors import safe_open

    with safe_open(str(src / wm[name]), framework="pt") as f:
        return f.get_tensor(name)


def load_packed_expert(pdir: Path, pwm: dict, stem: str):
    """Import one expert projection from the W4A16 artifact as a jang affine
    block — zero re-quantization (CONVERT.md Path B).

    poolside packs nibbles = q+8, little nibble first, which is byte-for-byte
    MLX's own affine 4-bit layout, so the packed words are reused verbatim;
    only the container changes. bias = -8*scale makes MLX's ``q*scale + bias``
    evaluate the symmetric grid ``q_sym*scale`` exactly.
    """
    packed = _load_raw_pt(pdir, pwm, f"{stem}.weight_packed").numpy()
    if packed.dtype != np.int32:
        raise SystemExit(f"{stem}.weight_packed dtype {packed.dtype}, expected int32")
    scale = _load_raw_pt(pdir, pwm, f"{stem}.weight_scale").float().numpy()
    return packed.view(np.uint32), scale.astype(np.float32)


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

    # ── Path B: certified QAT expert codes ──
    # Raptor's experts were trained through a symmetric-int4 g128 quantizer
    # (STE), and every certified number belongs to THAT function. The BF16
    # merge is the pre-snap master, so re-quantizing it — affine or even a
    # hand-rolled snap — yields a function nobody measured (a snap from bf16
    # reproduces ~99.4% of codes; the rest differ by 1 because poolside's
    # bf16 scale rounding differs by an ULP). Importing the packed codes is
    # exact by construction.
    PACKED = args.qat_packed.expanduser() if args.qat_packed else None
    pwm: dict = {}
    QAT_GS = 128
    if PACKED is not None:
        pidx = PACKED / "model.safetensors.index.json"
        if not pidx.exists():
            raise SystemExit(f"missing {pidx}")
        pwm = json.loads(pidx.read_text())["weight_map"]
        pmiss = sorted({s for s in pwm.values() if not (PACKED / s).exists()})
        if pmiss:
            raise SystemExit(
                f"QAT artifact incomplete: {len(pmiss)} shards missing; "
                f"first={pmiss[0]}")
        man = PACKED / "qat-export-manifest.json"
        if man.exists():
            m = json.loads(man.read_text())
            npack = m.get("expert_tensors_packed")
            nver = m.get("bit_exact_roundtrip_verified")
            print(f"  QAT artifact: packed={npack} bit_exact_verified={nver} "
                  f"group_size={m.get('group_size')}", flush=True)
            if nver != npack:
                raise SystemExit(
                    f"QAT artifact reports {nver}/{npack} tensors verified "
                    "bit-exact — refusing to import an unverified export")
            if m.get("group_size") != QAT_GS:
                raise SystemExit(
                    f"QAT group_size={m.get('group_size')}, converter assumes "
                    f"{QAT_GS}")
        if set(policy.routed_bits.values()) != {4}:
            raise SystemExit(
                f"--qat-packed imports 4-bit codes but profile "
                f"{policy.profile} wants routed bits {policy.routed_bits}")
        if args.awq is not None:
            raise SystemExit(
                "--qat-packed and --awq are mutually exclusive: folding AWQ "
                "scales would require re-quantizing the experts and discard "
                "the certified QAT function")
        # Experts are locked to the QAT grid's 128; non-experts keep the
        # profile's proven group size. Do NOT force the whole bundle to 128 —
        # measured 2026-07-29, coarsening the non-expert path to 128 made
        # greedy decode degenerate into repetition on open-ended prompts,
        # which poolside's certified W4A16 (BF16 non-experts) does not do.
        # laguna/runtime.py honours per-module group_size for this.
        print(f"  group_size: experts {QAT_GS} (QAT grid) / non-experts {gs}",
              flush=True)

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

        # routed experts: stack -> prestacked switch_mlp.
        for proj in _PROJS:
            base = f"{pre}.mlp.switch_mlp.{proj}"
            if PACKED is not None:
                # Path B: import the certified QAT codes verbatim.
                pk, sc = zip(*(
                    load_packed_expert(
                        PACKED, pwm, f"{pre}.mlp.experts.{e}.{proj}")
                    for e in range(NE)))
                pk = np.stack(pk)
                sc = np.stack(sc)
                writer.add(f"{base}.weight", pk)
                writer.add(f"{base}.scales", sc.astype(np.float16))
                writer.add(f"{base}.biases", (-8.0 * sc).astype(np.float16))
                overrides[base] = {"bits": 4, "group_size": QAT_GS,
                                   "mode": "affine"}
                del pk, sc
            else:
                rows = moe_inter if proj in ("gate_proj", "up_proj") else H
                cols = H if proj in ("gate_proj", "up_proj") else moe_inter
                stack = np.empty((NE, rows, cols), dtype=np.float32)
                for e in range(NE):
                    stack[e] = _load_pt(
                        SRC, wm, f"{pre}.mlp.experts.{e}.{proj}.weight")
                if scale is not None and proj in ("gate_proj", "up_proj"):
                    stack *= scale[None, None, :]
                emit_quant(base, stack, policy.routed_bits[proj])
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
    # Current Poolside revision e80da38 also defaults the template itself On;
    # older revisions defaulted it Off. Derive the fallback from the exact
    # copied template so jang_config and the prompt cannot disagree.
    gen_cfg: dict = {}
    gen_p = SRC / "generation_config.json"
    if gen_p.exists():
        gen_cfg = json.loads(gen_p.read_text())
    else:
        print("  WARNING: source has no generation_config.json — chat block "
              "will carry no vendor sampling defaults", flush=True)
    source_template_path = SRC / "chat_template.jinja"
    source_template_text = (
        source_template_path.read_text(encoding="utf-8")
        if source_template_path.exists()
        else None
    )
    chat_block = build_chat_block(
        gen_cfg,
        template_text=source_template_text,
    )
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

    # ── reconcile the template's literal thinking fallback with the vendor's
    # explicit declaration ──
    # Two surfaces declare the DEFAULT reasoning mode: the template's own
    # `enable_thinking | default(...)` fallback (what a plain transformers
    # caller gets) and generation_config.default_chat_template_kwargs
    # .enable_thinking (what vLLM and jang_config honour). Raptor-1.0-16B
    # ships them in direct contradiction — its template descends from
    # laguna_glm_thinking_v8, which still defaults false, while its
    # generation_config declares true, as the whole shipped Laguna-2.1
    # family does. Left alone the same bundle reasons or does not depending
    # on who loads it, and a reasoning-trained model served reasoning-OFF
    # measures far below its real capability (the OsaurusAgent-35B
    # post-mortem). The explicit vendor kwarg is authoritative, so align the
    # copied template to it. Only ever touches that one default() literal.
    tpl_out_p = OUT / "chat_template.jinja"
    declared = (gen_cfg.get("default_chat_template_kwargs") or {}).get(
        "enable_thinking")
    if declared is not None and tpl_out_p.exists():
        tpl_txt = tpl_out_p.read_text(encoding="utf-8")
        if _template_default_enable_thinking(tpl_txt) != bool(declared):
            new_txt, n = _set_template_thinking_default(tpl_txt, bool(declared))
            if n != 1:
                raise SystemExit(
                    "chat template disagrees with generation_config "
                    f"default_chat_template_kwargs.enable_thinking={declared}, "
                    f"but the `enable_thinking | default(...)` fallback was "
                    f"matched {n} times — cannot reconcile safely; fix the "
                    "template by hand before shipping"
                )
            tpl_out_p.write_text(new_txt, encoding="utf-8")
            print(f"  chat_template: aligned literal thinking fallback "
                  f"default({not bool(declared)}) -> default({bool(declared)}) "
                  "to match generation_config", flush=True)

    # Inline the REAL chat template into tokenizer_config. poolside ships
    # tokenizer_config.chat_template = "{% include 'chat_template.jinja' %}"
    # — a 35-char multi-file include stub, NOT a template. Only the newest
    # transformers resolve that include against the model dir; every other
    # consumer (vmlx among them) renders a broken/fallback template where
    # enable_thinking never reaches the real jinja, which is exactly the
    # "cannot enable Laguna reasoning" bug (2026-07-22). The shipped
    # Laguna-M.1 bundle inlines the full template — that is the house
    # convention. Treat an include-stub (or anything template-free) as
    # absent and inline the .jinja content.
    tok_cfg_p = OUT / "tokenizer_config.json"
    tpl_p = OUT / "chat_template.jinja"
    if tok_cfg_p.exists() and tpl_p.exists():
        tc = json.loads(tok_cfg_p.read_text())
        cur = tc.get("chat_template") or ""
        if not cur or "{% include" in cur or "{%- include" in cur:
            tc["chat_template"] = tpl_p.read_text(encoding="utf-8")
            tok_cfg_p.write_text(json.dumps(tc, indent=2, ensure_ascii=False))
            print("  chat_template: inlined full .jinja into tokenizer_config "
                  f"(was {len(cur)} chars{' include-stub' if cur else ''})",
                  flush=True)

    # ── sampling-defaults reconciliation + gate ──
    # Two files declare sampling: jang_config.chat.sampling_defaults (what
    # vmlx reads first) and the copied generation_config.json (what
    # transformers / vLLM / every non-JANG consumer reads). They MUST agree,
    # or the same bundle samples differently depending on who loads it —
    # exactly how the XS-2.1 lineup shipped without top_k=20 while its README
    # advertised it. Push card-documented keys into the copy, then hard-gate.
    gen_out_p = OUT / "generation_config.json"
    if gen_out_p.exists():
        gen_out = json.loads(gen_out_p.read_text())
        added = {}
        for k, v in chat_block["sampling_defaults"].items():
            if k not in gen_out:
                gen_out[k] = v
                added[k] = v
        if added:
            gen_out_p.write_text(json.dumps(gen_out, indent=2) + "\n")
            print(f"  generation_config: added card-documented {added}",
                  flush=True)
        disagree = {
            k: (v, gen_out.get(k))
            for k, v in chat_block["sampling_defaults"].items()
            if gen_out.get(k) != v
        }
        if disagree:
            raise SystemExit(
                "sampling defaults disagree between jang_config.chat and "
                f"generation_config.json (key: jang vs gen): {disagree} — "
                "refusing to ship a bundle that samples differently per "
                "consumer"
            )

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
    # The EFFECTIVE template (what tok will render) must be the real thing,
    # not the include stub — and it must actually read enable_thinking.
    # Environments differ on include resolution; gate on the tokenizer's
    # attribute so this fails HERE, not in a consumer.
    eff_tpl = getattr(tok, "chat_template", None) or ""
    if "{% include" in eff_tpl or "{%- include" in eff_tpl:
        raise SystemExit(
            "tokenizer.chat_template is still the include stub — the inline "
            "step failed; consumers without model-dir include resolution "
            "will render without the think protocol")
    if "enable_thinking" not in eff_tpl:
        raise SystemExit(
            "effective chat template does not read enable_thinking — "
            "reasoning cannot be toggled; refusing to ship")
    msgs = [{"role": "user", "content": "ping"}]
    think = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    nothink = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    default_render = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)
    for label, rendered, tail in (("think", think, "<assistant><think>"),
                                  ("no-think", nothink, "<assistant></think>"),
                                  (
                                      "default",
                                      default_render,
                                      (
                                          "<assistant><think>"
                                          if chat_block["reasoning"][
                                              "default_enabled"
                                          ]
                                          else "<assistant></think>"
                                      ),
                                  )):
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
