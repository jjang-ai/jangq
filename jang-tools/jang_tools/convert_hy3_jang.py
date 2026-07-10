"""Tencent Hy3 (final) -> smallest all-affine JANG converter.

Created by Jinho Jang (eric@jangq.ai) — 2026-07-09.

Target source: tencent/Hy3 (final release 2026-07-06, BF16, 597.6 GB,
model_type="hy_v3", HYV3ForCausalLM). NOT the FP8 repo — always quantize from
the highest-precision source.

Architecture (verified against config.json + safetensors index):
  - 80 decoder layers + 1 MTP layer (model.layers.80, DeepSeek-V3 style:
    eh_proj/enorm/hnorm/final_layernorm + its own full attention + MoE block)
  - GQA 64 q-heads / 8 kv-heads, head_dim 128, per-head q/k RMSNorm, standard
    RMSNorm blocks (NOT gemma (1+w) style — AWQ fold is a plain divide)
  - MoE: 192 routed experts (expert_hidden 1536), top-8, sigmoid router +
    expert_bias (DSV3 aux-free), route_norm, router_scaling_factor 2.826,
    1 shared expert; layer 0 dense (first_k_dense_replace=1, inter 13312)
  - untied embeddings, enable_lm_head_fp32=true, vocab 120832, 256K ctx

Profiles (all mx.quantize affine, spec: tests/test_hy3_jang_affine_policy.py):
  JANG_2L (smallest, default): group_size 128
      routed gate/up/down 2/2/2 · attention 8 · shared expert 8 ·
      dense FFN 8 · embed 6 · lm_head 8 · router/bias/norms fp16 passthrough
      MTP default preserve-affine8 (native speculative decode target)
  JANG_2K: routed 2/2/3, MTP dropped (NOT the campaign target)

AWQ on routed experts is DEFAULT-ON (Eric directive 2026-06-18): captured
per-input-channel scales (jang_tools.hy3.awq_capture) pre-scale routed AND
shared gate/up AND the router gate, with the inverse folded into
post_attention_layernorm (w/s for standard RMSNorm). Scaling the router gate
keeps routing logits bit-identical — a strict improvement over the M3 fold,
which left the router reading x/s. Refuses ≤3-bit routed conversion without
scales unless --no-awq is passed explicitly (which warns).

Output layout (loads via jang_tools.hy3.model sanitize with zero re-stacking):
  - routed experts prestacked: model.layers.{L}.mlp.switch_mlp.{proj}.{weight,scales,biases}
  - everything else keeps source names (router.gate, expert_bias, shared_mlp,
    q/k norms, ...) — sanitize renames at load
  - MTP layer re-namespaced to mtp.0.* (vmlx native_mtp detection keys off
    mtp.* tensor names) with FINAL param names so the runtime head loads with
    no sanitize gymnastics:
      mtp.0.{eh_proj,enorm,hnorm,final_layernorm}
      mtp.0.block.self_attn.* / input_layernorm / post_attention_layernorm
      mtp.0.block.mlp.gate.weight + gate.e_score_correction_bias (router)
      mtp.0.block.mlp.shared_experts.* + mlp.switch_mlp.* (prestacked)

Usage:
  python -m jang_tools.convert_hy3_jang \
      --src /Volumes/EricsLLMDrive/sources/Hy3 \
      --out /Volumes/EricsLLMDrive/jangq-ai/Hy3-JANG_2L \
      --awq /Volumes/EricsLLMDrive/sources/Hy3-awq-scales.safetensors \
      [--profile JANG_2L] [--mtp-policy preserve-affine8|preserve-affine4|drop]
      [--no-awq] [--dry-run]
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_PROJS = ("gate_proj", "up_proj", "down_proj")


# ── policy ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Hy3JangPolicy:
    profile: str
    group_size: int
    routed_bits: dict
    attention_bits: int
    shared_expert_bits: int
    dense_ffn_bits: int
    embed_bits: int
    lm_head_bits: int
    mtp_policy: str
    mtp_layer_start: int = 80


_PROFILES = {
    "JANG_2L": dict(
        group_size=128,
        routed_bits={"gate_proj": 2, "up_proj": 2, "down_proj": 2},
        attention_bits=8,
        shared_expert_bits=8,
        dense_ffn_bits=8,
        embed_bits=6,
        lm_head_bits=8,
        # MTP dropped by default: measured 2026-07-10, native speculative decode
        # is a ~-3% NET LOSS on the 2-bit routed backbone (verify forward is
        # per-row 2-bit MoE compute; the affine-8 head accepts at ~58% vs a ~59%
        # break-even). It only pays off with a higher-bit routed backbone. So the
        # smallest-affine profile ships without the 3.87 GB head. Pass
        # --mtp-policy preserve-affine8 to keep it (still supported).
        default_mtp_policy="drop",
    ),
    "JANG_2K": dict(
        group_size=128,
        routed_bits={"gate_proj": 2, "up_proj": 2, "down_proj": 3},
        attention_bits=8,
        shared_expert_bits=8,
        dense_ffn_bits=8,
        embed_bits=6,
        lm_head_bits=8,
        default_mtp_policy="drop",
    ),
}


def profile_policy(
    profile: str,
    mtp_policy: str | None = None,
    mtp_layer_start: int = 80,
) -> Hy3JangPolicy:
    key = profile.upper()
    if key not in _PROFILES:
        raise ValueError(
            f"unknown Hy3 JANG profile {profile!r}; expected one of {sorted(_PROFILES)}"
        )
    spec = _PROFILES[key]
    mtp = mtp_policy or spec["default_mtp_policy"]
    if mtp not in ("drop", "preserve-affine8", "preserve-affine4"):
        raise ValueError(f"unknown mtp_policy {mtp!r}")
    return Hy3JangPolicy(
        profile=key,
        group_size=spec["group_size"],
        routed_bits=dict(spec["routed_bits"]),
        attention_bits=spec["attention_bits"],
        shared_expert_bits=spec["shared_expert_bits"],
        dense_ffn_bits=spec["dense_ffn_bits"],
        embed_bits=spec["embed_bits"],
        lm_head_bits=spec["lm_head_bits"],
        mtp_policy=mtp,
        mtp_layer_start=mtp_layer_start,
    )


def _is_passthrough(name: str) -> bool:
    n = name
    if n.endswith(".bias"):
        return True
    if "norm" in n.lower():  # input/post_attention/q/k norms, enorm, hnorm, final_layernorm
        return True
    if n.endswith(".expert_bias") or "e_score_correction_bias" in n:
        return True
    if ".mlp.router.gate.weight" in n or n.endswith(".mlp.gate.weight"):
        return True
    return False


def classify_tensor(name: str, policy: Hy3JangPolicy) -> tuple[int, str]:
    """Classify a Hy3 source tensor -> (bits, method in {affine, passthrough, drop}).

    MTP-layer classification takes precedence: under preserve-affineN every
    2D weight in model.layers.{80}.* is affine-N (draft quality only affects
    speculative acceptance rate, never output correctness — the base model
    verifies), norms/biases pass through.
    """
    n = name
    is_mtp = n.startswith(f"model.layers.{policy.mtp_layer_start}.")

    if is_mtp:
        if policy.mtp_policy == "drop":
            return (0, "drop")
        mtp_bits = 8 if policy.mtp_policy.endswith("8") else 4
        if _is_passthrough(n):
            return (16, "passthrough")
        return (mtp_bits, "affine")

    if _is_passthrough(n):
        return (16, "passthrough")

    if ".mlp.experts." in n and any(f".{p}.weight" in n for p in _PROJS):
        for p in _PROJS:
            if f".{p}.weight" in n:
                return (policy.routed_bits[p], "affine")

    if ".shared_mlp." in n or ".shared_experts." in n:
        return (policy.shared_expert_bits, "affine")

    if ".mlp." in n and any(f".{p}.weight" in n for p in _PROJS):
        return (policy.dense_ffn_bits, "affine")

    if "embed_tokens" in n:
        return (policy.embed_bits, "affine")

    if n == "lm_head.weight" or n.endswith(".lm_head.weight"):
        return (policy.lm_head_bits, "affine")

    if "self_attn" in n and any(
        f".{p}" in n for p in ("q_proj", "k_proj", "v_proj", "o_proj")
    ):
        return (policy.attention_bits, "affine")

    # Any 2D matmul we missed stays safe at 8-bit affine.
    return (8, "affine")


# ── conversion ────────────────────────────────────────────────────────────

SHARD_BYTES = 4_500_000_000


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Tencent Hy3 -> all-affine JANG")
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--profile", default="JANG_2L")
    ap.add_argument("--mtp-policy", default=None,
                    choices=["drop", "preserve-affine8", "preserve-affine4"])
    ap.add_argument("--awq", type=Path, default=None,
                    help="Hy3 AWQ scales safetensors (jang_tools.hy3.awq_capture)")
    ap.add_argument("--no-awq", action="store_true",
                    help="EXPLICITLY ship unprotected low-bit routed experts (warns)")
    ap.add_argument("--group-size", type=int, default=None,
                    help="override policy group_size (e.g. 64 if g128 fails coherence gate)")
    ap.add_argument("--shard-bytes", type=int, default=SHARD_BYTES)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def _load_pt(src: Path, wm: dict, name: str) -> np.ndarray:
    """fp32 numpy load, dtype-agnostic (bf16 weights, fp32 router gates both OK)."""
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
    if cfg.get("model_type") != "hy_v3":
        raise SystemExit(f"source model_type={cfg.get('model_type')!r}; expected 'hy_v3'")

    NL = int(cfg["num_hidden_layers"])
    NE = int(cfg["num_experts"])
    H = int(cfg["hidden_size"])
    moe_inter = int(cfg.get("moe_intermediate_size", cfg.get("expert_hidden_dim", 1536)))
    n_mtp = int(cfg.get("num_nextn_predict_layers", 0))
    first_dense = int(cfg.get("first_k_dense_replace", 0))

    policy = profile_policy(args.profile, args.mtp_policy, mtp_layer_start=NL)
    gs = args.group_size or policy.group_size

    index_path = SRC / "model.safetensors.index.json"
    if not index_path.exists():
        raise SystemExit(f"missing {index_path}; refusing a partial Hy3 download")
    wm = json.loads(index_path.read_text())["weight_map"]
    missing = sorted({s for s in wm.values() if not (SRC / s).exists()})
    if missing:
        raise SystemExit(
            f"source incomplete: {len(missing)} shards missing; first={missing[0]}"
        )

    # ── AWQ scales (default-on for low-bit routed experts) ──
    min_routed = min(policy.routed_bits.values())
    awq_layer: dict[int, np.ndarray] = {}
    if args.no_awq:
        print("  WARNING: --no-awq — routed experts at "
              f"{policy.routed_bits} bits ship UNPROTECTED. Low-bit MoE without "
              "AWQ has measurably flattened arithmetic margins (M3 REAP proof). "
              "Stamping quantization.awq.enabled=false.", flush=True)
    else:
        if args.awq is None and min_routed <= 3:
            raise SystemExit(
                "AWQ scales are REQUIRED for <=3-bit routed experts "
                "(default-on directive 2026-06-18). Run "
                "`python -m jang_tools.hy3.awq_capture` on the bf16 source and "
                "pass --awq <scales.safetensors>, or pass --no-awq explicitly."
            )
        if args.awq is not None:
            from safetensors.numpy import load_file

            raw = load_file(str(args.awq.expanduser()))
            for li in range(first_dense, NL):
                k = f"model.layers.{li}.mlp.input_scale"
                if k in raw:
                    awq_layer[li] = raw[k].astype(np.float32)
            if len(awq_layer) != NL - first_dense:
                raise SystemExit(
                    f"AWQ scales cover {len(awq_layer)}/{NL - first_dense} sparse "
                    f"layers — refusing a partial fold (file: {args.awq})"
                )
            print(f"  AWQ: {len(awq_layer)} layer scales loaded", flush=True)

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
            "mtp_policy": policy.mtp_policy,
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

    print(f"  Hy3 -> {policy.profile}  layers={NL}+{n_mtp}MTP experts={NE} "
          f"H={H} moe_inter={moe_inter} gs={gs}", flush=True)
    print(f"  routed={policy.routed_bits} attn={policy.attention_bits} "
          f"shared={policy.shared_expert_bits} dense={policy.dense_ffn_bits} "
          f"embed={policy.embed_bits} lm_head={policy.lm_head_bits} "
          f"mtp={policy.mtp_policy} awq={'on' if awq_layer else 'OFF'}", flush=True)

    # ── bookends ──
    print("  bookends...", flush=True)
    emit_quant("model.embed_tokens", _load_pt(SRC, wm, "model.embed_tokens.weight"),
               policy.embed_bits)
    emit_quant("lm_head", _load_pt(SRC, wm, "lm_head.weight"), policy.lm_head_bits)
    emit_pass("model.norm.weight")

    # ── layers (0..NL-1 base, NL.. MTP re-namespaced to mtp.{i}.*) ──
    def convert_layer(li: int, is_mtp: bool) -> None:
        tl = time.time()
        pre = f"model.layers.{li}"          # source namespace
        mtp_bits = 0
        if is_mtp:
            if policy.mtp_policy == "drop":
                print(f"    L{li} MTP dropped", flush=True)
                return
            mtp_bits = 8 if policy.mtp_policy.endswith("8") else 4
            mtp_idx = li - NL
            head = f"mtp.{mtp_idx}"          # output namespace for the head
            blk = f"{head}.block"

        def bits_for(kind: str) -> int:
            if is_mtp:
                return mtp_bits
            return {
                "attn": policy.attention_bits,
                "shared": policy.shared_expert_bits,
                "dense": policy.dense_ffn_bits,
                "routed_gate_proj": policy.routed_bits["gate_proj"],
                "routed_up_proj": policy.routed_bits["up_proj"],
                "routed_down_proj": policy.routed_bits["down_proj"],
            }[kind]

        def out_name(src_suffix: str, final_suffix: str | None = None) -> str:
            """Map a layer-relative source suffix to the output tensor base."""
            if not is_mtp:
                return f"{pre}.{src_suffix}"
            return f"{blk}.{final_suffix or src_suffix}"

        # MTP head extras (eh_proj + enorm/hnorm/final_layernorm) live at the
        # head root, not inside the block.
        if is_mtp:
            k = f"{pre}.eh_proj.weight"
            if k in wm:
                emit_quant(f"{head}.eh_proj", _load_pt(SRC, wm, k), mtp_bits)
            for extra in ("enorm", "hnorm", "final_layernorm"):
                k = f"{pre}.{extra}.weight"
                if k in wm:
                    emit_pass(f"{head}.{extra}.weight", _load_pt(SRC, wm, k))

        # attention
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            emit_quant(out_name(f"self_attn.{proj}"),
                       _load_pt(SRC, wm, f"{pre}.self_attn.{proj}.weight"),
                       bits_for("attn"))
        for sub in ("q_norm", "k_norm"):
            k = f"{pre}.self_attn.{sub}.weight"
            if k in wm:
                emit_pass(out_name(f"self_attn.{sub}.weight"), _load_pt(SRC, wm, k))

        emit_pass(out_name("input_layernorm.weight"),
                  _load_pt(SRC, wm, f"{pre}.input_layernorm.weight"))

        is_moe = f"{pre}.mlp.router.gate.weight" in wm
        scale = awq_layer.get(li) if (is_moe and not is_mtp) else None

        # post_attention_layernorm feeds router + shared + routed experts.
        # Standard RMSNorm -> AWQ inverse fold is a plain divide.
        pw = _load_pt(SRC, wm, f"{pre}.post_attention_layernorm.weight")
        if scale is not None:
            pw = pw / scale
        emit_pass(out_name("post_attention_layernorm.weight"),
                  pw.astype(np.float16))

        if not is_moe:
            for proj in _PROJS:
                emit_quant(out_name(f"mlp.{proj}"),
                           _load_pt(SRC, wm, f"{pre}.mlp.{proj}.weight"),
                           bits_for("dense"))
            print(f"    L{li:2d} dense  {time.time() - tl:.1f}s", flush=True)
            return

        # router gate: scale by s to keep routing logits bit-identical to the
        # unfolded forward (the M3 fold skipped this; Hy3 does it properly).
        # Base layers keep source names (hy3 sanitize renames at load); the MTP
        # block gets FINAL names (gate.weight / gate.e_score_correction_bias).
        gate = _load_pt(SRC, wm, f"{pre}.mlp.router.gate.weight")  # (E, H)
        if scale is not None:
            gate = gate * scale[None, :]
        writer.add(out_name("mlp.router.gate.weight", "mlp.gate.weight"),
                   gate.astype(np.float16))
        bk = f"{pre}.mlp.expert_bias"
        if bk in wm:
            emit_pass(out_name("mlp.expert_bias",
                               "mlp.gate.e_score_correction_bias"),
                      _load_pt(SRC, wm, bk))

        # shared expert reads the same folded stream -> scale gate/up inputs.
        for proj in _PROJS:
            k = f"{pre}.mlp.shared_mlp.{proj}.weight"
            if k in wm:
                w = _load_pt(SRC, wm, k)
                if scale is not None and proj in ("gate_proj", "up_proj"):
                    w = w * scale[None, :]
                emit_quant(out_name(f"mlp.shared_mlp.{proj}",
                                    f"mlp.shared_experts.{proj}"),
                           w, bits_for("shared"))

        # routed experts: stack -> prestacked switch_mlp, quantized per policy.
        for proj in _PROJS:
            rows = moe_inter if proj in ("gate_proj", "up_proj") else H
            cols = H if proj in ("gate_proj", "up_proj") else moe_inter
            stack = np.empty((NE, rows, cols), dtype=np.float32)
            for e in range(NE):
                stack[e] = _load_pt(SRC, wm, f"{pre}.mlp.experts.{e}.{proj}.weight")
            if scale is not None and proj in ("gate_proj", "up_proj"):
                stack *= scale[None, None, :]
            emit_quant(out_name(f"mlp.switch_mlp.{proj}"), stack,
                       bits_for(f"routed_{proj}"))
            del stack
            gc.collect()
            mx.clear_cache()
        tag = "MTP-moe" if is_mtp else "moe"
        print(f"    L{li:2d} {tag} {time.time() - tl:.1f}s", flush=True)
        gc.collect()

    for li in range(NL):
        convert_layer(li, is_mtp=False)
    for li in range(NL, NL + n_mtp):
        convert_layer(li, is_mtp=True)

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

    mtp_preserved = policy.mtp_policy != "drop" and n_mtp > 0
    if not mtp_preserved:
        # The MTP tensors are not written, so the config must not advertise an
        # MTP head — otherwise a loader that keys head construction on
        # num_nextn_predict_layers>0 would try to build a head with no weights.
        out_cfg["num_nextn_predict_layers"] = 0
        if isinstance(out_cfg.get("text_config"), dict):
            out_cfg["text_config"]["num_nextn_predict_layers"] = 0
    runtime_block = {
        "bundle_has_mtp": mtp_preserved,
        "mtp_layers": n_mtp if mtp_preserved else 0,
        "mtp_mode": ("preserved_native_candidate" if mtp_preserved
                     else "dropped_for_smallest_affine"),
        "mtp_num_speculative_tokens": 2 if mtp_preserved else 0,
        "mtp_status": (
            "MTP layer preserved at affine-%d for native speculative decode "
            "(official serving config drafts 2 tokens); base model verifies, "
            "draft quality only affects acceptance rate."
            % (8 if policy.mtp_policy.endswith("8") else 4)
        ) if mtp_preserved else "MTP dropped (smallest-affine profile)",
    }
    # Seed block. Canonicalized below via jang_tools.capabilities so the
    # modalities/has_* fields match what verify_directory recomputes — a
    # hand-written block fails verify ("re-stamp at the very end").
    capabilities = {
        "reasoning_parser": "qwen3",
        "tool_parser": "hunyuan",
        "think_in_template": False,
        "supports_tools": True,
        "supports_thinking": True,
        "family": "hy_v3",
        "modality": "text",
        "cache_type": "kv",
    }
    out_cfg["runtime"] = runtime_block

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
    if mtp_preserved:
        bits_map["mtp"] = 8 if policy.mtp_policy.endswith("8") else 4
    # `quantization.bits` MUST be a scalar: vmlx's `_jang_default_bits` does
    # `int(quant["bits"])` for the default width of any module without an
    # explicit config.json override, and it is evaluated eagerly inside a
    # `setdefault(...)` call, so a dict here raises TypeError at load even
    # when config.json carries correct per-tensor overrides. The per-role map
    # lives under `bits_by_role`. Do NOT name it `mxtq_bits` — that key routes
    # the loader down the TurboQuant hydrate path, and this is an affine bundle.
    bit_widths_used = sorted(
        {policy.attention_bits, policy.shared_expert_bits, policy.dense_ffn_bits,
         policy.embed_bits, policy.lm_head_bits, *policy.routed_bits.values()}
        | ({8 if policy.mtp_policy.endswith("8") else 4} if mtp_preserved else set())
    )
    jang_cfg = {
        "format": "jang",
        "format_version": "2.0",
        "profile": policy.profile,
        "cache_subtype": "kv",
        "source_model": {"name": "Hy3", "org": "tencent", "architecture": "hy_v3"},
        "quantization": {
            "method": "jang-affine-mixed",
            "profile": policy.profile,
            "block_size": gs,
            "group_size": gs,
            "mode": "affine",
            "bits": 8,  # conservative default for un-overridden modules
            "bits_by_role": bits_map,
            "bit_widths_used": bit_widths_used,
            "routed_avg_bits": sum(policy.routed_bits.values()) / 3.0,
            "awq": {
                "enabled": bool(awq_layer),
                # Don't describe a fold that didn't happen.
                "scope": ("routed+shared gate/up + router gate fold"
                          if awq_layer else None),
            },
        },
        "architecture": {
            "type": "moe", "attention": "gqa+qk_norm",
            "has_vision": False, "has_moe": True, "cache_type": "kv",
        },
        "runtime": runtime_block,
        "bundle_has_mtp": mtp_preserved,
        "mtp_layers": n_mtp if mtp_preserved else 0,
        "capabilities": capabilities,
        "chat": {
            "reasoning": {
                "supported": True,
                "parser": "qwen3",
                "default_mode": "no_think",
                "modes": ["no_think", "low", "high"],
            },
            "tool_calling": {"supported": True, "parser": "hunyuan"},
            # Final-release official recommendation. The May PREVIEW JANG_2L
            # looped at temp 0.9 past ~1.5-2.2K tokens and was stamped greedy;
            # the final model is a new post-train — gate the loop explicitly
            # before overriding these.
            "sampling_defaults": {"temperature": 0.9, "top_p": 1.0, "top_k": -1},
        },
    }

    # Canonicalize capabilities from the FINAL jang_config + config + the
    # written tensor index (build_capabilities inspects tensor names to decide
    # modalities). Must run after every jang_config mutation, before writing.
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

    # ── sidecars ──
    for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json",
               "merges.txt", "special_tokens_map.json", "added_tokens.json",
               "chat_template.jinja", "chat_template.json",
               "generation_config.json"):
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

    print(f"\n  shards={nshard} on_disk={total / 1e9:.2f}GB "
          f"elapsed={(time.time() - t0) / 60:.1f}min")

    # Verify LAST and make failure loud: a bundle whose capabilities block
    # doesn't round-trip is a bundle the engine may mis-route at load.
    from jang_tools.capabilities import verify_directory

    ok, msg = verify_directory(OUT)
    print(f"  verify: ok={ok}  msg={msg}", flush=True)
    if not ok:
        raise SystemExit(f"capabilities verify FAILED for {OUT}: {msg}")

    print(f"  DONE -> {OUT}")


if __name__ == "__main__":
    main()
