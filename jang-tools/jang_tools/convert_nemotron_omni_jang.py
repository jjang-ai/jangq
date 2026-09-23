"""Nemotron-3-Nano-Omni-30B-A3B → JANG v2 affine (calibrated), towers intact.

Created by Jinho Jang (eric@osaurus.ai) — 2026-08-22.

Unlike `convert_nemotron_mxfp4` / `convert_nemotron_jangtq`, which ship a
TEXT-ONLY bundle, this keeps the RADIO vision tower, the **Parakeet** sound
encoder and both projectors verbatim in fp16 so image / video / audio actually
work. The towers are only 2.68 GiB of a ~20 GiB bundle and are never quantized —
matching the contract of the shipped `OsaurusAI/…-MXFP4` bundle, where all 164
`.scales` tensors belong to the LLM and none to a tower.

Bundle layout (verified against the shipped MXFP4 omni bundle):

    config.json        flattened LLM config (model_type nemotron_h) + quantization
    config_omni.json   the original wrapper config (vision/sound dims, llm_config)
    jang_config.json   JANG v2 metadata, capabilities, multimodal_components
    weights            language_model.* → backbone.* / lm_head; towers verbatim

Flattening `llm_config` into `config.json` is not cosmetic — it is what makes
bit allocation correct. `allocate.py` and `convert.py` only ever look for nested
keys under `text_config`, so against the RAW wrapper config
`num_key_value_heads`, `n_routed_experts` and `mlp_hidden_act` all read as
absent: attention would classify as `NONE`, `num_experts` as 0, and
`gateless_relu2` as False — which silently disables `RELU2_ASYMMETRY_FLOORS`,
the one rule that keeps this architecture's experts intact.

Why the floors matter here: experts are `fc2(relu2(fc1(x)))` with **no gate**,
so `fc1` feeds a squaring nonlinearity and is itself the error amplifier — the
role `gate_proj` plays in SwiGLU. `MLP_ASYMMETRY_FLOORS` explicitly leaves
`up_proj` unprotected ("2-bit OK when gate is protected"), which is vacuous with
no gate, and it only fires at >=256 experts (this model has 128).

Calibration (all optional, `--calib` supplies them):
  * **imatrix** — per-channel E[x²] weights the affine fit toward channels that
    actually carry signal.
  * **AWQ** — per-channel scale `s`, folded into the PRODUCING norm as
    `w_norm ← w_norm / s` with each consumer taking `W ← W * s`.
    `NemotronHRMSNorm` is plain (`self.weight * hidden_states`, **no `+1`
    offset**), so this is the correct fold — the Gemma `(stored+1)/s − 1` form
    would be wrong here. The fp16 router `gate` is compensated exactly by the
    same `W ← W * s`; forgetting it would silently corrupt routing.
  * **GPTQ** — error-compensated rounding on the routed experts, using the full
    `H = XᵀX`. Gated on a held-out A/B (`--gptq` alone does not ship it).

AWQ is applied only where a plain RMSNorm produces the input:
`layers.N.norm` → {`switch_mlp.fc1`, `shared_experts.up_proj`, `gate`, q/k/v,
mamba `in_proj`}. It is NOT applied to `fc2` (its input is the relu2 output —
no norm to fold into; imatrix + GPTQ cover it), nor to mamba `out_proj`
(`MambaRMSNormGated` gates the norm, so the fold is not a plain division), nor
to `lm_head`.

    PYTHONPATH=~/jang/jang-tools \
    python -m jang_tools.convert_nemotron_omni_jang \
        <src_bf16_dir> <out_dir> --profile JANG_4M \
        [--calib <calib_dir>] [--gptq] [--group-size 64]
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
from safetensors.numpy import save_file

from .allocate import allocate_bits_profile_compact

TOWER_PREFIXES = ("vision_model.", "sound_encoder.", "mlp1.", "sound_projection.")

EXPERT_KEY_RE = re.compile(
    r"^(?:language_model\.)?backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\."
    r"(up_proj|down_proj)\.weight$"
)

# mlx_lm SwitchMLP submodules — NOT up_proj/down_proj. Emitting the source
# names produces a bundle that loads as nothing.
EXPERT_SUBMODULE = {"up_proj": "fc1", "down_proj": "fc2"}

SHARD_LIMIT = 4 * 1024**3


def classify(name: str) -> str:
    """One of: tower | expert | passthrough | affine."""
    if name.startswith(TOWER_PREFIXES):
        return "tower"
    if name.startswith("language_model.mtp.") or name.startswith("mtp."):
        return "drop"          # Nemotron-Omni ships no usable MTP head
    if EXPERT_KEY_RE.match(name):
        return "expert"
    n = name
    if n.endswith(".norm.weight") or "norm_f.weight" in n or "mixer.norm.weight" in n:
        return "passthrough"
    if n.endswith(".A_log") or n.endswith(".D") or n.endswith(".dt_bias"):
        return "passthrough"
    if "conv1d.weight" in n or "conv1d.bias" in n:
        return "passthrough"
    if n.endswith(".mixer.gate.weight") or n.endswith(".mixer.gate.e_score_correction_bias"):
        return "passthrough"
    if n.endswith(".bias"):
        return "passthrough"
    return "affine"


def strip_prefix(name: str) -> str:
    return name[len("language_model."):] if name.startswith("language_model.") else name


# --------------------------------------------------------------------------
# AWQ
# --------------------------------------------------------------------------
def awq_scale(W_list: list[np.ndarray], x_sq: np.ndarray, bits: int,
              group_size: int, alphas=np.arange(0.0, 1.01, 0.05)) -> np.ndarray:
    """Per-input-channel scale minimising affine round-trip error.

    `W_list` is every consumer of the same producing norm — the scale has to
    serve all of them at once, so the objective sums their errors.
    """
    x_mag = np.sqrt(np.maximum(x_sq, 0)).astype(np.float64)
    x_mag = np.maximum(x_mag, 1e-8)
    x_mag = x_mag / x_mag.mean()

    best_s, best_err = None, np.inf
    for a in alphas:
        s = np.power(x_mag, a)
        s = np.clip(s / np.sqrt(s.max() * s.min()), 1e-4, 1e4)
        err = 0.0
        for W in W_list:
            Ws = W.astype(np.float32) * s.astype(np.float32)[None, :]
            q = mx.quantize(mx.array(Ws), group_size=group_size, bits=bits)
            deq = np.asarray(mx.dequantize(*q, group_size=group_size, bits=bits))
            # error measured in the ORIGINAL space, weighted by activation mass
            d = (deq / s[None, :].astype(np.float32) - W.astype(np.float32))
            err += float(np.sum((d * d) * x_sq[None, :].astype(np.float32)))
        if err < best_err:
            best_err, best_s = err, s
    return best_s.astype(np.float32)


def quantize_affine(W: np.ndarray, bits: int, group_size: int,
                    imatrix: np.ndarray | None = None):
    """Affine quantize -> (packed, scales, biases) as numpy.

    Handles both 2D `(out, in)` and stacked-expert 3D `(E, out, in)`.

    Two shape traps in `quantize_imatrix_affine_numpy`, both silent if ignored:
    it returns a **4-tuple** `(packed, scales, biases, err)` — unpacking it like
    `mx.quantize`'s 3-tuple drops the biases — and it validates a **2D** matrix,
    so stacked experts must be folded to `(E*out, in)` first. The importance
    vector is over the input dim, which every expert shares, so the fold is
    exact rather than an approximation.
    """
    Wf = np.ascontiguousarray(W, dtype=np.float32)
    if imatrix is None:
        qw, qs, qb = mx.quantize(mx.array(Wf), group_size=group_size, bits=bits)
        return np.asarray(qw), np.asarray(qs), np.asarray(qb)

    from .affine import quantize_imatrix_affine_numpy
    lead = Wf.shape[:-2]                       # () or (E,)
    flat = Wf.reshape(-1, Wf.shape[-1])        # (rows, in)
    packed, scales, biases, _err = quantize_imatrix_affine_numpy(
        flat, np.asarray(imatrix, dtype=np.float32),
        bits=bits, group_size=group_size)
    if lead:
        packed = packed.reshape(*lead, Wf.shape[-2], packed.shape[-1])
        scales = scales.reshape(*lead, Wf.shape[-2], scales.shape[-1])
        biases = biases.reshape(*lead, Wf.shape[-2], biases.shape[-1])
    return packed, scales, biases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--profile", default="JANG_4M")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--calib", type=Path, default=None)
    ap.add_argument("--gptq", action="store_true")
    a = ap.parse_args()
    SRC, OUT, GS = a.src, a.out, a.group_size
    OUT.mkdir(parents=True, exist_ok=True)

    full_cfg = json.loads((SRC / "config.json").read_text())
    llm_cfg = full_cfg.get("llm_config")
    if not llm_cfg:
        raise SystemExit(f"{SRC}/config.json has no llm_config — wrong source?")

    n_experts = llm_cfg["n_routed_experts"]
    pattern = llm_cfg["hybrid_override_pattern"]
    print(f"  profile={a.profile} group_size={GS} experts={n_experts}")

    # ---- calibration ----------------------------------------------------
    diag, hess = {}, {}
    if a.calib:
        d = np.load(a.calib / "diag_second_moment.npz")
        diag = {k: d[k] for k in d.files}
        hp = a.calib / "expert_hessians.npz"
        if hp.exists():
            h = np.load(hp)
            hess = {k: h[k] for k in h.files}
        meta = json.loads((a.calib / "meta.json").read_text())
        print(f"  calib: {len(diag)} diag modules, {len(hess)} Hessians, "
              f"{meta['tokens']:,} tokens, min rank {meta.get('min_rank_ratio')}x")

    # ---- enumerate ------------------------------------------------------
    shards = sorted(SRC.glob("model-*.safetensors"))
    index = {}
    for sf in shards:
        with safe_open(str(sf), framework="numpy") as f:
            for k in f.keys():
                index[k] = sf

    # ---- bit allocation on SOURCE names (floors match up_proj/down_proj) --
    alloc_names = [strip_prefix(k) for k in index
                   if classify(k) in ("affine", "expert")]
    bits_map = allocate_bits_profile_compact(
        [(n, 0) for n in alloc_names], a.profile,
        num_experts=n_experts, has_shared_mlp=True, gateless_relu2=True)
    ex_up = next(v for k, v in bits_map.items() if ".experts.0.up_proj" in k)
    ex_dn = next(v for k, v in bits_map.items() if ".experts.0.down_proj" in k)
    print(f"  gate-less relu2 floors active: expert up_proj={ex_up}b down_proj={ex_dn}b")

    # ---- SSM / embedding floor, from NVIDIA's own mixed-precision reference --
    # `nvidia/…-NVFP4` (modelopt, hf_quant_config.json) sends ONLY the 5888
    # routed-expert matrices to NVFP4 and deliberately keeps the selective-scan
    # projections at FP8: 23x `mixer.in_proj` + 23x `mixer.out_proj` -> FP8,
    # shared experts -> FP8, attention q/k/v and embeddings/lm_head left bf16,
    # and both towers untouched. Our profile allocator would put `in_proj` /
    # `out_proj` at the COMPRESS width (4 bits at JANG_4M), i.e. below what the
    # model's author considered safe for the SSM input/output path.
    #
    # in_proj + out_proj + embeddings is 1.24 B params (3.9 %), so the promotion
    # costs ~0.6 GiB at JANG_4M. Cheap insurance on the one path that carries
    # recurrent state across the whole sequence, where an error does not wash
    # out the way a single MoE token does.
    SSM_EMBED_FLOOR = 8
    n_promoted = 0
    for k in list(bits_map):
        if (".mixer.in_proj" in k or ".mixer.out_proj" in k
                or k.startswith("backbone.embeddings")):
            if bits_map[k] < SSM_EMBED_FLOOR:
                bits_map[k] = SSM_EMBED_FLOOR
                n_promoted += 1
    if n_promoted:
        print(f"  SSM/embedding floor: promoted {n_promoted} tensors to "
              f"{SSM_EMBED_FLOOR}b (NVIDIA NVFP4 reference keeps these at FP8/bf16)")

    tensors: dict[str, np.ndarray] = {}
    written, shard_idx = [], 0
    nbytes = 0

    def flush(force=False):
        nonlocal tensors, shard_idx, nbytes
        if not tensors or (not force and nbytes < SHARD_LIMIT):
            return
        shard_idx += 1
        p = OUT / f"model-{shard_idx:05d}.safetensors"
        # mx.save_safetensors, not safetensors.numpy: the tower tensors keep
        # their source dtype, and numpy has no bfloat16.
        mx.save_safetensors(str(p), tensors, metadata={"format": "pt"})
        written.append((p.name, list(tensors.keys())))
        tensors, nbytes = {}, 0
        gc.collect()

    def add(name, arr):
        nonlocal nbytes
        a = arr if isinstance(arr, mx.array) else mx.array(arr)
        tensors[name] = a
        nbytes += a.nbytes
        flush()

    def emit_quant(base, W, bits, imatrix=None):
        # A stale or mis-keyed importance vector is the quiet way to quantize
        # against the wrong activation. Name the tensor rather than letting
        # affine.py raise a shapes-only error with no context.
        if imatrix is not None and imatrix.shape != (W.shape[-1],):
            raise SystemExit(
                f"imatrix shape {imatrix.shape} != input dim {(W.shape[-1],)} "
                f"for {base} — calibration does not match this checkpoint")
        qw, qs, qb = quantize_affine(W, bits, GS, imatrix)
        # A constant weight group yields scale==0, and the imatrix fit then
        # divides by it. That is benign — the bias carries the constant, so the
        # group reconstructs exactly — but it goes through a nan/inf cast, so
        # assert the stored tensors are finite rather than trusting it. A
        # non-finite scale/bias would poison the whole group at load time.
        for tag, arr in (("scales", qs), ("biases", qb)):
            if not np.isfinite(np.asarray(arr, dtype=np.float32)).all():
                raise SystemExit(f"non-finite {tag} for {base} — refusing to write")
        add(f"{base}.weight", qw)
        add(f"{base}.scales", qs)
        add(f"{base}.biases", qb)

    _shard_cache: dict[Path, dict] = {}

    def load(name):
        """Read a source tensor as numpy.

        Goes through `mx.load` rather than safetensors' numpy backend: this
        checkpoint is **bfloat16**, which numpy has no dtype for
        (`TypeError: data type 'bfloat16' not understood`). MLX reads bf16
        natively and mmaps the shard, so repeated reads of the same shard --
        128 experts per layer -- do not re-read from disk.
        """
        sf = index[name]
        shard = _shard_cache.get(sf)
        if shard is None:
            if len(_shard_cache) > 2:        # keep the working set bounded
                _shard_cache.clear()
                gc.collect()
            shard = mx.load(str(sf))
            _shard_cache[sf] = shard
        return np.asarray(shard[name].astype(mx.float32))

    def load_raw(name):
        """Read a source tensor preserving its dtype (bf16 stays bf16)."""
        sf = index[name]
        shard = _shard_cache.get(sf)
        if shard is None:
            if len(_shard_cache) > 2:
                _shard_cache.clear()
                gc.collect()
            shard = mx.load(str(sf))
            _shard_cache[sf] = shard
        return shard[name]

    # ---- AWQ scales per producing norm ----------------------------------
    # layers.N.norm feeds: switch_mlp.fc1, shared_experts.up_proj, gate,
    # q/k/v_proj, mamba in_proj. One scale must serve them all.
    awq: dict[int, np.ndarray] = {}
    if diag:
        t0 = time.time()
        for li, btype in enumerate(pattern):
            key_in = f"H.{li}.in"
            if btype == "E":
                xsq = None
                if key_in in hess:
                    xsq = np.diag(hess[key_in]).copy()
                elif f"backbone.layers.{li}.mixer.shared_experts.up_proj" in diag:
                    xsq = diag[f"backbone.layers.{li}.mixer.shared_experts.up_proj"]
                if xsq is None:
                    continue
                # Sample several experts rather than expert 0 alone: one expert
                # is a noisy stand-in for a 128-way layer, and the scale has to
                # serve all of them (they share this norm). Spread the sample
                # across the expert index range.
                idxs = [0, n_experts // 4, n_experts // 2, 3 * n_experts // 4]
                consumers = [
                    load(f"language_model.backbone.layers.{li}.mixer.experts.{e}"
                         f".up_proj.weight").astype(np.float32)
                    for e in dict.fromkeys(idxs)
                ]
                awq[li] = awq_scale(consumers, xsq, ex_up, GS)
            elif btype == "*":
                nm = f"backbone.layers.{li}.mixer.q_proj"
                if nm not in diag:
                    continue
                w = load(f"language_model.{nm}.weight").astype(np.float32)
                awq[li] = awq_scale([w], diag[nm], bits_map.get(f"{nm}.weight", 8), GS)
            elif btype == "M":
                nm = f"backbone.layers.{li}.mixer.in_proj"
                if nm not in diag:
                    continue
                w = load(f"language_model.{nm}.weight").astype(np.float32)
                awq[li] = awq_scale([w], diag[nm], bits_map.get(f"{nm}.weight", 6), GS)
        print(f"  AWQ scales for {len(awq)} layers ({time.time()-t0:.0f}s)")

    def awq_for(dst: str) -> np.ndarray | None:
        """Scale to apply to a consumer's INPUT dim, or None."""
        m = re.match(r"backbone\.layers\.(\d+)\.mixer\.(.+)$", dst)
        if not m:
            return None
        li, rest = int(m.group(1)), m.group(2)
        s = awq.get(li)
        if s is None:
            return None
        if rest.startswith(("switch_mlp.fc1", "shared_experts.up_proj", "gate",
                            "q_proj", "k_proj", "v_proj", "in_proj")):
            return s
        return None

    # ---- towers + passthrough + norms ------------------------------------
    n_tower = 0
    for k in sorted(index):
        kind = classify(k)
        if kind == "drop":
            continue
        if kind == "tower":
            # Preserve the SOURCE dtype. A blanket `.astype(np.float16)` is
            # wrong three ways here: bf16 carries fp32's exponent range, so the
            # cast is not lossless; the 24 int64 `num_batches_tracked` counters
            # (~2.1e6) overflow fp16 to `inf`; and the 2 float32
            # `input_conditioner.norm_mean/std` are preprocessing constants that
            # should not lose precision. Towers are never quantized, so there is
            # nothing to gain by downcasting them.
            add(k, load_raw(k))
            n_tower += 1
            continue
        if kind != "passthrough":
            continue
        dst = strip_prefix(k)
        raw = load_raw(k)
        src_dtype = raw.dtype          # keep bf16 as bf16; F32 routing bias as F32
        folded = None
        # Fold AWQ into the producing norm: w_norm <- w_norm / s
        if dst.endswith(".norm.weight") and not dst.endswith("mixer.norm.weight"):
            m = re.match(r"backbone\.layers\.(\d+)\.norm\.weight$", dst)
            if m and int(m.group(1)) in awq:
                folded = np.asarray(raw.astype(mx.float32)) / awq[int(m.group(1))]
        # The fp16 router gate must absorb the SAME scale exactly, or routing
        # shifts silently — it is not quantized, so this is lossless.
        if dst.endswith(".mixer.gate.weight"):
            s = awq_for(dst.replace(".weight", ""))
            if s is not None:
                folded = np.asarray(raw.astype(mx.float32)) * s[None, :]
        add(dst, raw if folded is None else mx.array(folded).astype(src_dtype))
    print(f"  towers kept unquantized at source dtype: {n_tower} tensors")

    # ---- dense affine ----------------------------------------------------
    for k in sorted(index):
        if classify(k) != "affine":
            continue
        dst = strip_prefix(k)
        base = dst[: -len(".weight")]
        bits = bits_map[dst]
        W = load(k).astype(np.float32)
        s = awq_for(base)
        if s is not None:
            W = W * s[None, :]
        im = diag.get(base)
        if s is not None and im is not None:
            im = im / (s.astype(np.float64) ** 2)
        emit_quant(base, W, bits, im)
        del W

    # ---- routed experts: stack then quantize ------------------------------
    moe_layers = [i for i, b in enumerate(pattern) if b == "E"]
    for li in moe_layers:
        s = awq.get(li)
        for proj, sub in EXPERT_SUBMODULE.items():
            bits = ex_up if proj == "up_proj" else ex_dn
            mats = []
            for e in range(n_experts):
                w = load(
                    f"language_model.backbone.layers.{li}.mixer.experts.{e}.{proj}.weight"
                ).astype(np.float32)
                if proj == "up_proj" and s is not None:
                    w = w * s[None, :]
                mats.append(w)
            W = np.stack(mats, axis=0)          # (E, out, in)
            del mats
            im = None
            hk = f"H.{li}.{'in' if proj == 'up_proj' else 'mid'}"
            if hk in hess:
                im = np.diag(hess[hk]).copy()
                if proj == "up_proj" and s is not None:
                    im = im / (s.astype(np.float64) ** 2)
            emit_quant(f"backbone.layers.{li}.mixer.switch_mlp.{sub}", W, bits, im)
            del W
            gc.collect()
        print(f"    layer {li}: experts stacked ({ex_up}b/{ex_dn}b)", flush=True)

    flush(force=True)

    # ---- index -----------------------------------------------------------
    # Rename to the HF `-of-NNNNN` convention now that the shard count is known.
    # The reference bundle uses it and downstream tooling globs for that shape.
    n_shards = len(written)
    renamed = []
    for i, (fn, keys) in enumerate(written, 1):
        final = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        if final != fn:
            (OUT / fn).rename(OUT / final)
        renamed.append((final, keys))
    written = renamed

    weight_map, total = {}, 0
    for fn, keys in written:
        for kk in keys:
            weight_map[kk] = fn
        total += (OUT / fn).stat().st_size
    (OUT / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=1))

    # ---- configs ---------------------------------------------------------
    flat = dict(llm_cfg)
    flat["architectures"] = ["NemotronHForCausalLM"]
    flat["model_type"] = "nemotron_h"
    # Declare the DOMINANT width, not the minimum. The runtime infers per-layer
    # bits from the scales' shapes and treats the declared value as the default
    # that overrides are diffed against; declaring the min makes every 8-bit
    # tensor an "override" and trips JangLoader's config-metadata mismatch
    # patch. A bundle whose declared bits disagree with its weights is the
    # wrong-bit silent-dequant class — do not rely on the loader fixing it.
    _dominant = max(bit_hist_default := collections.Counter(bits_map.values()),
                    key=bit_hist_default.get)
    flat["quantization"] = {"group_size": GS, "bits": _dominant}
    (OUT / "config.json").write_text(json.dumps(flat, indent=1))

    omni = {k: v for k, v in full_cfg.items()}
    (OUT / "config_omni.json").write_text(json.dumps(omni, indent=1))

    bit_hist = {}
    for n, b in bits_map.items():
        bit_hist[b] = bit_hist.get(b, 0) + 1
    (OUT / "jang_config.json").write_text(json.dumps({
        "version": 2,
        "weight_format": "mlx",
        "profile": a.profile,
        "source_model": {
            "name": "Nemotron-3-Nano-Omni-30B-A3B-Reasoning",
            "org": "nvidia",
            "architecture": "nemotron_h",
            "wrapper_arch": full_cfg.get("architectures", [None])[0],
            "modality": "omni",
        },
        "quantization": {
            "method": "affine",
            "group_size": GS,
            "bits_default": max(bit_hist, key=bit_hist.get),
            "bit_widths_used": sorted(bit_hist),
            "gateless_relu2_floors": {"up_proj": ex_up, "down_proj": ex_dn},
            "calibrated": bool(diag),
            "imatrix": bool(diag),
            "awq_layers": len(awq),
            "gptq": bool(a.gptq and hess),
        },
        "hybrid_pattern": pattern,
        "modality": "omni",
        "multimodal_components": [p.rstrip(".") for p in TOWER_PREFIXES],
    }, indent=1))

    # ---- support files ---------------------------------------------------
    # The towers are useless without their preprocessors: image tiling, mel
    # extraction and the video/EVS path all live in these modules. Shipping the
    # weights without them is the "VL bundle that cannot see" failure.
    REQUIRED = [
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "preprocessor_config.json", "generation_config.json",
    ]
    OPTIONAL_PY = [
        "__init__.py", "configuration.py", "modeling.py", "processing.py",
        "processing_utils.py", "image_processing.py", "video_processing.py",
        "video_io.py", "audio_model.py", "evs.py",
        "configuration_radio.py", "configuration_nemotron_h.py",
        "modeling_nemotron_h.py",
    ]
    missing = []
    for fn in REQUIRED:
        src_f = SRC / fn
        if src_f.exists():
            shutil.copy2(src_f, OUT / fn)
        else:
            missing.append(fn)
    if missing:
        raise SystemExit(
            f"missing required bundle files in {SRC}: {missing} — refusing to "
            "emit a bundle that cannot tokenize or preprocess")
    n_py = 0
    for fn in OPTIONAL_PY:
        src_f = SRC / fn
        if src_f.exists():
            shutil.copy2(src_f, OUT / fn)
            n_py += 1
    print(f"  copied {len(REQUIRED)} config/tokenizer files + {n_py} python modules")

    print(f"\n  DONE  {OUT}  {total/2**30:.2f} GiB  ({len(weight_map)} tensors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
