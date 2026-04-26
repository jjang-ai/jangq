"""Patch DSV4-Flash bundle config.json in place to fix the
quant-metadata bug found 2026-04-24.

Bug: bundles shipped with `quantization = {bits: 2, group_size: 32}` only.
MLX applied 2-bit dequant to 8-bit-stored attention/embed/head weights →
silent garbage on hard prompts (HumanEval pass@1 42% instead of ~70%).

Fix strategy: enumerate every quantized Linear module by scanning
safetensors and write explicit per-module overrides — matching the
reference-quant shape (Thump604, mlx-community).

Bit-width inference per layer:
- Has `.tq_packed` (no `.weight`): MXTQ codebook → no override
  (TurboQuantSwitchLinear handles at runtime, MLX skips).
- Has `.weight` + `.scales` + `.biases`: standard affine; bits inferred
  from `weight.shape[-1] * 32 / in_features` (mlx packing).
- Else: passthrough fp16.

Usage:
    python3 -m jang_tools.patch_dsv4_quant_config /path/to/bundle1 ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from safetensors import safe_open


def _scan_quantized_layers(bundle: Path) -> dict[str, dict]:
    """Walk all shards. For every Linear-like quantized layer, infer
    (bits, group_size, mode) from on-disk shape vs weight metadata.
    Skip layers stored as MXTQ codebook (tq_packed only)."""
    out: dict[str, dict] = {}
    index_path = bundle / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"No model.safetensors.index.json in {bundle}")
    weight_map: dict[str, str] = json.loads(index_path.read_text())["weight_map"]

    # Group by shard, collect keys per shard.
    by_shard: dict[str, list[str]] = {}
    for k, fname in weight_map.items():
        by_shard.setdefault(fname, []).append(k)

    # Build set of paths that have scales (= are quantized affine/mxfp4).
    keys = set(weight_map.keys())
    quantized_paths: set[str] = set()
    mxtq_paths: set[str] = set()
    for k in keys:
        if k.endswith(".scales"):
            base = k[:-len(".scales")]
            if (base + ".weight") in keys and (base + ".biases") in keys:
                quantized_paths.add(base)
        elif k.endswith(".tq_packed"):
            mxtq_paths.add(k[:-len(".tq_packed")])

    # For each quantized path, read weight shape + scales shape to infer bits.
    # MLX packs `bits` codes per element of weight (uint32):
    #   weight.shape = (..., out, in_packed) where in_packed = in / (32 / bits)
    #   scales.shape = (..., out, n_groups) where n_groups = in / group_size
    # The .weight and .scales for the same module may live in different shards,
    # so look up each via weight_map.
    open_handles: dict[str, "safe_open"] = {}
    def get_handle(fname: str):
        if fname not in open_handles:
            open_handles[fname] = safe_open(str(bundle / fname), framework="numpy")
            open_handles[fname].__enter__()
        return open_handles[fname]
    try:
        for base in sorted(quantized_paths):
            wkey = base + ".weight"
            skey = base + ".scales"
            wfn = weight_map.get(wkey)
            sfn = weight_map.get(skey)
            if not wfn or not sfn:
                continue
            wf = get_handle(wfn)
            sf = get_handle(sfn)
            w_shape = list(wf.get_slice(wkey).get_shape())
            s_shape = list(sf.get_slice(skey).get_shape())
            in_packed = w_shape[-1]
            n_groups = s_shape[-1]
            if n_groups <= 0 or in_packed <= 0:
                continue
            # 32 * in_packed = bits * n_groups * group_size — solve for plausible (bits, gs).
            best = None
            for gs in (32, 64, 128):
                num = 32 * in_packed
                den = n_groups * gs
                if den == 0 or num % den != 0:
                    continue
                bits = num // den
                if bits in (2, 3, 4, 5, 6, 8):
                    best = (bits, gs)
                    break
            if best is None:
                continue
            bits, gs = best
            is_routed = ("switch_mlp" in base) or (".experts." in base)
            mode = "mxfp4" if (bits == 4 and is_routed) else "affine"
            out[base] = {"group_size": gs, "bits": bits, "mode": mode}
    finally:
        for h in open_handles.values():
            try:
                h.__exit__(None, None, None)
            except Exception:
                pass
    return out


def _detect_format(bundle: Path) -> str:
    """Returns 'jangtq' if any `.tq_packed` exists, else 'jang'."""
    index_path = bundle / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"No safetensors index in {bundle}")
    weight_map = json.loads(index_path.read_text())["weight_map"]
    for k in weight_map.keys():
        if k.endswith(".tq_packed"):
            return "jangtq"
    return "jang"


def _read_tq_bits(bundle: Path) -> int:
    """Read any one .tq_bits scalar (assume all same)."""
    weight_map = json.loads((bundle / "model.safetensors.index.json").read_text())["weight_map"]
    for k, fname in weight_map.items():
        if k.endswith(".tq_bits"):
            with safe_open(str(bundle / fname), framework="numpy") as f:
                return int(f.get_tensor(k).flat[0])
    return 0


def patch(bundle: Path) -> bool:
    cfg_path = bundle / "config.json"
    if not cfg_path.exists():
        print(f"[skip] {bundle}: no config.json", flush=True)
        return False
    src_cfg = json.loads(cfg_path.read_text())

    fmt = _detect_format(bundle)
    overrides = _scan_quantized_layers(bundle)
    profile_bits = _read_tq_bits(bundle) if fmt == "jangtq" else 0

    # Build new quant cfg: top-level 8-bit affine + per-module overrides.
    new_q: dict = {"group_size": 32, "bits": 8, "mode": "affine"}
    for path, ov in sorted(overrides.items()):
        new_q[path] = ov

    # Detect routed-expert profile bits for JANG (no tq_bits there).
    if fmt == "jang":
        sample_overrides = [v for k, v in overrides.items()
                            if "switch_mlp" in k or "experts" in k]
        if sample_overrides:
            profile_bits = sample_overrides[0]["bits"]

    # transformers ≥4.50 expects `rope_parameters` (with rope_theta inside).
    # Without it, mlx-lm load_tokenizer falls back to bare
    # PreTrainedTokenizerFast — drops chat_template + special tokens.
    # transformers validation also rejects int values — factor / beta_*
    # must be floats. Cast on write.
    rope_added = False
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
        for k in ("factor", "beta_fast", "beta_slow", "rope_theta"):
            if k in rp and not isinstance(rp[k], float):
                rp[k] = float(rp[k])
        src_cfg["rope_parameters"] = rp
        rope_added = True

    old_q = src_cfg.get("quantization", {})
    quant_unchanged = old_q == new_q

    if quant_unchanged and not rope_added:
        print(f"[ok ] {bundle}: already correct ({len(new_q)} entries)", flush=True)
        return False

    src_cfg["quantization"] = new_q
    src_cfg["routed_expert_bits"] = profile_bits
    src_cfg["group_size"] = 32
    src_cfg["mxtq_seed"] = src_cfg.get("mxtq_seed", 42)
    cfg_path.write_text(json.dumps(src_cfg, indent=2))

    n_old_overrides = sum(1 for v in old_q.values() if isinstance(v, dict))
    n_new_overrides = sum(1 for v in new_q.values() if isinstance(v, dict))
    parts = []
    if not quant_unchanged:
        parts.append(
            f"top-level bits {old_q.get('bits', '?')}→8 mode→affine, "
            f"overrides {n_old_overrides}→{n_new_overrides}"
        )
    if rope_added:
        parts.append("rope_parameters added")
    print(f"[FIX] {bundle.name}: fmt={fmt} profile_bits={profile_bits} {'; '.join(parts)}", flush=True)
    return True


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for arg in sys.argv[1:]:
        bundle = Path(arg)
        if not bundle.is_dir():
            print(f"[skip] {bundle}: not a directory")
            continue
        try:
            patch(bundle)
        except Exception as e:
            import traceback
            print(f"[ERR] {bundle}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
