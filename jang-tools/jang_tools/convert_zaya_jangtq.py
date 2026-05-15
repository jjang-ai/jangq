"""Convert Zyphra/ZAYA1-8B BF16 weights to a ZAYA JANGTQ bundle."""

from __future__ import annotations

import argparse
import gc
import json
import struct
import sys
from pathlib import Path

from jang_tools.convert_zaya_common import (
    CAPABILITIES,
    PROFILE_BITS,
    ZAYA_JANGTQ_K_BITS,
    affine_quantize,
    copy_sidecars_with_template,
    expert_output_base,
    finalize_shards,
    is_passthrough,
    load_json,
    load_tensor,
    regular_bits,
    scan_source,
    split_expert_fc1,
    total_shard_size,
    tq_quantize_experts_rowwise,
    write_json,
)
from jang_tools.progress import ProgressEmitter


MAX_SHARD = 1_000_000_000
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Zyphra/ZAYA1-8B BF16 source to JANGTQ."
    )
    parser.add_argument("src", type=Path, help="Local Zyphra/ZAYA1-8B BF16 directory")
    parser.add_argument("out", type=Path, help="Output bundle directory")
    parser.add_argument("profile", nargs="?", default="JANGTQ2")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true", help="Scan and print tensor policy without writing output")
    parser.add_argument("--progress", choices=["json", "off"], default="off")
    parser.add_argument("--quiet-text", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile_key = args.profile.upper()
    if profile_key not in PROFILE_BITS:
        raise SystemExit(f"unknown profile {args.profile!r}; expected one of {sorted(PROFILE_BITS)}")

    progress = ProgressEmitter(
        json_to_stderr=(args.progress == "json"),
        quiet_text=args.quiet_text,
    )
    src = args.src.expanduser()
    out = args.out.expanduser()
    expert_bits = PROFILE_BITS[profile_key]
    if expert_bits == "mixed":
        # JANGTQ_K: per-projection bit allocation. 4-bit on down_proj because
        # its output feeds the residual stream; 2-bit on gate/up that get
        # gated through SwiGLU. Same scheme as MiniMax-M2.7-JANGTQ_K. Targets
        # ZAYA's top-1 + MOD-passthrough failure mode that breaks plain
        # JANGTQ2 once cumulative output crosses the 2-3k coherence ceiling.
        expert_bits_map = dict(ZAYA_JANGTQ_K_BITS)
        profile = "JANGTQ_K"
    else:
        expert_bits_map = None
        profile = f"JANGTQ{expert_bits}"
    group = int(args.group_size)

    try:
        config = load_json(src / "config.json")
        hidden_size = int(config["hidden_size"])
        n_experts = int(config["num_experts"])

        print("=" * 70)
        print(f"  Zyphra/ZAYA1-8B -> {profile} conversion")
        print("=" * 70)
        print(f"  Source: {src}")
        print(f"  Output: {out}")
        print(f"  Expert layout: split linear_fc1 -> pre-stacked switch_mlp gate/up/down")
        if expert_bits_map is not None:
            bits_summary = ", ".join(f"{k}={v}" for k, v in expert_bits_map.items())
            print(f"  Profile: routed_expert=mxtq-mixed ({bits_summary}), attention/embed=affine-8")
        else:
            print(f"  Profile: routed_expert=mxtq-{expert_bits}, attention/embed=affine-8")
        if expert_bits == 3:
            print("  Note: using row-wise TQ packing for 2048-wide 3-bit experts")

        progress.phase(1, 3, "scan")
        regular, experts = scan_source(src)
        print(f"  Found {len(regular)} regular tensors + {len(experts)} expert tensors")
        if args.dry_run:
            pass_count = 0
            affine_count = 0
            for name, shape, _sf_path in regular:
                if is_passthrough(name) or len(shape) < 2:
                    pass_count += 1
                else:
                    affine_count += 1
            layers = sorted({layer for layer, _expert in experts})
            print("\n  Dry run policy:")
            print(f"    regular affine tensors: {affine_count}")
            print(f"    passthrough tensors:    {pass_count}")
            print(f"    expert layers:          {len(layers)}")
            print(f"    MXTQ expert groups:     {len(layers) * 3}")
            if expert_bits_map is not None:
                print(f"    expert bits:            mixed ({expert_bits_map})")
                for proj, b in expert_bits_map.items():
                    vals_per_u32 = 32 // b
                    packed_cols = (hidden_size + vals_per_u32 - 1) // vals_per_u32
                    print(f"      {proj} packed shape:  [{n_experts}, {hidden_size}, {packed_cols}] ({b}-bit)")
            else:
                vals_per_u32 = 32 // expert_bits
                packed_cols = (hidden_size + vals_per_u32 - 1) // vals_per_u32
                print(f"    expert bits:            {expert_bits}")
                print(f"    expert packed shape:    [{n_experts}, {hidden_size}, {packed_cols}] per projection")
            print("    no output written")
            progress.done(ok=True, output="dry-run")
            return

        import numpy as np
        from safetensors.numpy import save_file
        from tqdm import tqdm

        out.mkdir(parents=True, exist_ok=True)

        shard_idx = 0
        shard_tensors: dict[str, np.ndarray] = {}
        shard_bytes = 0
        shard_map: dict[str, str] = {}
        total_affine = 0
        total_mxtq = 0
        total_passthrough = 0

        done_keys: set[str] = set()
        existing = sorted(out.glob("model-*-of-XXXXX.safetensors"))
        if existing:
            for sf in existing:
                with sf.open("rb") as f:
                    hsize = struct.unpack("<Q", f.read(8))[0]
                    header = json.loads(f.read(hsize))
                for key in header:
                    if key != "__metadata__":
                        done_keys.add(key)
                        shard_map[key] = sf.name
                shard_idx = max(shard_idx, int(sf.name.split("-")[1]))
            print(f"  Resume: {len(done_keys)} keys already written")

        def flush_shard() -> None:
            nonlocal shard_idx, shard_tensors, shard_bytes
            if not shard_tensors:
                return
            shard_idx += 1
            name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
            save_file(shard_tensors, str(out / name))
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

        def affine_emit(base: str, arr: np.ndarray, bits: int = 8) -> bool:
            keys = (f"{base}.weight", f"{base}.scales", f"{base}.biases")
            if all(key in done_keys for key in keys):
                return False
            qw, qs, qb = affine_quantize(arr, bits=bits, group_size=group)
            add_tensor(keys[0], qw)
            add_tensor(keys[1], qs)
            add_tensor(keys[2], qb)
            return True

        def mxtq_emit(base: str, stack: np.ndarray, bits: int) -> bool:
            keys = (f"{base}.tq_packed", f"{base}.tq_norms", f"{base}.tq_bits")
            if all(key in done_keys for key in keys):
                return False
            result = tq_quantize_experts_rowwise(stack, bits=bits, seed=SEED)
            add_tensor(keys[0], result["packed"])
            add_tensor(keys[1], result["norms"])
            add_tensor(keys[2], np.array([bits], dtype=np.uint8))
            return True

        progress.phase(2, 3, "convert")
        print("\n  Converting regular tensors...")
        for name, shape, sf_path in tqdm(regular, desc="  Regular"):
            base = name[:-7] if name.endswith(".weight") else name
            if is_passthrough(name) or len(shape) < 2:
                if name in done_keys:
                    total_passthrough += 1
                    continue
                arr = load_tensor(sf_path, name, shape)
                add_tensor(name, arr.astype(np.float16))
                total_passthrough += 1
            else:
                keys = (f"{base}.weight", f"{base}.scales", f"{base}.biases")
                if all(key in done_keys for key in keys):
                    total_affine += 1
                    continue
                arr = load_tensor(sf_path, name, shape)
                if affine_emit(base, arr, regular_bits(name, 8, 8)):
                    total_affine += 1
            del arr
            if total_affine % 200 == 0:
                gc.collect()

        print("\n  Splitting + TQ-encoding experts...")
        layers = sorted({layer for layer, _expert in experts})
        for layer in tqdm(layers, desc="  Experts"):
            members = {expert: experts[(layer, expert)] for expert in range(n_experts) if (layer, expert) in experts}
            if len(members) != n_experts:
                print(f"  WARNING: layer {layer} has {len(members)} experts, expected {n_experts}")

            stacks = {
                "gate_proj": np.empty((n_experts, hidden_size, hidden_size), dtype=np.float32),
                "up_proj": np.empty((n_experts, hidden_size, hidden_size), dtype=np.float32),
                "down_proj": np.empty((n_experts, hidden_size, hidden_size), dtype=np.float32),
            }
            for expert in range(n_experts):
                parts = members[expert]
                fc1_shape, fc1_sf, fc1_name = parts["linear_fc1"]
                fc2_shape, fc2_sf, fc2_name = parts["linear_fc2"]
                fc1 = load_tensor(fc1_sf, fc1_name, fc1_shape)
                gate, up = split_expert_fc1(fc1, hidden_size)
                down = load_tensor(fc2_sf, fc2_name, fc2_shape)
                stacks["gate_proj"][expert] = gate
                stacks["up_proj"][expert] = up
                stacks["down_proj"][expert] = down
                del fc1, gate, up, down

            for proj, stack in stacks.items():
                proj_bits = (
                    expert_bits_map[proj] if expert_bits_map is not None else expert_bits
                )
                if mxtq_emit(expert_output_base(layer, proj), stack, proj_bits):
                    total_mxtq += 1
            del stacks
            gc.collect()

        flush_shard()

        progress.phase(3, 3, "write")
        shard_map = finalize_shards(out, shard_idx, shard_map)
        index = {
            "metadata": {
                "format": "jangtq",
                "total_size": total_shard_size(out, shard_map),
            },
            "weight_map": shard_map,
        }
        write_json(out / "model.safetensors.index.json", index)

        # routed_expert is the per-projection map for JANGTQ_K, scalar bits
        # otherwise. Loader's per-tensor `.tq_bits` is the source of truth for
        # routed experts; this entry is informational + drives the Swift
        # `LoadingLayout.jangtqRoutedBits` decode for non-uniform bundles.
        routed_expert_meta: object
        if expert_bits_map is not None:
            routed_expert_meta = dict(expert_bits_map)
        else:
            routed_expert_meta = expert_bits
        mxtq_bits = {
            "routed_expert": routed_expert_meta,
            "attention": 8,
            "router": 16,
            "embed_tokens": 8,
            "lm_head": 8,
            "cca_conv": 16,
            "norms_residual": 16,
        }
        tq_in_features = {
            expert_output_base(layer, proj): hidden_size
            for layer in layers
            for proj in ("gate_proj", "up_proj", "down_proj")
        }
        config.pop("quantization_config", None)
        config["weight_format"] = "mxtq"
        config["zaya_expert_layout"] = "split_switch_mlp"
        config.setdefault("tie_word_embeddings", True)
        config["mxtq_bits"] = mxtq_bits
        config["mxtq_seed"] = SEED
        config["quantization"] = {
            "bits": 8,
            "group_size": group,
            "mode": "affine",
            "routed_expert_bits": routed_expert_meta,
            "mxtq_bits": mxtq_bits,
            "expert_layout": "split_switch_mlp",
        }
        config["capabilities"] = CAPABILITIES
        write_json(out / "config.json", config)

        jang_config = {
            "version": 2,
            "weight_format": "mxtq",
            "profile": profile,
            "cache_subtype": "zaya_cca",
            "source_model": {
                "name": "ZAYA1-8B",
                "org": "Zyphra",
                "architecture": "zaya",
            },
            "expert_layout": "split_switch_mlp",
            "mxtq_seed": SEED,
            "mxtq_bits": mxtq_bits,
            "tq_in_features": tq_in_features,
            "quantization": {
                "method": "affine+mxtq",
                "group_size": group,
                "bits_default": expert_bits if expert_bits != "mixed" else "mixed",
            },
            "capabilities": CAPABILITIES,
        }
        write_json(out / "jang_config.json", jang_config)
        copy_sidecars_with_template(src, out)

        print("\n  Building jangtq_runtime.safetensors sidecar...")
        try:
            from jang_tools.build_jangtq_sidecar import main as build_sidecar

            saved_argv = list(sys.argv)
            sys.argv = ["build_jangtq_sidecar", str(out)]
            try:
                build_sidecar()
            finally:
                sys.argv = saved_argv
        except (Exception, SystemExit) as exc:
            print(f"  [sidecar] failed: {exc}")
            print(f"  Run manually before upload: python3 -m jang_tools.build_jangtq_sidecar {out}")

        print("\n  Done!")
        print(f"  MXTQ expert groups:  {total_mxtq}")
        print(f"  Affine tensors:      {total_affine}")
        print(f"  Passthrough tensors: {total_passthrough}")
        print(f"  Output:              {out}")
        progress.done(ok=True, output=str(out))
    except Exception as exc:
        progress.done(ok=False, error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
