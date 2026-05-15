"""Convert Zyphra/ZAYA1-8B BF16 weights to a ZAYA MXFP4/JANG affine bundle."""

from __future__ import annotations

import argparse
import gc
import json
import struct
from pathlib import Path

from jang_tools.convert_zaya_common import (
    CAPABILITIES,
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
    write_json,
)
from jang_tools.progress import ProgressEmitter


MAX_SHARD = 1_000_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Zyphra/ZAYA1-8B BF16 source to MXFP4 affine ZAYA bundle."
    )
    parser.add_argument("src", type=Path, help="Local Zyphra/ZAYA1-8B BF16 directory")
    parser.add_argument("out", type=Path, help="Output bundle directory")
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 6, 8])
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--embed-bits", type=int, default=8, choices=[4, 6, 8])
    parser.add_argument("--dry-run", action="store_true", help="Scan and print tensor policy without writing output")
    parser.add_argument("--progress", choices=["json", "off"], default="off")
    parser.add_argument("--quiet-text", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    progress = ProgressEmitter(
        json_to_stderr=(args.progress == "json"),
        quiet_text=args.quiet_text,
    )
    src = args.src.expanduser()
    out = args.out.expanduser()
    bits = int(args.bits)
    group = int(args.group_size)
    embed_bits = int(args.embed_bits)

    try:
        config = load_json(src / "config.json")
        hidden_size = int(config["hidden_size"])
        n_experts = int(config["num_experts"])

        print("=" * 70)
        print(f"  Zyphra/ZAYA1-8B -> MXFP{bits} affine conversion")
        print("=" * 70)
        print(f"  Source: {src}")
        print(f"  Output: {out}")
        print(f"  Expert layout: split linear_fc1 -> pre-stacked switch_mlp gate/up/down")
        print(f"  Bits: linears={bits}, embed={embed_bits}, router/CCA-state tensors=passthrough")

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
            print(f"    expert affine groups:   {len(layers) * 3}")
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
        total_passthrough = 0
        total_expert_groups = 0

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

        def affine_emit(base: str, arr: np.ndarray, b: int) -> bool:
            keys = (f"{base}.weight", f"{base}.scales", f"{base}.biases")
            if all(key in done_keys for key in keys):
                return False
            qw, qs, qb = affine_quantize(arr, bits=b, group_size=group)
            add_tensor(keys[0], qw)
            add_tensor(keys[1], qs)
            add_tensor(keys[2], qb)
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
                if all(key in done_keys for key in (f"{base}.weight", f"{base}.scales", f"{base}.biases")):
                    total_affine += 1
                    continue
                arr = load_tensor(sf_path, name, shape)
                if affine_emit(base, arr, regular_bits(name, bits, embed_bits)):
                    total_affine += 1
            del arr
            if total_affine % 200 == 0:
                gc.collect()

        print("\n  Splitting + pre-stacking experts...")
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
                base = expert_output_base(layer, proj)
                if affine_emit(base, stack, bits):
                    total_expert_groups += 1
            del stacks
            gc.collect()

        flush_shard()

        progress.phase(3, 3, "write")
        shard_map = finalize_shards(out, shard_idx, shard_map)
        index = {
            "metadata": {
                "format": "mxfp4",
                "total_size": total_shard_size(out, shard_map),
            },
            "weight_map": shard_map,
        }
        write_json(out / "model.safetensors.index.json", index)

        config.pop("quantization_config", None)
        config["weight_format"] = "mxfp4"
        config["zaya_expert_layout"] = "split_switch_mlp"
        config.setdefault("tie_word_embeddings", True)
        config["quantization"] = {
            "bits": bits,
            "group_size": group,
            "mode": "affine",
            "embed_bits": embed_bits,
            "router_bits": 16,
            "expert_layout": "split_switch_mlp",
        }
        config["capabilities"] = CAPABILITIES
        write_json(out / "config.json", config)

        jang_config = {
            "version": 2,
            "weight_format": "mxfp4",
            "profile": f"MXFP{bits}",
            "cache_subtype": "zaya_cca",
            "source_model": {
                "name": "ZAYA1-8B",
                "org": "Zyphra",
                "architecture": "zaya",
            },
            "expert_layout": "split_switch_mlp",
            "quantization": {
                "method": "affine",
                "group_size": group,
                "bits": bits,
                "embed_bits": embed_bits,
            },
            "capabilities": CAPABILITIES,
        }
        write_json(out / "jang_config.json", jang_config)
        copy_sidecars_with_template(src, out)

        print("\n  Done!")
        print(f"  Regular affine tensors: {total_affine}")
        print(f"  Expert affine groups:   {total_expert_groups}")
        print(f"  Passthrough tensors:    {total_passthrough}")
        print(f"  Output:                 {out}")
        progress.done(ok=True, output=str(out))
    except Exception as exc:
        progress.done(ok=False, error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
