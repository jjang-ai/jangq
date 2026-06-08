"""
Verify a MiniMax-M2.7-JANG_K bundle.
Created by Jinho Jang (eric@jangq.ai)

Three checks:
  1. Bit allocation audit — derive bits from on-disk packed shapes and
     compare to expected w1=2, w2=4, w3=2, attn=8, embed=6, lm_head=8.
  2. Stock mlx_lm.load() — confirms config + tensor names line up.
  3. Greedy decode against a known prompt with the MiniMax chat template.

Usage:
    python3 -m jang_tools.verify_minimax_jang <bundle_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open


EXPECTED = {
    "model.embed_tokens": (6, 128),
    "lm_head": (8, 128),
    "self_attn.q_proj": (8, 128),
    "self_attn.k_proj": (8, 128),
    "self_attn.v_proj": (8, 128),
    "self_attn.o_proj": (8, 128),
    "switch_mlp.gate_proj": (2, 128),
    "switch_mlp.up_proj":   (2, 128),
    "switch_mlp.down_proj": (4, 128),
}


def derive_bits_gs(weight_shape, scales_shape):
    """packed last dim → bits;  scales last dim → group_size."""
    packed_in = weight_shape[-1]   # packed cols
    scales_in = scales_shape[-1]
    # Recover original in_features from the scales axis:
    #     scales_in = original_in / gs   →  gs = ?  (need original_in)
    # And from packed_in: packed_in = original_in / (32 / bits)
    #                     → original_in = packed_in * 32 / bits
    # Two unknowns (bits, gs); use the constraint scales_in × gs = packed_in × 32 / bits
    # gs and bits both ≤ 128. Brute search:
    for bits in (2, 3, 4, 5, 6, 8):
        original_in = packed_in * 32 / bits
        if original_in != int(original_in):
            continue
        original_in = int(original_in)
        if original_in % scales_in != 0:
            continue
        gs = original_in // scales_in
        if gs in (32, 64, 128):
            return bits, gs
    return None, None


def audit(bundle_dir: Path):
    print("=" * 60)
    print(f"  BIT ALLOCATION AUDIT  →  {bundle_dir}")
    print("=" * 60)
    cfg = json.loads((bundle_dir / "config.json").read_text())
    n_layers = cfg["num_hidden_layers"]
    idx = json.loads((bundle_dir / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]

    # Collect (suffix, bits, gs, count) statistics
    from collections import Counter
    sigs = Counter()

    # Open each shard once, scan its weight tensors
    seen_shards = set()
    for tname, shard in wm.items():
        if not tname.endswith(".weight"):
            continue
        if shard in seen_shards:
            # Already grouped — skip per-shard re-open
            pass

    sample_seen = {}
    for tname, shard in sorted(wm.items()):
        if not tname.endswith(".weight"):
            continue
        # Find the matching scales tensor
        scales_name = tname[:-len(".weight")] + ".scales"
        if scales_name not in wm:
            continue  # passthrough fp16 weights have no scales
        with safe_open(str(bundle_dir / shard), framework="numpy") as f:
            keys = set(f.keys())
            if tname not in keys or scales_name not in keys:
                continue
            wshape = list(f.get_slice(tname).get_shape())
            sshape = list(f.get_slice(scales_name).get_shape())
        bits, gs = derive_bits_gs(wshape, sshape)
        # Bucket by suffix (drop layer index)
        suffix = ".".join(tname.split(".")[-3:-1])  # e.g. switch_mlp.gate_proj
        if "embed_tokens" in tname:
            suffix = "model.embed_tokens"
        elif tname == "lm_head.weight":
            suffix = "lm_head"
        sigs[(suffix, bits, gs)] += 1
        # Save one example shape per bucket
        sample_seen.setdefault((suffix, bits, gs), (tname, wshape, sshape))

    print(f"  layers in config: {n_layers}\n")
    print(f"  {'COUNT':>5} {'SUFFIX':<28} {'BITS':>5} {'GS':>5}  example shape")
    print(f"  {'-'*5} {'-'*28} {'-'*5} {'-'*5}  ----------------------------")
    pass_count, fail_count = 0, 0
    for (suffix, bits, gs), n in sorted(sigs.items(), key=lambda x: -x[1]):
        ex_name, ex_w, ex_s = sample_seen[(suffix, bits, gs)]
        marker = " "
        if suffix in EXPECTED:
            exp_b, exp_gs = EXPECTED[suffix]
            ok = bits == exp_b and gs == exp_gs
            marker = "✓" if ok else "✗"
            (pass_count if ok else fail_count).__add__
            if ok:
                pass_count += 1
            else:
                fail_count += 1
        print(f"  {n:>5} {suffix:<28} {bits!s:>5} {gs!s:>5}  "
              f"w={ex_w} s={ex_s} {marker}")
    print()
    print(f"  audit: {pass_count} buckets matched, {fail_count} mismatched")
    return fail_count == 0


def load_and_decode(bundle_dir: Path):
    print("=" * 60)
    print(f"  LOAD + GREEDY DECODE  →  {bundle_dir}")
    print("=" * 60)
    from mlx_lm import load, generate
    print("  loading...", flush=True)
    model, tok = load(str(bundle_dir))
    print(f"    loaded.  model class: {model.__class__.__name__}")
    print(f"    tokenizer: {tok.__class__.__name__}")
    print("  decoding 'What is the capital of France?' (greedy, max_tokens=32)...",
          flush=True)
    msgs = [{"role": "user", "content": "What is the capital of France?"}]
    if hasattr(tok, "apply_chat_template"):
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                         tokenize=False)
    else:
        prompt = "What is the capital of France?\n"
    out = generate(model, tok, prompt=prompt, max_tokens=32, verbose=False)
    print(f"\n  OUTPUT: {out!r}")
    return out


def main():
    if len(sys.argv) < 2:
        print("usage: python3 -m jang_tools.verify_minimax_jang <bundle_dir>",
              file=sys.stderr)
        sys.exit(2)
    bundle_dir = Path(sys.argv[1]).expanduser()
    if not bundle_dir.exists():
        raise SystemExit(f"bundle not found: {bundle_dir}")
    ok = audit(bundle_dir)
    if not ok:
        print("\n  ✗ bit allocation audit FAILED — skipping decode")
        sys.exit(1)
    print()
    out = load_and_decode(bundle_dir)
    if "Paris" in out or "paris" in out.lower():
        print("\n  ✓ decode contains 'Paris' — bundle looks coherent")
        sys.exit(0)
    print("\n  ✗ decode missing 'Paris' — coherence suspect")
    sys.exit(2)


if __name__ == "__main__":
    main()
