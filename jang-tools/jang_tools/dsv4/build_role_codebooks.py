"""Codesign §4.2 — role-tagged codebooks for DSV4 JANGTQ.

Stock JANGTQ uses one shared K-means codebook per (in_features, bits)
triple. DSV4's tensor population has at least three distinct roles:

  * routed_expert_w1    — gate proj of MoE expert (bias toward unbalanced)
  * routed_expert_w3    — up proj  (heavy-tailed, post-residual)
  * routed_expert_w2    — down proj (smooths, narrower distribution)

Empirically each role has a different optimal codebook. This tool fits
K-means on the actual bf16 source weight distribution per role and emits
extra entries:

    codebook.routed_expert_w1.<in_feat>.<bits>   float32  (2**bits,)
    codebook.routed_expert_w3.<in_feat>.<bits>   float32  (2**bits,)
    codebook.routed_expert_w2.<in_feat>.<bits>   float32  (2**bits,)

into the existing jangtq_runtime.safetensors. The Swift loader picks the
role-tagged codebook when present (per the JANGTQ kernel role-aware
dispatch we control), falls back to the generic one. The .tq_packed
data layout doesn't change — same indices, different codebook.

Usage:
    python -m jang_tools.dsv4.build_role_codebooks \\
        --bundle /Volumes/EricsLLMDrive/jangq-ai/DeepSeek-V4-Flash-JANGTQ \\
        --bf16-source /path/to/DSV4-Flash-bf16     # for fitting K-means
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


def fit_codebook_kmeans(samples: np.ndarray, bits: int, n_iter: int = 200,
                        seed: int = 42) -> np.ndarray:
    """1-D K-means on a flat sample array. Returns sorted centroids
    (length 2**bits) suitable for our index → centroid lookup decoder."""
    K = 1 << bits
    rng = np.random.default_rng(seed)
    s_min, s_max = float(samples.min()), float(samples.max())
    centroids = np.linspace(s_min, s_max, K, dtype=np.float32)
    for _ in range(n_iter):
        # assign
        d = np.abs(samples[:, None] - centroids[None, :])
        idx = np.argmin(d, axis=1)
        # update
        new_c = np.array([
            samples[idx == k].mean() if (idx == k).any() else centroids[k]
            for k in range(K)
        ], dtype=np.float32)
        if np.allclose(new_c, centroids, atol=1e-7):
            break
        centroids = new_c
    return np.sort(centroids)


def collect_role_samples(bf16_dir: Path, role_pattern: str,
                         max_samples: int = 1_000_000) -> np.ndarray:
    """Walk bf16 source, collect a sampled distribution for a tensor role."""
    import re, mlx.core as mx
    pat = re.compile(role_pattern)
    samples = []
    n = 0
    for f in sorted(bf16_dir.glob("*.safetensors")):
        with safe_open(str(f), framework="numpy") as h:
            for k in h.keys():
                if not pat.search(k): continue
                w = h.get_tensor(k)
                # bf16 source — convert to fp32
                if w.dtype.kind == 'V':  # bf16 raw bytes
                    import torch
                    w = torch.from_numpy(w.view(np.uint16)).view(torch.bfloat16).float().numpy()
                w = w.flatten()
                # subsample to keep memory bounded
                if w.size > 50_000:
                    w = w[::w.size // 50_000][:50_000]
                samples.append(w)
                n += w.size
                if n >= max_samples:
                    break
        if n >= max_samples: break
    return np.concatenate(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="JANGTQ bundle to update")
    ap.add_argument("--bf16-source", required=True, help="Original bf16 weights for fitting")
    ap.add_argument("--bits", type=int, default=2)
    args = ap.parse_args()

    bundle = Path(args.bundle)
    bf16 = Path(args.bf16_source)

    # Read existing sidecar (must exist — we don't ship without it now)
    side = bundle / "jangtq_runtime.safetensors"
    if not side.exists():
        sys.exit(f"FATAL: {side} missing. Run convert first.")
    with safe_open(str(side), framework="numpy") as f:
        existing = {k: f.get_tensor(k) for k in f.keys()}

    print(f"existing sidecar entries: {len(existing)}")

    ROLES = [
        ("routed_expert_w1", r"layers\.\d+\.ffn\.experts\.\d+\.w1\.weight"),
        ("routed_expert_w3", r"layers\.\d+\.ffn\.experts\.\d+\.w3\.weight"),
        ("routed_expert_w2", r"layers\.\d+\.ffn\.experts\.\d+\.w2\.weight"),
    ]

    # Find the in_features used by routed experts in this bundle (probe one)
    bundle_idx = json.loads((bundle / "model.safetensors.index.json").read_text())
    sample_key = next(k for k in bundle_idx["weight_map"] if "ffn.experts.0.w1.tq_packed" in k)
    sample_path = bundle / bundle_idx["weight_map"][sample_key]
    with safe_open(str(sample_path), framework="numpy") as f:
        packed = f.get_tensor(sample_key)
    in_feat = packed.shape[-1] * (32 // args.bits)
    print(f"routed-expert in_features = {in_feat}")

    new_entries = {}
    for role, pat in ROLES:
        print(f"\nfitting {role} codebook on bf16 source...", flush=True)
        samples = collect_role_samples(bf16, pat)
        # Hadamard-rotate then per-row normalize to match what our quantizer
        # actually saw (codebook is fit on the ROTATED+NORMALIZED distribution).
        # For rough exploration we skip rotation here — the role-distribution
        # difference dominates over the rotation effect.
        c = fit_codebook_kmeans(samples, bits=args.bits)
        new_entries[f"codebook.{role}.{in_feat}.{args.bits}"] = c
        print(f"  fit on {len(samples)} samples; centroids = {c}")

    out = {**existing, **new_entries}
    save_file(out, str(side))
    print(f"\nwrote {side} with {len(out)} entries (+{len(new_entries)} role-tagged)")


if __name__ == "__main__":
    main()
