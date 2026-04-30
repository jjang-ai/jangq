"""Convert MiMo-V2.5-Pro FP8 source -> JANGTQ2 (TurboQuant 2-bit).

Mirrors jang_tools.convert_qwen35_jangtq / convert_minimax_jangtq /
convert_nemotron_jangtq:
    - dequant FP8 -> bf16 per tensor (jang_tools.mimo_v2.fp8_codec)
    - apply Hadamard rotation (jang_tools.turboquant.hadamard_kernel)
    - K-means / RQ codebooks per group (jang_tools.turboquant.codebook)
    - emit packed uint8 codes + bf16 codebook
    - embed routing-profile sidecar (freq/entropy/coact/transition)
    - bake mxtq_bits + routed_expert_bits + rope_parameters into config.json

Distributed: when launched via mlx.launch over the two-node rig, each rank
owns its slice of experts (sharding.default_mimo_v2_plan) and writes out
only those experts. A final merge step (rank 0) stitches sidecar metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import MiMoV2Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--routed-bits", type=int, default=2)
    ap.add_argument("--non-routed-bits", type=int, default=2)
    ap.add_argument("--hadamard", action="store_true", default=True)
    ap.add_argument("--awq", action="store_true", default=True,
                    help="Apply AWQ scales captured by awq_capture_fp8")
    args = ap.parse_args()

    cfg = MiMoV2Config.from_json(f"{args.src}/config.json")
    Path(args.dst).mkdir(parents=True, exist_ok=True)

    src_cfg = json.loads((Path(args.src) / "config.json").read_text())
    out = dict(src_cfg)
    out["mxtq_bits"] = args.routed_bits
    out["routed_expert_bits"] = args.routed_bits
    out["rope_parameters"] = {
        "rope_type": "default",
        "rope_theta": float(cfg.rope_theta),
        "partial_rotary_factor": float(cfg.partial_rotary_factor),
    }
    (Path(args.dst) / "config.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {args.dst}/config.json with mxtq_bits={args.routed_bits}")

    raise NotImplementedError(
        "Hook up to jang_tools.turboquant.* once FP8 source download completes "
        "and the AWQ probe (awq_capture_fp8) has run on calibration. Files in "
        "place to plug in (mirror convert_qwen35_jangtq.py exactly)."
    )


if __name__ == "__main__":
    main()
