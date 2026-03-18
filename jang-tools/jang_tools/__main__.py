"""
JANG Tools CLI — Mixed-Precision Importance Quantization for Apple Silicon
Created by Jinho Jang (eric@jangq.ai)
"""

import argparse
import json
import sys

from . import __version__

BANNER = f"""
  ╔══════════════════════════════════════════════════════╗
  ║  JANG Tools v{__version__:<43s}║
  ║  Mixed-Precision Importance Quantization             ║
  ║  for Apple Silicon                                   ║
  ║                                                      ║
  ║  Created by Jinho Jang (eric@jangq.ai)                ║
  ╚══════════════════════════════════════════════════════╝
"""


def cmd_inspect(args):
    """Inspect a JANG model — show bit allocation, quality metrics, size."""
    from .format.reader import load_jang_model

    model = load_jang_model(args.model)
    summary = model.summary()

    print(f"\n  Model: {summary['source_model']}")
    print(f"  Target bits: {summary['target_bits']}")
    print(f"  Actual bits: {summary['actual_bits']}")
    print(f"  Block size: {summary['block_size']}")
    print(f"  Total blocks: {summary['total_blocks']:,}")
    print(f"  Weight tensors: {summary['total_weight_names']}")
    print(f"  Total size: {summary['total_qweight_gb']} GB")
    print()
    print("  Bit allocation:")
    for bw, info in summary["histogram"].items():
        bar = "█" * int(info["percent"] / 2)
        print(f"    {bw:>5s}: {info['count']:>8,} blocks ({info['percent']:>5.1f}%) {bar}")
    print()

    # Show quality metrics if available
    metrics = model.jang_config.get("quality_metrics", {})
    if metrics:
        print("  Quality metrics:")
        for key, val in metrics.items():
            print(f"    {key}: {val}")
        print()


def cmd_validate(args):
    """Validate a JANG model directory."""
    from .format.reader import is_jang_model, load_jang_model

    path = args.model
    if not is_jang_model(path):
        print(f"  ERROR: {path} is not a valid JANG model directory")
        sys.exit(1)

    try:
        model = load_jang_model(path)
        summary = model.summary()
        print(f"  VALID: {path}")
        print(f"  Source: {summary['source_model']}")
        print(f"  Bits: {summary['actual_bits']}")
        print(f"  Blocks: {summary['total_blocks']:,}")
        print(f"  Size: {summary['total_qweight_gb']} GB")
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)


def cmd_estimate(args):
    """Estimate JANG model size for a given parameter count and bit width."""
    from .format.spec import estimate_model_size

    # Parse parameter count (e.g., "70B", "14B", "7B")
    param_str = args.params.upper().replace(",", "")
    multipliers = {"B": 1e9, "M": 1e6, "K": 1e3}
    for suffix, mult in multipliers.items():
        if param_str.endswith(suffix):
            num_params = int(float(param_str[:-1]) * mult)
            break
    else:
        num_params = int(param_str)

    print(f"\n  Model: {args.params} parameters")
    print(f"  {'Profile':<12s} {'Nominal':>8s} {'Effective':>10s} {'Size (GB)':>10s}")
    print(f"  {'─' * 42}")

    for target in [2.0, 2.5, 3.0, 4.0, 6.0, 8.0]:
        info = estimate_model_size(num_params, target)
        print(
            f"  JANG-{target:<5.1f}  {info['nominal_bits']:>7.1f}b  "
            f"{info['effective_bits']:>9.2f}b  {info['weight_gb']:>9.1f}"
        )
    print()


def cmd_upgrade(args):
    """Upgrade a JANG v1 model to v2 format (MLX-native, instant load)."""
    try:
        from .loader import upgrade_v1_to_v2, is_jang_model, _is_v2_model
    except ImportError:
        print("  ERROR: 'jang upgrade' requires MLX (Apple Silicon only).")
        print("  Install with: pip install 'jang[mlx]'")
        sys.exit(1)
    from pathlib import Path

    path = Path(args.model)
    if not is_jang_model(path):
        print(f"  ERROR: {path} is not a JANG model directory")
        sys.exit(1)

    if _is_v2_model(path):
        print(f"  Already v2 format — loads instantly via mx.load() mmap")
        sys.exit(0)

    upgrade_v1_to_v2(path)


def cmd_convert(args):
    """Convert a HuggingFace model to JANG format."""
    from .convert import convert_model
    from .allocate import JANG_PROFILES, JANG_K_TARGETS, profile_for_bits, is_k_quant, k_quant_target

    # Resolve profile
    raw = args.profile
    if raw.isdigit():
        profile = profile_for_bits(int(raw))
        print(f"  Bit target {raw} → profile {profile}")
    else:
        profile = raw.upper()

    if profile not in JANG_PROFILES and not is_k_quant(profile):
        print(f"  ERROR: Unknown profile '{profile}'")
        all_profiles = sorted(JANG_PROFILES.keys()) + sorted(JANG_K_TARGETS.keys())
        print(f"  Available: {', '.join(all_profiles)}")
        print(f"  Or use a number 1-8 (e.g., jang convert model -p 2)")
        sys.exit(1)

    # Derive target_bits from profile
    if is_k_quant(profile):
        target_bits = k_quant_target(profile)
    else:
        # Extract from profile name: JANG_2S → 2.0, JANG_4M → 4.0
        for ch in profile.replace("JANG_", ""):
            if ch.isdigit():
                target_bits = float(ch)
                break
        else:
            target_bits = 4.0

    # Output path
    output = args.output
    if not output:
        import os
        name = os.path.basename(args.model.rstrip("/"))
        for suffix in ["-BF16", "-bf16", "-FP16", "-fp16"]:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        output = f"{name}-{profile}"

    result = convert_model(
        model_path=args.model,
        output_path=output,
        target_bits=target_bits,
        profile=profile,
        quantization_method=args.method,
    )

    print(f"\n  Profile: {profile}")
    print(f"  Actual bits: {result['actual_bits']}")
    print(f"  Weight size: {result['total_weight_gb']} GB")
    print(f"  Output: {output}")


def main():
    parser = argparse.ArgumentParser(
        prog="jang",
        description="JANG: Mixed-Precision Importance Quantization for Apple Silicon",
    )
    parser.add_argument("--version", action="version", version=f"jang-tools {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    # inspect
    p_inspect = subparsers.add_parser("inspect", help="Inspect a JANG model")
    p_inspect.add_argument("model", help="Path to JANG model directory")
    p_inspect.set_defaults(func=cmd_inspect)

    # validate
    p_validate = subparsers.add_parser("validate", help="Validate a JANG model directory")
    p_validate.add_argument("model", help="Path to JANG model directory")
    p_validate.set_defaults(func=cmd_validate)

    # estimate
    p_estimate = subparsers.add_parser("estimate", help="Estimate JANG model sizes")
    p_estimate.add_argument("params", help="Parameter count (e.g., 70B, 14B, 7B)")
    p_estimate.set_defaults(func=cmd_estimate)

    # convert
    p_convert = subparsers.add_parser("convert", help="Convert a HuggingFace model to JANG format")
    p_convert.add_argument("model", help="Path to HuggingFace model directory")
    p_convert.add_argument("-o", "--output", help="Output directory (default: auto)")
    p_convert.add_argument("-p", "--profile", default="2",
                          help="JANG profile (e.g., JANG_2L, JANG_3M) or number 1-8 (default: 2)")
    p_convert.add_argument("-m", "--method", default="mse", choices=["mse", "rtn", "mse-all"],
                          help="Quantization method (default: mse)")
    p_convert.set_defaults(func=cmd_convert)

    # upgrade
    p_upgrade = subparsers.add_parser("upgrade",
        help="Upgrade JANG v1 model to v2 (MLX-native, instant load)")
    p_upgrade.add_argument("model", help="Path to JANG v1 model directory")
    p_upgrade.set_defaults(func=cmd_upgrade)

    args = parser.parse_args()

    if args.command is None:
        print(BANNER)
        parser.print_help()
        sys.exit(0)

    print(BANNER)
    args.func(args)


if __name__ == "__main__":
    main()
