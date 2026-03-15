"""
MLXQ Tools CLI — Mixed-Precision Importance Quantization for Apple Silicon
Created by Eric Jang (eric@vmlx.net)
"""

import argparse
import json
import sys

from . import __version__

BANNER = f"""
  ╔══════════════════════════════════════════════════════╗
  ║  MLXQ Tools v{__version__:<43s}║
  ║  Mixed-Precision Importance Quantization             ║
  ║  for Apple Silicon                                   ║
  ║                                                      ║
  ║  Created by Eric Jang (eric@vmlx.net)                ║
  ╚══════════════════════════════════════════════════════╝
"""


def cmd_inspect(args):
    """Inspect an MXQ model — show bit allocation, quality metrics, size."""
    from .format.reader import load_mxq_model

    model = load_mxq_model(args.model)
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
    metrics = model.mxq_config.get("quality_metrics", {})
    if metrics:
        print("  Quality metrics:")
        for key, val in metrics.items():
            print(f"    {key}: {val}")
        print()


def cmd_validate(args):
    """Validate an MXQ model directory."""
    from .format.reader import is_mxq_model, load_mxq_model

    path = args.model
    if not is_mxq_model(path):
        print(f"  ERROR: {path} is not a valid MXQ model directory")
        sys.exit(1)

    try:
        model = load_mxq_model(path)
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
    """Estimate MXQ model size for a given parameter count and bit width."""
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
            f"  MXQ-{target:<5.1f}  {info['nominal_bits']:>7.1f}b  "
            f"{info['effective_bits']:>9.2f}b  {info['weight_gb']:>9.1f}"
        )
    print()


def main():
    parser = argparse.ArgumentParser(
        prog="mxq-tools",
        description="MLXQ: Mixed-Precision Importance Quantization for Apple Silicon",
    )
    parser.add_argument("--version", action="version", version=f"mxq-tools {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    # inspect
    p_inspect = subparsers.add_parser("inspect", help="Inspect an MXQ model")
    p_inspect.add_argument("model", help="Path to MXQ model directory")
    p_inspect.set_defaults(func=cmd_inspect)

    # validate
    p_validate = subparsers.add_parser("validate", help="Validate an MXQ model directory")
    p_validate.add_argument("model", help="Path to MXQ model directory")
    p_validate.set_defaults(func=cmd_validate)

    # estimate
    p_estimate = subparsers.add_parser("estimate", help="Estimate MXQ model sizes")
    p_estimate.add_argument("params", help="Parameter count (e.g., 70B, 14B, 7B)")
    p_estimate.set_defaults(func=cmd_estimate)

    args = parser.parse_args()

    if args.command is None:
        print(BANNER)
        parser.print_help()
        sys.exit(0)

    print(BANNER)
    args.func(args)


if __name__ == "__main__":
    main()
